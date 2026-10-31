"""
Whose personal data a task may touch, and keeping it out of shared places.

OpenTeddy has no user accounts. The one door that knows who is asking is a
Telegram private chat: the bridge sets the in-process origin "telegram"
(tools._context), which an HTTP client cannot. So a person is "tg:<chat
id>" — only for a private chat (a positive id; groups are negative), and
only when the request really came through the bridge or a schedule
running in that person's Telegram session. The web UI, CLI, MCP and
group chats have no person; personal Google data is refused there.

The web UI has no login and shows every session's history and live
events. So once a task reads someone's calendar or mail it is marked
private: what is stored (task summary, subtask results) and what is
broadcast to browsers is replaced by a note, memory doesn't keep it, and
only the Telegram reply carries the full answer.
"""
from __future__ import annotations

import time
from contextvars import ContextVar
from typing import Any, Dict, Optional

NOTE = "🔒 這則回答含有個人資料（Google 行事曆／Gmail），只顯示在本人的 Telegram，不保存在這裡。"

_scope: ContextVar[Optional[Dict[str, Any]]] = ContextVar("openteddy_personal_scope", default=None)
_private: Dict[str, float] = {}               # task_id → when it was marked


def person_for_chat(chat_id: Any) -> Optional[str]:
    s = str(chat_id or "").strip()
    return f"tg:{s}" if s.isdigit() and int(s) > 0 else None


async def resolve(req: Any, tracker: Any) -> Optional[str]:
    """The person behind this request, or None (web / CLI / MCP / group)."""
    from tools._context import get_triggered_by
    origin = get_triggered_by()
    if origin == "telegram":
        return person_for_chat((req.context or {}).get("telegram_chat_id"))
    if origin == "schedule" and req.session_id:
        try:
            sess = await tracker.get_session(req.session_id)
        except Exception:  # noqa: BLE001
            return None
        return person_for_chat((sess or {}).get("telegram_chat_id"))
    return None


def begin(task_id: str, person: Optional[str]) -> None:
    """Bind this task's person for everything it calls (tools included)."""
    _scope.set({"person": person, "task_id": task_id})


def current_person() -> Optional[str]:
    s = _scope.get()
    return (s or {}).get("person")


def mark(task_id: Optional[str] = None) -> None:
    """This task now carries personal data."""
    tid = task_id or (_scope.get() or {}).get("task_id")
    if not tid:
        return
    now = time.time()
    _private[tid] = now
    if len(_private) > 2000:                  # keep a day's worth
        for k in [k for k, t in _private.items() if now - t > 86400]:
            _private.pop(k, None)


def is_private(task_id: Optional[str]) -> bool:
    return bool(task_id) and task_id in _private


_EVENT_KEEP = {"type", "event", "task_id", "subtask_id", "session_id", "status", "order", "total",
               "round", "call_idx", "tool", "tool_name", "name", "success", "duration_ms",
               "confidence", "_ts"}


def scrub_event(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """What a browser may see of a private task's event; None = drop it."""
    if not is_private(data.get("task_id")):
        return data
    if str(data.get("type", "")).endswith("stream.delta"):
        return None
    out = {k: v for k, v in data.items() if k in _EVENT_KEEP}
    for k, v in data.items():
        if k not in out and isinstance(v, str) and v:
            out[k] = "🔒"
    out["private"] = True
    return out
