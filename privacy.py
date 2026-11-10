"""
Whose personal data a task may touch, and keeping it out of shared places.

OpenTeddy has no user accounts. The one door that knows who is asking is a
Telegram private chat: the bridge sets the in-process origin "telegram"
(tools._context), which an HTTP client cannot. So a person is "tg:<chat
id>" — only for a private chat (a positive id; groups are negative), and
only when the request really came through the bridge or a schedule
running in that person's Telegram session. Group chats have no person.

The web UI, CLI, MCP and Task API are one person between them,
WEB_PERSON: whoever uses this OpenTeddy's web UI. Its Settings page pairs
a Chrome for it, from any address that reaches the page — 127.0.0.1 on
the desktop app, the LAN or a VPN on a home server (the user, 2026-11-11:
"只要我可以連到設定裡面就應該取得該台電腦的配對碼"). On a server others can
reach, that means anyone who opens the web UI can use that Chrome, and
the web UI's task history shows what it read; Settings says so. Google
stays Telegram-only (has_account): a shared web UI must never become
everyone's calendar and mail.

Once a task reads someone's calendar, mail or browser it is marked
private: memory doesn't keep it and cloud models don't see it. For a
Telegram person the web UI also gets a note instead of the content
(stored summary, subtask results, live events) and only their Telegram
reply carries it; WEB_PERSON asked from the web UI, so it shows there.
"""
from __future__ import annotations

import os
import time
from contextvars import ContextVar
from typing import Any, Dict, Optional

NOTE = "🔒 這則回答含有個人資料（行事曆、信件或你的瀏覽器），只顯示在本人的 Telegram，不保存在這裡。"

WEB_PERSON = "web:owner"
_LOOPBACK = ("127.0.0.1", "localhost", "::1")

_scope: ContextVar[Optional[Dict[str, Any]]] = ContextVar("openteddy_personal_scope", default=None)
_private: Dict[str, float] = {}               # task_id → when it was marked
_web_view: Dict[str, float] = {}              # private tasks of WEB_PERSON: the web UI shows them


def local_only() -> bool:
    """True when this server listens on this computer only (run.sh and the
    desktop sidecar say so in OPENTEDDY_HOST). Settings then needn't warn
    that others can use the paired Chrome."""
    return os.environ.get("OPENTEDDY_HOST", "").strip().lower() in _LOOPBACK


def has_account(person: Optional[str]) -> bool:
    """May link and read personal accounts (Google): a real person, not the
    web UI that anyone reaching this server can open."""
    return bool(person) and person != WEB_PERSON


def person_for_chat(chat_id: Any) -> Optional[str]:
    s = str(chat_id or "").strip()
    return f"tg:{s}" if s.isdigit() and int(s) > 0 else None


async def resolve(req: Any, tracker: Any) -> Optional[str]:
    """The person behind this request: tg:<id>, WEB_PERSON (web UI, CLI,
    MCP, API), or None (a group chat)."""
    from tools._context import get_triggered_by
    import browser_bridge
    if browser_bridge.is_eval_task(req.id):          # the delivery eval's throwaway browser
        return browser_bridge.EVAL_PERSON
    origin = get_triggered_by()
    if origin == "telegram":
        return person_for_chat((req.context or {}).get("telegram_chat_id"))
    if origin == "schedule" and req.session_id:
        try:
            sess = await tracker.get_session(req.session_id)
        except Exception:  # noqa: BLE001
            sess = None
        tg = person_for_chat((sess or {}).get("telegram_chat_id"))
        if tg:
            return tg
    return WEB_PERSON


def begin(task_id: str, person: Optional[str]) -> None:
    """Bind this task's person for everything it calls (tools included)."""
    _scope.set({"person": person, "task_id": task_id})


def current_person() -> Optional[str]:
    s = _scope.get()
    return (s or {}).get("person")


def current_task() -> Optional[str]:
    s = _scope.get()
    return (s or {}).get("task_id")


def mark(task_id: Optional[str] = None) -> None:
    """This task now carries personal data. (Not the eval's own person: its
    browser is a throwaway one, and the eval has to read the answer.)"""
    tid = task_id or (_scope.get() or {}).get("task_id")
    if not tid or (current_person() or "").startswith("eval:"):
        return
    now = time.time()
    _private[tid] = now
    if current_person() == WEB_PERSON:
        _web_view[tid] = now
    if len(_private) > 2000:                  # keep a day's worth
        for d in (_private, _web_view):
            for k in [k for k, t in d.items() if now - t > 86400]:
                d.pop(k, None)


def is_private(task_id: Optional[str]) -> bool:
    """Personal data: keep it out of memory and off cloud models."""
    return bool(task_id) and task_id in _private


def hidden_from_web(task_id: Optional[str]) -> bool:
    """Personal data the web UI must not show — everything private except
    what the web UI's own person asked for there."""
    return is_private(task_id) and task_id not in _web_view


_EVENT_KEEP = {"type", "event", "task_id", "subtask_id", "session_id", "status", "order", "total",
               "round", "call_idx", "tool", "tool_name", "name", "success", "duration_ms",
               "confidence", "_ts"}


def scrub_event(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """What a browser may see of a private task's event; None = drop it."""
    if not hidden_from_web(data.get("task_id")):
        return data
    if str(data.get("type", "")).endswith("stream.delta"):
        return None
    out = {k: v for k, v in data.items() if k in _EVENT_KEEP}
    for k, v in data.items():
        if k not in out and isinstance(v, str) and v:
            out[k] = "🔒"
    out["private"] = True
    return out
