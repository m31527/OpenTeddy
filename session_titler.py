"""
Name sessions in OpenTeddy's quiet time.

A session starts as "New session". Once its first task has finished and
nothing has run for a while (idle.py), the fast chat model reads that
first exchange and gives it a short title. The person asked for it to
happen in the gaps, not on the hot path (2026-11-07), so a title never
delays an answer.

It is named once, from how it began, and never again: later turns don't
change it, and whoever wants another name renames it (the person's
words, 2026-11-07). A title the person typed (title_state 'user') is
never replaced, and a rename that lands while the model is thinking
wins (tracker.set_auto_title checks again).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from print_intent import _MANIFEST_RE

import idle
from config import config

logger = logging.getLogger(__name__)

QUIET_S = 30.0       # nothing has run for this long → the system is free
TICK_S = 15.0
BACKOFF_S = 300.0    # after a failed model call

SYSTEM = """You name chat sessions for a sidebar. Reply with only a short title for what the person asked for, in the person's language (Traditional Chinese for Chinese), at most 16 Chinese characters or 6 words. No quotes, no ending punctuation, no "Title:"."""

_tracker: Any = None
_orch: Any = None
_broadcast: Optional[Callable[[dict], Awaitable[None]]] = None
_loop_task: Optional[asyncio.Task] = None


def clean(text: str) -> Optional[str]:
    """The model's reply as a title; None when nothing is usable."""
    line = next((l for l in (text or "").splitlines() if l.strip()), "").strip()
    line = re.sub(r"^(標題|title)\s*[:：]\s*", "", line, flags=re.I)
    for _ in range(2):   # 「…」。 — quotes and the full stop, either order
        line = line.strip(" \t\"'`*#「」『』“”‘’").rstrip("。.!！")
    if not line or line.strip("-—_ ") == "":
        return None
    return line[:40].rstrip()


def conversation(task: dict) -> str:
    parts = [f"Person: {(task.get('goal') or '').strip()[:400]}"]
    s = (task.get("summary") or "").strip()
    if s:
        parts.append(f"OpenTeddy: {s[:300]}")
    return "\n".join(parts)


def from_request(goal: str) -> str:
    """The first line of what the person typed — when the model gives
    nothing usable."""
    text = _MANIFEST_RE.sub("", goal or "").strip()
    line = next((l for l in text.splitlines() if l.strip()), "").strip()
    return line[:30].rstrip() or "Session"


async def _ask_model(prompt: str, local_only: bool) -> str:
    from config import orchestrator_on_cloud, set_session_local_only
    set_session_local_only(local_only)   # this loop's own context only
    if orchestrator_on_cloud():
        r = await _orch.escalation.provider.complete_text(prompt, system=SYSTEM, max_tokens=60)
        return r.text or ""

    import local_engine
    payload: Dict[str, Any] = {
        "model": config.gemma_model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 60,
                    "num_ctx": int(getattr(config, "gemma_num_ctx", 16384))},
        "keep_alive": getattr(config, "ollama_keep_alive", "24h"),
    }
    th = local_engine.think_setting(config.gemma_model)
    if th is not None:
        payload["think"] = th
    t0 = time.monotonic()
    resp = await _orch._http.post(f"{config.gemma_base_url}/api/chat", json=payload)
    resp.raise_for_status()
    data = resp.json()
    try:
        t = local_engine.timings(data)
        out = data.get("eval_count", 0) or 0
        dur = data.get("eval_duration", 0) or 0
        await _tracker.record_usage(
            task_id="", model=config.gemma_model, model_provider="ollama",
            tokens_in=data.get("prompt_eval_count", 0) or 0, tokens_out=out,
            task_description="[session title]",
            duration_ms=int((data.get("total_duration") or 0) / 1e6) or int((time.monotonic() - t0) * 1000),
            tokens_per_sec=(out * 1e9 / dur) if out and dur else 0.0,
            load_ms=t["load_ms"], prefill_ms=t["prefill_ms"])
    except Exception:  # noqa: BLE001
        pass
    return (data.get("message") or {}).get("content") or ""


async def title_one() -> bool:
    """Name the most recently used session that needs it, from its first
    finished exchange. True if one was named."""
    rows = await _tracker.sessions_to_title(limit=1)
    if not rows:
        return False
    s = rows[0]
    first = (await _tracker.first_finished_tasks(s["id"], limit=1))[0]
    title = (clean(await _ask_model(conversation(first), bool(s.get("local_only"))))
             or from_request(first.get("goal") or ""))
    if await _tracker.set_auto_title(s["id"], title):
        logger.info("Session %s named %r", s["id"][:8], title)
        if _broadcast:
            await _broadcast({"type": "session.renamed", "session_id": s["id"], "title": title})
    return True


async def _run() -> None:
    while True:
        wait = TICK_S
        try:
            if idle.quiet_for() >= QUIET_S:
                if await title_one():
                    wait = 2.0   # backlog: keep going while it stays quiet
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Session titler: %s", exc)
            wait = BACKOFF_S
        await asyncio.sleep(wait)


async def start(tracker: Any, orch: Any, broadcast: Callable[[dict], Awaitable[None]]) -> None:
    global _tracker, _orch, _broadcast, _loop_task
    _tracker, _orch, _broadcast = tracker, orch, broadcast
    if _loop_task is None or _loop_task.done():
        _loop_task = asyncio.create_task(_run(), name="session_titler")


async def stop() -> None:
    global _loop_task
    if _loop_task:
        _loop_task.cancel()
        _loop_task = None
