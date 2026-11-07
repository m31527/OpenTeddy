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
from typing import Any, Awaitable, Callable, Optional

from print_intent import _MANIFEST_RE

import idle

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
    import fast_model
    return await fast_model.ask(_orch, SYSTEM, prompt, max_tokens=60,
                                label="[session title]", local_only=local_only)


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
