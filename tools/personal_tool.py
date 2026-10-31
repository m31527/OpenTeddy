"""
OpenTeddy Personal Tools — the asking person's own Google calendar and
Gmail (read-only) and one-shot reminders.

Whose data is never a tool argument: it is the person behind this task's
Telegram private chat (privacy.py). From the web UI, CLI, MCP or a group
the calendar and Gmail tools refuse, and a task that uses them is marked
private (its answer isn't stored or shown in the web UI).

Short questions ("今天有什麼行程", "有什麼新信", "明天 9 點提醒我…") are
answered before planning. These tools are for the same data as one step
of a bigger job: "看我明天的行程，空檔排一個健身", "整理今天的未讀信重點",
"會議前 15 分鐘提醒我".
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from tool_registry import RiskLevel, make_result


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


async def calendar_events(start_date: str = "", days: int = 1) -> Dict[str, Any]:
    import google_personal as G
    t0 = time.monotonic()
    try:
        first = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else datetime.now().date()
        n = max(1, min(int(days or 1), 31))
        evs = await G.events_for_days(first, n)
        label = G.day_label(first) + (f" 起 {n} 天" if n > 1 else "")
        return make_result(True, result=G.describe_events(evs, label, multi_day=n > 1), duration_ms=_ms(t0))
    except (G.PersonalError, ValueError) as exc:
        return make_result(False, error=str(exc), duration_ms=_ms(t0))


async def gmail_unread(query: str = "", limit: int = 10) -> Dict[str, Any]:
    import google_personal as G
    t0 = time.monotonic()
    q = (query or "").strip() or "category:primary is:unread newer_than:1d"
    try:
        mails = await G.unread(q.replace('"', "'"), max(1, min(int(limit or 10), 30)))
        return make_result(True, result=G.describe_mail(mails, f"符合「{q}」的信"), duration_ms=_ms(t0))
    except G.PersonalError as exc:
        return make_result(False, error=str(exc), duration_ms=_ms(t0))


async def reminder_create(text: str, when: str = "", minutes_from_now: int = 0,
                          via: str = "telegram") -> Dict[str, Any]:
    import reminders as R
    from tools._context import get_session_id
    t0 = time.monotonic()
    try:
        if minutes_from_now:
            fire = datetime.now().astimezone() + timedelta(minutes=int(minutes_from_now))
        elif when:
            fire = datetime.fromisoformat(when.strip().replace(" ", "T"))
            fire = fire if fire.tzinfo else fire.astimezone()
        else:
            return make_result(False, error="Give `when` (YYYY-MM-DD HH:MM) or minutes_from_now.", duration_ms=_ms(t0))
        if fire <= datetime.now().astimezone():
            return make_result(False, error=f"{fire:%Y-%m-%d %H:%M} is in the past.", duration_ms=_ms(t0))
        row = await R.create(R.Reminder(fire_at=fire, text=(text or "時間到了").strip()[:200],
                                        via="xiaoai" if via == "xiaoai" else "telegram"),
                             session_id=get_session_id() or "")
        return make_result(True, result=R.describe_created(row), duration_ms=_ms(t0))
    except (ValueError, RuntimeError) as exc:
        return make_result(False, error=str(exc), duration_ms=_ms(t0))


_SCHEMA_CAL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "calendar_events",
        "description": (
            "The asking person's Google Calendar events (read-only, all calendars they show): "
            "times, titles, locations, recurring events expanded. Use for anything that depends "
            "on their schedule. Only works in their Telegram private chat. Can't create or "
            "change events."
        ),
        "parameters": {"type": "object", "properties": {
            "start_date": {"type": "string", "description": "YYYY-MM-DD; empty = today"},
            "days": {"type": "integer", "description": "How many days from start_date (1–31). Default 1."},
        }},
    },
}

_SCHEMA_MAIL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "gmail_unread",
        "description": (
            "Senders, subjects and times of the asking person's Gmail messages (read-only; "
            "nothing is marked read). `query` is Gmail search syntax; default = unread in "
            "Primary from the last 24 hours. Only works in their Telegram private chat. "
            "Can't send, reply or delete."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Gmail search, e.g. 'is:unread from:bank newer_than:7d'"},
            "limit": {"type": "integer", "description": "Max messages (1–30). Default 10."},
        }},
    },
}

_SCHEMA_REM: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "reminder_create",
        "description": (
            "Remind the user once at a given time (Telegram, optionally spoken by the 小愛 "
            "speaker). For things that repeat, a schedule is the right tool, not this."
        ),
        "parameters": {"type": "object", "properties": {
            "text": {"type": "string", "description": "What to remind, in the user's words"},
            "when": {"type": "string", "description": "Local time, YYYY-MM-DD HH:MM"},
            "minutes_from_now": {"type": "integer", "description": "Instead of `when`"},
            "via": {"type": "string", "description": "telegram (default) or xiaoai"},
        }, "required": ["text"]},
    },
}

PERSONAL_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    (calendar_events, _SCHEMA_CAL, "low"),
    (gmail_unread, _SCHEMA_MAIL, "low"),
    (reminder_create, _SCHEMA_REM, "low"),
]
