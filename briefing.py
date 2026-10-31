"""
早安簡報 — the morning briefing, assembled without a model.

Weather (and an umbrella), a typhoon only when one is heading for Taiwan,
today's calendar, today's reminders, unread Primary mail, and devices at
home that went offline. Every part is optional: a part that isn't set up
is left out, a part that fails says so in one line, and the rest still
arrives. All parts are fetched at once, so the briefing takes as long as
the slowest source, not the sum.

"早安簡報" / "今日簡報" in chat, or a daily schedule with that goal.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)

_WEEKDAY = "一二三四五六日"
_ASK = re.compile(r"(?:早安|今日|今天的?|每日|晨間)\s*(?:簡報|简报|摘要)|morning briefing|daily briefing|^brief me", re.I)
_NOT = re.compile(r"每天|每日\s*\d|排程|做一份|做個|寫|ppt|投影片|簡報檔|會議|怎麼|如何", re.I)


def parse_briefing_request(text: str) -> bool:
    """ "早安簡報", "給我今日簡報", "產生早安簡報" (a schedule's goal) — not
    "幫我做一份簡報" or "每天早上給我早安簡報" (that one is a schedule)."""
    t = (text or "").strip()
    return bool(t and len(t) <= 24 and _ASK.search(t) and not _NOT.search(t))


async def _part(name: str, coro, timeout: float = 25.0) -> Optional[str]:
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        return f"⚠️ {name}：逾時，稍後再問一次"
    except Exception as exc:  # noqa: BLE001
        logger.info("briefing %s failed: %s", name, exc)
        return f"⚠️ {name}：{exc}" if str(exc) else f"⚠️ {name}：讀取失敗"


async def _weather() -> str:
    import weather
    return "🌤 " + await weather.brief()


async def _typhoon() -> Optional[str]:
    import typhoon
    near = [s for s in await typhoon.active_storms() if typhoon.is_near(s)]
    if not near:
        return None
    return "🌀 " + "、".join(f"颱風 {s['name']}（距台灣約 {s['distance']:,} 公里，預報最近約 {typhoon.closest_km(s):,} 公里）"
                             for s in near) + "，請留意中央氣象署警報"


async def _calendar() -> Optional[str]:
    import google_personal as G
    if not G.calendar_configured():
        return None
    try:
        return G.describe_events(await G.events_for_days(datetime.now().date(), 1), "今天")
    except G.PersonalError as exc:
        return f"⚠️ 行事曆：{exc}"


async def _reminders() -> Optional[str]:
    import reminders as R
    now = datetime.now().astimezone()
    end = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), now.tzinfo)
    rows = await R.between(now, end)
    if not rows:
        return None
    return "⏰ 今天的提醒：\n" + "\n".join(f"• {r['fire_at_local']:%H:%M} {r['text']}" for r in rows)


async def _mail() -> Optional[str]:
    import google_personal as G
    if not G.gmail_configured():
        return None
    try:
        mails = await G.unread(limit=5)
    except G.PersonalError as exc:
        return f"⚠️ Gmail：{exc}"
    return G.describe_mail(mails, "未讀的主要信件（24 小時內）")


async def _home() -> Optional[str]:
    import home_assistant as HA
    if not HA.configured():
        return None
    off = [d for d in await HA.devices() if not d["online"]]
    if not off:
        return None
    names = "、".join(d["name"] for d in off[:6]) + ("…" if len(off) > 6 else "")
    return f"📴 家裡有 {len(off)} 台設備離線：{names}"


async def morning_briefing() -> str:
    today = datetime.now().date()
    parts = await asyncio.gather(
        _part("天氣", _weather()), _part("颱風", _typhoon()), _part("行事曆", _calendar()),
        _part("提醒", _reminders()), _part("Gmail", _mail()), _part("家裡設備", _home()),
    )
    lines: List[str] = [f"☀️ 早安！今天是 {today.month}/{today.day}（週{_WEEKDAY[today.weekday()]}）"]
    lines += [p for p in parts if p]
    import google_personal as G
    if not G.calendar_configured() and not G.gmail_configured():
        lines.append("（行事曆和 Gmail 還沒連接：Settings → Google）")
    return "\n\n".join(lines)
