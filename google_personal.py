"""
Your Google calendar and Gmail, read-only — what a personal assistant
needs to know before it can be one.

The simple route, no Google Cloud project:
  • Calendar: the calendar's "Secret address in iCal format" (Settings →
    the calendar → Integrate calendar). Recurring events are expanded.
  • Gmail: IMAP with an app password (Google Account → Security → 2-Step
    Verification → App passwords). The mailbox is opened read-only, so
    nothing is ever marked read, moved or deleted.

Both secrets live in Settings (redacted in exports) and never reach a
prompt. Creating events or sending mail would need Google's OAuth APIs —
a later step, with confirmation before anything is sent.
"""
from __future__ import annotations

import asyncio
import imaplib
import logging
import re
import time
from datetime import date, datetime, timedelta
from email import message_from_bytes
from email.header import decode_header, make_header
from email.utils import parseaddr, parsedate_to_datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config import config

logger = logging.getLogger(__name__)

_WEEKDAY = "一二三四五六日"
_ics_cache: Tuple[float, str, Optional[bytes]] = (0.0, "", None)


class PersonalError(Exception):
    """A Google read that failed, with a message for the user."""


def local_tz():
    return datetime.now().astimezone().tzinfo


def calendar_configured() -> bool:
    return bool((getattr(config, "google_ical_url", "") or "").strip())


def gmail_configured() -> bool:
    return bool((getattr(config, "gmail_address", "") or "").strip()
                and (getattr(config, "gmail_app_password", "") or "").strip())


def day_label(d: date) -> str:
    return f"{d.month}/{d.day}（週{_WEEKDAY[d.weekday()]}）"


# ── Calendar ─────────────────────────────────────────────────────────────────

async def _ics() -> bytes:
    global _ics_cache
    url = (getattr(config, "google_ical_url", "") or "").strip()
    if not url:
        raise PersonalError("還沒連接 Google 行事曆：Settings → Google 貼上「iCal 格式的私人網址」。")
    if _ics_cache[2] is not None and _ics_cache[1] == url and time.monotonic() - _ics_cache[0] < 300:
        return _ics_cache[2]
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            r = await client.get(url)
    except httpx.HTTPError as exc:
        raise PersonalError(f"讀不到 Google 行事曆（{type(exc).__name__}）") from exc
    if r.status_code in (401, 403, 404):
        raise PersonalError("Google 行事曆網址失效了（可能被重設），請重新複製「iCal 格式的私人網址」。")
    if r.status_code >= 400 or b"BEGIN:VCALENDAR" not in r.content[:2000]:
        raise PersonalError(f"Google 行事曆回應異常（HTTP {r.status_code}）")
    _ics_cache = (time.monotonic(), url, r.content)
    return r.content


async def events_between(start: datetime, end: datetime) -> List[Dict[str, Any]]:
    """Events overlapping [start, end) in local time: {start, end, all_day, title, location}."""
    import icalendar
    import recurring_ical_events
    raw = await _ics()
    cal = icalendar.Calendar.from_ical(raw)
    tz = local_tz()
    out = []
    for ev in recurring_ical_events.of(cal).between(start, end):
        if str(ev.get("STATUS", "")).upper() == "CANCELLED":
            continue
        s = ev.get("DTSTART").dt
        e = ev.get("DTEND").dt if ev.get("DTEND") else s
        all_day = isinstance(s, date) and not isinstance(s, datetime)
        if not all_day:
            s = (s if s.tzinfo else s.replace(tzinfo=tz)).astimezone(tz)
            e = (e if e.tzinfo else e.replace(tzinfo=tz)).astimezone(tz)
        out.append({"start": s, "end": e, "all_day": all_day,
                    "title": str(ev.get("SUMMARY") or "（無標題）").strip(),
                    "location": str(ev.get("LOCATION") or "").strip()})
    # By day; within a day, all-day events first, then by start time.
    day = lambda x: x["start"] if x["all_day"] else x["start"].date()
    return sorted(out, key=lambda x: (day(x), not x["all_day"], "" if x["all_day"] else x["start"].isoformat()))


async def events_for_days(first: date, days: int = 1) -> List[Dict[str, Any]]:
    tz = local_tz()
    start = datetime.combine(first, datetime.min.time(), tz)
    return await events_between(start, start + timedelta(days=days))


def describe_events(events: List[Dict[str, Any]], label: str, multi_day: bool = False) -> str:
    if not events:
        return f"📅 {label}沒有行程。"
    lines = [f"📅 {label}的行程（{len(events)}）："]
    for e in events:
        if e["all_day"]:
            when = (day_label(e["start"]) + " " if multi_day else "") + "全天"
        else:
            when = (day_label(e["start"].date()) + " " if multi_day else "") + \
                f"{e['start']:%H:%M}–{e['end']:%H:%M}"
        where = f"（{e['location']}）" if e["location"] else ""
        lines.append(f"• {when} {e['title']}{where}")
    return "\n".join(lines)


# ── Gmail ────────────────────────────────────────────────────────────────────

def _decode(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value))).strip()
    except Exception:  # noqa: BLE001
        return value.strip()


def _unread_blocking(query: str, limit: int) -> List[Dict[str, Any]]:
    addr = (getattr(config, "gmail_address", "") or "").strip()
    pw = (getattr(config, "gmail_app_password", "") or "").replace(" ", "")
    if not addr or not pw:
        raise PersonalError("還沒連接 Gmail：Settings → Google 填 Gmail 地址和應用程式密碼。")
    try:
        imap = imaplib.IMAP4_SSL("imap.gmail.com", 993, timeout=20)
    except OSError as exc:
        raise PersonalError(f"連不到 Gmail（{type(exc).__name__}）") from exc
    try:
        try:
            imap.login(addr, pw)
        except imaplib.IMAP4.error as exc:
            raise PersonalError("Gmail 登入失敗：應用程式密碼不對，或 Gmail 的 IMAP 沒開。") from exc
        imap.select("INBOX", readonly=True)            # read-only: nothing gets marked read
        typ, data = imap.uid("SEARCH", "X-GM-RAW", f'"{query}"')
        uids = (data[0] or b"").split() if typ == "OK" else []
        out = []
        for uid in reversed(uids[-limit:]):
            typ, msg = imap.uid("FETCH", uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])")
            if typ != "OK" or not msg or not isinstance(msg[0], tuple):
                continue
            m = message_from_bytes(msg[0][1])
            name, email_addr = parseaddr(_decode(m.get("From")))
            try:
                when = parsedate_to_datetime(m.get("Date")).astimezone(local_tz())
            except Exception:  # noqa: BLE001
                when = None
            out.append({"from": name or email_addr, "email": email_addr,
                        "subject": _decode(m.get("Subject")) or "（無主旨）", "date": when})
        return out
    finally:
        try:
            imap.logout()
        except Exception:  # noqa: BLE001
            pass


async def unread(query: str = "category:primary is:unread newer_than:1d", limit: int = 8) -> List[Dict[str, Any]]:
    """Unread mail matching a Gmail search (default: Primary tab, last 24 h), newest first."""
    return await asyncio.to_thread(_unread_blocking, query, limit)


def describe_mail(mails: List[Dict[str, Any]], label: str = "未讀的重要信") -> str:
    if not mails:
        return f"✉️ 沒有{label}。"
    lines = [f"✉️ {label}（{len(mails)}）："]
    for m in mails:
        when = f"（{m['date']:%H:%M}）" if m.get("date") and m["date"].date() == datetime.now().date() else (
            f"（{m['date'].month}/{m['date'].day}）" if m.get("date") else "")
        lines.append(f"• {m['from']}：{m['subject'][:60]}{when}")
    return "\n".join(lines)


async def status() -> Dict[str, Any]:
    """For Settings → Test connection."""
    out: Dict[str, Any] = {"calendar": {"configured": calendar_configured()},
                           "gmail": {"configured": gmail_configured()}}
    if calendar_configured():
        try:
            evs = await events_for_days(datetime.now().date(), 7)
            out["calendar"].update(ok=True, events_7d=len(evs))
        except PersonalError as exc:
            out["calendar"].update(ok=False, error=str(exc))
    if gmail_configured():
        try:
            mails = await unread(limit=20)
            out["gmail"].update(ok=True, unread_primary_24h=len(mails))
        except PersonalError as exc:
            out["gmail"].update(ok=False, error=str(exc))
    return out


# ── Asking about it ──────────────────────────────────────────────────────────

_CAL_ASK = re.compile(r"(行程|會議|開會|約會|行事曆|schedule|calendar|meetings?)", re.I)
_MAIL_ASK = re.compile(r"(新信|新郵件|新的信|未讀|重要的信|重要信件|有人寄信|來信|信箱|email|e-mail|郵件|gmail|inbox|mail)", re.I)
_DAYWORD = re.compile(r"今天|今日|明天|明日|後天|這週|本週|這禮拜|下週|下周|下禮拜|一週|today|tomorrow|this week|next week", re.I)
_ASKY = re.compile(r"有|什麼|哪些|嗎|幾封|幾個|多少|查|看|列出|\?|？|any|what|check|show|list|unread|new", re.I)
# Doing something with mail or the calendar (and trip planning) is the agent's job.
_NOT_ASK = re.compile(r"提醒|排程|寄|回信|回覆|寫|草擬|翻譯|新增|建立|加入|取消|刪除|改到|移到|安排|整理|分析|摘要|總結|"
                      r"規劃|計畫|計劃|推薦|建議|旅遊|旅行|準備|做|產生|生成|製作|預約|每天|每週|是什麼|怎麼設定|"
                      r"draft|send|reply|plan|write", re.I)
_CJK = re.compile(r"[㐀-鿿]")


def _short(t: str) -> bool:
    return len(t) <= (30 if _CJK.search(t) else 60)


def parse_calendar_question(text: str) -> Optional[Tuple[date, int, str]]:
    """(first day, days, label) for "今天有什麼行程", "明天的會議", "這週行程"."""
    t = (text or "").strip()
    if (not t or not _short(t) or not _CAL_ASK.search(t) or _NOT_ASK.search(t)
            or not (_DAYWORD.search(t) or _ASKY.search(t))):
        return None
    today = datetime.now().date()
    if re.search(r"下週|下周|下禮拜|next week", t, re.I):
        start = today + timedelta(days=7 - today.weekday())
        return start, 7, "下週"
    if re.search(r"這週|本週|這禮拜|一週|this week", t, re.I):
        return today, 7 - today.weekday(), "這週"
    if re.search(r"後天", t):
        return today + timedelta(days=2), 1, "後天"
    if re.search(r"明天|明日|tomorrow", t, re.I):
        return today + timedelta(days=1), 1, "明天"
    return today, 1, "今天"


def parse_mail_question(text: str) -> bool:
    """ "有什麼新信", "有重要的信嗎", "check my email"."""
    t = (text or "").strip()
    return bool(t and _short(t) and _MAIL_ASK.search(t) and _ASKY.search(t) and not _NOT_ASK.search(t))
