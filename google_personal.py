"""
Each person's own Google calendar and Gmail, read-only.

Data is fetched with the token of the person asking (google_auth.py) — the
one behind a Telegram private chat (privacy.py). There is no machine-wide
account: the web UI, CLI, MCP and group chats can't read anyone's
calendar or mail, and a task that does is marked private so its answer
isn't stored or shown in the web UI.

  • Calendar API: every calendar the person shows in Google Calendar,
    recurring events expanded by Google, declined and cancelled ones left out.
  • Gmail API, gmail.readonly: headers fetched with format=metadata;
    nothing is marked read, moved, sent or deleted.
"""
from __future__ import annotations

import asyncio
import logging
import re
import urllib.parse
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx

import google_auth as GA
import privacy

logger = logging.getLogger(__name__)

CAL_BASE = "https://www.googleapis.com/calendar/v3"
GMAIL_BASE = "https://gmail.googleapis.com/gmail/v1"
_WEEKDAY = "一二三四五六日"

ONLY_TELEGRAM = ("🔒 個人的行事曆和信件只在本人的 Telegram 私訊裡使用（網頁、CLI、群組都不行）。"
                 "在 Telegram 私訊 bot 說「連結我的 Google」就可以開始。")


class PersonalError(Exception):
    """A Google read that failed, with a message for the person."""


def local_tz():
    return datetime.now().astimezone().tzinfo


def day_label(d: date) -> str:
    return f"{d.month}/{d.day}（週{_WEEKDAY[d.weekday()]}）"


def person_or_raise(person: Optional[str] = None) -> str:
    p = person or privacy.current_person()
    if not p:
        raise PersonalError(ONLY_TELEGRAM)
    return p


async def _get(person: str, url: str, params: Any = None) -> Dict[str, Any]:
    try:
        token = await GA.access_token(person)
    except GA.GoogleAuthError as exc:
        raise PersonalError(str(exc)) from exc
    async with httpx.AsyncClient(timeout=20.0) as client:
        for attempt in (0, 1):
            r = await client.get(url, params=params, headers={"Authorization": f"Bearer {token}"})
            if r.status_code == 401 and attempt == 0:
                try:
                    token = await GA.access_token(person, force=True)
                except GA.GoogleAuthError as exc:
                    raise PersonalError(str(exc)) from exc
                continue
            break
    if r.status_code == 403:
        what = "行事曆" if url.startswith(CAL_BASE) else "Gmail"
        raise PersonalError(f"Google 沒有授權讀取你的{what}：重新說「連結我的 Google」，同意畫面裡兩項都要勾選。")
    if r.status_code >= 400:
        raise PersonalError(f"Google 回應錯誤（HTTP {r.status_code}），請稍後再試。")
    return r.json()


# ── Calendar ─────────────────────────────────────────────────────────────────

def _when(t: Dict[str, Any], tz) -> Tuple[Any, bool]:
    if t.get("dateTime"):
        return datetime.fromisoformat(t["dateTime"].replace("Z", "+00:00")).astimezone(tz), False
    return date.fromisoformat(t["date"]), True


async def events_between(start: datetime, end: datetime, person: Optional[str] = None) -> List[Dict[str, Any]]:
    """Events overlapping [start, end) on the calendars the person shows: {start, end, all_day, title, location}."""
    who = person_or_raise(person)
    tz = local_tz()
    cals = await _get(who, f"{CAL_BASE}/users/me/calendarList", {"minAccessRole": "reader"})
    ids = [c["id"] for c in cals.get("items", []) if c.get("primary") or (c.get("selected") and not c.get("hidden"))][:12]
    window = {"timeMin": start.isoformat(), "timeMax": end.isoformat(), "singleEvents": "true",
              "orderBy": "startTime", "maxResults": "100"}
    pages = await asyncio.gather(*(_get(who, f"{CAL_BASE}/calendars/{urllib.parse.quote(i, safe='')}/events", window)
                                   for i in ids), return_exceptions=True)
    failed = [p for p in pages if isinstance(p, Exception)]
    if pages and len(failed) == len(pages):     # one bad shared calendar doesn't sink the rest
        raise failed[0] if isinstance(failed[0], PersonalError) else PersonalError("讀不到行事曆，請稍後再試。")
    out, seen = [], set()
    for page in pages:
        if isinstance(page, Exception):
            continue
        for ev in page.get("items", []):
            if ev.get("status") == "cancelled":
                continue
            if any(a.get("self") and a.get("responseStatus") == "declined" for a in ev.get("attendees", [])):
                continue
            s, all_day = _when(ev.get("start") or {}, tz)
            e, _ = _when(ev.get("end") or ev.get("start") or {}, tz)
            key = (ev.get("iCalUID") or ev.get("id"), str(s))
            if key in seen:                     # the same event on two shared calendars
                continue
            seen.add(key)
            out.append({"start": s, "end": e, "all_day": all_day,
                        "title": (ev.get("summary") or "（無標題）").strip(),
                        "location": (ev.get("location") or "").strip()})
    privacy.mark()
    # By day; within a day, all-day events first, then by start time.
    day = lambda x: x["start"] if x["all_day"] else x["start"].date()
    return sorted(out, key=lambda x: (day(x), not x["all_day"], "" if x["all_day"] else x["start"].isoformat()))


async def events_for_days(first: date, days: int = 1, person: Optional[str] = None) -> List[Dict[str, Any]]:
    tz = local_tz()
    start = datetime.combine(first, datetime.min.time(), tz)
    return await events_between(start, start + timedelta(days=days), person)


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

def _sender(value: str) -> Tuple[str, str]:
    from email.utils import parseaddr
    name, addr = parseaddr(value or "")
    return (name or addr).strip(), addr


async def unread(query: str = "category:primary is:unread newer_than:1d", limit: int = 8,
                 person: Optional[str] = None) -> List[Dict[str, Any]]:
    """Messages matching a Gmail search (default: unread in Primary, last 24 h), newest first."""
    who = person_or_raise(person)
    listing = await _get(who, f"{GMAIL_BASE}/users/me/messages", {"q": query, "maxResults": str(limit)})
    ids = [m["id"] for m in listing.get("messages", [])][:limit]
    params = [("format", "metadata"), ("metadataHeaders", "From"), ("metadataHeaders", "Subject"),
              ("metadataHeaders", "Date")]
    msgs = await asyncio.gather(*(_get(who, f"{GMAIL_BASE}/users/me/messages/{i}", params) for i in ids))
    out = []
    for m in msgs:
        h = {x.get("name", "").lower(): x.get("value", "") for x in (m.get("payload") or {}).get("headers", [])}
        name, addr = _sender(h.get("from", ""))
        when = None
        if m.get("internalDate"):
            when = datetime.fromtimestamp(int(m["internalDate"]) / 1000).astimezone(local_tz())
        out.append({"from": name or addr, "email": addr, "subject": (h.get("subject") or "").strip() or "（無主旨）",
                    "date": when})
    privacy.mark()
    return out


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
    """For Settings → Test: is the OAuth client set, how many people linked (no names)."""
    out: Dict[str, Any] = {"configured": GA.configured()}
    try:
        out["linked"] = await GA.count()
    except Exception:  # noqa: BLE001
        out["linked"] = 0
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
