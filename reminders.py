"""
One-shot reminders: "明天早上 9 點提醒我繳電話費", "30 分鐘後提醒我關火",
"下午 3 點叫小愛提醒我出門", "remind me at 5pm to call mom".

Schedules (scheduler.py) are for things that repeat; a reminder fires
once. Before this, "明天提醒我" went to the model, which either faked a
reminder or made a daily cron job out of it.

Rows live in the tracker DB (table `reminders`), so a restart doesn't lose
them; one background loop sleeps until the next one is due. A reminder
that came due while OpenTeddy was down is still delivered, marked late.
Delivery: Telegram (the chat it was asked from, else the default chat),
and through the 小愛 speaker when the request said so.
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from config import config

logger = logging.getLogger(__name__)

_WEEKDAY = "一二三四五六日"
_tracker: Any = None
_wake: Optional[asyncio.Event] = None
_loop_task: Optional[asyncio.Task] = None

_TABLE = (
    "CREATE TABLE IF NOT EXISTS reminders ("
    "  id TEXT PRIMARY KEY, session_id TEXT NOT NULL DEFAULT '', chat_id TEXT NOT NULL DEFAULT '',"
    "  fire_at TEXT NOT NULL, text TEXT NOT NULL, via TEXT NOT NULL DEFAULT 'telegram',"
    "  status TEXT NOT NULL DEFAULT 'pending', created_at TEXT NOT NULL, sent_at TEXT, error TEXT"
    ")"
)


def local_tz():
    return datetime.now().astimezone().tzinfo


def _now() -> datetime:
    return datetime.now().astimezone()


# ── Understanding the request ────────────────────────────────────────────────

@dataclass
class Reminder:
    fire_at: Optional[datetime]     # None = no time given; ask for one
    text: str
    via: str = "telegram"           # telegram | xiaoai


_CN_DIGIT = {"零": 0, "〇": 0, "一": 1, "二": 2, "兩": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_NUM = r"(?:\d{1,3}|[零〇一二兩两三四五六七八九十]{1,3})"


def _num(s: str) -> int:
    s = s.strip()
    if s.isdigit():
        return int(s)
    if "十" in s:
        a, _, b = s.partition("十")
        return (_CN_DIGIT.get(a, 1) if a else 1) * 10 + (_CN_DIGIT.get(b, 0) if b else 0)
    return _CN_DIGIT.get(s, 0)


_TRIGGER = re.compile(r"提醒我|提醒一下|記得提醒|叫我|remind me", re.I)
_CONDITIONAL = re.compile(r"如果|的話|假如|要是|一旦|完成後|好了再|\bif\b|\bwhen\b", re.I)
_XIAOAI = re.compile(r"(?:叫|讓|請|用|透過|要)?小愛(?:音箱|同學)?(?:跟我)?(?:提醒|說|講|播|叫)")
_REPEAT = re.compile(r"每天|每日|每週|每周|每星期|每個|每月|每年|每隔|每小時|每晚|每早|工作日|平日|"
                     r"週一到週五|周一到周五|every|daily|weekly|weekdays", re.I)
_REL = re.compile(rf"(?P<n>{_NUM}|半)\s*(?P<half>個半)?\s*(?P<u>分鐘|分钟|分|個小時|个小时|小時|小时|鐘頭|钟头)\s*(?:後|后|之後|之后|以後|以后)")
_REL_EN = re.compile(r"\bin\s+(?P<n>\d{1,3})\s*(?P<u>minutes?|mins?|hours?|hrs?)\b", re.I)
_DAY = re.compile(r"大後天|大后天|後天|后天|明天|明日|明早|明晚|今天|今日|今早|今晚|tomorrow|today|tonight|"
                  rf"(?:下|這|这|本)?(?:週|周|星期|禮拜|礼拜)[一二三四五六日天]|"
                  r"(?P<m>\d{1,2})\s*[月/]\s*(?P<d>\d{1,2})\s*(?:日|號|号)?", re.I)
_PERIOD = re.compile(r"凌晨|半夜|早上|早晨|清晨|上午|中午|下午|傍晚|晚上|今早|今晚|明早|明晚")
_CLOCK = re.compile(rf"(?P<h>{_NUM})\s*(?:點|点|時|时)\s*(?:(?P<half>半)|(?P<q>一刻|三刻)|(?P<min>{_NUM})\s*分?)?"
                    r"|(?P<h2>\d{1,2})\s*[:：]\s*(?P<min2>\d{2})")
_CLOCK_EN = re.compile(r"\bat\s+(?P<h>\d{1,2})(?::(?P<min>\d{2}))?\s*(?P<ap>am|pm)?\b", re.I)
_PERIOD_DEFAULT = {"凌晨": 6, "半夜": 0, "早上": 8, "早晨": 8, "清晨": 6, "今早": 8, "明早": 8,
                   "上午": 9, "中午": 12, "下午": 14, "傍晚": 17, "晚上": 20, "今晚": 20, "明晚": 20}
_FILLER = re.compile(r"^(?:要|記得|记得|一下|:|：|,|，|\s|to\b|that\b)+|(?:喔|哦|噢|唷|吧|啊|呀|。|！|!|\s)+$", re.I)


def _resolve_day(m: re.Match, today: date) -> date:
    s = m.group(0).lower()
    if m.group("m"):
        d = date(today.year, int(m.group("m")), int(m.group("d")))
        return d if d >= today else date(today.year + 1, d.month, d.day)
    if s in ("大後天", "大后天"):
        return today + timedelta(days=3)
    if s in ("後天", "后天"):
        return today + timedelta(days=2)
    if s in ("明天", "明日", "明早", "明晚", "tomorrow"):
        return today + timedelta(days=1)
    if s in ("今天", "今日", "今早", "今晚", "today", "tonight"):
        return today
    wd = "一二三四五六日天".index(s[-1]) % 7
    if s[0] == "下":                      # 下週X: that day in next (Monday-based) week
        return today + timedelta(days=7 - today.weekday() + wd)
    return today + timedelta(days=(wd - today.weekday()) % 7)


def parse_reminder(text: str, now: Optional[datetime] = None) -> Optional[Reminder]:
    """A one-shot reminder request, or None. Recurring wording ("每天…")
    is a schedule, not a reminder."""
    t = (text or "").strip()
    if not t or len(t) > 160 or _REPEAT.search(t) or _CONDITIONAL.search(t):
        return None
    trig = _TRIGGER.search(t)
    xiao = _XIAOAI.search(t)
    if not trig and not xiao:
        return None
    if re.search(r"取消|刪除|删除|有哪些|清單|列表|多少個|怎麼用|是什麼", t):
        return None
    now = now or _now()
    tz = now.tzinfo
    today = now.date()
    spans: List[Tuple[int, int]] = []
    fire: Optional[datetime] = None

    rel = _REL.search(t) or _REL_EN.search(t)
    if rel:
        spans.append(rel.span())
        n = rel.group("n")
        amount = 0.5 if n == "半" else float(_num(n))
        if rel.groupdict().get("half"):
            amount += 0.5
        unit = rel.group("u").lower()
        minutes = amount * 60 if unit[0] in "小個个鐘钟h" else amount
        fire = now + timedelta(minutes=minutes)
    else:
        day_m = _DAY.search(t)
        period_m = _PERIOD.search(t)
        clock_m = _CLOCK.search(t) or _CLOCK_EN.search(t)
        if clock_m and clock_m.re is _CLOCK and clock_m.group("h") == "一" and clock_m.start() > 0 \
                and t[clock_m.start() - 1] in "早晚快慢多少有差":
            clock_m = None                        # "晚一點" is "a bit later", not 1 o'clock
        for m in (day_m, period_m, clock_m):
            if m:
                spans.append(m.span())
        if day_m or period_m or clock_m:
            day = _resolve_day(day_m, today) if day_m else None
            period = period_m.group(0) if period_m else ""
            if not day and period in ("明早", "明晚"):
                day = today + timedelta(days=1)
            if clock_m:
                g = clock_m.groupdict()
                if clock_m.re is _CLOCK_EN:
                    hour, minute = int(g["h"]), int(g["min"] or 0)
                    ap = (g.get("ap") or "").lower()
                    if ap == "pm" and hour < 12:
                        hour += 12
                    elif ap == "am" and hour == 12:
                        hour = 0
                elif g.get("h2"):
                    hour, minute = int(g["h2"]), int(g["min2"])
                else:
                    hour = _num(g["h"])
                    minute = 30 if g.get("half") else 15 if g.get("q") == "一刻" else 45 if g.get("q") else (
                        _num(g["min"]) if g.get("min") else 0)
                if period in ("下午", "傍晚", "晚上", "今晚", "明晚") and hour < 12:
                    hour += 12
                elif period == "中午" and hour < 6:
                    hour += 12
                elif period in ("半夜", "凌晨") and hour == 12:
                    hour = 0
            elif not period and (day is None or day == today):
                hour = minute = -1                # "今天提醒我…": no time to go on
            else:
                hour, minute = _PERIOD_DEFAULT.get(period, 9), 0
            if hour > 23 or minute > 59:
                return None
            if hour < 0:
                fire = None
            else:
                at = lambda d: datetime.combine(d, datetime.min.time(), tz).replace(hour=hour, minute=minute)
                if day is None:
                    day = today if at(today) > now else today + timedelta(days=1)
                elif at(day) <= now and day_m and re.search(r"[週周期拜][一二三四五六日天]$", day_m.group(0)) \
                        and not day_m.group(0).startswith("下"):
                    day += timedelta(days=7)      # "週五" said on a Friday evening = next Friday
                fire = at(day)

    if xiao and not trig and fire is None:
        return None                           # "叫小愛說…" right now is a home command
    # What to say: the message minus the trigger and the time words.
    cut = sorted(spans + ([trig.span()] if trig else []) + ([xiao.span()] if xiao else []))
    pieces, pos = [], 0
    for a, b in cut:
        if a >= pos:
            pieces.append(t[pos:a])
            pos = b
        else:
            pos = max(pos, b)
    pieces.append(t[pos:])
    body = " ".join(p.strip(" ，,。:：") for p in pieces if p.strip(" ，,。:：")).strip()
    body = re.sub(r"^(?:請|请|幫我|帮我|麻煩|麻烦|please)\s*", "", body, flags=re.I)
    body = re.sub(r"(?:的時候|的时候|左右|整)\s*", "", body)
    body = _FILLER.sub("", body).strip()
    if not body:
        body = "時間到了"
    return Reminder(fire_at=fire, text=body[:200], via="xiaoai" if xiao else "telegram")


_LIST = re.compile(r"(我有|有哪些|有什麼|有什么|目前的|所有的?|查看|看一下|列出).{0,4}提醒|提醒(清單|列表|有哪些)|"
                   r"list (my )?reminders|my reminders", re.I)
_CANCEL = re.compile(r"(取消|刪除|删除|不用|不要)(了)?.{0,6}提醒|提醒.{0,4}(取消|刪掉|删掉|不用了)|cancel.{0,10}reminder", re.I)


def parse_list(text: str) -> bool:
    t = (text or "").strip()
    return bool(t and len(t) <= 30 and _LIST.search(t) and not _CANCEL.search(t))


def parse_cancel(text: str) -> Optional[str]:
    """'' = cancel the only / latest; '3f2a1b' = that one; None = not a cancel."""
    t = (text or "").strip()
    if not t or len(t) > 40 or not _CANCEL.search(t) or re.search(r"排程|schedule", t, re.I):
        return None
    m = re.search(r"\b([0-9a-f]{4,32})\b", t)
    return m.group(1) if m else ""


# ── Storage ──────────────────────────────────────────────────────────────────

def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).astimezone(local_tz())


def _row(r: Any) -> Dict[str, Any]:
    d = dict(r)
    d["fire_at_local"] = _parse_iso(d["fire_at"])
    return d


async def _db():
    if _tracker is None:
        raise RuntimeError("reminders not started")
    db = _tracker.db
    await db.execute(_TABLE)
    return db


async def create(rem: Reminder, session_id: str = "", chat_id: str = "") -> Dict[str, Any]:
    db = await _db()
    # The same request twice (a resend, "asked again") isn't two reminders.
    async with db.execute(
        "SELECT * FROM reminders WHERE status='pending' AND text=? AND session_id=?",
        (rem.text, session_id),
    ) as cur:
        for r in await cur.fetchall():
            if abs((_parse_iso(r["fire_at"]) - rem.fire_at).total_seconds()) < 90:
                return {**_row(r), "duplicate": True}
    rid = uuid.uuid4().hex
    await db.execute(
        "INSERT INTO reminders(id, session_id, chat_id, fire_at, text, via, status, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
        (rid, session_id, str(chat_id or ""), _iso(rem.fire_at), rem.text, rem.via, _iso(_now())),
    )
    await db.commit()
    if _wake:
        _wake.set()
    async with db.execute("SELECT * FROM reminders WHERE id=?", (rid,)) as cur:
        return _row(await cur.fetchone())


async def pending(session_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """One conversation's pending reminders. session_id=None (admin
    endpoint, startup count only) means all of them — never use it to
    answer a person: someone else's reminders are private."""
    db = await _db()
    sql, args = "SELECT * FROM reminders WHERE status='pending'", []
    if session_id is not None:
        sql += " AND session_id=?"
        args.append(session_id)
    async with db.execute(sql + " ORDER BY fire_at", args) as cur:
        return [_row(r) for r in await cur.fetchall()]


async def between(start: datetime, end: datetime, session_id: str) -> List[Dict[str, Any]]:
    db = await _db()
    async with db.execute(
        "SELECT * FROM reminders WHERE status='pending' AND session_id=? AND fire_at>=? AND fire_at<? "
        "ORDER BY fire_at", (session_id, _iso(start), _iso(end)),
    ) as cur:
        return [_row(r) for r in await cur.fetchall()]


async def cancel(prefix: str, session_id: str) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """(cancelled row, candidates when it's ambiguous). Only this
    conversation's reminders can be cancelled."""
    rows = await pending(session_id)
    if prefix:
        rows = [r for r in rows if r["id"].startswith(prefix.lower())]
    elif len(rows) > 1:
        # "取消提醒" with several: the one created last.
        rows = sorted(rows, key=lambda r: r["created_at"])[-1:]
    if len(rows) != 1:
        return None, rows
    db = await _db()
    await db.execute("UPDATE reminders SET status='cancelled' WHERE id=?", (rows[0]["id"],))
    await db.commit()
    return rows[0], []


# ── Saying it ────────────────────────────────────────────────────────────────

def when_label(dt: datetime, now: Optional[datetime] = None) -> str:
    now = now or _now()
    d = dt.date()
    diff = (d - now.date()).days
    day = {0: "今天", 1: "明天", 2: "後天"}.get(diff, f"{d.month}/{d.day}")
    return f"{day}（週{_WEEKDAY[d.weekday()]}）{dt:%H:%M}"


def describe_created(row: Dict[str, Any]) -> str:
    via = "，到時小愛會說出來" if row["via"] == "xiaoai" else ""
    again = "（已經設過了）" if row.get("duplicate") else ""
    return (f"⏰ 好，{when_label(row['fire_at_local'])}提醒你：{row['text']}{via}{again}\n"
            f"取消請說「取消提醒 {row['id'][:6]}」")


def describe_pending(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "⏰ 目前沒有待提醒的事。"
    lines = [f"⏰ 待提醒（{len(rows)}）："]
    for r in rows:
        via = " 🔊" if r["via"] == "xiaoai" else ""
        lines.append(f"• {when_label(r['fire_at_local'])} {r['text']}{via} · `{r['id'][:6]}`")
    return "\n".join(lines)


# ── Firing ───────────────────────────────────────────────────────────────────

async def _deliver(r: Dict[str, Any]) -> Optional[str]:
    """Send one reminder; returns an error when nothing reached the user."""
    late = (_now() - r["fire_at_local"]).total_seconds()
    note = ""
    if late > 120:
        note = f"\n（晚了 {int(late // 60)} 分鐘：原定 {when_label(r['fire_at_local'])}，那時 OpenTeddy 沒在執行）"
    msg = f"⏰ 提醒：{r['text']}{note}"
    errors: List[str] = []
    delivered = False
    if r["via"] == "xiaoai" and late < 1800:
        try:
            import home_assistant as HA
            await HA.xiaoai(f"提醒你，{r['text']}")
            delivered = True
        except Exception as exc:  # noqa: BLE001
            errors.append(f"小愛沒說出來：{exc}")
    chat = r.get("chat_id") or ""
    if not chat and r.get("session_id") and _tracker is not None:
        try:
            sess = await _tracker.get_session(r["session_id"])
            chat = (sess or {}).get("telegram_chat_id") or ""
        except Exception:  # noqa: BLE001
            pass
    chat = chat or (getattr(config, "telegram_default_chat_id", "") or "").strip()
    if chat and (getattr(config, "telegram_bot_token", "") or "").strip():
        try:
            from telegram_bridge import _send_rich
            await _send_rich(chat, msg + ("\n⚠️ " + "；".join(errors) if errors else ""))
            delivered = True
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Telegram 沒送出：{exc}")
    elif not delivered:
        errors.append("沒有可送的地方（Telegram 沒設定）")
    return None if delivered else "；".join(errors)


async def fire_due() -> int:
    db = await _db()
    async with db.execute(
        "SELECT * FROM reminders WHERE status='pending' AND fire_at<=? ORDER BY fire_at", (_iso(_now()),)
    ) as cur:
        due = [_row(r) for r in await cur.fetchall()]
    for r in due:
        err = await _deliver(r)
        await db.execute("UPDATE reminders SET status=?, sent_at=?, error=? WHERE id=?",
                         ("sent" if err is None else "failed", _iso(_now()), err, r["id"]))
        await db.commit()
        logger.info("Reminder %s %s: %r", r["id"][:6], "sent" if err is None else f"failed ({err})", r["text"])
    return len(due)


async def _next_due() -> Optional[datetime]:
    db = await _db()
    async with db.execute("SELECT MIN(fire_at) FROM reminders WHERE status='pending'") as cur:
        row = await cur.fetchone()
    return _parse_iso(row[0]) if row and row[0] else None


async def _run() -> None:
    while True:
        try:
            await fire_due()
            nxt = await _next_due()
            wait = 300.0 if nxt is None else max(0.5, min(300.0, (nxt - _now()).total_seconds()))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reminders loop: %s", exc)
            wait = 30.0
        _wake.clear()
        try:
            await asyncio.wait_for(_wake.wait(), timeout=wait)
        except asyncio.TimeoutError:
            pass


async def start(tracker: Any) -> None:
    """Called from the app lifespan: create the table, start the loop."""
    global _tracker, _wake, _loop_task
    _tracker = tracker
    await _db()
    await tracker.db.commit()
    _wake = asyncio.Event()
    if _loop_task is None or _loop_task.done():
        _loop_task = asyncio.create_task(_run(), name="reminders")
    n = len(await pending())
    logger.info("Reminders: %d pending", n)


async def stop() -> None:
    global _loop_task
    if _loop_task:
        _loop_task.cancel()
        _loop_task = None
