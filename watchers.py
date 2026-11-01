"""
Event triggers — OpenTeddy notices things and tells the right person,
without being asked.

Four watchers, each a cheap poll with no model call:

  typhoon   a storm heading for Taiwan → everyone's Telegram; once when it
            starts heading this way, once more when it is within 300 km
  meeting   N minutes (default 15) before an event on a linked person's
            Google Calendar → that person only
  offline   a home device offline for over N minutes (default 60) → the
            owner; once per outage, again only after it has come back.
            Devices already offline on the very first look aren't announced
  mail      new unread mail Gmail marks important (Primary) → that person
            only; what was already there on the first look isn't announced

Who was told what is kept in the tracker DB (`watch_state`, hashed keys —
no titles or subjects), so a restart doesn't repeat an alert. Offline and
mail alerts wait out quiet hours (default 23–7) and go out afterwards if
they still apply. Each watcher can be turned off in Settings → 🔔.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from config import config

logger = logging.getLogger(__name__)

_tracker: Any = None
_task: Optional[asyncio.Task] = None
_last_run: Dict[str, float] = {}
INTERVALS = {"typhoon": 1800, "meeting": 300, "offline": 300, "mail": 600}
TICK_S = 60

_TABLE = ("CREATE TABLE IF NOT EXISTS watch_state ("
          "  watcher TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL DEFAULT '',"
          "  updated_at TEXT NOT NULL, PRIMARY KEY (watcher, key))")


def _on(name: str) -> bool:
    return str(getattr(config, f"watch_{name}", "true")).strip().lower() not in ("0", "false", "off", "no", "")


def _h(*parts: Any) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:20]


def quiet_now(now: Optional[datetime] = None) -> bool:
    """Inside quiet hours ("23-7"; empty = never quiet)."""
    raw = (getattr(config, "watch_quiet_hours", "23-7") or "").strip()
    try:
        a, b = (int(x) for x in raw.split("-"))
    except ValueError:
        return False
    h = (now or datetime.now()).hour
    return (a <= h or h < b) if a > b else (a <= h < b)


# ── State ────────────────────────────────────────────────────────────────────

async def _db():
    db = _tracker.db
    await db.execute(_TABLE)
    return db


async def seen(watcher: str, key: str) -> Optional[str]:
    db = await _db()
    async with db.execute("SELECT value FROM watch_state WHERE watcher=? AND key=?", (watcher, key)) as cur:
        row = await cur.fetchone()
    return row[0] if row else None


async def remember(watcher: str, key: str, value: str = "1") -> None:
    db = await _db()
    await db.execute("INSERT INTO watch_state(watcher, key, value, updated_at) VALUES (?, ?, ?, ?) "
                     "ON CONFLICT(watcher, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                     (watcher, key, value, datetime.now(timezone.utc).isoformat()))
    await db.commit()


async def forget(watcher: str, key: str) -> None:
    db = await _db()
    await db.execute("DELETE FROM watch_state WHERE watcher=? AND key=?", (watcher, key))
    await db.commit()


async def keys(watcher: str) -> Dict[str, str]:
    db = await _db()
    async with db.execute("SELECT key, value FROM watch_state WHERE watcher=?", (watcher,)) as cur:
        return {r[0]: r[1] for r in await cur.fetchall()}


# ── Who to tell ──────────────────────────────────────────────────────────────

def telegram_ready() -> bool:
    return bool((getattr(config, "telegram_bot_token", "") or "").strip())


def owner_chat() -> str:
    return (getattr(config, "telegram_default_chat_id", "") or "").strip()


def household_chats() -> List[str]:
    """Everyone's private chat: the whitelist's people plus the owner."""
    raw = getattr(config, "telegram_inbound_chat_id_whitelist", "") or ""
    chats = [c.strip() for c in raw.split(",") if c.strip().isdigit() and int(c.strip()) > 0]
    if owner_chat() and owner_chat() not in chats:
        chats.insert(0, owner_chat())
    return chats


async def linked_people(scope: Optional[str] = None) -> List[str]:
    """People (tg:<chat>) who linked Google, with `scope` granted."""
    import google_auth as GA
    if not GA.configured():
        return []
    try:
        db = await GA._db()
        async with db.execute("SELECT person, scopes FROM google_accounts WHERE status='ok'") as cur:
            rows = await cur.fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [r[0] for r in rows if not scope or scope in (r[1] or "").split()]


Sender = Callable[[str, str], Awaitable[None]]


async def _send(chat: str, text: str) -> None:
    from telegram_bridge import _send_rich
    await _send_rich(chat, text)


# ── Watchers ─────────────────────────────────────────────────────────────────

async def watch_typhoon(send: Sender) -> int:
    import typhoon
    sent = 0
    for s in await typhoon.active_storms():
        if not typhoon.is_near(s):
            continue
        stage = "close" if s["distance"] <= typhoon.NOW_NEAR_KM else "near"
        key = _h(s.get("number") or s.get("name"))
        before = await seen("typhoon", key)
        if before == stage or (before == "close" and stage == "near"):
            continue
        head = "🌀 颱風逼近" if stage == "close" else "🌀 颱風往台灣靠近"
        text = (f"{head}：{s.get('name') or '颱風'} 距離台灣約 {s['distance']:,} 公里，"
                f"預報最近約 {typhoon.closest_km(s):,} 公里（日本氣象廳）。\n"
                "記得收好陽台的東西、檢查門窗。警報以中央氣象署為準。")
        for chat in household_chats():
            await send(chat, text)
        await remember("typhoon", key, stage)
        sent += 1
    return sent


async def watch_meeting(send: Sender, now: Optional[datetime] = None) -> int:
    import google_auth as GA
    import google_personal as G
    lead = int(getattr(config, "watch_meeting_minutes", 15) or 15)
    now = now or datetime.now().astimezone()
    sent = 0
    for person in await linked_people(GA.CAL_SCOPE):
        try:
            events = await G.events_between(now, now + timedelta(minutes=lead + 1), person)
        except G.PersonalError as exc:
            logger.info("watch meeting: %s: %s", person, exc)
            continue
        for e in events:
            if e["all_day"] or not (now < e["start"] <= now + timedelta(minutes=lead + 1)):
                continue
            key = _h(person, e["title"], e["start"].isoformat())
            if await seen("meeting", key):
                continue
            mins = max(1, round((e["start"] - now).total_seconds() / 60))
            where = f"\n📍 {e['location']}" if e["location"] else ""
            await send(person.split(":", 1)[1], f"📅 {mins} 分鐘後：{e['start']:%H:%M} {e['title']}{where}")
            await remember("meeting", key)
            sent += 1
    return sent


async def watch_offline(send: Sender, now: Optional[datetime] = None) -> int:
    import home_assistant as HA
    if not HA.configured() or not owner_chat():
        return 0
    after = int(getattr(config, "watch_offline_minutes", 60) or 60)
    ignore = {x.strip() for x in str(getattr(config, "watch_offline_ignore", "") or "").split(",") if x.strip()}
    now = now or datetime.now(timezone.utc)
    devs = await HA.devices()
    known = await keys("offline")
    current: Set[str] = set()
    due = []
    for d in devs:
        if d["online"] or d["name"] in ignore:
            continue
        key = _h(d["area"], d["name"])
        current.add(key)
        try:
            since = datetime.fromisoformat(d.get("offline_since") or "")
        except ValueError:
            continue
        if since.tzinfo is None:
            since = since.replace(tzinfo=timezone.utc)
        if (now - since).total_seconds() >= after * 60 and key not in known:
            due.append((key, d))
    for key in set(known) - current - {"_baseline"}:   # back online → a later outage is news again
        await forget("offline", key)
    if "_baseline" not in known:            # first look: long-known outages aren't news
        for key, _ in due:
            await remember("offline", key)
        await remember("offline", "_baseline")
        return 0
    if not due or quiet_now():
        return 0
    lines = [f"• {d['area'] + ' · ' if d['area'] else ''}{d['name']}（{d['offline_for']}）" for _, d in due]
    await send(owner_chat(), "📴 家裡有設備離線超過 "
               + (f"{after // 60} 小時" if after % 60 == 0 else f"{after} 分鐘") + "：\n" + "\n".join(lines))
    for key, _ in due:
        await remember("offline", key)
    return len(due)


async def watch_mail(send: Sender) -> int:
    import google_auth as GA
    import google_personal as G
    sent = 0
    for person in await linked_people(GA.MAIL_SCOPE):
        try:
            mails = await G.unread("is:unread is:important category:primary newer_than:2d", 10, person)
        except G.PersonalError as exc:
            logger.info("watch mail: %s: %s", person, exc)
            continue
        mark = await seen("mail", _h(person, "since"))
        newest = max([m.get("ts", 0) for m in mails] + [int(mark or 0)])
        if mark is None:                     # first look: what's already there isn't news
            await remember("mail", _h(person, "since"), str(newest or int(time.time() * 1000)))
            continue
        fresh = [m for m in mails if m.get("ts", 0) > int(mark)]
        if not fresh or quiet_now():
            continue
        lines = [f"• {m['from']}：{m['subject'][:60]}" for m in fresh[:3]]
        more = f"\n…還有 {len(fresh) - 3} 封" if len(fresh) > 3 else ""
        await send(person.split(":", 1)[1], "✉️ 重要新信：\n" + "\n".join(lines) + more)
        await remember("mail", _h(person, "since"), str(newest))
        sent += len(fresh)
    return sent


WATCHERS = {"typhoon": watch_typhoon, "meeting": watch_meeting, "offline": watch_offline, "mail": watch_mail}


async def tick(send: Sender = _send, force: bool = False) -> Dict[str, Any]:
    """Run whichever watchers are due. Returns {watcher: count | error}."""
    out: Dict[str, Any] = {}
    if not telegram_ready():
        return out
    for name, fn in WATCHERS.items():
        if not _on(name):
            continue
        if not force and time.monotonic() - _last_run.get(name, -1e9) < INTERVALS[name]:
            continue
        _last_run[name] = time.monotonic()
        try:
            out[name] = await fn(send)
        except Exception as exc:  # noqa: BLE001 — one watcher's trouble never stops the others
            logger.warning("watch %s failed: %s", name, exc)
            out[name] = f"error: {exc}"
    if any(isinstance(v, int) and v for v in out.values()):
        logger.info("Watchers sent: %s", {k: v for k, v in out.items() if v})
    return out


async def _loop() -> None:
    await asyncio.sleep(30)                  # let the rest of startup settle
    while True:
        try:
            await tick()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("Watchers tick failed: %s", exc)
        await asyncio.sleep(TICK_S)


async def start(tracker: Any) -> None:
    global _tracker, _task
    _tracker = tracker
    await _db()
    await tracker.db.commit()
    if _task is None or _task.done():
        _task = asyncio.create_task(_loop(), name="watchers")
    logger.info("Watchers: %s", ", ".join(n for n in WATCHERS if _on(n)) or "all off")


async def stop() -> None:
    global _task
    if _task:
        _task.cancel()
        _task = None
