"""
Event triggers (watchers.py): the right person hears about it, once.

  A. typhoon: everyone's private chat (not groups); once when it heads
     this way, again when it's close; never again for the same stage
  B. meeting: only the person whose calendar it is; once; all-day and
     later events ignored; no calendar grant → nothing
  C. offline: the owner, once per outage, after the threshold; ignored
     devices skipped; back online then offline again → told again;
     quiet hours hold the alert until they end
  D. mail: what was there at first isn't news; a newer important mail
     goes to that person; quiet hours hold it
  E. tick: toggles and intervals respected; one watcher failing doesn't
     stop the rest; state survives a restart; no Telegram → nothing

    .venv/bin/python tests/test_watchers.py
"""
from __future__ import annotations

import asyncio, logging, os, sys, tempfile
from datetime import datetime, timedelta, timezone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from config import config
TMP = tempfile.mkdtemp(prefix="ot-watch-")
config.db_path = os.path.join(TMP, "t.db")
config.telegram_bot_token, config.telegram_default_chat_id = "123:abc", "100"
config.telegram_inbound_chat_id_whitelist = "100, 200, -300"
config.google_client_id, config.google_client_secret = "cid", "secret"
config.watch_quiet_hours = ""

import watchers as W
import google_auth as GA
import google_personal as G
import home_assistant as HA
import typhoon

SENT: list = []


async def send(chat, text):
    SENT.append((chat, text))


async def test_typhoon() -> None:
    storm = {"number": "2617", "name": "FUNG-WONG", "distance": 650, "forecast": [(24, (22, 122), 280)]}
    far = {"number": "2618", "name": "FAR", "distance": 2500, "forecast": [(24, (30, 150), 2900)]}
    async def storms(): return [storm, far]
    typhoon.active_storms = storms
    SENT.clear()
    assert await W.watch_typhoon(send) == 1
    assert sorted(c for c, _ in SENT) == ["100", "200"], "everyone's private chat, not the group"
    assert SENT[0][1].startswith("🌀 颱風往台灣靠近：FUNG-WONG 距離台灣約 650 公里") and "FAR" not in SENT[0][1]
    SENT.clear()
    assert await W.watch_typhoon(send) == 0 and not SENT, "same stage → not again"
    storm["distance"] = 250
    assert await W.watch_typhoon(send) == 1 and SENT[0][1].startswith("🌀 颱風逼近")
    SENT.clear()
    storm["distance"] = 650
    assert await W.watch_typhoon(send) == 0, "moving back out isn't news"
    print("  ✓ typhoon: everyone once when it heads this way, once more when close; groups and far storms skipped")


async def link(person: str, scopes: str) -> None:
    db = await GA._db()
    await db.execute("INSERT OR REPLACE INTO google_accounts(person, email, refresh_token, scopes, linked_at, status) "
                     "VALUES (?, ?, 'x', ?, '2026-11-01', 'ok')", (person, person + "@x", scopes))
    await db.commit()


async def test_meeting() -> None:
    now = datetime.now().astimezone().replace(second=0, microsecond=0)
    await link("tg:200", f"openid {GA.CAL_SCOPE} {GA.MAIL_SCOPE}")
    await link("tg:300", "openid")                                  # linked, no calendar grant
    asked = []
    async def between(start, end, person):
        asked.append(person)
        return [
            {"start": now + timedelta(minutes=10), "end": now + timedelta(minutes=70), "all_day": False,
             "title": "季度檢討", "location": "3F 會議室"},
            {"start": now.date(), "end": now.date(), "all_day": True, "title": "媽媽生日", "location": ""},
            {"start": now + timedelta(minutes=40), "end": now + timedelta(minutes=50), "all_day": False,
             "title": "之後的會", "location": ""},
        ]
    G.events_between = between
    SENT.clear()
    assert await W.watch_meeting(send, now) == 1
    assert SENT == [("200", "📅 10 分鐘後：" + (now + timedelta(minutes=10)).strftime("%H:%M") + " 季度檢討\n📍 3F 會議室")], SENT
    assert asked == ["tg:200"], "no calendar grant → not even asked"
    SENT.clear()
    assert await W.watch_meeting(send, now + timedelta(minutes=5)) == 0, "once per event"
    print("  ✓ meeting: only that person, once; all-day and later events ignored; no grant → nothing")


async def test_offline() -> None:
    now = datetime.now(timezone.utc)
    devs = [
        {"name": "Roborock S5", "area": "客廳", "online": False, "offline_for": "2 小時",
         "offline_since": (now - timedelta(hours=2)).isoformat()},
        {"name": "Mi Control Hub", "area": "", "online": False, "offline_for": "10 分鐘",
         "offline_since": (now - timedelta(minutes=10)).isoformat()},
        {"name": "舊插座", "area": "陽台", "online": False, "offline_for": "30 天",
         "offline_since": (now - timedelta(days=30)).isoformat()},
        {"name": "桌燈", "area": "書房", "online": True, "offline_for": "", "offline_since": ""},
    ]
    async def devices(): return [dict(d) for d in devs]
    HA.devices, HA.configured = devices, (lambda: True)
    config.watch_offline_ignore = "舊插座"
    SENT.clear()
    assert await W.watch_offline(send, now) == 0 and not SENT, "first look: outages already known aren't news"
    devs.append({"name": "空氣淨化器", "area": "臥室", "online": False, "offline_for": "2 小時",
                 "offline_since": (now - timedelta(hours=2)).isoformat()})
    assert await W.watch_offline(send, now) == 1
    assert SENT == [("100", "📴 家裡有設備離線超過 1 小時：\n• 臥室 · 空氣淨化器（2 小時）")], SENT
    SENT.clear()
    assert await W.watch_offline(send, now) == 0, "once per outage"
    devs[-1]["online"] = True
    await W.watch_offline(send, now)
    devs[-1]["online"] = False
    h = datetime.now().hour
    config.watch_quiet_hours = f"{h}-{(h + 1) % 24}"
    assert W.quiet_now() and await W.watch_offline(send, now) == 0 and not SENT, "quiet hours hold it"
    config.watch_quiet_hours = ""
    assert await W.watch_offline(send, now) == 1 and SENT, "back online, then offline again → told again"
    print("  ✓ offline: known outages baselined; owner told once per new outage; ignored skipped; quiet hours wait")


async def test_mail() -> None:
    base = int(datetime.now().timestamp() * 1000)
    mails = [{"id": "a", "from": "台灣銀行", "subject": "帳單", "ts": base - 60000},
             {"id": "b", "from": "GitHub", "subject": "CI failed", "ts": base - 30000}]
    async def unread(query, limit, person):
        assert "is:important" in query and person == "tg:200"
        return list(mails)
    G.unread = unread
    SENT.clear()
    assert await W.watch_mail(send) == 0 and not SENT, "what's already there isn't news"
    mails.insert(0, {"id": "c", "from": "老闆", "subject": "明天的簡報", "ts": base + 1000})
    h = datetime.now().hour
    config.watch_quiet_hours = f"{h}-{(h + 1) % 24}"
    assert await W.watch_mail(send) == 0 and not SENT
    config.watch_quiet_hours = ""
    assert await W.watch_mail(send) == 1
    assert SENT == [("200", "✉️ 重要新信：\n• 老闆：明天的簡報")], SENT
    SENT.clear()
    assert await W.watch_mail(send) == 0
    print("  ✓ mail: the backlog isn't announced; a newer important mail goes to that person; quiet hours wait")


async def test_tick(tracker) -> None:
    async def boom(): raise RuntimeError("JMA down")
    typhoon.active_storms = boom
    W._last_run.clear()
    out = await W.tick(send, force=True)
    assert str(out["typhoon"]).startswith("error") and "offline" in out and "mail" in out, out
    config.watch_mail = "false"
    W._last_run.clear()
    assert "mail" not in await W.tick(send, force=True)
    config.watch_mail = "true"
    W._last_run.clear()
    await W.tick(send)
    assert await W.tick(send) == {}, "intervals: nothing is due a second later"
    rows = await W.keys("offline")
    W._last_run.clear()
    SENT.clear()
    await W.watch_offline(send)
    assert not SENT and rows, "state is in the DB — a restart doesn't repeat alerts"
    config.telegram_bot_token = ""
    W._last_run.clear()
    assert await W.tick(send, force=True) == {}
    config.telegram_bot_token = "123:abc"
    print("  ✓ tick: toggles and intervals respected; a failing watcher doesn't stop the rest; no Telegram → nothing")


async def main() -> None:
    from tracker import Tracker
    tracker = Tracker(config.db_path)
    await tracker.open()
    try:
        await W.start(tracker)
        await W.stop()
        await GA.start(tracker)
        print("A. typhoon"); await test_typhoon()
        print("B. meeting"); await test_meeting()
        print("C. offline"); await test_offline()
        print("D. mail"); await test_mail()
        print("E. tick"); await test_tick(tracker)
    finally:
        await tracker.close()
    print("\nALL WATCHER TESTS PASS")


asyncio.run(main())
