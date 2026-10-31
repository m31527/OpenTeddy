"""
Personal assistant basics: Google calendar (iCal) + Gmail (IMAP),
one-shot reminders, the morning briefing.

  A. calendar: recurring / all-day / timezone / cancelled events from an
     iCal feed; cache; a dead URL explained; questions parsed
  B. Gmail: read-only select, Gmail search, encoded headers decoded,
     newest first; wrong app password explained; questions parsed
  C. reminders: parsing (relative, clock, day, period, 小愛, English;
     recurring and conditional left alone); create / dedupe / list /
     cancel; delivery to Telegram and 小愛; late delivery; the loop fires
  D. briefing: every part, a failing part doesn't sink the rest
  E. orchestrator direct paths: reminder, list, cancel, calendar, mail,
     briefing; schedules don't swallow reminders; tools; log redaction

    .venv/bin/python tests/test_personal.py
"""
from __future__ import annotations

import asyncio, imaplib, logging, os, sys, tempfile, threading, time, types
from datetime import date, datetime, timedelta
from email.header import Header
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["TZ"] = "Asia/Taipei"
time.tzset()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from config import config
import google_personal as G
import reminders as R

TODAY = datetime.now().date()
HITS = {"ics": 0}


def _ics() -> bytes:
    d = lambda x: x.strftime("%Y%m%d")
    tomorrow = TODAY + timedelta(days=1)
    return f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//Google Inc//Google Calendar 70.9054//EN
X-WR-TIMEZONE:Asia/Taipei
BEGIN:VEVENT
DTSTART:20260105T010000Z
DTEND:20260105T013000Z
RRULE:FREQ=WEEKLY;BYDAY=MO
SUMMARY:週會
UID:weekly@x
END:VEVENT
BEGIN:VEVENT
DTSTART;VALUE=DATE:{d(tomorrow)}
DTEND;VALUE=DATE:{d(tomorrow + timedelta(days=1))}
SUMMARY:媽媽生日
UID:allday@x
END:VEVENT
BEGIN:VEVENT
DTSTART;TZID=Asia/Taipei:{d(TODAY)}T150000
DTEND;TZID=Asia/Taipei:{d(TODAY)}T160000
SUMMARY:看牙醫
LOCATION:信義區
UID:dentist@x
END:VEVENT
BEGIN:VEVENT
DTSTART;TZID=Asia/Taipei:{d(TODAY)}T170000
DTEND;TZID=Asia/Taipei:{d(TODAY)}T180000
SUMMARY:取消的會
STATUS:CANCELLED
UID:cancelled@x
END:VEVENT
END:VCALENDAR
""".replace("\n", "\r\n").encode()


class _Cal(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if "private-abc123" not in self.path:
            self.send_response(404); self.end_headers(); return
        HITS["ics"] += 1
        body = _ics()
        self.send_response(200)
        self.send_header("Content-Type", "text/calendar")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _expected_today() -> list:
    out = ["看牙醫"]
    if TODAY.weekday() == 0:
        out.insert(0, "週會")
    return out


async def test_calendar(base: str) -> None:
    config.google_ical_url = f"{base}/calendar/ical/me%40gmail.com/private-abc123/basic.ics"
    G._ics_cache = (0.0, "", None)
    evs = await G.events_for_days(TODAY, 1)
    assert [e["title"] for e in evs] == _expected_today(), evs
    dent = [e for e in evs if e["title"] == "看牙醫"][0]
    assert (dent["start"].hour, dent["end"].hour, dent["location"], dent["all_day"]) == (15, 16, "信義區", False)
    week = await G.events_for_days(TODAY, 7)
    titles = [e["title"] for e in week]
    assert titles.count("週會") == 1 and "媽媽生日" in titles and "取消的會" not in titles, titles
    mon = [e for e in week if e["title"] == "週會"][0]
    assert mon["start"].weekday() == 0 and mon["start"].strftime("%H:%M") == "09:00"
    assert [e["start"] if e["all_day"] else e["start"].date() for e in week] == sorted(
        e["start"] if e["all_day"] else e["start"].date() for e in week), "by day"
    assert HITS["ics"] == 1, "second read within 5 minutes comes from the cache"
    text = G.describe_events(evs, "今天")
    assert text.startswith("📅 今天的行程") and "• 15:00–16:00 看牙醫（信義區）" in text, text
    multi = G.describe_events(week, "這週", multi_day=True)
    assert "全天 媽媽生日" in multi and "（週一） 09:00–09:30 週會" in multi, multi
    assert G.describe_events([], "明天") == "📅 明天沒有行程。"
    print("  ✓ recurring (UTC), all-day, TZID and cancelled events; sorted; cached; described")

    config.google_ical_url = f"{base}/calendar/ical/me/private-zzz/basic.ics"
    try:
        await G.events_for_days(TODAY, 1); raise AssertionError("dead url")
    except G.PersonalError as exc:
        assert "網址失效" in str(exc)
    config.google_ical_url = ""
    try:
        await G.events_for_days(TODAY, 1); raise AssertionError("no url")
    except G.PersonalError as exc:
        assert "還沒連接 Google 行事曆" in str(exc)
    print("  ✓ a reset / missing iCal address explained")

    pc = G.parse_calendar_question
    assert pc("今天有什麼行程") == (TODAY, 1, "今天")
    assert pc("明天的會議")[1:] == (1, "明天") and pc("後天有約會嗎")[2] == "後天"
    assert pc("這週有哪些會議")[2] == "這週" and pc("下週行程")[1:] == (7, "下週")
    assert pc("what meetings do I have tomorrow")[2] == "明天"
    for t in ("幫我規劃東京五天行程", "明天要開會，幫我準備簡報", "提醒我明天的會議", "會議記錄整理一下",
              "行程", "幫我把會議改到下午", "每天早上給我今天的行程"):
        assert pc(t) is None, t
    print("  ✓ calendar questions vs. planning / editing / reminders")


HEADERS = {
    b"101": (f"From: {Header('台灣銀行', 'utf-8').encode()} <notice@bot.com.tw>\r\n"
             f"Subject: {Header('信用卡帳單通知', 'utf-8').encode()}\r\n"
             "Date: Fri, 30 Oct 2026 23:10:00 +0000\r\n\r\n").encode(),
    b"102": b"From: GitHub <noreply@github.com>\r\nSubject: [shop] CI failed on main\r\n"
            b"Date: Sat, 31 Oct 2026 01:05:00 +0000\r\n\r\n",
    b"103": b"From: amy@example.com\r\nSubject: \r\nDate: nonsense\r\n\r\n",
}
IMAP_CALLS: list = []


class FakeIMAP:
    def __init__(self, host, port, timeout=None):
        IMAP_CALLS.append(("connect", host, port))

    def login(self, user, pw):
        if pw != "abcdabcdabcdabcd":
            raise imaplib.IMAP4.error("[AUTHENTICATIONFAILED] Invalid credentials")
        IMAP_CALLS.append(("login", user))

    def select(self, box, readonly=False):
        IMAP_CALLS.append(("select", box, readonly))
        return "OK", [b"3"]

    def uid(self, cmd, *args):
        if cmd == "SEARCH":
            IMAP_CALLS.append(("search",) + args)
            return "OK", [b"101 102 103"]
        IMAP_CALLS.append(("fetch",) + args)
        return "OK", [(b"1 (UID " + args[0] + b" BODY[HEADER.FIELDS (FROM SUBJECT DATE)] {99}", HEADERS[args[0]]), b")"]

    def logout(self):
        IMAP_CALLS.append(("logout",))


async def test_gmail() -> None:
    G.imaplib.IMAP4_SSL = FakeIMAP
    config.gmail_address, config.gmail_app_password = "me@gmail.com", "abcd abcd abcd abcd"
    mails = await G.unread(limit=8)
    assert ("connect", "imap.gmail.com", 993) in IMAP_CALLS and ("login", "me@gmail.com") in IMAP_CALLS
    assert ("select", "INBOX", True) in IMAP_CALLS, "mailbox opened read-only"
    assert ("search", "X-GM-RAW", '"category:primary is:unread newer_than:1d"') in IMAP_CALLS
    fetches = [c for c in IMAP_CALLS if c[0] == "fetch"]
    assert all("BODY.PEEK" in c[2] for c in fetches), "PEEK: nothing marked read"
    assert [c[1] for c in fetches] == [b"103", b"102", b"101"], "newest first"
    assert IMAP_CALLS[-1] == ("logout",)
    assert mails[2]["from"] == "台灣銀行" and mails[2]["subject"] == "信用卡帳單通知"
    assert mails[1]["from"] == "GitHub" and mails[0]["subject"] == "（無主旨）" and mails[0]["date"] is None
    assert mails[2]["date"].hour == 7 and mails[2]["date"].day == 31, "converted to local time"
    text = G.describe_mail(mails)
    assert text.startswith("✉️ 未讀的重要信（3）") and "• 台灣銀行：信用卡帳單通知" in text, text
    assert G.describe_mail([]) == "✉️ 沒有未讀的重要信。"
    print("  ✓ read-only select, Gmail search, PEEK, encoded names decoded, newest first, local time")

    config.gmail_app_password = "wrong"
    try:
        await G.unread(); raise AssertionError("bad password")
    except G.PersonalError as exc:
        assert "應用程式密碼不對" in str(exc)
    config.gmail_app_password = "abcdabcdabcdabcd"
    pm = G.parse_mail_question
    for t in ("有什麼新信", "有重要的信嗎", "gmail 有沒有新信", "幫我看一下 email", "check my email"):
        assert pm(t), t
    for t in ("幫我寫一封 email 給老闆", "回信給 amy 說我會晚到", "整理今天的未讀信重點", "我的信箱滿了怎麼辦", "email 是什麼"):
        assert not pm(t), t
    print("  ✓ a wrong app password explained; mail questions vs. writing / summarising")


def test_parse_reminders() -> None:
    now = datetime(2026, 10, 31, 10, 15).astimezone()     # a Saturday
    P = lambda t: R.parse_reminder(t, now)
    show = lambda r: (r.fire_at.strftime("%m/%d %H:%M") if r.fire_at else None, r.text, r.via)
    cases = {
        "明天早上9點提醒我繳電話費": ("11/01 09:00", "繳電話費", "telegram"),
        "提醒我明天早上9點開會": ("11/01 09:00", "開會", "telegram"),
        "30分鐘後提醒我關火": ("10/31 10:45", "關火", "telegram"),
        "半小時後提醒我": ("10/31 10:45", "時間到了", "telegram"),
        "一個半小時後提醒我收衣服": ("10/31 11:45", "收衣服", "telegram"),
        "十分鐘後提醒我": ("10/31 10:25", "時間到了", "telegram"),
        "今晚8點半提醒我倒垃圾": ("10/31 20:30", "倒垃圾", "telegram"),
        "下午3點叫小愛提醒我出門": ("10/31 15:00", "出門", "xiaoai"),
        "7:40 叫小愛說該出門了": ("11/01 07:40", "該出門了", "xiaoai"),
        "下週一提醒我交報告": ("11/02 09:00", "交報告", "telegram"),
        "週五下午兩點提醒我看牙醫": ("11/06 14:00", "看牙醫", "telegram"),
        "11/5 提醒我繳信用卡": ("11/05 09:00", "繳信用卡", "telegram"),
        "明天7點叫我起床": ("11/01 07:00", "起床", "telegram"),
        "早上8點提醒我": ("11/01 08:00", "時間到了", "telegram"),
        "中午12點半提醒我吃飯": ("10/31 12:30", "吃飯", "telegram"),
        "明天晚上十一點提醒我睡覺": ("11/01 23:00", "睡覺", "telegram"),
        "凌晨一點提醒我": ("11/01 01:00", "時間到了", "telegram"),
        "remind me at 5pm to call mom": ("10/31 17:00", "call mom", "telegram"),
        "remind me in 20 minutes to check the oven": ("10/31 10:35", "check the oven", "telegram"),
        "提醒我買牛奶": (None, "買牛奶", "telegram"),
        "今天提醒我回信": (None, "回信", "telegram"),
    }
    for t, want in cases.items():
        r = P(t)
        assert r is not None and show(r) == want, (t, r and show(r), want)
    assert P("晚一點提醒我打電話").fire_at is None, "晚一點 is not 1 o'clock"
    for t in ("每天早上8點提醒我吃藥", "每週一提醒我倒垃圾", "如果明天下雨提醒我帶傘", "叫小愛說早安",
              "我有哪些提醒", "取消提醒 3f2a1b", "提醒功能怎麼用", "明天天氣如何"):
        assert P(t) is None, t
    assert R.parse_list("我有哪些提醒") and R.parse_list("提醒清單") and not R.parse_list("取消提醒")
    assert R.parse_cancel("取消提醒 3f2a1b") == "3f2a1b" and R.parse_cancel("取消剛剛的提醒") == ""
    assert R.parse_cancel("取消那個排程") is None and R.parse_cancel("明天提醒我") is None
    print(f"  ✓ {len(cases)} phrasings → time / text / 小愛; recurring, conditional, list, cancel left alone")


SENT: list = []
SPOKEN: list = []


async def test_reminder_store(tracker) -> None:
    import telegram_bridge, home_assistant as HA
    async def fake_send(chat, text): SENT.append((chat, text))
    async def fake_xiaoai(text, execute=False): SPOKEN.append(text); return {"done": True}
    telegram_bridge._send_rich, HA.xiaoai = fake_send, fake_xiaoai
    config.telegram_bot_token, config.telegram_default_chat_id = "123:abc", "999"

    await R.start(tracker)
    now = datetime.now().astimezone()
    a = await R.create(R.Reminder(now + timedelta(hours=2), "繳電話費"), session_id="s1", chat_id="555")
    again = await R.create(R.Reminder(now + timedelta(hours=2, seconds=30), "繳電話費"), session_id="s1")
    assert again.get("duplicate") and again["id"] == a["id"], "same request twice = one reminder"
    b = await R.create(R.Reminder(now + timedelta(hours=3), "出門", "xiaoai"), session_id="s1")
    rows = await R.pending("s1")
    assert [r["text"] for r in rows] == ["繳電話費", "出門"]
    txt = R.describe_created(a)
    assert "提醒你：繳電話費" in txt and f"取消提醒 {a['id'][:6]}" in txt
    assert "小愛會說出來" in R.describe_created(b)
    lst = R.describe_pending(rows)
    assert lst.startswith("⏰ 待提醒（2）") and "出門 🔊" in lst
    done, cand = await R.cancel(b["id"][:6], "s1")
    assert done["id"] == b["id"] and not cand and len(await R.pending()) == 1
    none, cand = await R.cancel("ffffff", "s1")
    assert none is None and cand == []
    print("  ✓ create, duplicate ignored, list, cancel by id; unknown id")

    # Delivery: Telegram to the chat it came from; 小愛 too; late ones say so.
    db = tracker.db
    late = (datetime.now().astimezone() - timedelta(minutes=40)).astimezone(R.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    await db.execute("INSERT INTO reminders(id, session_id, chat_id, fire_at, text, via, status, created_at) "
                     "VALUES ('late1', 's1', '', ?, '吃藥', 'xiaoai', 'pending', ?)", (late, late))
    await db.commit()
    SENT.clear(); SPOKEN.clear()
    assert await R.fire_due() == 1
    assert SENT[0][0] == "999" and SENT[0][1].startswith("⏰ 提醒：吃藥") and "晚了 40 分鐘" in SENT[0][1], SENT
    assert SPOKEN == [], "a reminder 40 minutes late isn't shouted into the room"
    async with db.execute("SELECT status FROM reminders WHERE id='late1'") as cur:
        assert (await cur.fetchone())[0] == "sent"

    SENT.clear()
    soon = await R.create(R.Reminder(datetime.now().astimezone() + timedelta(seconds=1), "關火", "xiaoai"),
                          session_id="s1", chat_id="555")
    for _ in range(40):
        if SENT:
            break
        await asyncio.sleep(0.1)
    assert SENT == [("555", "⏰ 提醒：關火")], SENT
    assert SPOKEN == ["提醒你，關火"], SPOKEN
    print("  ✓ the loop fires on time to the asking chat and 小愛; a late one is marked late")

    config.telegram_bot_token = ""
    await db.execute("UPDATE reminders SET fire_at=? WHERE id=?", (late, a["id"]))
    await db.commit()
    SENT.clear()
    async def broken(*a, **k): raise HA.HAError("speaker offline")
    HA.xiaoai = broken
    await R.fire_due()
    async with db.execute("SELECT status, error FROM reminders WHERE id=?", (a["id"],)) as cur:
        st, err = await cur.fetchone()
    assert st == "failed" and "Telegram 沒設定" in err, (st, err)
    config.telegram_bot_token = "123:abc"
    HA.xiaoai = fake_xiaoai
    print("  ✓ nowhere to send → recorded as failed, with why")


async def test_briefing() -> None:
    import briefing, weather, typhoon, home_assistant as HA
    async def brief(place=""): return "家裡：🌧 小雨，22–27°C，降雨機率 70%，☂️ 記得帶傘"
    async def storms():
        return [{"name": "KONG-REY", "distance": 650, "forecast": [(24, (22, 122), 280)]},
                {"name": "FAR", "distance": 2500, "forecast": [(24, (30, 150), 2900)]}]
    weather.brief, typhoon.active_storms = brief, storms
    async def devs(): return [{"name": "客廳燈", "online": False}, {"name": "冷氣", "online": True}]
    HA.devices = devs
    real_configured = HA.configured
    HA.configured = lambda: True
    await R.create(R.Reminder(datetime.now().astimezone() + timedelta(minutes=1), "打給媽媽"), session_id="s1")
    text = await briefing.morning_briefing()
    parts = text.split("\n\n")
    assert parts[0].startswith(f"☀️ 早安！今天是 {TODAY.month}/{TODAY.day}"), parts[0]
    assert "☂️ 記得帶傘" in parts[1]
    assert "颱風 KONG-REY" in text and "FAR" not in text, "only a storm heading for Taiwan"
    assert "📅 今天的行程" in text and "看牙醫" in text
    assert "⏰ 今天的提醒" in text and "打給媽媽" in text
    assert "✉️ 未讀的主要信件" in text and "信用卡帳單通知" in text
    assert "📴 家裡有 1 台設備離線：客廳燈" in text
    print("  ✓ weather, near typhoon only, calendar, reminders, mail, offline devices")

    async def boom(place=""): raise RuntimeError("Open-Meteo down")
    weather.brief = boom
    async def calm(): return [{"name": "FAR", "distance": 2500, "forecast": [(24, (30, 150), 2900)]}]
    typhoon.active_storms = calm
    HA.configured = real_configured
    config.ha_token = ""
    text = await briefing.morning_briefing()
    assert "⚠️ 天氣：Open-Meteo down" in text and "看牙醫" in text and "🌀" not in text and "📴" not in text
    assert briefing.parse_briefing_request("早安簡報") and briefing.parse_briefing_request("今日簡報")
    assert briefing.parse_briefing_request("morning briefing") and briefing.parse_briefing_request("產生早安簡報")
    for t in ("幫我做一份簡報", "每天早上 7:30 給我早安簡報", "簡報怎麼做比較好", "今天會議的摘要", "早安簡報怎麼設定"):
        assert not briefing.parse_briefing_request(t), t
    print("  ✓ a failing part says so in one line and the rest still arrives; requests parsed")


async def test_orchestrator() -> None:
    from orchestrator import Orchestrator
    from models import TaskRequest, TaskStatus
    rows = []

    class _Tracker:
        async def get_session(self, sid): return {"id": sid, "mode": "chat", "workspace_dir": None}
        async def create_task(self, req): rows.append(("create", req.goal))
        async def update_task_status(self, tid, status, summary=None): rows.append(("status", status, summary))
        async def create_subtask(self, st): pass
        async def update_subtask(self, st): pass
        async def list_tasks(self, **k): return []

    async def no_event(*a, **k): pass
    async def planner(*a, **k): raise RuntimeError("planner reached")
    o = types.SimpleNamespace(tracker=_Tracker(), _emit_task_event=no_event, _plan=planner,
                              _classify_intent=planner, _fast_chat_response=planner)
    for n in ("_finish_direct", "_claude_code", "_asked_again", "_reminder", "_personal"):
        setattr(o, n, (lambda name: lambda *a, **k: getattr(Orchestrator, name)(o, *a, **k))(n))
    run = lambda goal, **ctx: Orchestrator.run(o, TaskRequest(goal=goal, session_id="s9", context=ctx))

    res = await run("明天早上9點提醒我繳電話費", telegram_chat_id="777")
    assert res.status == TaskStatus.COMPLETED and "09:00提醒你：繳電話費" in res.summary, res.summary
    rid = res.summary.split("取消提醒 ")[1][:6]
    mine = [r for r in await R.pending("s9")]
    assert len(mine) == 1 and mine[0]["chat_id"] == "777"
    res = await run("提醒我買牛奶")
    assert res.status == TaskStatus.FAILED and "要什麼時候提醒你「買牛奶」" in res.summary
    res = await run("我有哪些提醒")
    assert "繳電話費" in res.summary and rid in res.summary
    res = await run(f"取消提醒 {rid}")
    assert res.summary.startswith("🗑 已取消") and not await R.pending("s9")
    try:
        await run("提醒我出門", triggered_by="schedule")
        raise AssertionError("a scheduled goal isn't a new reminder")
    except RuntimeError as exc:
        assert "planner reached" in str(exc)
    print("  ✓ reminder: create (with the chat), ask for a time, list, cancel; scheduled goals skip it")

    res = await run("今天有什麼行程")
    assert res.summary.startswith("📅 今天的行程") and "看牙醫" in res.summary, res.summary
    res = await run("有什麼新信")
    assert res.summary.startswith("✉️ 未讀的重要信") and "台灣銀行" in res.summary
    res = await run("早安簡報")
    assert res.summary.startswith("☀️ 早安！")
    config.google_ical_url, saved = "", config.google_ical_url
    res = await run("明天有什麼行程")
    assert res.status == TaskStatus.FAILED and "整合日曆" in res.summary
    config.google_ical_url = saved
    print("  ✓ calendar, mail and briefing answered without the model; not connected → how to connect")

    import scheduling_intent
    async def no_llm(text): raise AssertionError("LLM called for a reminder")
    scheduling_intent.llm_extract = no_llm
    assert await scheduling_intent.detect_scheduling_intent("明天早上 9:00 提醒我開會") is None
    print("  ✓ a one-shot reminder never becomes a cron schedule (and costs no LLM call)")


async def test_tools_and_logs() -> None:
    from tools.personal_tool import calendar_events, gmail_unread, reminder_create
    r = await calendar_events(TODAY.isoformat(), 7)
    assert r["success"] and "週會" in r["result"], r
    r = await gmail_unread("is:unread from:bank", 5)
    assert r["success"] and ("search", "X-GM-RAW", '"is:unread from:bank"') in IMAP_CALLS
    r = await reminder_create("開會前準備", minutes_from_now=15)
    assert r["success"] and "開會前準備" in r["result"]
    r = await reminder_create("x", when="2020-01-01 09:00")
    assert not r["success"] and "past" in r["error"]
    from tool_registry import tool_registry
    names = {t["function"]["name"] for t in tool_registry.get_schemas()} if hasattr(tool_registry, "get_schemas") else set()
    if names:
        assert {"calendar_events", "gmail_unread", "reminder_create"} <= names
    import log_redact
    line = "HTTP Request: GET https://calendar.google.com/calendar/ical/me%40gmail.com/private-0123456789abcdef/basic.ics"
    assert "0123456789abcdef" not in log_redact.redact(line) and "private-<redacted>" in log_redact.redact(line)
    print("  ✓ agent tools (calendar, Gmail search, reminder); the iCal secret is masked in logs")


async def main() -> None:
    from tracker import Tracker
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Cal)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    tmp = tempfile.mkdtemp(prefix="ot-personal-")
    tracker = Tracker(os.path.join(tmp, "t.db"))
    await tracker.open()
    try:
        print("A. calendar"); await test_calendar(base)
        config.google_ical_url = f"{base}/calendar/ical/me%40gmail.com/private-abc123/basic.ics"
        print("B. gmail"); await test_gmail()
        print("C. reminders"); test_parse_reminders(); await test_reminder_store(tracker)
        print("D. briefing"); await test_briefing()
        print("E. orchestrator"); await test_orchestrator(); await test_tools_and_logs()
    finally:
        await R.stop()
        await tracker.close()
        srv.shutdown()
    print("\nALL PERSONAL TESTS PASS")


asyncio.run(main())
