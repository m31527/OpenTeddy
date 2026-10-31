"""
Personal assistant: each person's own Google calendar + Gmail (OAuth,
read-only), one-shot reminders, the morning briefing — and privacy.

  A. linking: commands, the sign-in link (PKCE, offline, consent), the
     pasted-back address bound to the chat that asked, one use, refresh
     token encrypted at rest, missing scopes reported, refresh, expiry,
     unlink revokes at Google
  B. data: calendars the person shows (hidden ones not read), recurring
     instances from Google, declined / cancelled / duplicate left out;
     Gmail search + headers; no grant → explained
  C. who: a Telegram private chat is a person; web, CLI, MCP, groups and a
     spoofed request context are nobody; a schedule in a person's
     Telegram session is that person
  D. privacy: a task that read calendar / mail is stored and broadcast as
     a note, never in full; memory doesn't keep it
  E. reminders: parsing, store, delivery, other people's stay private
  F. briefing, orchestrator paths, tools, log redaction

    .venv/bin/python tests/test_personal.py
"""
from __future__ import annotations

import asyncio, base64, hashlib, json, logging, os, stat, sys, tempfile, threading, time, types
import urllib.parse
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ["TZ"] = "Asia/Taipei"
time.tzset()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from config import config
import google_auth as GA
import google_personal as G
import privacy
import reminders as R
from tools._context import set_triggered_by, reset_triggered_by

TODAY = datetime.now().date()
TMP = tempfile.mkdtemp(prefix="ot-personal-")
config.db_path = os.path.join(TMP, "t.db")
config.google_client_id, config.google_client_secret = "123-abc.apps.googleusercontent.com", "GOCSPX-test"
GOOG = {"challenge": "", "revoked": [], "q": [], "fetched": [], "auth": []}
ALICE, BOB = "tg:777", "tg:888"


def _id_token(email: str) -> str:
    b = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
    return f"{b({'alg': 'RS256'})}.{b({'email': email})}.sig"


def _iso(d: date, h: int, m: int = 0) -> str:
    return datetime.combine(d, datetime.min.time()).replace(hour=h, minute=m).astimezone().isoformat()


def _events(cal: str) -> list:
    tomorrow = TODAY + timedelta(days=1)
    if cal == "primary":
        return [
            {"id": "e1", "iCalUID": "dentist@x", "summary": "看牙醫", "location": "信義區",
             "start": {"dateTime": _iso(TODAY, 15)}, "end": {"dateTime": _iso(TODAY, 16)}},
            {"id": "e2", "iCalUID": "bday@x", "summary": "媽媽生日",
             "start": {"date": tomorrow.isoformat()}, "end": {"date": (tomorrow + timedelta(days=1)).isoformat()}},
            {"id": "e3", "status": "cancelled", "summary": "取消的會",
             "start": {"dateTime": _iso(TODAY, 17)}, "end": {"dateTime": _iso(TODAY, 18)}},
            {"id": "e4", "summary": "我拒絕的會", "attendees": [{"self": True, "responseStatus": "declined"}],
             "start": {"dateTime": _iso(TODAY, 11)}, "end": {"dateTime": _iso(TODAY, 12)}},
        ]
    if cal == "family@group.calendar.google.com":
        return [
            {"id": "f1", "iCalUID": "dentist@x", "summary": "看牙醫", "location": "信義區",
             "start": {"dateTime": _iso(TODAY, 15)}, "end": {"dateTime": _iso(TODAY, 16)}},
            {"id": "f2", "iCalUID": "dinner@x", "summary": "家庭聚餐",
             "start": {"dateTime": _iso(TODAY, 19)}, "end": {"dateTime": _iso(TODAY, 21)}},
        ]
    return [{"id": "h1", "summary": "秘密", "start": {"dateTime": _iso(TODAY, 9)}, "end": {"dateTime": _iso(TODAY, 10)}}]


MAILS = {
    "m3": {"From": "台灣銀行 <notice@bot.com.tw>", "Subject": "信用卡帳單通知", "t": _iso(TODAY, 7)},
    "m2": {"From": "GitHub <noreply@github.com>", "Subject": "[shop] CI failed on main", "t": _iso(TODAY, 6)},
    "m1": {"From": "amy@example.com", "Subject": "", "t": _iso(TODAY - timedelta(days=1), 22)},
}


class _Google(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        form = dict(urllib.parse.parse_qsl(self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()))
        if self.path == "/revoke":
            GOOG["revoked"].append(form.get("token"))
            return self._send({})
        assert form["client_id"] == config.google_client_id and form["client_secret"] == config.google_client_secret
        if form["grant_type"] == "authorization_code":
            ch = base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"].encode()).digest()).rstrip(b"=").decode()
            if form["code"] not in ("good-code", "partial-code") or ch != GOOG["challenge"]:
                return self._send({"error": "invalid_grant"}, 400)
            assert form["redirect_uri"] == GA.REDIRECT_URI
            scope = f"openid email {GA.CAL_SCOPE}" + ("" if form["code"] == "partial-code" else f" {GA.MAIL_SCOPE}")
            return self._send({"access_token": "at-1", "expires_in": 3599, "refresh_token": "rt-secret-123",
                               "scope": scope, "id_token": _id_token("alice@gmail.com")})
        if form["refresh_token"] == "rt-secret-123" and "rt-secret-123" not in GOOG["revoked"]:
            return self._send({"access_token": "at-2", "expires_in": 3599})
        return self._send({"error": "invalid_grant"}, 400)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        auth = self.headers.get("Authorization", "")
        GOOG["auth"].append(auth)
        if auth not in ("Bearer at-1", "Bearer at-2"):
            return self._send({"error": {"code": 401}}, 401)
        if GOOG.get("deny"):
            return self._send({"error": {"code": 403, "message": "insufficient scope"}}, 403)
        if u.path == "/cal/users/me/calendarList":
            return self._send({"items": [{"id": "primary", "primary": True, "selected": True},
                                         {"id": "family@group.calendar.google.com", "selected": True},
                                         {"id": "hidden@group.calendar.google.com", "selected": True, "hidden": True},
                                         {"id": "unselected@x", "selected": False}]})
        if u.path.startswith("/cal/calendars/"):
            cal = urllib.parse.unquote(u.path.split("/")[3])
            GOOG["fetched"].append(cal)
            assert q["singleEvents"] == ["true"] and q["timeMin"] and q["timeMax"]
            lo, hi = (datetime.fromisoformat(q[k][0]) for k in ("timeMin", "timeMax"))
            at = lambda t: (datetime.fromisoformat(t["dateTime"]) if "dateTime" in t else
                            datetime.combine(date.fromisoformat(t["date"]), datetime.min.time()).astimezone())
            return self._send({"items": [e for e in _events(cal) if at(e["start"]) < hi and at(e["end"]) > lo]})
        if u.path == "/gmail/users/me/messages":
            GOOG["q"].append(q["q"][0])
            return self._send({"messages": [{"id": "m3"}, {"id": "m2"}, {"id": "m1"}]})
        if u.path.startswith("/gmail/users/me/messages/"):
            m = MAILS[u.path.rsplit("/", 1)[1]]
            assert q["format"] == ["metadata"]
            ts = int(datetime.fromisoformat(m["t"]).timestamp() * 1000)
            return self._send({"internalDate": str(ts), "payload": {"headers": [
                {"name": "From", "value": m["From"]}, {"name": "Subject", "value": m["Subject"]}]}})
        self._send({}, 404)


async def test_linking(tracker) -> None:
    pc = GA.parse_command
    assert pc("連結我的 Google") == "link" and pc("/google") == "link" and pc("綁定 gmail") == "link"
    assert pc("解除 Google") == "unlink" and pc("取消連結我的 google") == "unlink"
    assert pc("Google 連結了嗎") == "status" and pc("今天天氣") is None and pc("google 是什麼公司") is None
    url = "http://127.0.0.1:8749/?state=abc%2Bx&code=4/0AbC-dE&scope=email%20openid"
    assert GA.parse_callback(url) == ("4/0AbC-dE", "abc+x") and GA.parse_callback("code=1 但沒有 state") is None
    print("  ✓ link / unlink / status commands; the pasted-back address parsed")

    await GA.start(tracker)
    link = GA.begin(ALICE)
    q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(link).query))
    assert q["redirect_uri"] == "http://127.0.0.1:8749/" and q["code_challenge_method"] == "S256"
    assert q["access_type"] == "offline" and q["prompt"] == "consent"
    assert set(q["scope"].split()) == {"openid", "email", GA.CAL_SCOPE, GA.MAIL_SCOPE}
    GOOG["challenge"] = q["code_challenge"]
    try:
        await GA.finish(BOB, "good-code", q["state"]); raise AssertionError("other chat")
    except GA.GoogleAuthError as exc:
        assert "不是給這個聊天的" in str(exc)
    res = await GA.finish(ALICE, "good-code", q["state"])
    assert res == {"email": "alice@gmail.com", "missing": []}, res
    try:
        await GA.finish(ALICE, "good-code", q["state"]); raise AssertionError("reused")
    except GA.GoogleAuthError as exc:
        assert "過期或用過了" in str(exc)
    async with tracker.db.execute("SELECT * FROM google_accounts") as cur:
        rows = [dict(r) for r in await cur.fetchall()]
    assert len(rows) == 1 and rows[0]["email"] == "alice@gmail.com"
    assert "rt-secret-123" not in json.dumps(rows), "refresh token encrypted at rest"
    key = os.path.join(TMP, "google_tokens.key")
    assert stat.S_IMODE(os.stat(key).st_mode) == 0o600
    acct = await GA.account(ALICE)
    assert acct["status"] == "ok" and "refresh_token" not in acct
    print("  ✓ PKCE link bound to the asking chat, one use; token encrypted (key file 600)")

    link = GA.begin(BOB)
    GOOG["challenge"] = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(link).query))["code_challenge"]
    res = await GA.finish(BOB, "partial-code", dict(urllib.parse.parse_qsl(urllib.parse.urlparse(link).query))["state"])
    assert res["missing"] == [GA.MAIL_SCOPE], "unticked Gmail is reported"
    assert await GA.count() == 2
    assert await GA.access_token(ALICE) == "at-1"
    assert await GA.access_token(ALICE, force=True) == "at-2"
    assert await GA.unlink(BOB) == "alice@gmail.com" and "rt-secret-123" in GOOG["revoked"]
    assert await GA.account(BOB) is None and await GA.unlink(BOB) is None
    GOOG["revoked"].clear()
    print("  ✓ missing scopes reported; access token cached and refreshed; unlink revokes and forgets")


async def test_data() -> None:
    GA._access[ALICE] = ("stale-token", time.time() + 3600)        # Google says 401 → refresh once
    privacy.begin("task-cal", ALICE)
    evs = await G.events_for_days(TODAY, 2)
    titles = [e["title"] for e in evs]
    assert titles == ["看牙醫", "家庭聚餐", "媽媽生日"], titles
    assert "hidden@group.calendar.google.com" not in GOOG["fetched"] and "unselected@x" not in GOOG["fetched"]
    dent = evs[0]
    assert (dent["start"].hour, dent["end"].hour, dent["location"], dent["all_day"]) == (15, 16, "信義區", False)
    assert evs[2]["all_day"] and evs[2]["start"] == TODAY + timedelta(days=1)
    assert privacy.is_private("task-cal"), "a task that read the calendar is private"
    assert "Bearer stale-token" in GOOG["auth"] and GOOG["auth"][-1] == "Bearer at-2"
    text = G.describe_events(evs, "這兩天", multi_day=True)
    assert "15:00–16:00 看牙醫（信義區）" in text and "全天 媽媽生日" in text
    print("  ✓ calendars the person shows; duplicates, declined, cancelled and hidden left out; 401 → refresh")

    privacy.begin("task-mail", ALICE)
    mails = await G.unread()
    assert GOOG["q"][-1] == "category:primary is:unread newer_than:1d"
    assert [m["from"] for m in mails] == ["台灣銀行", "GitHub", "amy@example.com"]
    assert mails[0]["subject"] == "信用卡帳單通知" and mails[2]["subject"] == "（無主旨）"
    assert mails[0]["date"].hour == 7 and privacy.is_private("task-mail")
    assert G.describe_mail(mails).startswith("✉️ 未讀的重要信（3）")
    GOOG["deny"] = True
    try:
        await G.unread(); raise AssertionError("403")
    except G.PersonalError as exc:
        assert "沒有授權讀取你的Gmail" in str(exc)
    GOOG["deny"] = False
    privacy.begin("task-web", None)
    try:
        await G.unread(); raise AssertionError("no person")
    except G.PersonalError as exc:
        assert "只在本人的 Telegram 私訊" in str(exc)
    try:
        await G.events_for_days(TODAY, 1, BOB); raise AssertionError("not linked")
    except G.PersonalError as exc:
        assert "還沒連結 Google" in str(exc)
    pm = G.parse_mail_question
    for t in ("有什麼新信", "有重要的信嗎", "gmail 有沒有新信", "check my email"):
        assert pm(t), t
    for t in ("幫我寫一封 email 給老闆", "整理今天的未讀信重點", "email 是什麼"):
        assert not pm(t), t
    pcq = G.parse_calendar_question
    assert pcq("今天有什麼行程") == (TODAY, 1, "今天") and pcq("下週行程")[1:] == (7, "下週")
    for t in ("幫我規劃東京五天行程", "明天要開會，幫我準備簡報", "提醒我明天的會議"):
        assert pcq(t) is None, t
    print("  ✓ Gmail search + headers; no grant / nobody / not linked each explained; questions parsed")


async def test_who(tracker) -> None:
    assert privacy.person_for_chat("777") == ALICE and privacy.person_for_chat(777) == ALICE
    assert privacy.person_for_chat("-1001234") is None and privacy.person_for_chat("") is None
    from models import TaskRequest
    class _T:
        async def get_session(self, sid): return {"telegram_chat_id": {"s-tg": "777", "s-grp": "-55"}.get(sid, "")}
    req = lambda **c: TaskRequest(goal="x", session_id=c.pop("sid", "s-web"), context=c)
    assert await privacy.resolve(req(telegram_chat_id="777"), _T()) is None, "web: spoofed context is nobody"
    tok = set_triggered_by("telegram")
    try:
        assert await privacy.resolve(req(telegram_chat_id="777"), _T()) == ALICE
        assert await privacy.resolve(req(telegram_chat_id="-55"), _T()) is None, "a group is nobody"
    finally:
        reset_triggered_by(tok)
    tok = set_triggered_by("schedule")
    try:
        assert await privacy.resolve(req(sid="s-tg"), _T()) == ALICE
        assert await privacy.resolve(req(sid="s-web"), _T()) is None
    finally:
        reset_triggered_by(tok)
    tok = set_triggered_by("api_trusted")
    try:
        assert await privacy.resolve(req(sid="s-tg", telegram_chat_id="777"), _T()) is None
    finally:
        reset_triggered_by(tok)
    print("  ✓ a Telegram private chat is a person; web / API / group / spoofed context are nobody; schedules")


async def test_storage(tracker) -> None:
    from models import TaskRequest, TaskStatus, SubTask, AgentRole
    for tid in ("p1", "n1"):
        await tracker.create_task(TaskRequest(id=tid, goal="今天有什麼行程", session_id="s"))
    privacy.mark("p1")
    await tracker.update_task_status("p1", TaskStatus.COMPLETED, "📅 今天：15:00 看牙醫")
    await tracker.update_task_status("n1", TaskStatus.COMPLETED, "☀️ 晴")
    st = SubTask(parent_task_id="p1", description="x", agent=AgentRole.EXECUTOR, order=0,
                 status=TaskStatus.COMPLETED, result="看牙醫 15:00")
    await tracker.create_subtask(st)
    await tracker.update_subtask(st)
    assert (await tracker.get_task("p1"))["summary"] == privacy.NOTE
    assert (await tracker.get_task("n1"))["summary"] == "☀️ 晴"
    assert (await tracker.get_subtasks("p1"))[0].result == privacy.NOTE
    ev = privacy.scrub_event({"type": "tool_result", "task_id": "p1", "tool": "gmail_unread",
                              "result": "台灣銀行：帳單", "data": {"x": 1}, "_ts": 1.0})
    assert ev == {"type": "tool_result", "task_id": "p1", "tool": "gmail_unread", "_ts": 1.0,
                  "result": "🔒", "private": True}, ev
    assert privacy.scrub_event({"type": "chat.stream.delta", "task_id": "p1", "text": "看牙"}) is None
    assert privacy.scrub_event({"type": "tool_result", "task_id": "n1", "result": "ok"})["result"] == "ok"
    from memory import MemoryManager
    stored = []
    m = MemoryManager.__new__(MemoryManager)
    type(m).is_available = property(lambda self: (stored.append(1), False)[1])
    await m.summarize_and_store("p1", "goal", "看牙醫")
    assert not stored, "a private task never reaches memory"
    print("  ✓ private task: stored and broadcast as a note, live text dropped, memory skipped")

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
    other = await R.create(R.Reminder(now + timedelta(hours=4), "看診"), session_id="s2")
    assert [r["text"] for r in await R.pending("s1")] == ["繳電話費"], "s2's reminder isn't s1's business"
    assert await R.cancel(other["id"][:6], "s1") == (None, []), "s1 can't cancel s2's reminder"
    assert [r["text"] for r in await R.pending("s2")] == ["看診"]
    print("  ✓ create, duplicate ignored, list, cancel by id; unknown id; other people's stay private")

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
    privacy.begin("brief-1", ALICE)
    text = await briefing.morning_briefing("s1", ALICE)
    assert "看診" not in text, "another conversation's reminder never shows up in this briefing"
    parts = text.split("\n\n")
    assert parts[0].startswith(f"☀️ 早安！今天是 {TODAY.month}/{TODAY.day}"), parts[0]
    assert "☂️ 記得帶傘" in parts[1]
    assert "颱風 KONG-REY" in text and "FAR" not in text, "only a storm heading for Taiwan"
    assert "📅 今天的行程" in text and "看牙醫" in text and "媽媽生日" not in text
    assert "⏰ 今天的提醒" in text and "打給媽媽" in text
    assert "✉️ 未讀的主要信件" in text and "信用卡帳單通知" in text
    assert "📴 家裡有 1 台設備離線：客廳燈" in text and privacy.is_private("brief-1")
    print("  ✓ the person's calendar and mail, their reminders, weather, near typhoon only, offline devices")

    text = await briefing.morning_briefing("s-web", None)
    assert "看牙醫" not in text and "信用卡" not in text and "連結我的 Google" not in text, "web: nobody's data"
    text = await briefing.morning_briefing("s2", BOB)
    assert "看牙醫" not in text and "說「連結我的 Google」" in text, "not linked → how to add it"
    async def boom(place=""): raise RuntimeError("Open-Meteo down")
    weather.brief = boom
    async def calm(): return [{"name": "FAR", "distance": 2500, "forecast": [(24, (30, 150), 2900)]}]
    typhoon.active_storms = calm
    HA.configured = real_configured
    config.ha_token = ""
    text = await briefing.morning_briefing("s1", ALICE)
    assert "⚠️ 天氣：Open-Meteo down" in text and "看牙醫" in text and "🌀" not in text and "📴" not in text
    assert briefing.parse_briefing_request("早安簡報") and briefing.parse_briefing_request("產生早安簡報")
    for t in ("幫我做一份簡報", "每天早上 7:30 給我早安簡報", "早安簡報怎麼設定"):
        assert not briefing.parse_briefing_request(t), t
    print("  ✓ web: nobody's data; not linked: a hint; a failing part is one line; requests parsed")


async def test_orchestrator() -> None:
    from orchestrator import Orchestrator
    from models import TaskRequest, TaskStatus
    rows = []

    class _Tracker:
        async def get_session(self, sid): return {"id": sid, "mode": "chat", "workspace_dir": None,
                                                  "telegram_chat_id": "777" if sid == "s9" else ""}
        async def create_task(self, req): rows.append(("create", req.goal))
        async def update_task_status(self, tid, status, summary=None): rows.append(("status", status, summary))
        async def create_subtask(self, st): pass
        async def update_subtask(self, st): pass
        async def list_tasks(self, **k): return []

    async def no_event(*a, **k): pass
    async def planner(*a, **k): raise RuntimeError("planner reached")
    o = types.SimpleNamespace(tracker=_Tracker(), _emit_task_event=no_event, _plan=planner,
                              _classify_intent=planner, _fast_chat_response=planner)
    for n in ("_finish_direct", "_claude_code", "_asked_again", "_reminder", "_personal", "_google_link"):
        setattr(o, n, (lambda name: lambda *a, **k: getattr(Orchestrator, name)(o, *a, **k))(n))

    async def run(goal, origin="telegram", chat="777", **ctx):
        tok = set_triggered_by(origin) if origin else None
        try:
            c = dict(ctx, **({"telegram_chat_id": chat} if chat else {}))
            return await Orchestrator.run(o, TaskRequest(goal=goal, session_id="s9", context=c))
        finally:
            if tok:
                reset_triggered_by(tok)

    res = await run("明天早上9點提醒我繳電話費")
    assert res.status == TaskStatus.COMPLETED and "09:00提醒你：繳電話費" in res.summary, res.summary
    rid = res.summary.split("取消提醒 ")[1][:6]
    assert [r["chat_id"] for r in await R.pending("s9")] == ["777"]
    res = await run("提醒我買牛奶")
    assert res.status == TaskStatus.FAILED and "要什麼時候提醒你「買牛奶」" in res.summary
    res = await run("我有哪些提醒")
    assert "繳電話費" in res.summary and rid in res.summary and "看診" not in res.summary
    res = await run(f"取消提醒 {rid}")
    assert res.summary.startswith("🗑 已取消") and not await R.pending("s9")
    try:
        await run("提醒我出門", origin="schedule", chat="", triggered_by="schedule")
        raise AssertionError("a scheduled goal isn't a new reminder")
    except RuntimeError as exc:
        assert "planner reached" in str(exc)
    print("  ✓ reminder: create (with the chat), ask for a time, list, cancel; scheduled goals skip it")

    res = await run("今天有什麼行程", origin="", chat="777")
    assert res.status == TaskStatus.FAILED and "只在本人的 Telegram 私訊" in res.summary, "web: refused"
    res = await run("連結我的 Google", origin="telegram", chat="-55")
    assert "只在本人的 Telegram 私訊" in res.summary, "group: refused"
    res = await run("連結我的 Google")
    assert "https://accounts.google.com/" in res.summary and "貼回這裡" in res.summary
    link = res.summary.split("登入：")[1].split()[0]
    lq = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(link).query))
    GOOG["challenge"] = lq["code_challenge"]
    res = await run(f"http://127.0.0.1:8749/?state={urllib.parse.quote(lq['state'])}&code=good-code&scope=email")
    assert res.summary.startswith("✅ 已連結 Google：alice@gmail.com"), res.summary
    res = await run("今天有什麼行程")
    assert res.summary.startswith("📅 今天的行程") and "看牙醫" in res.summary and "家庭聚餐" in res.summary
    res = await run("有什麼新信")
    assert res.summary.startswith("✉️ 未讀的重要信") and "台灣銀行" in res.summary
    res = await run("早安簡報")
    assert res.summary.startswith("☀️ 早安！") and "看牙醫" in res.summary
    res = await run("早安簡報", origin="schedule", chat="", triggered_by="schedule")
    assert "看牙醫" in res.summary, "a schedule in the person's Telegram session is that person"
    res = await run("Google 連結了嗎")
    assert "alice@gmail.com（正常）" in res.summary
    res = await run("解除 Google")
    assert res.summary.startswith("🗑 已解除") and "rt-secret-123" in GOOG["revoked"]
    res = await run("今天有什麼行程")
    assert res.status == TaskStatus.FAILED and "還沒連結 Google" in res.summary
    saved = config.google_client_secret
    config.google_client_secret = ""
    res = await run("今天有什麼行程")
    assert "還沒設定 Google 登入" in res.summary
    config.google_client_secret = saved
    print("  ✓ link from Telegram, paste back, calendar / mail / briefing; web and groups refused; unlink")

    import scheduling_intent
    async def no_llm(text): raise AssertionError("LLM called")
    scheduling_intent.llm_extract = no_llm
    assert await scheduling_intent.detect_scheduling_intent("明天早上 9:00 提醒我開會") is None
    assert await scheduling_intent.detect_scheduling_intent(
        "http://127.0.0.1:8749/?state=a&code=4/0Ab&scope=x") is None
    print("  ✓ reminders and pasted sign-in addresses never become cron schedules")


async def test_tools_and_logs() -> None:
    from tools.personal_tool import calendar_events, gmail_unread, reminder_create
    GOOG["revoked"].clear()
    link = GA.begin(ALICE)
    lq = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(link).query))
    GOOG["challenge"] = lq["code_challenge"]
    await GA.finish(ALICE, "good-code", lq["state"])
    privacy.begin("tool-web", None)
    r = await calendar_events(TODAY.isoformat(), 2)
    assert not r["success"] and "Telegram" in r["error"], "the agent can't read anyone's calendar from the web"
    privacy.begin("tool-tg", ALICE)
    r = await calendar_events(TODAY.isoformat(), 2)
    assert r["success"] and "看牙醫" in r["result"] and privacy.is_private("tool-tg")
    r = await gmail_unread("is:unread from:bank", 5)
    assert r["success"] and GOOG["q"][-1] == "is:unread from:bank"
    r = await reminder_create("開會前準備", minutes_from_now=15)
    assert r["success"] and "開會前準備" in r["result"]
    r = await reminder_create("x", when="2020-01-01 09:00")
    assert not r["success"] and "past" in r["error"]
    import log_redact
    line = "Telegram message: http://127.0.0.1:8749/?state=abc&code=4/0AbCdEf123&scope=email"
    assert "4/0AbCdEf123" not in log_redact.redact(line) and "code=<redacted>" in log_redact.redact(line)
    print("  ✓ agent tools follow the person (refused from the web); sign-in codes masked in logs")


async def main() -> None:
    from tracker import Tracker
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Google)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    GA.TOKEN_URL, GA.REVOKE_URL = f"{base}/token", f"{base}/revoke"
    G.CAL_BASE, G.GMAIL_BASE = f"{base}/cal", f"{base}/gmail"
    tracker = Tracker(config.db_path)
    await tracker.open()
    try:
        print("A. linking"); await test_linking(tracker)
        print("B. data"); await test_data()
        print("C. who"); await test_who(tracker)
        print("D. privacy"); await test_storage(tracker)
        print("E. reminders"); test_parse_reminders(); await test_reminder_store(tracker)
        print("F. briefing / orchestrator / tools"); await test_briefing(); await test_orchestrator()
        await test_tools_and_logs()
    finally:
        await R.stop()
        await tracker.close()
        srv.shutdown()
    print("\nALL PERSONAL TESTS PASS")


asyncio.run(main())
