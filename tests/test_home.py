"""
Home Assistant bridge, against a fake Home Assistant.

  A. home_intent: short device commands, sensitive words, "確認 1234"
  B. home_assistant: not configured, list/filter, control, the
     confirmation hold (locks, garage covers), single-use + expiry,
     auth / network errors, Assist parsing, 小愛 say / command + fallback
  C. home tools: results and errors
  D. orchestrator: "關客廳燈" answered by Assist without planning; Assist
     not understanding → normal pipeline; "確認 code" runs the held action;
     scheduled runs can't confirm; unconfigured → untouched

    .venv/bin/python tests/test_home.py
"""
from __future__ import annotations

import asyncio, logging, os, sys, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

import httpx
from config import config
import home_assistant as HA
import home_intent as HI

STATES = [
    {"entity_id": "light.living_room", "state": "on", "attributes": {"friendly_name": "客廳燈", "brightness": 200}},
    {"entity_id": "climate.bedroom_ac", "state": "cool", "attributes": {"friendly_name": "臥室冷氣", "temperature": 26, "current_temperature": 28.5}},
    {"entity_id": "lock.front_door", "state": "locked", "attributes": {"friendly_name": "大門鎖"}},
    {"entity_id": "cover.garage", "state": "closed", "attributes": {"friendly_name": "車庫門", "device_class": "garage"}},
    {"entity_id": "cover.curtain", "state": "open", "attributes": {"friendly_name": "客廳窗簾", "device_class": "curtain"}},
    {"entity_id": "media_player.xiaomi_lx06_7538_play_control", "state": "idle", "attributes": {"friendly_name": "小愛音箱Pro"}},
    {"entity_id": "sun.sun", "state": "below_horizon", "attributes": {}},
]


import json as _json
from datetime import datetime, timedelta, timezone
_NOW = datetime.now(timezone.utc)
TEMPLATE_ROWS = [  # entity → device, the shape HA's template engine returns
    {"e": "light.lamp5", "st": "on", "lc": _NOW.isoformat(), "d": "dev1", "dn": "皓檯燈", "et": "", "a": "皓房間"},
    {"e": "button.lamp5_reset", "st": "unknown", "lc": _NOW.isoformat(), "d": "dev1", "dn": "皓檯燈", "et": "", "a": "皓房間"},
    {"e": "sensor.hub_illum", "st": "unavailable", "lc": (_NOW - timedelta(days=3)).isoformat(), "d": "dev2", "dn": "Mi Control Hub", "et": "", "a": "客廳"},
    {"e": "button.hub_x", "st": "unknown", "lc": _NOW.isoformat(), "d": "dev2", "dn": "Mi Control Hub", "et": "", "a": "客廳"},
    {"e": "vacuum.s5", "st": "unavailable", "lc": (_NOW - timedelta(hours=5)).isoformat(), "d": "dev3", "dn": "Roborock S5", "et": "", "a": "客廳"},
    {"e": "sensor.cam_status", "st": "unavailable", "lc": _NOW.isoformat(), "d": "dev4", "dn": "攝影機", "et": "", "a": ""},
    {"e": "switch.cam_motion", "st": "on", "lc": _NOW.isoformat(), "d": "dev4", "dn": "攝影機", "et": "", "a": ""},
    {"e": "button.repeater", "st": "unknown", "lc": _NOW.isoformat(), "d": "dev5", "dn": "房間前wifi", "et": "", "a": ""},
    {"e": "sensor.sun_next_dawn", "st": "2026-10-12", "lc": _NOW.isoformat(), "d": "dev6", "dn": "Sun", "et": "DeviceEntryType.SERVICE", "a": ""},
    {"e": "sensor.xiaomi_message", "st": "hi", "lc": _NOW.isoformat(), "d": "", "dn": "Xiaomi message", "et": "", "a": ""},
    {"e": "update.hacs", "st": "off", "lc": _NOW.isoformat(), "d": "dev7", "dn": "Updater", "et": "", "a": ""},
]


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._p, self.text = status, payload, str(payload)
    def json(self): return self._p


class _Client:
    calls: list = []
    fail: dict = {}          # path → status code
    down = False
    assist_reply: dict = {}
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def request(self, method, url, json=None, headers=None):
        if _Client.down:
            raise httpx.ConnectError("refused")
        path = url.split("8123", 1)[1]
        _Client.calls.append((method, path, json, headers))
        if path in _Client.fail:
            return _Resp(_Client.fail[path], {"message": "nope"})
        if path == "/api/states":
            return _Resp(200, STATES)
        if path.startswith("/api/states/"):
            eid = path.rsplit("/", 1)[1]
            return _Resp(200, next(s for s in STATES if s["entity_id"] == eid))
        if path == "/api/config":
            return _Resp(200, {"version": "2026.10.1", "location_name": "家", "language": "zh-Hant"})
        if path == "/api/template":
            return _Resp(200, TEMPLATE_ROWS)
        if path == "/api/conversation/process":
            return _Resp(200, _Client.assist_reply)
        if path.startswith("/api/services/"):
            return _Resp(200, [{"entity_id": (json or {}).get("entity_id")}])
        return _Resp(404, {})


def setup(token="tok-123"):
    HA.httpx = types.SimpleNamespace(AsyncClient=_Client, HTTPError=httpx.HTTPError)  # type: ignore[assignment]
    config.ha_url, config.ha_token = "http://127.0.0.1:8123", token
    config.ha_language, config.xiaoai_entity = "zh-tw", ""
    config.ha_confirm_domains = HA.DEFAULT_CONFIRM_DOMAINS
    _Client.calls, _Client.fail, _Client.down = [], {}, False
    HA._pending.clear()


def posts():
    return [(p, j) for m, p, j, _ in _Client.calls if m == "POST" and p.startswith("/api/services/")]


# ── A ────────────────────────────────────────────────────────────────────────

def test_intent() -> None:
    for t in ("關客廳燈", "冷氣開 26 度", "電視音量調小", "打開臥室的窗簾", "turn off the living room lights", "冷氣調到除濕模式"):
        assert HI.parse_home_command(t) == t, t
    for t in ("把大門鎖上", "開車庫門", "關閉保全", "關於台北今天溫度", "開會提醒", "今天天氣如何",
              "幫我總結這份報告", "open the github page", "請把客廳的燈關掉然後幫我整理今天的所有行程並寄信給老闆",
              "[Attachments in workspace — use these paths directly, do not re-upload:\n  - uploads/a.pdf (1 KB)\n]\n\n開燈"):
        assert HI.parse_home_command(t) is None, t
    assert HI.parse_confirmation("確認 1234") == "1234" and HI.parse_confirmation("確認：5678") == "5678"
    assert HI.parse_confirmation("confirm 4321") == "4321"
    for t in ("確認 12345", "我確認 1234", "確認一下 1234 是什麼", "1234"):
        assert HI.parse_confirmation(t) is None, t
    for t, kind in (("家裡有哪些設備離線", "offline"), ("有設備斷線嗎", "offline"), ("what devices are offline", "offline"),
                    ("家裡有哪些設備", "list"), ("設備清單", "list"), ("列出所有裝置", "list")):
        assert HI.parse_home_query(t) == kind, t
    for t in ("我的網路離線了", "the wifi is offline", "關客廳燈", "離線時提醒我並寄信給老闆", "今天天氣"):
        assert HI.parse_home_query(t) is None, t
    print("  ✓ device commands recognised; sensitive words, other work and attachments are not; 確認 code exact")
    print("  ✓ 哪些設備離線 / 家裡有哪些設備 recognised; 'my internet is offline' is not")
    xa = {"叫小愛說：早安，該出門了！": ("say", "早安，該出門了！"), "請小愛音箱提醒出門": ("say", "提醒你：出門"),
          "用小愛播報 今天會下雨": ("say", "今天會下雨"), "叫小愛打開冷氣": ("command", "打開冷氣"),
          "跟小愛說打開客廳的冷氣": ("command", "打開客廳的冷氣")}
    for t, want in xa.items():
        assert HI.parse_xiaoai(t) == want, (t, HI.parse_xiaoai(t))
    for t in ("請小愛音箱每天週一到週五早上7：40分提醒出門", "叫小愛明天提醒我開會", "小愛音箱還在嗎", "every weekday tell xiaoai hi"):
        assert HI.parse_xiaoai(t) is None, t
    print("  ✓ 叫小愛說 / 提醒 → say, 叫小愛打開… / 跟小愛說… → command; recurring or future wording never fires now")
    music = {"幫我播放家裡的，小米音箱播放音樂": "播放音樂", "播放音樂": "播放音樂", "放點輕音樂": "放點輕音樂",
             "我想聽周杰倫的歌": "我想聽周杰倫的歌", "叫小愛放音樂": "放音樂", "小愛音量調到30": "音量調到30",
             "小米音箱下一首": "下一首", "停止音樂": "暫停播放", "play some music": "播放音樂", "stop the music": "暫停播放"}
    for t, cmd in music.items():
        assert HI.parse_xiaoai(t) == ("command", cmd), (t, HI.parse_xiaoai(t))
    for t in ("我要製作睡眠音樂", "電視播放音樂", "幫我整理一份播放音樂的清單", "小愛音箱還在嗎"):
        assert HI.parse_xiaoai(t) is None, t
    assert HI.looks_home_related("把氣氛弄浪漫一點的燈光") and HI.looks_home_related("小米音箱")
    assert not HI.looks_home_related("今天星期幾")
    print("  ✓ music → 小愛 command (小米音箱 / 音箱 names, stop, volume, English mapped to Chinese); "
          "TV, creative and multi-step requests are not")


# ── B ────────────────────────────────────────────────────────────────────────

async def test_client() -> None:
    setup(token="")
    assert not HA.configured()
    try:
        await HA.list_entities(); raise AssertionError
    except HA.HAError as exc:
        assert "Settings" in str(exc)
    setup()
    ents = await HA.list_entities()
    assert [e["entity_id"] for e in ents][:2] == ["light.living_room", "climate.bedroom_ac"]
    assert not any(e["entity_id"].startswith("sun.") for e in ents)
    assert [e["entity_id"] for e in await HA.list_entities(query="客廳")] == ["light.living_room", "cover.curtain"]
    assert [e["entity_id"] for e in await HA.list_entities(domain="climate")] == ["climate.bedroom_ac"]
    ac = (await HA.list_entities(domain="climate"))[0]
    assert ac["temperature"] == 26 and ac["current_temperature"] == 28.5
    assert _Client.calls[-1][3]["Authorization"] == "Bearer tok-123"
    st = await HA.status()
    assert st["version"] == "2026.10.1" and st["entities"] == len(STATES)
    assert st["xiaoai"] == "media_player.xiaomi_lx06_7538_play_control"
    print("  ✓ list / filter by name or domain, bearer token, status with 小愛 found automatically")

    r = await HA.control("light.living_room", "turn_off")
    assert r["done"] and posts()[-1] == ("/api/services/light/turn_off", {"entity_id": "light.living_room"})
    await HA.control("climate.bedroom_ac", "climate.set_temperature", {"temperature": 25})
    assert posts()[-1] == ("/api/services/climate/set_temperature", {"entity_id": "climate.bedroom_ac", "temperature": 25})
    await HA.control("cover.curtain", "close_cover")
    assert posts()[-1][0] == "/api/services/cover/close_cover"
    print("  ✓ control: lights, air-con with data, curtains run directly")

    n = len(posts())
    held = await HA.control("lock.front_door", "unlock")
    assert held["needs_confirmation"] and len(posts()) == n, "a lock is never touched before confirmation"
    garage = await HA.control("cover.garage", "open_cover")
    assert garage["needs_confirmation"] and len(posts()) == n
    r = await HA.confirm(held["needs_confirmation"])
    assert r["done"] and posts()[-1] == ("/api/services/lock/unlock", {"entity_id": "lock.front_door"})
    try:
        await HA.confirm(held["needs_confirmation"]); raise AssertionError
    except HA.HAError as exc:
        assert "過期" in str(exc)
    HA._pending[garage["needs_confirmation"]]["expires"] = time.monotonic() - 1
    try:
        await HA.confirm(garage["needs_confirmation"]); raise AssertionError
    except HA.HAError:
        pass
    assert len(posts()) == n + 1
    config.ha_confirm_domains = ""
    assert (await HA.control("lock.front_door", "lock"))["done"], "an emptied list turns confirmation off"
    config.ha_confirm_domains = HA.DEFAULT_CONFIRM_DOMAINS
    print("  ✓ locks and garage doors held for 確認 code; single use; expires; configurable")

    _Client.fail = {"/api/states": 401}
    try:
        await HA.list_entities(); raise AssertionError
    except HA.HAError as exc:
        assert "權杖" in str(exc)
    _Client.fail, _Client.down = {}, True
    try:
        await HA.status(); raise AssertionError
    except HA.HAError as exc:
        assert "連不到" in str(exc)
    _Client.down = False
    print("  ✓ rejected token and unreachable Home Assistant explained")

    _Client.assist_reply = {"response": {"response_type": "action_done", "speech": {"plain": {"speech": "已關閉客廳燈"}},
                                          "data": {"success": [{"name": "客廳燈", "id": "light.living_room"}], "failed": []}}}
    r = await HA.assist("關客廳燈")
    assert r == {"ok": True, "kind": "action_done", "speech": "已關閉客廳燈", "targets": ["客廳燈"], "failed": [], "code": None}
    assert _Client.calls[-1][2] == {"text": "關客廳燈", "language": "zh-tw"}
    _Client.assist_reply = {"response": {"response_type": "error", "speech": {"plain": {"speech": "抱歉，我聽不懂"}},
                                          "data": {"code": "no_intent_match"}}}
    r = await HA.assist("把氣氛弄浪漫一點")
    assert not r["ok"] and r["code"] == "no_intent_match"
    print("  ✓ Assist: done / not understood")

    await HA.xiaoai("晚餐好了")
    assert posts()[-1] == ("/api/services/xiaomi_miot/intelligent_speaker",
                           {"entity_id": "media_player.xiaomi_lx06_7538_play_control", "text": "晚餐好了",
                            "execute": False, "silent": False})
    await HA.xiaoai("打開冷氣", execute=True)
    assert posts()[-1][1]["execute"] is True
    _Client.fail = {"/api/services/xiaomi_miot/intelligent_speaker": 400}
    await HA.xiaoai("打開冷氣", execute=True)
    assert posts()[-1] == ("/api/services/text/set_value",
                           {"entity_id": "text.xiaomi_lx06_7538_play_control_execute_text_directive", "value": "打開冷氣"})
    _Client.fail = {}
    print("  ✓ 小愛: say, run as a spoken command, fallback to the text entity")

    devs = await HA.devices()
    names = {d["name"]: d for d in devs}
    assert set(names) == {"皓檯燈", "Mi Control Hub", "Roborock S5", "攝影機", "房間前wifi"}, set(names)
    assert names["皓檯燈"]["online"] and names["攝影機"]["online"] and names["房間前wifi"]["online"]
    assert not names["Mi Control Hub"]["online"] and names["Mi Control Hub"]["offline_for"] == "3 天"
    assert not names["Roborock S5"]["online"] and names["Roborock S5"]["offline_for"] == "5 小時"
    off = HA.describe_devices(devs, only_offline=True)
    assert off.startswith("📴 離線的設備（2 台）：") and "• 客廳 · Mi Control Hub，已離線 3 天" in off, off
    assert "其他 3 台在線" in off
    full = HA.describe_devices(devs, only_offline=False)
    assert full.startswith("🏠 家裡共有 5 台設備，3 台在線、2 台離線") and "**皓房間**" in full and "Roborock S5 📴 離線" in full, full
    assert HA.describe_devices([d for d in devs if d["online"]], True) == "✅ 所有 3 台設備都在線。"
    print("  ✓ devices: one row per physical device (no Sun / service / device-less / updates); "
          "offline only when nothing has a real state; how long; grouped by room")


# ── C ────────────────────────────────────────────────────────────────────────

async def test_tools() -> None:
    from tools import home_tool
    setup()
    r = await home_tool.home_devices(query="冷氣")
    assert r["success"] and r["result"][0]["entity_id"] == "climate.bedroom_ac"
    r = await home_tool.home_control("lock.front_door", "unlock")
    assert r["success"] and r["result"]["needs_confirmation"]
    r = await home_tool.home_control("nonsense", "turn_on")
    assert not r["success"] and "設備 ID" in r["error"]
    r = await home_tool.xiaoai("你好", mode="say")
    assert r["success"] and r["result"]["mode"] == "say"
    from tool_registry import tool_registry
    tool_registry.auto_register_all()
    r = await home_tool.home_overview(only_offline=True)
    assert r["success"] and r["result"]["summary"].startswith("📴")
    for name in ("home_devices", "home_overview", "home_state", "home_control", "home_assist", "xiaoai"):
        assert tool_registry.risk_of(name) == "low", name
    print("  ✓ tools: devices, held lock, bad id, 小愛; registered")


# ── D ────────────────────────────────────────────────────────────────────────

class _Tracker:
    status = None
    async def get_session(self, sid): return {"id": sid, "mode": "code"}
    async def create_task(self, req): pass
    async def update_task_status(self, tid, status, summary=None): _Tracker.status = (status, summary)
    async def create_subtask(self, st): pass
    async def update_subtask(self, st): pass
    async def list_tasks(self, **k): return []


async def test_orchestrator() -> None:
    from orchestrator import Orchestrator
    from models import TaskRequest, TaskStatus
    setup()

    class _ReachedPlanner(Exception):
        pass

    async def no_event(*a, **k): pass
    async def planner(*a, **k): raise _ReachedPlanner()

    def fake():
        o = types.SimpleNamespace(tracker=_Tracker(), _emit_task_event=no_event, _plan=planner,
                                  _classify_intent=planner, _fast_chat_response=planner)
        for name in ("_finish_direct", "_home_fast", "_home_confirm", "_home_overview", "_home_xiaoai", "_asked_again", "_typhoon",
                     "_direct_print", "_recent_uploads"):
            setattr(o, name, (lambda n: lambda *a, **k: getattr(Orchestrator, n)(o, *a, **k))(name))
        return o

    async def run(goal, **ctx):
        try:
            return await Orchestrator.run(fake(), TaskRequest(goal=goal, session_id="s1", context=ctx))
        except _ReachedPlanner:
            return "planner"
        except Exception as exc:  # noqa: BLE001 — the fake stops short of a real pipeline
            return f"pipeline:{type(exc).__name__}"

    _Client.assist_reply = {"response": {"response_type": "action_done", "speech": {"plain": {"speech": "已關閉客廳燈"}},
                                          "data": {"success": [{"name": "客廳燈"}], "failed": []}}}
    res = await run("關客廳燈")
    assert res.status == TaskStatus.COMPLETED and res.summary == "🏠 已關閉客廳燈", res
    print("  ✓ 關客廳燈 → Assist → done, no planning")

    _Client.assist_reply = {"response": {"response_type": "error", "data": {"code": "no_intent_match"}}}
    res = await run("把冷氣開到很冷")
    assert isinstance(res, str), f"Assist didn't understand → pipeline, got {res}"
    print("  ✓ Assist didn't understand → the normal pipeline takes over")

    held = await HA.control("lock.front_door", "unlock")
    code = held["needs_confirmation"]
    res = await run(f"確認 {code}", triggered_by="schedule")
    assert isinstance(res, str) and code in HA._pending, "a scheduled run can't confirm"
    n = len(posts())
    res = await run(f"確認 {code}")
    assert res.status == TaskStatus.COMPLETED and "lock.front_door" in res.summary
    assert posts()[n:] == [("/api/services/lock/unlock", {"entity_id": "lock.front_door"})]
    res = await run(f"確認 {code}")
    assert res.status == TaskStatus.FAILED and "過期" in res.summary
    print("  ✓ 確認 code from a person runs the held lock once; scheduled runs can't")

    res = await run("叫小愛說：早安，該出門了！")
    assert res.status == TaskStatus.COMPLETED and res.summary == "🔊 小愛說了：「早安，該出門了！」", res
    assert posts()[-1][1]["text"] == "早安，該出門了！" and posts()[-1][1]["execute"] is False
    res = await run("叫小愛打開冷氣")
    assert posts()[-1][1] == {"entity_id": "media_player.xiaomi_lx06_7538_play_control", "text": "打開冷氣",
                              "execute": True, "silent": False}
    print("  ✓ 叫小愛說… / 叫小愛打開冷氣 go straight to the speaker, no planning")

    res = await run("幫我播放家裡的，小米音箱播放音樂")
    assert res.summary == "🔊 已請小愛執行：「播放音樂」" and posts()[-1][1]["execute"] is True, res
    print("  ✓ the request that failed in real use now plays through 小愛")

    # A home request Assist can't handle must reach the planner, never the
    # no-tools chat lane (the classifier once said "the LLM can't control devices").
    calls = []
    async def classify(*a, **k):
        calls.append("classifier")
        return {"needs_tools": False, "is_pure_chat": True, "confidence": 0.95}
    async def chat(*a, **k): calls.append("chat"); raise _ReachedPlanner()
    async def plan(req, *a, **k):
        calls.append("planner")
        calls.append(str(getattr(req.mode, "value", req.mode)))
        raise _ReachedPlanner()
    _Client.assist_reply = {"response": {"response_type": "error", "data": {"code": "no_intent_match"}}}
    o = fake()
    o._classify_intent, o._fast_chat_response, o._plan = classify, chat, plan
    try:
        await Orchestrator.run(o, TaskRequest(goal="把氣氛弄浪漫一點的燈光", session_id="s1"))
    except Exception:  # noqa: BLE001
        pass
    assert "classifier" not in calls and "chat" not in calls, calls
    calls.clear()
    o = fake()
    o._classify_intent, o._fast_chat_response, o._plan = classify, chat, plan
    try:
        await Orchestrator.run(o, TaskRequest(goal="把燈光弄浪漫一點", session_id="s1",
                                              mode="chat", context={"mode_override": True}))
    except Exception:  # noqa: BLE001
        pass
    assert "chat" not in calls and calls[-2:] == ["planner", "code"], f"auto chat lane kept for a home request: {calls}"
    print("  ✓ home requests Assist can't handle skip the classifier and the auto chat lane")

    res = await run("家裡有哪些設備離線")
    assert res.status == TaskStatus.COMPLETED and res.summary.startswith("📴 離線的設備（2 台）"), res
    res = await run("家裡有哪些設備")
    assert res.summary.startswith("🏠 家裡共有 5 台設備"), res.summary
    print("  ✓ 哪些設備離線 / 家裡有哪些設備 answered straight from Home Assistant")

    _Client.down = True
    res = await run("關客廳燈")
    assert res.status == TaskStatus.FAILED and "連不到" in res.summary
    _Client.down = False
    config.ha_token = ""
    n = len(_Client.calls)
    res = await run("關客廳燈")
    assert isinstance(res, str) and len(_Client.calls) == n, "unconfigured → never contacted"
    print("  ✓ Home Assistant down → clear failure; not configured → untouched")


async def main() -> None:
    print("A. intent"); test_intent()
    print("B. client"); await test_client()
    print("C. tools"); await test_tools()
    print("D. orchestrator"); await test_orchestrator()
    print("\nALL HOME TESTS PASS")


asyncio.run(main())
