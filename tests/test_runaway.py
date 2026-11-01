"""
The lamp that took eighteen minutes — and what stops it happening again.

"幫我把皓房間檯燈打開" missed the lamp called 皓檯燈, the agent looped for
8.5 minutes; the reply "燈真的打開了，但你好像執行非常久" ran the device
pipeline for another 10, until Telegram's limit, and the task stayed
"running" and couldn't be cancelled.

  A. remarks: a comment on what happened is not a command (chat lane)
  B. names: 皓房間檯燈 → 皓檯燈 by name / alias; ambiguity and parameters left alone
  C. home fast path: Assist misses → the name match turns the lamp on in one call
  D. retries: a looped / out-of-rounds attempt isn't run again locally;
     other retries carry what went wrong
  E. executor records why an attempt ended; the retry prompt includes the failure
  F. Telegram: a stopped task is recorded; /tasks/{id}/cancel reaches Telegram tasks

    .venv/bin/python tests/test_runaway.py
"""
from __future__ import annotations

import asyncio, logging, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from config import config
import home_assistant as HA
from home_intent import is_remark, parse_home_command, looks_home_related


def test_remarks() -> None:
    for t in ("但是燈真的打開了，但你好像執行非常久的時間", "燈開了", "謝謝，燈打開了", "為什麼要開那麼久",
              ):
        assert is_remark(t), t
        assert parse_home_command(t) is None and not looks_home_related(t), t
    for t in ("幫我把燈打開", "關燈", "燈打開了嗎", "燈沒有打開", "你好像沒有開燈", "再幫我打開一次燈",
              "打開客廳燈", "冷氣開 26 度"):
        assert not is_remark(t), t
    assert parse_home_command("幫我把皓房間檯燈打開") and looks_home_related("燈沒有打開")
    print("  ✓ remarks about a past action aren't commands; requests, questions and 'it didn't work' are")


STATES = [
    {"entity_id": "light.yeelink_lamp5_2937_light", "state": "off", "attributes": {"friendly_name": "Light"}},
    {"entity_id": "light.xiaomi_lamp32_9303_light", "state": "on", "attributes": {"friendly_name": "灯"}},
    {"entity_id": "light.yeelight_lamp1_5d78_light", "state": "off", "attributes": {"friendly_name": "灯"}},
    {"entity_id": "fan.zhimi_ma4_a2c0_air_purifier", "state": "on", "attributes": {"friendly_name": "空氣淨化器"}},
    {"entity_id": "switch.living_wall", "state": "off", "attributes": {"friendly_name": "客廳開關"}},
    {"entity_id": "lock.front", "state": "locked", "attributes": {"friendly_name": "大門"}},
    {"entity_id": "sensor.temp", "state": "26", "attributes": {"friendly_name": "皓房間溫度"}},
    {"entity_id": "light.study_a", "state": "off", "attributes": {"friendly_name": "書房燈"}},
    {"entity_id": "light.study_b", "state": "off", "attributes": {"friendly_name": "書房吊燈"}},
]
NAMES = {"light.yeelink_lamp5_2937_light": ["Light", "皓檯燈"],
         "light.xiaomi_lamp32_9303_light": ["灯", "桌燈", "LED檯燈"],
         "light.yeelight_lamp1_5d78_light": ["灯", "檯燈"]}
CALLS: list = []


async def _fake_request(method, path, json=None, timeout=10.0):
    CALLS.append((method, path, json))
    if path == "/api/states":
        return STATES
    if path.startswith("/api/services/"):
        return [{"entity_id": json["entity_id"]}]
    if path == "/api/conversation/process":
        return {"response": {"response_type": "error", "data": {"code": "no_valid_targets"}}}
    return {}


async def test_names() -> None:
    HA._request = _fake_request
    async def names(ids): return NAMES
    HA._registry_names = names
    f = HA.find_on_off
    hit = await f("幫我把皓房間檯燈打開")
    assert hit == {"entity_id": "light.yeelink_lamp5_2937_light", "name": "皓檯燈", "service": "turn_on"}, hit
    assert (await f("關掉桌燈"))["entity_id"] == "light.xiaomi_lamp32_9303_light"
    assert (await f("幫我關空氣淨化器"))["service"] == "turn_off"
    assert (await f("打開客廳開關"))["entity_id"] == "switch.living_wall", "開關 is a noun, not on + off"
    assert (await f("打開檯燈"))["entity_id"] == "light.yeelight_lamp1_5d78_light", "an exact alias wins"
    for t in ("打開燈", "打開書房", "冷氣開26度", "把燈調亮", "打開大門", "皓房間溫度"):
        assert await f(t) is None, t
    print("  ✓ 皓房間檯燈 → 皓檯燈 by alias; exact alias wins; ties, numbers, locks, sensors left alone")


async def test_home_fast() -> None:
    from orchestrator import Orchestrator
    from models import TaskRequest, TaskStatus
    rows = []

    class _T:
        async def create_subtask(self, st): pass
        async def update_subtask(self, st): pass
        async def update_task_status(self, tid, status, summary=None): rows.append(summary)
    o = types.SimpleNamespace(tracker=_T())
    o._finish_direct = lambda *a, **k: Orchestrator._finish_direct(o, *a, **k)
    CALLS.clear()
    res = await Orchestrator._home_fast(o, TaskRequest(goal="幫我把皓房間檯燈打開"), "幫我把皓房間檯燈打開")
    assert res.status == TaskStatus.COMPLETED and res.summary == "🏠 已打開「皓檯燈」", res.summary
    assert ("POST", "/api/services/light/turn_on", {"entity_id": "light.yeelink_lamp5_2937_light"}) in CALLS
    res = await Orchestrator._home_fast(o, TaskRequest(goal="把房間的燈打開"), "把房間的燈打開")
    assert res is None, "no single match → the agent as before"
    print("  ✓ Assist misses the name → one direct call turns 皓檯燈 on (no agent)")


async def test_remark_lane() -> None:
    from orchestrator import Orchestrator
    from models import TaskRequest, TaskStatus, TaskResult
    chats = []

    class _T:
        async def get_session(self, sid): return {"id": sid, "mode": "code", "workspace_dir": None}
        async def create_task(self, req): pass
        async def update_task_status(self, *a, **k): pass
        async def create_subtask(self, st): pass
        async def update_subtask(self, st): pass
        async def list_tasks(self, **k): return []

    async def no_event(*a, **k): pass
    async def planner(*a, **k): raise RuntimeError("planner reached")
    async def chat(req, mode, intent):
        chats.append(req.goal)
        return TaskResult(task_id=req.id, status=TaskStatus.COMPLETED, summary="抱歉讓你久等了")
    o = types.SimpleNamespace(tracker=_T(), _emit_task_event=no_event, _plan=planner,
                              _classify_intent=planner, _fast_chat_response=chat)
    for n in ("_finish_direct", "_claude_code", "_asked_again", "_reminder", "_personal", "_google_link"):
        setattr(o, n, (lambda name: lambda *a, **k: getattr(Orchestrator, name)(o, *a, **k))(n))
    async def boom(*a, **k): raise AssertionError("device pipeline reached")
    o._home_fast = boom
    config.ha_token = "x"
    res = await Orchestrator.run(o, TaskRequest(goal="但是燈真的打開了，但你好像執行非常久的時間", session_id="s"))
    assert res.summary == "抱歉讓你久等了" and chats, "a remark gets a reply on the chat lane"
    config.ha_token = ""
    print("  ✓ the remark gets one chat reply — no Assist, no agent, no ten minutes")


async def test_retries() -> None:
    from orchestrator import Orchestrator
    from models import SubTask, TaskStatus, AgentRole
    seen, escalated = [], []

    class _Exec:
        def __init__(self, reasons): self.reasons = list(reasons)
        async def execute(self, st, ctx, mode="code"):
            seen.append(dict(ctx))
            st.status, st.confidence, st.result = TaskStatus.COMPLETED, 0.3, "I am a language model"
            st.error = None
            self.reason = self.reasons.pop(0) if self.reasons else None
            return st
        def pop_exit_reason(self, sid):
            r, self.reason = getattr(self, "reason", None), None
            return r

    class _Esc:
        async def resolve(self, st, ctx):
            escalated.append(st.id)
            st.status = TaskStatus.COMPLETED
            return st

    class _T:
        async def update_subtask(self, st): pass

    async def run(reasons):
        seen.clear(); escalated.clear()
        o = types.SimpleNamespace(executor=_Exec(reasons), escalation=_Esc(), tracker=_T(), _ollama_busy=False)
        st = SubTask(parent_task_id="t", description="打開皓檯燈", agent=AgentRole.EXECUTOR, order=0)
        await Orchestrator._run_subtask(o, st, {"x": 1}, mode="code")
    config.llm_mode = "mixed"
    await run(["loop"])
    assert len(seen) == 1 and escalated, "a looped attempt isn't run again locally"
    await run(["max_rounds"])
    assert len(seen) == 1 and escalated
    await run([None, None, None])
    assert len(seen) == 3 and escalated
    assert "previous_attempt_failed" not in seen[0]
    assert "low confidence (0.30)" in seen[1]["previous_attempt_failed"] and seen[1]["x"] == 1
    print("  ✓ loop / round limit → no identical local retry; other retries say what failed")


async def test_executor_reason() -> None:
    import executor as EX
    from tool_registry import ToolRegistry

    class _Reg:
        def get_schemas(self):
            return [{"type": "function", "function": {"name": "home_state", "description": "",
                                                       "parameters": {"type": "object", "properties": {}}}}]
        def get_schemas_by_names(self, n): return self.get_schemas()
        def risk_of(self, n): return "medium"
        async def execute(self, name, args, **k): return {"success": True, "result": {"state": "on"}, "error": None}

    class _Http:
        def __init__(self): self.payloads = []
        async def post(self, url, json=None, **k):
            self.payloads.append(json)
            tc = [{"function": {"name": "home_state", "arguments": {"entity_id": "light.x"}}}]
            final = len(self.payloads) > 8 or json.get("tools") is None
            data = {"message": {"content": '{"result": "done", "confidence": 0.8}' if final else "",
                                "tool_calls": [] if final else tc}, "prompt_eval_count": 10, "eval_count": 5}
            class _R:
                def raise_for_status(self): pass
                def json(self): return data
            return _R()
        async def aclose(self): pass

    class _Tr:
        async def record_usage(self, **k): pass
        async def append_task_artifact(self, *a, **k): pass
    config.streaming_enabled = False
    ex = EX.Executor(tracker=_Tr(), skill_factory=None, registry=_Reg())
    await ex._http.aclose()
    ex._http = _Http()
    await ex._qwen_execute("打開皓檯燈", {"previous_attempt_failed": "loop detector: home_state × 3"},
                           task_id="t", subtask_id="s1")
    first = [m for m in ex._http.payloads[0]["messages"] if m["role"] == "user"][0]["content"]
    assert "PREVIOUS ATTEMPT AT THIS STEP FAILED" in first and "home_state × 3" in first
    assert ex.pop_exit_reason("s1") == "loop" and ex.pop_exit_reason("s1") is None
    print("  ✓ executor: the retry prompt carries the failure; a loop is reported as 'loop'")


async def test_telegram_stop() -> None:
    import telegram_bridge as TB
    import main
    from models import TaskStatus
    marked = []

    class _T:
        async def update_task_status(self, tid, status, summary=None): marked.append((tid, status, summary))
    TB._tracker = _T()
    await TB._mark_stopped("t9", "⌛ 超過 10 分鐘，已強制中止")
    assert marked == [("t9", TaskStatus.FAILED, "⌛ 超過 10 分鐘，已強制中止")]
    t = asyncio.ensure_future(asyncio.sleep(30))
    TB.RUNNING["tg-task"] = t
    assert (await main.cancel_task("tg-task")) == {"cancelled": True, "task_id": "tg-task"}
    await asyncio.sleep(0)
    assert t.cancelled()
    TB.RUNNING.clear()
    print("  ✓ a timed-out Telegram task is recorded as failed; cancel reaches Telegram tasks")


async def main_() -> None:
    print("A. remarks"); test_remarks()
    print("B. names"); await test_names()
    print("C. home fast path"); await test_home_fast()
    print("D. remark lane"); await test_remark_lane()
    print("E. retries"); await test_retries(); await test_executor_reason()
    print("F. Telegram"); await test_telegram_stop()
    print("\nALL RUNAWAY TESTS PASS")


asyncio.run(main_())
