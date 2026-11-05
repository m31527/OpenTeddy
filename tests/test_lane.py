"""
Lane picking + the thinking switch — the two fixes for "579 s to say
what day it is".

    .venv/bin/python tests/test_lane.py
"""
from __future__ import annotations

import asyncio, logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

from config import config
config.decision_mode = "off"
import decide as _decide
async def _nolog(rec): pass
_decide.set_log_sink(_nolog)

import lane
import local_engine as L


def test_lane() -> None:
    """No keyword rules any more: a message nobody picked a mode for starts
    in the chat lane with the gate; files attached go straight to work; a
    session the person set to chat stays chat."""
    assert lane.auto_lane("code") == ("chat", {"mode_override": True, "gate": "code"})
    assert lane.auto_lane("analytic") == ("chat", {"mode_override": True, "gate": "analytic"})
    assert lane.auto_lane("code", has_attachments=True) == ("code", {})
    assert lane.auto_lane("chat") == ("chat", {})
    assert not hasattr(lane, "looks_conversational"), "the keyword rules are gone"
    print("  ✓ auto_lane: chat lane with the gate; attachments → work; a chat session stays chat")


def test_think() -> None:
    config.ollama_think = False
    assert L.think_setting("qwen3.8:27b") is False
    assert L.think_setting("gemma4:26b") is False
    assert L.think_setting("deepseek-r1:14b") is False
    assert L.think_setting("qwen2.5:3b") is None          # no switch on this family → never sent
    assert L.think_setting("gemma3:4b") is None
    assert L.think_setting("llama3.1:8b") is None
    p = L.build_payload(model="qwen3.8:27b", messages=[{"role": "user", "content": "hi"}], system="s",
                        tools=None, stream=False, temperature=0.1, num_predict=10)
    assert p.get("think") is False and p["messages"][0]["role"] == "system"
    p = L.build_payload(model="qwen2.5:3b", messages=[{"role": "user", "content": "hi"}], system="s",
                        tools=None, stream=False, temperature=0.1, num_predict=10)
    assert "think" not in p
    config.ollama_think = True
    assert L.think_setting("qwen3.8:27b") is None
    p = L.build_payload(model="qwen3.8:27b", messages=[{"role": "user", "content": "hi"}], system=None,
                        tools=None, stream=False, temperature=0.1, num_predict=10)
    assert "think" not in p
    config.ollama_think = False
    print("  ✓ think switch: off for thinking families, never sent to others, OLLAMA_THINK=true restores default")


def test_fast_chat_sentinel() -> None:
    """_gemma_complete returns "[]" on error; the chat lane must treat it
    as a failure (with the reason), never as the answer."""
    import types
    from models import TaskRequest
    from orchestrator import Orchestrator

    async def complete_sentinel(*a, **k): return "[]"
    async def complete_ok(*a, **k): return "今天是星期四。"
    async def no_recent(*a, **k): return ""
    fake = types.SimpleNamespace(memory=None, _last_gemma_error="ReadTimeout",
                                 _orchestrator_complete=complete_sentinel,
                                 _recent_turns_block=no_recent,
                                 tracker=types.SimpleNamespace())
    req = TaskRequest(goal="Hi", session_id="s")

    async def run():
        try:
            await Orchestrator._fast_chat_response(fake, req, "chat", {"confidence": 0.9})
            raise AssertionError("sentinel was accepted as an answer")
        except RuntimeError as exc:
            assert "ReadTimeout" in str(exc) and "no answer" in str(exc), exc
        # a real answer still goes through to the tracker write (which we stub)
        calls = []
        async def _rec(*a, **k): calls.append(a)
        fake.tracker = types.SimpleNamespace(create_subtask=_rec, update_subtask=_rec, update_task_status=_rec)
        fake._orchestrator_complete = complete_ok
        r = await Orchestrator._fast_chat_response(fake, req, "chat", {"confidence": 0.9})
        assert r.summary.strip() == "今天是星期四。" and len(calls) == 3, (r, calls)
    asyncio.run(run())
    print("  ✓ fast chat: '[]' sentinel raises with the planner's error; a real answer passes")


def test_chat_promise() -> None:
    """A chat-lane answer that only promises to act goes to the work lane —
    unless the person set the session to chat themselves."""
    import types
    from models import SessionMode, TaskRequest, TaskStatus
    import orchestrator as O

    promise = "好的，我會為您查詢 momo 購物網上 Vivo X300 的最低價格。\n\n請稍候，我正在執行搜尋..."
    said = {"text": promise}
    async def complete(*a, **k): return said["text"]
    async def no_recent(*a, **k): return ""
    async def _rec(*a, **k): pass
    fake = types.SimpleNamespace(memory=None, _orchestrator_complete=complete, _recent_turns_block=no_recent,
                                 tracker=types.SimpleNamespace(create_subtask=_rec, update_subtask=_rec,
                                                               update_task_status=_rec))
    auto = TaskRequest(goal="去 momo 購物官網看看 vivo x300 最便宜多少錢", session_id="s",
                       mode="chat", context={"mode_override": True})
    mine = TaskRequest(goal="你會幫我查資料嗎", session_id="s", mode="chat")

    async def run():
        for answer in (promise, "Sure — let me check momo for you.", "稍等一下，我馬上幫你查"):
            said["text"] = answer
            try:
                await O.Orchestrator._fast_chat_response(fake, auto, "chat", {"confidence": 0.9})
                raise AssertionError(f"promise accepted as an answer: {answer!r}")
            except O._NeedsWork:
                pass
        said["text"] = promise
        r = await O.Orchestrator._fast_chat_response(fake, mine, "chat", {"confidence": 0.9})
        assert r.status == TaskStatus.COMPLETED, "an explicit chat session stays chat"
        for answer in ("今天是星期四。", "我會為您查詢的話，需要先連接瀏覽器。" + "說明" * 100):
            said["text"] = answer
            r = await O.Orchestrator._fast_chat_response(fake, auto, "chat", {"confidence": 0.9})
            assert r.status == TaskStatus.COMPLETED, answer[:30]

        # Through run(): the promise never reaches the person; the planner
        # gets the request in the work lane, with tools.
        class _Tracker:
            async def get_session(self, sid): return {"id": sid, "mode": "code"}
            async def create_task(self, req): pass
            async def update_task_status(self, *a, **k): pass
            async def create_subtask(self, st): pass
            async def update_subtask(self, st): pass
            async def list_tasks(self, **k): return []

        class _Planned(Exception):
            pass
        seen = []
        async def no_event(*a, **k): pass
        async def classify(*a, **k):
            seen.append("classifier")
            return {"needs_tools": False, "is_pure_chat": True, "confidence": 0.95}
        async def chat(*a, **k):
            seen.append("chat")
            raise O._NeedsWork(promise[:80])
        async def plan(req):
            seen.append(("planner", str(getattr(req.mode, "value", req.mode))))
            raise _Planned()
        o = types.SimpleNamespace(tracker=_Tracker(), _emit_task_event=no_event, _plan=plan,
                                  _classify_intent=classify, _fast_chat_response=chat)
        for name in ("_finish_direct", "_asked_again", "_typhoon", "_direct_print", "_recent_uploads"):
            setattr(o, name, (lambda n: lambda *a, **k: getattr(O.Orchestrator, n)(o, *a, **k))(name))
        req = TaskRequest(goal="你好嗎？我想知道 momo 的事", session_id="s1", mode="chat",
                          context={"mode_override": True})
        try:
            await O.Orchestrator.run(o, req)
        except Exception:  # noqa: BLE001 — the fake stops short of a real pipeline
            pass
        assert seen == ["chat", ("planner", "code")], seen
        assert req.mode == SessionMode.CODE
    asyncio.run(run())
    print("  ✓ chat lane: 「請稍候，我正在執行搜尋…」 goes to the planner in the work lane; "
          "a chat session the person picked stays chat")


def test_gate() -> None:
    """The gate: one call to the chat model with start_task. An answer is
    the answer; a start_task call sends the request to the session's own
    mode; the prompt lists the registered tools."""
    import types
    from models import SessionMode, TaskRequest, TaskStatus
    import gate as G
    import orchestrator as O
    from tool_registry import tool_registry

    tool_registry.auto_register_all()
    G._system_cache.clear()
    sp = G.system_prompt()
    assert "- browser_open:" in sp and "- weather_forecast:" in sp and "start_task" in sp, sp[:400]
    print("  ✓ gate prompt: the agent's abilities come from the tool registry (%d tools)"
          % sp.count("\n- "))

    reply = {}
    sent = []

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"message": dict(reply), "eval_count": 3, "eval_duration": 10**7}

    async def post(url, json=None):
        sent.append((url, json))
        return _Resp()

    async def no_recent(*a, **k): return ""
    async def _rec(*a, **k): pass
    trk = types.SimpleNamespace(create_subtask=_rec, update_subtask=_rec, update_task_status=_rec, record_usage=_rec)
    fake = types.SimpleNamespace(memory=None, _recent_turns_block=no_recent, tracker=trk,
                                 _http=types.SimpleNamespace(post=post))
    req = TaskRequest(goal="去 momo 購物官網看看 vivo x300 最便宜多少錢", session_id="s", mode="chat",
                      context={"mode_override": True, "gate": "code"})

    async def run():
        reply.clear()
        reply.update({"content": "", "tool_calls": [{"function": {"name": "start_task",
                                                                  "arguments": {"what": "momo 查 vivo x300 價格"}}}]})
        try:
            await O.Orchestrator._fast_chat_response(fake, req, "chat", {"confidence": 0.9})
            raise AssertionError("start_task was not followed")
        except O._NeedsWork as exc:
            assert "momo 查 vivo x300 價格" in str(exc), exc
        url, body = sent[-1]
        assert url.endswith("/api/chat") and body["tools"][0]["function"]["name"] == "start_task"
        assert body["messages"][0]["role"] == "system" and "vivo x300" in body["messages"][1]["content"]
        reply.clear()
        reply.update({"content": "今天是星期四。"})
        r = await O.Orchestrator._fast_chat_response(fake, req, "chat", {"confidence": 0.9})
        assert r.status == TaskStatus.COMPLETED and r.summary == "今天是星期四。", r
        reply.clear()
        reply.update({"content": "好的，我去 momo 幫你查，請稍候…"})
        try:
            await O.Orchestrator._fast_chat_response(fake, req, "chat", {"confidence": 0.9})
            raise AssertionError("a written promise got through the gate")
        except O._NeedsWork:
            pass

        # Through run(): start_task → the planner, in the session's own mode.
        class _Tracker:
            async def get_session(self, sid): return {"id": sid, "mode": "analytic"}
            async def create_task(self, req): pass
            async def update_task_status(self, *a, **k): pass
            async def create_subtask(self, st): pass
            async def update_subtask(self, st): pass
            async def list_tasks(self, **k): return []
        seen = []
        async def no_event(*a, **k): pass
        async def classify(*a, **k):
            seen.append("classifier")
            return {"needs_tools": False, "is_pure_chat": True, "confidence": 0.95}
        async def chat(*a, **k):
            seen.append("gate")
            raise O._NeedsWork("start_task: 查營收")
        async def plan(r):
            seen.append(("planner", r.mode.value))
            raise RuntimeError("stop here")
        o = types.SimpleNamespace(tracker=_Tracker(), _emit_task_event=no_event, _plan=plan,
                                  _classify_intent=classify, _fast_chat_response=chat)
        for name in ("_finish_direct", "_asked_again", "_typhoon", "_direct_print", "_recent_uploads"):
            setattr(o, name, (lambda n: lambda *a, **k: getattr(O.Orchestrator, n)(o, *a, **k))(name))
        r2 = TaskRequest(goal="上個月北區營收多少", session_id="s1", mode="chat",
                         context={"mode_override": True, "gate": "analytic"})
        try:
            await O.Orchestrator.run(o, r2)
        except Exception:  # noqa: BLE001
            pass
        assert seen == ["gate", ("planner", "analytic")], seen
        assert r2.mode == SessionMode.ANALYTIC
    asyncio.run(run())
    print("  ✓ gate: start_task → the session's own mode (analytic stays analytic); an answer is the answer; "
          "a written promise still goes to work")


if __name__ == "__main__":
    test_lane()
    test_think()
    test_fast_chat_sentinel()
    test_chat_promise()
    test_gate()
    print("\nALL LANE TESTS PASS")
