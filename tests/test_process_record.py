"""
The iPhone 18 price list that said "not on sale yet" (2026-11-11).

The run read Shopee, PChome and momo and wrote a PChome price table on the
way; momo went round in circles and the step ended on half a sentence
("All results on page 1 are Pro and"). A keyword rule failed it for not
writing a file ("以程式碼輸出為 JSON 或 Markdown 表格"), and the summary —
shown 500 chars of that half sentence — said iPhone 18 wasn't released.
Claude Fix then ran out of turns and replaced even that with "ran 15 turns
without converging". Whoever didn't watch the process saw a useless system.

  A. executor: what the model writes between tool calls is kept
  B. summary input: whole results, plus the notes when they add something
  C. file check: the model decides; an answer given in the reply passes
  D. Claude Fix: out of turns → one more call answers with what it found;
     a fix that brings nothing back keeps the earlier answer
  E. process record: notes and steps saved with the task, served when
     opened; a private task keeps none; a Claude Fix is appended after it

    .venv/bin/python tests/test_process_record.py
"""
from __future__ import annotations

import asyncio, logging, os, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from config import config

TABLE = ("PChome 24h購物 - iPhone 18 價格整理\n\n| 型號 | 容量 | 價格 |\n|---|---|---|\n"
         "| iPhone 18 | 256GB | $30,900 |\n| iPhone 18 Pro | 256GB | $43,900 |\n"
         "| iPhone 18 Pro Max | 256GB | $50,900 |")
HALF = "All results on page 1 are Pro and"


class _Reg:
    def get_schemas(self):
        return [{"type": "function", "function": {"name": "browser_read", "description": "",
                                                   "parameters": {"type": "object", "properties": {}}}}]
    def get_schemas_by_names(self, n): return self.get_schemas()
    def risk_of(self, n): return "low"
    async def execute(self, name, args, **k): return {"success": True, "result": "page text", "error": None}


def _executor(replies):
    """An Executor whose model replies with `replies` in turn: (content, call a tool?)."""
    import executor as EX

    class _Http:
        def __init__(self): self.n = 0
        async def post(self, url, json=None, **k):
            content, tool = replies[min(self.n, len(replies) - 1)]
            self.n += 1
            tc = [{"function": {"name": "browser_read", "arguments": {"tab": self.n}}}] if tool else []
            data = {"message": {"content": content, "tool_calls": tc}, "prompt_eval_count": 10, "eval_count": 5}
            class _R:
                def raise_for_status(self): pass
                def json(self): return data
            return _R()
        async def aclose(self): pass

    class _Tr:
        async def record_usage(self, **k): pass
        async def append_task_artifact(self, *a, **k): pass

    events = []

    async def _cb(ev): events.append(ev)
    ex = EX.Executor(tracker=_Tr(), skill_factory=None, registry=_Reg(), ws_callback=_cb)
    return ex, _Http, events


async def test_notes() -> None:
    config.streaming_enabled = False
    ex, Http, events = _executor([("I'll read each site.", True), (TABLE, True),
                                  ("Let me read momo.", True), (HALF, False)])
    await ex._http.aclose()
    ex._http = Http()
    result, *_ = await ex._qwen_execute("找 iPhone 18 價格", {}, task_id="t1", subtask_id="s1")
    assert result.strip() == HALF
    notes = [e["text"] for e in events if e.get("type") == "process.note"]
    assert notes == ["I'll read each site.", TABLE, "Let me read momo."], notes
    kept = ex.pop_notes("s1")
    assert TABLE in kept and kept.index("I'll read") < kept.index("PChome") and ex.pop_notes("s1") == ""

    ex._notes["s2"] = ["a" * 50, "b" * 2000, "c" * 20, "d" * 1500]
    kept = ex.pop_notes("s2", budget=3000)
    assert "b" * 2000 in kept and "d" not in kept and kept.index("a") < kept.index("b") < kept.index("c")
    print("  ✓ the model's words between tool calls are kept (and sent on when not streamed); "
          "over budget the long ones stay, in order")


async def test_step_log() -> None:
    from orchestrator import Orchestrator
    from models import SubTask, TaskStatus, AgentRole
    ex, _, _ = _executor([("", False)])
    await ex._http.aclose()
    st = SubTask(parent_task_id="t1", description="搜尋各平台 iPhone 18 售價", agent=AgentRole.EXECUTOR, order=0)
    st.status, st.result = TaskStatus.FAILED, HALF
    ex._notes[st.id] = ["I'll read each site.", TABLE]
    long_st = SubTask(parent_task_id="t1", description="整理成表格", agent=AgentRole.EXECUTOR, order=1)
    long_st.status, long_st.result = TaskStatus.COMPLETED, "| x |\n" * 900
    ex._notes[long_st.id] = ["| x |"]                      # already in the result → not repeated
    o = types.SimpleNamespace(executor=ex)
    lines, notes = Orchestrator._step_log(o, "搜尋 iPhone 18 的售價", [st, long_st])
    log = "\n".join(lines)
    assert "過程筆記" in log and "$30,900" in log and HALF in log
    assert log.count("| x |") > 500, "a long table isn't cut at 500 chars"
    assert notes == ["I'll read each site.\n\n" + TABLE]
    lines, _ = Orchestrator._step_log(o, "find iPhone 18 prices", [st])
    assert "Working notes" not in "\n".join(lines), "notes were taken once"
    print("  ✓ summary input: whole results, a cut-off step's notes (the price table), each once")


async def test_file_check() -> None:
    from orchestrator import Orchestrator
    from models import SubTask, TaskStatus, AgentRole
    from config import set_session_workspace, set_session_local_only
    set_session_workspace(tempfile.mkdtemp(prefix="ot-guard-"))
    set_session_local_only(True)          # a failed step stops here instead of escalating
    asked = []

    class _Exec:
        async def execute(self, st, ctx, mode="code"):
            st.status, st.confidence, st.result, st.error = TaskStatus.COMPLETED, 0.8, TABLE, None
            return st
        def pop_subtask_artifacts(self, sid): return []
        def pop_exit_reason(self, sid): return None

    class _T:
        async def update_subtask(self, st): pass

    async def run(must):
        async def _must(st):
            asked.append(st.description)
            return must
        async def _none(*a, **k): return None
        o = types.SimpleNamespace(executor=_Exec(), tracker=_T(), _ollama_busy=False,
                                  _step_must_change_files=_must, _verify_subtask=_none,
                                  _verify_deliverable=_none, escalation=None)
        st = SubTask(parent_task_id="t", agent=AgentRole.EXECUTOR, order=0,
                     description="搜尋 iPhone 18 售價，將結果以程式碼輸出為結構化格式（如 JSON 或 Markdown 表格）")
        return await Orchestrator._run_subtask(o, st, {}, mode="code")

    config.llm_mode = "local"
    st = await run(False)
    assert st.status == TaskStatus.COMPLETED and st.result == TABLE and len(asked) == 1
    asked.clear()
    st = await run(True)
    assert st.status == TaskStatus.FAILED and "沒有任何檔案被改動" in (st.error or "") and len(asked) == 3
    set_session_local_only(False)
    print("  ✓ file check: the model says whether a file had to change — an answer in the reply passes")


async def test_claude_fix() -> None:
    import escalation as ES
    from llm_provider import LLMToolTurnResponse, LLMToolUse, LLMUsage

    class _Hist:
        def __init__(self): self.results = []
        def add_user_message(self, t): pass
        def add_assistant_turn(self, r): pass
        def add_tool_results(self, rs): self.results.append(rs)

    class _Prov:
        provider_name, model_name = "anthropic", "claude-test"
        def __init__(self, wrap): self.calls, self.wrap, self.hist = 0, wrap, None
        def new_history(self):
            self.hist = _Hist()
            return self.hist
        async def complete_with_tools(self, history, tools, system=None, max_tokens=4096):
            self.calls += 1
            if self.calls > 3:                                  # the wrap-up call
                return LLMToolTurnResponse(text=self.wrap, usage=LLMUsage())
            text = TABLE if self.calls == 2 else "Reading momo."
            return LLMToolTurnResponse(text=text, usage=LLMUsage(),
                                       tool_uses=[LLMToolUse(id=f"c{self.calls}", name="browser_read", input={})])

    class _Tr:
        async def record_usage(self, **k): pass

    events = []

    async def _cb(ev): events.append(ev)
    agent = ES.EscalationAgent.__new__(ES.EscalationAgent)
    agent.tracker = _Tr()
    agent._escalation_blocked = lambda: None
    for wrap, want in (("PChome 有 3 款：…；momo 沒有讀完。", "PChome 有 3 款：…；momo 沒有讀完。"), ("", TABLE)):
        agent.provider = _Prov(wrap)
        events.clear()
        r = await agent.resolve_whole_task(task={"id": "t1", "goal": "iPhone 18 價格"}, subtasks=[],
                                           tool_registry=_Reg(), max_turns=3, ws_callback=_cb)
        assert r["partial"] and not r["success"] and r["summary"] == want, r
        last = agent.provider.hist.results[-1][-1].content
        assert "last tool turn" in last and "Do NOT call" in last
        assert [e["text"] for e in events if e.get("type") == "process.note"][1] == TABLE
    print("  ✓ Claude Fix out of turns: one more call answers with what it found "
          "(else the fullest note); its notes go to the process record")

    import main
    from tracker import Tracker
    from models import TaskRequest, TaskStatus
    tr = Tracker(os.path.join(tempfile.mkdtemp(prefix="ot-fix-"), "t.db"))
    await tr.open()
    old = (getattr(main, "tracker", None), getattr(main, "escalation_agent", None))
    try:
        main.tracker = tr
        for tid, res, want_status, want_in in (
            ("f1", {"success": False, "summary": "anthropic API error: 529"}, "escalated", "PChome 找到 3 款"),
            ("f2", {"success": False, "partial": True, "summary": "部分結果：PChome …"}, "completed", "部分結果"),
        ):
            await tr.create_task(TaskRequest(id=tid, goal="iPhone 18 價格", session_id=""))
            await tr.update_task_status(tid, TaskStatus.FAILED, "PChome 找到 3 款")

            class _Esc:
                async def resolve_whole_task(self, **k): return res
            main.escalation_agent = _Esc()
            await main.claude_fix_task(tid)
            for _ in range(50):
                await asyncio.sleep(0.02)
                row = await tr.get_task(tid)
                if row["status"] != "failed":
                    break
            assert row["status"] == want_status and want_in in row["summary"], row
        assert "PChome 找到 3 款" in (await tr.get_task("f1"))["summary"]
    finally:
        main.tracker, main.escalation_agent = old
        await tr.close()
    print("  ✓ a fix that fails keeps the answer that was there; a partial one is shown as the answer")


async def test_process_record() -> None:
    import process_log as P
    import privacy
    from tracker import Tracker
    from models import TaskRequest
    tr = Tracker(os.path.join(tempfile.mkdtemp(prefix="ot-proc-"), "t.db"))
    await tr.open()
    try:
        await tr.create_task(TaskRequest(id="p1", goal="iPhone 18 價格", session_id="s"))
        evs = [
            {"type": "chat.stream.delta", "task_id": "p1", "text": "I'll read "},
            {"type": "chat.stream.delta", "task_id": "p1", "text": "each site."},
            {"type": "chat.stream.end", "task_id": "p1"},
            {"event": "tool_call", "task_id": "p1", "round": 0, "call_idx": 0, "tool": "browser_read",
             "args": {"tab": 1, "find": "x" * 900}},
            {"event": "tool_result", "task_id": "p1", "round": 0, "call_idx": 0, "tool": "browser_read",
             "success": True, "output": "PChome page"},
            {"type": "chat.stream.delta", "task_id": "p1", "text": TABLE},
            {"event": "tool_call", "task_id": "p1", "round": 1, "call_idx": 0, "tool": "browser_read", "args": {}},
            {"event": "tool_result", "task_id": "p1", "round": 1, "call_idx": 0, "tool": "browser_read",
             "success": False, "error": "Loop detected — refusing to re-run"},
            {"event": "tool_call", "task_id": "p1", "round": 2, "call_idx": 0, "tool": "browser_click", "args": {}},
            {"type": "task.done", "task_id": "p1", "status": "failed"},
        ]
        for ev in evs:
            await P.listener(tr, ev)
        items = await tr.get_task_process("p1")
        assert [i["k"] for i in items] == ["note", "step", "note", "step", "step"], items
        assert items[0]["text"] == "I'll read each site." and items[2]["text"] == TABLE
        assert items[1]["ok"] and items[1]["out"] == "PChome page" and len(items[1]["args"]["find"]) < 320
        assert items[3]["ok"] is False and "Loop detected" in items[3]["out"] and "ok" not in items[4]
        listed = (await tr.list_tasks(session_id="s"))[0]
        assert listed["has_process"] and "process" not in listed

        for ev in ({"event": "tool_call", "task_id": "p1", "round": 100, "call_idx": 0, "tool": "browser_read",
                    "args": {}, "source": "claude-fix"},
                   {"type": "process.note", "task_id": "p1", "text": "momo: 2 more", "source": "claude-fix"},
                   {"event": "claude_fix_done", "task_id": "p1", "status": "completed"}):
            await P.listener(tr, ev)
        items = await tr.get_task_process("p1")
        assert [i["k"] for i in items][5:] == ["phase", "step", "note"] and items[5]["text"] == "claude-fix"
        print("  ✓ notes and steps are saved with the task in order; a Claude Fix is appended after a marker")

        import main
        old = getattr(main, "tracker", None)
        main.tracker = tr
        try:
            assert len((await main.get_task_process("p1"))["items"]) == 8
            await tr.create_task(TaskRequest(id="p2", goal="我的行事曆", session_id="s"))
            privacy.begin("p2", "tg:alice")
            privacy.mark()
            await P.listener(tr, {"type": "process.note", "task_id": "p2", "text": "週三 10:00 看牙醫"})
            await P.listener(tr, {"type": "task.done", "task_id": "p2"})
            assert await tr.get_task_process("p2") == [] and (await main.get_task_process("p2"))["items"] == []
        finally:
            main.tracker = old
        for i in range(P._MAX_TASKS + 5):
            P.record({"type": "process.note", "task_id": f"leak{i}", "text": "x"})
        assert len(P._live) <= P._MAX_TASKS
        print("  ✓ a private task keeps no record; tasks that never end don't pile up")
    finally:
        await tr.close()


async def main_() -> None:
    print("A. executor keeps its notes"); await test_notes()
    print("B. summary input"); await test_step_log()
    print("C. file check"); await test_file_check()
    print("D. Claude Fix"); await test_claude_fix()
    print("E. process record"); await test_process_record()
    print("all passed")


if __name__ == "__main__":
    asyncio.run(main_())
