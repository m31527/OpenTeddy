"""
A step the cloud model takes over is done with tools and checked like a
local one.

In the 2026-11-14 eval, 7 of qwen3.6:35b's 13 misses were cloud takeovers
with no tools: "invoices.json written" with no file, code for nobody to
run, "revenue 1,234,567.89" made up, all accepted as done.

  A. numbers: an answer's numbers have to appear in what the tools returned;
     DeepSeek's reasoning is passed back, or its tool loop dies on turn 2
  B. tools: in code / analytic mode the cloud model gets the workspace tools
     (within the agent's scope, never the person's browser or Google) and
     really runs them; chat mode stays text
  C. a made-up number is sent back once; one still unbacked is marked
  D. out of turns: not done, and what it found is kept
  E. the orchestrator: a cloud "done" that had to write a file and didn't
     is not done; files the cloud wrote get file cards
  F. executor: a step whose plan names a tool, answered without any tool,
     is sent back once

    .venv/bin/python tests/test_cloud_step.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from config import config
from escalation import EscalationAgent, ungrounded_numbers
from llm_provider import LLMTextResponse, LLMToolTurnResponse, LLMToolUse, LLMUsage
from models import SubTask, TaskStatus


def test_numbers() -> None:
    src = ["region,revenue\n北區,102470\n中區,29980", "合計 132450"]
    assert ungrounded_numbers("北區 102,470 元最高，合計 132,450 元", src) == []
    assert ungrounded_numbers("總營收 1,234,567.89，共 452 筆", src) == ["1234567.89", "452"]
    assert ungrounded_numbers("1280.0 和 1280 一樣，3 筆不算", ["1280"]) == []
    assert ungrounded_numbers("total 4000", []) == ["4000"]
    assert ungrounded_numbers("你要的 2025 年 amount 總和無法計算", ["讀 report_2025.csv"]) == []
    print("  ✓ an answer's numbers must appear in the tool results (73,440 = 73440; one digit ignored)")


def test_deepseek_reasoning() -> None:
    from llm_provider import DeepseekProvider, _OpenAICompatHistory
    call = [{"id": "c1", "function": {"name": "read_file", "arguments": "{\"path\": \"a\"}"}}]
    h = _OpenAICompatHistory()
    # Its reasoning comes back with the calls, even empty, and the text it
    # wrote alongside stays in the same message (two messages → 400).
    h.add_assistant_turn(DeepseekProvider._parse_response({"choices": [{"message": {
        "content": "先看檔案", "reasoning_content": "", "tool_calls": call}}]}))
    assert len(h.messages) == 1 and h.messages[0]["reasoning_content"] == ""
    assert h.messages[0]["content"] == "先看檔案" and h.messages[0]["tool_calls"]
    # A model without reasoning keeps the split, as before.
    h = _OpenAICompatHistory()
    h.add_assistant_turn(DeepseekProvider._parse_response({"choices": [{"message": {
        "content": "let me look", "tool_calls": call}}]}))
    assert [m["content"] for m in h.messages] == ["let me look", None]
    assert all("reasoning_content" not in m for m in h.messages)
    print("  ✓ DeepSeek's reasoning goes back with its tool calls (else 400 on the next turn)")


class _Hist:
    def __init__(self): self.msgs = []
    def add_user_message(self, t): self.msgs.append(("user", t))
    def add_assistant_turn(self, r): self.msgs.append(("assistant", r.text, [u.name for u in r.tool_uses]))
    def add_tool_results(self, rs): self.msgs.append(("tools", [r.content for r in rs]))


class _Provider:
    provider_name, model_name = "fake", "fake-1"
    def __init__(self, turns): self.turns, self.calls, self.text_calls, self.hist = list(turns), [], 0, None
    def is_configured(self): return True
    def new_history(self):
        self.hist = _Hist()
        return self.hist
    async def complete_with_tools(self, history, tools, system, max_tokens):
        self.calls.append(sorted(t["function"]["name"] for t in tools))
        t = self.turns.pop(0) if self.turns else self.turns_default
        return LLMToolTurnResponse(text=t[0], tool_uses=[LLMToolUse(id=f"u{i}", name=n, input=a)
                                                         for i, (n, a) in enumerate(t[1])],
                                   usage=LLMUsage(10, 5))
    async def complete_text(self, user_message, system=None, max_tokens=1024):
        self.text_calls += 1
        return LLMTextResponse(text="text answer", usage=LLMUsage(1, 1))


class _Reg:
    NAMES = ("read_file", "write_file", "python_exec", "list_directory", "browser_open",
             "calendar_events", "home_control", "telegram_send")
    def __init__(self, outputs=None): self.ran, self.outputs = [], outputs or {}
    def list_tools(self): return [{"name": n} for n in self.NAMES]
    def get_schemas_by_names(self, names):
        return [{"type": "function", "function": {"name": n}} for n in names if n in self.NAMES]
    async def execute(self, name, args, task_id="", allowed_tools=None, **k):
        self.ran.append((name, sorted(allowed_tools or [])))
        if allowed_tools and name not in allowed_tools:
            return {"success": False, "error": "outside scope"}
        return {"success": True, "result": {"stdout": self.outputs.get(name, "ok")}}


class _Tracker:
    def __init__(self): self.usage, self.updates = [], []
    async def get_task(self, tid): return {"goal": "sales.csv 2026 年 9 月的總營收是多少？"}
    async def update_subtask(self, st): self.updates.append(st.status)
    async def record_usage(self, **k): self.usage.append(k)


def _step(**k):
    return SubTask(parent_task_id="t1", description="用 python_exec 讀取 sales.csv 算 9 月營收", **k)


async def test_tools() -> None:
    config.escalation_enabled = True
    events = []
    async def push(ev): events.append(ev)
    prov = _Provider([("先看資料", [("python_exec", {"code": "print(73440)"})]),
                      ("2026 年 9 月總營收是 73,440 元。", [])])
    reg = _Reg({"python_exec": "73440"})
    tr = _Tracker()
    agent = EscalationAgent(tr, provider=prov, registry=reg, push_event=push)
    st = await agent.resolve(_step(result="I need to check", error="answer lacked a tool"),
                             {"allowed_tools": [], "skill_guides": "guide text"}, mode="analytic")
    assert st.status == TaskStatus.COMPLETED and st.confidence == 1.0 and "73,440" in st.result, st
    assert prov.calls[0] == ["list_directory", "python_exec", "read_file", "write_file"], prov.calls[0]
    # No declared scope → none passed on: the registry would read one as
    # leave to auto-approve unattended runs.
    assert reg.ran == [("python_exec", [])], reg.ran
    first = prov.hist.msgs[0][1]
    assert "sales.csv 2026 年 9 月" in first and "I need to check" in first and "answer lacked a tool" in first
    kinds = [(e.get("event") or e.get("type"), e.get("source")) for e in events]
    assert ("tool_call", "cloud-step") in kinds and ("tool_result", "cloud-step") in kinds
    assert ("process.note", "cloud-step") in kinds and all(e["task_id"] == "t1" for e in events)
    assert len(tr.usage) == 2 and tr.usage[0]["task_description"].startswith("[cloud-step]")

    # The agent's tool scope still holds and reaches the registry; a tool
    # the cloud wasn't offered never runs. Chat mode answers in text.
    prov = _Provider([("", [("browser_open", {"url": "x"}), ("read_file", {"path": "a"})]), ("讀到了", [])])
    reg = _Reg()
    agent = EscalationAgent(_Tracker(), provider=prov, registry=reg, push_event=push)
    await agent.resolve(_step(), {"allowed_tools": ["read_file", "browser_open"]}, mode="code")
    assert prov.calls[0] == ["read_file"], prov.calls
    assert reg.ran == [("read_file", ["browser_open", "read_file"])], reg.ran
    assert "isn't available here" in prov.hist.msgs[2][1][0], prov.hist.msgs
    prov = _Provider([])
    st = await EscalationAgent(_Tracker(), provider=prov, registry=_Reg()).resolve(_step(), {}, mode="chat")
    assert prov.text_calls == 1 and not prov.calls and st.result == "text answer"
    print("  ✓ the cloud model runs workspace tools (scope kept, no browser/Google/home); chat stays text")


async def test_made_up_numbers() -> None:
    prov = _Provider([("總營收 1,234,567.89 元，共 452 筆", []),
                      ("", [("python_exec", {"code": "..."})]),
                      ("總營收 73,440 元", [])])
    agent = EscalationAgent(_Tracker(), provider=prov, registry=_Reg({"python_exec": "73440"}))
    st = await agent.resolve(_step(), {}, mode="analytic")
    nudge = [m for m in prov.hist.msgs if m[0] == "user"][1][1]
    assert "1234567.89" in nudge and "452" in nudge, nudge
    assert st.status == TaskStatus.COMPLETED and st.result == "總營收 73,440 元" and st.confidence == 1.0

    prov = _Provider([("總營收 1,234,567 元", []), ("總營收 1,234,567 元", [])])
    st = await EscalationAgent(_Tracker(), provider=prov, registry=_Reg()).resolve(_step(), {}, mode="analytic")
    assert "沒有驗證過：1234567" in st.result and st.confidence == 0.6, st.result
    print("  ✓ a number no tool returned is sent back once; one still unbacked is marked unverified")


async def test_out_of_turns() -> None:
    prov = _Provider([])
    prov.turns_default = ("找到北區 102470", [("read_file", {"path": "x"})])
    st = await EscalationAgent(_Tracker(), provider=prov, registry=_Reg()).resolve(_step(), {}, mode="code")
    assert st.status == TaskStatus.FAILED and "輪內還沒做完" in st.error and "102470" in st.result, st
    assert len(prov.calls) == 11
    print("  ✓ out of turns: not done, and what it found is kept")


async def test_orchestrator_check() -> None:
    import orchestrator as O
    import workspace_changes

    ws = tempfile.mkdtemp()
    old = workspace_changes.snapshot
    workspace_changes.snapshot = lambda root=None: old(ws)
    old_since = workspace_changes.changed_since
    workspace_changes.changed_since = lambda before, root=None: old_since(before, ws)
    must = {"v": True}
    async def must_change(self, st): return must["v"]
    real_must = O.Orchestrator._step_must_change_files
    O.Orchestrator._step_must_change_files = must_change
    try:
        cards = []
        class _Exec:
            async def emit_files(self, st, paths): cards.append([os.path.basename(p) for p in paths])
        class _Esc:
            def __init__(self, write): self.write = write
            async def resolve(self, st, ctx, mode="code"):
                if self.write:
                    open(os.path.join(ws, "invoices.json"), "w").write("[]")
                st.status, st.confidence, st.result = TaskStatus.COMPLETED, 1.0, "invoices.json written"
                return st
        tr = _Tracker()
        o = types.SimpleNamespace(escalation=_Esc(False), executor=_Exec(), tracker=tr)
        before = workspace_changes.snapshot()
        st = await O._cloud_step(o, _step(), {}, "code", before)
        assert st.status == TaskStatus.FAILED and "不算完成" in st.error, st

        must["v"] = False          # a question: nothing to write
        st = await O._cloud_step(o, _step(), {}, "analytic", before)
        assert st.status == TaskStatus.COMPLETED

        must["v"] = True
        o.escalation = _Esc(True)
        st = await O._cloud_step(o, _step(), {}, "code", workspace_changes.snapshot())
        assert st.status == TaskStatus.COMPLETED and cards[-1] == ["invoices.json"], cards
    finally:
        workspace_changes.snapshot, workspace_changes.changed_since = old, old_since
        O.Orchestrator._step_must_change_files = real_must
    print("  ✓ a cloud 'done' that had to write a file and didn't is not done; its files get cards")


async def test_executor_named_tool() -> None:
    import executor as EX

    class _Reg2:
        def get_schemas(self):
            return [{"type": "function", "function": {"name": n, "description": "",
                                                       "parameters": {"type": "object", "properties": {}}}}
                    for n in ("python_exec", "read_file")]
        def get_schemas_by_names(self, n): return self.get_schemas()
        def risk_of(self, n): return "medium"
        async def execute(self, name, args, **k):
            return {"success": True, "result": {"stdout": "4000"}, "error": None}

    class _Http:
        def __init__(self, replies): self.payloads, self.replies = [], list(replies)
        async def post(self, url, json=None, **k):
            self.payloads.append(json)
            content, calls = self.replies.pop(0)
            data = {"message": {"content": content, "tool_calls": calls},
                    "prompt_eval_count": 10, "eval_count": 5}
            class _R:
                def raise_for_status(self): pass
                def json(self): return data
            return _R()
        async def aclose(self): pass

    class _Tr:
        async def record_usage(self, **k): pass
        async def append_task_artifact(self, *a, **k): pass

    config.streaming_enabled = False
    run = [{"function": {"name": "python_exec", "arguments": {"code": "print(4000)"}}}]
    final = '{"result": "總和是 4000", "confidence": 0.9}'
    ex = EX.Executor(tracker=_Tr(), skill_factory=None, registry=_Reg2())
    await ex._http.aclose()
    ex._http = _Http([('{"result": "需要先確認檔案是否存在", "confidence": 0.8}', []), ("", run), (final, [])])
    out = await ex._qwen_execute("用 python_exec 讀取 report_2025.csv 算 amount 總和", {},
                                 task_id="t", subtask_id="s1", mode="analytic")
    sent_back = [m["content"] for m in ex._http.payloads[1]["messages"] if m["role"] == "user"][-1]
    assert "This step says to use python_exec" in sent_back, sent_back
    assert "4000" in out[0] and len(ex._http.payloads) == 3, (out, len(ex._http.payloads))

    # A step that names no tool, or one that already ran a tool, isn't sent back.
    ex._http = _Http([(final, [])])
    await ex._qwen_execute("整理上面的結果，寫成一段摘要", {}, task_id="t", subtask_id="s2", mode="analytic")
    assert len(ex._http.payloads) == 1
    assert EX._tools_named_in("用 read_file_x 和 python_exec", _Reg2().get_schemas()) == ["python_exec"]
    print("  ✓ executor: a step whose plan names a tool, answered with no tool run, is sent back once")


async def main_() -> None:
    print("A. numbers"); test_numbers(); test_deepseek_reasoning()
    print("B. tools"); await test_tools()
    print("C. made-up numbers"); await test_made_up_numbers()
    print("D. out of turns"); await test_out_of_turns()
    print("E. orchestrator"); await test_orchestrator_check()
    print("F. executor"); await test_executor_named_tool()
    print("\nALL CLOUD STEP TESTS PASS")


asyncio.run(main_())
