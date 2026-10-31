"""
System prompts are fixed text, so Ollama can reuse its KV cache.

The clock used to open every system prompt (to the second) and discovery
memos were appended to the system prompt — in front of the tool schemas
— so every call re-read ~12k tokens: 14.1 s → 1.5 s per new subtask on
the home GX10 once moved (scripts/prompt_cache_bench.py).

  A. executor / planner system prompts: identical across calls, no date
  B. the clock note: date, weekday, minutes — for the user message
  C. executor loop: the system prompt of round 2 is byte-identical to
     round 1, the clock is in the first user message, and a discovery
     memo comes after the conversation (not stored in the history)
  D. per-call timing: load / prefill / decode split and logged

    .venv/bin/python tests/test_prompt_cache.py
"""
from __future__ import annotations

import asyncio, logging, os, sys, time
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from config import config
import executor as EX
import orchestrator as OR
import local_engine


def test_static_prompts() -> None:
    today = datetime.now().strftime("%Y-%m-%d")
    for mode in ("chat", "code"):
        for model in ("qwen3.5:2b", "qwen3.8:27b", "qwen3.6:35b"):
            a = EX._system_prompt_for_mode(mode, model)
            time.sleep(0.01)
            assert a == EX._system_prompt_for_mode(mode, model) and today not in a, (mode, model)
            assert "Current system clock" not in a
    for mode in ("chat", "code", "analytic"):
        for model in ("gemma4:e4b", "gemma4:26b"):
            p = OR._plan_prompt_for_mode(mode, model)
            assert p == OR._plan_prompt_for_mode(mode, model) and today not in p, (mode, model)
    print("  ✓ executor and planner system prompts are fixed text (no clock)")


def test_clock_note() -> None:
    now = datetime.now().astimezone()
    n = OR._clock_note()
    assert n.startswith(f"[Now: {now:%Y-%m-%d} ") and now.strftime("%A") in n and "農曆" in n
    assert ":" in n.split()[2] and len(n.split()[2]) == 5, "to the minute, not the second"
    assert "週" in n and "UTC" in n
    print("  ✓ clock note: date, weekday, HH:MM, timezone — for the user message")


class _Resp:
    def __init__(self, data): self._d = data
    def raise_for_status(self): pass
    def json(self): return self._d


class _Http:
    def __init__(self, answers): self.answers, self.payloads = list(answers), []
    async def post(self, url, json=None, **k):
        self.payloads.append(json)
        return _Resp(self.answers.pop(0))
    async def aclose(self): pass


class _Registry:
    def get_schemas(self):
        return [{"type": "function", "function": {"name": "list_directory", "description": "ls",
                                                   "parameters": {"type": "object", "properties": {}}}}]
    def get_schemas_by_names(self, names): return self.get_schemas()
    def risk_of(self, name): return "low"
    async def execute(self, name, args, **k):
        return {"success": True, "result": {"path": ".", "entries": ["a.csv", "b.pdf"]}, "error": None}


class _Tracker:
    async def record_usage(self, **k): self.last = k
    async def append_task_artifact(self, *a, **k): pass


def _ollama(content="", tool_calls=None):
    return {"message": {"content": content, "tool_calls": tool_calls or []},
            "prompt_eval_count": 12000, "eval_count": 40, "load_duration": 15_000_000,
            "prompt_eval_duration": 1_530_000_000, "eval_duration": 4_000_000_000,
            "total_duration": 5_600_000_000, "done_reason": "stop"}


async def test_executor_loop() -> None:
    config.streaming_enabled = False
    config.local_engine = "ollama"
    ex = EX.Executor(tracker=_Tracker(), skill_factory=None, registry=_Registry())
    await ex._http.aclose()
    ex._http = _Http([
        _ollama(tool_calls=[{"function": {"name": "list_directory", "arguments": {"path": "."}}}]),
        _ollama(content='{"result": "a.csv, b.pdf", "confidence": 0.9}'),
    ])
    await ex._qwen_execute("列出 workspace 的檔案", {}, task_id="t1", subtask_id="s1", mode="code")
    p1, p2 = ex._http.payloads
    sys1 = [m for m in p1["messages"] if m["role"] == "system"]
    sys2 = [m for m in p2["messages"] if m["role"] == "system"]
    assert sys1 == sys2 and sys1, "round 2 reuses round 1's system prompt byte for byte"
    first = [m for m in p1["messages"] if m["role"] == "user"][0]["content"]
    assert first.startswith("[Now: ") and "列出 workspace 的檔案" in first
    assert "Discovery memo" not in sys2[0]["content"]
    last = p2["messages"][-1]
    assert last["role"] == "user" and last["content"].startswith("[Discovery memo"), last
    assert p2["messages"][:len(p1["messages"])] == p1["messages"], "round 1's messages are a prefix of round 2's"
    assert p1["options"]["num_ctx"] == p2["options"]["num_ctx"] == int(config.qwen_num_ctx)
    print("  ✓ executor: same system prompt every round, clock in the task message, memo after the history")
    assert ex.tracker.last["prefill_ms"] == 1530 and ex.tracker.last["load_ms"] == 15
    print("  ✓ usage records carry load and prefill time")


def test_timing() -> None:
    t = local_engine.timings(_ollama())
    assert t == {"load_ms": 15, "prefill_ms": 1530, "decode_ms": 4000, "total_ms": 5600}
    assert local_engine.timings({}) == {"load_ms": 0, "prefill_ms": 0, "decode_ms": 0, "total_ms": 0}
    print("  ✓ timings: load / prefill / decode / total in ms")


async def main() -> None:
    print("A. static prompts"); test_static_prompts()
    print("B. clock"); test_clock_note()
    print("C. executor loop"); await test_executor_loop()
    print("D. timing"); test_timing()
    print("\nALL PROMPT CACHE TESTS PASS")


asyncio.run(main())
