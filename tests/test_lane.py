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
    chat = ["今天是星期幾", "你好", "早安！", "現在幾點？", "你是誰", "謝謝",
            "what day is it?", "hi", "are you there?", "為什麼會這樣？"]
    work = ["幫我查今天的訂單", "分析 sales.csv 產出報告", "每天早上 8 點查昨日營收",
            "部署最新版 OpenTeddy", "寫一個 CSV 轉 JSON 的 CLI", "請你整理一下客訴",
            "deploy the app", "generate a chart from ~/data.csv", "run the tests",
            "https://example.com 這頁在講什麼", "查一下上個月北區退貨率？",
            "今天是星期幾" * 12]                      # 72 chars > MAX_CHAT_CHARS → work
    for q in chat:
        assert lane.looks_conversational(q), f"should be chat: {q!r}"
    for q in work:
        assert not lane.looks_conversational(q), f"should be work: {q!r}"
    assert not lane.looks_conversational("你好", has_attachments=True)
    print("  ✓ conversational detector: %d chat / %d work cases, ties go to work" % (len(chat), len(work)))

    async def run():
        assert await lane.pick_mode("今天是星期幾", "code") == "chat"
        assert await lane.pick_mode("幫我查訂單", "analytic") == "analytic"
        assert await lane.pick_mode("早安", "chat") == "chat"
    asyncio.run(run())
    print("  ✓ pick_mode: chat for conversation, session mode otherwise")


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


if __name__ == "__main__":
    test_lane()
    test_think()
    test_fast_chat_sentinel()
    print("\nALL LANE TESTS PASS")
