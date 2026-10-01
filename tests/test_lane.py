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


if __name__ == "__main__":
    test_lane()
    test_think()
    print("\nALL LANE TESTS PASS")
