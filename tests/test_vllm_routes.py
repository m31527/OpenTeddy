"""
Each local model goes to the engine that serves it.

One vLLM server serves one model. To keep the gemma planner / qwen
executor split on vLLM — or to put only the executor on vLLM — calls are
routed per model: `vllm_models` = "qwen3.6:35b=http://127.0.0.1:8001, …".

  A. routes: parsed from the setting; a listed model goes to its server,
     the rest stay on Ollama; local_engine=vllm alone keeps the planner on
     Ollama (as before); nothing is routed on macOS
  B. payloads: OpenAI shape for a routed model, with thinking off through
     the chat template and JSON mode spelled per engine; replies are read
     by their shape
  C. the planner's calls (fast model, gate) follow gemma's route
  D. local only: no cloud call from a local-only session — not a step,
     not a summary, not a skill

    .venv/bin/python tests/test_vllm_routes.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import platform
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

platform.system = lambda: "Linux"          # routes are off on macOS; test the Linux box

from config import config, set_session_local_only
import local_engine as LE

QWEN, GEMMA = "qwen3.6:35b", "gemma4:e4b"


def _setup(routes: str, engine: str = "ollama") -> None:
    config.vllm_models, config.local_engine = routes, engine
    config.qwen_model, config.gemma_model = QWEN, GEMMA
    config.qwen_base_url = config.gemma_base_url = "http://127.0.0.1:11434"
    config.vllm_base_url = "http://127.0.0.1:8001"


def test_routes() -> None:
    _setup(f" {QWEN}=http://127.0.0.1:8001/ ,\n{GEMMA} = http://127.0.0.1:8002, junk, =x")
    assert LE.vllm_routes() == {QWEN: "http://127.0.0.1:8001", GEMMA: "http://127.0.0.1:8002"}
    assert LE.chat_endpoint(QWEN) == "http://127.0.0.1:8001/v1/chat/completions"
    assert LE.chat_endpoint(GEMMA) == "http://127.0.0.1:8002/v1/chat/completions"
    assert LE.is_vllm("qwen3-vl:8b-instruct") is False
    assert LE.chat_endpoint("qwen3-vl:8b-instruct") == "http://127.0.0.1:11434/api/chat"

    _setup(f"{QWEN}=http://127.0.0.1:8001")          # executor only (group B)
    assert LE.routed(QWEN) and not LE.routed(GEMMA)
    assert LE.usage_provider_label(QWEN) == "vllm" and LE.usage_provider_label(GEMMA) == "ollama"
    assert LE.supports_streaming(GEMMA) and not LE.supports_streaming(QWEN)

    _setup("", engine="vllm")                          # the old global switch
    assert LE.is_vllm(QWEN) and not LE.routed(GEMMA), "planner stays on Ollama as before"
    assert LE.chat_endpoint() == "http://127.0.0.1:8001/v1/chat/completions"

    _setup("")
    assert not LE.is_vllm(QWEN) and LE.chat_endpoint(QWEN) == "http://127.0.0.1:11434/api/chat"
    platform.system = lambda: "Darwin"
    _setup(f"{QWEN}=http://127.0.0.1:8001")
    assert LE.vllm_routes() == {} and not LE.is_vllm(QWEN)
    platform.system = lambda: "Linux"
    print("  ✓ a listed model goes to its own vLLM server; the rest stay on Ollama; none on macOS")


def test_payloads() -> None:
    _setup(f"{QWEN}=http://127.0.0.1:8001")
    p = LE.build_payload(model=QWEN, messages=[{"role": "user", "content": "hi"}], system="sys",
                         tools=None, stream=False, temperature=0.2, num_predict=50, num_ctx=32768,
                         json_mode=True)
    assert p["messages"][0] == {"role": "system", "content": "sys"} and "options" not in p
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert p["response_format"] == {"type": "json_object"} and p["max_tokens"] == 50
    o = LE.build_payload(model=GEMMA, messages=[], system=None, tools=None, stream=False,
                         temperature=0.1, num_predict=10, json_mode=True)
    assert o["format"] == "json" and o["think"] is False and "chat_template_kwargs" not in o
    openai = {"choices": [{"message": {"content": "ok", "tool_calls": []}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 7, "completion_tokens": 2}}
    n = LE.normalize_response(openai)
    assert n["message"]["content"] == "ok" and n["prompt_eval_count"] == 7
    ollama = {"message": {"content": "x"}, "eval_count": 3}
    assert LE.normalize_response(ollama) is ollama
    print("  ✓ OpenAI shape for a routed model (thinking off, JSON mode); replies read by their shape")


class _Http:
    def __init__(self, reply): self.reply, self.calls = reply, []
    async def post(self, url, json=None, **k):
        self.calls.append((url, json))
        reply = self.reply
        class _R:
            def raise_for_status(self): pass
            def json(self): return reply
        return _R()


class _Tr:
    async def record_usage(self, **k): self.last = k


async def test_planner_calls() -> None:
    import fast_model
    import gate
    _setup(f"{QWEN}=http://127.0.0.1:8001,{GEMMA}=http://127.0.0.1:8002")
    reply = {"choices": [{"message": {"content": '{"ok": true}', "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "start_task", "arguments": "{\"what\": \"做報表\"}"}}]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
    orch = types.SimpleNamespace(_http=_Http(reply), tracker=_Tr())
    out = await fast_model.ask(orch, "sys", "q", json_mode=True)
    url, body = orch._http.calls[-1]
    assert url == "http://127.0.0.1:8002/v1/chat/completions" and body["response_format"]
    assert out == '{"ok": true}' and orch.tracker.last["model_provider"] == "vllm"
    text, what = await gate.ask(orch, "幫我做報表")
    url, body = orch._http.calls[-1]
    assert url.startswith("http://127.0.0.1:8002") and body["tools"][0]["function"]["name"] == "start_task"
    assert what == "做報表", what

    _setup(f"{QWEN}=http://127.0.0.1:8001")          # gemma not routed → Ollama as before
    orch = types.SimpleNamespace(_http=_Http({"message": {"content": "x"}, "eval_count": 1}), tracker=_Tr())
    await fast_model.ask(orch, "sys", "q")
    assert orch._http.calls[-1][0] == "http://127.0.0.1:11434/api/chat"
    print("  ✓ the planner's quick calls and the gate follow gemma's route (vLLM or Ollama)")


async def test_local_only() -> None:
    from escalation import EscalationAgent
    from skill_factory import SkillFactory
    config.escalation_enabled = True

    class _Prov:
        provider_name, model_name = "fake", "f"
        def is_configured(self): return True
        async def complete_text(self, **k): raise AssertionError("cloud called")
    agent = EscalationAgent(types.SimpleNamespace(), provider=_Prov())
    set_session_local_only(True)
    try:
        assert "local only" in (agent._escalation_blocked() or "")
        s = await agent.synthesize_summary("goal", ["a"], task_id="t")
        assert "a" in s, s
        sf = SkillFactory.__new__(SkillFactory)
        try:
            await sf.generate_skill("csv_top_products", "top products", auto=True)
            raise AssertionError("created")
        except ValueError as exc:
            assert "stays local" in str(exc)
    finally:
        set_session_local_only(False)
    assert agent._escalation_blocked() is None
    print("  ✓ a local-only session calls no cloud model: no takeover, no summary, no skill")


async def main() -> None:
    print("A. routes"); test_routes()
    print("B. payloads"); test_payloads()
    print("C. planner"); await test_planner_calls()
    print("D. local only"); await test_local_only()
    print("\nALL VLLM ROUTE TESTS PASS")


asyncio.run(main())
