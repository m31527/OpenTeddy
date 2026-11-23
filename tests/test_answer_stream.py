"""
answer_stream.py and the gate's streamed call (gate._streamed).

  A. listen / push / end / close: only a listened-to task streams.
  B. Ollama NDJSON: each piece pushed as it comes, END after; the counts and
     the whole text come back as a plain call's would.
  C. vLLM SSE: content deltas pushed; a tool call assembled from its
     fragments and nothing pushed (the hand-off).
  D. gate.ask streams only when someone listens; the answer is the same.

    .venv/bin/python tests/test_answer_stream.py
"""
from __future__ import annotations

import asyncio, json, logging, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import httpx

from config import config
import answer_stream
import gate


def run(coro):
    return asyncio.run(coro)


def drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_registry():
    assert not answer_stream.wanted("t1")
    q = answer_stream.listen("t1")
    assert answer_stream.wanted("t1")
    answer_stream.push("t1", "你")
    answer_stream.push("t1", "")
    answer_stream.push("t2", "別人的")
    answer_stream.end("t1")
    assert drain(q) == ["你", answer_stream.END]
    answer_stream.close("t1", q)
    assert not answer_stream.wanted("t1")
    print("  ✓ listen → pieces then END; nothing for a task nobody listens to; close forgets it")


def orch_with(handler):
    return types.SimpleNamespace(_http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def test_ollama():
    lines = [{"message": {"content": "退貨"}}, {"message": {"content": "期限是"}},
             {"message": {"content": "七天。"}},
             {"message": {"content": ""}, "done": True, "prompt_eval_count": 2100, "eval_count": 9,
              "eval_duration": 90_000_000, "total_duration": 300_000_000}]
    body = "\n".join(json.dumps(x, ensure_ascii=False) for x in lines)
    seen = {}

    def handler(req):
        seen["payload"] = json.loads(req.content)
        return httpx.Response(200, text=body)
    q = answer_stream.listen("t-ol")
    data = run(gate._streamed(orch_with(handler), "http://x/api/chat", {"stream": True}, False, "t-ol"))
    assert drain(q) == ["退貨", "期限是", "七天。"]
    assert data["message"]["content"] == "退貨期限是七天。" and not data["message"]["tool_calls"]
    assert data["eval_count"] == 9 and data["prompt_eval_count"] == 2100
    answer_stream.close("t-ol", q)

    tool = "\n".join(json.dumps(x) for x in [
        {"message": {"content": "", "tool_calls": [{"function": {"name": "start_task", "arguments": {"what": "查天氣"}}}]}},
        {"message": {"content": ""}, "done": True}])
    q = answer_stream.listen("t-ol2")
    data = run(gate._streamed(orch_with(lambda r: httpx.Response(200, text=tool)), "http://x/v1/chat/completions", {}, False, "t-ol2"))
    assert drain(q) == [] and gate._task_from(data["message"]["tool_calls"]) == "查天氣"
    answer_stream.close("t-ol2", q)
    print("  ✓ Ollama: pieces pushed as they come, counts kept; a start_task call pushes nothing")


def test_vllm():
    def sse(objs):
        return "".join("data: " + json.dumps(o, ensure_ascii=False) + "\n\n" for o in objs) + "data: [DONE]\n\n"
    text = sse([{"choices": [{"delta": {"role": "assistant"}}]},
                {"choices": [{"delta": {"content": "可以，"}}]},
                {"choices": [{"delta": {"content": "七天內。"}}]},
                {"choices": [], "usage": {"prompt_tokens": 2000, "completion_tokens": 6}}])
    q = answer_stream.listen("t-v")
    data = run(gate._streamed(orch_with(lambda r: httpx.Response(200, text=text)), "http://x/v1/chat/completions", {}, True, "t-v"))
    assert drain(q) == ["可以，", "七天內。"] and data["message"]["content"] == "可以，七天內。"
    assert data["eval_count"] == 6 and data["prompt_eval_count"] == 2000
    answer_stream.close("t-v", q)

    tool = sse([{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "start_task", "arguments": ""}}]}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"what\": "}}]}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"訂位\"}"}}]}}]}])
    q = answer_stream.listen("t-v2")
    data = run(gate._streamed(orch_with(lambda r: httpx.Response(200, text=tool)), "http://x/v1/chat/completions", {}, True, "t-v2"))
    assert drain(q) == [] and gate._task_from(data["message"]["tool_calls"]) == "訂位"
    answer_stream.close("t-v2", q)
    print("  ✓ vLLM: content deltas pushed, usage kept; a tool call assembled from its fragments, nothing pushed")


def test_ask():
    config.llm_mode, config.orchestrator_backend = "local", "local"
    config.gemma_model, config.gemma_base_url = "gemma4:e4b", "http://ollama"
    import local_engine
    local_engine.vllm_routes = lambda: {}
    calls = []

    def handler(req):
        p = json.loads(req.content)
        calls.append(p["stream"])
        if p["stream"]:
            return httpx.Response(200, text="\n".join(json.dumps(x, ensure_ascii=False) for x in [
                {"message": {"content": "你好！"}}, {"message": {"content": ""}, "done": True}]))
        return httpx.Response(200, json={"message": {"content": "你好！"}, "done": True})

    async def record_usage(**kw):
        pass
    orch = orch_with(handler)
    orch.tracker = types.SimpleNamespace(record_usage=record_usage)
    gate._system_cache["s"] = "system"
    assert run(gate.ask(orch, "hi", task_id="nobody")) == ("你好！", None)
    q = answer_stream.listen("someone")
    assert run(gate.ask(orch, "hi", task_id="someone")) == ("你好！", None)
    assert calls == [False, True] and drain(q) == ["你好！", answer_stream.END]
    answer_stream.close("someone", q)
    print("  ✓ gate.ask streams only when someone listens; same answer either way; END after the text")


if __name__ == "__main__":
    print("A. registry"); test_registry()
    print("B. Ollama"); test_ollama()
    print("C. vLLM"); test_vllm()
    print("D. gate.ask"); test_ask()
    print("\nALL ANSWER STREAM TESTS PASS")
