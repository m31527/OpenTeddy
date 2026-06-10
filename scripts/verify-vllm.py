#!/usr/bin/env python3
"""
Standalone verification that OpenTeddy's local_engine abstraction can
talk to a running vLLM server — without booting the whole orchestrator.

Run on the DGX after starting a vLLM server, with the engine env vars set:

    export OPENTEDDY_LOCAL_ENGINE=vllm
    export VLLM_BASE_URL=http://127.0.0.1:8001
    export QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
    .venv/bin/python scripts/verify-vllm.py

It exercises three things, each the way the real executor does:
  1. A plain chat completion  → checks build_payload + normalize_response.
  2. A tool-use turn          → checks tool_calls survive the round-trip
                                 (this is the bit that needs vLLM's
                                 --enable-auto-tool-choice --tool-call-parser).
  3. A rough tokens/sec number → so you can eyeball vLLM vs Ollama speed.

Prints a clear PASS/FAIL per step. Exit code 0 = all good.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

sys.path.insert(0, ".")

import httpx
import local_engine as le
from config import config


GREEN = "\033[32m"; RED = "\033[31m"; DIM = "\033[2m"; OFF = "\033[0m"
def ok(m):   print(f"  {GREEN}✓{OFF} {m}")
def bad(m):  print(f"  {RED}✗{OFF} {m}")
def info(m): print(f"  {DIM}{m}{OFF}")


async def main() -> int:
    engine = le.active_engine()
    print(f"\nActive engine : {engine}")
    print(f"Endpoint      : {le.chat_endpoint()}")
    print(f"Model         : {config.qwen_model}")
    print(f"Streaming     : {le.supports_streaming()}\n")

    if engine != "vllm":
        bad(f"active_engine is '{engine}', not 'vllm'. Set "
            "OPENTEDDY_LOCAL_ENGINE=vllm (and you must be on a non-Mac host).")
        return 1

    http = httpx.AsyncClient(timeout=120)
    failures = 0

    # ── 1. Plain chat ────────────────────────────────────────────────────────
    print("[1/3] Plain chat completion")
    try:
        payload = le.build_payload(
            model=config.qwen_model,
            messages=[{"role": "user", "content": "Reply with exactly: PONG"}],
            system="You are a terse test bot.",
            tools=None, stream=False, temperature=0.0, num_predict=16,
        )
        t0 = time.monotonic()
        r = await http.post(le.chat_endpoint(), json=payload)
        r.raise_for_status()
        data = le.normalize_response(r.json())
        dt = time.monotonic() - t0
        content = (data.get("message") or {}).get("content", "")
        toks = data.get("eval_count", 0)
        if content.strip():
            ok(f"got reply: {content.strip()[:60]!r}")
            info(f"{toks} output tokens in {dt:.2f}s "
                 f"(~{toks/dt:.1f} tok/s wall-clock)" if toks else f"{dt:.2f}s")
        else:
            bad("empty content — normalize_response mapping may be off")
            info(f"raw keys: {sorted(r.json().keys())}")
            failures += 1
    except Exception as exc:
        bad(f"chat call failed: {exc}")
        info("Is the vLLM server up? curl http://127.0.0.1:8001/v1/models")
        failures += 1

    # ── 2. Tool-use round-trip ───────────────────────────────────────────────
    print("\n[2/3] Tool-use turn (needs --enable-auto-tool-choice)")
    try:
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }]
        payload = le.build_payload(
            model=config.qwen_model,
            messages=[{"role": "user",
                       "content": "What's the weather in Taipei? Use the tool."}],
            system="You are a helpful assistant. Use tools when relevant.",
            tools=tools, stream=False, temperature=0.0, num_predict=128,
        )
        r = await http.post(le.chat_endpoint(), json=payload)
        r.raise_for_status()
        data = le.normalize_response(r.json())
        tcs = (data.get("message") or {}).get("tool_calls") or []
        if tcs:
            fn = (tcs[0].get("function") or {})
            ok(f"model emitted tool_call: {fn.get('name')}({fn.get('arguments')})")
        else:
            bad("no tool_calls returned — vLLM likely started WITHOUT "
                "--enable-auto-tool-choice --tool-call-parser hermes")
            info(f"message.content was: "
                 f"{(data.get('message') or {}).get('content','')[:80]!r}")
            failures += 1
    except Exception as exc:
        bad(f"tool-use call failed: {exc}")
        failures += 1

    # ── 3. Throughput sample ─────────────────────────────────────────────────
    print("\n[3/3] Throughput sample (longer generation)")
    try:
        payload = le.build_payload(
            model=config.qwen_model,
            messages=[{"role": "user",
                       "content": "Write a 150-word paragraph about mold prevention."}],
            system=None, tools=None, stream=False, temperature=0.7, num_predict=256,
        )
        t0 = time.monotonic()
        r = await http.post(le.chat_endpoint(), json=payload)
        r.raise_for_status()
        data = le.normalize_response(r.json())
        dt = time.monotonic() - t0
        toks = data.get("eval_count", 0)
        if toks:
            ok(f"{toks} tokens in {dt:.2f}s → ~{toks/dt:.1f} tok/s")
            info("Compare against your Ollama tok/s for the same model size "
                 "(Settings → Model Settings shows Ollama's number).")
        else:
            info(f"completed in {dt:.2f}s (no token count returned)")
    except Exception as exc:
        bad(f"throughput call failed: {exc}")
        failures += 1

    await http.aclose()

    print()
    if failures == 0:
        print(f"{GREEN}ALL PASS{OFF} — OpenTeddy's local_engine talks to vLLM "
              "correctly. Safe to run the full backend with "
              "OPENTEDDY_LOCAL_ENGINE=vllm.\n")
        return 0
    print(f"{RED}{failures} step(s) failed{OFF} — see hints above.\n")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
