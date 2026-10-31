"""
How much prompt re-reading the KV cache saves — old prompt layout vs new.

Sends OpenTeddy's real executor prompt (system text + every tool schema)
to Ollama the way a task does, in two layouts:

  old  the clock (to the second) opens the system prompt; discovery memos
       are appended to the system prompt, i.e. in front of the tools
  new  fixed system prompt; the clock is in the first user message; memos
       are a note after the conversation

and reports, for the second call of each pair, how many prompt tokens
Ollama had to evaluate and how long that took (prefill). One answer
token per call, so it takes seconds, not minutes.

    .venv/bin/python scripts/prompt_cache_bench.py --url http://192.168.50.86:11434 --model qwen3.8:27b

Use the runtime's num_ctx (default 16384): a different one makes Ollama
reload the model.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)


def _call(url: str, model: str, system: str, messages: list, tools: list, num_ctx: int) -> dict:
    body = {
        "model": model, "stream": False, "keep_alive": "24h", "think": False,
        "messages": [{"role": "system", "content": system}] + messages,
        "tools": tools,
        "options": {"num_predict": 1, "temperature": 0, "num_ctx": num_ctx},
    }
    t0 = time.monotonic()
    r = httpx.post(f"{url}/api/chat", json=body, timeout=600)
    r.raise_for_status()
    d = r.json()
    d["_wall_s"] = time.monotonic() - t0
    return d


def _row(label: str, d: dict) -> str:
    return (f"  {label:<44} prefill {d.get('prompt_eval_count', 0):>6} tok "
            f"{(d.get('prompt_eval_duration') or 0) / 1e9:>6.2f} s · load "
            f"{(d.get('load_duration') or 0) / 1e9:>5.2f} s · wall {d['_wall_s']:>6.2f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--num-ctx", type=int, default=16384)
    a = ap.parse_args()

    from executor import _system_prompt_for_mode
    from orchestrator import _clock_note, _current_time_header
    from tool_registry import tool_registry
    tool_registry.auto_register_all()
    tools = tool_registry.get_schemas()
    static = _system_prompt_for_mode("code", a.model)
    task_a = "Task: 列出 workspace 裡最大的 5 個檔案\n\nContext: {}"
    task_b = "Task: 查一下這台機器的磁碟剩多少空間\n\nContext: {}"
    memo = ("[Discovery memo — what you ALREADY learned in this subtask. Do NOT call these tools "
            "again on the same file/path; use the info below directly:]\n  • list_directory(.) → 14 entries")
    round1 = [{"role": "assistant", "content": "", "tool_calls": [
                  {"function": {"name": "list_directory", "arguments": {"path": "."}}}]},
              {"role": "tool", "content": json.dumps({"success": True, "result": ["a.csv", "b.pdf", "src/"]})}]
    print(f"{a.model} @ {a.url} · {len(tools)} tool schemas · system {len(static)} chars · num_ctx {a.num_ctx}\n")

    warm = _call(a.url, a.model, static, [{"role": "user", "content": "hi"}], tools, a.num_ctx)
    print(_row("warm-up (loads the model if needed)", warm))

    print("\nA new subtask (different task, a few seconds later):")
    old_hdr = lambda: _current_time_header()
    _call(a.url, a.model, old_hdr() + static, [{"role": "user", "content": task_a}], tools, a.num_ctx)
    time.sleep(1.1)                         # the old header changes every second
    old = _call(a.url, a.model, old_hdr() + static, [{"role": "user", "content": task_b}], tools, a.num_ctx)
    print(_row("old: clock at top of system prompt", old))
    _call(a.url, a.model, static, [{"role": "user", "content": _clock_note() + task_a}], tools, a.num_ctx)
    new = _call(a.url, a.model, static, [{"role": "user", "content": _clock_note() + task_b}], tools, a.num_ctx)
    print(_row("new: fixed system prompt, clock in message", new))

    print("\nThe next tool round, after a discovery memo was added:")
    hdr = old_hdr()
    _call(a.url, a.model, hdr + static, [{"role": "user", "content": task_a}], tools, a.num_ctx)
    old_r = _call(a.url, a.model, hdr + static + "\n\n" + memo,
                  [{"role": "user", "content": task_a}] + round1, tools, a.num_ctx)
    print(_row("old: memo appended to system prompt", old_r))
    first = [{"role": "user", "content": _clock_note() + task_a}]
    _call(a.url, a.model, static, first, tools, a.num_ctx)
    new_r = _call(a.url, a.model, static, first + round1 + [{"role": "user", "content": memo}], tools, a.num_ctx)
    print(_row("new: memo after the conversation", new_r))

    s = lambda d: (d.get("prompt_eval_duration") or 0) / 1e9
    print(f"\nprefill per new subtask: {s(old):.2f} s → {s(new):.2f} s; "
          f"per memo round: {s(old_r):.2f} s → {s(new_r):.2f} s")


if __name__ == "__main__":
    main()
