"""
OpenTeddy gate — the chat model decides whether a message is chat or work.

A message that arrives without an explicit mode (Telegram, the Task API's
auto_lane) goes first to the fast chat model with one tool, start_task.
It either answers right there (one call: "今天星期幾", "你好") or calls
start_task, and the request goes to the full agent with its tools.

This replaced lane.py's keyword rules. Each new kind of job needed a new
rule there: 「去 momo 購物官網看看 vivo x300 最便宜多少錢」 looked like a
question (多少), took the chat lane — no browser — and the person got
「請稍候，我正在執行搜尋…」 with nothing behind it (2026-11-05). The model
already knows what it can't do in a reply; what it needs is the list of
what the agent can do, and that comes from the tool registry, so a new
tool needs no new rule.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from config import config

logger = logging.getLogger(__name__)

START_TASK = {"type": "function", "function": {
    "name": "start_task",
    "description": "Hand the request to OpenTeddy's agent, which uses the tools listed in the instructions.",
    "parameters": {"type": "object",
                   "properties": {"what": {"type": "string", "description": "what needs doing, in a few words"}},
                   "required": ["what"]}}}

_SYSTEM = """You are OpenTeddy, a helpful, concise personal assistant. Reply in the person's language (Traditional Chinese for Chinese).

You see each message first. Do exactly one of these:
1. Answer it yourself, now, in this reply — when your own knowledge, this conversation and the date/time given are enough: greetings, small talk, opinions, advice, explanations, general knowledge, questions about what you can do.
2. Call start_task — when the answer needs something only a tool can give or do: anything current or live (news, weather, prices, a website, a company you don't know well), the person's browser, files, code, this machine, devices, messages, calendar or mail, reminders, schedules, printing, or the state of their tasks.

OpenTeddy's agent has these tools:
{tools}

Questions ABOUT a task are yours to answer: "how do I…?", "how does … work?", "what do you think…?", "why did you…?", "would it be better to…?", "are you able to…?", or a remark on how something went.
But a polite request is still a request: 「可以幫我…嗎」, 「可以把…給我嗎」, "can you … for me?" ask for it now → start_task. If such a request lacks details, call start_task anyway; the agent can see the files and the conversation and will ask.

Never say you are going to do something ("請稍候", "我去查", "let me check") — call start_task instead. Never invent live facts."""

_system_cache: Dict[str, str] = {}


def system_prompt() -> str:
    """Built once from the registered tools; fixed afterwards, so the
    model server keeps it as a cached prefix."""
    if "s" not in _system_cache:
        from tool_registry import tool_registry
        lines = []
        for s in tool_registry.get_schemas():
            f = s.get("function") or {}
            first = (f.get("description") or "").strip().split("\n")[0].split(". ")[0][:80]
            lines.append(f"- {f.get('name')}: {first}")
        if not lines:                     # registry not loaded yet: don't cache an empty list
            return _SYSTEM.format(tools="(web search, browser, files, code, devices, calendar, schedules)")
        _system_cache["s"] = _SYSTEM.format(tools="\n".join(lines))
    return _system_cache["s"]


def _task_from(tool_calls: Any) -> Optional[str]:
    for c in tool_calls or []:
        fn = (c or {}).get("function") or {}
        if fn.get("name") == "start_task":
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                import json
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            return str((args or {}).get("what") or "").strip() or "start_task"
    return None


async def ask(orch: Any, prompt: str, task_id: str = "") -> Tuple[str, Optional[str]]:
    """(answer, None) when the model answered; ("", what) when it called
    start_task. Raises on a failed call (the caller treats it like any
    fast-chat failure)."""
    from config import orchestrator_on_cloud
    if orchestrator_on_cloud():
        provider = orch.escalation.provider
        hist = provider.new_history()
        hist.add_user_message(prompt)
        r = await provider.complete_with_tools(hist, [START_TASK], system=system_prompt(), max_tokens=2048)
        for u in r.tool_uses or []:
            if u.name == "start_task":
                return "", str((u.input or {}).get("what") or "").strip() or "start_task"
        return (r.text or "").strip(), None

    import local_engine
    payload = {
        "model": config.gemma_model,
        "messages": [{"role": "system", "content": system_prompt()}, {"role": "user", "content": prompt}],
        "tools": [START_TASK],
        "stream": False,
        "options": {"temperature": float(getattr(config, "gemma_temperature", 0.1)),
                    "num_predict": config.gemma_max_tokens,
                    "num_ctx": int(getattr(config, "gemma_num_ctx", 16384))},
        "keep_alive": getattr(config, "ollama_keep_alive", "24h"),
    }
    th = local_engine.think_setting(config.gemma_model)
    if th is not None:
        payload["think"] = th
    t0 = time.monotonic()
    resp = await orch._http.post(f"{config.gemma_base_url}/api/chat", json=payload)
    resp.raise_for_status()
    data = resp.json()
    t = local_engine.log_timing("[gate]", config.gemma_model, data)
    try:
        out = data.get("eval_count", 0) or 0
        dur = data.get("eval_duration", 0) or 0
        await orch.tracker.record_usage(
            task_id=task_id, model=config.gemma_model, model_provider="ollama",
            tokens_in=data.get("prompt_eval_count", 0) or 0, tokens_out=out,
            task_description="[gate]", duration_ms=int((data.get("total_duration") or 0) / 1e6)
            or int((time.monotonic() - t0) * 1000),
            tokens_per_sec=(out * 1e9 / dur) if out and dur else 0.0,
            load_ms=t["load_ms"], prefill_ms=t["prefill_ms"])
    except Exception:  # noqa: BLE001
        pass
    msg = data.get("message") or {}
    what = _task_from(msg.get("tool_calls"))
    if what:
        return "", what
    return (msg.get("content") or "").strip(), None
