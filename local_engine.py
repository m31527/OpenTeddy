"""
OpenTeddy local-inference engine abstraction.

Two backends serve the local executor model:

  - "ollama" (default) — cross-platform (macOS / Linux / Windows). Hits
    Ollama's NATIVE /api/chat endpoint so we keep per-request `num_ctx`
    and `keep_alive`, which the context watchdog + warm-model UX depend
    on. This is what every OpenTeddy install has used to date.

  - "vllm" — Linux + CUDA only. Hits vLLM's OpenAI-compatible
    /v1/chat/completions. On DGX-class hardware vLLM delivers ~3-5x the
    throughput of Ollama, chiefly via continuous batching + automatic
    prefix caching. The prefix-cache win is large for OpenTeddy
    specifically: the planner / executor system prompts are ~2k tokens
    each and are byte-identical across calls, so caching their KV once
    instead of re-prefilling every call is most of the speedup.

Why an abstraction layer instead of just swapping base URLs:

  The two engines speak different request/response dialects:

    Ollama /api/chat          | vLLM /v1/chat/completions (OpenAI)
    --------------------------|----------------------------------------
    `system` is top-level     | `system` is a messages[0] entry
    options:{num_ctx,...}     | flat temperature / max_tokens
    keep_alive per request    | N/A (model resident at serve time)
    resp.message.{...}        | resp.choices[0].message.{...}
    resp.eval_count etc.      | resp.usage.completion_tokens
    NDJSON streaming          | SSE streaming

  This module normalises both to the Ollama-shaped response dict the
  executor already consumes, so the executor's tool-use loop stays
  engine-agnostic — it builds a request via build_payload(), POSTs to
  chat_endpoint(), and reads normalize_response().

macOS hard-gate:

  vLLM has no Metal backend and won't pip-install on a Mac. So even if a
  user carries a `local_engine=vllm` setting over from a Linux box (e.g.
  via synced settings), active_engine() forces "ollama" on Darwin. A Mac
  user can never end up pointed at a non-existent vLLM server.
"""

from __future__ import annotations

import logging
import platform
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Engine resolution ─────────────────────────────────────────────────────────


def active_engine() -> str:
    """Resolve the effective local engine, applying the macOS hard-gate.

    Returns "ollama" or "vllm". Anything unrecognised → "ollama"."""
    from config import config
    requested = (getattr(config, "local_engine", "ollama") or "ollama").lower()
    if requested == "vllm" and platform.system() == "Darwin":
        # vLLM can't run on macOS — silently fall back rather than point
        # the executor at a server that will never exist.
        return "ollama"
    return requested if requested in ("ollama", "vllm") else "ollama"


def vllm_routes() -> Dict[str, str]:
    """model → base URL of the vLLM server that serves it, from the
    `vllm_models` setting ("qwen3.6:35b=http://127.0.0.1:8001, …"). One
    vLLM server serves one model, so a planner and an executor on vLLM
    are two servers, and a model left out stays where it was (Ollama).
    Empty on macOS, for the same reason as active_engine()."""
    from config import config
    if platform.system() == "Darwin":
        return {}
    out: Dict[str, str] = {}
    for part in re.split(r"[,\n]", getattr(config, "vllm_models", "") or ""):
        model, _, url = part.partition("=")
        if model.strip() and url.strip():
            out[model.strip()] = url.strip().rstrip("/")
    return out


def routed(model: str) -> bool:
    """`model` has a vLLM server of its own (vllm_models). The planner's
    calls check this, not is_vllm(): with local_engine=vllm alone the one
    vLLM server serves the executor, and the planner stays on Ollama."""
    return model in vllm_routes()


def is_vllm(model: Optional[str] = None) -> bool:
    """Whether a call for `model` goes to vLLM: its own server when it is
    in vllm_models, else whatever local_engine says."""
    if model is not None and model in vllm_routes():
        return True
    return active_engine() == "vllm"


def supports_streaming(model: Optional[str] = None) -> bool:
    """Whether to use token-streaming for the active engine.

    Ollama streams NDJSON and we render word-by-word in the chat UI.
    vLLM streaming with tool-call deltas uses the fiddlier OpenAI SSE
    delta format; for the first cut we run vLLM non-streamed (it's the
    throughput-for-batch engine anyway — word-by-word UI matters less
    for the scheduled / fleet workloads vLLM targets). Streaming for
    vLLM is a follow-up enhancement, not a correctness requirement."""
    return not is_vllm(model)


def base_url(model: Optional[str] = None) -> str:
    from config import config
    if model is not None and model in vllm_routes():
        return vllm_routes()[model]
    if is_vllm():
        return getattr(config, "vllm_base_url", "http://127.0.0.1:8001")
    return config.qwen_base_url


def chat_endpoint(model: Optional[str] = None) -> str:
    """Full POST URL for a chat completion of `model` (default: the
    executor's) on the engine that serves it."""
    b = base_url(model).rstrip("/")
    return f"{b}/v1/chat/completions" if is_vllm(model) else f"{b}/api/chat"


def usage_provider_label(model: Optional[str] = None) -> str:
    """String stamped on usage records' model_provider column so the
    Usage tab can break local cost/speed down by engine."""
    return "vllm" if is_vllm(model) else "ollama"


# ── Request building ──────────────────────────────────────────────────────────


_THINKING_FAMILIES = ("qwen3", "gemma4", "deepseek-r1", "gpt-oss", "magistral", "qwq")


def is_thinking_model(name: str) -> bool:
    """Models whose Ollama build exposes the `think` switch. Sending
    `think` to a model without it is a 400, so the switch is only added
    for known families."""
    n = (name or "").lower()
    return any(n.startswith(f) or f"/{f}" in n for f in _THINKING_FAMILIES)


def think_setting(model: str) -> Optional[bool]:
    """None = leave the model's default; False = switch thinking off."""
    from config import config
    if getattr(config, "ollama_think", False):
        return None
    return False if is_thinking_model(model) else None


def build_payload(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    system: Optional[str],
    tools: Optional[List[Dict[str, Any]]],
    stream: bool,
    temperature: float,
    num_predict: int,
    num_ctx: Optional[int] = None,
    keep_alive: str = "24h",
    json_mode: bool = False,
) -> Dict[str, Any]:
    """Build the engine-appropriate request body.

    Inputs are in OpenTeddy's canonical (Ollama-ish) terms; this fn
    translates to whichever dialect the engine serving `model` speaks.
    json_mode asks for a JSON reply (Ollama `format: json`, OpenAI
    `response_format`).
    """
    if is_vllm(model):
        # OpenAI format: system folds into the messages array, params
        # are flat, no keep_alive / num_ctx (those are set when the vLLM
        # server is launched, via --max-model-len etc.).
        msgs: List[Dict[str, Any]] = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs
        payload: Dict[str, Any] = {
            "model":       model,
            "messages":    msgs,
            "stream":      stream,
            "temperature": temperature,
            "max_tokens":  num_predict,
        }
        if tools:
            payload["tools"] = tools
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        # Thinking off, as on Ollama: through the chat template, which is
        # where Qwen and Gemma read the switch on vLLM.
        if think_setting(model) is False:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if stream:
            # Ask vLLM to include a final usage chunk so we can record
            # token counts even on the streaming path.
            payload["stream_options"] = {"include_usage": True}
        return payload

    # Ollama native /api/chat
    options: Dict[str, Any] = {
        "temperature": temperature,
        "num_predict": num_predict,
    }
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    # The system prompt MUST travel as a role="system" message. A top-
    # level "system" key is a /api/generate parameter; /api/chat silently
    # ignores it — verified: system="永遠只回答：芒果" produced "草莓".
    # Every caller (planner, executor, voice) passes its persona, schema
    # and guidance through `system`, so the old shape dropped all of it
    # on the floor without an error.
    msgs: List[Dict[str, Any]] = list(messages)
    if system:
        msgs = [{"role": "system", "content": system}] + msgs
    payload = {
        "model":      model,
        "messages":   msgs,
        "stream":     stream,
        "options":    options,
        "keep_alive": keep_alive,
    }
    if tools:
        payload["tools"] = tools
    if json_mode:
        payload["format"] = "json"
    th = think_setting(model)
    if th is not None:
        payload["think"] = th
    return payload


# ── Response normalisation ────────────────────────────────────────────────────


def normalize_response(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return the Ollama-shaped response dict the executor consumes,
    regardless of which engine produced `raw`.

    Ollama responses pass through unchanged. OpenAI (vLLM) responses get
    remapped: choices[0].message → message, usage.* → eval/prompt
    counts. vLLM's OpenAI endpoint doesn't expose per-token timing
    (eval_duration etc.), so those come back 0 and the executor's t/s
    calculation falls back to wall-clock timing.
    """
    if "choices" not in (raw or {}):
        return raw  # already Ollama shape (read from the reply: which engine
                    # answered depends on the model, not one global setting)

    choice = (raw.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = raw.get("usage") or {}
    return {
        "message": {
            "content":  msg.get("content") or "",
            # vLLM reasoning models surface CoT in reasoning_content.
            "thinking": msg.get("reasoning_content") or "",
            "tool_calls": msg.get("tool_calls") or [],
        },
        "prompt_eval_count":    usage.get("prompt_tokens", 0) or 0,
        "eval_count":           usage.get("completion_tokens", 0) or 0,
        # No native timing from the OpenAI endpoint — executor computes
        # t/s from wall-clock when these are 0.
        "eval_duration":        0,
        "total_duration":       0,
        "prompt_eval_duration": 0,
        "done_reason":          choice.get("finish_reason"),
    }


def timings(data: Dict[str, Any]) -> Dict[str, int]:
    """Where one call's time went, in ms: load (bringing the model into
    memory), prefill (reading the prompt — what a reused KV-cache prefix
    saves) and decode (writing the answer). Ollama only; vLLM reports 0."""
    ms = lambda k: int((data.get(k) or 0) / 1_000_000)
    return {"load_ms": ms("load_duration"), "prefill_ms": ms("prompt_eval_duration"),
            "decode_ms": ms("eval_duration"), "total_ms": ms("total_duration")}


def log_timing(kind: str, model: str, data: Dict[str, Any]) -> Dict[str, int]:
    """One log line per model call, e.g.
    "LLM executor qwen3.8:27b: load 0 ms · prefill 412 tok 0.9 s · decode 88 tok 6.1 s · 7.2 s"."""
    t = timings(data)
    if t["total_ms"]:
        logger.info("LLM %s %s: load %d ms · prefill %d tok %.1f s · decode %d tok %.1f s · %.1f s",
                    kind, model, t["load_ms"], data.get("prompt_eval_count") or 0, t["prefill_ms"] / 1000,
                    data.get("eval_count") or 0, t["decode_ms"] / 1000, t["total_ms"] / 1000)
    return t


def normalize_stream_chunk(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a single streaming chunk to the Ollama shape the
    executor's streaming loop reads (chunk.message.content /
    .thinking / .tool_calls + chunk.done + the trailing stats).

    Only meaningful for engines where supports_streaming() is True. For
    the current cut that's Ollama only, so this is effectively an
    identity passthrough — but it's here so the executor's streaming
    block can route through it uniformly, and so vLLM SSE streaming can
    be slotted in later without touching the executor.
    """
    if "choices" not in (chunk or {}):
        return chunk
    # vLLM SSE delta → Ollama-ish chunk. (Wired for completeness; not
    # exercised until supports_streaming() returns True for vLLM.)
    choice = (chunk.get("choices") or [{}])[0]
    delta = choice.get("delta") or {}
    usage = chunk.get("usage") or {}
    done = choice.get("finish_reason") is not None
    return {
        "message": {
            "content":    delta.get("content") or "",
            "thinking":   delta.get("reasoning_content") or "",
            "tool_calls": delta.get("tool_calls") or [],
        },
        "done": done,
        "prompt_eval_count": usage.get("prompt_tokens", 0) or 0,
        "eval_count":        usage.get("completion_tokens", 0) or 0,
        "done_reason":       choice.get("finish_reason"),
    }
