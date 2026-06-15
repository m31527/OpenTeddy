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

import platform
from typing import Any, Dict, List, Optional


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


def is_vllm() -> bool:
    return active_engine() == "vllm"


def unified_model() -> str:
    """The configured single-model id, or "" if the 2-model split is in
    effect. Stripped so a stray-whitespace setting reads as unset."""
    from config import config
    return (getattr(config, "unified_model", "") or "").strip()


def local_model(role: str = "executor") -> str:
    """Resolve the model id for a role on the active engine.

    Three cases, in priority order:

    1. `unified_model` set → that one id for EVERY role. This is the
       single-model design: planner and executor share one model (and,
       on vLLM, one resident instance) with no switch on the
       plan→execute boundary.
    2. vLLM with no unified id → still one id, because a vLLM server
       hosts exactly one model. Both roles use the executor's model
       (qwen_model) — whatever vLLM was launched serving. A vLLM user
       can't have a separate Gemma planner; there's only one served
       model.
    3. Ollama, 2-model split (the historical default) → planner uses
       gemma_model, executor uses qwen_model.

    `role` is "planner" or "executor" (anything non-"planner" is
    treated as executor)."""
    from config import config
    u = unified_model()
    if u:
        return u
    if is_vllm():
        return config.qwen_model
    return config.gemma_model if role == "planner" else config.qwen_model


def supports_streaming() -> bool:
    """Whether to use token-streaming for the active engine.

    Ollama streams NDJSON and we render word-by-word in the chat UI.
    vLLM streaming with tool-call deltas uses the fiddlier OpenAI SSE
    delta format; for the first cut we run vLLM non-streamed (it's the
    throughput-for-batch engine anyway — word-by-word UI matters less
    for the scheduled / fleet workloads vLLM targets). Streaming for
    vLLM is a follow-up enhancement, not a correctness requirement."""
    return not is_vllm()


def base_url() -> str:
    from config import config
    if is_vllm():
        return getattr(config, "vllm_base_url", "http://127.0.0.1:8001")
    return config.qwen_base_url


def chat_endpoint() -> str:
    """Full POST URL for a chat completion against the active engine."""
    b = base_url().rstrip("/")
    return f"{b}/v1/chat/completions" if is_vllm() else f"{b}/api/chat"


def usage_provider_label() -> str:
    """String stamped on usage records' model_provider column so the
    Usage tab can break local cost/speed down by engine."""
    return "vllm" if is_vllm() else "ollama"


# ── Request building ──────────────────────────────────────────────────────────


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
    translates to whichever dialect the active engine speaks.

    json_mode forces a strict-JSON response. The two engines spell this
    differently — Ollama's `format: "json"` vs OpenAI/vLLM's
    `response_format: {type: "json_object"}` — so callers that need a
    machine-parseable verdict (e.g. the deliverable verifier) pass
    json_mode=True and stay engine-agnostic.
    """
    if is_vllm():
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
    payload = {
        "model":      model,
        "messages":   messages,
        "system":     system,
        "stream":     stream,
        "options":    options,
        "keep_alive": keep_alive,
    }
    if tools:
        payload["tools"] = tools
    if json_mode:
        payload["format"] = "json"
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
    if not is_vllm():
        return raw  # already Ollama shape

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
    if not is_vllm():
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
