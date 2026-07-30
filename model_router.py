"""
OpenTeddy model router — task-first routing between local and strong models.

Three strategies (skill-plus spec §9):

  LOCAL_PREFERRED          routine work: classification, summarization,
                           intent, skill lookup, simple planning.
  STRONG_MODEL_PREFERRED   skill generation / repair, complex coding,
                           complex reasoning, repeated local failure.
  LOCAL_ONLY               user policy: NO cloud call may happen — not
                           even as a hidden fallback.

This module does NOT redesign OpenTeddy as cloud-first. It formalises
routing that already exists informally (llm_mode local/mixed/cloud +
per-session local_only ContextVar + escalation) into one place the Skill
Builder and future callers can consult, so "which model does this task
deserve" stops being decided ad-hoc inside business logic.

The hard guarantee: when the effective strategy is LOCAL_ONLY,
`completer_for()` returns a callable that never touches the cloud
provider object at all — enforcement by construction, not by a flag
check inside the cloud path. Scenario E in the acceptance tests asserts
this with a provider spy.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Tuple

import httpx

from config import config, is_local_mode

logger = logging.getLogger(__name__)


class ModelStrategy(str, Enum):
    LOCAL_PREFERRED = "local_preferred"
    STRONG_MODEL_PREFERRED = "strong_model_preferred"
    LOCAL_ONLY = "local_only"


# Task kinds that deserve the strong model when one is allowed+configured.
_STRONG_TASK_KINDS = {
    "skill_generation", "skill_repair", "skill_test_generation",
    "complex_coding", "complex_reasoning", "difficult_planning",
}


def resolve_strategy(task_kind: str) -> ModelStrategy:
    """Map a task kind to the effective strategy, honouring user policy.

    Local-only policy (global llm_mode='local' OR the current session's
    privacy flag) always wins — it is a user promise, not a preference.
    """
    if is_local_mode():
        return ModelStrategy.LOCAL_ONLY
    if task_kind in _STRONG_TASK_KINDS:
        return ModelStrategy.STRONG_MODEL_PREFERRED
    return ModelStrategy.LOCAL_PREFERRED


# A text completer: (user_message, system, max_tokens) -> text
TextCompleter = Callable[[str, Optional[str], int], Awaitable[str]]


async def _local_complete(
    user_message: str, system: Optional[str], max_tokens: int,
) -> str:
    """One-shot completion on the LOCAL engine (Ollama or vLLM via
    local_engine), using the executor model. No cloud contact."""
    import local_engine
    payload = local_engine.build_payload(
        model=config.qwen_model,
        messages=[{"role": "user", "content": user_message}],
        system=system,
        tools=None,
        stream=False,
        temperature=0.2,
        num_predict=max_tokens,
        num_ctx=int(getattr(config, "qwen_num_ctx", 16384)),
        keep_alive=getattr(config, "ollama_keep_alive", "24h"),
    )
    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(local_engine.chat_endpoint(), json=payload)
        resp.raise_for_status()
        msg = local_engine.normalize_response(resp.json()).get("message") or {}
        return (msg.get("content") or msg.get("thinking") or "").strip()


def completer_for(
    strategy: ModelStrategy, provider: Any = None,
) -> Tuple[TextCompleter, str]:
    """Return (completer, model_label) for a strategy.

    provider: an llm_provider.LLMProvider (injected for testability).
    Selection rules:
      LOCAL_ONLY               → local completer, provider object untouched.
      STRONG_MODEL_PREFERRED   → cloud provider when configured, else local.
      LOCAL_PREFERRED          → local completer (cloud never needed).
    """
    if strategy == ModelStrategy.LOCAL_ONLY:
        logger.info("model.route.selected strategy=local_only model=%s",
                    config.qwen_model)
        return _local_complete, config.qwen_model

    if strategy == ModelStrategy.STRONG_MODEL_PREFERRED and provider is not None:
        try:
            configured = provider.is_configured()
        except Exception:  # noqa: BLE001
            configured = False
        if configured:
            model_label = getattr(provider, "model_name", "cloud")

            async def _cloud_complete(
                user_message: str, system: Optional[str], max_tokens: int,
            ) -> str:
                resp = await provider.complete_text(
                    user_message=user_message, system=system,
                    max_tokens=max_tokens,
                )
                return (resp.text or "").strip()

            logger.info("model.route.selected strategy=strong model=%s",
                        model_label)
            return _cloud_complete, model_label

    logger.info("model.route.selected strategy=%s model=%s (local)",
                strategy.value, config.qwen_model)
    return _local_complete, config.qwen_model
