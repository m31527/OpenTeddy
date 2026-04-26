"""
OpenTeddy Model Profile
─────────────────────────────────────────────────────────────────────────────
Lightweight introspection on the local LLM tags so the orchestrator and
executor can adapt their prompts to the model's capability tier.

A 2 B "thinking" model needs an example-heavy, strictly-formatted system
prompt or it'll happily wander off into tool-use freeform; a 27 B model
benefits from the opposite — a brief, open-ended directive.

We try to keep this purely lexical (parse the Ollama tag) so the agent
loop never blocks on a network round-trip just to size the model.
"""
from __future__ import annotations

import re
from functools import lru_cache

# Match patterns like: ":2b", ":7b", ":35b", ":e2b", ":e4b", ":1.5b", ":0.8b".
# Group 1 captures the numeric value, group 2 the lowercase 'b' marker.
_TAG_SIZE_RE = re.compile(r":(?:e)?(\d+(?:\.\d+)?)b\b", re.IGNORECASE)


@lru_cache(maxsize=128)
def model_size_billions(name: str) -> float:
    """Parse parameter count (in billions) from an Ollama model tag.

    Returns 0.0 when the tag doesn't expose a number (e.g. ``:latest``).
    Callers should treat 0.0 as "unknown — use the balanced default".

    Examples
    --------
    >>> model_size_billions("qwen3.5:2b")     # 2.0
    >>> model_size_billions("gemma4:e4b")     # 4.0
    >>> model_size_billions("qwen3.5:0.8b")   # 0.8
    >>> model_size_billions("gemma4:26b")     # 26.0
    >>> model_size_billions("qwen3.5:latest") # 0.0  (unknown)
    """
    if not name:
        return 0.0
    m = _TAG_SIZE_RE.search(name)
    if m:
        try:
            return float(m.group(1))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def prompt_tier(size_b: float) -> str:
    """Map model size → prompt-strictness tier.

    • ``strict``   — under 4 B. Small / "thinking" models. Need explicit
                     few-shot examples and a "ONLY output X" preamble or
                     they'll hallucinate prose around tool calls.
    • ``balanced`` — 4–11 B. The current OpenTeddy prompts target this.
    • ``open``     — 12 B+. Trust the model, give it room to reason.

    Unknown size (0.0) falls back to ``balanced`` so behaviour matches
    the historical default.
    """
    if size_b <= 0:
        return "balanced"
    if size_b < 4.0:
        return "strict"
    if size_b < 12.0:
        return "balanced"
    return "open"


def model_tier(name: str) -> str:
    """Convenience: tag → prompt tier in one call."""
    return prompt_tier(model_size_billions(name))
