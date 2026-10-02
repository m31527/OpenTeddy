"""
Live model lists for the cloud LLM providers.

The Settings page used to offer hard-coded <datalist> suggestions per
provider, written into index.html by hand — they were a generation behind
within months (the Claude list still recommended 4.7 / 4.6). Every
provider publishes its own list, so ask it:

    anthropic   GET https://api.anthropic.com/v1/models            (x-api-key)
    openai      GET https://api.openai.com/v1/models               (Bearer)
    gemini      GET https://generativelanguage.googleapis.com/v1beta/models?key=…
    deepseek    GET https://api.deepseek.com/models                (Bearer)
    openrouter  GET https://openrouter.ai/api/v1/models            (public)

Results are cached per provider for 10 minutes. Without a key, or when
the provider can't be reached, the caller gets `source: "fallback"` with
a short curated list (or none) and the error — the UI keeps whatever
suggestions it already has, and the field stays free text either way.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config import config

logger = logging.getLogger(__name__)

PROVIDERS = ("anthropic", "openai", "gemini", "deepseek", "openrouter")
_TTL_S = 600.0
_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}

_KEY_FIELD = {
    "anthropic": "anthropic_api_key",
    "openai": "openai_api_key",
    "gemini": "gemini_api_key",
    "deepseek": "deepseek_api_key",
    "openrouter": "openrouter_api_key",
}

# Used only when a live list can't be fetched. Current generation only;
# anything else the operator can still type by hand.
_FALLBACK: Dict[str, List[Dict[str, Any]]] = {
    "anthropic": [
        {"id": "claude-sonnet-5-5", "label": "Claude Sonnet 5.5"},
        {"id": "claude-opus-5-5", "label": "Claude Opus 5.5"},
        {"id": "claude-fable-5-1", "label": "Claude Fable 5.1"},
        {"id": "claude-opus-5", "label": "Claude Opus 5"},
        {"id": "claude-sonnet-5", "label": "Claude Sonnet 5"},
        {"id": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5"},
    ],
}

# OpenAI's /v1/models also lists embeddings, audio, image and moderation
# models; only chat-capable ones belong in a planner/escalation picker.
_OPENAI_KEEP = re.compile(r"^(gpt-|o\d|chatgpt-)", re.I)
_OPENAI_DROP = re.compile(
    r"(embedding|tts|whisper|dall-e|audio|realtime|transcribe|moderation|image|search|instruct|codex-mini)",
    re.I,
)
_GEMINI_DROP = re.compile(r"(embedding|aqa|imagen|veo|tts|learnlm)", re.I)


def _key_for(provider: str, override: Optional[str] = None) -> str:
    k = (override or "").strip()
    if k:
        return k
    return (getattr(config, _KEY_FIELD.get(provider, ""), "") or "").strip()


async def _fetch(provider: str, key: str) -> List[Dict[str, Any]]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as c:
        if provider == "anthropic":
            out: List[Dict[str, Any]] = []
            params: Dict[str, Any] = {"limit": 100}
            for _ in range(10):                       # paginate, bounded
                r = await c.get("https://api.anthropic.com/v1/models", params=params,
                                headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
                r.raise_for_status()
                d = r.json()
                for m in d.get("data") or []:
                    out.append({"id": m.get("id"), "label": m.get("display_name") or m.get("id"),
                                "created": m.get("created_at")})
                if not d.get("has_more") or not d.get("last_id"):
                    break
                params = {"limit": 100, "after_id": d["last_id"]}
            return out                                 # API returns newest first
        if provider == "openai":
            r = await c.get("https://api.openai.com/v1/models",
                            headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            ms = [m for m in r.json().get("data") or []
                  if _OPENAI_KEEP.search(m.get("id", "")) and not _OPENAI_DROP.search(m.get("id", ""))]
            ms.sort(key=lambda m: m.get("created") or 0, reverse=True)
            return [{"id": m["id"], "label": m["id"], "created": m.get("created")} for m in ms]
        if provider == "gemini":
            r = await c.get("https://generativelanguage.googleapis.com/v1beta/models",
                            params={"key": key, "pageSize": 1000})
            r.raise_for_status()
            out = []
            for m in r.json().get("models") or []:
                mid = str(m.get("name", "")).removeprefix("models/")
                if "generateContent" not in (m.get("supportedGenerationMethods") or []):
                    continue
                if not mid or _GEMINI_DROP.search(mid):
                    continue
                out.append({"id": mid, "label": m.get("displayName") or mid, "created": None})
            out.sort(key=lambda m: m["id"], reverse=True)
            return out
        if provider == "deepseek":
            r = await c.get("https://api.deepseek.com/models",
                            headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            return [{"id": m["id"], "label": m["id"], "created": m.get("created")}
                    for m in r.json().get("data") or [] if m.get("id")]
        if provider == "openrouter":
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            r = await c.get("https://openrouter.ai/api/v1/models", headers=headers)
            r.raise_for_status()
            ms = r.json().get("data") or []
            ms.sort(key=lambda m: m.get("created") or 0, reverse=True)
            return [{"id": m["id"], "label": m.get("name") or m["id"], "created": m.get("created")}
                    for m in ms if m.get("id")]
    raise ValueError(f"unknown provider {provider!r}")


async def list_models(provider: str, *, refresh: bool = False,
                      api_key: Optional[str] = None) -> Dict[str, Any]:
    """Return {provider, source, fetched_at, models: [{id, label, created}], error}."""
    provider = (provider or "").strip().lower()
    if provider not in PROVIDERS:
        return {"provider": provider, "source": "error", "models": [],
                "error": f"unknown provider; one of {', '.join(PROVIDERS)}"}
    key = _key_for(provider, api_key)
    cache_key = f"{provider}|{key[-6:] if key else ''}"
    hit = _cache.get(cache_key)
    if hit and not refresh and time.monotonic() - hit[0] < _TTL_S:
        return {**hit[1], "cached": True}
    if not key and provider != "openrouter":
        return {"provider": provider, "source": "fallback", "fetched_at": None,
                "models": _FALLBACK.get(provider, []),
                "error": "no API key for this provider — showing the built-in list"}
    try:
        models = [m for m in await _fetch(provider, key) if m.get("id")]
        res = {"provider": provider, "source": "live", "fetched_at": time.time(),
               "models": models, "error": None}
        _cache[cache_key] = (time.monotonic(), res)
        return res
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"[:240]
        logger.warning("model list for %s failed: %s", provider, msg)
        return {"provider": provider, "source": "fallback", "fetched_at": None,
                "models": _FALLBACK.get(provider, []), "error": msg}


def clear_cache() -> None:
    _cache.clear()
