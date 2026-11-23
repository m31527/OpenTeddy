"""
What actually runs each model role, and whether it is up — for Settings.

The model dropdowns say what is configured; this says where each call
really goes, by the same rules the calls use:

  planner (chat · plan · summary)  gemma_model → its vLLM server when
                                   vllm_models routes it, else Ollama at
                                   gemma_base_url; the cloud provider when
                                   the planner is on the cloud.
  executor                         qwen_model → vLLM when routed (or
                                   local_engine=vllm), else Ollama at
                                   qwen_base_url; each step on the cloud
                                   in cloud mode.
  vision                           vision_model, Ollama at qwen_base_url
                                   (tools/browser_ext_tool.py).
  embed                            knowledge_embed_model, Ollama at
                                   gemma_base_url (knowledge.py).
  voice                            the voice service (voice_bridge.py).

and asks each server what it has: Ollama /api/tags + /api/ps (installed,
loaded), vLLM /v1/models (served), the voice service /health. A spark-a
executor on vLLM showed in the dropdown as an Ollama vision model, and a
Save made it one (2026-11-23); this is the view that would have shown it.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import httpx

from config import config

TIMEOUT = 3.0


def _base(name: str) -> str:
    """Ollama names a model "bge-m3:latest" when it was pulled as "bge-m3"."""
    return name if ":" in name else f"{name}:latest"


def _provider() -> Dict[str, str]:
    p = (getattr(config, "llm_provider", "") or "anthropic").strip().lower()
    model = {"anthropic": "claude_model", "openai": "openai_model", "gemini": "gemini_model",
             "deepseek": "deepseek_model", "openrouter": "openrouter_model"}.get(p, "claude_model")
    key = {"anthropic": "anthropic_api_key"}.get(p, f"{p}_api_key")
    return {"provider": p, "model": str(getattr(config, model, "") or ""),
            "has_key": "yes" if (getattr(config, key, "") or "").strip() else "no"}


def roles() -> List[Dict[str, Any]]:
    """Each role: the model, the engine and the address its calls go to."""
    import local_engine
    out: List[Dict[str, Any]] = []
    mode = (getattr(config, "llm_mode", "") or "local").strip().lower()
    cloud = _provider()

    planner = (getattr(config, "gemma_model", "") or "").strip()
    if mode == "cloud" or (mode == "mixed" and getattr(config, "orchestrator_backend", "local") == "cloud"):
        cm = (getattr(config, "orchestrator_cloud_model", "") or "").strip() or cloud["model"]
        out.append({"role": "planner", "model": cm, "engine": "cloud", "url": cloud["provider"],
                    "has_key": cloud["has_key"]})
    elif local_engine.routed(planner):
        out.append({"role": "planner", "model": planner, "engine": "vllm", "url": local_engine.vllm_routes()[planner]})
    else:
        out.append({"role": "planner", "model": planner, "engine": "ollama",
                    "url": (config.gemma_base_url or "").rstrip("/")})

    executor = (getattr(config, "qwen_model", "") or "").strip()
    if mode == "cloud":
        out.append({"role": "executor", "model": cloud["model"], "engine": "cloud", "url": cloud["provider"],
                    "has_key": cloud["has_key"]})
    elif local_engine.is_vllm(executor):
        out.append({"role": "executor", "model": executor, "engine": "vllm",
                    "url": local_engine.base_url(executor).rstrip("/")})
    else:
        out.append({"role": "executor", "model": executor, "engine": "ollama",
                    "url": (config.qwen_base_url or "").rstrip("/")})

    vision = (getattr(config, "vision_model", "") or "").strip()
    if vision:
        out.append({"role": "vision", "model": vision, "engine": "ollama",
                    "url": (config.qwen_base_url or "").rstrip("/")})
    embed = (getattr(config, "knowledge_embed_model", "") or "").strip()
    if embed:
        out.append({"role": "embed", "model": embed, "engine": "ollama",
                    "url": (config.gemma_base_url or "").rstrip("/")})
    voice_url = (getattr(config, "voice_service_url", "") or "").strip().rstrip("/")
    if voice_url:
        out.append({"role": "voice", "model": "", "engine": "voice", "url": voice_url,
                    "enabled": bool(getattr(config, "voice_enabled", False))})
    return out


async def _ollama(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
    try:
        tags, ps = await asyncio.gather(client.get(f"{url}/api/tags"), client.get(f"{url}/api/ps"))
        installed = {m.get("name") for m in tags.json().get("models", [])}
        loaded = {m.get("name"): m for m in ps.json().get("models", [])}
        return {"ok": True, "installed": installed, "loaded": loaded}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__}


async def _vllm(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
    try:
        r = await client.get(f"{url}/v1/models")
        r.raise_for_status()
        return {"ok": True, "served": {m.get("id"): m for m in r.json().get("data", [])}}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__}


async def _voice(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
    try:
        h = (await client.get(f"{url}/health")).json()
        return {"ok": True, "health": h}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": type(exc).__name__}


def judge(role: Dict[str, Any], server: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The role plus a `state` the UI shows: loaded / installed / missing /
    up / not_served / ready / starting / off / unreachable / cloud / no_key."""
    r = dict(role)
    eng = r["engine"]
    if eng == "cloud":
        r["state"] = "cloud" if r.get("has_key") == "yes" else "no_key"
        return r
    if not server or not server.get("ok"):
        r["state"] = "unreachable"
        return r
    if eng == "ollama":
        name = _base(r["model"])
        if name in server["loaded"]:
            m = server["loaded"][name]
            r["state"] = "loaded"
            # `size`, not `size_vram`: on GB10's shared memory Ollama counts
            # only a sliver as VRAM (0.6 GB for a 6.6 GB gemma4:e4b).
            r["vram_gb"] = round((m.get("size") or m.get("size_vram") or 0) / 1e9, 1)
            r["context"] = m.get("context_length")
        elif name in server["installed"]:
            r["state"] = "installed"
        else:
            r["state"] = "missing"
    elif eng == "vllm":
        m = server["served"].get(r["model"])
        r["state"] = "up" if m else "not_served"
        if m:
            r["context"] = m.get("max_model_len")
            r["weights"] = m.get("root")
        else:
            r["served"] = sorted(server["served"])
    elif eng == "voice":
        h = server["health"]
        r["model"] = " · ".join(x.split("/")[-1] for x in (h.get("asr"), h.get("tts")) if x)
        r["state"] = ("ready" if h.get("ok") else "starting") if r.get("enabled") else "off"
        if not h.get("ok"):
            r["stage"] = h.get("stage")
    return r


async def snapshot() -> Dict[str, Any]:
    """Every role with its engine, address and live state."""
    rs = roles()
    checks: Dict[tuple, Any] = {}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for r in rs:
            key = (r["engine"], r["url"])
            if key in checks or r["engine"] == "cloud":
                continue
            fn = {"ollama": _ollama, "vllm": _vllm, "voice": _voice}[r["engine"]]
            checks[key] = fn(client, r["url"])
        done = dict(zip(checks, await asyncio.gather(*checks.values()))) if checks else {}
    return {"mode": (getattr(config, "llm_mode", "") or "local"),
            "roles": [judge(r, done.get((r["engine"], r["url"]))) for r in rs]}
