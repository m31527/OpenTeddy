"""
Cloud planner + live model lists.

  A. cloud_models: each provider's list endpoint is parsed and filtered
     correctly, no-key fallback, caching, failure fallback.
  B. orchestrator routing: which backend the planner/chat calls take in
     local / mixed+local / mixed+cloud / local_only session / cloud mode,
     and that the chosen cloud model is passed through.

    .venv/bin/python tests/test_planner_cloud.py
"""
from __future__ import annotations

import asyncio, logging, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

from config import config
import cloud_models as CM


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status
    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")
    def json(self): return self._p


class _Client:
    routes: dict = {}
    calls: list = []
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def get(self, url, params=None, headers=None):
        _Client.calls.append((url, params, headers))
        r = _Client.routes[url]
        return r(params) if callable(r) else r


def setup_routes():
    _Client.routes = {
        "https://api.anthropic.com/v1/models": lambda p: _Resp(
            {"data": [{"id": "claude-opus-5-5", "display_name": "Claude Opus 5.5", "created_at": "2026-08-01T00:00:00Z"}],
             "has_more": True, "last_id": "claude-opus-5-5"} if not (p or {}).get("after_id") else
            {"data": [{"id": "claude-sonnet-5", "display_name": "Claude Sonnet 5", "created_at": "2026-01-01T00:00:00Z"}],
             "has_more": False, "last_id": "claude-sonnet-5"}),
        "https://api.openai.com/v1/models": _Resp({"data": [
            {"id": "gpt-5.4", "created": 300}, {"id": "text-embedding-3-large", "created": 400},
            {"id": "gpt-4o-mini-tts", "created": 350}, {"id": "o5-mini", "created": 200},
            {"id": "dall-e-3", "created": 100}, {"id": "whisper-1", "created": 90}]}),
        "https://generativelanguage.googleapis.com/v1beta/models": _Resp({"models": [
            {"name": "models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/gemini-3-pro", "displayName": "Gemini 3 Pro", "supportedGenerationMethods": ["generateContent", "countTokens"]},
            {"name": "models/imagen-4", "supportedGenerationMethods": ["predict"]}]}),
        "https://api.deepseek.com/models": _Resp({"data": [{"id": "deepseek-v4-flash"}, {"id": "deepseek-reasoner"}]}),
        "https://openrouter.ai/api/v1/models": _Resp({"data": [
            {"id": "a/old", "name": "Old", "created": 1}, {"id": "b/new", "name": "New", "created": 9}]}),
    }


async def test_models() -> None:
    CM.httpx = types.SimpleNamespace(AsyncClient=_Client, Timeout=lambda *a, **k: None)  # type: ignore[assignment]
    setup_routes(); CM.clear_cache()
    for f in ("anthropic_api_key", "openai_api_key", "gemini_api_key", "deepseek_api_key", "openrouter_api_key"):
        setattr(config, f, "")

    r = await CM.list_models("anthropic")
    assert r["source"] == "fallback" and r["models"] and "no API key" in r["error"] and not _Client.calls
    print("  ✓ no key → built-in fallback, no network")

    r = await CM.list_models("anthropic", api_key="sk-ant-x")
    ids = [m["id"] for m in r["models"]]
    assert r["source"] == "live" and ids == ["claude-opus-5-5", "claude-sonnet-5"], ids
    h = _Client.calls[-1][2]
    assert h["x-api-key"] == "sk-ant-x" and h["anthropic-version"] == "2023-06-01"
    print("  ✓ anthropic: paginated, newest first, x-api-key + anthropic-version")

    r = await CM.list_models("openai", api_key="sk-x")
    assert [m["id"] for m in r["models"]] == ["gpt-5.4", "o5-mini"], r["models"]
    print("  ✓ openai: chat models only (embeddings / tts / image / audio dropped), newest first")

    r = await CM.list_models("gemini", api_key="g-x")
    assert [m["id"] for m in r["models"]] == ["gemini-3-pro", "gemini-2.5-flash"], r["models"]
    print("  ✓ gemini: generateContent only, 'models/' prefix stripped")

    r = await CM.list_models("deepseek", api_key="d-x")
    assert [m["id"] for m in r["models"]] == ["deepseek-v4-flash", "deepseek-reasoner"]
    r = await CM.list_models("openrouter")          # public, no key needed
    assert r["source"] == "live" and [m["id"] for m in r["models"]] == ["b/new", "a/old"]
    print("  ✓ deepseek parsed; openrouter works without a key, newest first")

    n = len(_Client.calls)
    await CM.list_models("deepseek", api_key="d-x")
    assert len(_Client.calls) == n
    await CM.list_models("deepseek", api_key="d-x", refresh=True)
    assert len(_Client.calls) == n + 1
    print("  ✓ cached for 10 min; refresh=True bypasses")

    _Client.routes["https://api.openai.com/v1/models"] = _Resp({}, status=401)
    CM.clear_cache()
    r = await CM.list_models("openai", api_key="bad")
    assert r["source"] == "fallback" and "401" in r["error"]
    r = await CM.list_models("nope")
    assert r["source"] == "error"
    print("  ✓ provider error → fallback with the reason; unknown provider rejected")


async def test_routing() -> None:
    from orchestrator import Orchestrator
    from config import set_session_local_only, orchestrator_on_cloud
    calls = {"gemma": 0, "cloud": []}

    async def gemma(*a, **k):
        calls["gemma"] += 1; return "local answer"

    class Provider:
        model_name = "claude-opus-5-5"; provider_name = "anthropic"
        async def complete_text(self, user_message, system=None, max_tokens=2048, model=None):
            calls["cloud"].append(model)
            return types.SimpleNamespace(text="cloud answer", usage=types.SimpleNamespace(input_tokens=1, output_tokens=1))

    async def _rec(**k): pass
    fake = types.SimpleNamespace(_gemma_complete=gemma, escalation=types.SimpleNamespace(provider=Provider()),
                                 tracker=types.SimpleNamespace(record_usage=_rec))
    run = lambda: Orchestrator._orchestrator_complete(fake, "plan this", "sys")

    config.llm_mode, config.orchestrator_backend, config.orchestrator_cloud_model = "mixed", "local", ""
    assert await run() == "local answer" and calls["gemma"] == 1 and not calls["cloud"]
    print("  ✓ mixed + local planner → Ollama")

    config.orchestrator_backend, config.orchestrator_cloud_model = "cloud", "claude-sonnet-5"
    assert await run() == "cloud answer" and calls["cloud"][-1] == "claude-sonnet-5"
    print("  ✓ mixed + cloud planner → provider, with the chosen model (not the escalation model)")

    set_session_local_only(True)
    try:
        assert not orchestrator_on_cloud()
        assert await run() == "local answer"
    finally:
        set_session_local_only(False)
    print("  ✓ local_only session → stays local even with a cloud planner")

    config.llm_mode = "local"
    assert await run() == "local answer"
    config.llm_mode = "cloud"
    assert await run() == "cloud answer" and calls["cloud"][-1] is None
    print("  ✓ local mode → Ollama; cloud mode → provider with its own model")
    config.llm_mode, config.orchestrator_backend, config.orchestrator_cloud_model = "mixed", "local", ""


async def main() -> None:
    print("A. live model lists"); await test_models()
    print("B. planner routing"); await test_routing()
    print("\nALL PLANNER/CLOUD TESTS PASS")


asyncio.run(main())
