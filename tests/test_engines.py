"""
engines.py: Settings' "running now" — where each role's calls really go.

  A. roles(): planner / executor on vLLM when vllm_models routes them, else
     Ollama at their base URL; the cloud in cloud mode (and for a cloud
     planner in mixed mode); vision and embeddings on Ollama; the voice
     service when one is set.
  B. judge(): loaded / installed / missing on Ollama (bge-m3 ≡ bge-m3:latest),
     up / not_served on vLLM, ready / starting / off for voice, unreachable,
     cloud with or without a key.
  C. snapshot(): one check per server, shared by the roles on it.

    .venv/bin/python tests/test_engines.py
"""
from __future__ import annotations

import asyncio, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
import engines
import local_engine

ROUTES = {}
local_engine.vllm_routes = lambda: dict(ROUTES)      # (empty on macOS otherwise)


def setup(**kw):
    base = dict(llm_mode="mixed", orchestrator_backend="local", gemma_model="gemma4:e4b",
                qwen_model="qwen3.6:35b", vision_model="qwen3-vl:8b-instruct", knowledge_embed_model="bge-m3",
                gemma_base_url="http://localhost:11434", qwen_base_url="http://localhost:11434",
                voice_service_url="http://192.168.100.2:8095", voice_enabled=True, local_engine="ollama",
                llm_provider="openai", openai_model="gpt-5", openai_api_key="", orchestrator_cloud_model="")
    base.update(kw)
    for k, v in base.items():
        setattr(config, k, v)


def by_role(rs):
    return {r["role"]: r for r in rs}


def test_roles():
    ROUTES.clear(); ROUTES["qwen3.6:35b"] = "http://127.0.0.1:8001"
    setup()
    r = by_role(engines.roles())
    assert (r["planner"]["engine"], r["planner"]["url"]) == ("ollama", "http://localhost:11434")
    assert (r["executor"]["engine"], r["executor"]["url"]) == ("vllm", "http://127.0.0.1:8001"), r["executor"]
    assert r["vision"]["engine"] == "ollama" and r["embed"]["model"] == "bge-m3"
    assert r["voice"]["url"] == "http://192.168.100.2:8095" and r["voice"]["enabled"]
    print("  ✓ spark-a: gemma on Ollama, qwen3.6:35b on its vLLM server, vision/embed on Ollama, voice on B")

    setup(qwen_model="qwen3-vl:8b-instruct")
    assert by_role(engines.roles())["executor"]["engine"] == "ollama", "the Save bug: an executor that isn't routed"
    setup(gemma_model="qwen3.6:35b")
    assert by_role(engines.roles())["planner"]["engine"] == "vllm"
    print("  ✓ a model vllm_models doesn't route goes to Ollama; a routed planner to vLLM")

    setup(orchestrator_backend="cloud", orchestrator_cloud_model="gpt-5-mini", openai_api_key="sk-x")
    r = by_role(engines.roles())
    assert r["planner"]["engine"] == "cloud" and r["planner"]["model"] == "gpt-5-mini" and r["planner"]["has_key"] == "yes"
    assert r["executor"]["engine"] == "vllm"
    setup(llm_mode="cloud")
    r = by_role(engines.roles())
    assert r["planner"]["engine"] == r["executor"]["engine"] == "cloud" and r["executor"]["model"] == "gpt-5"
    assert r["executor"]["has_key"] == "no"
    setup(vision_model="", knowledge_embed_model="", voice_service_url="")
    assert [x["role"] for x in engines.roles()] == ["planner", "executor"]
    print("  ✓ cloud planner in mixed mode, everything on the cloud in cloud mode; unset roles left out")


def test_judge():
    ollama = {"ok": True, "installed": {"gemma4:e4b", "bge-m3:latest", "qwen3-vl:8b-instruct"},
              "loaded": {"gemma4:e4b": {"size": 7_200_000_000, "size_vram": 559_000_000, "context_length": 16384}}}
    j = engines.judge({"role": "planner", "model": "gemma4:e4b", "engine": "ollama", "url": "u"}, ollama)
    assert j["state"] == "loaded" and j["context"] == 16384 and "vram_gb" not in j
    assert engines.judge({"role": "embed", "model": "bge-m3", "engine": "ollama", "url": "u"}, ollama)["state"] == "installed"
    assert engines.judge({"role": "x", "model": "llama9", "engine": "ollama", "url": "u"}, ollama)["state"] == "missing"
    print("  ✓ Ollama: loaded (context; no memory figure) / installed (bge-m3 = bge-m3:latest) / missing")

    vllm = {"ok": True, "served": {"qwen3.6:35b": {"max_model_len": 65536, "root": "nvidia/Qwen3.6-35B-A3B-NVFP4"}}}
    j = engines.judge({"role": "executor", "model": "qwen3.6:35b", "engine": "vllm", "url": "u"}, vllm)
    assert j["state"] == "up" and j["context"] == 65536 and j["weights"].startswith("nvidia/")
    j = engines.judge({"role": "executor", "model": "other", "engine": "vllm", "url": "u"}, vllm)
    assert j["state"] == "not_served" and j["served"] == ["qwen3.6:35b"]
    assert engines.judge({"role": "executor", "model": "m", "engine": "vllm", "url": "u"}, {"ok": False})["state"] == "unreachable"
    print("  ✓ vLLM: serving (context, weights) / not on that server (what it serves) / unreachable")

    h = {"ok": True, "health": {"ok": True, "asr": "FunAudioLLM/SenseVoiceSmall", "tts": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"}}
    j = engines.judge({"role": "voice", "model": "", "engine": "voice", "url": "u", "enabled": True}, h)
    assert j["state"] == "ready" and j["model"] == "SenseVoiceSmall · Fun-CosyVoice3-0.5B-2512"
    assert engines.judge({"role": "voice", "model": "", "engine": "voice", "url": "u", "enabled": False}, h)["state"] == "off"
    j = engines.judge({"role": "voice", "model": "", "engine": "voice", "url": "u", "enabled": True},
                      {"ok": True, "health": {"ok": False, "stage": "downloading"}})
    assert j["state"] == "starting" and j["stage"] == "downloading"
    assert engines.judge({"role": "p", "model": "m", "engine": "cloud", "url": "openai", "has_key": "no"}, None)["state"] == "no_key"
    assert engines.judge({"role": "p", "model": "m", "engine": "cloud", "url": "openai", "has_key": "yes"}, None)["state"] == "cloud"
    print("  ✓ voice ready / off / starting; cloud with and without a key")


def test_snapshot():
    ROUTES.clear(); ROUTES["qwen3.6:35b"] = "http://127.0.0.1:8001"
    setup()
    calls = []

    async def fake_ollama(client, url):
        calls.append(("ollama", url))
        return {"ok": True, "installed": {"gemma4:e4b", "bge-m3:latest", "qwen3-vl:8b-instruct"}, "loaded": {}}

    async def fake_vllm(client, url):
        calls.append(("vllm", url))
        return {"ok": True, "served": {"qwen3.6:35b": {"max_model_len": 65536}}}

    async def fake_voice(client, url):
        calls.append(("voice", url))
        return {"ok": False, "error": "ConnectError"}
    engines._ollama, engines._vllm, engines._voice = fake_ollama, fake_vllm, fake_voice
    snap = asyncio.run(engines.snapshot())
    states = {r["role"]: r["state"] for r in snap["roles"]}
    assert states == {"planner": "installed", "executor": "up", "vision": "installed", "embed": "installed",
                      "voice": "unreachable"}, states
    assert sorted(calls) == [("ollama", "http://localhost:11434"), ("vllm", "http://127.0.0.1:8001"),
                             ("voice", "http://192.168.100.2:8095")], calls
    print("  ✓ snapshot: one check per server, shared by the three roles on Ollama")


if __name__ == "__main__":
    print("A. roles"); test_roles()
    print("B. judge"); test_judge()
    print("C. snapshot"); test_snapshot()
    print("\nALL ENGINE TESTS PASS")
