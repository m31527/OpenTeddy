#!/usr/bin/env python3
"""
Acceptance tests for the self-learning skill architecture
(feature/self-learning-skills — skill-plus.md §16, scenarios A–G).

Standalone runner (the repo has no pytest): run with

    .venv/bin/python tests/test_self_learning_skills.py

Every scenario uses a REAL SQLite tracker + the real matcher/builder/
runtime; only the LLM is faked (SkillFactory's injectable provider), so
no commercial API calls happen. Each scenario gets a fresh DB +
skills_dir for isolation.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

GOOD_COUNT = (
    "async def run(input_data: dict) -> str:\n"
    "    text = str(input_data.get('text', ''))\n"
    "    return f'word_count={len(text.split())}'\n"
)
GOOD_TEMP = (
    "async def run(input_data: dict) -> str:\n"
    "    c = float(input_data.get('celsius', 0))\n"
    "    return f'{c * 9 / 5 + 32}F'\n"
)
BAD_RAISES = (
    "async def run(input_data: dict) -> str:\n"
    "    raise RuntimeError('boom')\n"
)

SPEC_TEMP = ('{"name": "celsius_to_fahrenheit", '
             '"description": "convert celsius temperatures to fahrenheit", '
             '"capabilities": ["temperature", "celsius", "fahrenheit", "convert"], '
             '"input_keys": ["celsius"]}')
PERMS_NONE = ('{"filesystem": {"read": [], "write": []}, '
              '"network": {"domains": []}, "commands": [], '
              '"credentials": [], "services": []}')
PERMS_NET = ('{"filesystem": {"read": [], "write": []}, '
             '"network": {"domains": ["api.example.com"]}, "commands": [], '
             '"credentials": [], "services": []}')


class FakeResp:
    def __init__(self, text): self.text = text


class FakeProvider:
    """Marker-routed scripted provider. Counts calls per kind so tests can
    assert 'the Skill Builder was NOT called'."""
    model_name = "fake-cloud"

    def __init__(self, code=GOOD_TEMP, spec=SPEC_TEMP, perms=PERMS_NONE,
                 repair=None):
        self.code, self.spec, self.perms = code, spec, perms
        self.repair = repair or code
        self.calls: dict[str, int] = {}

    def is_configured(self): return True
    def get_unconfigured_message(self): return "no key"

    async def complete_text(self, user_message, system=None, max_tokens=0):
        if "Define a REUSABLE skill" in user_message:
            kind = "spec"; out = self.spec
        elif "smoke test" in user_message:
            kind = "test_input"; out = '{"celsius": 100}'
        elif "MINIMUM permissions" in user_message:
            kind = "permissions"; out = self.perms
        elif "FAILED" in user_message:
            kind = "repair"; out = self.repair
        else:
            kind = "generate"; out = self.code
        self.calls[kind] = self.calls.get(kind, 0) + 1
        return FakeResp(out)


async def fresh_env(provider, runtime=None):
    from config import config
    from tracker import Tracker
    from skill_factory import SkillFactory
    config.skills_dir = tempfile.mkdtemp()
    config.skill_test_before_register = True
    config.skill_repair_max_attempts = 3
    config.llm_mode = "mixed"
    t = Tracker(db_path=tempfile.mktemp(suffix=".db"))
    await t.open()
    return t, SkillFactory(t, provider=provider, runtime=runtime)


async def seed_active(t, name, code, description, capabilities):
    from datetime import datetime
    from models import SkillMetadata, SkillStatus
    await t.upsert_skill(SkillMetadata(
        name=name, description=description, code=code, version=1,
        status=SkillStatus.ACTIVE, test_status="passed",
        capabilities=capabilities,
        created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
    ))


# ── Scenario A — existing skill is reused, builder NOT called ────────────────
async def scenario_a():
    p = FakeProvider()
    t, sf = await fresh_env(p)
    try:
        await seed_active(t, "count_words", GOOD_COUNT,
                          "count words in a text string",
                          ["count", "words", "text"])
        name, built = await sf.ensure_skill("count words in this text")
        assert name == "count_words" and built is False, (name, built)
        assert p.calls.get("generate", 0) == 0, "builder must NOT run"
        ok, out = await sf.invoke_skill(name, "sub-a", {"text": "one two three"})
        assert ok and out == "word_count=3", (ok, out)
        print("A) existing skill reused, builder not called          OK")
    finally:
        await t.close()


# ── Scenario B — no skill → build, test, register, execute ──────────────────
async def scenario_b():
    p = FakeProvider()
    t, sf = await fresh_env(p)
    try:
        name, built = await sf.ensure_skill(
            "convert celsius temperatures to fahrenheit")
        assert built is True and name == "celsius_to_fahrenheit", (name, built)
        skill = await t.get_skill(name)
        assert skill.status.value == "active" and skill.test_status == "passed"
        assert skill.model_used == "fake-cloud"
        ok, out = await sf.invoke_skill(name, "sub-b", {"celsius": 0})
        assert ok and out == "32.0F", (ok, out)
        print("B) no skill -> build -> test -> register -> execute   OK")
    finally:
        await t.close()


# ── Scenario C — first generation fails, repair loop fixes it ────────────────
async def scenario_c():
    p = FakeProvider(code=BAD_RAISES, repair=GOOD_TEMP)
    t, sf = await fresh_env(p)
    try:
        name, built = await sf.ensure_skill("convert celsius to fahrenheit")
        assert built and name, (name, built)
        assert p.calls.get("repair", 0) == 1, p.calls
        skill = await t.get_skill(name)
        assert skill.test_status == "passed"
        # repair attempts are capped by config
        from config import config
        assert p.calls["repair"] <= config.skill_repair_max_attempts
        print("C) broken generation -> repair -> verified            OK")
    finally:
        await t.close()


# ── Scenario D — reuse: second identical task builds NOTHING ─────────────────
async def scenario_d():
    p = FakeProvider()
    t, sf = await fresh_env(p)
    try:
        n1, built1 = await sf.ensure_skill("convert celsius temperatures to fahrenheit")
        assert built1 is True
        gen_calls_after_first = p.calls.get("generate", 0)
        n2, built2 = await sf.ensure_skill("convert celsius temperatures to fahrenheit")
        assert built2 is False and n2 == n1, (n1, n2, built2)
        assert p.calls.get("generate", 0) == gen_calls_after_first, \
            "Skill Builder call count on second run must be 0"
        print("D) second run reuses skill, builder call count = 0    OK")
    finally:
        await t.close()


# ── Scenario E — Local Only: cloud provider must never be touched ────────────
async def scenario_e():
    import model_router
    p = FakeProvider()
    t, sf = await fresh_env(p)
    from config import config
    config.llm_mode = "local"          # user policy: local only

    local_calls = {"n": 0}
    script = [SPEC_TEMP, GOOD_TEMP, '{"celsius": 100}', PERMS_NONE]

    async def fake_local(user_message, system, max_tokens):
        local_calls["n"] += 1
        return script.pop(0)

    orig = model_router._local_complete
    model_router._local_complete = fake_local
    try:
        name, built = await sf.ensure_skill("convert celsius temperatures to fahrenheit")
        assert built is True and name, (name, built)
        assert sum(p.calls.values()) == 0, f"cloud provider was called: {p.calls}"
        assert local_calls["n"] >= 2, "local model should have done the work"
        skill = await t.get_skill(name)
        assert skill.model_used == config.qwen_model
        print("E) Local Only: zero cloud calls, local model used     OK")
    finally:
        model_router._local_complete = orig
        config.llm_mode = "mixed"
        await t.close()


# ── Scenario F — permissions declared, propagated into the runtime ───────────
async def scenario_f():
    from skill_runtime import NativeRuntime

    class SpyRuntime(NativeRuntime):
        def __init__(self): self.seen_permissions = []
        async def execute(self, skill_name, code, input_data, context):
            self.seen_permissions.append(context.permissions)
            return await NativeRuntime.execute(self, skill_name, code,
                                               input_data, context)

    spy = SpyRuntime()
    p = FakeProvider(perms=PERMS_NET)
    t, sf = await fresh_env(p, runtime=spy)
    try:
        name, built = await sf.ensure_skill("convert celsius temperatures to fahrenheit")
        assert built is True
        skill = await t.get_skill(name)
        assert skill.permissions.network.domains == ["api.example.com"], \
            skill.permissions.model_dump()
        await sf.invoke_skill(name, "sub-f", {"celsius": 1})
        assert spy.seen_permissions[-1].network.domains == ["api.example.com"], \
            "runtime did not receive the declared permissions"
        print("F) permissions declared + propagated to runtime       OK")
    finally:
        await t.close()


# ── Scenario G — broken skill self-repairs; old version recoverable ──────────
async def scenario_g():
    p = FakeProvider(repair=GOOD_TEMP)
    t, sf = await fresh_env(p)
    try:
        await seed_active(t, "temp_conv", BAD_RAISES,
                          "convert celsius to fahrenheit",
                          ["temperature", "celsius", "fahrenheit"])
        ok, out = await sf.invoke_skill("temp_conv", "sub-g", {"celsius": 100})
        assert ok is False
        await asyncio.sleep(0.5)          # background self-repair
        repaired = await t.get_skill("temp_conv")
        assert repaired.version == 2 and "9 / 5" in repaired.code, repaired.version
        ok2, out2 = await sf.invoke_skill("temp_conv", "sub-g2", {"celsius": 100})
        assert ok2 and out2 == "212.0F", (ok2, out2)
        # old (broken) version is preserved in history and rollback works
        versions = await t.list_skill_versions("temp_conv")
        assert any(v["version"] == 1 and "boom" in v["code"] for v in versions), versions
        assert await t.rollback_skill("temp_conv", 1) is True
        rolled = await t.get_skill("temp_conv")
        assert rolled.version == 3 and "boom" in rolled.code
        print("G) runtime break -> self-repair v2 -> rollback works  OK")
    finally:
        await t.close()


async def main():
    for s in (scenario_a, scenario_b, scenario_c, scenario_d,
              scenario_e, scenario_f, scenario_g):
        await s()
    print("\nALL 7 ACCEPTANCE SCENARIOS PASS")

if __name__ == "__main__":
    asyncio.run(main())
