"""
OpenTeddy Skill Factory
Dynamically generates, tests, and registers new Python skills using
the configured cloud LLM (today: Claude via :class:`AnthropicProvider`,
later: any provider exposed by :func:`llm_provider.get_default_provider`).

Skills are sandboxed async functions that can be loaded at runtime.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import os
import textwrap
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from config import config
from llm_provider import (
    LLMProvider,
    LLMProviderError,
    get_default_provider,
)
from models import SkillInvocation, SkillMetadata, SkillStatus
from tracker import Tracker

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_GENERATION_SYSTEM = """\
You are SkillForge, an expert Python engineer embedded in OpenTeddy — a self-growing \
multi-agent system. Your job is to write a single async Python function called `run` \
that implements the requested skill.

Rules:
1. The function signature MUST be: async def run(input_data: dict) -> str
2. Use only standard library + these allowed packages: httpx, aiofiles, json, re, \
   datetime, pathlib, math, base64, hashlib, asyncio.
3. Return a human-readable string summarising what was done / the result.
4. Handle exceptions internally — never let the function raise.
5. Do NOT include import statements outside the function body.
6. Keep the function under {max_tokens} tokens.
7. Output ONLY the raw Python source code — no markdown fences, no explanations.
"""

_GENERATION_USER = """\
Skill name: {name}
Description: {description}
Example input keys: {input_keys}

Write the `run` async function now.
"""

_TEST_INPUT_USER = """\
Below is a Python skill for an agent system. Produce ONE realistic JSON \
object to pass as `input_data` to `run(input_data)` for a smoke test. \
Prefer inputs with no external side effects (use example.com for URLs, \
/tmp paths for files). Output ONLY the raw JSON object — no markdown, \
no explanations.

Skill name: {name}
Description: {description}

```python
{code}
```
"""

_REPAIR_USER = """\
The Python skill below FAILED. Fix it and return the complete corrected \
source. Keep the same contract: `async def run(input_data: dict) -> str`, \
same allowed packages, handle exceptions internally. Output ONLY the raw \
Python source code — no markdown fences, no explanations.

Skill name: {name}
Description: {description}

FAILING INPUT:
{failing_input}

ERROR:
{error}

CURRENT CODE:
```python
{code}
```
"""


class SkillFactory:
    """Creates, validates, stores, and loads skills dynamically."""

    def __init__(
        self,
        tracker: Tracker,
        provider: Optional[LLMProvider] = None,
    ) -> None:
        self.tracker = tracker
        # Provider is injectable for tests; production code uses the
        # default. Same pattern as :class:`EscalationAgent`.
        self.provider = provider or get_default_provider()
        self._loaded: Dict[str, Any] = {}   # name → callable
        # Names with a background self-repair in flight — prevents a skill
        # that fails repeatedly in quick succession from spawning parallel
        # regenerations of itself.
        self._regen_inflight: set = set()

    # ── Public API ────────────────────────────────────────────────────────────

    async def generate_skill(
        self,
        name: str,
        description: str,
        input_keys: Optional[list] = None,
    ) -> SkillMetadata:
        """Ask the LLM to write a new skill, TEST it, and persist to DB + disk.

        Loop A of the self-improvement design (test-before-register):
        syntax-checking alone let "compiles but doesn't work" skills into
        the library. Now the generated code must survive one real run
        against an LLM-suggested test input before it's stored. Failures
        feed the error back to the LLM for up to 2 repair attempts; if it
        still can't produce working code we raise rather than register a
        dud.

        A behaviour-tested skill registers as ACTIVE directly — the passed
        test IS the success evidence. (This also breaks the old deadlock:
        the executor only matches ACTIVE skills, but promotion to ACTIVE
        required successes that could only come from being matched — so
        TESTING skills sat unreachable forever.) With
        skill_test_before_register off, the old syntax-check-only path and
        TESTING status are preserved.
        """
        logger.info("Generating skill '%s'…", name)
        input_keys = input_keys or []
        test_on = bool(getattr(config, "skill_test_before_register", True))

        code = await self._generate_code(name, description, input_keys)
        max_repairs = 2
        last_err = ""
        test_input: Dict[str, Any] = {}
        tested = False

        for attempt in range(max_repairs + 1):
            valid, err = self._validate_syntax(code)
            if valid:
                if not test_on:
                    break  # legacy path: syntax check only
                if attempt == 0:
                    # One test input for all attempts — repairs should be
                    # judged against the same bar they failed.
                    test_input = await self._generate_test_input(
                        name, description, code,
                    )
                ok, err = await self._behavior_test(name, code, test_input)
                if ok:
                    tested = True
                    break
            last_err = err
            if attempt == max_repairs:
                raise ValueError(
                    f"Skill '{name}' failed generation after "
                    f"{max_repairs} repair attempts: {last_err}\n\nCode:\n{code}"
                )
            logger.info(
                "Skill '%s' failed %s (attempt %d) — asking LLM to repair: %s",
                name, "behaviour test" if valid else "syntax check",
                attempt + 1, err[:200],
            )
            code = await self._repair_code(
                name, description, code, err, test_input,
            )

        skill = SkillMetadata(
            name=name,
            description=description,
            code=code,
            version=1,
            status=SkillStatus.ACTIVE if tested else SkillStatus.TESTING,
            success_count=1 if tested else 0,   # the passed test
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        await self.tracker.upsert_skill(skill)
        self._write_skill_file(name, code)
        logger.info(
            "Skill '%s' saved (%s%s).",
            name, skill.status.value,
            ", behaviour-tested" if tested else "",
        )
        return skill

    async def invoke_skill(
        self,
        skill_name: str,
        subtask_id: str,
        input_data: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Run a skill by name. Returns (success, output)."""
        skill = await self.tracker.get_skill(skill_name)
        if not skill:
            return False, f"Skill '{skill_name}' not found."

        fn = await self._load_skill(skill_name, skill.code)
        start = time.monotonic()
        success = False
        output = ""
        try:
            output = await fn(input_data)
            success = True
        except Exception as exc:  # noqa: BLE001
            output = f"Skill error: {exc}\n{traceback.format_exc()}"
            logger.warning("Skill '%s' raised: %s", skill_name, exc)
        finally:
            elapsed_ms = int((time.monotonic() - start) * 1000)

        inv = SkillInvocation(
            skill_name=skill_name,
            subtask_id=subtask_id,
            input_data=input_data,
            output_data=output,
            success=success,
            duration_ms=elapsed_ms,
        )
        await self.tracker.record_skill_invocation(inv)
        await self.tracker.promote_skill_if_ready(skill_name)

        # Loop A, runtime half: a raise means the CODE is broken (skills
        # are contractually required to handle exceptions internally), so
        # self-repair in the background using the real failing input as
        # the regression test. Fire-and-forget — the caller already got
        # its (False, error) result and the executor is falling back to
        # the normal tool loop; repair must never block that.
        if not success:
            import asyncio
            asyncio.create_task(
                self._maybe_regenerate(skill, input_data, output)
            )
        return success, output

    async def list_active_skills(self) -> list[SkillMetadata]:
        return await self.tracker.list_skills(status=SkillStatus.ACTIVE)

    async def list_all_skills(self) -> list[SkillMetadata]:
        return await self.tracker.list_skills()

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _generate_code(
        self, name: str, description: str, input_keys: list
    ) -> str:
        if not self.provider.is_configured():
            # Re-raise as RuntimeError (preserved API contract — callers
            # of generate_skill catch RuntimeError specifically).
            raise RuntimeError(self.provider.get_unconfigured_message())
        system = _GENERATION_SYSTEM.format(max_tokens=config.max_skill_tokens)
        user = _GENERATION_USER.format(
            name=name,
            description=description,
            input_keys=", ".join(input_keys) if input_keys else "any",
        )
        try:
            response = await self.provider.complete_text(
                user_message=user,
                system=system,
                max_tokens=config.max_skill_tokens,
            )
        except LLMProviderError as exc:
            raise RuntimeError(f"Skill generation failed: {exc}") from exc

        raw = response.text
        # Strip accidental markdown fences
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                l for l in lines if not l.startswith("```")
            )
        return textwrap.dedent(raw)

    async def _generate_test_input(
        self, name: str, description: str, code: str,
    ) -> Dict[str, Any]:
        """Ask the LLM for one realistic input_data dict to smoke-test the
        skill. Any failure (LLM error, unparseable output) degrades to {}
        — an empty dict still exercises the code path, and rule 4 of the
        contract says the skill must handle bad input without raising."""
        import json as _json
        try:
            resp = await self.provider.complete_text(
                user_message=_TEST_INPUT_USER.format(
                    name=name, description=description, code=code[:4000],
                ),
                system=None,
                max_tokens=400,
            )
            raw = (resp.text or "").strip()
            if raw.startswith("```"):
                raw = "\n".join(
                    l for l in raw.splitlines() if not l.startswith("```")
                )
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                parsed = _json.loads(raw[start:end + 1])
                if isinstance(parsed, dict):
                    return parsed
        except Exception as exc:  # noqa: BLE001
            logger.debug("test-input generation degraded to {}: %s", exc)
        return {}

    async def _behavior_test(
        self, name: str, code: str, test_input: Dict[str, Any],
        timeout_s: float = 30.0,
    ) -> Tuple[bool, str]:
        """Actually run the candidate code once against test_input.

        Loaded into a throwaway module namespace — NOT written to
        skills_dir — so a broken candidate can never clobber a working
        version on disk. This is a behaviour test, not a security sandbox:
        the code runs in-process at the same trust level it would at
        invocation time, just earlier, where a failure is cheap.
        """
        import asyncio
        import types
        try:
            mod = types.ModuleType(f"skills_test.{name}")
            exec(compile(code, f"<skill-test:{name}>", "exec"), mod.__dict__)
            fn = getattr(mod, "run", None)
            if fn is None:
                return False, "no 'run' function after exec"
            out = await asyncio.wait_for(fn(dict(test_input)), timeout=timeout_s)
            if not isinstance(out, str):
                return False, (
                    f"run() must return str, got {type(out).__name__}"
                )
            return True, out
        except asyncio.TimeoutError:
            return False, f"behaviour test timed out after {timeout_s:.0f}s"
        except Exception as exc:  # noqa: BLE001
            return False, f"{exc}\n{traceback.format_exc()[-1500:]}"

    async def _repair_code(
        self, name: str, description: str, code: str,
        error: str, failing_input: Dict[str, Any],
    ) -> str:
        """Feed the failure back to the LLM and get a corrected version."""
        import json as _json
        system = _GENERATION_SYSTEM.format(max_tokens=config.max_skill_tokens)
        try:
            failing_repr = _json.dumps(
                failing_input, ensure_ascii=False, default=str,
            )[:800]
        except Exception:  # noqa: BLE001
            failing_repr = str(failing_input)[:800]
        response = await self.provider.complete_text(
            user_message=_REPAIR_USER.format(
                name=name,
                description=description,
                failing_input=failing_repr,
                error=error[:1200],
                code=code,
            ),
            system=system,
            max_tokens=config.max_skill_tokens,
        )
        raw = response.text
        if raw.startswith("```"):
            raw = "\n".join(
                l for l in raw.splitlines() if not l.startswith("```")
            )
        return textwrap.dedent(raw)

    async def _maybe_regenerate(
        self, skill: SkillMetadata, failing_input: Dict[str, Any],
        error_output: str,
    ) -> None:
        """Background self-repair after a runtime failure (Loop A).

        Repair → syntax check → re-test against the REAL input that just
        failed → store as version+1 and invalidate the loaded-function
        cache. At skill_regen_max_versions the skill is RETIRED instead:
        it stops being matched, so a hopeless skill can't burn an LLM call
        on every failure forever. All errors are swallowed — self-repair
        must never take down anything else.
        """
        name = skill.name
        if name in self._regen_inflight:
            return
        self._regen_inflight.add(name)
        try:
            if not self.provider.is_configured():
                return
            max_versions = int(
                getattr(config, "skill_regen_max_versions", 3) or 3
            )
            if skill.version >= max_versions:
                await self.tracker.set_skill_status(name, SkillStatus.RETIRED)
                logger.warning(
                    "Skill '%s' hit the self-repair cap (v%d) — RETIRED.",
                    name, skill.version,
                )
                return

            logger.info(
                "Skill '%s' v%d failed at runtime — attempting self-repair…",
                name, skill.version,
            )
            new_code = await self._repair_code(
                name, skill.description, skill.code,
                error_output, failing_input,
            )
            valid, err = self._validate_syntax(new_code)
            if not valid:
                logger.warning(
                    "Skill '%s' self-repair produced invalid code — kept "
                    "old version. (%s)", name, err[:200],
                )
                return
            ok, err = await self._behavior_test(name, new_code, failing_input)
            if not ok:
                logger.warning(
                    "Skill '%s' self-repair still fails its own regression "
                    "input — kept old version. (%s)", name, err[:200],
                )
                return

            repaired = skill.model_copy(update={
                "code":          new_code,
                "version":       skill.version + 1,
                "updated_at":    datetime.utcnow(),
                # Survived the exact input that broke the old version.
                "success_count": skill.success_count + 1,
            })
            await self.tracker.upsert_skill(repaired)
            self._write_skill_file(name, new_code)
            # Drop the cached callable so the next invocation loads the
            # repaired version instead of the stale in-memory one.
            self._loaded.pop(name, None)
            logger.info(
                "Skill '%s' self-repaired → v%d (regression input passes).",
                name, repaired.version,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Skill '%s' self-repair crashed (non-fatal): %s", name, exc,
            )
        finally:
            self._regen_inflight.discard(name)

    @staticmethod
    def _validate_syntax(code: str) -> Tuple[bool, str]:
        try:
            ast.parse(code)
            # Ensure there's an async def run
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef) and node.name == "run":
                    return True, ""
            return False, "No 'async def run(input_data)' found in generated code."
        except SyntaxError as exc:
            return False, str(exc)

    async def _load_skill(self, name: str, code: str) -> Any:
        """Dynamically load (or reload) a skill function from source."""
        if name in self._loaded:
            return self._loaded[name]

        skill_path = os.path.join(config.skills_dir, f"{name}.py")
        if not os.path.exists(skill_path):
            self._write_skill_file(name, code)

        spec = importlib.util.spec_from_file_location(f"skills.{name}", skill_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load skill module: {skill_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        fn = getattr(module, "run", None)
        if fn is None:
            raise AttributeError(f"Skill '{name}' has no 'run' function.")
        self._loaded[name] = fn
        return fn

    def _write_skill_file(self, name: str, code: str) -> None:
        os.makedirs(config.skills_dir, exist_ok=True)
        path = os.path.join(config.skills_dir, f"{name}.py")
        header = (
            f'"""\nAuto-generated skill: {name}\n'
            f'Created: {datetime.utcnow().isoformat()}\n"""\n\n'
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(header + code + "\n")
        logger.debug("Skill file written: %s", path)
