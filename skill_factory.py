"""
OpenTeddy Skill Factory
Dynamically generates, tests, and registers new Python skills using
the configured cloud LLM (today: Claude via :class:`AnthropicProvider`,
later: any provider exposed by :func:`llm_provider.get_default_provider`).

Skills are sandboxed async functions that can be loaded at runtime.
"""

from __future__ import annotations

import ast
import logging
import os
import re
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
from models import SkillInvocation, SkillMetadata, SkillPermissions, SkillStatus
from skill_runtime import RuntimeContext, SkillRuntime, default_runtime
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

_SKILL_SPEC_USER = """\
A user asked an agent to do a task the agent has no existing skill for. \
Define a REUSABLE skill that would handle this task AND similar future \
requests (generalise: "convert these invoices" → an invoice-conversion \
skill, not a one-off). Output ONLY a raw JSON object:

{{"name": "<snake_case_short_name>",
  "description": "<one sentence, what the skill does>",
  "capabilities": ["<tag>", "..."],
  "input_keys": ["<expected input_data key>", "..."]}}

TASK: {task}
"""

_PERMISSIONS_USER = """\
Below is a Python skill. Declare the MINIMUM permissions it needs to run. \
Output ONLY a raw JSON object in exactly this shape (empty lists for \
anything unused):

{{"filesystem": {{"read": [], "write": []}},
  "network": {{"domains": []}},
  "commands": [], "credentials": [], "services": []}}

Rules: list network domains it contacts; list filesystem path PREFIXES it \
reads/writes; do NOT invent permissions the code does not use.

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
        runtime: Optional[SkillRuntime] = None,
    ) -> None:
        self.tracker = tracker
        # Provider is injectable for tests; production code uses the
        # default. Same pattern as :class:`EscalationAgent`.
        self.provider = provider or get_default_provider()
        # Execution backend — injectable (skill_runtime.SkillRuntime), so
        # generated-code execution can move to a sandboxed runtime later
        # without touching builder logic.
        self.runtime = runtime or default_runtime
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
        capabilities: Optional[list] = None,
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
        logger.info("skill.build.started skill=%s", name)
        input_keys = input_keys or []
        test_on = bool(getattr(config, "skill_test_before_register", True))

        code, model_used = await self._generate_code(name, description, input_keys)
        logger.info("skill.build.generated skill=%s model=%s", name, model_used)
        max_repairs = max(0, int(getattr(config, "skill_repair_max_attempts", 3)))
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
                logger.info("skill.test.started skill=%s attempt=%d", name, attempt + 1)
                ok, err = await self._behavior_test(name, code, test_input)
                if ok:
                    tested = True
                    logger.info("skill.test.passed skill=%s attempt=%d", name, attempt + 1)
                    break
                logger.info("skill.test.failed skill=%s attempt=%d", name, attempt + 1)
            last_err = err
            if attempt == max_repairs:
                raise ValueError(
                    f"Skill '{name}' failed generation after "
                    f"{max_repairs} repair attempts: {last_err}\n\nCode:\n{code}"
                )
            logger.info(
                "skill.repair.started skill=%s attempt=%d cause=%s", name,
                attempt + 1,
                ("behaviour" if valid else "syntax") + ":" + err[:120],
            )
            code = await self._repair_code(
                name, description, code, err, test_input,
            )
            logger.info("skill.repair.completed skill=%s attempt=%d", name, attempt + 1)

        # Least-privilege declaration for the final code (best-effort —
        # unparseable/missing → empty declaration, never blocks saving).
        permissions = await self._declare_permissions(code) if tested \
            else SkillPermissions()

        skill = SkillMetadata(
            name=name,
            description=description,
            code=code,
            version=1,
            status=SkillStatus.ACTIVE if tested else SkillStatus.TESTING,
            success_count=1 if tested else 0,   # the passed test
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            capabilities=capabilities or [],
            input_keys=list(input_keys),
            permissions=permissions,
            source_type="generated",
            model_used=model_used,
            test_status="passed" if tested else "untested",
        )
        await self.tracker.upsert_skill(skill)
        await self.tracker.save_skill_version(
            name, 1, code, note=f"initial ({'tested' if tested else 'untested'})",
        )
        self._write_skill_file(name, code)
        logger.info(
            "skill.registered skill=%s status=%s tested=%s model=%s",
            name, skill.status.value, tested, model_used,
        )
        return skill

    async def invoke_skill(
        self,
        skill_name: str,
        subtask_id: str,
        input_data: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Run a skill by name. Returns (success, output).

        Execution goes through the pluggable SkillRuntime with the
        skill's DECLARED PERMISSIONS in the runtime context (propagation
        per skill-plus §11–12). Executing straight from the DB row also
        removes the old file/module cache, whose stale entries could keep
        running pre-repair code after a version bump.
        """
        skill = await self.tracker.get_skill(skill_name)
        if not skill:
            return False, f"Skill '{skill_name}' not found."
        if not getattr(skill, "enabled", True):
            return False, f"Skill '{skill_name}' is disabled."

        from config import effective_workspace_dir
        try:
            ws = effective_workspace_dir()
        except Exception:  # noqa: BLE001
            ws = ""
        ctx = RuntimeContext(
            permissions=skill.permissions,
            timeout_s=60.0,
            workspace_dir=ws,
        )
        result = await self.runtime.execute(skill_name, skill.code, input_data, ctx)
        success = result.success
        output = result.output if success else f"Skill error: {result.error}"
        elapsed_ms = result.duration_ms

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

    async def ensure_skill(
        self,
        task_text: str,
        skill_hint: Optional[str] = None,
    ) -> Tuple[Optional[str], bool]:
        """The learn-or-reuse entry point (skill-plus §2 core loop).

        Search existing skills first; only when nothing matches, define a
        reusable skill for the task (spec stage), build, test, repair and
        register it. Returns (skill_name, was_built) — (None, False) when
        no match exists and building failed / produced nothing usable.

        "I know how → do it. I don't know how → learn it, test it,
        remember it. I learned it before → reuse it."
        """
        import json as _json
        import skill_matcher

        skills = await self.tracker.list_skills(status=SkillStatus.ACTIVE)
        threshold = float(getattr(config, "skill_match_threshold", 0.4) or 0.4)
        m = skill_matcher.match(task_text, skills, threshold, skill_hint=skill_hint)
        if m.matched:
            return m.skill_name, False

        # No existing capability → define a REUSABLE skill spec for the
        # task (name/description/capabilities/input_keys), then build it.
        try:
            spec_raw, _model = await self._complete(
                "skill_generation",
                _SKILL_SPEC_USER.format(task=task_text[:800]),
                system=None, max_tokens=400,
            )
            start, end = spec_raw.find("{"), spec_raw.rfind("}")
            spec = _json.loads(spec_raw[start:end + 1]) if start >= 0 else {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("ensure_skill: spec stage failed (%s) — not building", exc)
            return None, False

        name = re.sub(r"[^a-z0-9_]", "", str(spec.get("name", "")).lower().replace("-", "_"))
        if not name:
            return None, False
        # Name collision with a non-matching existing skill → suffix.
        if any(s.name == name for s in skills):
            name = f"{name}_v2"
        try:
            skill = await self.generate_skill(
                name,
                str(spec.get("description") or task_text[:200]),
                input_keys=list(spec.get("input_keys") or []),
                capabilities=list(spec.get("capabilities") or []),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ensure_skill: build failed for '%s': %s", name, exc)
            return None, False
        return (skill.name, True) if skill.status == SkillStatus.ACTIVE else (None, False)

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _complete(
        self, task_kind: str, user_message: str,
        system: Optional[str], max_tokens: int,
    ) -> Tuple[str, str]:
        """Route one text completion through the model router.

        Returns (text, model_label). Strategy honours user policy: in
        Local-Only mode the returned completer never touches the cloud
        provider object (enforcement by construction — see model_router);
        otherwise skill work prefers the strong model.
        """
        import model_router
        strategy = model_router.resolve_strategy(task_kind)
        if (strategy is not model_router.ModelStrategy.LOCAL_ONLY
                and not self.provider.is_configured()):
            # Preserved API contract: outside Local-Only, skill building
            # requires a configured cloud provider (a 2B local model
            # writing skills produces garbage; better to say so).
            raise RuntimeError(self.provider.get_unconfigured_message())
        completer, model_label = model_router.completer_for(strategy, self.provider)
        try:
            text = await completer(user_message, system, max_tokens)
        except LLMProviderError as exc:
            raise RuntimeError(f"Skill generation failed: {exc}") from exc
        return text, model_label

    async def _generate_code(
        self, name: str, description: str, input_keys: list
    ) -> Tuple[str, str]:
        """Returns (code, model_label)."""
        system = _GENERATION_SYSTEM.format(max_tokens=config.max_skill_tokens)
        user = _GENERATION_USER.format(
            name=name,
            description=description,
            input_keys=", ".join(input_keys) if input_keys else "any",
        )
        raw, model_label = await self._complete(
            "skill_generation", user, system, config.max_skill_tokens,
        )
        # Strip accidental markdown fences
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                l for l in lines if not l.startswith("```")
            )
        return textwrap.dedent(raw), model_label

    async def _declare_permissions(self, code: str) -> SkillPermissions:
        """Ask the LLM for the skill's least-privilege declaration.
        Best-effort: any failure returns an EMPTY declaration (deny-by-
        default posture — nothing is granted that wasn't declared)."""
        import json as _json
        try:
            raw, _model = await self._complete(
                "skill_generation",
                _PERMISSIONS_USER.format(code=code[:4000]),
                system=None, max_tokens=300,
            )
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                data = _json.loads(raw[start:end + 1])
                if isinstance(data, dict):
                    return SkillPermissions(**data)
        except Exception as exc:  # noqa: BLE001
            logger.debug("permission declaration degraded to empty: %s", exc)
        return SkillPermissions()

    async def _generate_test_input(
        self, name: str, description: str, code: str,
    ) -> Dict[str, Any]:
        """Ask the LLM for one realistic input_data dict to smoke-test the
        skill. Any failure (LLM error, unparseable output) degrades to {}
        — an empty dict still exercises the code path, and rule 4 of the
        contract says the skill must handle bad input without raising."""
        import json as _json
        try:
            raw, _model = await self._complete(
                "skill_test_generation",
                _TEST_INPUT_USER.format(
                    name=name, description=description, code=code[:4000],
                ),
                system=None, max_tokens=400,
            )
            raw = raw.strip()
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
        """Run the candidate code once against test_input via the runtime.

        Candidate code is executed from source — NOT written to
        skills_dir — so a broken candidate can never clobber a working
        version on disk. This is a behaviour test, not a security sandbox
        (see skill_runtime.NativeRuntime): failures are just cheaper here
        than at real invocation time.
        """
        result = await self.runtime.execute(
            f"{name}__candidate", code, dict(test_input),
            RuntimeContext(timeout_s=timeout_s),
        )
        return result.success, (result.output if result.success else result.error)

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
        raw, _model = await self._complete(
            "skill_repair",
            _REPAIR_USER.format(
                name=name,
                description=description,
                failing_input=failing_repr,
                error=error[:1200],
                code=code,
            ),
            system=system,
            max_tokens=config.max_skill_tokens,
        )
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
            # No pre-flight provider check: _complete routes by policy —
            # in Local-Only mode repair runs on the local model; outside
            # it, an unconfigured provider raises and lands in the
            # swallow-all handler below (repair is best-effort anyway).
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

            # Snapshot the OLD version before overwriting — the last
            # known-working code stays recoverable via rollback_skill.
            await self.tracker.save_skill_version(
                name, skill.version, skill.code,
                note=f"pre-repair snapshot (runtime failure: {error_output[:120]})",
            )
            repaired = skill.model_copy(update={
                "code":          new_code,
                "version":       skill.version + 1,
                "updated_at":    datetime.utcnow(),
                # Survived the exact input that broke the old version.
                "success_count": skill.success_count + 1,
                "test_status":   "passed",
            })
            await self.tracker.upsert_skill(repaired)
            await self.tracker.save_skill_version(
                name, repaired.version, new_code, note="self-repair result",
            )
            self._write_skill_file(name, new_code)
            logger.info(
                "skill.repair.completed skill=%s new_version=%d "
                "(regression input passes)", name, repaired.version,
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
