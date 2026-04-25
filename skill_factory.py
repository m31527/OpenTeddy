"""
OpenTeddy Skill Factory
Dynamically generates, tests, and registers new Python skills using Claude.
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

import anthropic

from config import config
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


class SkillFactory:
    """Creates, validates, stores, and loads skills dynamically."""

    def __init__(self, tracker: Tracker) -> None:
        self.tracker = tracker
        self._claude_key: str | None = None
        self._claude: anthropic.AsyncAnthropic | None = None
        self._loaded: Dict[str, Any] = {}   # name → callable

    @property
    def _client(self) -> anthropic.AsyncAnthropic:
        # Same pattern as EscalationAgent: rebuild the Anthropic client
        # whenever config.anthropic_api_key changes via the settings UI.
        # Pass None (not "") when no key is configured — see escalation.py
        # for why ("" makes the SDK throw before a fallback is attempted).
        key = config.anthropic_api_key or None
        if self._claude is None or self._claude_key != key:
            self._claude = anthropic.AsyncAnthropic(api_key=key)
            self._claude_key = key
        return self._claude

    # ── Public API ────────────────────────────────────────────────────────────

    async def generate_skill(
        self,
        name: str,
        description: str,
        input_keys: Optional[list] = None,
    ) -> SkillMetadata:
        """Ask Claude to write a new skill, validate it, and persist to DB + disk."""
        logger.info("Generating skill '%s'…", name)
        input_keys = input_keys or []

        # 1. Generate code via Claude
        code = await self._generate_code(name, description, input_keys)

        # 2. Syntax-check the generated code
        valid, err = self._validate_syntax(code)
        if not valid:
            raise ValueError(f"Skill '{name}' syntax error: {err}\n\nCode:\n{code}")

        # 3. Persist skill (status=TESTING)
        skill = SkillMetadata(
            name=name,
            description=description,
            code=code,
            version=1,
            status=SkillStatus.TESTING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )
        await self.tracker.upsert_skill(skill)
        self._write_skill_file(name, code)
        logger.info("Skill '%s' saved (TESTING).", name)
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
        return success, output

    async def list_active_skills(self) -> list[SkillMetadata]:
        return await self.tracker.list_skills(status=SkillStatus.ACTIVE)

    async def list_all_skills(self) -> list[SkillMetadata]:
        return await self.tracker.list_skills()

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _generate_code(
        self, name: str, description: str, input_keys: list
    ) -> str:
        if not (config.anthropic_api_key or "").strip():
            raise RuntimeError(
                "Claude API key is not configured. Set one in Settings → "
                "Model Settings → Claude API Key, then retry skill generation."
            )
        system = _GENERATION_SYSTEM.format(max_tokens=config.max_skill_tokens)
        user = _GENERATION_USER.format(
            name=name,
            description=description,
            input_keys=", ".join(input_keys) if input_keys else "any",
        )
        message = await self._client.messages.create(
            model=config.claude_model,
            max_tokens=config.max_skill_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = message.content[0].text.strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                l for l in lines if not l.startswith("```")
            )
        return textwrap.dedent(raw)

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
