"""
OpenTeddy skill runtime abstraction.

Decouples "run this skill's code against this input" from the Skill
Builder / registry, so the execution backend can be swapped without
touching skill generation logic (skill-plus spec §12–13):

    NativeRuntime      in-process execution (today's behaviour, extracted)
    DockerRuntime      future: containerised execution
    OpenShellRuntime   future: NVIDIA OpenShell secure backend

The RuntimeContext carries the skill's declared permissions into the
runtime. v1 propagates + surfaces them (the module sees its own
declaration, the runtime logs it); actual OS-level enforcement is what
the future sandboxed runtimes are for. NativeRuntime is a behaviour
boundary, NOT a security sandbox — generated code runs at the same trust
level it always has, just behind a uniform interface.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from models import SkillPermissions

logger = logging.getLogger(__name__)


@dataclass
class RuntimeContext:
    """Everything a runtime needs beyond the code + input."""
    permissions: SkillPermissions = field(default_factory=SkillPermissions)
    timeout_s: float = 30.0
    workspace_dir: str = ""


@dataclass
class SkillExecutionResult:
    success: bool
    output: str = ""
    error: str = ""
    duration_ms: int = 0


class SkillRuntime(ABC):
    """Interface every execution backend implements."""

    @abstractmethod
    async def execute(
        self, skill_name: str, code: str,
        input_data: Dict[str, Any], context: RuntimeContext,
    ) -> SkillExecutionResult:
        ...


class NativeRuntime(SkillRuntime):
    """In-process execution — the historical OpenTeddy behaviour.

    The code is exec'd into a throwaway module namespace (never imported
    from disk here), `run(input_data)` is awaited under the context's
    timeout, and the declared permissions are exposed to the module as
    ``__skill_permissions__`` so both the skill and any inspection
    tooling can see exactly what was granted.
    """

    async def execute(
        self, skill_name: str, code: str,
        input_data: Dict[str, Any], context: RuntimeContext,
    ) -> SkillExecutionResult:
        start = time.monotonic()
        perms = context.permissions
        if not perms.is_empty():
            logger.info(
                "skill.execution.started skill=%s permissions=%s",
                skill_name, perms.model_dump_json(),
            )
        else:
            logger.info("skill.execution.started skill=%s permissions=(none declared)",
                        skill_name)
        try:
            mod = types.ModuleType(f"skills_runtime.{skill_name}")
            mod.__skill_permissions__ = perms.model_dump()
            exec(compile(code, f"<skill:{skill_name}>", "exec"), mod.__dict__)
            fn = getattr(mod, "run", None)
            if fn is None:
                raise AttributeError("no 'run' function in skill code")
            out = await asyncio.wait_for(
                fn(dict(input_data)), timeout=context.timeout_s,
            )
            if not isinstance(out, str):
                raise TypeError(
                    f"run() must return str, got {type(out).__name__}"
                )
            dur = int((time.monotonic() - start) * 1000)
            logger.info("skill.execution.completed skill=%s duration_ms=%d",
                        skill_name, dur)
            return SkillExecutionResult(True, output=out, duration_ms=dur)
        except asyncio.TimeoutError:
            dur = int((time.monotonic() - start) * 1000)
            logger.warning("skill.execution.failed skill=%s reason=timeout(%.0fs)",
                           skill_name, context.timeout_s)
            return SkillExecutionResult(
                False, error=f"timed out after {context.timeout_s:.0f}s",
                duration_ms=dur,
            )
        except Exception as exc:  # noqa: BLE001
            dur = int((time.monotonic() - start) * 1000)
            logger.warning("skill.execution.failed skill=%s error=%s",
                           skill_name, exc)
            return SkillExecutionResult(
                False,
                error=f"{exc}\n{traceback.format_exc()[-1500:]}",
                duration_ms=dur,
            )


# Module-level default so callers that don't need a custom backend share
# one instance. Injectable everywhere it's consumed (SkillFactory takes a
# runtime parameter) — tests swap in spies here.
default_runtime: SkillRuntime = NativeRuntime()
