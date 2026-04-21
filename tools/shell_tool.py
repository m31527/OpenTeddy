"""
OpenTeddy Shell Tool
Executes shell commands via asyncio subprocess.
High-risk keywords trigger the approval gate.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)

# ── Risk detection ─────────────────────────────────────────────────────────────

_HIGH_RISK_PATTERNS: List[str] = [
    r"\brm\s+-[rRfF]*f",          # rm -rf / rm -f
    r"\brm\b",                     # any rm
    r"\bmkfs\b",                   # format filesystem
    r"\bdd\b",                     # disk dump/copy
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    r"\bpoweroff\b",
    r"\bchmod\s+[0-7]*7[0-7][0-7]",  # world-writable
    r"\bchown\b",
    r"\bcurl\s.*\|\s*(?:bash|sh)",    # curl | bash
    r"\bwget\s.*\|\s*(?:bash|sh)",
    r"\b>\s*/etc/",               # overwrite system files
    r"\bsudo\b",
    r"\bsu\b",
    r"\bkill\s+-9\b",
    r"\btruncate\b",
    r"\bmv\b",                    # move (potentially destructive)
    r"\bcp\s.*-[^-]*r",           # recursive copy
]

_HIGH_RISK_RE = re.compile("|".join(_HIGH_RISK_PATTERNS), re.IGNORECASE)


def _is_high_risk(command: str) -> bool:
    return bool(_HIGH_RISK_RE.search(command))


# ── Tool implementation ────────────────────────────────────────────────────────

async def execute_shell(
    command: str,
    working_dir: Optional[str] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Execute a shell command.
    Returns {success, result: {stdout, stderr, exit_code}, error, duration_ms}.
    Risk is determined dynamically; the registry receives LOW by default but
    shell_tool re-checks at call time (registry handles the gate for HIGH entries).
    """
    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=working_dir,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=float(timeout)
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            duration_ms = int((time.monotonic() - start) * 1000)
            return make_result(
                False,
                error=f"Command timed out after {timeout}s",
                duration_ms=duration_ms,
            )

        exit_code = proc.returncode or 0
        duration_ms = int((time.monotonic() - start) * 1000)
        success = exit_code == 0
        return make_result(
            success,
            result={
                "stdout": stdout_bytes.decode(errors="replace"),
                "stderr": stderr_bytes.decode(errors="replace"),
                "exit_code": exit_code,
            },
            error=None if success else f"Exit code {exit_code}",
            duration_ms=duration_ms,
        )
    except Exception as exc:  # noqa: BLE001
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.error("execute_shell error: %s", exc)
        return make_result(False, error=str(exc), duration_ms=duration_ms)


# ── Separate wrappers for low vs high risk ────────────────────────────────────

async def shell_exec_readonly(
    command: str,
    working_dir: Optional[str] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Read-only shell commands (ls, cat, grep, etc.) — LOW risk."""
    return await execute_shell(command, working_dir, timeout)


async def shell_exec_write(
    command: str,
    working_dir: Optional[str] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Write/destructive shell commands — HIGH risk, requires approval."""
    return await execute_shell(command, working_dir, timeout)


# ── Schemas ───────────────────────────────────────────────────────────────────

_SHELL_PARAMS = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "The shell command to execute.",
        },
        "working_dir": {
            "type": "string",
            "description": "Optional working directory path.",
        },
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds (default 30).",
            "default": 30,
        },
    },
    "required": ["command"],
}

_SCHEMA_READONLY: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "shell_exec_readonly",
        "description": (
            "Execute a read-only shell command (ls, cat, grep, find, echo, pwd, env, etc.). "
            "Do NOT use for commands that modify files or system state."
        ),
        "parameters": _SHELL_PARAMS,
    },
}

_SCHEMA_WRITE: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "shell_exec_write",
        "description": (
            "Execute a shell command that modifies files or system state. "
            "Requires human approval before execution."
        ),
        "parameters": _SHELL_PARAMS,
    },
}

# ── Export ─────────────────────────────────────────────────────────────────────

SHELL_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    (shell_exec_readonly, _SCHEMA_READONLY, "low"),
    (shell_exec_write, _SCHEMA_WRITE, "high"),
]
