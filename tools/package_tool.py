"""
OpenTeddy Package Tool
Install Python packages into a dedicated venv (VENV_PATH env var).
HIGH risk — requires approval.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)


def _pip_executable() -> str:
    """Resolve pip inside VENV_PATH, falling back to current interpreter."""
    venv = os.environ.get("VENV_PATH", "")
    if venv:
        # Unix
        candidate = Path(venv) / "bin" / "pip"
        if candidate.exists():
            return str(candidate)
        # Windows
        candidate = Path(venv) / "Scripts" / "pip.exe"
        if candidate.exists():
            return str(candidate)
    # Fallback: use the same Python's pip module
    return f"{sys.executable} -m pip"


async def pip_install(package: str) -> Dict[str, Any]:
    """
    Install a Python package into the project venv.
    HIGH risk — requires approval.
    """
    start = time.monotonic()
    pip = _pip_executable()
    cmd = f"{pip} install {package}"
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=120.0)
        exit_code = proc.returncode or 0
        success = exit_code == 0
        return make_result(
            success,
            result={
                "package": package,
                "stdout": stdout_b.decode(errors="replace"),
                "stderr": stderr_b.decode(errors="replace"),
                "exit_code": exit_code,
            },
            error=None if success else f"pip install failed (exit {exit_code})",
            duration_ms=_ms(start),
        )
    except asyncio.TimeoutError:
        return make_result(False, error="pip install timed out (120s)",
                           duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=str(exc), duration_ms=_ms(start))


async def pip_list() -> Dict[str, Any]:
    """List installed packages in the venv. LOW risk."""
    start = time.monotonic()
    pip = _pip_executable()
    cmd = f"{pip} list --format=json"
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=30.0)
        import json
        packages = json.loads(stdout_b.decode(errors="replace"))
        return make_result(True, result={"packages": packages},
                           duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=str(exc), duration_ms=_ms(start))


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


# ── Schemas ───────────────────────────────────────────────────────────────────

_SCHEMA_INSTALL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "pip_install",
        "description": (
            "Install a Python package into the project virtual environment. "
            "Requires human approval. Isolated to VENV_PATH."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "package": {
                    "type": "string",
                    "description": "Package name (optionally with version, e.g. 'requests>=2.28').",
                },
            },
            "required": ["package"],
        },
    },
}

_SCHEMA_LIST: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "pip_list",
        "description": "List all installed Python packages in the project venv.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}

# ── Export ─────────────────────────────────────────────────────────────────────

PACKAGE_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    (pip_install, _SCHEMA_INSTALL, "high"),
    (pip_list,    _SCHEMA_LIST,    "low"),
]
