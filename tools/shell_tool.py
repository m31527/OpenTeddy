"""
OpenTeddy Shell Tool
Executes shell commands via asyncio subprocess.
High-risk keywords trigger the approval gate.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)

# ── Output truncation limits ───────────────────────────────────────────────────
MAX_OUTPUT_LINES = 200
MAX_OUTPUT_CHARS = 8000

# ── Docker-specific timeouts (seconds) ────────────────────────────────────────
_DOCKER_TIMEOUTS: List[Tuple[str, int]] = [
    ("docker compose up",    180),
    ("docker-compose up",    180),
    ("docker compose build", 300),
    ("docker-compose build", 300),
    ("docker build",         300),
]

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


def _sanitize_command(cmd: str) -> str:
    """Rewrite known problematic commands to be safe for non-interactive execution.

    - ``docker compose logs`` without ``--tail`` → add ``--tail=50 --no-color``
      (prevents infinite blocking when a container is still running)
    """
    if ("docker compose logs" in cmd or "docker-compose logs" in cmd):
        if "--tail" not in cmd:
            cmd = cmd.replace("docker compose logs", "docker compose logs --tail=50 --no-color")
            cmd = cmd.replace("docker-compose logs", "docker-compose logs --tail=50 --no-color")
    return cmd


# ── Docker Compose context check ──────────────────────────────────────────────

_COMPOSE_FILENAMES = (
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
)
_COMPOSE_FILE_FLAG_RE = re.compile(r"(?:^|\s)(?:-f|--file)(?:\s+|=)\S+")
_COMPOSE_CD_RE = re.compile(r"\bcd\s+([^\s&|;]+)")


def _docker_compose_context_note(
    command: str, working_dir: Optional[str],
) -> Optional[str]:
    """If a ``docker compose`` command is run without ``-f``, resolve which
    compose file will actually be used and return a short note. Helps surface
    the classic "wrong directory" bug where the CLI silently falls back to the
    shell's CWD compose file.

    Returns ``None`` for non-compose commands or when ``-f`` was given.
    """
    if "docker compose" not in command and "docker-compose" not in command:
        return None
    if _COMPOSE_FILE_FLAG_RE.search(command):
        return None

    # Prefer a cd target in the command itself, then working_dir, then CWD.
    cd_match = _COMPOSE_CD_RE.search(command)
    if cd_match:
        resolved_dir = os.path.expanduser(cd_match.group(1))
        source = "cd in command"
    elif working_dir:
        resolved_dir = working_dir
        source = "working_dir arg"
    else:
        resolved_dir = os.getcwd()
        source = "process cwd"

    for name in _COMPOSE_FILENAMES:
        path = os.path.join(resolved_dir, name)
        if os.path.isfile(path):
            return (
                f"[shell_tool] docker compose will use: {path} "
                f"(resolved via {source}). If this is the wrong file, pass "
                f"-f <path> or cd to the correct directory first."
            )

    return (
        f"[shell_tool] WARNING: no compose file found in {resolved_dir} "
        f"({source}). docker compose will fail or fall back to a parent dir."
    )


def _docker_timeout(cmd: str, default: int) -> int:
    """Return a command-specific timeout for known long-running Docker operations."""
    for pattern, t in _DOCKER_TIMEOUTS:
        if pattern in cmd:
            return t
    return default


def _truncate_output(text: str) -> str:
    """Truncate stdout/stderr to avoid overwhelming the model context."""
    lines = text.split("\n")
    if len(lines) > MAX_OUTPUT_LINES:
        text = "\n".join(lines[:MAX_OUTPUT_LINES]) + (
            f"\n... [截斷，共 {len(lines)} 行，只顯示前 {MAX_OUTPUT_LINES} 行]"
        )
    if len(text) > MAX_OUTPUT_CHARS:
        text = text[:MAX_OUTPUT_CHARS] + (
            f"\n... [截斷，共 {len(text)} 字元，只顯示前 {MAX_OUTPUT_CHARS} 字元]"
        )
    return text


# ── Tool implementation ────────────────────────────────────────────────────────

def _resolve_working_dir(working_dir: Optional[str]) -> str:
    """Pick the effective working directory for this command.

    Priority:
      1. An explicit ``working_dir`` arg from the LLM (absolute or relative)
      2. ``config.agent_workspace_dir`` — the project-wide default
      3. Falls back to the current process CWD if even that fails

    The directory is created on-demand so a freshly cloned project doesn't
    have to ``mkdir`` it manually. Returns an absolute path.
    """
    # Late import to avoid a circular dep when tool_registry imports this
    # module during early module initialisation.
    from config import config as _cfg
    chosen = working_dir or getattr(_cfg, "agent_workspace_dir", None) or os.getcwd()
    chosen_abs = os.path.abspath(chosen)
    try:
        os.makedirs(chosen_abs, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not ensure working dir %s exists: %s", chosen_abs, exc)
    return chosen_abs


async def _drain_with_silence_timeout(
    proc: asyncio.subprocess.Process,
    silence_timeout: float,
    wall_timeout: float,
) -> Tuple[bytes, bytes, str]:
    """Read stdout + stderr concurrently, kill the process if nothing arrives
    for ``silence_timeout`` seconds, AND cap total runtime at ``wall_timeout``.

    Returns (stdout_bytes, stderr_bytes, reason) where reason is one of:
      "exit"              — process exited normally
      "silence_timeout"   — idle too long, killed
      "wall_timeout"      — exceeded absolute ceiling, killed

    The whole point of this function is that `docker compose up --build`
    can legitimately run for 10+ minutes while actively printing layer
    progress. A fixed wall-clock timeout misjudges that as "hung". Here
    each chunk of output resets the silence clock; a truly stuck command
    (DNS failure, interactive prompt, lock contention with no logging)
    still gets caught within ``silence_timeout`` seconds.
    """
    stdout_buf = bytearray()
    stderr_buf = bytearray()
    start = time.monotonic()
    last_activity = start

    async def _reader(stream: asyncio.StreamReader, buf: bytearray) -> None:
        while True:
            chunk = await stream.read(4096)
            if not chunk:
                return
            buf.extend(chunk)
            nonlocal last_activity
            last_activity = time.monotonic()

    assert proc.stdout is not None and proc.stderr is not None
    stdout_task = asyncio.create_task(_reader(proc.stdout, stdout_buf))
    stderr_task = asyncio.create_task(_reader(proc.stderr, stderr_buf))
    wait_task   = asyncio.create_task(proc.wait())

    try:
        while True:
            # Poll at most every `silence_timeout`; this loop exits on one of
            # three conditions: exit, silence exceeded, wall-clock exceeded.
            idle_for = time.monotonic() - last_activity
            remaining_silence = max(0.5, silence_timeout - idle_for) if silence_timeout > 0 else 999999
            elapsed = time.monotonic() - start
            remaining_wall = max(0.5, wall_timeout - elapsed) if wall_timeout > 0 else 999999
            sleep_for = min(remaining_silence, remaining_wall)

            done, _ = await asyncio.wait(
                {wait_task}, timeout=sleep_for,
            )
            if wait_task in done:
                # Process has exited — drain any tail output.
                # Readers will hit EOF shortly.
                await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
                return bytes(stdout_buf), bytes(stderr_buf), "exit"

            # Woke up on timeout. Which limit tripped?
            now = time.monotonic()
            if silence_timeout > 0 and (now - last_activity) >= silence_timeout:
                return bytes(stdout_buf), bytes(stderr_buf), "silence_timeout"
            if wall_timeout > 0 and (now - start) >= wall_timeout:
                return bytes(stdout_buf), bytes(stderr_buf), "wall_timeout"
            # Otherwise: spurious wake, loop
    finally:
        # Clean up readers — they'll auto-exit on EOF but we belt-and-braces.
        for t in (stdout_task, stderr_task, wait_task):
            if not t.done():
                t.cancel()


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

    Applies five safety measures automatically:
    1. ``_resolve_working_dir`` defaults the cwd to ``agent_workspace_dir`` when
       the LLM didn't supply one, so git clones etc. land in a known place.
    2. ``_sanitize_command`` rewrites dangerous/blocking variants (e.g. docker logs).
    3. ``_docker_timeout`` overrides the timeout for known long-running Docker ops.
    4. **Silence timeout** (``config.shell_silence_timeout``, default 90s) —
       command is killed only if it produces no output for that many seconds.
       This lets long-but-active commands (docker build, pip install) run to
       completion while still catching real hangs.
    5. ``_truncate_output`` caps stdout/stderr to avoid context overflow.

    Cleanup: the subprocess is ALWAYS killed if we exit this function without
    a clean exit — including when our caller cancels us via asyncio.CancelledError.
    No more zombie `docker build` processes chewing CPU in the background.
    """
    command = _sanitize_command(command)
    effective_timeout = _docker_timeout(command, timeout)
    if effective_timeout != timeout:
        logger.info(
            "execute_shell: overriding timeout %ds → %ds for command: %s",
            timeout, effective_timeout, command[:80],
        )

    effective_dir = _resolve_working_dir(working_dir)
    if not working_dir:
        logger.info(
            "execute_shell: no working_dir supplied, defaulting to %s",
            effective_dir,
        )

    # Late import of config so tests can override it without module reload.
    from config import config as _cfg
    silence_timeout = float(getattr(_cfg, "shell_silence_timeout", 90) or 0)

    compose_note = _docker_compose_context_note(command, effective_dir)
    if compose_note:
        logger.info("%s", compose_note)

    start = time.monotonic()
    proc: Optional[asyncio.subprocess.Process] = None
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=effective_dir,
        )
        stdout_bytes, stderr_bytes, reason = await _drain_with_silence_timeout(
            proc,
            silence_timeout=silence_timeout,
            wall_timeout=float(effective_timeout),
        )

        if reason in ("silence_timeout", "wall_timeout"):
            # Kill the straggler before returning — otherwise docker build
            # keeps running orphaned, burning CPU and confusing the user's
            # next attempt. `proc.wait()` with a short timeout gives the
            # OS a moment to reap.
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                pass

            duration_ms = int((time.monotonic() - start) * 1000)
            idle_msg = (
                f"no output for {silence_timeout}s (silence timeout — looks hung)"
                if reason == "silence_timeout"
                else f"exceeded wall-clock limit of {effective_timeout}s"
            )
            return make_result(
                False,
                result={
                    "stdout": _truncate_output(stdout_bytes.decode(errors="replace")),
                    "stderr": _truncate_output(stderr_bytes.decode(errors="replace")),
                    "exit_code": -1,
                },
                error=f"Command killed — {idle_msg}",
                duration_ms=duration_ms,
            )

        exit_code = proc.returncode or 0
        duration_ms = int((time.monotonic() - start) * 1000)
        success = exit_code == 0

        stdout = _truncate_output(stdout_bytes.decode(errors="replace"))
        stderr = _truncate_output(stderr_bytes.decode(errors="replace"))

        if compose_note:
            stderr = f"{compose_note}\n{stderr}" if stderr else compose_note

        return make_result(
            success,
            result={
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
            },
            error=None if success else f"Exit code {exit_code}",
            duration_ms=duration_ms,
        )
    except asyncio.CancelledError:
        # The orchestrator cancelled us (outer subtask_timeout, user hit Stop,
        # or upstream escalation). Kill the subprocess so we don't leak a
        # runaway docker build into the background.
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                pass
        raise
    except Exception as exc:  # noqa: BLE001
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
                await asyncio.wait_for(proc.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                pass
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
