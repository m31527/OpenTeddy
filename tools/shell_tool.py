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


def _fix_duplicate_workspace_prefix(
    cmd: str, effective_dir: str,
) -> Tuple[str, Optional[str]]:
    """Strip a duplicate ``agent-workspace/`` prefix from a leading ``cd``.

    Bug this fixes: Gemma sometimes plans commands like
        ``cd ./agent-workspace/Pixelle-Video && docker compose up``
    because ``docker_project_detect`` accepts that shape (it resolves via
    Python's os.path.abspath, which is relative to the uvicorn process
    cwd). But shell_exec_* runs in ``agent_workspace_dir`` already, so
    the shell sees ``cd ./agent-workspace/agent-workspace/...`` and the
    `cd` silently fails, breaking the `&&` chain. The user then sees
    "deploy completed 100%" while nothing actually started.

    If the subprocess cwd ends in (e.g.) ``/agent-workspace`` AND the
    command starts with ``cd ./agent-workspace/X`` or ``cd agent-workspace/X``,
    rewrite to ``cd X``. Absolute paths are left alone.

    Returns (fixed_cmd, reason_or_None). reason is a human-readable
    note when a rewrite happened, for logging.
    """
    if not effective_dir:
        return cmd, None
    ws_basename = os.path.basename(os.path.abspath(effective_dir).rstrip("/"))
    if not ws_basename:
        return cmd, None

    # Match "cd agent-workspace/X..." or "cd ./agent-workspace/X..." at start.
    # The `(?:\./)?` prevents matching "cd /agent-workspace/X" (absolute).
    pattern = rf"^cd\s+(?:\./)?({re.escape(ws_basename)}/)"
    m = re.match(pattern, cmd)
    if not m:
        return cmd, None

    new_cmd = re.sub(pattern, "cd ", cmd, count=1)
    reason = (
        f"auto-stripped duplicate '{m.group(1)}' prefix — shell cwd is "
        f"already '{effective_dir}'. Was: `{cmd[:80]}` → `{new_cmd[:80]}`"
    )
    return new_cmd, reason


_DEPLOY_SUCCESS_EMPTY_RE = re.compile(
    r"\bdocker\s+compose\s+(?:up|ps)\b", re.IGNORECASE,
)

_CD_FAILURE_RE = re.compile(
    r"(?:cd: can'?t cd to |cd: no such file|no such file or directory)",
    re.IGNORECASE,
)


def _build_cwd_diagnostic(effective_dir: str) -> str:
    """Render a short block showing the subprocess's cwd and its contents.

    Appended to stderr on failures that smell like a cwd mismatch
    (cd errors, "no such file or directory"). Lets Qwen — and the
    human reading the UI — see at a glance WHERE the shell actually
    was, versus where the LLM thought it was planning from. This is
    the single most effective debug aid for path-resolution bugs
    because the alternative is staring at "can't cd" with zero
    clues about the current working directory.
    """
    lines: List[str] = [
        "",
        "[OpenTeddy cwd diagnostic]",
        f"  effective cwd: {effective_dir}",
    ]
    try:
        if os.path.isdir(effective_dir):
            entries = sorted(os.listdir(effective_dir))
            preview = "  ".join(entries[:20])
            more = f"  (+{len(entries) - 20} more)" if len(entries) > 20 else ""
            lines.append(f"  contents ({len(entries)}): {preview}{more}")
        else:
            lines.append("  (this directory does NOT exist on disk)")
    except Exception as exc:  # noqa: BLE001
        lines.append(f"  (could not list: {exc})")
    try:
        from config import config as _cfg
        ws = getattr(_cfg, "agent_workspace_dir", None)
        if ws and os.path.abspath(ws) != effective_dir:
            lines.append(
                f"  note: agent_workspace_dir is {ws} — if clones went "
                "there but this cwd is different, that's the bug. "
                "Check /debug/workspace for the authoritative path."
            )
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


def _openteddy_project_root() -> str:
    """Absolute path of OpenTeddy's own source tree (parent of tools/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _is_openteddy_source_path(path: str) -> bool:
    """True when ``path`` is inside OpenTeddy's own source tree BUT NOT
    inside an ``agent-workspace/`` subdir.

    Protects against the "Qwen passes working_dir='/path/to/OpenTeddy'
    and runs `docker compose up --build` there" failure mode, which
    could clobber the running uvicorn / DB / skills directory. The
    ``agent-workspace/`` subdir is explicitly allowed since that's the
    legitimate sandbox.
    """
    try:
        abs_path = os.path.abspath(path)
    except Exception:  # noqa: BLE001
        return False
    root = _openteddy_project_root()
    ws   = os.path.join(root, "agent-workspace")
    if abs_path == root:
        return True
    if abs_path.startswith(root + os.sep):
        # Inside project root — allowed only if under agent-workspace/
        return not (abs_path == ws or abs_path.startswith(ws + os.sep))
    return False


def _looks_like_empty_compose_result(command: str, stdout: str, stderr: str) -> bool:
    """After ``docker compose up`` / ``ps``, an EMPTY container list is a
    red flag, not a success. The header ``NAME STATUS SERVICE`` with no
    rows means either the compose project never started or the agent ran
    ``ps`` in the wrong directory. Return True so the caller can flip
    this into a visible failure signal."""
    if not _DEPLOY_SUCCESS_EMPTY_RE.search(command):
        return False
    # `docker compose ps` with zero services emits either just the header
    # or no output at all (depending on version).
    rendered = (stdout or "") + (stderr or "")
    if not rendered.strip():
        return True
    lines = [ln for ln in rendered.splitlines() if ln.strip()]
    # If only a header row (contains "NAME" and "STATUS" or "SERVICE") and
    # nothing else, it's empty.
    if len(lines) <= 1:
        header = lines[0] if lines else ""
        if "NAME" in header.upper() and ("STATUS" in header.upper() or "SERVICE" in header.upper()):
            return True
    return False


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
      1. An explicit ``working_dir`` arg from the LLM — resolved against
         the session's effective workspace when relative, so ``../X``
         can't escape.
      2. The session's effective workspace (per-session override or
         global default — see config.effective_workspace_dir).
      3. Falls back to the current process CWD if even that fails.

    The directory is created on-demand for the GLOBAL fallback workspace
    only (it's our sandbox). Per-session workspaces are assumed to be
    real project directories and are NOT auto-created — if you point
    a session at a bogus path we want you to see the failure, not
    silently create an empty directory.

    Returns an absolute path.
    """
    # Late import to avoid a circular dep when tool_registry imports this
    # module during early module initialisation.
    from config import config as _cfg, effective_workspace_dir
    ws = effective_workspace_dir() or os.getcwd()
    ws = os.path.abspath(ws)
    is_global_ws = (ws == _cfg.agent_workspace_dir)

    if not working_dir:
        chosen_abs = ws
    elif os.path.isabs(working_dir):
        chosen_abs = working_dir
    else:
        # Resolve relative paths against the effective workspace, NOT
        # the uvicorn cwd — same as for the session default.
        chosen_abs = os.path.abspath(os.path.join(ws, working_dir))

    # Only auto-create the GLOBAL sandbox. Per-session paths must exist.
    if is_global_ws and chosen_abs.startswith(ws):
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

    effective_dir = _resolve_working_dir(working_dir)
    if not working_dir:
        logger.info(
            "execute_shell: no working_dir supplied, defaulting to %s",
            effective_dir,
        )

    # ── Hard block: OpenTeddy's own source tree ────────────────────────────
    # If Qwen passed a working_dir like "/home/user/OpenTeddy", refuse
    # flat out — running `docker compose up` or `rm` there would break
    # the live uvicorn, clobber openteddy.db, overwrite skills/. Only
    # agent-workspace/ under the project root is allowed.
    if _is_openteddy_source_path(effective_dir):
        msg = (
            f"Refused: working_dir '{effective_dir}' is inside OpenTeddy's "
            f"own source tree. This would conflict with the running agent. "
            f"Use a path under the agent workspace, or set this session's "
            f"workspace via the 📂 icon in the chat header to point at the "
            f"project you actually want to work on."
        )
        logger.warning("execute_shell blocked: %s", msg)
        return make_result(False, error=msg)

    # Also catch `cd /path/to/OpenTeddy && ...` at the start of the command.
    cd_match = re.match(r"^\s*cd\s+(['\"]?)([^'\"\s&|;]+)\1", command)
    if cd_match:
        cd_target = cd_match.group(2)
        # Resolve cd target relative to the subprocess cwd (effective_dir)
        cd_abs = cd_target if os.path.isabs(cd_target) else os.path.abspath(os.path.join(effective_dir, cd_target))
        if _is_openteddy_source_path(cd_abs):
            msg = (
                f"Refused: command tried to `cd` into '{cd_abs}', which is "
                f"inside OpenTeddy's own source tree. Only agent-workspace/ "
                f"and its subdirs are allowed. Fix the plan or set a "
                f"session-specific workspace via 📂."
            )
            logger.warning("execute_shell blocked: %s", msg)
            return make_result(False, error=msg)

    # Path-hygiene fix for the most common Gemma planning mistake: it
    # sometimes prefixes cd targets with ./agent-workspace/ even though
    # the shell is already rooted there. We fix it transparently and log
    # a note so the user can see it happened.
    fixed, fix_note = _fix_duplicate_workspace_prefix(command, effective_dir)
    if fix_note:
        logger.info("execute_shell: %s", fix_note)
        command = fixed

    # ── Soft warning: session workspace boundary ───────────────────────────
    # If the user set a session-specific workspace AND the effective_dir
    # escaped it, attach a note so Qwen sees "you're drifting outside the
    # session's declared workspace" and can correct course.
    try:
        from config import config as _cfg, effective_workspace_dir
        sess_ws = os.path.abspath(effective_workspace_dir())
        global_ws = _cfg.agent_workspace_dir
        if sess_ws != global_ws and not effective_dir.startswith(sess_ws):
            logger.warning(
                "execute_shell: cwd %s escapes session workspace %s",
                effective_dir, sess_ws,
            )
    except Exception:  # noqa: BLE001
        pass

    effective_timeout = _docker_timeout(command, timeout)
    if effective_timeout != timeout:
        logger.info(
            "execute_shell: overriding timeout %ds → %ds for command: %s",
            timeout, effective_timeout, command[:80],
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

        # If this looks like a cwd / path mismatch, tell Qwen AND the user
        # exactly where the shell was and what's in it. Before this block,
        # the `cd: can't cd to worldmonitor` error gave zero clues about
        # which directory the shell was in — now it does.
        combined = (stdout or "") + "\n" + (stderr or "")
        if not success and _CD_FAILURE_RE.search(combined):
            stderr = (stderr + _build_cwd_diagnostic(effective_dir)).strip()

        # Sanity guard: `docker compose up/ps` that exits 0 with ZERO rows
        # is almost always a false success (wrong cwd, missing project
        # file, stale context). Flip it to failure so Qwen escalates
        # instead of reporting "deployed successfully" with no containers.
        if success and _looks_like_empty_compose_result(command, stdout, stderr):
            logger.warning(
                "execute_shell: docker compose returned OK but NO containers "
                "are present — likely ran in the wrong directory or the project "
                "name is off. Flipping result to failure so the agent retries."
            )
            hint = (
                "\n\n[OpenTeddy hint] docker compose reported zero containers. "
                "This usually means the command ran in the wrong directory, or "
                "the compose file wasn't picked up. Try running `docker compose "
                "config` inside the project dir, or use an explicit `-f <file>`."
            )
            stderr = (stderr + hint).strip()
            return make_result(
                False,
                result={
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code,
                },
                error="docker compose returned zero containers (likely wrong cwd)",
                duration_ms=duration_ms,
            )

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
