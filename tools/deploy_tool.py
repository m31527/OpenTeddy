"""
OpenTeddy Deploy Tool
Primitive building blocks for Code-mode deployment intelligence.

These are intentionally small and composable — Qwen orchestrates them
in its function-calling loop. Four tools here:

  port_probe            (low)  — is this port in use? by whom?
  docker_project_detect (low)  — scan a dir for Dockerfile / compose / services
  docker_diagnose       (low)  — bundle inspect + logs + ps + health hints
  port_free             (HIGH) — kill the process holding a port (approval gated)

Why not one big deploy_project() that does it all? Because when the real
world breaks — unhealthy container, port conflict, missing .env — the
agent needs leverage to recover step-by-step. A monolithic tool hides the
failure mode and forces retry-whole-thing, which wastes tokens and often
never recovers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import socket
from typing import Any, Dict, List, Optional, Tuple

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)

# Try PyYAML for proper compose parsing; fall back to regex if unavailable.
try:
    import yaml as _yaml
    _HAVE_YAML = True
except ImportError:  # pragma: no cover
    _HAVE_YAML = False


# ── Internals ────────────────────────────────────────────────────────────────

async def _run_cmd(
    argv: List[str], *, cwd: Optional[str] = None, timeout: int = 30,
) -> Tuple[int, str, str]:
    """Run a command and return (exit_code, stdout, stderr). No shell — argv only.

    Using create_subprocess_exec instead of _shell avoids injection risk when
    parameters come from the LLM (e.g. container names with weird chars).
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
    except FileNotFoundError:
        return 127, "", f"command not found: {argv[0]}"
    try:
        out_b, err_b = await asyncio.wait_for(
            proc.communicate(), timeout=float(timeout)
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return 124, "", f"timeout after {timeout}s"
    return proc.returncode or 0, out_b.decode(errors="replace"), err_b.decode(errors="replace")


def _resolve_cwd(working_dir: Optional[str]) -> str:
    """Same cwd logic as shell_tool — fall back to the session's effective
    workspace (per-session override, else config.agent_workspace_dir).

    When ``working_dir`` is a *relative* path, resolve it **against the
    effective workspace** (not against whatever cwd Python happens to be
    in). This keeps every tool's cwd semantics identical — the source
    of last month's worldmonitor/Pixelle-Video confusion was
    shell_tool vs deploy_tool resolving the same relative string
    against different anchors.
    """
    from config import effective_workspace_dir
    ws = effective_workspace_dir() or os.getcwd()
    ws = os.path.abspath(ws)
    if not working_dir:
        return ws
    chosen = working_dir if os.path.isabs(working_dir) else os.path.join(ws, working_dir)
    return os.path.abspath(chosen)


# ── Tool: port_probe ─────────────────────────────────────────────────────────

# Processes that port_probe flags as "do not kill by default" — includes
# typical web servers / databases / daemons. When port_probe sees one of
# these owning a contested port, it tells Qwen to remap the container
# port instead of calling port_free (which would disrupt the host).
_IMPORTANT_PROCESS_NAMES = frozenset({
    "uvicorn", "gunicorn", "hypercorn", "daphne",       # Python ASGI/WSGI
    "node", "npm", "next-server",                       # Node web servers
    "nginx", "apache", "apache2", "httpd", "caddy", "haproxy",
    "mysqld", "postgres", "redis-server", "mongod",     # Databases
    "ollama", "ollama-server",                          # Ollama daemon
    "docker", "dockerd", "containerd", "containerd-shim",
    "sshd", "cupsd", "launchd", "systemd",
    "chrome", "firefox", "safari",                      # Browsers often hold ephemeral ports
})


async def port_probe(
    port: int, host: str = "localhost",
) -> Dict[str, Any]:
    """Check whether a TCP port is in use. Returns PID/process info when
    ``lsof`` is available; otherwise only the boolean.

    Critically also flags whether the holding process is OpenTeddy itself,
    or a known-important host service — because port_free on those would
    kill the agent or a critical system daemon. Qwen reads the
    ``safe_to_kill_hint`` + ``recommendation`` fields to decide whether
    to remap the container's port instead.

    Output schema (as the 'result' field of the tool response):
      {
        "port": int,
        "host": str,
        "in_use": bool,
        "pid": int | None,
        "process": str | None,
        "user": str | None,
        "listening_addresses": list[str],
        "is_self": bool,              # PID matches our process or parent
        "is_important": bool,         # PID is a well-known daemon/server
        "safe_to_kill_hint": bool,    # False when is_self or is_important
        "recommendation": str,        # human-readable suggestion for Qwen
        "raw_lsof": str               # preserved for debugging
      }
    """
    port = int(port)
    out: Dict[str, Any] = {
        "port": port,
        "host": host,
        "in_use": False,
        "pid": None,
        "process": None,
        "user": None,
        "listening_addresses": [],
        "is_self": False,
        "is_important": False,
        "safe_to_kill_hint": True,
        "recommendation": "",
        "raw_lsof": "",
    }

    # First, a quick no-privilege check via bind(). If the OS lets us bind,
    # the port is definitely free. If it refuses, it's in use (or we lack
    # permission). This works even without lsof installed.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.close()
        # Port was free at bind time — return early.
        return make_result(True, result=out)
    except OSError:
        out["in_use"] = True
    except Exception:  # noqa: BLE001
        out["in_use"] = True

    # Upgrade with lsof details if available. Running on macOS/Linux only.
    if shutil.which("lsof"):
        # -P skip port→service names, -n skip DNS, -iTCP:<port> -sTCP:LISTEN
        rc, stdout, _ = await _run_cmd(
            ["lsof", "-P", "-n", f"-iTCP:{port}", "-sTCP:LISTEN"],
            timeout=5,
        )
        out["raw_lsof"] = stdout.strip()
        if rc == 0 and stdout:
            # Parse the first listener line. Header + data format:
            # COMMAND   PID   USER   FD   TYPE   DEVICE   SIZE/OFF  NODE  NAME
            lines = [ln for ln in stdout.splitlines() if ln.strip() and not ln.startswith("COMMAND")]
            addrs: List[str] = []
            for ln in lines:
                parts = ln.split()
                if len(parts) >= 9:
                    if out["pid"] is None:
                        out["process"] = parts[0]
                        try:
                            out["pid"] = int(parts[1])
                        except ValueError:
                            pass
                        out["user"] = parts[2]
                    addrs.append(parts[8])  # NAME column (e.g. *:8000, 127.0.0.1:8000)
            out["listening_addresses"] = addrs

    # Safety analysis — the whole point of this field is to stop Qwen
    # from calling port_free on OpenTeddy's own uvicorn (would kill the
    # agent) or on shared host daemons like mysqld / ollama.
    pid = out["pid"]
    if pid is not None:
        our_pid = os.getpid()
        parent_pid = os.getppid()
        if pid in (our_pid, parent_pid):
            out["is_self"] = True
        else:
            # Walk up the PID's ancestry to detect "this is a child of
            # OpenTeddy" (e.g. a uvicorn worker forked by the parent
            # gunicorn process). Cheap best-effort — skips on error.
            try:
                pid_cursor = pid
                for _ in range(6):  # bounded to avoid weird loops
                    rc_pp, pp_out, _ = await _run_cmd(
                        ["ps", "-o", "ppid=", "-p", str(pid_cursor)], timeout=2,
                    )
                    if rc_pp != 0 or not pp_out.strip():
                        break
                    try:
                        pid_cursor = int(pp_out.strip())
                    except ValueError:
                        break
                    if pid_cursor in (our_pid, parent_pid, 0, 1):
                        if pid_cursor in (our_pid, parent_pid):
                            out["is_self"] = True
                        break
            except Exception:  # noqa: BLE001
                pass

    pname = (out["process"] or "").lower()
    if pname in _IMPORTANT_PROCESS_NAMES:
        out["is_important"] = True

    out["safe_to_kill_hint"] = not (out["is_self"] or out["is_important"])

    # Recommendation string the plan-prompt tells Gemma/Qwen to read.
    if not out["in_use"]:
        out["recommendation"] = "Port is free — proceed with the bind."
    elif out["is_self"]:
        out["recommendation"] = (
            f"Port {port} is held by OpenTeddy's own process (PID {pid}). "
            "DO NOT call port_free — that would kill the agent. "
            "Use compose_remap_port to move the container to a different host port."
        )
    elif out["is_important"]:
        out["recommendation"] = (
            f"Port {port} is held by '{out['process']}' (PID {pid}), "
            "a known-important host service. Prefer compose_remap_port to "
            "rebind the container; only port_free if the user explicitly "
            "confirms stopping this service is OK."
        )
    elif pid is not None:
        out["recommendation"] = (
            f"Port {port} is held by '{out['process']}' (PID {pid}). "
            "Looks like a normal user process — port_free is reasonable, "
            "or compose_remap_port if you want to avoid disrupting it."
        )
    else:
        out["recommendation"] = (
            f"Port {port} is in use but lsof did not report a PID "
            "(may require elevated privileges). Consider compose_remap_port."
        )

    return make_result(True, result=out)


# ── Tool: docker_project_detect ──────────────────────────────────────────────

_COMPOSE_FILENAMES = (
    "docker-compose.yml",
    "docker-compose.yaml",
    "docker-compose.override.yml",
    "docker-compose.override.yaml",
    "compose.yml",
    "compose.yaml",
)


def _parse_compose(path: str) -> Dict[str, Any]:
    """Best-effort parse of a docker-compose file. Returns services + ports."""
    info: Dict[str, Any] = {"services": [], "exposed_ports": [], "errors": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as exc:  # noqa: BLE001
        info["errors"].append(f"read failed: {exc}")
        return info

    if _HAVE_YAML:
        try:
            data = _yaml.safe_load(content) or {}
            services = data.get("services", {}) or {}
            for name, svc in services.items():
                if not isinstance(svc, dict):
                    continue
                entry: Dict[str, Any] = {"name": name}
                if svc.get("image"):
                    entry["image"] = svc["image"]
                if svc.get("build"):
                    entry["build"] = svc["build"]
                ports = svc.get("ports", []) or []
                # ports can be "8080:80", "127.0.0.1:8080:80", or {published, target}
                host_ports: List[int] = []
                for p in ports:
                    if isinstance(p, str):
                        m = re.match(r"(?:[\d\.]+:)?(\d+):\d+", p)
                        if m:
                            host_ports.append(int(m.group(1)))
                    elif isinstance(p, dict) and p.get("published"):
                        try:
                            host_ports.append(int(p["published"]))
                        except (TypeError, ValueError):
                            pass
                if host_ports:
                    entry["host_ports"] = host_ports
                    info["exposed_ports"].extend(host_ports)
                info["services"].append(entry)
        except Exception as exc:  # noqa: BLE001
            info["errors"].append(f"yaml parse failed: {exc}")
    else:
        # Regex fallback: find `services:` section then list top-level keys.
        m = re.search(r"^services:\s*\n((?:[ \t].*\n)+)", content, re.MULTILINE)
        if m:
            body = m.group(1)
            for sm in re.finditer(r"^[ \t]{2}([a-zA-Z0-9_.-]+):\s*$", body, re.MULTILINE):
                info["services"].append({"name": sm.group(1)})
        # Ports: "8080:80"-style anywhere in the file
        for pm in re.finditer(r'["\']?(\d{2,5}):\d+["\']?', content):
            try:
                info["exposed_ports"].append(int(pm.group(1)))
            except ValueError:
                pass

    # Deduplicate ports while preserving order
    seen: set[int] = set()
    info["exposed_ports"] = [p for p in info["exposed_ports"] if not (p in seen or seen.add(p))]
    return info


async def docker_project_detect(
    working_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Scan a directory for Docker-related config and return structured metadata.

    Output schema:
      {
        "working_dir": str,
        "has_dockerfile": bool,
        "dockerfile_paths": [str],
        "compose_files": [str],
        "services": [ { "name", "image"?, "build"?, "host_ports"? } ],
        "exposed_ports": [int],
        "env_example_exists": bool,
        "env_exists": bool,
        "has_readme": bool,
        "likely_deploy_hint": str     # short suggestion for Qwen
      }
    """
    root = _resolve_cwd(working_dir)
    if not os.path.isdir(root):
        # Most common cause: Gemma planned to detect a repo that hasn't been
        # cloned yet. Spell it out so the next plan step is obvious.
        hint = (
            f"Directory does not exist: {root}. "
            "If this is a repo you intended to work on, git clone it into "
            "the agent workspace FIRST (e.g. `git clone <URL>`), then retry. "
            "Do not use `..` paths — all work should happen inside the "
            "workspace."
        )
        return make_result(False, error=hint)
    try:
        entries = set(os.listdir(root))
    except Exception as exc:  # noqa: BLE001
        return make_result(
            False,
            error=f"cannot list {root}: {exc}. "
                  "Check permissions or try a different working_dir.",
        )

    # Case-insensitive lookup
    entries_lower = {e.lower(): e for e in entries}

    dockerfiles: List[str] = []
    if "dockerfile" in entries_lower:
        dockerfiles.append(os.path.join(root, entries_lower["dockerfile"]))
    # Common variants
    for name in list(entries):
        if name.lower().startswith("dockerfile") and name.lower() != "dockerfile":
            dockerfiles.append(os.path.join(root, name))

    compose_paths: List[str] = []
    for cf in _COMPOSE_FILENAMES:
        if cf in entries_lower:
            compose_paths.append(os.path.join(root, entries_lower[cf]))

    services: List[Dict[str, Any]] = []
    exposed_ports: List[int] = []
    for cp in compose_paths:
        parsed = _parse_compose(cp)
        services.extend(parsed.get("services", []))
        exposed_ports.extend(parsed.get("exposed_ports", []))

    # Deduplicate services by name (keep first occurrence; overrides merge later)
    seen_svc: set[str] = set()
    services_uniq: List[Dict[str, Any]] = []
    for s in services:
        if s["name"] not in seen_svc:
            seen_svc.add(s["name"])
            services_uniq.append(s)

    seen_ports: set[int] = set()
    exposed_ports = [p for p in exposed_ports if not (p in seen_ports or seen_ports.add(p))]

    env_example = any(
        n.lower() in ("env.example", ".env.example", "env.template", ".env.template")
        for n in entries
    )
    env_exists  = ".env" in entries_lower
    has_readme  = any(n.lower().startswith("readme") for n in entries)

    # Deployment hint heuristic — tells Qwen the fastest next step.
    if compose_paths:
        hint = f"docker compose -f {os.path.basename(compose_paths[0])} up -d"
        if not env_exists and env_example:
            hint = "cp .env.example .env  # (edit values as needed), then " + hint
    elif dockerfiles:
        hint = "docker build -t app . && docker run -d --name app app"
    else:
        hint = "No Docker config found — check the README for manual install steps"

    result = {
        "working_dir": root,
        "has_dockerfile": bool(dockerfiles),
        "dockerfile_paths": dockerfiles,
        "compose_files": compose_paths,
        "services": services_uniq,
        "exposed_ports": exposed_ports,
        "env_example_exists": env_example,
        "env_exists": env_exists,
        "has_readme": has_readme,
        "likely_deploy_hint": hint,
    }
    return make_result(True, result=result)


# ── Tool: docker_diagnose ────────────────────────────────────────────────────

# Log-pattern heuristics — cheap to evaluate, high signal.
_DIAGNOSIS_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?i)out of memory|oom-?killed|killed.*out of memory"),
     "OOM killed — container exceeded memory limit. Increase `deploy.resources.limits.memory` "
     "in compose or reduce workload."),
    (re.compile(r"(?i)bind:?.{0,10}address already in use|port is already allocated"),
     "Port conflict — something else is holding the port. Use `port_probe` then `port_free`, "
     "or change the host-side port in the compose file."),
    (re.compile(r"(?i)no space left on device"),
     "Disk full — free space with `docker system prune -a -f` (careful) or clean up volumes."),
    (re.compile(r"(?i)permission denied"),
     "Permission error — likely a volume mount or entrypoint uid/gid issue. Check `chown` "
     "on the mounted directory."),
    (re.compile(r"(?i)error response from daemon"),
     "Docker daemon error — check `docker info` and the daemon's own logs (journalctl -u docker)."),
    (re.compile(r"(?i)pull access denied|manifest unknown|repository does not exist"),
     "Image pull failed — check the image name/tag and `docker login` for private registries."),
    (re.compile(r"(?i)no such container|no such image"),
     "Target not found — may need to `docker compose build` or `docker compose up --no-start` first."),
    (re.compile(r"(?i)connection refused.*:\d+"),
     "Internal service not listening yet — container may still be booting. Wait ~10s and retry, "
     "or check the service's healthcheck."),
    (re.compile(r"(?i)exit code 137"),
     "Exit code 137 = SIGKILL, almost always OOM. Raise memory limit."),
    (re.compile(r"(?i)exit code 139"),
     "Exit code 139 = SIGSEGV (segfault). Likely an architecture mismatch (arm64 vs amd64) "
     "or a binary bug."),
    (re.compile(r"(?i)\.env.*(?:not found|no such file)"),
     "Missing .env — copy from .env.example if present, then fill in required values."),
]


def _run_diagnosis(text: str) -> Optional[str]:
    for pattern, hint in _DIAGNOSIS_PATTERNS:
        if pattern.search(text):
            return hint
    return None


async def docker_diagnose(
    target: str,
) -> Dict[str, Any]:
    """Bundle docker inspect + logs + ps + diagnosis into one call.

    ``target`` is a container name or id (not a service name — use
    `docker compose ps` first to resolve service→container if needed).

    Output schema:
      {
        "target": str,
        "exists": bool,
        "status": str,              # 'running' | 'exited' | 'restarting' | 'unhealthy' | ...
        "health": str | None,       # from docker healthcheck, if defined
        "exit_code": int | None,
        "restart_count": int,
        "started_at": str | None,
        "finished_at": str | None,
        "port_bindings": [str],     # ["0.0.0.0:8080->80/tcp", ...]
        "image": str | None,
        "recent_logs": str,         # last 50 lines
        "diagnosis_hint": str | None
      }
    """
    if not target or not isinstance(target, str):
        return make_result(False, error="target must be a non-empty string")
    if not shutil.which("docker"):
        return make_result(False, error="docker CLI not available on this host")

    out: Dict[str, Any] = {
        "target": target, "exists": False, "status": "unknown",
        "health": None, "exit_code": None, "restart_count": 0,
        "started_at": None, "finished_at": None, "port_bindings": [],
        "image": None, "recent_logs": "", "diagnosis_hint": None,
    }

    # 1. docker inspect — structured state
    rc, stdout, stderr = await _run_cmd(
        ["docker", "inspect", "--format",
         "{{.State.Status}}|{{.State.Health.Status}}|{{.State.ExitCode}}|"
         "{{.RestartCount}}|{{.State.StartedAt}}|{{.State.FinishedAt}}|"
         "{{.Config.Image}}",
         target],
        timeout=10,
    )
    if rc != 0:
        # Container/image may not exist
        err = (stderr or "").strip().splitlines()[0] if stderr else f"inspect rc={rc}"
        out["exists"] = False
        out["diagnosis_hint"] = f"docker inspect failed: {err}. " + (
            "Container may have been removed or never created — check `docker ps -a`."
        )
        return make_result(True, result=out)

    parts = stdout.strip().split("|")
    if len(parts) >= 7:
        out["exists"]       = True
        out["status"]       = parts[0] or "unknown"
        out["health"]       = parts[1] if parts[1] not in ("", "<no value>") else None
        try:
            out["exit_code"] = int(parts[2])
        except ValueError:
            out["exit_code"] = None
        try:
            out["restart_count"] = int(parts[3])
        except ValueError:
            out["restart_count"] = 0
        out["started_at"]  = parts[4] if parts[4] != "0001-01-01T00:00:00Z" else None
        out["finished_at"] = parts[5] if parts[5] != "0001-01-01T00:00:00Z" else None
        out["image"]       = parts[6]

    # 2. Port bindings — separate call because the format is different
    rc2, stdout2, _ = await _run_cmd(
        ["docker", "port", target],
        timeout=5,
    )
    if rc2 == 0 and stdout2.strip():
        out["port_bindings"] = stdout2.strip().splitlines()

    # 3. Recent logs — stdout + stderr combined
    rc3, logs_out, logs_err = await _run_cmd(
        ["docker", "logs", "--tail=50", "--timestamps", target],
        timeout=10,
    )
    # docker logs emits container stdout on our stdout and stderr on our stderr
    combined = (logs_out + ("\n" + logs_err if logs_err else "")).strip()
    # Truncate to keep the result small — 50 lines max ~4KB
    if len(combined) > 4000:
        combined = combined[:4000] + "\n... (truncated)"
    out["recent_logs"] = combined

    # 4. Heuristic hint — scan logs + status for known patterns
    scan_text = combined + " " + (out["status"] or "") + " " + (out["health"] or "")
    hint = _run_diagnosis(scan_text)
    if not hint:
        # Fall back to generic status-based hints
        if out["status"] == "restarting" and out["restart_count"] > 2:
            hint = (f"Container is in a crash loop (restarted {out['restart_count']} times). "
                    "Inspect `recent_logs` above for the underlying error.")
        elif out["status"] == "exited" and out["exit_code"] not in (0, None):
            hint = (f"Container exited with code {out['exit_code']}. "
                    "Check `recent_logs` for the last error before shutdown.")
        elif out["health"] == "unhealthy":
            hint = ("Healthcheck is failing. The service is running but its own probe says "
                    "it's not ready — check the healthcheck command and service readiness.")
    out["diagnosis_hint"] = hint

    return make_result(True, result=out)


# ── Tool: compose_validate ───────────────────────────────────────────────────

# Parse "line N" out of docker-compose error blurbs, e.g.:
#   yaml: unmarshal errors:
#     line 15: cannot unmarshal !!str `ANTHROP...` into cli.named
_COMPOSE_ERROR_LINE_RE = re.compile(r"line\s+(\d+)", re.IGNORECASE)


def _extract_context_lines(
    path: str, line_no: int, radius: int = 3,
) -> str:
    """Return a small snippet around `line_no` so Qwen can see what's wrong."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:  # noqa: BLE001
        return ""
    lo = max(0, line_no - 1 - radius)
    hi = min(len(lines), line_no + radius)
    snippet: List[str] = []
    for i in range(lo, hi):
        marker = " > " if i == line_no - 1 else "   "
        snippet.append(f"{marker}{i+1:4d}: {lines[i].rstrip()}")
    return "\n".join(snippet)


async def compose_validate(
    working_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run `docker compose config --quiet` to validate the compose file
    without actually starting anything. This is the cheapest way to catch
    YAML syntax errors, bad variable substitutions, and unresolved
    references BEFORE `up` wastes 30+ seconds building images that will
    never run.

    Pairs with env_file_lint — a .env file with multi-line values is the
    #1 cause of "cannot unmarshal !!str `XXX...` into cli.named" errors,
    because variable substitution splices the bad value directly into
    the YAML.

    Output schema:
      {
        "working_dir": str,
        "valid": bool,
        "stdout": str,                # rendered compose if valid
        "stderr": str,                # raw error if invalid
        "error_line": int | None,     # parsed line number if available
        "error_file": str | None,     # which file the line refers to (compose vs .env)
        "context_snippet": str,       # file lines around the error
        "diagnosis_hint": str | None  # short human-readable next step
      }
    """
    root = _resolve_cwd(working_dir)
    if not shutil.which("docker"):
        return make_result(False, error="docker CLI not available")

    rc, stdout, stderr = await _run_cmd(
        ["docker", "compose", "config", "--quiet"],
        cwd=root, timeout=15,
    )
    valid = rc == 0
    out: Dict[str, Any] = {
        "working_dir": root,
        "valid": valid,
        "stdout": stdout[:2000],
        "stderr": stderr.strip(),
        "error_line": None,
        "error_file": None,
        "context_snippet": "",
        "diagnosis_hint": None,
    }

    if not valid:
        # Try to pull a line number out of the error and slice the file.
        # First figure out WHICH file the error is about — docker-compose
        # reports errors in .env with the same "line N" syntax as YAML
        # errors, so we have to read the stderr to tell them apart.
        m = _COMPOSE_ERROR_LINE_RE.search(stderr)
        compose_paths = [
            os.path.join(root, n) for n in _COMPOSE_FILENAMES
            if os.path.isfile(os.path.join(root, n))
        ]
        compose_file = compose_paths[0] if compose_paths else None

        # If stderr mentions a .env path, take the context from that file.
        env_path_match = re.search(r"(\S+\.env)", stderr)
        env_ref_path = None
        if env_path_match:
            env_ref_path = env_path_match.group(1)
            # Normalise to absolute
            if not os.path.isabs(env_ref_path):
                env_ref_path = os.path.join(root, env_ref_path)

        if m:
            try:
                line_no = int(m.group(1))
                out["error_line"] = line_no
                # Prefer the env file if stderr pointed there
                target_file = env_ref_path if env_ref_path and os.path.isfile(env_ref_path) else compose_file
                if target_file:
                    out["context_snippet"] = _extract_context_lines(target_file, line_no)
                    out["error_file"] = target_file
            except ValueError:
                pass

        # Heuristic hints — the common compose-config failure modes.
        lowered = stderr.lower()
        # Most specific first: .env-origin errors point away from YAML.
        if env_ref_path and ("failed to read" in lowered or "key cannot contain" in lowered
                              or "line" in lowered):
            out["diagnosis_hint"] = (
                f"Error is in the .env file (see error_file above). "
                f"Run env_file_lint('{os.path.basename(env_ref_path)}') to see the "
                "structural issue, then fix the offending line."
            )
        elif "unmarshal" in lowered and ("cli.named" in lowered or "into" in lowered):
            out["diagnosis_hint"] = (
                "YAML unmarshal error — almost always a .env file problem: a "
                "value contains newlines or unescaped special chars, and "
                "variable substitution spliced garbage into the YAML. Run "
                "env_file_lint(.env) to pinpoint the bad line, then fix/quote it."
            )
        elif "variable is not set" in lowered or "required variable" in lowered:
            out["diagnosis_hint"] = (
                "A compose variable is unset. Copy .env.example → .env and "
                "fill in the missing key, then re-run compose_validate."
            )
        elif "yaml: line" in lowered or "mapping values are not allowed" in lowered:
            out["diagnosis_hint"] = (
                "YAML syntax error (indentation or quoting). Look at the "
                "context_snippet above — the line with ' > ' is the error."
            )
        elif "no such file" in lowered and ".env" in lowered:
            out["diagnosis_hint"] = (
                "Compose is pointing at a .env file that doesn't exist. "
                "Run `cp .env.example .env` first."
            )
        else:
            out["diagnosis_hint"] = (
                "compose config failed. Read the stderr above for the specific "
                "error; if unclear, try `docker compose config` (without --quiet) "
                "to see more context."
            )

    return make_result(True, result=out)


# ── Tool: env_file_lint ──────────────────────────────────────────────────────

async def env_file_lint(
    env_file: str = ".env",
    working_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Check a .env file for structural problems that break docker-compose
    variable substitution. The #1 cause of cryptic compose YAML errors:
    a key whose value spans multiple lines (e.g. a pasted JSON blob or
    a copied multi-line secret), because compose reads KEY=... per line
    and splices the result into the YAML verbatim.

    Also catches:
      • Duplicate keys (last one wins silently)
      • Keys with spaces/punctuation in the name
      • Values that look multi-line but aren't quoted
      • Lines that should be comments but forgot the '#'

    Output schema:
      {
        "env_file": str,                       # absolute path resolved
        "exists": bool,
        "line_count": int,
        "issues": [ {"line": int, "severity": "error"|"warn", "message": str, "content": str} ],
        "duplicate_keys": [str],
        "summary": str
      }
    """
    root = _resolve_cwd(working_dir)
    # Allow relative .env paths against the workspace
    path = env_file if os.path.isabs(env_file) else os.path.join(root, env_file)

    out: Dict[str, Any] = {
        "env_file": path, "exists": False, "line_count": 0,
        "issues": [], "duplicate_keys": [], "summary": "",
    }

    if not os.path.isfile(path):
        out["summary"] = f".env file not found at {path}"
        return make_result(True, result=out)

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=f"cannot read {path}: {exc}")

    out["exists"] = True
    out["line_count"] = len(raw_lines)

    issues: List[Dict[str, Any]] = []
    seen_keys: Dict[str, int] = {}  # key -> first line it appeared on
    dup_set: set[str] = set()

    # State machine: are we mid-way through a "looks multi-line" value?
    # Heuristic: a value ending with an unmatched quote or a backslash
    # suggests the next line is a continuation — which breaks compose.
    pending_multiline_from: Optional[int] = None

    for idx, raw in enumerate(raw_lines, start=1):
        line = raw.rstrip("\n")
        stripped = line.strip()

        # Blank / comment
        if not stripped or stripped.startswith("#"):
            if pending_multiline_from is not None:
                # A multi-line value that continues past a blank/comment is
                # almost always broken — flag the original line.
                issues.append({
                    "line": pending_multiline_from,
                    "severity": "error",
                    "message": "Value spans blank or comment lines — compose "
                               "will only see the first line, the rest becomes "
                               "YAML salad.",
                    "content": raw_lines[pending_multiline_from - 1].rstrip("\n"),
                })
                pending_multiline_from = None
            continue

        # Inside an apparent multi-line value
        if pending_multiline_from is not None:
            # Any further KEY=... line on its own terminates the suspicion,
            # but we still flag the earlier line.
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", stripped):
                issues.append({
                    "line": pending_multiline_from,
                    "severity": "error",
                    "message": "Multi-line value without triple-quote / single "
                               "line — compose substitution will corrupt the "
                               "YAML. Quote the value or put it on one line.",
                    "content": raw_lines[pending_multiline_from - 1].rstrip("\n"),
                })
                pending_multiline_from = None
            else:
                # Still continuation
                continue

        # KEY=VALUE line
        eq = stripped.find("=")
        if eq <= 0:
            issues.append({
                "line": idx, "severity": "warn",
                "message": "Line has no '=' — did you forget a comment '#' prefix?",
                "content": line,
            })
            continue

        key = stripped[:eq].strip()
        value = stripped[eq + 1 :]

        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            issues.append({
                "line": idx, "severity": "warn",
                "message": f"Non-standard env key '{key}' — use [A-Z_][A-Z0-9_]*.",
                "content": line,
            })

        if key in seen_keys:
            if key not in dup_set:
                dup_set.add(key)
                out["duplicate_keys"].append(key)
            issues.append({
                "line": idx, "severity": "warn",
                "message": f"Duplicate key '{key}' (first seen at line {seen_keys[key]}). "
                           "docker-compose uses the last occurrence.",
                "content": line,
            })
        else:
            seen_keys[key] = idx

        # Multi-line detection:
        # - value ends with backslash → line continuation (shell-like)
        # - odd number of ' or "  → unterminated quote
        # - value starts with { [ ( but doesn't close on this line
        if value.endswith("\\"):
            pending_multiline_from = idx
            continue
        q_double = value.count('"') - value.count('\\"')
        q_single = value.count("'") - value.count("\\'")
        if q_double % 2 == 1 or q_single % 2 == 1:
            pending_multiline_from = idx
            continue

    # Trailing unclosed multi-line at EOF
    if pending_multiline_from is not None:
        issues.append({
            "line": pending_multiline_from,
            "severity": "error",
            "message": "Multi-line value never terminates before end of file.",
            "content": raw_lines[pending_multiline_from - 1].rstrip("\n"),
        })

    out["issues"] = issues
    errors = sum(1 for i in issues if i["severity"] == "error")
    warns  = sum(1 for i in issues if i["severity"] == "warn")
    out["summary"] = (
        f"{out['line_count']} lines scanned: {errors} errors, {warns} warnings"
        + (f", {len(out['duplicate_keys'])} duplicate keys" if out["duplicate_keys"] else "")
    )
    return make_result(True, result=out)


# ── Tool: compose_remap_port ─────────────────────────────────────────────────

# Match a docker-compose port entry like:
#   - "8000:80"
#   - "8000:80/tcp"
#   - 8000:80
#   - "127.0.0.1:8000:80"
# Capture groups: (prefix, host_port, suffix)
# prefix  = optional quote + optional IP bind (e.g. '"127.0.0.1:')
# suffix  = everything after the host port (':80', ':80/tcp', '"')
_PORT_ENTRY_RE = re.compile(
    r"""
    (?P<prefix>
        -\s*           # list dash
        ["']?          # opening quote (optional)
        (?:[\d\.]+:)?  # optional IP bind like '127.0.0.1:'
    )
    (?P<host>\d+)
    (?P<suffix>
        :\d+           # container port
        (?:/\w+)?      # /tcp or /udp (optional)
        ["']?          # closing quote (optional)
    )
    """,
    re.VERBOSE,
)


def _edit_compose_port(
    content: str, service: str, from_port: int, to_port: int,
) -> Tuple[str, int]:
    """Regex edit of a compose file's ports mapping.

    Returns (new_content, occurrences_changed).

    We deliberately use regex rather than round-tripping through PyYAML
    because YAML dump loses comments and reformats whitespace — which
    breaks diffs, merges, and user trust in "what did the tool change?"
    Downside: if the compose file uses an exotic port syntax this might
    miss it. That's an acceptable trade for the common case.
    """
    # Locate the service block: line starting with "  <service>:" at
    # indent 2 (compose default) or 4 (nested). End at the next
    # top-level service or end of file.
    # We support 2-space and 4-space top-level indent.
    m = re.search(
        rf"(^(?P<indent>[ \t]{{2,4}}){re.escape(service)}:\s*\n)"
        rf"(?P<body>(?:(?P=indent)[ \t].*\n|\n)+)",
        content,
        re.MULTILINE,
    )
    if not m:
        return content, 0
    body_start = m.start("body")
    body_end   = m.end("body")
    body       = content[body_start:body_end]

    count = 0
    def _sub(match: re.Match) -> str:
        nonlocal count
        if int(match.group("host")) != int(from_port):
            return match.group(0)
        count += 1
        return f"{match.group('prefix')}{to_port}{match.group('suffix')}"

    new_body = _PORT_ENTRY_RE.sub(_sub, body)
    return content[:body_start] + new_body + content[body_end:], count


async def compose_remap_port(
    compose_file: str,
    service: str,
    from_port: int,
    to_port: int,
) -> Dict[str, Any]:
    """Rewrite a docker-compose file's host-side port binding for one service.

    The common use case is port conflict recovery: something on the host
    is holding ``from_port`` (e.g. OpenTeddy's own uvicorn on 8000), so
    move the container's binding to ``to_port`` instead of killing the
    host process. Only touches the specified service; comments and
    formatting elsewhere are preserved.

    Output schema:
      {
        "compose_file": str,
        "service": str,
        "from_port": int,
        "to_port": int,
        "occurrences_changed": int,    # 0 means nothing matched
        "backup_path": str,            # '<file>.bak' — for rollback
        "preview": str                 # unified-diff-ish snippet
      }

    Low risk: writes a .bak before modifying, and only touches a single
    numeric literal. The caller can roll back with mv.
    """
    try:
        with open(compose_file, "r", encoding="utf-8") as f:
            original = f.read()
    except FileNotFoundError:
        return make_result(False, error=f"compose file not found: {compose_file}")
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=f"read failed: {exc}")

    from_port = int(from_port)
    to_port = int(to_port)
    if from_port == to_port:
        return make_result(False, error="from_port and to_port are identical")

    new_content, count = _edit_compose_port(original, service, int(from_port), int(to_port))
    if count == 0:
        return make_result(
            True,
            result={
                "compose_file": compose_file,
                "service": service,
                "from_port": from_port,
                "to_port": to_port,
                "occurrences_changed": 0,
                "backup_path": "",
                "preview": "",
                "message": (
                    f"No port mapping matching {from_port}:* was found under "
                    f"service '{service}'. Check the service name and file "
                    "contents with file_read first."
                ),
            },
        )

    # Write backup + new content
    backup = compose_file + ".bak"
    try:
        with open(backup, "w", encoding="utf-8") as f:
            f.write(original)
        with open(compose_file, "w", encoding="utf-8") as f:
            f.write(new_content)
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=f"write failed: {exc}")

    # Tiny "diff preview" — the changed lines only, to keep tokens low.
    old_lines = original.splitlines()
    new_lines = new_content.splitlines()
    preview_lines: List[str] = []
    for i, (ol, nl) in enumerate(zip(old_lines, new_lines)):
        if ol != nl:
            preview_lines.append(f"- {ol}")
            preview_lines.append(f"+ {nl}")
            if len(preview_lines) >= 12:
                break
    preview = "\n".join(preview_lines) if preview_lines else "(no textual diff)"

    return make_result(
        True,
        result={
            "compose_file": compose_file,
            "service": service,
            "from_port": from_port,
            "to_port": to_port,
            "occurrences_changed": count,
            "backup_path": backup,
            "preview": preview,
            "message": (
                f"Changed {count} port mapping(s) in service '{service}' "
                f"from {from_port} → {to_port}. "
                f"Backup saved at {backup}. "
                "Restart with `docker compose up -d` (or re-create the service)."
            ),
        },
    )


# ── Tool: port_free (HIGH risk) ──────────────────────────────────────────────

async def port_free(
    port: int,
) -> Dict[str, Any]:
    """Find the process listening on ``port`` and terminate it.

    HIGH risk — the registry gates this behind user approval before it runs.
    Tries SIGTERM first, escalates to SIGKILL after 2s if the process is
    still alive. Only one PID is killed per call (if multiple processes
    share the port somehow, the caller should call again).

    Output schema:
      {
        "port": int,
        "killed": bool,
        "pid": int | None,
        "process": str | None,
        "signal_used": "TERM" | "KILL" | None,
        "message": str
      }
    """
    port = int(port)
    out: Dict[str, Any] = {
        "port": port, "killed": False, "pid": None,
        "process": None, "signal_used": None, "message": "",
    }

    if not shutil.which("lsof"):
        return make_result(False, error="lsof not available — cannot identify PID holding the port")

    rc, stdout, _ = await _run_cmd(
        ["lsof", "-P", "-n", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
        timeout=5,
    )
    if rc != 0 or not stdout.strip():
        out["message"] = f"No listening process found on port {port}."
        return make_result(True, result=out)

    pids = [int(p) for p in stdout.split() if p.isdigit()]
    if not pids:
        out["message"] = f"Could not parse lsof output for port {port}."
        return make_result(True, result=out)

    pid = pids[0]
    out["pid"] = pid

    # Look up the process name for the caller's audit log
    rc_ps, ps_out, _ = await _run_cmd(["ps", "-p", str(pid), "-o", "comm="], timeout=5)
    if rc_ps == 0 and ps_out.strip():
        out["process"] = ps_out.strip().splitlines()[0].strip()

    # SIGTERM first
    try:
        os.kill(pid, 15)   # SIGTERM
        out["signal_used"] = "TERM"
    except ProcessLookupError:
        out["message"] = f"PID {pid} was already gone before we could signal it."
        return make_result(True, result=out)
    except PermissionError:
        return make_result(
            False,
            error=f"Permission denied killing PID {pid} (owned by another user? try sudo).",
        )

    # Give it 2s to exit cleanly
    await asyncio.sleep(2)

    # Still alive? Escalate to SIGKILL
    try:
        os.kill(pid, 0)  # probe
        try:
            os.kill(pid, 9)  # SIGKILL
            out["signal_used"] = "KILL"
        except ProcessLookupError:
            pass  # died between probe and kill
    except ProcessLookupError:
        pass  # cleanly exited on SIGTERM — great

    out["killed"] = True
    out["message"] = (
        f"Terminated PID {pid} ({out['process'] or 'unknown'}) on port {port} "
        f"via SIG{out['signal_used']}."
    )
    return make_result(True, result=out)


# ── Schemas ───────────────────────────────────────────────────────────────────

_SCHEMA_PORT_PROBE: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "port_probe",
        "description": (
            "Check whether a TCP port is currently in use on a host, and if so "
            "return the PID/process holding it. Use BEFORE starting a service "
            "to avoid 'bind: address already in use', or when diagnosing such "
            "an error. Low risk — purely informational."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "port": {"type": "integer", "description": "TCP port number (1-65535)."},
                "host": {"type": "string", "description": "Host to probe. Default 'localhost'.", "default": "localhost"},
            },
            "required": ["port"],
        },
    },
}

_SCHEMA_DOCKER_DETECT: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "docker_project_detect",
        "description": (
            "Scan a project directory for Docker config (Dockerfile, "
            "docker-compose.yml, .env.example) and return structured "
            "metadata: services, exposed ports, and a suggested deploy "
            "command. Use as the FIRST step of any deployment task — it "
            "tells you whether to use compose, plain docker, or neither, "
            "and which ports are about to be bound. Low risk."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "working_dir": {
                    "type": "string",
                    "description": "Absolute or relative path to scan. "
                                   "If omitted, defaults to the agent workspace.",
                },
            },
            "required": [],
        },
    },
}

_SCHEMA_DOCKER_DIAGNOSE: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "docker_diagnose",
        "description": (
            "One-shot diagnosis of a Docker container by name or id: bundles "
            "`docker inspect`, `docker logs --tail=50`, and `docker port` "
            "into one call, plus a heuristic diagnosis_hint (OOM, port "
            "conflict, missing .env, pull access, etc.). Use whenever a "
            "container is unhealthy, restarting, exited, or failed to start. "
            "Low risk — read-only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Container name or id. If you only have a "
                                   "compose service name, resolve it with "
                                   "`docker compose ps` first.",
                },
            },
            "required": ["target"],
        },
    },
}

_SCHEMA_COMPOSE_VALIDATE: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "compose_validate",
        "description": (
            "Run `docker compose config --quiet` to validate a compose file "
            "without starting anything. PRE-FLIGHT CHECK — always run this "
            "right after `cp .env.example .env` and BEFORE `docker compose "
            "up`, because YAML/substitution errors take 30s+ to surface via "
            "up and crash nothing useful. Returns the error_line and a "
            "context_snippet around it when invalid, plus a diagnosis_hint "
            "pointing to the likely fix (env_file_lint, missing variable, "
            "syntax typo, etc.). Low risk — read-only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "working_dir": {
                    "type": "string",
                    "description": "Directory containing docker-compose.yml. "
                                   "Defaults to the agent workspace.",
                },
            },
            "required": [],
        },
    },
}

_SCHEMA_ENV_FILE_LINT: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "env_file_lint",
        "description": (
            "Scan a .env file for the structural problems that break "
            "docker-compose variable substitution: multi-line values, "
            "unterminated quotes, duplicate keys, non-standard key names. "
            "Run this when compose_validate reports a YAML unmarshal error "
            "with a value fragment (e.g. '`ANTHROP...` into cli.named') — "
            "the culprit is almost always a bad .env line. Low risk."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "env_file": {
                    "type": "string",
                    "description": "Path to the .env file. Defaults to '.env' "
                                   "relative to working_dir.",
                    "default": ".env",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Base directory for resolving env_file. "
                                   "Defaults to the agent workspace.",
                },
            },
            "required": [],
        },
    },
}

_SCHEMA_COMPOSE_REMAP: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "compose_remap_port",
        "description": (
            "Rewrite a docker-compose file's host-side port binding for a "
            "specific service. Use this when port_probe reports a conflict "
            "with safe_to_kill_hint=False — the host process is important "
            "(OpenTeddy itself, a database, a daemon) and shouldn't be "
            "killed. Moves the container's published port instead. "
            "Writes a .bak before modifying and returns a diff preview. "
            "Low risk — single numeric literal change, easy to roll back."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "compose_file": {
                    "type": "string",
                    "description": "Absolute or relative path to docker-compose.yml",
                },
                "service": {
                    "type": "string",
                    "description": "Service name as declared in the compose file (e.g. 'web', 'api').",
                },
                "from_port": {
                    "type": "integer",
                    "description": "Current host-side port (the one in conflict).",
                },
                "to_port": {
                    "type": "integer",
                    "description": "New host-side port to bind. Pick something unlikely to clash — e.g. original + 10000.",
                },
            },
            "required": ["compose_file", "service", "from_port", "to_port"],
        },
    },
}

_SCHEMA_PORT_FREE: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "port_free",
        "description": (
            "Find the process listening on the given TCP port and terminate "
            "it (SIGTERM → SIGKILL after 2s if needed). HIGH RISK — requires "
            "user approval before running. Use only after `port_probe` has "
            "confirmed the port is held by a stale or unwanted process "
            "(safe_to_kill_hint=True), NOT when is_self/is_important is set."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "port": {"type": "integer", "description": "TCP port number."},
            },
            "required": ["port"],
        },
    },
}


# ── Export ────────────────────────────────────────────────────────────────────

DEPLOY_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    (port_probe,            _SCHEMA_PORT_PROBE,       "low"),
    (docker_project_detect, _SCHEMA_DOCKER_DETECT,    "low"),
    (docker_diagnose,       _SCHEMA_DOCKER_DIAGNOSE,  "low"),
    (compose_validate,      _SCHEMA_COMPOSE_VALIDATE, "low"),
    (env_file_lint,         _SCHEMA_ENV_FILE_LINT,    "low"),
    (compose_remap_port,    _SCHEMA_COMPOSE_REMAP,    "low"),
    (port_free,             _SCHEMA_PORT_FREE,        "high"),
]
