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
    """Same cwd logic as shell_tool — fall back to config.agent_workspace_dir."""
    from config import config as _cfg
    chosen = working_dir or getattr(_cfg, "agent_workspace_dir", None) or os.getcwd()
    return os.path.abspath(chosen)


# ── Tool: port_probe ─────────────────────────────────────────────────────────

async def port_probe(
    port: int, host: str = "localhost",
) -> Dict[str, Any]:
    """Check whether a TCP port is in use. Returns PID/process info when
    ``lsof`` is available; otherwise only the boolean.

    Output schema (as the 'result' field of the tool response):
      {
        "port": int,
        "host": str,
        "in_use": bool,
        "pid": int | None,
        "process": str | None,
        "user": str | None,
        "listening_addresses": list[str],
        "raw_lsof": str     # preserved for debugging
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
    try:
        entries = set(os.listdir(root))
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=f"cannot list {root}: {exc}")

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

_SCHEMA_PORT_FREE: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "port_free",
        "description": (
            "Find the process listening on the given TCP port and terminate "
            "it (SIGTERM → SIGKILL after 2s if needed). HIGH RISK — requires "
            "user approval before running. Use only after `port_probe` has "
            "confirmed the port is held by a stale or unwanted process, not "
            "a service the user might still need."
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
    (port_probe,            _SCHEMA_PORT_PROBE,     "low"),
    (docker_project_detect, _SCHEMA_DOCKER_DETECT,  "low"),
    (docker_diagnose,       _SCHEMA_DOCKER_DIAGNOSE,"low"),
    (port_free,             _SCHEMA_PORT_FREE,      "high"),
]
