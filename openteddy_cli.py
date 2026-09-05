#!/usr/bin/env python3
"""
openteddy — the command-line door into the OpenTeddy Runtime.

The Runtime is the FastAPI process (run.sh); this is a thin client over
its Task API. Nothing here plans, executes or decides — it submits work,
follows the task's event stream, and lets a human answer approval
prompts from the terminal. That last part is what makes headless
operation real: with the web UI closed, a high-risk tool call still has
somewhere to be approved.

    openteddy run "Analyze ~/sales.xlsx and create a chart"
    openteddy task list | task status <id> [--follow] | task cancel <id>
    openteddy skill list | skill run <name> --input '{"x": 1}'
    openteddy agent list | agent scope <name> db_query http_get
    openteddy tools | models | health
    openteddy service install [--host 0.0.0.0] | status | logs | uninstall

Connects to $OPENTEDDY_URL (default http://127.0.0.1:8000) or --url.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Iterator, List, Optional

try:
    import httpx
except ImportError:  # pragma: no cover
    sys.stderr.write("openteddy: httpx missing — run from the repo (.venv) or pip install httpx\n")
    sys.exit(2)

DEFAULT_URL = os.environ.get("OPENTEDDY_URL", "http://127.0.0.1:8000")
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
TERMINAL = {"completed", "failed", "escalated"}
EXIT_BY_STATUS = {"completed": 0, "failed": 1, "escalated": 2}


def die(msg: str, code: int = 1) -> None:
    sys.stderr.write(f"✗ {msg}\n")
    sys.exit(code)


def _short(obj: Any, n: int = 100) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        s = str(obj)
    return s if len(s) <= n else s[: n - 1] + "…"


def _fmt_dur(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


class Runtime:
    """HTTP client with the runtime's failure modes turned into sentences."""

    def __init__(self, url: str, json_out: bool = False) -> None:
        self.url = url.rstrip("/")
        self.json_out = json_out
        self.c = httpx.Client(base_url=self.url, timeout=httpx.Timeout(30.0, connect=5.0))

    def req(self, method: str, path: str, **kw: Any) -> Any:
        try:
            r = self.c.request(method, path, **kw)
        except httpx.ConnectError:
            die(f"Runtime not reachable at {self.url}.\n"
                f"  Start it:            ./run.sh\n"
                f"  Or run it as a service so it is always there: openteddy service install")
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail")
            except Exception:  # noqa: BLE001
                detail = r.text[:300]
            die(f"HTTP {r.status_code}: {detail}")
        return r.json() if r.content else {}

    def out(self, data: Any, human: Optional[str] = None) -> None:
        if self.json_out or human is None:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            print(human)

    # ── event stream ───────────────────────────────────────────────────────

    def events(self, task_id: str) -> Iterator[Dict[str, Any]]:
        """Yield the task's SSE events until the stream closes."""
        with self.c.stream(
            "GET", f"/tasks/{task_id}/events",
            timeout=httpx.Timeout(None, connect=10.0),
        ) as resp:
            if resp.status_code >= 400:
                die(f"HTTP {resp.status_code} on event stream")
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                try:
                    yield json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue

    def follow(self, task_id: str, auto_approve: bool = False) -> int:
        """Render a task's progress to the terminal; answer approval
        prompts; return a process exit code from the final status."""
        t0 = time.monotonic()
        artifacts: List[str] = []
        plan_started = False
        for ev in self.events(task_id):
            kind = ev.get("type") or ev.get("event") or ""
            if kind == "plan.stream.delta":
                if not plan_started:
                    print("⋯ planning", end="", flush=True)
                    plan_started = True
                continue
            if kind == "plan.stream.end":
                print("\r✓ Plan ready          ")
                continue
            if kind == "task.started":
                print(f"✓ Running  {task_id[:8]}")
                continue
            if kind == "subtask.progress":
                print(f"▸ subtask {ev.get('order')}/{ev.get('total')}  {ev.get('status', '')}")
                continue
            if kind == "tool_call":
                print(f"  → {ev.get('tool')}  {_short(ev.get('args', {}))}")
                continue
            if kind == "tool_result":
                ok = ev.get("success", True)
                dur = ev.get("duration_ms")
                tail = f"  {dur}ms" if dur is not None else ""
                err = "" if ok else f"  {_short(ev.get('error', ''), 120)}"
                print(f"  {'✓' if ok else '✗'} {ev.get('tool')}{tail}{err}")
                continue
            if kind == "artifact":
                path = ev.get("path") or ev.get("name") or ""
                if path:
                    artifacts.append(path)
                    print(f"  📎 {path}")
                continue
            if kind == "circuit_breaker":
                print(f"  ! circuit breaker: {_short(ev.get('reason') or ev, 160)}")
                continue
            if kind == "approval_request":
                self._approval(ev, auto_approve)
                continue
            if kind == "task.done":
                status = ev.get("status", "unknown")
                mark = {"completed": "✓", "failed": "✗", "escalated": "↑"}.get(status, "•")
                print(f"\n{mark} Done ({status}) in {_fmt_dur(time.monotonic() - t0)}")
                summary = (ev.get("summary") or "").strip()
                if summary:
                    print(summary)
                if artifacts:
                    print("\nArtifacts:")
                    for p in dict.fromkeys(artifacts):
                        print(f"  {p}")
                return EXIT_BY_STATUS.get(status, 1)
        # Stream ended without task.done — report what the tracker says.
        row = self.req("GET", f"/tasks/{task_id}")
        print(f"\n• stream closed; status = {row.get('status')}")
        return EXIT_BY_STATUS.get(row.get("status", ""), 1)

    def _approval(self, ev: Dict[str, Any], auto_approve: bool) -> None:
        tool, args, aid = ev.get("tool"), ev.get("args", {}), ev.get("approval_id")
        print(f"\n⚠ {tool} needs approval:\n    {_short(args, 300)}")
        if auto_approve:
            self.req("POST", f"/approvals/{aid}/approve")
            print("  ✓ approved (--yes)")
            return
        if not sys.stdin.isatty():
            print(f"  (no terminal to ask — approve elsewhere: openteddy task approve {ev.get('task_id', '')[:8]})")
            return
        try:
            ans = input("  approve? [y/N] ").strip().lower()
        except EOFError:
            ans = ""
        if ans in ("y", "yes"):
            self.req("POST", f"/approvals/{aid}/approve")
            print("  ✓ approved")
        else:
            self.req("POST", f"/approvals/{aid}/reject")
            print("  ✗ rejected — the agent will try another way")

    # ── helpers ────────────────────────────────────────────────────────────

    def resolve_agent(self, ref: str) -> Dict[str, Any]:
        agents = self.req("GET", "/agents").get("agents", [])
        hits = [a for a in agents if a.get("id") == ref or a.get("id", "").startswith(ref)
                or a.get("name") == ref]
        if not hits:
            die(f"No agent matches '{ref}'. Known: {', '.join(a['name'] for a in agents) or '(none)'}")
        if len(hits) > 1:
            die(f"'{ref}' is ambiguous: " + ", ".join(f"{a['name']} ({a['id'][:8]})" for a in hits))
        return hits[0]

    def resolve_task(self, ref: str) -> str:
        if len(ref) >= 32:
            return ref
        rows = self.req("GET", "/tasks", params={"limit": 200})
        hits = [r for r in rows if str(r.get("id", "")).startswith(ref)]
        if len(hits) != 1:
            die(f"Task id '{ref}' is {'ambiguous' if hits else 'unknown'} — use more characters")
        return hits[0]["id"]


# ── commands ───────────────────────────────────────────────────────────────────

def cmd_run(rt: Runtime, a: argparse.Namespace) -> int:
    body: Dict[str, Any] = {"intent": a.intent}
    if a.agent:
        body["agent_id"] = rt.resolve_agent(a.agent)["id"]
    if a.session:
        body["session_id"] = a.session
    if a.mode:
        body["mode"] = a.mode
    if a.local_only:
        body["privacy"] = "local_only"
    if a.unattended:
        body["require_approval"] = False
    acc = rt.req("POST", "/tasks", json=body)
    if rt.json_out or a.no_follow:
        rt.out(acc, f"✓ Task accepted  {acc['task_id']}  (session {acc.get('session_id', '')[:8]})\n"
                    f"  follow: openteddy task status {acc['task_id'][:8]} --follow")
        return 0
    if acc.get("status") == "completed" and acc.get("message"):
        # scheduling shortcut or an instant answer — nothing to follow
        print(acc["message"])
        return 0
    print(f"✓ Task accepted  {acc['task_id'][:8]}  (session {acc.get('session_id', '')[:8]})")
    return rt.follow(acc["task_id"], auto_approve=a.yes)


def cmd_task(rt: Runtime, a: argparse.Namespace) -> int:
    if a.task_cmd == "list":
        params: Dict[str, Any] = {"limit": a.limit}
        if a.session:
            params["session_id"] = a.session
        rows = rt.req("GET", "/tasks", params=params)
        if rt.json_out:
            rt.out(rows)
            return 0
        if not rows:
            print("(no tasks)")
            return 0
        print(f"{'ID':8}  {'STATUS':10} {'CREATED':16}  GOAL")
        for r in rows:
            goal = (r.get("goal") or r.get("description") or r.get("summary") or "").replace("\n", " ")
            print(f"{str(r.get('id', ''))[:8]:8}  {str(r.get('status', '')):10} "
                  f"{str(r.get('created_at', ''))[:16]:16}  {goal[:70]}")
        return 0
    tid = rt.resolve_task(a.id)
    if a.task_cmd == "status":
        row = rt.req("GET", f"/tasks/{tid}")
        if rt.json_out and not a.follow:
            rt.out(row)
            return EXIT_BY_STATUS.get(row.get("status", ""), 0)
        print(f"task    {tid}\nstatus  {row.get('status')}\nsession {row.get('session_id') or '-'}")
        for st in row.get("subtasks") or []:
            print(f"  {st.get('order', 0) + 1:>2}. [{st.get('status')}] {(st.get('description') or '')[:80]}")
        if row.get("summary"):
            print("\n" + row["summary"])
        if a.follow and row.get("status") not in TERMINAL:
            print("— following —")
            return rt.follow(tid, auto_approve=a.yes)
        return EXIT_BY_STATUS.get(row.get("status", ""), 0)
    if a.task_cmd in ("approve", "reject"):
        res = rt.req("POST", f"/tasks/{tid}/{a.task_cmd}")
        rt.out(res, f"✓ {res.get('resolved', 0)} pending call(s) {res.get('status')}")
        return 0
    if a.task_cmd == "cancel":
        res = rt.req("POST", f"/tasks/{tid}/cancel")
        rt.out(res, f"✓ cancel requested for {tid[:8]}")
        return 0
    return 2


def cmd_skill(rt: Runtime, a: argparse.Namespace) -> int:
    if a.skill_cmd == "list":
        skills = rt.req("GET", "/skills").get("skills", [])
        if rt.json_out:
            rt.out(skills)
            return 0
        if not skills:
            print("(no skills yet — the agent creates them as it works)")
            return 0
        print(f"{'NAME':28} {'VER':>3} {'STATUS':8} {'OK':>4} {'FAIL':>4}  DESCRIPTION")
        for s in skills:
            print(f"{s.get('name', '')[:28]:28} {s.get('version', 1):>3} {str(s.get('status', '')):8} "
                  f"{s.get('success_count', 0):>4} {s.get('failure_count', 0):>4}  {(s.get('description') or '')[:60]}")
        return 0
    if a.skill_cmd == "run":
        try:
            payload = json.loads(a.input) if a.input else {}
        except json.JSONDecodeError as exc:
            die(f"--input must be JSON: {exc}")
        res = rt.req("POST", f"/skills/{a.name}/run", json={"input": payload})
        if rt.json_out:
            rt.out(res)
        else:
            print(f"{'✓' if res.get('success') else '✗'} {a.name}  {res.get('duration_ms')}ms\n{res.get('output', '')}")
        return 0 if res.get("success") else 1
    return 2


def cmd_agent(rt: Runtime, a: argparse.Namespace) -> int:
    if a.agent_cmd == "list":
        agents = rt.req("GET", "/agents").get("agents", [])
        if rt.json_out:
            rt.out(agents)
            return 0
        if not agents:
            print("(no agents)")
            return 0
        print(f"{'ID':8}  {'NAME':20} {'MODE':9} {'DB':3} {'SCOPE'}")
        for ag in agents:
            scope = ag.get("allowed_tools") or []
            print(f"{ag['id'][:8]:8}  {ag['name'][:20]:20} {str(ag.get('mode', '')):9} "
                  f"{'yes' if ag.get('has_db') else '-':3} "
                  f"{', '.join(scope) if scope else '(unrestricted)'}")
        return 0
    if a.agent_cmd == "scope":
        ag = rt.resolve_agent(a.agent)
        tools = [] if a.clear else list(a.tools)
        if not tools and not a.clear:
            print(f"{ag['name']}: {', '.join(ag.get('allowed_tools') or []) or '(unrestricted)'}")
            return 0
        res = rt.req("PATCH", f"/agents/{ag['id']}", json={"allowed_tools": tools})
        new = (res.get("agent") or res).get("allowed_tools") or []
        rt.out(res, f"✓ {ag['name']} scope → {', '.join(new) if new else '(unrestricted)'}"
                    + ("" if new else "\n  note: unattended runs need a non-empty scope"))
        return 0
    return 2


def cmd_tools(rt: Runtime, a: argparse.Namespace) -> int:
    tools = rt.req("GET", "/tools").get("tools", [])
    if rt.json_out:
        rt.out(tools)
        return 0
    for t in sorted(tools, key=lambda x: (x.get("risk_level") != "high", x["name"])):
        print(f"{'HIGH' if t.get('risk_level') == 'high' else 'low ':4}  {t['name']:28} {(t.get('description') or '')[:70]}")
    return 0


def cmd_models(rt: Runtime, a: argparse.Namespace) -> int:
    m = rt.req("GET", "/models")
    cloud = ", ".join(f"{k}:{v['model']}" for k, v in (m.get("cloud") or {}).items() if v.get("configured")) or "(none)"
    rt.out(m, f"engine    {m.get('engine')}  @ {m.get('base_url')}\n"
              f"planner   {m.get('planner')}\nexecutor  {m.get('executor')}\nvoice     {m.get('voice') or '-'}\n"
              f"cloud     {cloud}")
    return 0


def cmd_health(rt: Runtime, a: argparse.Namespace) -> int:
    h = rt.req("GET", "/health")
    rt.out(h, f"✓ runtime up  v{h.get('version', '?')}  @ {rt.url}")
    return 0


# ── service (launchd / systemd) ────────────────────────────────────────────────

LAUNCHD_LABEL = "net.openteddy.runtime"


def _plist(host: str, port: int, log: str) -> str:
    args = "".join(f"\n      <string>{x}</string>" for x in
                   ["/bin/bash", os.path.join(REPO_DIR, "run.sh"), "--no-reload",
                    "--host", host, "--port", str(port)])
    path_env = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LAUNCHD_LABEL}</string>
  <key>ProgramArguments</key>
  <array>{args}
  </array>
  <key>WorkingDirectory</key><string>{REPO_DIR}</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{log}</string>
  <key>StandardErrorPath</key><string>{log}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>{path_env}</string>
    <key>OPENTEDDY_HOST</key><string>{host}</string>
    <key>OPENTEDDY_PORT</key><string>{port}</string>
  </dict>
</dict>
</plist>
"""


def _systemd_unit(host: str, port: int, system: bool) -> str:
    user_line = f"User={getpass.getuser()}\n" if system else ""
    wanted = "multi-user.target" if system else "default.target"
    return f"""[Unit]
Description=OpenTeddy Runtime
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
{user_line}WorkingDirectory={REPO_DIR}
ExecStart=/bin/bash {os.path.join(REPO_DIR, "run.sh")} --no-reload --host {host} --port {port}
Restart=on-failure
RestartSec=3
Environment=PATH=/usr/local/bin:/usr/bin:/bin:%h/.local/bin
Environment=OPENTEDDY_HOST={host}
Environment=OPENTEDDY_PORT={port}

[Install]
WantedBy={wanted}
"""


def _sh(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    print("  $ " + " ".join(cmd))
    return subprocess.run(cmd, check=check, text=True, capture_output=False)


def cmd_service(rt: Runtime, a: argparse.Namespace) -> int:
    sysname = platform.system()
    if a.service_cmd == "install":
        host, port = a.host, a.port
        if sysname == "Darwin":
            plist_path = os.path.expanduser(f"~/Library/LaunchAgents/{LAUNCHD_LABEL}.plist")
            log = os.path.join(REPO_DIR, "logs", "runtime.log")
            content = _plist(host, port, log)
            if a.dry_run:
                print(f"# would write {plist_path}\n{content}")
                return 0
            os.makedirs(os.path.dirname(plist_path), exist_ok=True)
            os.makedirs(os.path.dirname(log), exist_ok=True)
            with open(plist_path, "w") as fh:
                fh.write(content)
            uid = os.getuid()
            subprocess.run(["launchctl", "bootout", f"gui/{uid}", plist_path],
                           capture_output=True, text=True)
            r = subprocess.run(["launchctl", "bootstrap", f"gui/{uid}", plist_path],
                               capture_output=True, text=True)
            if r.returncode != 0:
                _sh(["launchctl", "load", "-w", plist_path])
            print(f"✓ installed {LAUNCHD_LABEL} (starts at login, restarts if it dies)\n"
                  f"  logs: {log}\n  check: openteddy health")
            return 0
        if sysname == "Linux":
            if not shutil.which("systemctl"):
                die("systemd not found — run ./run.sh under your own supervisor")
            content = _systemd_unit(host, port, a.system)
            if a.system:
                unit_path = "/etc/systemd/system/openteddy.service"
            else:
                unit_path = os.path.expanduser("~/.config/systemd/user/openteddy.service")
            if a.dry_run:
                print(f"# would write {unit_path}\n{content}")
                return 0
            if a.system:
                p = subprocess.run(["sudo", "tee", unit_path], input=content, text=True,
                                   capture_output=True)
                if p.returncode != 0:
                    die(f"could not write {unit_path}: {p.stderr.strip()}")
                _sh(["sudo", "systemctl", "daemon-reload"])
                _sh(["sudo", "systemctl", "enable", "--now", "openteddy"])
                print("✓ installed system service 'openteddy' (starts at boot)\n"
                      "  logs:  sudo journalctl -u openteddy -f\n  check: openteddy health")
                return 0
            os.makedirs(os.path.dirname(unit_path), exist_ok=True)
            with open(unit_path, "w") as fh:
                fh.write(content)
            _sh(["systemctl", "--user", "daemon-reload"])
            _sh(["systemctl", "--user", "enable", "--now", "openteddy"])
            print("✓ installed user service 'openteddy'\n"
                  f"  To start at BOOT without a login (headless server), run once:\n"
                  f"    sudo loginctl enable-linger {getpass.getuser()}\n"
                  "  logs:  journalctl --user -u openteddy -f\n  check: openteddy health")
            return 0
        die(f"service install not supported on {sysname}")
    if a.service_cmd == "uninstall":
        if sysname == "Darwin":
            plist_path = os.path.expanduser(f"~/Library/LaunchAgents/{LAUNCHD_LABEL}.plist")
            subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", plist_path],
                           capture_output=True, text=True)
            if os.path.exists(plist_path):
                os.remove(plist_path)
            print("✓ removed launch agent")
            return 0
        if sysname == "Linux":
            if a.system:
                _sh(["sudo", "systemctl", "disable", "--now", "openteddy"], check=False)
                _sh(["sudo", "rm", "-f", "/etc/systemd/system/openteddy.service"], check=False)
                _sh(["sudo", "systemctl", "daemon-reload"], check=False)
            else:
                _sh(["systemctl", "--user", "disable", "--now", "openteddy"], check=False)
                p = os.path.expanduser("~/.config/systemd/user/openteddy.service")
                if os.path.exists(p):
                    os.remove(p)
                _sh(["systemctl", "--user", "daemon-reload"], check=False)
            print("✓ removed service")
            return 0
        die(f"not supported on {sysname}")
    if a.service_cmd == "status":
        if sysname == "Darwin":
            r = subprocess.run(["launchctl", "list"], capture_output=True, text=True)
            loaded = LAUNCHD_LABEL in r.stdout
            print(f"launchd: {'loaded' if loaded else 'not installed'}")
        elif sysname == "Linux" and shutil.which("systemctl"):
            scope = ["sudo", "systemctl"] if a.system else ["systemctl", "--user"]
            r = subprocess.run(scope + ["is-active", "openteddy"], capture_output=True, text=True)
            print(f"systemd: {r.stdout.strip() or 'not installed'}")
        try:
            h = rt.c.get("/health")
            print(f"runtime: up (v{h.json().get('version', '?')}) @ {rt.url}")
        except Exception:  # noqa: BLE001
            print(f"runtime: DOWN @ {rt.url}")
        return 0
    if a.service_cmd == "logs":
        if sysname == "Darwin":
            log = os.path.join(REPO_DIR, "logs", "runtime.log")
            os.execvp("tail", ["tail", "-n", "80", "-f", log])
        scope = ["sudo", "journalctl"] if a.system else ["journalctl", "--user"]
        os.execvp(scope[0], scope + ["-u", "openteddy", "-n", "80", "-f"])
    return 2


# ── argparse ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openteddy", description="OpenTeddy Runtime CLI")
    p.add_argument("--url", default=DEFAULT_URL, help=f"runtime URL (default {DEFAULT_URL})")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="submit a natural-language task and follow it")
    r.add_argument("intent")
    r.add_argument("--agent", help="agent name or id to run as")
    r.add_argument("--session", help="existing session id")
    r.add_argument("--mode", choices=["chat", "code", "analytic"])
    r.add_argument("--local-only", action="store_true", help="never use a cloud model for this task")
    r.add_argument("--unattended", action="store_true",
                   help="run without approval prompts (needs a scoped agent)")
    r.add_argument("--yes", action="store_true", help="approve every prompt from this terminal")
    r.add_argument("--no-follow", action="store_true", help="print the task id and exit")
    r.set_defaults(fn=cmd_run)

    t = sub.add_parser("task", help="list / inspect / control tasks")
    ts = t.add_subparsers(dest="task_cmd", required=True)
    tl = ts.add_parser("list"); tl.add_argument("--limit", type=int, default=20); tl.add_argument("--session")
    tst = ts.add_parser("status"); tst.add_argument("id"); tst.add_argument("--follow", action="store_true")
    tst.add_argument("--yes", action="store_true")
    for name in ("approve", "reject", "cancel"):
        ts.add_parser(name).add_argument("id")
    t.set_defaults(fn=cmd_task)

    s = sub.add_parser("skill", help="list skills or run one directly")
    ss = s.add_subparsers(dest="skill_cmd", required=True)
    ss.add_parser("list")
    sr = ss.add_parser("run"); sr.add_argument("name"); sr.add_argument("--input", help="JSON object")
    s.set_defaults(fn=cmd_skill)

    ag = sub.add_parser("agent", help="list agents or set an agent's tool scope")
    ags = ag.add_subparsers(dest="agent_cmd", required=True)
    ags.add_parser("list")
    sc = ags.add_parser("scope", help="show or set allowed_tools (the permission boundary)")
    sc.add_argument("agent"); sc.add_argument("tools", nargs="*"); sc.add_argument("--clear", action="store_true")
    ag.set_defaults(fn=cmd_agent)

    sub.add_parser("tools", help="list tools and their risk level").set_defaults(fn=cmd_tools)
    sub.add_parser("models", help="which models/engine the runtime uses").set_defaults(fn=cmd_models)
    sub.add_parser("health", help="is the runtime up").set_defaults(fn=cmd_health)

    sv = sub.add_parser("service", help="run the runtime as an always-on background service")
    svs = sv.add_subparsers(dest="service_cmd", required=True)
    si = svs.add_parser("install")
    si.add_argument("--host", default="127.0.0.1", help="bind host (0.0.0.0 for LAN/Tailscale)")
    si.add_argument("--port", type=int, default=8000)
    si.add_argument("--system", action="store_true", help="Linux: system-wide unit via sudo (starts at boot)")
    si.add_argument("--dry-run", action="store_true", help="print the unit file, change nothing")
    for name in ("uninstall", "status", "logs"):
        x = svs.add_parser(name); x.add_argument("--system", action="store_true")
    sv.set_defaults(fn=cmd_service)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    rt = Runtime(a.url, json_out=a.json)
    try:
        return a.fn(rt, a) or 0
    except KeyboardInterrupt:
        print("\n(interrupted — the task keeps running; openteddy task cancel <id> to stop it)")
        return 130


if __name__ == "__main__":
    sys.exit(main())
