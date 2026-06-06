#!/usr/bin/env python3
"""
OpenTeddy benchmark runner — drive a YAML task list against the
currently-running server, capture metrics, write JSON.

Assumes a server is reachable at --server (default
http://localhost:8000). Doesn't start one for you — that's deliberate,
because the whole point of this script is to measure the live
behaviour of a specific branch's server build, including its current
.env / Settings state. Spinning a fresh subprocess server every run
would change THOSE inputs too.

Usage:
    python tests/bench/run.py \
        --tasks tests/bench/golden_tasks.yaml \
        --runs 3 \
        --output tests/bench/results/baseline.json

Each task is run --runs times. The runner records every individual
attempt (so you can inspect variance if a number looks suspicious),
and compare.py takes the median per task — robust against the
occasional Ollama hiccup that would skew an average.

Cancel-safe: ^C between runs writes whatever's been collected so far,
so a 15-min bench interrupted at minute 12 still leaves usable data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Bench has no httpx dep of its own — fall back from httpx to urllib
# so this runs in any Python the user has handy, including the
# desktop's PyInstaller-bundled env when poking remote dev servers.
try:
    import httpx
    _HAVE_HTTPX = True
except ImportError:
    import urllib.request
    import urllib.error
    _HAVE_HTTPX = False

try:
    import yaml
except ImportError:
    sys.stderr.write("PyYAML required: pip install PyYAML\n")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OpenTeddy benchmark runner")
    p.add_argument("--tasks", required=True, help="Path to YAML task list")
    p.add_argument("--runs", type=int, default=3,
                   help="Times to repeat each task (median is used by compare.py)")
    p.add_argument("--output", required=True, help="Where to write the JSON result")
    p.add_argument("--server", default=os.environ.get("OPENTEDDY_URL", "http://localhost:8000"),
                   help="Base URL of the running OpenTeddy server")
    p.add_argument("--session", default="", help="Existing session id (creates a new one if blank)")
    p.add_argument("--task-timeout", type=int, default=300,
                   help="Per-task wall-clock cap, seconds. Beyond this we record a 'timeout' status.")
    p.add_argument("--filter", default="",
                   help="Only run tasks whose id starts with this prefix — useful for smoke-testing one entry")
    return p.parse_args()


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def _post_json(url: str, body: dict, timeout: float = 30.0) -> dict:
    if _HAVE_HTTPX:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=body)
            r.raise_for_status()
            return r.json()
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, timeout: float = 15.0) -> dict:
    if _HAVE_HTTPX:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(url)
            r.raise_for_status()
            return r.json()
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── Single-task execution ───────────────────────────────────────────────────

def _create_session(server: str, mode: str) -> str:
    """Spawn a fresh session for this run so previous bench tasks don't
    pollute memory / workspace state."""
    body = {"title": f"bench {uuid.uuid4().hex[:6]}", "mode": mode}
    res = _post_json(f"{server}/sessions", body)
    return res["id"]


def _run_one(server: str, session_id: str, task: Dict[str, Any],
             timeout_s: int) -> Dict[str, Any]:
    """Submit one task, poll until completion (or timeout), return
    captured metrics. Does NOT raise on task failure — failure is data
    we want to record."""
    task_id = str(uuid.uuid4())
    started = time.monotonic()
    metrics: Dict[str, Any] = {
        "task_id":     task_id,
        "goal":        task["goal"],
        "mode":        task.get("mode", "code"),
        "duration_s":  None,
        "status":      "unknown",
        "tokens_in":   0,
        "tokens_out":  0,
        "cost_usd":    0.0,
        "subtask_count": 0,
        "tool_calls":  0,
        "tools_used":  [],
        "summary_chars": 0,
        "error":       None,
    }

    try:
        _post_json(
            f"{server}/run",
            {
                "task_id":   task_id,
                "goal":      task["goal"],
                "mode":      task.get("mode", "code"),
                "session_id": session_id,
                "priority":  1,
            },
            timeout=10.0,
        )
    except Exception as exc:  # noqa: BLE001
        metrics["status"] = "submit_failed"
        metrics["error"] = str(exc)[:300]
        metrics["duration_s"] = round(time.monotonic() - started, 2)
        return metrics

    # Poll. Server's WS would be nicer but a simple poll matches the
    # bench's "single-shot script" character better than wiring up
    # websocket clients.
    last_status = None
    while True:
        if time.monotonic() - started > timeout_s:
            metrics["status"] = "timeout"
            metrics["error"] = f"bench-side timeout after {timeout_s}s"
            break
        try:
            status_resp = _get_json(f"{server}/tasks/{task_id}")
        except Exception as exc:  # noqa: BLE001
            metrics["status"] = "poll_failed"
            metrics["error"] = str(exc)[:300]
            break
        last_status = status_resp.get("status", "unknown")
        if last_status in ("completed", "failed", "escalated"):
            break
        time.sleep(0.5)

    metrics["duration_s"] = round(time.monotonic() - started, 2)
    if metrics["status"] == "unknown":
        metrics["status"] = last_status or "unknown"

    # Pull richer metrics now that the run is over
    try:
        task_resp = _get_json(f"{server}/tasks/{task_id}")
        subtasks = task_resp.get("subtasks") or []
        metrics["subtask_count"] = len(subtasks)
        summary = task_resp.get("summary") or ""
        metrics["summary_chars"] = len(summary)
        # Tool-call counting reads each subtask result and counts
        # markdown code fences that look like tool invocations. Crude
        # but cheap and runs against the public API.
        tools_used: List[str] = []
        for st in subtasks:
            result_text = (st.get("result") or "")
            for token in ("browser_fetch", "fetch_url", "shell_exec_write",
                          "shell_exec_readonly", "file_write", "read_file",
                          "python_exec", "pdf_extract_text", "web_search",
                          "telegram_send", "db_query"):
                if token in result_text:
                    tools_used.append(token)
        metrics["tools_used"] = sorted(set(tools_used))
        metrics["tool_calls"] = len(tools_used)
    except Exception as exc:  # noqa: BLE001
        metrics["error"] = (metrics.get("error") or "") + f" [post-metrics: {exc}]"

    # Token + cost — admin endpoint exposes per-task usage
    try:
        usage = _get_json(f"{server}/admin/perf?task_id={task_id}")
        if isinstance(usage, dict):
            metrics["tokens_in"] = int(usage.get("tokens_in") or 0)
            metrics["tokens_out"] = int(usage.get("tokens_out") or 0)
            metrics["cost_usd"] = float(usage.get("cost_usd") or 0.0)
    except Exception:  # noqa: BLE001
        # /admin/perf may not accept ?task_id — that's fine, leave zeros
        pass

    return metrics


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.tasks, "r", encoding="utf-8") as fh:
        tasks: List[Dict[str, Any]] = yaml.safe_load(fh) or []
    if args.filter:
        tasks = [t for t in tasks if (t.get("id") or "").startswith(args.filter)]
    if not tasks:
        sys.stderr.write("No tasks to run — empty list or filter matched nothing\n")
        return 1

    # Resolve / create the bench session up front so all tasks share it.
    if args.session:
        session_id = args.session
        print(f"Reusing session {session_id}")
    else:
        session_id = _create_session(args.server, mode="code")
        print(f"Created bench session {session_id}")

    print(f"Server: {args.server}")
    print(f"Running {len(tasks)} task(s), {args.runs} run(s) each, "
          f"max {args.task_timeout}s per task")
    print()

    result: Dict[str, Any] = {
        "server":    args.server,
        "tasks_path": str(args.tasks),
        "runs_per_task": args.runs,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "session_id": session_id,
        "tasks": {},
    }

    try:
        for ti, task in enumerate(tasks, 1):
            tid = task.get("id") or f"task_{ti}"
            print(f"[{ti}/{len(tasks)}] {tid} — {task.get('goal','')[:60]}")
            runs: List[Dict[str, Any]] = []
            for r in range(args.runs):
                print(f"    run {r+1}/{args.runs}...", end="", flush=True)
                m = _run_one(args.server, session_id, task, args.task_timeout)
                runs.append(m)
                print(f" {m['duration_s']}s ({m['status']})")
            result["tasks"][tid] = {
                "id":     tid,
                "goal":   task.get("goal"),
                "mode":   task.get("mode", "code"),
                "expect": {k: v for k, v in task.items() if k.startswith("expect_")},
                "runs":   runs,
            }
    except KeyboardInterrupt:
        print("\n^C — writing partial results to %s" % out_path)
    finally:
        result["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
