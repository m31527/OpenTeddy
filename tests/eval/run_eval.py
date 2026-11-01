#!/usr/bin/env python3
"""
Delivery eval: send each task to a running OpenTeddy, check what it
actually delivered, and report the numbers.

Run it ON the machine OpenTeddy runs on — it gives every task its own
folder under agent-workspace/eval/ and reads the files, the task rows and
the per-call timings from there.

    .venv/bin/python tests/eval/run_eval.py                    # all tasks, once
    .venv/bin/python tests/eval/run_eval.py --runs 3           # stability: 3 runs each
    .venv/bin/python tests/eval/run_eval.py --only code_slugify,doc_contract
    .venv/bin/python tests/eval/run_eval.py --executor qwen3.6:35b   # this model for the run, then back
    .venv/bin/python tests/eval/run_eval.py --compare results/a.json results/b.json

Approvals are answered automatically, except deletes and other dangerous
commands, which are refused (that is what dangerous_delete tests).
Between tasks the GPU temperature is checked: at --max-temp it waits until
the GPU cools to --cool-to — sustained load has powered these boxes off.

Writes results/<stamp>.json and results/<stamp>.md.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from checks import run_checks  # noqa: E402

DANGEROUS = re.compile(r"\brm\b|rmdir|unlink|shutil\.rmtree|os\.remove|DROP\s+TABLE|TRUNCATE|DELETE\s+FROM|mkfs|\bdd\s+if=", re.I)


# ── Heat ─────────────────────────────────────────────────────────────────────

def gpu_temp() -> Optional[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return max(int(x) for x in out if x.strip().isdigit()) if out else None
    except Exception:  # noqa: BLE001
        return None


def cool_down(max_temp: int, cool_to: int) -> float:
    """Wait while the GPU is hot. Returns seconds waited."""
    t = gpu_temp()
    if t is None or t < max_temp:
        return 0.0
    print(f"   🌡 GPU {t}°C ≥ {max_temp}°C — waiting to cool to {cool_to}°C", flush=True)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1800:
        time.sleep(20)
        t = gpu_temp()
        if t is None or t <= cool_to:
            break
    return time.monotonic() - t0


# ── Trying another executor model ────────────────────────────────────────────

def switch_executor(client: httpx.Client, model: str, ollama: str) -> str:
    """Make `model` the executor for this run; returns the one to put back.
    Refuses a model Ollama doesn't have, and loads it before the first task
    so the first task's time isn't the model loading."""
    tags = httpx.get(f"{ollama}/api/tags", timeout=15).json().get("models", [])
    if model not in {m.get("name") for m in tags}:
        raise SystemExit(f"✗ Ollama doesn't have {model} — pull it first (ollama pull {model})")
    before = ((client.get("/settings", timeout=15).json().get("data") or {}).get("executor_model") or {}).get("value", "")
    r = client.post("/settings", json={"executor_model": model}, timeout=30).json()
    if not r.get("success"):
        raise SystemExit(f"✗ couldn't switch the executor: {r.get('error')}")
    print(f"   executor {before} → {model} (put back afterwards); loading it…", flush=True)
    try:
        httpx.post(f"{ollama}/api/chat", timeout=600, json={
            "model": model, "messages": [{"role": "user", "content": "hi"}], "stream": False,
            "keep_alive": "24h", "options": {"num_predict": 1, "num_ctx": 16384}})
    except httpx.HTTPError as exc:
        print(f"   (warm-up failed: {exc})")
    return before


def restore_executor(client: httpx.Client, model: str) -> None:
    for _ in range(5):
        try:
            if client.post("/settings", json={"executor_model": model}, timeout=30).json().get("success"):
                print(f"   executor put back to {model}")
                return
        except httpx.HTTPError:
            pass
        time.sleep(10)
    print(f"⚠ couldn't put the executor back to {model} — set it in Settings")


# ── One task ─────────────────────────────────────────────────────────────────

def follow(client: httpx.Client, task_id: str, timeout_s: float, log: Dict[str, Any]) -> None:
    """Read the task's events until task.done; answer approvals."""
    deadline = time.monotonic() + timeout_s
    try:
        with client.stream("GET", f"/tasks/{task_id}/events",
                           timeout=httpx.Timeout(timeout_s + 30, connect=10)) as resp:
            for line in resp.iter_lines():
                if time.monotonic() > deadline:
                    log["timed_out"] = True
                    return
                if not line.startswith("data:"):
                    continue
                try:
                    ev = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                kind = ev.get("type") or ev.get("event") or ""
                if kind == "tool_call" and ev.get("tool"):
                    log["tools"].append(ev["tool"])
                elif kind == "circuit_breaker":
                    log["guards"] += 1
                elif kind == "approval_request":
                    args = json.dumps(ev.get("args", {}), ensure_ascii=False)
                    verdict = "reject" if DANGEROUS.search(args) else "approve"
                    if verdict == "reject":
                        log["rejected"].append(f"{ev.get('tool')}: {args[:120]}")
                    try:
                        client.post(f"/approvals/{ev.get('approval_id')}/{verdict}", timeout=15)
                    except httpx.HTTPError:
                        pass
                elif kind == "task.done":
                    log["status"] = ev.get("status") or log["status"]
                    log["answer"] = ev.get("summary") or log["answer"]
                    return
    except httpx.HTTPError as exc:
        log["stream_error"] = str(exc)


def usage(db_path: str, task_id: str) -> Dict[str, Any]:
    out = {"llm_calls": 0, "prefill_s": 0.0, "decode_s": 0.0, "load_s": 0.0, "cloud_calls": 0, "cost_usd": 0.0}
    try:
        c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
        for prov, dur, pre, load, cost in c.execute(
                "SELECT model_provider, duration_ms, prefill_ms, load_ms, cost_usd FROM usage_records WHERE task_id=?",
                (task_id,)):
            out["llm_calls"] += 1
            if prov in ("ollama", "vllm", "local"):
                out["prefill_s"] += (pre or 0) / 1000
                out["load_s"] += (load or 0) / 1000
                out["decode_s"] += max(0, (dur or 0) - (pre or 0) - (load or 0)) / 1000
            else:
                out["cloud_calls"] += 1
                out["cost_usd"] += float(cost or 0)
        c.close()
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return {k: round(v, 2) if isinstance(v, float) else v for k, v in out.items()}


def ensure_fixtures(tasks: List[Dict[str, Any]]) -> None:
    """Rebuild the inputs if any is missing (they're generated, and a
    *.db ignore rule once kept shop.db off a machine)."""
    need = {n for t in tasks for n in t.get("files", [])}
    if any(not os.path.exists(os.path.join(HERE, "fixtures", n)) for n in need):
        print("   rebuilding fixtures (make_fixtures.py)", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, "make_fixtures.py")], check=True,
                       capture_output=True)


def run_task(client: httpx.Client, task: Dict[str, Any], ws: str, args: argparse.Namespace) -> Dict[str, Any]:
    os.makedirs(ws, exist_ok=True)
    for name in task.get("files", []):
        shutil.copy(os.path.join(HERE, "fixtures", name), ws)
    log: Dict[str, Any] = {"tools": [], "rejected": [], "guards": 0, "status": "unknown", "answer": "",
                           "timed_out": False}
    t0 = time.monotonic()
    try:
        acc = client.post("/tasks", json={"intent": task["goal"], "workspace_dir": ws, "auto_lane": True,
                                          "context": {"triggered_by": "eval", "eval_task": task["id"]}},
                          timeout=60).json()
    except Exception as exc:  # noqa: BLE001
        return {"id": task["id"], "error": f"could not start: {exc}", "passed": False, "seconds": 0}
    task_id = acc.get("task_id", "")
    if acc.get("status") in ("completed", "failed") and acc.get("message") != "accepted":
        log["status"], log["answer"] = acc["status"], acc.get("message", "")      # answered on the spot
    else:
        timeout_s = float(task.get("timeout", args.timeout))
        th = threading.Thread(target=follow, args=(client, task_id, timeout_s, log), daemon=True)
        th.start()
        th.join(timeout_s + 60)
        if log["timed_out"] or th.is_alive():
            log["timed_out"] = True
            try:
                client.post(f"/tasks/{task_id}/cancel", timeout=15)
            except httpx.HTTPError:
                pass
    seconds = round(time.monotonic() - t0, 1)
    try:
        row = client.get(f"/tasks/{task_id}", timeout=15).json()
        log["status"] = row.get("status") or log["status"]
        log["answer"] = log["answer"] or row.get("summary") or row.get("message") or ""
    except Exception:  # noqa: BLE001
        pass
    ctx = {"ws": ws, "answer": log["answer"] or "", "tools": log["tools"], "rejected": log["rejected"],
           "fixtures": os.path.join(HERE, "fixtures"), "hidden": os.path.join(HERE, "hidden"),
           "python": sys.executable}
    results = run_checks(ctx, task["checks"])
    passed = all(r["ok"] for r in results) and not log["timed_out"]
    return {
        "id": task["id"], "category": task["category"], "task_id": task_id, "session_id": acc.get("session_id"),
        "status": log["status"], "passed": passed,
        # Said it finished, but the checks say otherwise: the failure that costs trust.
        "false_success": (log["status"] == "completed" and not passed),
        "timed_out": log["timed_out"], "seconds": seconds,
        "slow": seconds > float(task.get("seconds", 1e9)),
        "tools": log["tools"], "rejected": log["rejected"], "guards": log["guards"],
        "checks": results, "answer": (log["answer"] or "")[:600],
        **usage(args.db, task_id),
    }


# ── Report ───────────────────────────────────────────────────────────────────

def pct(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    return round(vals[min(len(vals) - 1, int(round(q * (len(vals) - 1))))], 1)


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results) or 1
    secs = [r["seconds"] for r in results]
    by_cat: Dict[str, List[bool]] = {}
    by_task: Dict[str, List[bool]] = {}
    for r in results:
        by_cat.setdefault(r.get("category") or "?", []).append(r["passed"])
        by_task.setdefault(r["id"], []).append(r["passed"])
    return {
        "runs": len(results),
        "pass_rate": round(sum(r["passed"] for r in results) / n, 3),
        "false_success_rate": round(sum(r.get("false_success", False) for r in results) / n, 3),
        "timeouts": sum(r.get("timed_out", False) for r in results),
        "seconds_p50": pct(secs, 0.5), "seconds_p95": pct(secs, 0.95),
        "gpu_seconds": round(sum(r.get("prefill_s", 0) + r.get("decode_s", 0) + r.get("load_s", 0) for r in results), 1),
        "llm_calls_avg": round(statistics.mean([r.get("llm_calls", 0) for r in results]) if results else 0, 1),
        "cloud_calls": sum(r.get("cloud_calls", 0) for r in results),
        "cost_usd": round(sum(r.get("cost_usd", 0) for r in results), 4),
        "loop_guards": sum(r.get("guards", 0) for r in results),
        "by_category": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_cat.items())},
        # pass^k: delivered on every run (stability), not just once.
        "always_passed": sorted(k for k, v in by_task.items() if all(v)),
        "never_passed": sorted(k for k, v in by_task.items() if not any(v)),
        "flaky": sorted(k for k, v in by_task.items() if any(v) and not all(v)),
    }


def markdown(meta: Dict[str, Any], s: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
    L = [f"# 交付壓測 {meta['stamp']}", "",
         f"OpenTeddy {meta.get('version', '?')} · {meta.get('executor', '?')} / {meta.get('planner', '?')} · "
         f"{meta['tasks']} 題 × {meta['runs']} 次", "",
         "| 指標 | 數值 |", "|---|---|",
         f"| **實際成功率**（通過自動檢查） | **{s['pass_rate']:.0%}** |",
         f"| **假成功率**（回報完成但檢查沒過） | **{s['false_success_rate']:.0%}** |",
         f"| 逾時 | {s['timeouts']} |",
         f"| 時間 p50 / p95 | {s['seconds_p50']} s / {s['seconds_p95']} s |",
         f"| 地端模型時間合計 | {s['gpu_seconds']} s |",
         f"| 平均每題模型呼叫 | {s['llm_calls_avg']} |",
         f"| 升級雲端 | {s['cloud_calls']} 次，US${s['cost_usd']} |",
         f"| 迴圈保護觸發 | {s['loop_guards']} |", "",
         "## 各類別成功率", "", "| 類別 | 成功率 |", "|---|---|"]
    L += [f"| {k} | {v:.0%} |" for k, v in s["by_category"].items()]
    if meta["runs"] > 1:
        L += ["", f"每次都成功：{', '.join(s['always_passed']) or '—'}",
              f"時好時壞：{', '.join(s['flaky']) or '—'}", f"從沒成功：{', '.join(s['never_passed']) or '—'}"]
    L += ["", "## 每題結果", "", "| 題目 | 結果 | 狀態 | 秒 | 呼叫 | 沒過的檢查 |", "|---|---|---|---|---|---|"]
    for r in results:
        bad = "；".join(f"{c['check']}: {c['detail']}" for c in r.get("checks", []) if not c["ok"])
        mark = "✅" if r["passed"] else ("⚠️ 假成功" if r.get("false_success") else "❌")
        L.append(f"| {r['id']} | {mark} | {r.get('status')}{' ⌛' if r.get('timed_out') else ''} | "
                 f"{r['seconds']}{' 🐢' if r.get('slow') else ''} | {r.get('llm_calls', 0)} | {bad.replace('|', '/')[:200]} |")
    return "\n".join(L) + "\n"


def _save(meta: Dict[str, Any], results: List[Dict[str, Any]], stamp: str, partial: bool = False):
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    name = f"{stamp}-{(meta.get('executor') or 'model').replace(':', '_').replace('/', '_')}"
    s = summarize(results)
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": dict(meta, partial=partial), "summary": s, "results": results}, fh,
                  ensure_ascii=False, indent=2)
    md = markdown(meta, s, results)
    if partial:
        md = md.replace("\n", f"\n> 進行中：已完成 {len(results)} / {meta['tasks'] * meta['runs']} 題\n", 1)
    with open(os.path.join(out_dir, f"{name}.md"), "w", encoding="utf-8") as fh:
        fh.write(md)
    return name, md


def compare(a_path: str, b_path: str) -> None:
    a, b = (json.load(open(p)) for p in (a_path, b_path))
    sa, sb = a["summary"], b["summary"]
    print(f"{'指標':<16}{a['meta']['stamp']:>18}{b['meta']['stamp']:>18}")
    for k, label in (("pass_rate", "實際成功率"), ("false_success_rate", "假成功率"), ("seconds_p50", "時間 p50"),
                     ("seconds_p95", "時間 p95"), ("gpu_seconds", "地端模型秒數"), ("llm_calls_avg", "平均呼叫"),
                     ("cloud_calls", "升級雲端"), ("timeouts", "逾時")):
        print(f"{label:<16}{sa.get(k, '-')!s:>18}{sb.get(k, '-')!s:>18}")
    pa = {r["id"]: r["passed"] for r in a["results"]}
    for r in b["results"]:
        if r["id"] in pa and pa[r["id"]] != r["passed"]:
            print(f"  {'↑ 變好' if r['passed'] else '↓ 退步'}  {r['id']}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", default=os.environ.get("OPENTEDDY_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--tasks", default=os.path.join(HERE, "tasks.yaml"))
    ap.add_argument("--only", default="", help="comma-separated task ids")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600, help="seconds per task before it's cancelled")
    ap.add_argument("--db", default=os.path.join(ROOT, "openteddy.db"))
    ap.add_argument("--workspace", default=os.path.join(ROOT, "agent-workspace", "eval"))
    ap.add_argument("--max-temp", type=int, default=80)
    ap.add_argument("--cool-to", type=int, default=65)
    ap.add_argument("--keep-sessions", action="store_true", help="don't delete the eval sessions afterwards")
    ap.add_argument("--executor", default="", help="run with this executor model, then put the previous one back")
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434"))
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
        return 0

    tasks = yaml.safe_load(open(args.tasks, encoding="utf-8"))
    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        tasks = [t for t in tasks if t["id"] in want]
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    client = httpx.Client(base_url=args.server.rstrip("/"))
    meta = {"stamp": stamp, "server": args.server, "tasks": len(tasks), "runs": args.runs}
    try:
        h = client.get("/health", timeout=10).json()
        meta["version"] = h.get("version")
        st = (client.get("/settings", timeout=10).json().get("data") or {})
        meta["executor"] = (st.get("executor_model") or {}).get("value")
        meta["planner"] = (st.get("orchestrator_model") or {}).get("value")
    except Exception as exc:  # noqa: BLE001
        print(f"✗ OpenTeddy not reachable at {args.server}: {exc}")
        return 1
    # One eval at a time: two would fight over the executor setting and the GPU.
    import fcntl
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    lock = open(os.path.join(HERE, "results", ".lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("✗ another eval is running")
        return 1
    # A stop (timer, Ctrl+C) still puts the executor back: SIGTERM → finally.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    restore = ""
    ensure_fixtures(tasks)
    if args.executor and args.executor != meta["executor"]:
        restore = switch_executor(client, args.executor, args.ollama.rstrip("/"))
        meta["executor"] = args.executor
    print(f"Eval {stamp}: {len(tasks)} tasks × {args.runs} on {meta['executor']} / {meta['planner']}")
    try:
        return _run_all(client, tasks, args, meta, stamp)
    finally:
        if restore:
            restore_executor(client, restore)


def _run_all(client: httpx.Client, tasks: List[Dict[str, Any]], args: argparse.Namespace,
             meta: Dict[str, Any], stamp: str) -> int:
    results: List[Dict[str, Any]] = []
    cooled = 0.0
    for run in range(1, args.runs + 1):
        for task in tasks:
            cooled += cool_down(args.max_temp, args.cool_to)
            ws = os.path.join(args.workspace, stamp, f"{task['id']}-r{run}")
            print(f"▸ [{run}/{args.runs}] {task['id']:<24}", end="", flush=True)
            try:
                r = run_task(client, task, ws, args)
            except Exception as exc:  # noqa: BLE001 — one broken task never ends the run
                r = {"id": task["id"], "category": task.get("category"), "passed": False, "seconds": 0,
                     "status": "harness-error", "error": f"{type(exc).__name__}: {exc}", "checks": []}
            r["run"] = run
            results.append(r)
            mark = "✅" if r["passed"] else ("⚠️ false success" if r.get("false_success") else "❌")
            print(f" {mark}  {r['seconds']:>6}s  calls={r.get('llm_calls', 0)}"
                  + ("" if r["passed"] else f"  — {next((c['detail'] for c in r.get('checks', []) if not c['ok']), r.get('error', ''))[:90]}"),
                  flush=True)
            _save(meta, results, stamp, partial=True)    # a crash still leaves a report
    meta["cooldown_seconds"] = round(cooled)
    name, md = _save(meta, results, stamp)
    out_dir = os.path.join(HERE, "results")
    if not args.keep_sessions:
        for r in results:
            if r.get("session_id"):
                try:
                    client.delete(f"/sessions/{r['session_id']}", timeout=15)
                except httpx.HTTPError:
                    pass
    print("\n" + md)
    print(f"→ {os.path.join(out_dir, name + '.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
