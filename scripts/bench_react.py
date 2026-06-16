#!/usr/bin/env python3
"""
bench_react.py — A/B the ReAct lane against the full plan→execute→summary flow.

Runs each goal twice — once with react_lane_enabled OFF, once ON — against a
running OpenTeddy server, and prints a side-by-side of wall-clock + the LLM
call breakdown pulled from the Usage records. The gemma call count is the
proof of whether the lane actually engaged:

    OFF  → ~3 gemma calls  (intent + plan + summary)
    ON   → ~1 gemma call   (intent only; plan + summary skipped)

If ON still shows ~3 gemma calls, the lane degraded back to the full flow
(the single loop came back empty) — the script flags that explicitly so you
don't mistake "it fell back" for "it's just not faster".

Usage:
    .venv/bin/python scripts/bench_react.py
    .venv/bin/python scripts/bench_react.py --repeats 2 \
        --goals "列出 ~/OpenTeddy 下有幾個 .py 檔，回報數量" \
                "整理今天 github top 10 熱門專案，產生中文 html"

Notes:
  - Talks to the SERVER, so run it on (or pointed at) the DGX that's running
    OpenTeddy. --base-url defaults to the local uvicorn on :8000.
  - Leaves react_lane_enabled restored to whatever it was before the run.
  - It does NOT touch your engine / verification settings — set those how you
    want (Ollama, verification off) before running.
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid

try:
    import httpx
except ImportError:  # pragma: no cover
    print("httpx not found — run with the project venv: .venv/bin/python scripts/bench_react.py")
    sys.exit(1)


# A light single-tool task (where the lane should clearly win) + a heavier
# deliverable task (where tool + output time dominate and the win is smaller).
DEFAULT_GOALS = [
    "列出 ~/OpenTeddy 目錄下有幾個 .py 檔，只要回報數量",
    "整理今天 GitHub 前 10 名熱門專案，產生一份中文 HTML 報告",
]


def set_react(client: httpx.Client, base: str, on: bool) -> None:
    """Flip react_lane_enabled and hot-reload (POST /settings)."""
    r = client.post(
        f"{base}/settings",
        json={"react_lane_enabled": "true" if on else "false"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(
            f"POST /settings rejected: {data.get('error')}. "
            "Is the server on a build that knows react_lane_enabled "
            "(git pull on the DGX)?"
        )


def get_react(client: httpx.Client, base: str) -> bool:
    r = client.get(f"{base}/settings", timeout=30)
    r.raise_for_status()
    v = (r.json().get("data", {}).get("react_lane_enabled", {}) or {}).get("value", "false")
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")


def run_goal(client: httpx.Client, base: str, goal: str) -> tuple[str, float, str]:
    """POST /run (blocks to completion). Returns (task_id, wall_seconds, status)."""
    task_id = str(uuid.uuid4())
    t0 = time.monotonic()
    r = client.post(
        f"{base}/run",
        json={"goal": goal, "task_id": task_id, "mode": "code"},
        timeout=900,  # tasks can run minutes
    )
    wall = time.monotonic() - t0
    status = "?"
    try:
        status = r.json().get("status", "?")
    except Exception:  # noqa: BLE001
        status = f"HTTP {r.status_code}"
    return task_id, wall, str(status)


def llm_breakdown(client: httpx.Client, base: str, task_id: str) -> dict:
    """Pull this task's LLM call records from /usage and break them down by
    model. Returns {n_calls, llm_ms, by_model: {model: (count, ms)}}."""
    r = client.get(f"{base}/usage", params={"page": 1, "page_size": 100}, timeout=30)
    r.raise_for_status()
    items = [it for it in r.json().get("items", []) if it.get("task_id") == task_id]
    by_model: dict[str, list[int]] = {}
    total_ms = 0
    for it in items:
        m = it.get("model", "?")
        dur = int(it.get("duration_ms", 0) or 0)
        slot = by_model.setdefault(m, [0, 0])
        slot[0] += 1
        slot[1] += dur
        total_ms += dur
    return {
        "n_calls": len(items),
        "llm_ms":  total_ms,
        "by_model": {m: tuple(v) for m, v in by_model.items()},
    }


def is_planner(model: str) -> bool:
    """Heuristic: the planner/summary model is the Gemma one."""
    return "gemma" in (model or "").lower()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--repeats", type=int, default=1,
                    help="runs per (goal, variant); reports the average")
    ap.add_argument("--goals", nargs="*", default=None,
                    help="override the default goal set")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the unmeasured warm-up run")
    args = ap.parse_args()

    base  = args.base_url.rstrip("/")
    goals = args.goals or DEFAULT_GOALS
    client = httpx.Client()

    # Remember the original setting so we can restore it afterwards.
    try:
        original = get_react(client, base)
    except Exception as exc:  # noqa: BLE001
        print(f"✗ Can't reach {base}/settings ({exc}). Is OpenTeddy running?")
        return 1

    print(f"▶ OpenTeddy @ {base}")
    print(f"▶ react_lane_enabled currently: {original}")
    print(f"▶ goals: {len(goals)} · repeats: {args.repeats}\n")

    # Warm-up: one throwaway run so the first measured task isn't paying a
    # cold model load. Uses the lightest goal.
    if not args.no_warmup:
        print("… warm-up run (not measured) …")
        try:
            set_react(client, base, False)
            run_goal(client, base, goals[0])
        except Exception as exc:  # noqa: BLE001
            print(f"  (warm-up failed, continuing: {exc})")

    # results[goal][variant] = list of (wall, n_calls, gemma_calls, qwen_calls, llm_ms, status)
    results: dict[str, dict[str, list]] = {g: {"OFF": [], "ON": []} for g in goals}

    try:
        for variant, on in (("OFF", False), ("ON", True)):
            set_react(client, base, on)
            for g in goals:
                for i in range(args.repeats):
                    tid, wall, status = run_goal(client, base, g)
                    bd = llm_breakdown(client, base, tid)
                    gemma = sum(c for m, (c, _) in bd["by_model"].items() if is_planner(m))
                    qwen  = bd["n_calls"] - gemma
                    results[g][variant].append(
                        (wall, bd["n_calls"], gemma, qwen, bd["llm_ms"], status)
                    )
                    print(f"  [{variant}] {g[:42]:42}  "
                          f"{wall:6.1f}s  calls={bd['n_calls']:2d} "
                          f"(gemma={gemma} qwen={qwen})  {status}")
    finally:
        # Restore the original setting no matter what.
        try:
            set_react(client, base, original)
            print(f"\n▶ restored react_lane_enabled → {original}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n⚠ could not restore react_lane_enabled: {exc}")

    # ── Report ────────────────────────────────────────────────────────────
    def avg(rows, idx):
        return sum(r[idx] for r in rows) / len(rows) if rows else 0.0

    print("\n" + "═" * 74)
    print("RESULTS  (averaged over repeats)")
    print("═" * 74)
    for g in goals:
        off, on = results[g]["OFF"], results[g]["ON"]
        if not off or not on:
            print(f"\n• {g}\n  (incomplete — a variant produced no runs)")
            continue
        o_wall, n_wall = avg(off, 0), avg(on, 0)
        o_gem,  n_gem  = avg(off, 2), avg(on, 2)
        o_calls, n_calls = avg(off, 1), avg(on, 1)
        saved = o_wall - n_wall
        pct   = (saved / o_wall * 100) if o_wall else 0.0
        print(f"\n• {g}")
        print(f"    OFF : {o_wall:6.1f}s   calls={o_calls:.0f}  gemma={o_gem:.0f}")
        print(f"    ON  : {n_wall:6.1f}s   calls={n_calls:.0f}  gemma={n_gem:.0f}")
        arrow = "faster" if saved > 0 else "SLOWER"
        print(f"    Δ   : {saved:+6.1f}s  ({pct:+.0f}%)  {arrow}")
        # Did the lane actually engage?
        if n_gem >= o_gem - 0.5:
            print("    ⚠ ReAct did NOT cut gemma calls — the lane likely "
                  "DEGRADED to the full flow (single loop came back empty). "
                  "Check the server log for 'ReAct lane degraded'.")
        elif saved <= 2:
            print("    ℹ lane engaged but the win is small — this task is "
                  "tool/output-bound, not round-trip-bound.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
