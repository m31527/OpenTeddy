#!/usr/bin/env python3
"""
OpenTeddy benchmark comparator — diff two run.py JSON outputs, print
a human-readable per-task table to stdout.

Median is used (not mean) so a single Ollama hiccup in one of 3 runs
doesn't dominate the comparison.

Usage:
    python tests/bench/compare.py baseline.json candidate.json
    python tests/bench/compare.py baseline.json candidate.json --threshold 10

`--threshold` is the % change at which a difference is reported as a
real improvement / regression. Defaults to 5 % — anything tighter is
noise on a 4B-class local model.

Exit code: 0 always. Bench reports, never gates — the user decides
what to do with the numbers. CI integration is a v2 concern.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diff two bench JSON outputs")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--threshold", type=float, default=5.0,
                   help="Pct change required to flag improvement / regression")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of the table")
    return p.parse_args()


# ── Median + warning helpers ─────────────────────────────────────────────────

def _median(runs: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [r.get(key) for r in runs if r.get(key) is not None]
    if not values:
        return None
    try:
        return float(statistics.median(values))
    except statistics.StatisticsError:
        return None


def _success_rate(runs: List[Dict[str, Any]]) -> float:
    if not runs:
        return 0.0
    n_ok = sum(1 for r in runs if r.get("status") == "completed")
    return n_ok / len(runs)


def _pct_change(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None:
        return None
    if before == 0:
        return None if after == 0 else float("inf")
    return (after - before) / before * 100.0


def _arrow(pct: Optional[float], threshold: float, lower_is_better: bool = True) -> str:
    """Return ✓ / 🎉 / ⚠ / —. lower_is_better controls direction."""
    if pct is None:
        return "—"
    if abs(pct) < threshold:
        return "✓"  # within noise
    improvement = (pct < 0) if lower_is_better else (pct > 0)
    if improvement:
        return "🎉" if abs(pct) >= 25 else "✓"
    return "⚠"  # regression past threshold


# ── Expect-warning collation ─────────────────────────────────────────────────

def _expect_warnings(task_record: Dict[str, Any]) -> List[str]:
    """Look at the task's median run vs its declared expect_* fields.
    Returns a list of human-readable strings — empty if everything's
    within bounds. These are WARNINGS not failures."""
    runs = task_record.get("runs") or []
    if not runs:
        return []
    expect = task_record.get("expect") or {}
    out: List[str] = []
    med_dur     = _median(runs, "duration_s")
    med_subs    = _median(runs, "subtask_count")
    med_tok_out = _median(runs, "tokens_out")
    if (cap := expect.get("expect_max_duration_s")) and med_dur and med_dur > cap:
        out.append(f"duration {med_dur:.1f}s > expected {cap}s")
    if (cap := expect.get("expect_max_subtasks")) and med_subs and med_subs > cap:
        out.append(f"subtasks {int(med_subs)} > expected {cap}")
    if (cap := expect.get("expect_max_tokens_out")) and med_tok_out and med_tok_out > cap:
        out.append(f"tokens_out {int(med_tok_out)} > expected {cap}")
    must_use = expect.get("expect_must_use_tool")
    must_not = expect.get("expect_must_not_use_tool")
    if must_use:
        all_tools_used: set = set()
        for r in runs:
            all_tools_used.update(r.get("tools_used") or [])
        if must_use not in all_tools_used:
            out.append(f"never called expected tool '{must_use}'")
    if must_not:
        for r in runs:
            if must_not in (r.get("tools_used") or []):
                out.append(f"called forbidden tool '{must_not}'")
                break
    return out


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    with open(args.baseline, "r", encoding="utf-8") as fh:
        b = json.load(fh)
    with open(args.candidate, "r", encoding="utf-8") as fh:
        c = json.load(fh)

    b_tasks = b.get("tasks") or {}
    c_tasks = c.get("tasks") or {}
    all_ids = sorted(set(b_tasks.keys()) | set(c_tasks.keys()))

    rows: List[Dict[str, Any]] = []
    n_improved = n_regressed = n_unchanged = 0

    for tid in all_ids:
        br = b_tasks.get(tid, {}).get("runs") or []
        cr = c_tasks.get(tid, {}).get("runs") or []
        if not br and not cr:
            continue
        b_dur = _median(br, "duration_s")
        c_dur = _median(cr, "duration_s")
        pct = _pct_change(b_dur, c_dur)
        arrow = _arrow(pct, args.threshold, lower_is_better=True)
        # Outcome tally — duration arrow is the canonical signal
        if arrow == "✓":
            n_unchanged += 1
        elif arrow in ("🎉",) or (arrow == "✓" and pct is not None and pct < -args.threshold):
            n_improved += 1
        elif arrow == "⚠":
            n_regressed += 1

        rows.append({
            "id":             tid,
            "baseline_s":     b_dur,
            "candidate_s":    c_dur,
            "pct_dur":        pct,
            "arrow":          arrow,
            "baseline_status_rate":  _success_rate(br),
            "candidate_status_rate": _success_rate(cr),
            "baseline_subtasks":  _median(br, "subtask_count"),
            "candidate_subtasks": _median(cr, "subtask_count"),
            "baseline_tokens_out": _median(br, "tokens_out"),
            "candidate_tokens_out": _median(cr, "tokens_out"),
            "warnings_candidate": _expect_warnings(c_tasks.get(tid, {})),
        })

    summary = {
        "baseline":          args.baseline,
        "candidate":         args.candidate,
        "threshold_pct":     args.threshold,
        "improvements":      n_improved,
        "regressions":       n_regressed,
        "unchanged":         n_unchanged,
        "tasks":             rows,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    # ── Pretty table ─────────────────────────────────────────────────
    print(f"=== {args.baseline} → {args.candidate} ===")
    print(f"threshold: ±{args.threshold:.1f}%   "
          f"({len(rows)} tasks, median of N runs)")
    print()
    print(f"{'id':30} {'baseline':>10} {'candidate':>10} {'change':>10}  flag")
    print("-" * 78)
    for r in rows:
        b = r["baseline_s"]
        c = r["candidate_s"]
        pct = r["pct_dur"]
        pct_str = f"{pct:+.1f}%" if pct is not None else "—"
        b_str = f"{b:.1f}s" if b is not None else "—"
        c_str = f"{c:.1f}s" if c is not None else "—"
        print(f"{r['id']:30} {b_str:>10} {c_str:>10} {pct_str:>10}  {r['arrow']}")
        for w in r.get("warnings_candidate") or []:
            print(f"    ⚠ {w}")
    print()
    print(f"improvements (≥{args.threshold:.0f}%):  {n_improved}")
    print(f"regressions  (≥{args.threshold:.0f}%):  {n_regressed}")
    print(f"unchanged:                {n_unchanged}")
    print()
    if n_regressed > 0:
        print("⚠ regressions detected — investigate before merging")
    elif n_improved > 0:
        print("✓ improvements with no regressions — looks shippable")
    else:
        print("→ no significant deltas — change is mostly invisible at this scale")

    return 0


if __name__ == "__main__":
    sys.exit(main())
