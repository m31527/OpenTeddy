#!/usr/bin/env python3
"""
Decision engine benchmark — is Laya good enough to promote, and at what
temperature?

Three things, each optional:

  --latency   warm-call latency for an English and a Chinese state on THIS
              machine (the number that matters is per-call, after warm-up)
  --labelled  accuracy / ECE / best temperature on a small built-in labelled
              set for the deliverable judge and the notify gate, zh + en.
              Zero-shot; this is the "day 0" number before any fine-tuning
  --replay    replay the runtime's `decisions` table: how often Laya agreed
              with the real verdict per kind, how the agreement changes with
              a confidence threshold, and the temperature that best fits the
              logged outcomes. This is the number that decides promotion

    .venv/bin/python scripts/decision_bench.py                # all three
    .venv/bin/python scripts/decision_bench.py --replay --hours 336
    .venv/bin/python scripts/decision_bench.py --json out.json

Temperature fitting: minimise negative log-likelihood of the (calibrated)
probability of the *correct* answer over a grid; report ECE before/after.
Paste the result into OPENTEDDY_DECISION_TEMPERATURES.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import sqlite3
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# ── built-in labelled set (zero-shot "day 0" check) ──────────────────────────
# (state, instructions-kind, label). Deliberately includes the failure modes
# seen in production: an honest zero-figure report (real), a description of
# a report (not real), a normal day (quiet), a breach (notify).

JUDGE_Q = ("Is this file a real, finished deliverable that matches the goal "
           "(not a placeholder, skeleton, or a description of one)? A report "
           "whose figures are zero but presented honestly is still real.")
NOTIFY_Q = "依通知條件，這份結果需要通知負責人嗎？（異常、失敗、資料缺失也算需要）"

JUDGE_SET: List[Tuple[str, bool]] = [
    ("GOAL: 統計兩日訂單數並產出含數量對比的 Markdown 報告\nFILE: order_daily.html\nCONTENT:\n<h1>訂單數量日報</h1><table><tr><td>2026-09-05</td><td>0</td></tr><tr><td>2026-09-06</td><td>0</td></tr></table><p>資料最新到 2026-08-30</p><canvas id='chart'></canvas>", True),
    ("GOAL: Analyze sales.csv and create a chart report\nFILE: sales_report.html\nCONTENT:\n<html><h1>Sales Report</h1><p>Total revenue 42,180. Top product: Teddy Classic (14,020).</p><script>new Chart(ctx,{type:'bar',data:{labels:['Jan','Feb'],datasets:[{data:[1255,1644]}]}})</script></html>", True),
    ("GOAL: Analyze sales.csv and create a chart report\nFILE: report.md\nCONTENT:\n# Report Plan\nThis report will contain a summary of sales, a chart of monthly revenue, and top products. TODO: run the analysis and fill in numbers.", False),
    ("GOAL: implement the snake game in JS\nFILE: snake.js\nCONTENT:\n// Snake game\n// TODO: implement game loop\nfunction init(){}\nfunction update(){ /* placeholder */ }\n", False),
    ("GOAL: implement the snake game in JS\nFILE: snake.js\nCONTENT:\nconst canvas=document.getElementById('c');const ctx=canvas.getContext('2d');let snake=[{x:5,y:5}],dir={x:1,y:0},food={x:8,y:8};document.addEventListener('keydown',e=>{if(e.key==='ArrowUp')dir={x:0,y:-1};});function tick(){const h={x:snake[0].x+dir.x,y:snake[0].y+dir.y};snake.unshift(h);if(h.x===food.x&&h.y===food.y){food={x:Math.floor(Math.random()*20),y:Math.floor(Math.random()*20)}}else snake.pop();draw();}setInterval(tick,120);", True),
    ("GOAL: 產出本週營收分析報告 HTML\nFILE: weekly.html\nCONTENT:\n<html><body><h1>本週營收分析</h1><p>本週營收 128 萬，較上週 +12%。北區 54 萬、南區 38 萬、東區 36 萬。</p><table><tr><th>區域</th><th>營收</th></tr><tr><td>北區</td><td>540,000</td></tr></table></body></html>", True),
    ("GOAL: 產出本週營收分析報告 HTML\nFILE: weekly.html\nCONTENT:\n<html><body><h1>營收分析報告</h1><p>此報告將分析本週營收。內容包含：營收總覽、區域比較、圖表。</p><p>（資料待補）</p></body></html>", False),
    ("GOAL: Create a Python CLI that converts CSV to JSON with a --pretty flag and a pytest test\nFILE: csv2json.py\nCONTENT:\nimport argparse, csv, json, sys\ndef main():\n    p=argparse.ArgumentParser(); p.add_argument('path'); p.add_argument('--pretty',action='store_true'); a=p.parse_args()\n    rows=list(csv.DictReader(open(a.path)))\n    print(json.dumps(rows, indent=2 if a.pretty else None, ensure_ascii=False))\nif __name__=='__main__': main()", True),
    ("GOAL: Create a Python CLI that converts CSV to JSON\nFILE: csv2json.py\nCONTENT:\n\"\"\"CSV to JSON converter.\n\nUsage: python csv2json.py input.csv --pretty\n\nThis script would read the CSV and output JSON.\n\"\"\"\npass", False),
    ("GOAL: 分析客訴資料並產出摘要報告\nFILE: complaints.md\nCONTENT:\n# 客訴摘要（9/1–9/7）\n共 23 件：物流 11、品質 7、客服 5。物流客訴較上週 +38%，主因為北區配送延遲。\n\n| 類別 | 件數 | 佔比 |\n|---|---|---|\n| 物流 | 11 | 48% |\n| 品質 | 7 | 30% |", True),
    ("GOAL: 分析客訴資料並產出摘要報告\nFILE: complaints.md\nCONTENT:\n# 客訴摘要\n\n將依類別統計客訴件數並比較上週。步驟：1. 讀取資料 2. 分類 3. 產出表格。", False),
    ("GOAL: write a deployment shell script for the app\nFILE: deploy.sh\nCONTENT:\n#!/usr/bin/env bash\nset -euo pipefail\ngit pull --ff-only\n.venv/bin/pip install -q -r requirements.txt\nsystemctl --user restart openteddy\ncurl -fs localhost:8000/health", True),
]

NOTIFY_SET: List[Tuple[str, bool]] = [
    ("任務目標：查昨日營收並跟前 7 日均值比\n通知條件：偏差超過 15%\n\n任務結果：\n昨日營收 128 萬元，較前 7 日均值 124 萬 +3.2%。各區正常。", False),
    ("任務目標：查昨日營收並跟前 7 日均值比\n通知條件：偏差超過 15%\n\n任務結果：\n昨日營收 98 萬元，較前 7 日均值 124 萬 -21%。南區下滑最多。", True),
    ("任務目標：查昨日新增訂單數\n通知條件：偏差超過 20%\n\n任務結果：\n昨日新增訂單 0 筆，前 7 日均值 3.2 筆（-100%）。資料最新到 9/12。", True),
    ("任務目標：查昨日新增訂單數\n通知條件：偏差超過 20%\n\n任務結果：\n昨日新增訂單 3 筆，前 7 日均值 3.2 筆（-6%）。", False),
    ("任務目標：檢查低於安全庫存的品項\n通知條件：有異常或失敗\n\n任務結果：\n低於安全庫存共 3 項：A100 剩 12、B220 剩 5、C310 剩 0。", True),
    ("任務目標：檢查低於安全庫存的品項\n通知條件：有異常或失敗\n\n任務結果：\n所有品項庫存皆高於安全水位。", False),
    ("任務目標：Meta 廣告成效日報\n通知條件：任何 campaign 的 CPA 超過目標 300 的 1.5 倍\n\n任務結果：\n三個 campaign：A CPA 210、B CPA 280、C CPA 190。整體 ROAS 3.1。", False),
    ("任務目標：Meta 廣告成效日報\n通知條件：任何 campaign 的 CPA 超過目標 300 的 1.5 倍\n\n任務結果：\n三個 campaign：A CPA 210、B CPA 520、C CPA 190。B 曝光 4,100 購買 2。", True),
    ("任務目標：彙整昨日客訴\n通知條件：有異常或失敗\n\n任務結果：\n資料庫連線被拒絕 (connection refused)，無法取得客訴資料。", True),
    ("任務目標：彙整昨日客訴\n通知條件：有異常或失敗\n\n任務結果：\n昨日客訴 2 件，皆為一般詢問，已由客服結案。", False),
    ("Goal: daily revenue vs 7-day mean\nNotify when: deviation above 15%\n\nResult:\nRevenue 1.28M, +2.9% vs 7-day mean 1.244M. All regions nominal.", False),
    ("Goal: daily revenue vs 7-day mean\nNotify when: deviation above 15%\n\nResult:\nRevenue 0.91M, -27% vs 7-day mean 1.244M. East region dropped 41%.", True),
    ("Goal: nightly backup check\nNotify when: anything failed\n\nResult:\nBackup completed 02:14, 41 GB, checksum verified.", False),
    ("Goal: nightly backup check\nNotify when: anything failed\n\nResult:\nBackup job exited with code 1 at 02:03: disk full on /backup.", True),
]


# ── metrics ──────────────────────────────────────────────────────────────────

def ece(confs: List[float], correct: List[bool], bins: int = 10) -> float:
    n = len(confs)
    if not n:
        return float("nan")
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo < c <= hi) or (b == 0 and c == 0)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confs[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(acc - conf)
    return total


def fit_temperature(p_true: List[float], labels: List[bool]) -> Tuple[float, float, float]:
    """Grid-search T minimising NLL of P(label). Returns (T, nll_before, nll_after)."""
    from decide import scale_prob

    def nll(T: float) -> float:
        s = 0.0
        for p, y in zip(p_true, labels):
            q = scale_prob(p, T)
            q = min(max(q, 1e-6), 1 - 1e-6)
            s += -math.log(q if y else 1 - q)
        return s / max(1, len(p_true))
    grid = [round(0.3 + 0.05 * i, 2) for i in range(95)]      # 0.30 … 5.00
    best = min(grid, key=nll)
    return best, nll(1.0), nll(best)


def summarise(name: str, p_true: List[float], labels: List[bool], lat: List[int]) -> Dict[str, Any]:
    from decide import scale_prob
    preds = [p >= 0.5 for p in p_true]
    correct = [a == b for a, b in zip(preds, labels)]
    conf_raw = [p if p >= 0.5 else 1 - p for p in p_true]
    T, nll0, nll1 = fit_temperature(p_true, labels)
    cal = [scale_prob(p, T) for p in p_true]
    conf_cal = [p if p >= 0.5 else 1 - p for p in cal]
    majority = max(sum(labels), len(labels) - sum(labels)) / len(labels)
    out = {
        "n": len(labels), "accuracy": round(sum(correct) / len(labels), 3),
        "majority_baseline": round(majority, 3),
        "ece_raw": round(ece(conf_raw, correct), 3), "ece_calibrated": round(ece(conf_cal, correct), 3),
        "temperature": T, "nll_raw": round(nll0, 3), "nll_calibrated": round(nll1, 3),
        "confident_share_at_0.85": round(sum(c >= 0.85 for c in conf_cal) / len(labels), 3),
        "confident_accuracy_at_0.85": (round(sum(correct[i] for i in range(len(labels)) if conf_cal[i] >= 0.85)
                                             / max(1, sum(c >= 0.85 for c in conf_cal)), 3)),
        "latency_p50_ms": int(statistics.median(lat)) if lat else None,
        "errors": [i for i, c in enumerate(correct) if not c],
    }
    print(f"\n[{name}] n={out['n']}  acc={out['accuracy']}  (majority {out['majority_baseline']})  "
          f"ECE raw {out['ece_raw']} → cal {out['ece_calibrated']} @T={T}  "
          f"confident≥0.85: {out['confident_share_at_0.85']*100:.0f}% of cases, acc {out['confident_accuracy_at_0.85']}  "
          f"p50 {out['latency_p50_ms']} ms")
    if out["errors"]:
        print(f"   wrong on cases: {out['errors']}")
    return out


# ── sections ─────────────────────────────────────────────────────────────────

async def sec_latency(D, rounds: int) -> Dict[str, Any]:
    en = "Report: daily order count. Both days 0 orders. Data latest to 2026-08-30. Chart included."
    zh = "昨日新增訂單 0 筆，較前 7 日均值 3.2 筆下降 100%。廣告花費 1,280，曝光 4,100，購買 0。"
    res: Dict[str, Any] = {}
    for tag, state in (("en", en), ("zh", zh)):
        lat = []
        for _ in range(rounds):
            p = await D.probe_yes_no("bench", state, "Does this need attention?")
            if p.error:
                res[tag] = {"error": p.error}
                break
            lat.append(p.latency_ms)
        if lat:
            res[tag] = {"p50_ms": int(statistics.median(lat)), "min_ms": min(lat), "max_ms": max(lat), "model": p.model}
            print(f"  {tag}: p50 {res[tag]['p50_ms']} ms  (min {min(lat)}, max {max(lat)})  via {p.model}")
    return res


async def _probe(D, name: str, state: str, q: str, model: Optional[str]):
    """Through the engine (auto-routed), or pinned to one Laya checkpoint."""
    if not model:
        return await D.probe_yes_no(name, state, q)
    p = D.Probe(kind=name, dtype="noul")
    t = time.monotonic()
    try:
        res = await asyncio.to_thread(D._provider._router.predict, state[:D._MAX_STATE_CHARS],
                                      {"q": {"type": "noul", "instructions": q}}, model=model)
        pt = float(res["answers"]["q"]["noul"])
        p.answer, p.probabilities = pt >= 0.5, {"true": pt, "false": 1 - pt}
        p.model = str((res.get("routing") or {}).get("model") or model)
    except Exception as exc:  # noqa: BLE001
        p.error = f"{type(exc).__name__}: {exc}"
    p.latency_ms = int((time.monotonic() - t) * 1000)
    return p


async def sec_labelled(D, model: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, data, q in (("judge.deliverable", JUDGE_SET, JUDGE_Q), ("notify.gate", NOTIFY_SET, NOTIFY_Q)):
        p_true, labels, lat = [], [], []
        for state, label in data:
            p = await _probe(D, name, state, q, model)
            if p.error:
                print(f"  {name}: Laya error: {p.error}")
                break
            p_true.append(p.probabilities["true"]); labels.append(label); lat.append(p.latency_ms)
        if p_true:
            out[name] = summarise(f"{name}{' @' + model if model else ''}", p_true, labels, lat)
    return out


def sec_replay(db_path: str, hours: int) -> Dict[str, Any]:
    from decide import scale_prob
    con = sqlite3.connect(db_path); con.row_factory = sqlite3.Row
    cutoff_sql = f"datetime('now', '-{int(hours)} hours')"
    rows = [dict(r) for r in con.execute(
        f"SELECT * FROM decisions WHERE ts >= {cutoff_sql} AND laya_answer IS NOT NULL AND agree IS NOT NULL ORDER BY ts")]
    out: Dict[str, Any] = {"hours": hours, "compared": len(rows), "kinds": {}}
    print(f"\n[replay] {len(rows)} decisions with both a Laya answer and a real verdict in the last {hours}h")
    by: Dict[str, List[dict]] = {}
    for r in rows:
        by.setdefault(r["kind"], []).append(r)
    for kind, rs in sorted(by.items()):
        agree = [bool(r["agree"]) for r in rs]
        # P(true) for noul kinds can be recovered from answer+confidence
        p_true, labels = [], []
        for r in rs:
            if r["dtype"] != "noul":
                continue
            try:
                la = json.loads(r["laya_answer"]); fa = json.loads(r["answer"])
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(la, bool) or not isinstance(fa, bool):
                continue
            c = float(r["laya_raw_confidence"] or r["laya_confidence"] or 0.5)
            p_true.append(c if la else 1 - c); labels.append(fa)
        k: Dict[str, Any] = {"n": len(rs), "agreement": round(sum(agree) / len(rs), 3),
                             "providers": {}, "laya_p50_ms": int(statistics.median([r["laya_latency_ms"] or 0 for r in rs])),
                             "fallback_p50_ms": (int(statistics.median([r["fallback_latency_ms"] for r in rs if r["fallback_latency_ms"] is not None]))
                                                 if any(r["fallback_latency_ms"] is not None for r in rs) else None)}
        for r in rs:
            k["providers"][r["provider"]] = k["providers"].get(r["provider"], 0) + 1
        if len(p_true) >= 5:
            T, nll0, nll1 = fit_temperature(p_true, labels)
            cal = [scale_prob(p, T) for p in p_true]
            conf = [p if p >= 0.5 else 1 - p for p in cal]
            corr = [(p >= 0.5) == y for p, y in zip(p_true, labels)]
            thr = {}
            for t in (0.7, 0.8, 0.85, 0.9, 0.95):
                idx = [i for i, c in enumerate(conf) if c >= t]
                thr[str(t)] = {"share": round(len(idx) / len(conf), 3),
                               "agreement": (round(sum(corr[i] for i in idx) / len(idx), 3) if idx else None)}
            k.update({"temperature": T, "nll_raw": round(nll0, 3), "nll_calibrated": round(nll1, 3),
                      "ece_raw": round(ece([p if p >= 0.5 else 1 - p for p in p_true], corr), 3),
                      "ece_calibrated": round(ece(conf, corr), 3), "by_threshold": thr})
        out["kinds"][kind] = k
        print(f"  {kind:20} n={k['n']:<4} agree={k['agreement']}  laya p50 {k['laya_p50_ms']} ms  "
              f"fallback p50 {k['fallback_p50_ms']} ms  T={k.get('temperature', '-')}")
        for t, v in (k.get("by_threshold") or {}).items():
            print(f"      ≥{t}: {v['share']*100:.0f}% of cases, agreement {v['agreement']}")
    if not rows:
        print("  (nothing yet — run the runtime in shadow mode for a while, then replay)")
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latency", action="store_true"); ap.add_argument("--labelled", action="store_true")
    ap.add_argument("--replay", action="store_true"); ap.add_argument("--hours", type=int, default=336)
    ap.add_argument("--rounds", type=int, default=20); ap.add_argument("--json", help="write full results here")
    ap.add_argument("--db", default=None, help="tracker db for --replay (default: config.db_path)")
    ap.add_argument("--model", default=None, help="pin a Laya checkpoint for --labelled: english | multilingual | typed-decisions")
    a = ap.parse_args()
    if not (a.latency or a.labelled or a.replay):
        a.latency = a.labelled = a.replay = True

    from config import config
    config.decision_mode = "shadow"
    import decide as D
    async def _nolog(rec): pass
    D.set_log_sink(_nolog)

    info = {"machine": platform.platform(), "python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["device"] = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    except Exception as exc:  # noqa: BLE001
        info["torch"] = f"missing ({exc})"
    print(f"machine {info['machine']}  torch {info.get('torch')}  device {info.get('device')}")
    results: Dict[str, Any] = {"info": info}

    if a.latency or a.labelled:
        if not D._provider.installed():
            print("laya not installed:  .venv/bin/pip install laya"); return 2
        t = time.monotonic(); ok = await D._provider.load()
        print(f"laya load: {'ok' if ok else D._provider.status()} in {time.monotonic() - t:.1f}s")
        if not ok:
            return 2
        await D.preload()
    if a.latency:
        print("\n[latency] warm calls"); results["latency"] = await sec_latency(D, a.rounds)
    if a.labelled:
        print("\n[labelled] zero-shot on the built-in set" + (f" (checkpoint: {a.model})" if a.model else ""))
        results["labelled"] = await sec_labelled(D, a.model)
    if a.replay:
        db = a.db or config.db_path
        if os.path.exists(db):
            results["replay"] = sec_replay(db, a.hours)
        else:
            print(f"\n[replay] no db at {db}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"\nwrote {a.json}")
    temps = {k: v["temperature"] for k, v in (results.get("labelled") or {}).items() if "temperature" in v}
    for k, v in (results.get("replay", {}).get("kinds") or {}).items():
        if "temperature" in v:
            temps[k] = v["temperature"]
    if temps:
        print("\nSuggested .env:\n  OPENTEDDY_DECISION_TEMPERATURES='" + json.dumps(temps) + "'")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
