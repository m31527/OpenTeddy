"""
每週自我體檢 — what went wrong this week, found in OpenTeddy's own records.

The failures that made OpenTeddy look dumb were never "the model is too
small": a skill hijacking every "fetch" (the weather answered with the
Taipei Dome's calendar 25 times, each counted a success), a music request
sent to the no-tools chat lane, a missing package that silently disabled
every schedule. All of them were sitting in the database. This reads it
and says what it finds, with a suggested fix — rules only, no model call,
so it is instant, free and doesn't heat the box.

    report = await build_report(tracker, days=7)
    report["text"]     # Markdown for Telegram / chat
    report["findings"] # structured, for the API / tests

Checks
  repeats      the same question asked again within an hour — the strongest
               sign the first answer was wrong
  give_up      "completed" answers that say they couldn't
  same_answer  one answer given to different questions (a hijacking skill)
  skills       skills returning the same output to different goals, or
               named like a tool
  failures     failed / escalated tasks, grouped by reason
  schedules    failing or disabled schedules, runs this period
  speed        median / slowest tasks, model compute time per model
  environment  scheduler, missing packages, Home Assistant, Ollama
"""
from __future__ import annotations

import json
import logging
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_GIVE_UP = re.compile(
    r"(無法|沒辦法|做不到|未能|找不到|未找到|抱歉|建議(?:後續|您)?改用|請(?:您)?手動|沒有權限|"
    r"cannot|can't|unable to|not able to|sorry|don't have access)", re.I)
_PUNCT = re.compile(r"[\s\W_]+", re.U)
_NUM = re.compile(r"\d+")
_UUIDISH = re.compile(r"\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b", re.I)


def _norm(s: str) -> str:
    return _PUNCT.sub("", (s or "").lower())


def _parse(ts: Optional[str]) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts) if ts else None
    except ValueError:
        return None


def _local(ts: datetime) -> str:
    """UTC-naive DB time → local "10/21 18:48"."""
    loc = ts.replace(tzinfo=timezone.utc).astimezone()
    return f"{loc.month}/{loc.day} {loc:%H:%M}"


def _clip(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _goal(goal: str) -> str:
    """Display form: an attachment manifest becomes "[附件] …"."""
    if (goal or "").startswith("[Attachments in workspace"):
        try:
            from print_intent import split_manifest
            paths, rest = split_manifest(goal)
            return f"[附件 {len(paths)}] {rest}"
        except Exception:  # noqa: BLE001
            return "[附件] …"
    return goal or ""


def _origin(ctx: str) -> str:
    try:
        return (json.loads(ctx or "{}") or {}).get("triggered_by") or "web"
    except (ValueError, TypeError):
        return "web"


async def _rows(db, sql: str, args: Tuple = ()) -> List[Dict[str, Any]]:
    async with db.execute(sql, args) as cur:
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in await cur.fetchall()]


# ── Checks ───────────────────────────────────────────────────────────────────

def find_repeats(tasks: List[Dict[str, Any]], window_min: int = 60) -> List[Dict[str, Any]]:
    """Clusters of near-identical questions in one session within an hour."""
    by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in tasks:
        if t["origin"] != "schedule" and len(_norm(t["goal"])) >= 2:
            by_session[t["session_id"] or ""].append(t)
    out = []
    for rows in by_session.values():
        rows.sort(key=lambda t: t["created"])
        cluster: List[Dict[str, Any]] = []
        for t in rows + [None]:                      # sentinel flushes the last cluster
            if t and cluster:
                prev = cluster[-1]
                close = (t["created"] - prev["created"]) <= timedelta(minutes=window_min)
                same = SequenceMatcher(None, _norm(prev["goal"]), _norm(t["goal"])).ratio() >= 0.85
                if close and same:
                    cluster.append(t)
                    continue
            if len(cluster) >= 2:
                out.append({"goal": cluster[0]["goal"], "times": len(cluster),
                            "first": cluster[0]["created"], "answer": cluster[-1]["summary"]})
            cluster = [t] if t else []
    # One row per question: the same question failing in two sittings is one finding.
    merged: Dict[str, Dict[str, Any]] = {}
    for c in out:
        k = _norm(c["goal"])
        if k in merged:
            m = merged[k]
            m["times"] += c["times"]
            m["episodes"] += 1
            if c["first"] > m["last"]:
                m["answer"], m["last"] = c["answer"], c["first"]
            m["first"] = min(m["first"], c["first"])
        else:
            merged[k] = {**c, "episodes": 1, "last": c["first"]}
    return sorted(merged.values(), key=lambda c: -c["times"])


def find_give_ups(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for t in tasks:
        s = t["summary"] or ""
        if t["status"] == "completed" and not s.startswith("❌") and _GIVE_UP.search(s[:600]):
            out.append({"goal": t["goal"], "answer": s, "at": t["created"]})
    return out


# Deterministic answers (device lists, weather, print receipts) repeat by design.
_DIRECT = ("🏠", "📴", "🔊", "📍", "🖨", "印表機：", "Printer:", "✅ 所有", "⏰", "🩺")


def find_same_answers(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for t in tasks:
        if (t["status"] == "completed" and len(t["summary"] or "") >= 40
                and not t["summary"].startswith(_DIRECT)):
            fp = _NUM.sub("", _norm(t["summary"]))[:80]
            groups[fp].append(t)
    out = []
    for rows in groups.values():
        goals = {_norm(r["goal"]) for r in rows}
        if len(rows) >= 3 and len(goals) >= 2:
            out.append({"answer": rows[0]["summary"], "count": len(rows),
                        "goals": list(dict.fromkeys(r["goal"] for r in rows))[:4]})
    return sorted(out, key=lambda g: -g["count"])


def group_failures(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    c: Counter = Counter()
    example: Dict[str, str] = {}
    for t in tasks:
        if t["status"] in ("failed", "escalated"):
            first = next((ln for ln in (t["summary"] or "").splitlines() if ln.strip()), "(沒有說明)")
            key = _NUM.sub("#", _UUIDISH.sub("<id>", _clip(first, 90)))
            c[key] += 1
            example.setdefault(key, t["goal"])
    return [{"reason": k, "count": n, "example": example[k]} for k, n in c.most_common(5)]


async def check_skills(db, since: str) -> List[Dict[str, Any]]:
    rows = await _rows(db, (
        "SELECT i.skill_name, i.output_data, i.success, t.goal "
        "FROM skill_invocations i LEFT JOIN subtasks s ON s.id = i.subtask_id "
        "LEFT JOIN tasks t ON t.id = s.parent_task_id WHERE i.invoked_at >= ?"), (since,))
    try:
        from tool_registry import tool_registry
        tools = set(getattr(tool_registry, "_tools", {}))
    except Exception:  # noqa: BLE001
        tools = set()
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[r["skill_name"]].append(r)
    out = []
    for name, inv in by.items():
        outputs = {_norm(r["output_data"] or "")[:200] for r in inv}
        goals = {_norm(r["goal"] or "") for r in inv if r["goal"]}
        # The same output handed to different questions is the tell; two
        # different questions are enough once it has happened three times.
        hijack = len(inv) >= 3 and len(outputs) <= 2 and len(goals) >= 2
        out.append({"name": name, "uses": len(inv), "successes": sum(1 for r in inv if r["success"]),
                    "distinct_goals": len(goals), "same_output": len(outputs) <= 2,
                    "suspect": hijack or name in tools,
                    "why": ("same output for different questions" if hijack else
                            "named like a tool" if name in tools else "")})
    return sorted(out, key=lambda s: (not s["suspect"], -s["uses"]))


def _expected_runs(cron: str, since: datetime, until: datetime, created: Optional[datetime]) -> Optional[int]:
    """How many times `cron` should have fired in the window (None if unknown)."""
    try:
        from scheduler import _cron_to_trigger
        trig = _cron_to_trigger(cron)
    except Exception:  # noqa: BLE001
        return None
    start = max(since, created or since).replace(tzinfo=timezone.utc)
    end = until.replace(tzinfo=timezone.utc)
    n, prev, t = 0, None, trig.get_next_fire_time(None, start)
    while t and t <= end and n < 1000:
        n += 1
        prev, t = t, trig.get_next_fire_time(t, t + timedelta(seconds=1))
    return n


async def check_schedules(db, tasks: List[Dict[str, Any]], since: Optional[datetime] = None,
                          until: Optional[datetime] = None) -> List[Dict[str, Any]]:
    rows = await _rows(db, "SELECT id, goal, cron, enabled, consecutive_failures, last_status, last_error, "
                           "created_at FROM scheduled_tasks")
    until = until or datetime.utcnow()
    since = since or until - timedelta(days=7)
    same_time: Counter = Counter(r["cron"].strip() for r in rows if r["enabled"])
    runs: Dict[str, Counter] = defaultdict(Counter)
    for t in tasks:
        if t["origin"] == "schedule":
            runs[_norm(t["goal"])][t["status"]] += 1
    out = []
    for r in rows:
        c = runs.get(_norm(r["goal"]), Counter())
        ran = sum(c.values())
        expected = _expected_runs(r["cron"], since, until, _parse(r["created_at"])) if r["enabled"] else None
        missed = expected is not None and expected > 0 and ran < expected
        duplicate = bool(r["enabled"]) and same_time[r["cron"].strip()] > 1
        out.append({"id": r["id"][:8], "goal": r["goal"], "cron": r["cron"], "enabled": bool(r["enabled"]),
                    "ok": c.get("completed", 0), "failed": c.get("failed", 0) + c.get("escalated", 0),
                    "expected": expected, "missed": missed, "duplicate": duplicate,
                    "streak": int(r["consecutive_failures"] or 0), "last_error": r["last_error"] or "",
                    "problem": ((not r["enabled"] and bool(r["last_error"]))
                                or int(r["consecutive_failures"] or 0) > 0 or missed or duplicate)})
    return out


async def check_usage(db, since: str) -> List[Dict[str, Any]]:
    rows = await _rows(db, "SELECT model, COUNT(*) AS calls, COALESCE(SUM(duration_ms),0) AS ms, "
                           "COALESCE(SUM(tokens_out),0) AS tok FROM usage_records WHERE created_at >= ? "
                           "GROUP BY model ORDER BY ms DESC", (since,))
    return [{"model": r["model"] or "?", "calls": r["calls"], "minutes": round(r["ms"] / 60000, 1),
             "tokens_out": r["tok"]} for r in rows]


async def check_environment() -> List[Dict[str, Any]]:
    env: List[Dict[str, Any]] = []
    try:
        import scheduler
        state = "running" if scheduler.running() else ("stopped" if scheduler.available() else "disabled")
        env.append({"name": "排程器", "ok": state == "running", "detail": state,
                    "fix": "" if state == "running" else
                    ".venv/bin/pip install -r requirements.txt，然後重新啟動"})
    except Exception as exc:  # noqa: BLE001
        env.append({"name": "排程器", "ok": False, "detail": str(exc), "fix": ""})
    try:
        import importlib.util, os
        spec = importlib.util.spec_from_file_location(
            "check_requirements", os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "check_requirements.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        miss = mod.missing()
        env.append({"name": "套件", "ok": not miss, "detail": "齊全" if not miss else "缺少 " + "、".join(miss),
                    "fix": "" if not miss else "openteddy update 或 ./deploy.sh 會自動補裝"})
    except Exception as exc:  # noqa: BLE001
        logger.debug("requirements check failed: %s", exc)
    try:
        import home_assistant as HA
        if HA.configured():
            try:
                st = await HA.status()
                env.append({"name": "Home Assistant", "ok": True, "detail": f"{st.get('version')} · {st.get('entities')} 個項目"})
            except HA.HAError as exc:
                env.append({"name": "Home Assistant", "ok": False, "detail": str(exc), "fix": ""})
    except Exception:  # noqa: BLE001
        pass
    try:
        import httpx
        from config import config
        async with httpx.AsyncClient(timeout=3.0) as c:
            r = await c.get(config.gemma_base_url.rstrip("/") + "/api/tags")
        env.append({"name": "Ollama", "ok": r.status_code == 200, "detail": "連線正常" if r.status_code == 200 else f"HTTP {r.status_code}"})
    except Exception as exc:  # noqa: BLE001
        env.append({"name": "Ollama", "ok": False, "detail": f"連不到（{type(exc).__name__}）",
                    "fix": "sudo systemctl restart ollama"})
    return env


# ── Report ───────────────────────────────────────────────────────────────────

async def build_report(tracker: Any, days: int = 7) -> Dict[str, Any]:
    days = max(1, min(int(days or 7), 90))
    now = datetime.utcnow()
    since_dt = now - timedelta(days=days)
    since = since_dt.isoformat()
    db = tracker.db
    raw = await _rows(db, "SELECT id, goal, context, status, summary, created_at, completed_at, session_id "
                          "FROM tasks WHERE created_at >= ? ORDER BY created_at", (since,))
    tasks = []
    for r in raw:
        created = _parse(r["created_at"])
        if not created:
            continue
        done = _parse(r["completed_at"])
        tasks.append({**r, "created": created, "origin": _origin(r["context"]),
                      "seconds": (done - created).total_seconds() if done else None,
                      "goal": r["goal"] or "", "summary": r["summary"] or ""})
    # The report's own runs are left out of the answer checks — but still
    # counted as schedule runs (the check-up schedule once looked "missed").
    all_tasks = tasks
    tasks = [t for t in tasks if not t["summary"].startswith("🩺")]

    f: Dict[str, Any] = {
        "repeats": find_repeats(tasks), "give_up": find_give_ups(tasks),
        "same_answer": find_same_answers(tasks), "failures": group_failures(tasks),
        "skills": await check_skills(db, since), "schedules": await check_schedules(db, all_tasks, since_dt, now),
        "usage": await check_usage(db, since), "environment": await check_environment(),
    }
    status = Counter(t["status"] for t in tasks)
    secs = sorted(t["seconds"] for t in tasks if t["seconds"] is not None and t["status"] == "completed")
    f["overview"] = {
        "tasks": len(tasks), "status": dict(status),
        "origins": dict(Counter(t["origin"] for t in tasks)),
        "median_s": round(statistics.median(secs), 1) if secs else None,
        "slowest": [{"goal": t["goal"], "seconds": round(t["seconds"])} for t in
                    sorted((t for t in tasks if t["seconds"]), key=lambda t: -t["seconds"])[:3]],
    }
    return {"days": days, "since": since, "findings": f, "text": render(f, since_dt, now, days)}


def render(f: Dict[str, Any], since: datetime, now: datetime, days: int) -> str:
    o = f["overview"]
    st = o["status"]
    L: List[str] = [f"🩺 OpenTeddy 自我體檢 · {_local(since).split()[0]}–{_local(now).split()[0]}（{days} 天）"]
    L.append(f"任務 {o['tasks']} 個：完成 {st.get('completed', 0)} · 失敗 {st.get('failed', 0)}"
             f" · 升級 {st.get('escalated', 0)}" + (f" · 中位數 {o['median_s']} 秒" if o["median_s"] is not None else ""))
    issues = 0

    sus: List[str] = []
    for c in f["repeats"][:4]:
        when = _local(c["first"]) + (f" 起 {c['episodes']} 段時間" if c.get("episodes", 1) > 1 else "")
        sus.append(f"• 「{_clip(_goal(c['goal']), 24)}」{when}連問 {c['times']} 次 — 最後的回答：{_clip(c['answer'], 50)}")
    for g in f["same_answer"][:2]:
        qs = "、".join(f"「{_clip(_goal(x), 14)}」" for x in g["goals"][:3])
        sus.append(f"• {g['count']} 個不同的問題（{qs}）得到同一個回答：{_clip(g['answer'], 40)}")
    if f["give_up"]:
        ex = f["give_up"][0]
        sus.append(f"• {len(f['give_up'])} 個「完成」的回答其實在說做不到，例如「{_clip(_goal(ex['goal']), 20)}」→ {_clip(ex['answer'], 40)}")
    if sus:
        issues += len(sus)
        L += ["", f"**⚠️ 可能答錯的（{len(sus)}）**", *sus,
              "建議：把這些問題再問一次看是否已修好；一直答不好的，告訴我要補哪種工具或資料來源。"]

    bad_skills = [s for s in f["skills"] if s["suspect"]]
    if bad_skills:
        issues += len(bad_skills)
        L += ["", "**🧩 可疑的技能**"]
        for s in bad_skills:
            why = "同一個結果回答不同問題" if s["why"].startswith("same") else "名稱和工具相同"
            L.append(f"• `{s['name']}`：用了 {s['uses']} 次，涵蓋 {s['distinct_goals']} 種問題，{why}")
        L.append("建議：在 Capabilities 把它降級或刪除（`POST /skills/<name>/demote`）。")

    if f["failures"]:
        n = sum(x["count"] for x in f["failures"])
        issues += 1
        L += ["", f"**❌ 失敗（{n}）**"]
        for x in f["failures"][:3]:
            L.append(f"• {x['count']} 次：{x['reason']}（例：「{_clip(_goal(x['example']), 20)}」）")

    sch = f["schedules"]
    if sch:
        L += ["", "**⏰ 排程**"]
        for s in sch:
            flag = "⚠️ " if s["problem"] else ""
            state = "" if s["enabled"] else "（已停用）"
            line = f"• {flag}{_clip(s['goal'], 26)}{state}（`{s['id']}`）：這段期間成功 {s['ok']}、失敗 {s['failed']}"
            if s["missed"]:
                line += f"，應執行 {s['expected']} 次"
            if s["streak"]:
                line += f"，連續失敗 {s['streak']} 次"
            if s["duplicate"]:
                line += "，和另一個排程同一時間（可能重複）"
            if s["last_error"] and s["problem"]:
                line += f" — {_clip(s['last_error'], 60)}"
            L.append(line)
            issues += int(s["problem"])

    if f["usage"] or o["slowest"]:
        L += ["", "**⏱ 速度與運算**"]
        if f["usage"]:
            L.append("• 模型運算時間：" + "、".join(f"{u['model']} {u['minutes']} 分鐘（{u['calls']} 次）" for u in f["usage"][:3]))
        for s in o["slowest"]:
            if s["seconds"] >= 60:
                L.append(f"• 最慢：「{_clip(_goal(s['goal']), 22)}」{s['seconds']} 秒")

    env = f["environment"]
    if env:
        bad = [e for e in env if not e["ok"]]
        issues += len(bad)
        L += ["", "**🔧 環境**", "• " + " · ".join(f"{'✅' if e['ok'] else '❌'} {e['name']} {e['detail']}" for e in env)]
        for e in bad:
            if e.get("fix"):
                L.append(f"  修法：{e['fix']}")

    L += ["", "✅ 沒有發現問題。" if issues == 0 else f"共 {issues} 項需要注意。"]
    return "\n".join(L)


_ASK = re.compile(r"(自我體檢|系統體檢|體檢報告|做體檢|本週體檢|這週體檢|self[- ]?review|health ?report)", re.I)


def parse_review_request(goal: str) -> Optional[int]:
    """Days to cover when the message asks for the check-up, else None."""
    t = (goal or "").strip()
    if not t or len(t) > 40 or not _ASK.search(t):
        return None
    m = re.search(r"(\d{1,2}|[一二兩两三四五六七八九十])\s*(?:天|日|days?)", t, re.I)
    if m:
        cn = {"一": 1, "二": 2, "兩": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
              "七": 7, "八": 8, "九": 9, "十": 10}
        n = int(m.group(1)) if m.group(1).isdigit() else cn[m.group(1)]
        return max(1, min(n, 90))
    if re.search(r"今天|今日|today", t, re.I):
        return 1
    if re.search(r"這個月|本月|一個月|month", t, re.I):
        return 30
    return 7
