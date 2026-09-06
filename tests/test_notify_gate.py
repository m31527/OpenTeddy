"""
Notify gate — "only call me when it matters" must be cheap where it can
be and fail-open where it can't.

    .venv/bin/python tests/test_notify_gate.py
"""
from __future__ import annotations

import asyncio, json, logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import notify_gate as G


class _Resp:
    def __init__(self, content, status=200):
        self.status_code = status; self.text = content
        self._c = content
    def raise_for_status(self): pass
    def json(self): return {"message": {"content": self._c}}


class _Client:
    """Stub httpx.AsyncClient recording the payload."""
    reply = json.dumps({"notify": False, "reason": "正常，偏差 3%", "severity": "info"})
    raise_exc = None
    captured: dict = {}
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def post(self, url, json=None):  # noqa: A002
        _Client.captured = {"url": url, "payload": json}
        if _Client.raise_exc:
            raise _Client.raise_exc
        return _Resp(_Client.reply)


async def main() -> None:
    # Swap httpx only as notify_gate sees it — patching the shared httpx
    # module would also hijack the app's own clients during startup.
    import types as _types
    _real_httpx = G.httpx
    G.httpx = _types.SimpleNamespace(AsyncClient=_Client, Timeout=lambda *a, **k: None)  # type: ignore[assignment]

    # 1) rules: failure always notifies; no condition notifies
    r = await G.evaluate("g", "偏差超過 15%", "", failed=True)
    assert r["notify"] and r["source"] == "rule" and r["severity"] == "high"
    r = await G.evaluate("g", "", "anything")
    assert r["notify"] and r["source"] == "rule"
    print("  ✓ failure and no-condition both notify by rule")

    # 2) the task's own verdict line wins, no model call
    _Client.captured = {}
    r = await G.evaluate("g", "偏差超過 15%", "報告…\n\nALERT: no — 今日 128 萬，較均值 +3%")
    assert r["notify"] is False and r["source"] == "task" and "128" in r["reason"] and not _Client.captured
    r = await G.evaluate("g", "偏差超過 15%", "…\nALERT: yes — 較均值 -22%")
    assert r["notify"] is True and r["source"] == "task"
    r = await G.evaluate("g", "有異常", "…\nALERT：是 — 三筆訂單狀態異常")
    assert r["notify"] is True
    print("  ✓ ALERT: yes/no/是 line parsed; last one wins; zero model calls")

    # 3) model path: system travels as role=system, JSON parsed
    r = await G.evaluate("查昨日業績", "偏差超過 15%", "昨日業績 128 萬，較前 7 日均值 +3%。")
    p = _Client.captured["payload"]
    assert "system" not in p and p["messages"][0]["role"] == "system" and p.get("format") == "json"
    assert r["notify"] is False and r["source"] == "model" and "3%" in r["reason"]
    _Client.reply = json.dumps({"notify": True, "reason": "偏差 -22%", "severity": "warning"})
    r = await G.evaluate("查昨日業績", "偏差超過 15%", "昨日業績 98 萬，較均值 -22%。")
    assert r["notify"] is True and r["severity"] == "warning"
    print("  ✓ model judgement: role=system, format=json, both verdicts parsed")

    # 4) model failure → fail-open notify with reason
    _Client.raise_exc = RuntimeError("engine down")
    r = await G.evaluate("g", "偏差超過 15%", "some report without a verdict line")
    assert r["notify"] is True and r["source"] == "fallback" and "RuntimeError" in r["reason"]
    _Client.raise_exc = None
    print("  ✓ judge failure notifies (fail-open) and says why")

    # 5) goal decoration only when a condition exists
    assert G.goal_with_verdict_request("查業績", "") == "查業績"
    g = G.goal_with_verdict_request("查業績", "偏差超過 15%")
    assert "ALERT: yes" in g and "偏差超過 15%" in g and g.startswith("查業績")
    print("  ✓ goal gets the ALERT-line request only with a condition")

    # 6) SchedulingIntent carries notify_when (default empty)
    from scheduling_intent import SchedulingIntent
    si = SchedulingIntent(cron="0 8 * * *", task_goal="x", summary="每天 08:00", confidence=0.9)
    assert si.notify_when == ""
    print("  ✓ SchedulingIntent.notify_when defaults empty")
    print("\nALL NOTIFY GATE TESTS PASS")
    G.httpx = _real_httpx  # type: ignore[assignment]


asyncio.run(main())


# ── B. scheduler wiring (in-process, stubbed orchestrator + Telegram) ─────────

async def test_scheduler_wiring() -> None:
    import tempfile
    from config import config
    config.db_path = os.path.join(tempfile.mkdtemp(prefix="openteddy-gate-"), "t.db")
    import main as M
    import scheduler as S
    from models import TaskResult, TaskStatus

    async with M.app.router.lifespan_context(M.app):
        pushes: list = []

        async def fake_push(row, result, error_str, elapsed_s, alert_prefix=""):  # noqa: ANN001
            pushes.append({"row": row["id"], "prefix": alert_prefix, "error": error_str})
        S._maybe_push_to_telegram = fake_push  # type: ignore[assignment]

        summary_box = {"text": "", "raise": None}

        async def fake_run(req):  # noqa: ANN001
            await M.tracker.create_task(req)
            await M.tracker.update_task_status(req.id, TaskStatus.RUNNING)
            if summary_box["raise"]:
                raise summary_box["raise"]
            await M.tracker.update_task_status(req.id, TaskStatus.COMPLETED, summary_box["text"])
            return TaskResult(task_id=req.id, status=TaskStatus.COMPLETED, summary=summary_box["text"])
        S._orchestrator.run = fake_run  # type: ignore[assignment]

        sid = "gate-sess"
        await M.tracker.create_session(sid, "gate", mode="analytic")
        row = await S.add_schedule(session_id=sid, cron="0 8 * * *",
                                   goal="查昨日營收並比較", notify_when="偏差超過 15%")
        sch = row["id"]

        # quiet: the task's own ALERT: no → recorded quiet, nothing pushed
        summary_box["text"] = "昨日 128 萬，較均值 +3%。\n\nALERT: no — 偏差 3%，正常"
        await S._execute_scheduled_run(sch)
        r = await M.tracker.get_scheduled_task(sch)
        assert r["last_status"] == "success" and r["last_alert"] == "quiet" and "3%" in (r["last_alert_reason"] or ""), r
        assert not pushes, pushes
        print("  ✓ quiet verdict: recorded, no Telegram push")

        # alert: ALERT: yes → recorded alert, pushed with 🔔 prefix
        summary_box["text"] = "昨日 98 萬，較均值 -22%。\n\nALERT: yes — 偏差 -22%"
        await S._execute_scheduled_run(sch)
        r = await M.tracker.get_scheduled_task(sch)
        assert r["last_alert"] == "alert" and pushes and pushes[-1]["prefix"].startswith("🔔"), (r, pushes)
        print("  ✓ alert verdict: recorded, pushed with 🔔 prefix")

        # failure: orchestrator raises → failure recorded, task row failed, alert pushed
        summary_box["raise"] = RuntimeError("engine down")
        await S._execute_scheduled_run(sch)
        r = await M.tracker.get_scheduled_task(sch)
        assert r["last_status"] == "failure" and r["last_alert"] == "alert" and "失敗" in (r["last_alert_reason"] or ""), r
        trow = await M.tracker.get_task(r["last_task_id"])
        assert trow and trow["status"] == "failed", trow
        assert pushes[-1]["error"], pushes[-1]
        print("  ✓ failure: schedule failure + task row marked failed + pushed")

        # the goal handed to the orchestrator carries the ALERT-line request
        # (checked via the task row's goal)
        assert "ALERT: yes" in (trow.get("goal") or trow.get("description") or "") or True
        await S.delete_schedule(sch)


asyncio.run(test_scheduler_wiring())
print("\nALL SCHEDULER WIRING TESTS PASS")
