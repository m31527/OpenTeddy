"""
Task API v1 tests — the runtime contract every interface relies on.

  A. tool_registry: tool scope is enforced at execution, unattended
     policy needs a scope, denylist still wins, approval_request event.
  B. HTTP: POST /tasks accepts-and-returns, binds to an agent, refuses
     unattended runs without a scope, streams task.done over SSE,
     wait=true blocks, privacy tightens, /run still works, /models.

Standalone runner (no pytest in this repo):
    .venv/bin/python tests/test_task_api.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

# Isolate persistence BEFORE main is imported (Tracker binds config.db_path
# at construction).
from config import config  # noqa: E402

_TMP = tempfile.mkdtemp(prefix="openteddy-taskapi-")
config.db_path = os.path.join(_TMP, "tracker.db")

from tool_registry import ToolRegistry, make_result  # noqa: E402
from approval_store import ApprovalStore  # noqa: E402
from tools._context import reset_triggered_by, set_triggered_by  # noqa: E402


def _schema(name: str) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": "test tool",
        "parameters": {"type": "object", "properties": {}}}}


async def low_tool(**kw):
    return make_result(True, result="low")


async def high_tool(cmd: str = "", **kw):
    return make_result(True, result=f"high:{cmd}")


# ── A. registry ──────────────────────────────────────────────────────────────

async def test_registry() -> None:
    reg = ToolRegistry()
    reg._store = ApprovalStore()
    reg.register(low_tool, _schema("low_tool"), "low")
    reg.register(high_tool, _schema("high_tool"), "high")
    events: list = []

    async def on_approval(ev):  # noqa: ANN001
        events.append(ev)
    reg.on_approval = on_approval

    # scope enforcement, independent of risk
    r = await reg.execute("low_tool", {}, allowed_tools=["other_tool"])
    assert r["success"] is False and "outside" in r["error"], r
    r = await reg.execute("low_tool", {}, allowed_tools=["low_tool"])
    assert r["success"] is True
    r = await reg.execute("low_tool", {}, allowed_tools=None)
    assert r["success"] is True
    print("  ✓ scope: out-of-scope refused, in-scope and unscoped run")

    # unattended origin WITH scope → high-risk runs, no approval created
    tok = set_triggered_by("schedule")
    try:
        r = await reg.execute("high_tool", {"cmd": "ls"}, allowed_tools=["high_tool"])
        assert r["success"] is True and not await reg._store.get_pending(), r
        # denylist still wins even when unattended
        r = await reg.execute("high_tool", {"cmd": "rm -rf /"}, allowed_tools=["high_tool"])
        assert r["success"] is False and "Blocked" in r["error"], r
        # unattended origin WITHOUT scope → gated (approval created, event fired)
        async def _no(*a, **k):
            return False
        reg._store.wait_for_resolution = _no
        r = await reg.execute("high_tool", {"cmd": "ls"}, allowed_tools=None)
        assert r["success"] is False and "rejected" in r["error"].lower(), r
        assert events and events[-1]["event"] == "approval_request" \
            and events[-1]["tool"] == "high_tool"
    finally:
        reset_triggered_by(tok)
    print("  ✓ unattended: scoped schedule auto-runs, denylist blocks, unscoped is gated + event")

    tok = set_triggered_by("api_trusted")
    try:
        r = await reg.execute("high_tool", {"cmd": "ok"}, allowed_tools=["high_tool"])
        assert r["success"] is True
    finally:
        reset_triggered_by(tok)
    print("  ✓ api_trusted + scope auto-approves")


# ── B. HTTP ──────────────────────────────────────────────────────────────────

async def test_http() -> None:
    import main as M
    from httpx import ASGITransport, AsyncClient
    from models import TaskResult, TaskStatus

    async with M.app.router.lifespan_context(M.app):
        calls: list = []

        async def fake_run(req):  # noqa: ANN001
            # Stand-in for the orchestrator: persist like the real one and
            # emit the lifecycle events the SSE stream keys on.
            calls.append(req)
            await M.tracker.create_task(req)
            await M.tracker.update_task_status(req.id, TaskStatus.RUNNING)
            await M.ws_manager.broadcast({"type": "task.started", "task_id": req.id})
            await asyncio.sleep(0.05)
            await M.tracker.update_task_status(req.id, TaskStatus.COMPLETED, "ok:" + req.goal)
            await M.ws_manager.broadcast({"type": "task.done", "task_id": req.id,
                                          "status": "completed", "summary": "ok"})
            return TaskResult(task_id=req.id, status=TaskStatus.COMPLETED,
                              summary="ok:" + req.goal)
        M.orchestrator.run = fake_run
        created_sessions: list = []

        async with AsyncClient(transport=ASGITransport(app=M.app), base_url="http://t") as c:
            names = [t["name"] for t in (await c.get("/tools")).json()["tools"]]
            pick = "db_query" if "db_query" in names else names[0]

            r = await c.post("/agents", json={"name": "bad", "mode": "analytic",
                                              "allowed_tools": ["nope_tool"]})
            assert r.status_code == 400 and "nope_tool" in r.text, r.text
            r = await c.post("/agents", json={"name": "scoped", "mode": "analytic",
                                              "allowed_tools": [pick]})
            assert r.status_code == 200, r.text
            scoped = r.json()["agent"]
            assert scoped["allowed_tools"] == [pick], scoped
            r = await c.post("/agents", json={"name": "open", "mode": "analytic"})
            open_agent = r.json()["agent"]
            print("  ✓ agents: unknown tool → 400, scope stored and returned")

            r = await c.post("/tasks", json={"intent": "x", "agent_id": open_agent["id"],
                                             "require_approval": False})
            assert r.status_code == 400 and "scope" in r.text, r.text
            print("  ✓ POST /tasks require_approval=false without scope → 400")

            r = await c.post("/tasks", json={"intent": "do it", "agent_id": scoped["id"],
                                             "require_approval": False})
            assert r.status_code == 202, r.text
            d = r.json(); assert d["status"] == "running" and d["task_id"] and d["session_id"]
            created_sessions.append(d["session_id"])
            sess = await M.tracker.get_session(d["session_id"])
            assert sess and sess.get("agent_id") == scoped["id"], sess
            print("  ✓ POST /tasks returns immediately; session built from agent")

            body = ""
            async with c.stream("GET", f"/tasks/{d['task_id']}/events") as resp:
                assert resp.headers["content-type"].startswith("text/event-stream")
                async for chunk in resp.aiter_text():
                    body += chunk
                    if "task.done" in body:
                        break
            assert "task.done" in body, body[:300]
            print("  ✓ GET /tasks/{id}/events streams and closes on task.done")

            r = await c.post("/tasks", json={"intent": "sync please", "wait": True})
            assert r.status_code == 202 and r.json()["status"] == "completed", r.text
            assert "sync please" in r.json()["message"]
            created_sessions.append(r.json()["session_id"])
            print("  ✓ wait=true blocks and returns the final status")

            r = await c.post("/tasks", json={"intent": "private", "privacy": "local_only",
                                             "wait": True})
            s2 = await M.tracker.get_session(r.json()["session_id"])
            created_sessions.append(r.json()["session_id"])
            assert s2 and bool(s2.get("local_only")), s2
            assert any(req.context.get("privacy") == "local_only" for req in calls)
            print("  ✓ privacy=local_only tightens the new session and rides in context")

            r = await c.post("/run", json={"goal": "legacy"})
            assert r.status_code == 202 and r.json()["status"] == "completed", r.text
            print("  ✓ POST /run still works (thin wrapper)")

            r = await c.get(f"/tasks/{d['task_id']}")
            assert r.status_code == 200 and r.json().get("session_id") == d["session_id"], r.text
            r = await c.get("/models")
            assert r.status_code == 200 and "engine" in r.json() and "executor" in r.json()
            print("  ✓ GET /tasks/{id} carries session_id; GET /models answers")

            # tidy: agents + sessions this test created
            for a in (scoped, open_agent):
                await c.delete(f"/agents/{a['id']}")
            for sid in created_sessions:
                try:
                    await M.tracker.delete_session(sid)
                except Exception:  # noqa: BLE001
                    pass


async def main() -> None:
    print("A. registry")
    await test_registry()
    print("B. http")
    await test_http()
    print("\nALL TASK API TESTS PASS")


if __name__ == "__main__":
    asyncio.run(main())
