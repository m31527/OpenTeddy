"""
OpenTeddy Fleet — worker node (spoke).

Runs on nodes with OPENTEDDY_FLEET_ROLE=worker. Dials the central
orchestrator's WebSocket, registers, heartbeats, and — crucially —
runs any dispatched goal on THIS node's existing single-machine
orchestrator.run(). Fleet is a thin shell around the mature agent, not
a reimplementation of it.

Lifecycle (started from main.py lifespan only when role=worker):
    start_fleet_worker()  → connect loop runs as a background task
    stop_fleet_worker()   → cancels it on shutdown

Resilience: the connect loop auto-reconnects with backoff, so a central
restart / transient network blip doesn't permanently detach the worker.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

from . import protocol as P

logger = logging.getLogger("openteddy.fleet.worker")


# ── Config ────────────────────────────────────────────────────────────────────
def _central_url() -> str:
    """ws:// URL of the central orchestrator. Operator sets
    OPENTEDDY_FLEET_CENTRAL (e.g. ws://dgx-01.local:8770)."""
    return (os.getenv("OPENTEDDY_FLEET_CENTRAL", "ws://127.0.0.1:8770") or "").strip()


def _node_id() -> str:
    """Stable id for this node. Defaults to the hostname so the central's
    registry shows human-meaningful names; override with
    OPENTEDDY_FLEET_NODE_ID."""
    explicit = os.getenv("OPENTEDDY_FLEET_NODE_ID")
    if explicit:
        return explicit.strip()
    import socket
    return socket.gethostname()


def _node_role() -> str:
    """This node's functional role in the fleet (finance / external /
    secops / …). Free-form label the central can route on later.
    Defaults to 'worker'."""
    return (os.getenv("OPENTEDDY_FLEET_NODE_ROLE", "worker") or "worker").strip()


_HEARTBEAT_INTERVAL_S = 15.0
_RECONNECT_BACKOFF_START = 2.0
_RECONNECT_BACKOFF_MAX = 30.0


class FleetWorker:
    """Dials the central, services dispatches. One per worker process."""

    def __init__(self) -> None:
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._running_tasks: Dict[str, asyncio.Task] = {}

    # ── Connect loop with backoff ────────────────────────────────────────────

    async def run(self) -> None:
        backoff = _RECONNECT_BACKOFF_START
        while not self._stop.is_set():
            try:
                await self._connect_once()
                backoff = _RECONNECT_BACKOFF_START  # healthy session → reset
            except Exception as exc:  # noqa: BLE001
                logger.info("fleet worker: connection ended (%s); "
                            "reconnecting in %.0fs", exc, backoff)
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_BACKOFF_MAX)

    async def _connect_once(self) -> None:
        import websockets  # lazy — only a worker needs the dep
        url = _central_url()
        async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
            # ── Register ──
            caps_tools, caps_model = _local_capabilities()
            await ws.send(json.dumps(P.make_register(
                _node_id(), _node_role(),
                tools=caps_tools, model=caps_model,
            )))
            raw = await asyncio.wait_for(ws.recv(), timeout=15)
            ack = P.parse_frame(json.loads(raw))
            if ack.get("type") != P.MSG_REGISTER_ACK or not ack.get("assigned"):
                reason = ack.get("reason", "unknown")
                logger.error("fleet worker: registration rejected — %s", reason)
                # Bad token / config — back off hard rather than hammer.
                await asyncio.sleep(_RECONNECT_BACKOFF_MAX)
                return
            logger.info("fleet worker: registered with central as id=%s role=%s",
                        _node_id(), _node_role())

            # ── Heartbeat + watcher + receive concurrently ──
            hb_task = asyncio.create_task(self._heartbeat_loop(ws))
            # Watcher pushes proactive alerts over THIS socket. It only
            # actually starts if OPENTEDDY_FLEET_WATCH_ENABLED is set;
            # otherwise start() is a no-op. We give it a send closure
            # bound to this connection so a reconnect gets a fresh one.
            watcher = None
            try:
                from .watcher import FleetWatcher

                async def _send(frame: Dict[str, Any]) -> None:
                    await ws.send(json.dumps(frame))

                watcher = FleetWatcher(_node_id(), _send, _get_local_orchestrator)
                watcher.start()
            except Exception as exc:  # noqa: BLE001
                logger.debug("fleet worker: watcher not started: %s", exc)
            try:
                async for raw in ws:
                    try:
                        frame = P.parse_frame(json.loads(raw))
                    except (P.ProtocolError, json.JSONDecodeError) as exc:
                        logger.debug("fleet worker: bad frame: %s", exc)
                        continue
                    await self._on_frame(ws, frame)
            finally:
                hb_task.cancel()
                if watcher is not None:
                    await watcher.stop()

    async def _heartbeat_loop(self, ws: Any) -> None:
        while not self._stop.is_set():
            status = "busy" if self._running_tasks else "idle"
            try:
                await ws.send(json.dumps(P.make_heartbeat(
                    _node_id(), status=status, ts=time.time(),
                    load={"running_tasks": len(self._running_tasks)},
                )))
            except Exception:  # noqa: BLE001
                return  # connection dead — outer loop will reconnect
            await asyncio.sleep(_HEARTBEAT_INTERVAL_S)

    # ── Frame handling ───────────────────────────────────────────────────────

    async def _on_frame(self, ws: Any, frame: Dict[str, Any]) -> None:
        mtype = frame.get("type")

        if mtype == P.MSG_DISPATCH:
            task_id = frame.get("task_id") or str(uuid.uuid4())
            # Run the goal in its own task so we keep heartbeating + can
            # accept a cancel while it executes.
            t = asyncio.create_task(self._run_dispatched(ws, frame))
            self._running_tasks[task_id] = t
            t.add_done_callback(lambda _t, tid=task_id: self._running_tasks.pop(tid, None))
            return

        if mtype == P.MSG_CANCEL:
            task_id = frame.get("task_id")
            t = self._running_tasks.get(task_id)
            if t and not t.done():
                t.cancel()
            return

    async def _run_dispatched(self, ws: Any, frame: Dict[str, Any]) -> None:
        """Run a dispatched goal on THIS node's existing orchestrator and
        report the result. This is the whole point of fleet: reuse the
        single-machine agent unchanged."""
        task_id = frame.get("task_id")
        goal = frame.get("goal") or ""
        mode = frame.get("mode") or "code"
        context = frame.get("context") or {}

        status = "failed"
        summary = ""
        artifacts: list = []
        try:
            # Import here so a worker that never receives a dispatch (or a
            # single-machine install that never starts a worker) doesn't
            # eagerly pull the orchestrator graph at module load.
            from orchestrator import Orchestrator  # noqa
            from models import TaskRequest, SessionMode

            orch = _get_local_orchestrator()
            req = TaskRequest(
                id=task_id or str(uuid.uuid4()),
                goal=goal,
                context={**context, "triggered_by": "fleet"},
                priority=1,
                session_id=context.get("session_id") or f"fleet-{task_id[:8]}",
                mode=SessionMode(mode) if mode in (m.value for m in SessionMode) else SessionMode.CODE,
            )
            result = await orch.run(req)
            # TaskStatus is a str-Enum ("completed" / "failed" / …); compare
            # its string value so we don't depend on the enum identity.
            raw_status = str(getattr(result, "status", "") or "").lower()
            status = "completed" if "completed" in raw_status else "failed"
            summary = getattr(result, "summary", "") or ""
        except asyncio.CancelledError:
            status = "failed"
            summary = "⏹️ task cancelled by central"
            raise
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            summary = f"fleet worker error: {exc}"
            logger.exception("fleet worker: dispatched task failed")
        finally:
            try:
                await ws.send(json.dumps(P.make_result(
                    task_id, _node_id(),
                    status=status, summary=summary, artifacts=artifacts,
                )))
            except Exception:  # noqa: BLE001
                logger.debug("fleet worker: failed to send result (conn gone)")

    # ── Start / stop ─────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run(), name="fleet_worker")

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            self._task = None


# ── Local helpers ─────────────────────────────────────────────────────────────


def _local_capabilities() -> tuple:
    """Best-effort (tools, model) self-description sent at register. Kept
    defensive — never let capability probing block registration."""
    tools: list = []
    model = ""
    try:
        from config import config
        model = getattr(config, "qwen_model", "")
    except Exception:  # noqa: BLE001
        pass
    try:
        from tool_registry import tool_registry
        tools = [name for name in getattr(tool_registry, "_tools", {}).keys()]
    except Exception:  # noqa: BLE001
        pass
    return tools, model


_LOCAL_ORCH: Any = None


def _get_local_orchestrator() -> Any:
    """Reuse the same Orchestrator the single-machine app builds, if one
    exists; otherwise construct a default. We import main lazily and look
    for its module-level orchestrator so a worker shares the exact same
    tool registry / memory / config as a local run would."""
    global _LOCAL_ORCH
    if _LOCAL_ORCH is not None:
        return _LOCAL_ORCH
    try:
        import main as _main
        cand = getattr(_main, "orchestrator", None)
        if cand is not None:
            _LOCAL_ORCH = cand
            return _LOCAL_ORCH
    except Exception:  # noqa: BLE001
        pass
    from orchestrator import Orchestrator
    _LOCAL_ORCH = Orchestrator()
    return _LOCAL_ORCH


# ── Module-level singleton + lifespan hooks ───────────────────────────────────

_INSTANCE: Optional[FleetWorker] = None


async def start_fleet_worker() -> Optional[FleetWorker]:
    """Called from main.py lifespan when role=worker. Idempotent."""
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE
    if not P.fleet_token():
        logger.error(
            "fleet worker: OPENTEDDY_FLEET_TOKEN is not set — central will "
            "reject registration (token fails closed). Set the shared token "
            "and restart.")
        # Still create the instance so stop() is safe; it just won't auth.
    inst = FleetWorker()
    inst.start()
    _INSTANCE = inst
    logger.info("fleet worker: started, dialing %s", _central_url())
    return inst


async def stop_fleet_worker() -> None:
    global _INSTANCE
    if _INSTANCE is not None:
        await _INSTANCE.stop()
        _INSTANCE = None
