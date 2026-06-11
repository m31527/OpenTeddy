"""
OpenTeddy Fleet — central orchestrator (hub).

Runs on the node with OPENTEDDY_FLEET_ROLE=orchestrator. Opens a
WebSocket server that worker nodes dial into; tracks who's connected
+ alive; dispatches goals to a chosen worker and collects the result.

Lifecycle (started from main.py's lifespan only when role=orchestrator):
    start_fleet_orchestrator()  → binds the WS server as a background task
    stop_fleet_orchestrator()   → cancels it on shutdown

Public API used by the rest of OpenTeddy (e.g. an operator-facing HTTP
endpoint or the chat UI):
    registry()                      → snapshot of known nodes + status
    dispatch(node_id, goal, ...)    → send a task, await its result

This module imports `websockets` lazily inside start_* so a
single-machine install (which never calls these) doesn't even require
the dependency to be present.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from . import protocol as P

logger = logging.getLogger("openteddy.fleet.orchestrator")


# ── Defaults ──────────────────────────────────────────────────────────────────
import os

_BIND_HOST = os.getenv("OPENTEDDY_FLEET_HOST", "0.0.0.0")
_BIND_PORT = int(os.getenv("OPENTEDDY_FLEET_PORT", "8770"))
# A node is considered offline if we haven't seen a heartbeat within this
# many seconds. Workers heartbeat every 15s, so 3 missed = ~45s grace.
_HEARTBEAT_TIMEOUT_S = float(os.getenv("OPENTEDDY_FLEET_HEARTBEAT_TIMEOUT", "45"))
_SERVER_VERSION = "openteddy-fleet-orchestrator/0"


# ── Connected-node state ──────────────────────────────────────────────────────


class _Node:
    """Live state for one connected worker. Held in memory only — the
    registry is rebuilt as workers reconnect, so a central restart loses
    nothing important (workers re-register within one heartbeat)."""

    __slots__ = ("node_id", "role", "ws", "capabilities", "status",
                 "last_seen", "last_load")

    def __init__(self, node_id: str, role: str, ws: Any,
                 capabilities: Dict[str, Any]):
        self.node_id = node_id
        self.role = role
        self.ws = ws
        self.capabilities = capabilities
        self.status = "idle"
        self.last_seen = 0.0      # monotonic seconds; set on each heartbeat
        self.last_load: Dict[str, Any] = {}


class FleetOrchestrator:
    """The hub. One instance per orchestrator process, created by
    start_fleet_orchestrator()."""

    def __init__(self) -> None:
        self._nodes: Dict[str, _Node] = {}
        self._server: Any = None
        self._server_task: Optional[asyncio.Task] = None
        # task_id → Future that resolves when the worker sends a result.
        self._pending: Dict[str, asyncio.Future] = {}

    # ── Registry / status ────────────────────────────────────────────────────

    def registry(self) -> List[Dict[str, Any]]:
        """Snapshot of all known nodes for status display. Marks each
        node online/offline by heartbeat recency."""
        now = _monotonic()
        out: List[Dict[str, Any]] = []
        for n in self._nodes.values():
            online = (now - n.last_seen) <= _HEARTBEAT_TIMEOUT_S if n.last_seen else False
            out.append({
                "node_id":      n.node_id,
                "role":         n.role,
                "status":       n.status,
                "online":       online,
                "model":        n.capabilities.get("model", ""),
                "tools":        n.capabilities.get("tools", []),
                "last_seen_s":  round(now - n.last_seen, 1) if n.last_seen else None,
                "load":         n.last_load,
            })
        return out

    # ── Auto node selection ──────────────────────────────────────────────────

    def pick_node(self, role: Optional[str] = None) -> Optional[str]:
        """Choose a node to run a goal when the operator didn't pick one.

        Policy:
          1. Consider only ONLINE nodes (recent heartbeat). Optionally
             filter to a `role`.
          2. Prefer an IDLE node (status != busy AND 0 running tasks).
             Among idle nodes, any is fine — pick the one with the
             smallest running_tasks (ties → first registered).
          3. If none are idle, fall back to the LEAST-LOADED online node
             (smallest running_tasks) so the goal queues on whoever has
             the shortest line rather than failing.
        Returns a node_id, or None if there are no eligible online nodes.
        """
        now = _monotonic()
        online = [
            n for n in self._nodes.values()
            if n.last_seen and (now - n.last_seen) <= _HEARTBEAT_TIMEOUT_S
            and (role is None or n.role == role)
        ]
        if not online:
            return None

        def _running(n: "_Node") -> int:
            try:
                return int((n.last_load or {}).get("running_tasks", 0))
            except (TypeError, ValueError):
                return 0

        idle = [n for n in online if n.status != "busy" and _running(n) == 0]
        pool = idle if idle else online
        # Smallest running_tasks wins; stable order preserves registration
        # order for ties (dict preserves insertion order).
        best = min(pool, key=_running)
        return best.node_id

    # ── Dispatch ─────────────────────────────────────────────────────────────

    async def dispatch(
        self,
        node_id: str,
        goal: str,
        *,
        mode: str = "code",
        context: Optional[Dict[str, Any]] = None,
        timeout_s: float = 600.0,
    ) -> Dict[str, Any]:
        """Send a goal to `node_id` and await its result frame.

        Returns the result dict ({status, summary, artifacts, ...}) or
        raises on unknown node / offline node / timeout. The caller
        (operator endpoint) turns exceptions into user-facing errors.
        """
        node = self._nodes.get(node_id)
        if node is None:
            raise ValueError(f"unknown node: {node_id!r}")
        now = _monotonic()
        if node.last_seen and (now - node.last_seen) > _HEARTBEAT_TIMEOUT_S:
            raise ValueError(f"node {node_id!r} is offline (no recent heartbeat)")

        task_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[task_id] = fut

        frame = P.make_dispatch(task_id, goal, mode=mode, context=context)
        try:
            await node.ws.send(json.dumps(frame))
        except Exception as exc:  # noqa: BLE001
            self._pending.pop(task_id, None)
            raise ValueError(f"failed to send to {node_id!r}: {exc}") from exc

        try:
            return await asyncio.wait_for(fut, timeout=timeout_s)
        except asyncio.TimeoutError:
            self._pending.pop(task_id, None)
            raise TimeoutError(
                f"node {node_id!r} did not return a result for task "
                f"{task_id[:8]} within {timeout_s:.0f}s"
            )

    # ── Connection handler ───────────────────────────────────────────────────

    async def _handle_connection(self, ws: Any) -> None:
        """One coroutine per connected worker. First frame MUST be a
        valid register with a good token, else we close immediately."""
        node: Optional[_Node] = None
        try:
            # ── Registration handshake ──
            raw = await asyncio.wait_for(ws.recv(), timeout=15)
            frame = P.parse_frame(json.loads(raw))
            if frame.get("type") != P.MSG_REGISTER:
                await ws.send(json.dumps(P.make_register_ack(
                    assigned=False, server_version=_SERVER_VERSION,
                    reason="first frame must be 'register'")))
                return
            if not P.verify_token(frame.get("token")):
                logger.warning("fleet: rejected node %s — bad token",
                               frame.get("node_id"))
                await ws.send(json.dumps(P.make_register_ack(
                    assigned=False, server_version=_SERVER_VERSION,
                    reason="invalid token")))
                return

            node_id = str(frame.get("node_id") or "").strip()
            role = str(frame.get("role") or "worker").strip()
            if not node_id:
                await ws.send(json.dumps(P.make_register_ack(
                    assigned=False, server_version=_SERVER_VERSION,
                    reason="missing node_id")))
                return

            node = _Node(node_id, role, ws, frame.get("capabilities") or {})
            node.last_seen = _monotonic()
            self._nodes[node_id] = node
            await ws.send(json.dumps(P.make_register_ack(
                assigned=True, server_version=_SERVER_VERSION)))
            logger.info("fleet: node registered — id=%s role=%s model=%s",
                        node_id, role, node.capabilities.get("model", "?"))

            # ── Frame loop ──
            async for raw in ws:
                try:
                    frame = P.parse_frame(json.loads(raw))
                except (P.ProtocolError, json.JSONDecodeError) as exc:
                    logger.debug("fleet: dropped bad frame from %s: %s",
                                 node_id, exc)
                    continue
                await self._on_frame(node, frame)

        except asyncio.TimeoutError:
            logger.debug("fleet: registration timed out, closing connection")
        except Exception as exc:  # noqa: BLE001
            logger.debug("fleet: connection error: %s", exc)
        finally:
            if node is not None and self._nodes.get(node.node_id) is node:
                # Only drop if THIS connection still owns the slot — a
                # reconnect may have already replaced us.
                del self._nodes[node.node_id]
                logger.info("fleet: node disconnected — id=%s", node.node_id)

    async def _on_frame(self, node: _Node, frame: Dict[str, Any]) -> None:
        """Route a post-registration frame from a worker."""
        mtype = frame.get("type")

        if mtype == P.MSG_HEARTBEAT:
            node.last_seen = _monotonic()
            node.status = frame.get("status") or node.status
            node.last_load = frame.get("load") or {}
            return

        if mtype == P.MSG_RESULT:
            task_id = frame.get("task_id")
            fut = self._pending.pop(task_id, None)
            if fut is not None and not fut.done():
                fut.set_result(frame)
            else:
                logger.debug("fleet: result for unknown/expired task %s",
                             task_id)
            return

        if mtype == P.MSG_ALERT:
            # Milestone 2 — for now just log; later this fans out to
            # operator notification channels (Telegram / Slack / UI toast).
            logger.warning("fleet ALERT from %s [%s]: %s (conf=%.2f)",
                           node.node_id, frame.get("severity"),
                           frame.get("observation"), frame.get("confidence", 0))
            return

        logger.debug("fleet: ignoring unexpected frame type %s from %s",
                     mtype, node.node_id)

    # ── Server start / stop ──────────────────────────────────────────────────

    async def start(self) -> None:
        import websockets  # lazy — only needed on an orchestrator node
        if not P.fleet_token():
            logger.error(
                "fleet: OPENTEDDY_FLEET_TOKEN is not set — refusing to start "
                "the orchestrator (would accept no workers anyway, since "
                "verify_token fails closed). Set a shared token and restart.")
            return
        self._server = await websockets.serve(
            self._handle_connection, _BIND_HOST, _BIND_PORT,
            ping_interval=20, ping_timeout=20,
        )
        logger.info("fleet: orchestrator listening on ws://%s:%d",
                    _BIND_HOST, _BIND_PORT)

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        logger.info("fleet: orchestrator stopped")


def _monotonic() -> float:
    import time
    return time.monotonic()


# ── Module-level singleton + lifespan hooks ───────────────────────────────────

_INSTANCE: Optional[FleetOrchestrator] = None


def get_orchestrator() -> Optional[FleetOrchestrator]:
    return _INSTANCE


async def start_fleet_orchestrator() -> Optional[FleetOrchestrator]:
    """Called from main.py lifespan when role=orchestrator. Idempotent."""
    global _INSTANCE
    if _INSTANCE is not None:
        return _INSTANCE
    inst = FleetOrchestrator()
    await inst.start()
    _INSTANCE = inst
    return inst


async def stop_fleet_orchestrator() -> None:
    global _INSTANCE
    if _INSTANCE is not None:
        await _INSTANCE.stop()
        _INSTANCE = None
