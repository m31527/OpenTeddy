"""
OpenTeddy Fleet — wire protocol.

All node↔central traffic is JSON objects with a `type` discriminator.
This module is the single source of truth for the message shapes plus
the helpers both sides use to build / parse / validate them. Keeping it
transport-agnostic (pure dict in / dict out) means a future swap from
WebSocket to NATS / gRPC touches only orchestrator.py + worker.py, never
the message contracts or the orchestration logic above them.

Auth model (MVP): a shared bearer token (OPENTEDDY_FLEET_TOKEN) travels
in the `register` frame only. After a connection is authenticated, later
frames on that connection are trusted — we don't re-send the token per
message. The token check is funnelled through verify_token() so swapping
to mTLS / per-node certs later is a localized change.
"""

from __future__ import annotations

import hmac
import os
import time
from typing import Any, Dict, List, Optional


# ── Protocol version ──────────────────────────────────────────────────────────
# Bumped when a breaking change to message shapes ships. register_ack
# echoes the server's version so a worker can warn on mismatch.
PROTOCOL_VERSION = "fleet/0"


# ── Message type constants ────────────────────────────────────────────────────
# worker → orchestrator
MSG_REGISTER  = "register"
MSG_HEARTBEAT = "heartbeat"
MSG_RESULT    = "result"
MSG_ALERT     = "alert"        # milestone 2

# orchestrator → worker
MSG_REGISTER_ACK = "register_ack"
MSG_DISPATCH     = "dispatch"
MSG_CANCEL       = "cancel"


# ── Token / auth ──────────────────────────────────────────────────────────────


def fleet_token() -> str:
    """The shared bearer token both sides expect. Empty string means
    'no token configured' — verify_token() then rejects ALL connections
    on the orchestrator side (fail closed), so a misconfigured central
    never accepts unauthenticated workers."""
    return (os.getenv("OPENTEDDY_FLEET_TOKEN", "") or "").strip()


def verify_token(presented: Optional[str]) -> bool:
    """Constant-time compare of a presented token against the configured
    one. Fail-closed: if no token is configured on this host, reject.
    Centralising the check here keeps the upgrade path to mTLS / signed
    certs a one-function change."""
    expected = fleet_token()
    if not expected:
        return False  # fail closed — never accept when unconfigured
    if not presented:
        return False
    return hmac.compare_digest(str(presented), expected)


# ── Builders (worker → orchestrator) ──────────────────────────────────────────


def make_register(
    node_id: str,
    role: str,
    *,
    tools: Optional[List[str]] = None,
    model: str = "",
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """First frame a worker sends. Carries the token + the node's
    self-described capabilities so the central can route by role / tool
    availability later (milestone 4)."""
    return {
        "type":    MSG_REGISTER,
        "node_id": node_id,
        "role":    role,
        "token":   token if token is not None else fleet_token(),
        "protocol_version": PROTOCOL_VERSION,
        "capabilities": {
            "tools": tools or [],
            "model": model,
        },
    }


def make_heartbeat(
    node_id: str,
    *,
    status: str = "idle",
    ts: float,
    load: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Periodic liveness ping. `ts` is passed in (not computed here) so
    the caller controls the clock — important because Date.now-style
    calls are awkward to test and we want deterministic replay."""
    return {
        "type":    MSG_HEARTBEAT,
        "node_id": node_id,
        "status":  status,        # "idle" | "busy"
        "ts":      ts,
        "load":    load or {},
    }


def make_result(
    task_id: str,
    node_id: str,
    *,
    status: str,
    summary: str,
    artifacts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Sent when a dispatched task finishes. `status` is "completed" or
    "failed"; `summary` is the orchestrator's final text; `artifacts`
    is the same producer-tool artifact list the single-machine UI shows."""
    return {
        "type":      MSG_RESULT,
        "task_id":   task_id,
        "node_id":   node_id,
        "status":    status,       # "completed" | "failed"
        "summary":   summary,
        "artifacts": artifacts or [],
    }


def make_alert(
    node_id: str,
    *,
    severity: str,
    observation: str,
    confidence: float,
) -> Dict[str, Any]:
    """Proactive anomaly report (milestone 2). Pushed by a worker's
    watcher loop without the central having asked."""
    return {
        "type":        MSG_ALERT,
        "node_id":     node_id,
        "severity":    severity,    # "info" | "warning" | "critical"
        "observation": observation,
        "confidence":  confidence,
    }


# ── Builders (orchestrator → worker) ──────────────────────────────────────────


def make_register_ack(
    *,
    assigned: bool,
    server_version: str,
    reason: str = "",
) -> Dict[str, Any]:
    """Orchestrator's reply to register. `assigned=False` + a reason
    means the worker was rejected (bad token / unknown role); the worker
    should log the reason and back off rather than spin-retry."""
    return {
        "type":           MSG_REGISTER_ACK,
        "assigned":       assigned,
        "server_version": server_version,
        "protocol_version": PROTOCOL_VERSION,
        "reason":         reason,
    }


def make_dispatch(
    task_id: str,
    goal: str,
    *,
    mode: str = "code",
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Central hands a goal to a worker. The worker runs it on its own
    orchestrator.run() — `mode` + `context` mirror the single-machine
    TaskRequest fields so the worker can construct one verbatim."""
    return {
        "type":    MSG_DISPATCH,
        "task_id": task_id,
        "goal":    goal,
        "mode":    mode,
        "context": context or {},
    }


def make_cancel(task_id: str) -> Dict[str, Any]:
    return {"type": MSG_CANCEL, "task_id": task_id}


# ── Validation ────────────────────────────────────────────────────────────────


class ProtocolError(Exception):
    """Raised on a malformed / unexpected frame. Callers catch + log +
    drop the frame rather than crashing the connection."""


def parse_frame(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Light validation of an inbound frame: must be a dict with a known
    `type`. Returns the frame unchanged on success. We deliberately keep
    this permissive on extra fields (forward-compat) but strict on the
    discriminator."""
    if not isinstance(raw, dict):
        raise ProtocolError(f"frame is not a dict: {type(raw).__name__}")
    mtype = raw.get("type")
    known = {
        MSG_REGISTER, MSG_HEARTBEAT, MSG_RESULT, MSG_ALERT,
        MSG_REGISTER_ACK, MSG_DISPATCH, MSG_CANCEL,
    }
    if mtype not in known:
        raise ProtocolError(f"unknown frame type: {mtype!r}")
    return raw
