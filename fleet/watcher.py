"""
OpenTeddy Fleet — worker watcher loop.

A worker's "proactive" half. On an interval, the worker asks ITSELF
(via its local orchestrator) whether anything in its area of
responsibility looks anomalous, and if so pushes an `alert` frame to
the central — without the operator having asked. This is the capability
that makes a fleet node feel like an employee rather than a query
endpoint: it notices problems and raises a hand.

Design choices for the MVP:
  - The self-check PROMPT is the part that's genuinely site-specific
    ("check /data/finance for unusual payments" vs "check auth logs for
    brute-force"). We DON'T hardcode a real one — instead there's a
    generic placeholder, overridable per node via OPENTEDDY_FLEET_WATCH_PROMPT.
    This lets the pipeline (timer → self-check → judge → push) be proven
    end-to-end before the operator writes the domain-specific prompts.
  - The watcher runs the check on the SAME local orchestrator a dispatch
    would, so the node uses its own model + tools + data. No new
    execution path.
  - It's strictly opt-in: disabled unless OPENTEDDY_FLEET_WATCH_ENABLED
    is truthy. A worker with watching off behaves exactly as before.
  - The check result is parsed for an anomaly signal. To keep the MVP
    robust against free-form model output, the self-check prompt asks
    the model to end its answer with a machine-readable line:
        ANOMALY: yes|no
        SEVERITY: info|warning|critical
        CONFIDENCE: 0.0-1.0
    We parse those; anything missing defaults to "no anomaly" (fail
    quiet — a watcher that cries wolf on every parse hiccup is worse
    than one that occasionally misses).

Cadence + jitter: checks fire every OPENTEDDY_FLEET_WATCH_INTERVAL
seconds (default 900 = 15 min) with a small random jitter so a fleet of
nodes doesn't all hit their data sources on the same tick.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, Optional

from . import protocol as P

logger = logging.getLogger("openteddy.fleet.watcher")


# ── Config ────────────────────────────────────────────────────────────────────


def watch_enabled() -> bool:
    return (os.getenv("OPENTEDDY_FLEET_WATCH_ENABLED", "") or "").strip().lower() \
        in ("1", "true", "yes", "on")


def _watch_interval_s() -> float:
    try:
        return max(60.0, float(os.getenv("OPENTEDDY_FLEET_WATCH_INTERVAL", "900")))
    except (TypeError, ValueError):
        return 900.0


# Generic placeholder self-check. Operators override per node with a
# domain-specific prompt via OPENTEDDY_FLEET_WATCH_PROMPT. The trailing
# machine-readable block is appended automatically (see _build_prompt) so
# operators only write the "what to check" part.
_DEFAULT_WATCH_PROMPT = (
    "You are a monitoring agent for this node. Using the tools and data "
    "available to you, briefly check whether anything in your area of "
    "responsibility looks unusual or warrants a human's attention right "
    "now. If you have no specific data source configured, say so and "
    "report no anomaly."
)

_MACHINE_TAIL = (
    "\n\nAfter your assessment, end your reply with EXACTLY these three "
    "lines, nothing after them:\n"
    "ANOMALY: <yes|no>\n"
    "SEVERITY: <info|warning|critical>\n"
    "CONFIDENCE: <0.0-1.0>"
)


def _has_custom_watch_prompt() -> bool:
    """True only when the operator has defined what this node should
    actually check. The generic placeholder doesn't count — a reasoning
    model handed a vague 'is anything unusual?' with no real data source
    invents anomalies and spams warning/critical alerts every cycle."""
    return bool((os.getenv("OPENTEDDY_FLEET_WATCH_PROMPT", "") or "").strip())


def _alert_cooldown_s() -> float:
    """Minimum seconds before the SAME observation may alert again. Stops
    a genuine-but-persistent condition (e.g. disk 85%) from re-firing
    every cycle. Default 1h."""
    try:
        return max(0.0, float(os.getenv("OPENTEDDY_FLEET_ALERT_COOLDOWN", "3600")))
    except (TypeError, ValueError):
        return 3600.0


def _build_prompt() -> str:
    base = (os.getenv("OPENTEDDY_FLEET_WATCH_PROMPT", "") or "").strip() \
        or _DEFAULT_WATCH_PROMPT
    return base + _MACHINE_TAIL


# ── Result parsing ────────────────────────────────────────────────────────────


_ANOMALY_RE   = re.compile(r"ANOMALY:\s*(yes|no)", re.IGNORECASE)
_SEVERITY_RE  = re.compile(r"SEVERITY:\s*(info|warning|critical)", re.IGNORECASE)
_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)")


def parse_watch_result(text: str) -> Optional[Dict[str, Any]]:
    """Parse the machine-readable tail of a self-check answer. Returns a
    dict {severity, confidence, observation} when an anomaly is flagged,
    or None when no anomaly / unparseable (fail quiet)."""
    if not text:
        return None
    m_anom = _ANOMALY_RE.search(text)
    if not m_anom or m_anom.group(1).lower() != "yes":
        return None
    sev = "warning"
    m_sev = _SEVERITY_RE.search(text)
    if m_sev:
        sev = m_sev.group(1).lower()
    conf = 0.5
    m_conf = _CONFIDENCE_RE.search(text)
    if m_conf:
        try:
            conf = max(0.0, min(1.0, float(m_conf.group(1))))
        except ValueError:
            pass
    # The human-readable observation is everything BEFORE the machine
    # tail — strip the three marker lines so the alert reads cleanly.
    observation = re.split(r"\n?ANOMALY:", text, maxsplit=1)[0].strip()
    if len(observation) > 1500:
        observation = observation[:1500] + " …"
    return {"severity": sev, "confidence": conf, "observation": observation}


# ── Watcher loop ──────────────────────────────────────────────────────────────


class FleetWatcher:
    """Runs inside a worker. Periodically self-checks and pushes alerts
    to the central over the SAME websocket the worker is connected on.
    The worker hands us a `send` coroutine + its node id; we never own
    the socket lifecycle."""

    def __init__(self, node_id: str, send_frame, get_orchestrator) -> None:
        self._node_id = node_id
        self._send = send_frame            # async (dict) -> None
        self._get_orch = get_orchestrator  # () -> Orchestrator
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        # {alert_signature: last_sent_monotonic} — drives the cooldown so
        # the same observation doesn't re-alert every cycle.
        self._alert_history: Dict[str, float] = {}

    def start(self) -> None:
        if self._task is None and watch_enabled():
            self._task = asyncio.create_task(self._loop(), name="fleet_watcher")
            logger.info("fleet watcher: enabled, interval=%.0fs",
                        _watch_interval_s())
            if not _has_custom_watch_prompt():
                # Loud, actionable warning: enabled but no real job → it
                # will run no-ops and never alert (by design, to avoid the
                # placeholder-prompt false-positive storm).
                logger.warning(
                    "fleet watcher: ENABLED but OPENTEDDY_FLEET_WATCH_PROMPT "
                    "is not set — it will NOT raise alerts. Define what this "
                    "node should check (e.g. disk/log/service health) in "
                    "OPENTEDDY_FLEET_WATCH_PROMPT to activate alerting."
                )

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        import random
        # Initial stagger so a freshly-booted fleet doesn't all check at
        # once. Vary by node id hash so it's stable per node, not random
        # per process restart.
        await asyncio.sleep(5 + (hash(self._node_id) % 30))
        while not self._stop.is_set():
            try:
                await self._run_one_check()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.warning("fleet watcher: check raised (ignored): %s", exc)
            # Interval + up to ±15% jitter.
            interval = _watch_interval_s()
            jitter = interval * 0.15 * (random.random() * 2 - 1)
            await asyncio.sleep(max(60.0, interval + jitter))

    async def _run_one_check(self) -> None:
        """Run one self-check on the local orchestrator; push an alert
        if the model flags an anomaly."""
        from models import TaskRequest, SessionMode
        orch = self._get_orch()
        if orch is None:
            return
        # No real job configured → skip the check entirely. The generic
        # placeholder prompt + a reasoning model invents anomalies and
        # spams warning/critical every cycle; skipping also saves the
        # wasted LLM call. Operators activate alerting by defining
        # OPENTEDDY_FLEET_WATCH_PROMPT (what THIS node should check).
        if not _has_custom_watch_prompt():
            logger.debug(
                "fleet watcher: no OPENTEDDY_FLEET_WATCH_PROMPT set — "
                "skipping check (no alerts until a real check is defined)."
            )
            return
        req = TaskRequest(
            id=str(uuid.uuid4()),
            goal=_build_prompt(),
            context={"triggered_by": "fleet_watcher"},
            # TaskRequest enforces priority >= 1; watcher checks are
            # low-priority but can't go below 1.
            priority=1,
            session_id=f"watch-{self._node_id}",
            mode=SessionMode.CODE,
        )
        # Bind the origin ContextVar the tool registry actually reads when
        # deciding the approval policy. WITHOUT this, the watcher's
        # health-check shell commands (uptime / df / docker ps …) hit the
        # web-UI approval gate and pile up forever — nobody is watching the
        # UI for an autonomous monitor. "fleet_watcher" auto-approves
        # non-destructive tools (destructive ones stay hard-blocked by the
        # denylist), exactly like the Telegram channel. The dict field in
        # `context` above is NOT read by the registry — this is.
        from tools._context import set_triggered_by, reset_triggered_by
        _origin_tok = set_triggered_by("fleet_watcher")
        try:
            result = await orch.run(req)
        finally:
            reset_triggered_by(_origin_tok)
        summary = getattr(result, "summary", "") or ""
        parsed = parse_watch_result(summary)
        if parsed is None:
            logger.debug("fleet watcher: no anomaly this cycle")
            return

        # ── De-dup / cooldown ────────────────────────────────────────────
        # A genuine but persistent condition (disk 85%, a service down)
        # would otherwise re-alert EVERY cycle. Suppress the same
        # observation until the cooldown elapses, so the operator gets one
        # ping, not a stream. Signature = severity + a hash of the
        # normalised observation (case/whitespace-folded, capped) so near-
        # identical wordings collapse to one.
        import hashlib
        norm = " ".join((parsed["observation"] or "").lower().split())[:200]
        sig = f'{parsed["severity"]}:{hashlib.sha1(norm.encode()).hexdigest()[:12]}'
        now = asyncio.get_event_loop().time()
        cooldown = _alert_cooldown_s()
        last = self._alert_history.get(sig)
        if last is not None and (now - last) < cooldown:
            logger.info(
                "fleet watcher: duplicate %s alert within cooldown (%.0fs) — "
                "suppressed", parsed["severity"], cooldown,
            )
            return
        self._alert_history[sig] = now
        # Bound memory — drop entries well past their cooldown.
        self._alert_history = {
            k: v for k, v in self._alert_history.items() if now - v < cooldown * 2
        }

        alert = P.make_alert(
            self._node_id,
            severity=parsed["severity"],
            observation=parsed["observation"],
            confidence=parsed["confidence"],
        )
        try:
            await self._send(alert)
            logger.info("fleet watcher: pushed %s alert (conf=%.2f)",
                        parsed["severity"], parsed["confidence"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("fleet watcher: failed to push alert: %s", exc)
