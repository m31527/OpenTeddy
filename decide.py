"""
OpenTeddy Decision Engine v0.1 — how much intelligence is this decision worth?

Several places in the runtime make a small, typed decision — is this file a
real deliverable? should this result notify the owner? is this message a
schedule request? which lane should a spoken question take? — and each
one currently costs a full LLM call (seconds on a 35B model) or a regex.
A non-autoregressive decision model (Laya, ~421M params, Apache-2.0)
answers the same typed question in ~30–60 ms with a probability attached
and nothing to parse. This module routes every such decision through one
ladder and records what happened:

    rule      deterministic verdict supplied by the caller (task's own
              ALERT line, hard policy) — always wins
    laya      when installed and the kind is ACTIVE and calibrated
              confidence ≥ the threshold
    fallback  the caller's existing path (LLM call / regex)

Per-kind modes:
    off      Laya never runs
    shadow   Laya runs and is LOGGED next to the real verdict; the
             fallback still decides  — the default, and where data comes from
    active   Laya decides when confident; otherwise fallback (still logged)

Why shadow first: Laya zero-shot is near chance on decisions it was not
trained for (its own benchmarks say so, and a first run here judged an
honest "0 orders" report "not a real artifact" at 0.91). Shadow mode turns
weeks of real traffic into an agreement rate per kind and a labelled set
to fine-tune on — only then does a kind get promoted to active.

Confidence is temperature-scaled per kind (raw Laya confidence is
over-confident; ECE 0.25 raw vs 0.08 fitted in its own numbers). Until a
temperature has been fitted from logged data, thresholds mean little —
another reason the default is shadow.

The decision never blocks on model loading: if Laya is still loading
(first run downloads ~1.5 GB), the call proceeds on the fallback and logs
that Laya was unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from config import config

logger = logging.getLogger(__name__)

# Decision kinds wired in v0.1 (documentation; any string is accepted).
KIND_JUDGE = "judge.deliverable"      # noul — is the produced file a real deliverable?
KIND_NOTIFY = "notify.gate"           # noul — does this scheduled result need the owner?
KIND_SCHEDULE = "schedule.intent"     # noul — is the message asking to schedule, not to do?
KIND_VOICE = "voice.route"            # choice — template / answer / work

_MAX_STATE_CHARS = 3000               # Laya context is 512–1024 tokens; keep the head
_LAYA_CALL_TIMEOUT_S = 5.0            # a warm call is ~50 ms; anything slower is a stall


# ── Laya provider ─────────────────────────────────────────────────────────────

class _LayaProvider:
    """Lazy, single-instance Laya router.

    Every Laya call — loading included — runs on ONE dedicated thread.
    `asyncio.to_thread` hands consecutive calls to whichever pool thread
    is free, and PyTorch's Metal backend (MPS) is not safe to drive from
    changing threads: the first version crashed a process outright with
    "failed assertion _status < MTLCommandBufferStatusCommitted" and made
    in-server calls take seconds instead of milliseconds. A single worker
    is also the right shape for CUDA (one stream, no contention) and it
    serialises forward passes on the shared model for free."""

    def __init__(self) -> None:
        self._router: Any = None
        self._loading = False
        self._load_error: Optional[str] = None
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya")

    async def _run(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: fn(*args, **kwargs))

    @staticmethod
    def installed() -> bool:
        try:
            import laya  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    def ready(self) -> bool:
        return self._router is not None

    def status(self) -> str:
        if self._router is not None:
            return "ready"
        if self._load_error:
            return f"error: {self._load_error}"
        if self._loading:
            return "loading"
        return "not loaded" if self.installed() else "not installed"

    def _load_sync(self) -> None:
        """Build the router AND warm it, on the worker thread, before
        anything is marked ready. On MPS the first forward pass of each
        checkpoint compiles Metal kernels (~4 s); if `ready()` flipped
        before that, real probes queued behind the warm-up and every one
        of them hit the 5 s timeout — the first in-server run lost all
        five voice probes exactly this way. Both checkpoints and both
        question types are exercised so no real call pays first-use."""
        from laya import Router
        router = Router(device=getattr(config, "decision_device", None) or None)
        noul = {"q": {"type": "noul", "instructions": "ok?"}}
        choice = {"q": {"type": "choice", "instructions": "which?",
                        "criteria": {"a": "first", "b": "second"}}}
        for state in ("warm-up: nothing to see here", "暖機：沒有需要注意的事"):
            for qs in (noul, choice):
                try:
                    router.predict(state, qs)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Laya warm-up call failed: %s", exc)
        self._router = router

    async def load(self) -> bool:
        """Load the router (downloads checkpoints on first use). Returns
        True when usable. Never raises."""
        if self._router is not None:
            return True
        if self._loading:
            return False
        self._loading = True
        try:
            await self._run(self._load_sync)
            self._load_error = None
            return True
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Laya unavailable: %s", self._load_error)
            return False
        finally:
            self._loading = False

    async def predict(self, state: str, questions: Dict[str, Any]) -> Dict[str, Any]:
        if self._router is None:
            raise RuntimeError(f"laya {self.status()}")
        # The single-worker executor already serialises calls; wait_for
        # bounds a stalled backend without leaving the worker wedged
        # (the future keeps running, the caller just stops waiting).
        return await asyncio.wait_for(
            self._run(self._router.predict, state, questions),
            timeout=_LAYA_CALL_TIMEOUT_S,
        )


_provider: Any = _LayaProvider()
_log_sink: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None


def set_provider(provider: Any) -> None:
    """Test hook: anything with .ready()/.status()/.predict()/.installed()."""
    global _provider
    _provider = provider


def set_log_sink(sink: Optional[Callable[[Dict[str, Any]], Awaitable[None]]]) -> None:
    """Test hook / alternative store for decision records."""
    global _log_sink
    _log_sink = sink


async def preload() -> bool:
    """Warm Laya at startup so the first real decision doesn't wait on a
    download. Loads both the English and multilingual checkpoints."""
    if mode_for("*") == "off" or not _provider.installed():
        return False
    t0 = time.monotonic()
    ok = await _provider.load()          # load + warm-up, on the worker thread
    if ok:
        logger.info("Laya ready (loaded + warmed) in %.1fs", time.monotonic() - t0)
    return ok


def provider_status() -> Dict[str, Any]:
    return {
        "installed": _provider.installed(),
        "status": _provider.status(),
        "mode": getattr(config, "decision_mode", "shadow"),
        "active_kinds": list(getattr(config, "decision_active_kinds", []) or []),
        "off_kinds": list(getattr(config, "decision_off_kinds", []) or []),
        "min_confidence": float(getattr(config, "decision_min_confidence", 0.85) or 0.85),
        "temperatures": dict(getattr(config, "decision_temperatures", {}) or {}),
    }


# ── Modes / calibration ──────────────────────────────────────────────────────

def mode_for(kind: str) -> str:
    base = str(getattr(config, "decision_mode", "shadow") or "shadow").lower()
    if base not in ("off", "shadow", "active"):
        base = "shadow"
    if kind in (getattr(config, "decision_off_kinds", []) or []):
        return "off"
    if kind in (getattr(config, "decision_active_kinds", []) or []):
        return "active" if base != "off" else "off"
    return base


def temperature_for(kind: str) -> float:
    temps = getattr(config, "decision_temperatures", {}) or {}
    try:
        return max(0.05, float(temps.get(kind, 1.0)))
    except Exception:  # noqa: BLE001
        return 1.0


def scale_prob(p: float, temperature: float) -> float:
    """Temperature-scale a binary probability via its logit. T=1 is the
    identity (returned as-is, not round-tripped through exp/log)."""
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    if abs(temperature - 1.0) < 1e-9:
        return p
    z = math.log(p / (1 - p)) / temperature
    return 1.0 / (1.0 + math.exp(-z))


def scale_probs(probs: Dict[str, float], temperature: float) -> Dict[str, float]:
    logits = {k: math.log(max(float(v), 1e-9)) / temperature for k, v in probs.items()}
    m = max(logits.values())
    exps = {k: math.exp(v - m) for k, v in logits.items()}
    z = sum(exps.values()) or 1.0
    return {k: v / z for k, v in exps.items()}


# ── Records ─────────────────────────────────────────────────────────────────

@dataclass
class Probe:
    """What Laya said, before anyone decided anything."""
    kind: str
    dtype: str                      # noul | choice
    answer: Any = None              # bool for noul, label for choice
    confidence: float = 0.0         # calibrated, in [0,1]
    raw_confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    latency_ms: int = 0
    model: str = ""                 # routing.model (english / multilingual)
    input_tokens: int = 0
    error: str = ""                 # non-empty when Laya did not answer
    mode: str = "shadow"


@dataclass
class Decision:
    id: str
    kind: str
    dtype: str
    answer: Any
    confidence: float
    provider: str                   # rule | laya | fallback
    reason: str = ""
    mode: str = "shadow"
    latency_ms: int = 0
    probe: Optional[Probe] = None
    fallback_answer: Any = None
    fallback_latency_ms: Optional[int] = None
    task_id: str = ""
    agree: Optional[bool] = None    # laya answer == final answer, when both exist

    @property
    def laya_would_decide(self) -> bool:
        return bool(self.probe and not self.probe.error
                    and self.probe.confidence >= _min_confidence())


def _min_confidence() -> float:
    return float(getattr(config, "decision_min_confidence", 0.85) or 0.85)


def _state_text(state: Any) -> str:
    s = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    return s[:_MAX_STATE_CHARS]


# ── Probing Laya ─────────────────────────────────────────────────────────────

async def probe_yes_no(kind: str, state: Any, instructions: str) -> Probe:
    p = Probe(kind=kind, dtype="noul", mode=mode_for(kind))
    if p.mode == "off":
        p.error = "off"
        return p
    if not _provider.ready():
        p.error = _provider.status()
        # Kick off loading in the background so a later call can use it.
        if _provider.installed() and _provider.status() == "not loaded":
            asyncio.create_task(_provider.load())
        return p
    t0 = time.monotonic()
    try:
        res = await _provider.predict(_state_text(state), {"q": {"type": "noul", "instructions": instructions}})
        a = (res.get("answers") or {}).get("q") or {}
        raw_p = float(a.get("noul", 0.5))
        T = temperature_for(kind)
        cal_p = scale_prob(raw_p, T)
        p.answer = cal_p >= 0.5
        p.confidence = cal_p if p.answer else 1.0 - cal_p
        p.raw_confidence = raw_p if raw_p >= 0.5 else 1.0 - raw_p
        p.probabilities = {"true": cal_p, "false": 1.0 - cal_p}
        p.model = str((res.get("routing") or {}).get("model") or res.get("model") or "")
        p.input_tokens = int((res.get("usage") or {}).get("input_tokens") or 0)
    except Exception as exc:  # noqa: BLE001
        p.error = f"{type(exc).__name__}: {exc}"
    p.latency_ms = int((time.monotonic() - t0) * 1000)
    return p


async def probe_choice(kind: str, state: Any, instructions: str, criteria: Dict[str, str]) -> Probe:
    p = Probe(kind=kind, dtype="choice", mode=mode_for(kind))
    if p.mode == "off":
        p.error = "off"
        return p
    if not _provider.ready():
        p.error = _provider.status()
        if _provider.installed() and _provider.status() == "not loaded":
            asyncio.create_task(_provider.load())
        return p
    t0 = time.monotonic()
    try:
        res = await _provider.predict(_state_text(state), {"q": {"type": "choice", "instructions": instructions, "criteria": criteria}})
        a = (res.get("answers") or {}).get("q") or {}
        raw = {k: float(v) for k, v in (a.get("probabilities") or {}).items()}
        if not raw:
            raise RuntimeError("no probabilities in Laya answer")
        cal = scale_probs(raw, temperature_for(kind))
        best = max(cal, key=cal.get)
        p.answer = best
        p.confidence = cal[best]
        p.raw_confidence = max(raw.values())
        p.probabilities = cal
        p.model = str((res.get("routing") or {}).get("model") or res.get("model") or "")
        p.input_tokens = int((res.get("usage") or {}).get("input_tokens") or 0)
    except Exception as exc:  # noqa: BLE001
        p.error = f"{type(exc).__name__}: {exc}"
    p.latency_ms = int((time.monotonic() - t0) * 1000)
    return p


# ── Deciding ────────────────────────────────────────────────────────────────

Fallback = Callable[[], Awaitable[Tuple[Any, str]]]


async def _decide(kind: str, dtype: str, probe_coro: Awaitable[Probe],
                  fallback: Optional[Fallback], rule: Optional[Tuple[Any, str]],
                  task_id: str) -> Decision:
    t0 = time.monotonic()
    mode = mode_for(kind)
    d = Decision(id=uuid.uuid4().hex[:12], kind=kind, dtype=dtype, answer=None,
                 confidence=0.0, provider="fallback", mode=mode, task_id=task_id)

    if mode == "off":
        # Laya never runs; behave exactly as before this module existed.
        if rule is not None:
            d.answer, d.reason, d.provider, d.confidence = rule[0], rule[1], "rule", 1.0
        elif fallback is not None:
            ft = time.monotonic()
            d.answer, d.reason = await fallback()
            d.fallback_latency_ms = int((time.monotonic() - ft) * 1000)
            d.fallback_answer = d.answer
        d.latency_ms = int((time.monotonic() - t0) * 1000)
        await _log(d)
        return d

    if mode == "active" and rule is None:
        # Laya first; fallback only when it can't decide confidently.
        probe = await probe_coro
        d.probe = probe
        if not probe.error and probe.confidence >= _min_confidence():
            d.answer, d.confidence, d.provider = probe.answer, probe.confidence, "laya"
            d.reason = f"laya {probe.model} p={probe.confidence:.2f}"
        elif fallback is not None:
            ft = time.monotonic()
            d.answer, d.reason = await fallback()
            d.fallback_latency_ms = int((time.monotonic() - ft) * 1000)
            d.fallback_answer = d.answer
    else:
        # shadow (or active with a rule): the verdict comes from the rule /
        # fallback; Laya runs alongside so its answer can be compared.
        # Run concurrently so shadow adds no latency to the real path.
        async def _fb() -> Tuple[Any, str, int]:
            if rule is not None:
                return rule[0], rule[1], 0
            if fallback is None:
                return None, "", 0
            ft = time.monotonic()
            a, r = await fallback()
            return a, r, int((time.monotonic() - ft) * 1000)
        probe, (ans, reason, fb_ms) = await asyncio.gather(probe_coro, _fb())
        d.probe = probe
        d.answer, d.reason = ans, reason
        if rule is not None:
            d.provider, d.confidence = "rule", 1.0
        else:
            d.provider = "fallback"
            d.fallback_answer, d.fallback_latency_ms = ans, fb_ms

    if d.probe and not d.probe.error and d.answer is not None:
        d.agree = (d.probe.answer == d.answer)
    d.latency_ms = int((time.monotonic() - t0) * 1000)
    await _log(d)
    return d


async def yes_no(kind: str, state: Any, instructions: str, *,
                 fallback: Optional[Fallback] = None,
                 rule: Optional[Tuple[bool, str]] = None,
                 task_id: str = "") -> Decision:
    """Typed yes/no decision through the ladder. `fallback` returns
    (answer, reason) — answer may be None when the fallback couldn't tell."""
    return await _decide(kind, "noul", probe_yes_no(kind, state, instructions),
                         fallback, rule, task_id)


async def choice(kind: str, state: Any, instructions: str, criteria: Dict[str, str], *,
                 fallback: Optional[Fallback] = None,
                 rule: Optional[Tuple[str, str]] = None,
                 task_id: str = "") -> Decision:
    return await _decide(kind, "choice", probe_choice(kind, state, instructions, criteria),
                         fallback, rule, task_id)


async def log_probe(probe: Probe, final_answer: Any, provider: str = "fallback",
                    reason: str = "", task_id: str = "", latency_ms: int = 0) -> Decision:
    """For call sites whose verdict is only known at the end (voice
    routing): record a probe next to the path actually taken."""
    d = Decision(id=uuid.uuid4().hex[:12], kind=probe.kind, dtype=probe.dtype,
                 answer=final_answer, confidence=0.0, provider=provider, reason=reason,
                 mode=probe.mode, latency_ms=latency_ms, probe=probe, task_id=task_id)
    if not probe.error and final_answer is not None:
        d.agree = (probe.answer == final_answer)
    await _log(d)
    return d


# ── Logging ─────────────────────────────────────────────────────────────────

def _record(d: Decision) -> Dict[str, Any]:
    p = d.probe
    return {
        "id": d.id, "kind": d.kind, "dtype": d.dtype, "mode": d.mode,
        "answer": json.dumps(d.answer, ensure_ascii=False), "confidence": round(d.confidence, 4),
        "provider": d.provider, "reason": (d.reason or "")[:300], "latency_ms": d.latency_ms,
        "laya_answer": (json.dumps(p.answer, ensure_ascii=False) if p and not p.error else None),
        "laya_confidence": (round(p.confidence, 4) if p and not p.error else None),
        "laya_raw_confidence": (round(p.raw_confidence, 4) if p and not p.error else None),
        "laya_latency_ms": (p.latency_ms if p else None),
        "laya_model": (p.model if p else None),
        "laya_error": (p.error if p and p.error else None),
        "fallback_answer": (json.dumps(d.fallback_answer, ensure_ascii=False)
                            if d.fallback_answer is not None else None),
        "fallback_latency_ms": d.fallback_latency_ms,
        "agree": (None if d.agree is None else int(d.agree)),
        "task_id": d.task_id or "",
    }


async def _log(d: Decision) -> None:
    if d.mode == "off":
        return              # off means off — no engine, no log rows
    rec = _record(d)
    try:
        if _log_sink is not None:
            await _log_sink(rec)
            return
        import main as _main_module
        await _main_module.tracker.log_decision(rec)
    except Exception as exc:  # noqa: BLE001
        logger.debug("decision log failed: %s", exc)
