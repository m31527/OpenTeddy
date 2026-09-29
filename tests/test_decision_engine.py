"""
Decision engine — the ladder (rule → Laya → fallback) and the three modes,
with a fake provider so no torch/weights are needed.

    .venv/bin/python tests/test_decision_engine.py
"""
from __future__ import annotations

import asyncio, logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

from config import config
import decide as D


class FakeLaya:
    """Answers scripted per call; records every predict."""
    def __init__(self, ready=True, installed=True):
        self._ready, self._installed = ready, installed
        self.calls: list = []
        self.next: list = []          # queue of answer dicts
    def installed(self): return self._installed
    def ready(self): return self._ready
    def status(self): return "ready" if self._ready else ("not loaded" if self._installed else "not installed")
    async def load(self): self._ready = True; return True
    async def predict(self, state, questions):
        self.calls.append((state, questions))
        q = next(iter(questions.values()))
        ans = self.next.pop(0) if self.next else {"noul": 0.5}
        if q["type"] == "noul":
            p = ans["noul"]
            return {"answers": {"q": {"type": "noul", "noul": p, "confidence": max(p, 1 - p)}},
                    "routing": {"model": "fake"}, "usage": {"input_tokens": 12}}
        probs = ans["probs"]
        best = max(probs, key=probs.get)
        return {"answers": {"q": {"type": "choice", "choice": best, "probabilities": probs}},
                "routing": {"model": "fake"}, "usage": {"input_tokens": 12}}


LOG: list = []
async def sink(rec): LOG.append(rec)


def reset(mode="shadow", active=(), off=(), min_conf=0.85, temps=None):
    config.decision_mode = mode
    config.decision_active_kinds = list(active)
    config.decision_off_kinds = list(off)
    config.decision_min_confidence = min_conf
    config.decision_temperatures = temps or {}
    LOG.clear()


async def main() -> None:
    fake = FakeLaya(); D.set_provider(fake); D.set_log_sink(sink)
    calls = {"fb": 0}
    async def fb_yes():
        calls["fb"] += 1; return True, "llm says yes"
    async def fb_none():
        calls["fb"] += 1; return None, "llm unsure"

    # 1) shadow: Laya runs + is logged, fallback decides; agree computed
    reset("shadow"); calls["fb"] = 0; fake.next = [{"noul": 0.1}]
    d = await D.yes_no("k", "state", "q?", fallback=fb_yes)
    assert d.provider == "fallback" and d.answer is True and calls["fb"] == 1
    assert d.probe and d.probe.answer is False and d.agree is False and d.probe.latency_ms >= 0
    assert LOG[-1]["provider"] == "fallback" and LOG[-1]["laya_answer"] == "false" and LOG[-1]["agree"] == 0
    print("  ✓ shadow: fallback decides, Laya logged, disagreement recorded")

    # 2) shadow + rule: rule wins, fallback NOT called, Laya still logged
    calls["fb"] = 0; fake.next = [{"noul": 0.95}]
    d = await D.yes_no("k", "state", "q?", rule=(True, "task said yes"), fallback=fb_yes)
    assert d.provider == "rule" and d.answer is True and calls["fb"] == 0 and d.agree is True
    print("  ✓ shadow + rule: rule wins without calling the fallback; Laya compared")

    # 3) active + confident: Laya decides, fallback skipped
    reset("active"); calls["fb"] = 0; fake.next = [{"noul": 0.97}]
    d = await D.yes_no("k", "state", "q?", fallback=fb_yes)
    assert d.provider == "laya" and d.answer is True and calls["fb"] == 0 and abs(d.confidence - 0.97) < 1e-9
    print("  ✓ active + confident: Laya decides, no fallback call")

    # 4) active + not confident: fallback decides, Laya logged
    calls["fb"] = 0; fake.next = [{"noul": 0.6}]
    d = await D.yes_no("k", "state", "q?", fallback=fb_yes)
    assert d.provider == "fallback" and calls["fb"] == 1 and d.probe.confidence < 0.85
    print("  ✓ active + unsure: fallback decides")

    # 5) active per-kind list on a shadow base; off list wins
    reset("shadow", active=["k"], off=["z"]); fake.next = [{"noul": 0.99}]
    d = await D.yes_no("k", "s", "q", fallback=fb_yes); assert d.provider == "laya"
    calls["fb"] = 0; d = await D.yes_no("z", "s", "q", fallback=fb_yes)
    assert d.provider == "fallback" and d.probe is None and calls["fb"] == 1 and not any(r["kind"] == "z" for r in LOG)
    print("  ✓ per-kind active/off lists; off kinds never touch Laya and are not logged")

    # 6) off: no Laya, no log
    reset("off"); calls["fb"] = 0; fake.calls.clear()
    d = await D.yes_no("k", "s", "q", fallback=fb_yes)
    assert d.provider == "fallback" and d.probe is None and not fake.calls and not LOG
    print("  ✓ off: behaves as before the engine existed")

    # 7) provider not ready → fallback, error logged, nothing blocks
    reset("active"); slow = FakeLaya(ready=False); D.set_provider(slow); calls["fb"] = 0
    d = await D.yes_no("k", "s", "q", fallback=fb_yes)
    assert d.provider == "fallback" and d.probe.error == "not loaded" and calls["fb"] == 1
    assert LOG[-1]["laya_error"] == "not loaded"
    await asyncio.sleep(0)   # let the background load task run
    D.set_provider(fake)
    print("  ✓ not loaded: fallback immediately, error recorded, load kicked off in background")

    # 8) fallback couldn't tell → answer None; in active mode Laya may still decide
    reset("shadow"); fake.next = [{"noul": 0.2}]
    d = await D.yes_no("k", "s", "q", fallback=fb_none); assert d.answer is None and d.agree is None
    reset("active"); fake.next = [{"noul": 0.02}]
    d = await D.yes_no("k", "s", "q", fallback=fb_none); assert d.provider == "laya" and d.answer is False
    print("  ✓ undecided fallback → None; active Laya can still answer")

    # 9) choice + temperature: calibration flattens over-confidence
    reset("active", temps={"c": 2.0}); fake.next = [{"probs": {"a": 0.9, "b": 0.08, "c": 0.02}}]
    d = await D.choice("c", "s", "which?", {"a": "", "b": "", "c": ""}, fallback=None)
    assert d.probe.answer == "a" and d.probe.raw_confidence == 0.9 and d.probe.confidence < 0.9
    assert abs(sum(d.probe.probabilities.values()) - 1) < 1e-6
    assert d.provider == "fallback" and d.answer is None   # cooled below 0.85 → not confident
    assert abs(D.scale_prob(0.9, 1.0) - 0.9) < 1e-9 and D.scale_prob(0.9, 2.0) < 0.9 and D.scale_prob(0.9, 0.5) > 0.9
    print("  ✓ temperature scaling: T>1 cools, T<1 sharpens, probabilities renormalise")

    # 10) log_probe for end-of-path call sites
    reset("shadow"); fake.next = [{"probs": {"template": 0.7, "answer": 0.2, "work": 0.1}}]
    p = await D.probe_choice("voice.route", "早安", "lane?", {"template": "", "answer": "", "work": ""})
    d = await D.log_probe(p, "template", provider="rule")
    assert d.agree is True and LOG[-1]["kind"] == "voice.route" and LOG[-1]["answer"] == '"template"'
    print("  ✓ log_probe records a late verdict next to the probe")

    # 11) state is truncated to the model's window; dict state serialised
    reset("shadow"); fake.calls.clear(); fake.next = [{"noul": 0.5}]
    await D.yes_no("k", {"x": "y" * 10000}, "q", fallback=fb_yes)
    assert len(fake.calls[-1][0]) <= D._MAX_STATE_CHARS
    print("  ✓ state capped at the context window")
    print("\nALL DECISION ENGINE TESTS PASS")


asyncio.run(main())
