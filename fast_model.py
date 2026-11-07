"""
One short call to the fast chat model, for small background judgments:
naming a session, picking which skills a task needs.

Same routing as the gate: the local model (config.gemma_model), or the
cloud provider when the orchestrator runs there — never for a session
marked local-only.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from config import config


async def ask(orch: Any, system: str, prompt: str, *, max_tokens: int = 60,
              label: str = "[fast]", local_only: Optional[bool] = None,
              json_mode: bool = False, task_id: str = "") -> str:
    """The model's reply text. Raises on a failed call."""
    from config import orchestrator_on_cloud, set_session_local_only
    if local_only is not None:
        set_session_local_only(local_only)   # the caller's own context only
    if orchestrator_on_cloud():
        r = await orch.escalation.provider.complete_text(prompt, system=system, max_tokens=max_tokens)
        return r.text or ""

    import local_engine
    payload: Dict[str, Any] = {
        "model": config.gemma_model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": max_tokens,
                    "num_ctx": int(getattr(config, "gemma_num_ctx", 16384))},
        "keep_alive": getattr(config, "ollama_keep_alive", "24h"),
    }
    if json_mode:
        payload["format"] = "json"
    th = local_engine.think_setting(config.gemma_model)
    if th is not None:
        payload["think"] = th
    t0 = time.monotonic()
    resp = await orch._http.post(f"{config.gemma_base_url}/api/chat", json=payload)
    resp.raise_for_status()
    data = resp.json()
    try:
        t = local_engine.timings(data)
        out = data.get("eval_count", 0) or 0
        dur = data.get("eval_duration", 0) or 0
        await orch.tracker.record_usage(
            task_id=task_id, model=config.gemma_model, model_provider="ollama",
            tokens_in=data.get("prompt_eval_count", 0) or 0, tokens_out=out,
            task_description=label,
            duration_ms=int((data.get("total_duration") or 0) / 1e6) or int((time.monotonic() - t0) * 1000),
            tokens_per_sec=(out * 1e9 / dur) if out and dur else 0.0,
            load_ms=t["load_ms"], prefill_ms=t["prefill_ms"])
    except Exception:  # noqa: BLE001
        pass
    return (data.get("message") or {}).get("content") or ""
