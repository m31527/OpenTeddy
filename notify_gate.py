"""
OpenTeddy notify gate — "only call me when it matters".

A schedule can carry a `notify_when` condition ("較前 7 日均值偏差超過
15%", "有異常或失敗才通知"). After the scheduled task finishes, this module
decides whether the result deserves a notification or should be recorded
quietly. That decision is what turns a scheduler from "runs on time" into
"an agent that only interrupts you for a reason".

Decision order — cheapest first, and fail-open:

  1. task failed                → notify (a broken job is always news)
  2. no condition on the row    → notify (the historical behaviour)
  3. the task's own verdict     → the scheduled goal asks the model to end
                                  its report with `ALERT: yes|no — reason`;
                                  if that line exists, trust it (0 cost)
  4. a model judgement          → executor model reads goal + condition +
                                  result and answers JSON
  5. anything goes wrong        → notify, with the reason attached

Fail-open on purpose: an extra ping costs a glance; a swallowed alert can
cost a quarter.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

import httpx

import local_engine
from config import config

logger = logging.getLogger(__name__)

_VERDICT = re.compile(
    r"ALERT\s*[:：]\s*(yes|no|是|否|true|false)\s*(?:[—\-–:：]\s*(.+))?",
    re.IGNORECASE,
)

_SYSTEM = """你是排程結果的通知守門員。負責人只想在「值得打擾」時被通知。
你會拿到：任務目標、通知條件、任務結果（報告全文）。
判斷結果是否符合通知條件。只輸出 JSON，不要其他文字：
{"notify": true 或 false, "reason": "一句話說明（引用結果裡的數字）", "severity": "info" 或 "warning" 或 "high"}
規則：
- 條件成立、或結果顯示異常／失敗／資料缺失 → notify true
- 條件不成立、一切正常 → notify false，reason 簡述「正常，數字為…」
- 無法從結果判斷 → notify true，reason 說明無法判斷的原因"""


def verdict_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Parse the task's own `ALERT: yes|no — reason` line, last one wins."""
    hits = list(_VERDICT.finditer(text or ""))
    if not hits:
        return None
    m = hits[-1]
    flag = m.group(1).lower()
    notify = flag in ("yes", "是", "true")
    reason = (m.group(2) or "").strip() or ("條件成立" if notify else "條件未成立")
    return {"notify": notify, "reason": reason[:300],
            "severity": "warning" if notify else "info", "source": "task"}


async def evaluate(
    goal: str,
    condition: str,
    result_text: str,
    *,
    failed: bool = False,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Return {notify, reason, severity, source}. Never raises."""
    if failed:
        return {"notify": True, "reason": "任務失敗", "severity": "high", "source": "rule"}
    condition = (condition or "").strip()
    if not condition:
        return {"notify": True, "reason": "此排程未設通知條件，每次通知",
                "severity": "info", "source": "rule"}

    own = verdict_from_text(result_text)
    if own is not None:
        return own

    model = model or getattr(config, "notify_gate_model", "") or config.qwen_model
    user = (
        f"任務目標：{goal}\n通知條件：{condition}\n\n任務結果：\n"
        + (result_text or "（沒有結果）")[:6000]
    )
    try:
        payload = local_engine.build_payload(
            model=model,
            messages=[{"role": "user", "content": user}],
            system=_SYSTEM,
            tools=None, stream=False, temperature=0.1, num_predict=160,
            num_ctx=8192, keep_alive=getattr(config, "ollama_keep_alive", "24h"),
        )
        if not local_engine.is_vllm():
            payload["format"] = "json"
            payload["think"] = False
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=5.0)) as c:
            resp = await c.post(local_engine.chat_endpoint(), json=payload)
            if resp.status_code == 400 and "think" in resp.text.lower():
                payload.pop("think", None)
                resp = await c.post(local_engine.chat_endpoint(), json=payload)
            resp.raise_for_status()
            msg = local_engine.normalize_response(resp.json()).get("message") or {}
        raw = (msg.get("content") or "").strip()
        m = re.search(r"\{[\s\S]*\}", raw)
        data = json.loads(m.group(0) if m else raw)
        notify = bool(data.get("notify"))
        return {
            "notify": notify,
            "reason": str(data.get("reason") or ("條件成立" if notify else "正常"))[:300],
            "severity": str(data.get("severity") or ("warning" if notify else "info")),
            "source": "model",
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("notify gate could not judge (%s: %s) — notifying", type(exc).__name__, exc)
        return {"notify": True, "reason": f"無法判斷條件（{type(exc).__name__}），保守通知",
                "severity": "warning", "source": "fallback"}


def goal_with_verdict_request(goal: str, condition: str) -> str:
    """Ask the scheduled task itself to end with the ALERT line, so the
    common case costs no extra model call."""
    condition = (condition or "").strip()
    if not condition:
        return goal
    return (
        f"{goal}\n\n[通知條件：{condition}]\n"
        "完成報告後，最後一行必須是 `ALERT: yes` 或 `ALERT: no`（依上述條件判斷"
        "是否需要通知負責人），後面接一句原因並引用關鍵數字。"
    )
