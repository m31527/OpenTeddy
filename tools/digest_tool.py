"""
OpenTeddy Digest Tool
─────────────────────────────────────────────────────────────────────────────
One question, the whole picture.

Scheduled jobs each report into their own session, which is correct for
running the work and useless for oversight: an owner with a dozen daily
jobs would have to open a dozen sessions and read each result on its own.
This tool pulls the latest run of every schedule into a single answer, so
"how did the company do today?" can be asked once, anywhere — in a chat,
or as a scheduled job of its own that pushes the summary to Telegram.

Reporting rules baked into the output rather than left to the model:

  * a schedule that FAILED, or that hasn't reported inside the window, is
    surfaced first and never dropped. Silence is the most dangerous
    result to omit — an owner reading a clean digest would conclude
    nothing was wrong, when in fact a job stopped running.
  * results are returned verbatim. Summarising a summary is where
    numbers drift; the caller sees what the job actually said.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)

_MAX_RESULT_CHARS = 1200


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


async def read_schedule_digest(hours: int = 24) -> Dict[str, Any]:
    """Summarise the latest run of every scheduled job. LOW risk."""
    start = time.monotonic()
    try:
        import main as _main_module
        rows: List[dict] = await _main_module.tracker.schedule_digest(hours=hours)
    except Exception as exc:  # noqa: BLE001
        logger.error("digest error: %s", exc)
        return make_result(False, error=str(exc), duration_ms=_ms(start))

    if not rows:
        return make_result(
            True,
            result={
                "schedules": 0,
                "note": "No enabled schedules exist yet — nothing to report.",
            },
            duration_ms=_ms(start),
        )

    needs_attention: List[dict] = []
    reported: List[dict] = []
    for r in rows:
        item = {
            "session": r.get("session_title") or "(untitled)",
            "session_id": r.get("session_id"),
            "goal": r.get("goal"),
            "cron": r.get("cron"),
            "last_run_at": r.get("last_run_at"),
            "status": r.get("last_status"),
        }
        if r.get("last_status") == "failure" or r.get("stale"):
            item["problem"] = (
                r.get("last_error")
                or (f"No run recorded in the last {hours}h — this job may "
                    f"have stopped firing.")
            )
            item["consecutive_failures"] = r.get("consecutive_failures") or 0
            needs_attention.append(item)
        else:
            res = (r.get("result") or "").strip()
            item["result"] = (
                res[:_MAX_RESULT_CHARS] + " …(truncated)"
                if len(res) > _MAX_RESULT_CHARS else res
            ) or "(the job reported no summary)"
            reported.append(item)

    return make_result(
        True,
        result={
            "window_hours": hours,
            "schedules": len(rows),
            # Ordered deliberately: problems before results, so a reader
            # who stops early still sees what is broken.
            "needs_attention": needs_attention,
            "reported": reported,
            "instruction": (
                "Report needs_attention FIRST and never omit it — a job "
                "that failed or went silent is the most important item "
                "here. Then summarise the reported results. Quote the "
                "numbers each job produced; do not re-derive or estimate "
                "them."
            ),
        },
        duration_ms=_ms(start),
    )


_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_schedule_digest",
        "description": (
            "Get the latest result of EVERY scheduled job across all "
            "sessions, plus any job that failed or stopped reporting. "
            "Use this for questions about overall status — 'how are "
            "things today', 'what did the scheduled reports say', "
            "'anything broken'. Returns what each job actually reported."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "hours": {
                    "type": "integer",
                    "description": (
                        "How recent a run must be to count as current "
                        "(default 24). Older jobs are flagged as stale."
                    ),
                },
            },
            "required": [],
        },
    },
}

DIGEST_TOOLS: List[tuple] = [
    (read_schedule_digest, _SCHEMA, "low"),
]
