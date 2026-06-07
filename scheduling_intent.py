"""
Natural-language scheduling intent detector.

Goal: any chat surface (Telegram, web, desktop) should let users say
"每天早上 9 點幫我抓 GitHub trending top 10" and have it auto-scheduled,
no /cron prefix or cron-syntax knowledge required.

Pipeline:

  ┌─────────────────────┐   no match (99% of msgs)   ┌──────────────┐
  │ raw user message    │ ─────────────────────────▶ │ return None  │
  └──────────┬──────────┘                            └──────────────┘
             │ regex hits a time-recurrence hint
             ▼
  ┌─────────────────────┐    LLM says is_schedule=false / low conf
  │ small LLM call:     │ ─────────────────────────▶  return None
  │ "is this a sched?   │
  │ if yes give me cron"│    LLM says yes + cron + goal + confidence
  └──────────┬──────────┘                            ┌──────────────┐
             └─────────────────────────────────────▶ │ SchedulingIntent│
                                                    └──────────────┘

We pre-filter with regex because the LLM call is the expensive part
(~100-300 ms on a small Ollama model, more for a cloud model). The
overwhelming majority of chat turns ("draft an email", "what's wrong
with this code") never look anything like a schedule, and we should
short-circuit them for free. False positives on the regex are fine —
the LLM filters them out. False negatives are the real cost (we'd
silently treat a real schedule request as a one-shot), so the regex
is intentionally generous.

The cron syntax we return is APScheduler-compatible 5-field
"minute hour day-of-month month day-of-week", same format scheduler.py's
``_validate_cron`` already accepts. So the caller's "add_schedule()"
just works without extra glue.

Cancellation goes through the existing natural-language cancel path
in telegram_bridge (and a parallel path we'll add to the web chat) —
"取消那個 GitHub trending 的排程" / "cancel my morning schedule" already
work, no need to duplicate the detection here.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ── Result type ──────────────────────────────────────────────────────────────


@dataclass
class SchedulingIntent:
    """Output of the detector. Caller passes ``cron`` + ``task_goal`` to
    ``scheduler.add_schedule()`` and uses ``summary`` for the
    user-facing confirmation message ("好的，我會每天 09:00 執行...")."""
    cron: str           # 5-field crontab (APScheduler-compatible)
    task_goal: str      # what the agent will run when the schedule fires
    summary: str        # short human-readable timing description
    confidence: float   # 0-1, 0.7 default threshold


# ── Stage 1: regex pre-filter ────────────────────────────────────────────────

# Looks for ANY of:
#   - Chinese time recurrence words: 每天 / 每週 / 每月 / 每小時 / 每分 /
#                                    每年 / 每隔
#   - Chinese weekday markers: 週X / 星期X
#   - Chinese time-of-day words: 凌晨 / 早上 / 清晨 / 中午 / 下午 / 傍晚 /
#                                晚上 / 半夜 / 午夜
#   - Clock formats: 9:30, 9：30, 9 點, 9點半 (covered by 9 點 + 9:30)
#   - English time-recurrence: every X / each X / daily / weekly / monthly
#                              / hourly / nightly
#   - English clock: at 9, at 9:30, at 9am
#   - English time-of-day: noon / midnight / morning / evening
#   - Literal: cron
#
# We DON'T match plain dates ("明天", "tomorrow") since those are
# one-shot tasks — scheduler.py only supports recurring crons today.
# When we add one-shot date triggers we'll widen this.
_PREFILTER_RE = re.compile(
    r"""
    (
        每\s*(天|日|週|周|月|小時|個小時|分鐘?|年|隔)
      | (週|周|星期)\s*(一|二|三|四|五|六|日|天)
      | (凌晨|早上|清晨|中午|下午|傍晚|晚上|半夜|午夜)
      | \d{1,2}\s*[:：.]\s*\d{2}
      | \d{1,2}\s*點(半|鐘)?
      | \b(every|each)\s+(day|week|month|hour|minute|morning|evening|
                          monday|tuesday|wednesday|thursday|friday|
                          saturday|sunday|night)\b
      | \b(daily|weekly|monthly|hourly|nightly)\b
      | \bat\s+\d{1,2}(:\d{2})?\s*(am|pm)?\b
      | \b(noon|midnight)\b
      | \bcron\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def regex_might_be_schedule(text: str) -> bool:
    """Fast & cheap pre-screen. False positives are OK (the LLM call
    filters them); false negatives skip real schedule requests."""
    if not text or not text.strip():
        return False
    return bool(_PREFILTER_RE.search(text))


# ── Stage 2: LLM extraction ──────────────────────────────────────────────────

_LLM_SYSTEM_PROMPT = """You are a scheduling-intent classifier for an AI \
agent. Read the user's message and decide if they're asking to SCHEDULE \
a recurring task (not execute one immediately).

Examples that ARE scheduling (is_schedule=true):
  - "每天早上 9 點去 GitHub 抓 trending top 10"
  - "every Monday at 8am summarise the top HN posts"
  - "每小時檢查一次系統狀態"
  - "weekdays at 18:00 send me my unread Slack messages"

Examples that are NOT scheduling (is_schedule=false):
  - "現在幫我抓 GitHub trending"            (one-shot, "now")
  - "What time is it?"                       (question)
  - "明天提醒我開會"                          (one-shot single date)
  - "9 點開會"                                (statement of fact, no action)
  - "draft an email"                          (no time component)

Cron format (5 fields, APScheduler-compatible):
  minute hour day-of-month month day-of-week
  - "每天早上 9 點"          → "0 9 * * *"
  - "每天 09:30"             → "30 9 * * *"
  - "每週一早上 8 點"         → "0 8 * * 1"
  - "工作日 18 點"            → "0 18 * * 1-5"
  - "每小時"                  → "0 * * * *"
  - "每月 1 號 8 點"          → "0 8 1 * *"

day-of-week: 0=Sunday, 1=Monday, ..., 6=Saturday.

Reply with ONLY a JSON object (no markdown fences, no explanation):
{
  "is_schedule": true | false,
  "cron":        "<5-field cron, or empty if not a schedule>",
  "task_goal":   "<what the agent should DO at that time, written as \
an actionable instruction in the user's original language>",
  "summary":     "<short human-readable timing, in user's language. \
e.g. '每天 09:00' or 'every Monday at 08:00'>",
  "confidence":  <float 0.0 to 1.0>
}

Set confidence high (>0.8) when both the time AND the action are clear.
Set low (<0.5) when the time is fuzzy or the action is missing.
"""


async def llm_extract(text: str) -> Optional[SchedulingIntent]:
    """Run the LLM classifier. Uses the configured default provider
    (Anthropic / Ollama / OpenAI / etc.) via ``complete_text``.

    Failures (provider not configured, network error, malformed JSON)
    all silently return ``None`` — better to fall through to the normal
    planner than to surface an opaque classifier crash to the user.
    """
    try:
        from llm_provider import get_default_provider, LLMProviderError
    except ImportError:
        logger.debug("scheduling_intent: llm_provider not importable")
        return None

    provider = get_default_provider()
    if not provider.is_configured():
        # No key configured — silently skip. The user can still create
        # schedules through the API or Telegram /cron command.
        logger.debug("scheduling_intent: provider not configured, skipping LLM")
        return None

    try:
        resp = await provider.complete_text(
            user_message=text,
            system=_LLM_SYSTEM_PROMPT,
            max_tokens=300,
        )
        raw = (resp.text or "").strip()
    except LLMProviderError as exc:
        logger.info("scheduling_intent: provider error %s", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("scheduling_intent: unexpected LLM error: %s", exc)
        return None

    if not raw:
        return None

    # Strip code fences in case the model added them despite instructions.
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        # Some models prepend "Here's the JSON:" before the payload —
        # try to recover the first {...} block.
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            logger.info(
                "scheduling_intent: non-JSON response (%s): %r",
                exc, raw[:200],
            )
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            logger.info(
                "scheduling_intent: nested JSON also failed: %r", raw[:200]
            )
            return None

    if not data.get("is_schedule"):
        return None

    cron = (data.get("cron") or "").strip()
    task_goal = (data.get("task_goal") or "").strip()
    if not cron or not task_goal:
        return None

    return SchedulingIntent(
        cron=cron,
        task_goal=task_goal,
        summary=(data.get("summary") or "").strip() or cron,
        confidence=float(data.get("confidence") or 0.0),
    )


# ── Public entrypoint ────────────────────────────────────────────────────────


async def detect_scheduling_intent(
    text: str,
    *,
    confidence_threshold: float = 0.7,
) -> Optional[SchedulingIntent]:
    """Two-stage detect. Returns ``None`` for the common case (normal
    chat message — planner runs as usual). Returns a
    :class:`SchedulingIntent` when the LLM is confident this is a
    recurring schedule request.

    Caller workflow:

        intent = await detect_scheduling_intent(user_message)
        if intent:
            row = await scheduler.add_schedule(
                session_id=session_id, cron=intent.cron, goal=intent.task_goal,
            )
            await reply(f"✓ 排好了：{intent.summary}\\n任務：{intent.task_goal}\\n"
                        f"(id: {row['id'][:8]} · 取消請說「取消那個排程」)")
            return  # skip the planner
        # otherwise: normal planner flow
    """
    if not regex_might_be_schedule(text):
        return None
    intent = await llm_extract(text)
    if intent is None:
        return None
    if intent.confidence < confidence_threshold:
        logger.info(
            "scheduling_intent: flagged but confidence %.2f < %.2f — "
            "treating as one-shot task",
            intent.confidence, confidence_threshold,
        )
        return None
    return intent
