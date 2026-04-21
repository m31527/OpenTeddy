"""
OpenTeddy Escalation Agent
When Qwen's confidence is too low or repeated failures occur,
this agent calls Claude to resolve the subtask.
All Anthropic API calls are recorded to the usage_records table.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict

import anthropic

from config import config
from models import SubTask, TaskStatus
from tracker import Tracker

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are Teddy-Prime, an expert AI assistant acting as the escalation layer in \
OpenTeddy — a self-growing multi-agent system.
A junior agent (Qwen) attempted the task below but lacked confidence.
Your job: complete the task accurately and completely.
Be concise but thorough.  If you need to write code, write it.
"""


class EscalationAgent:
    """Claude-based escalation handler."""

    def __init__(self, tracker: Tracker) -> None:
        self.tracker = tracker
        self._claude = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    async def resolve(self, subtask: SubTask, context: Dict) -> SubTask:
        """
        Use Claude to resolve a failed/low-confidence subtask.
        Updates and returns the subtask.
        Records token usage to usage_records.
        """
        logger.info(
            "Escalating subtask %s to Claude (confidence was %.2f)",
            subtask.id,
            subtask.confidence,
        )
        subtask.status = TaskStatus.ESCALATED
        await self.tracker.update_subtask(subtask)

        prior_attempt = subtask.result or "(none)"
        user_message = (
            f"Task description: {subtask.description}\n\n"
            f"Context: {str(context)[:3000]}\n\n"
            f"Previous attempt result: {prior_attempt}\n\n"
            "Please provide the correct, complete answer."
        )

        try:
            response = await self._claude.messages.create(
                model=config.claude_model,
                max_tokens=2048,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            result_text = response.content[0].text.strip()
            subtask.result = result_text
            subtask.confidence = 1.0
            subtask.status = TaskStatus.COMPLETED

            # ── Record usage ─────────────────────────────────────────────────
            usage = response.usage
            await self.tracker.record_usage(
                task_id=subtask.parent_task_id,
                model=config.claude_model,
                model_provider="anthropic",
                tokens_in=usage.input_tokens,
                tokens_out=usage.output_tokens,
                task_description=subtask.description[:300],
            )

        except anthropic.APIError as exc:
            logger.error("Claude escalation failed: %s", exc)
            subtask.error = f"Escalation error: {exc}"
            subtask.status = TaskStatus.FAILED

        subtask.completed_at = datetime.utcnow()
        await self.tracker.update_subtask(subtask)
        return subtask

    async def synthesize_summary(
        self,
        goal: str,
        subtask_results: list[str],
        task_id: str = "",
    ) -> str:
        """
        Ask Claude to write a final coherent summary from all subtask outputs.
        Records token usage to usage_records.
        """
        numbered = "\n".join(
            f"{i+1}. {r}" for i, r in enumerate(subtask_results)
        )
        prompt = (
            f"Original goal: {goal}\n\n"
            f"Completed sub-results:\n{numbered}\n\n"
            "Write a concise, well-structured final summary that directly answers "
            "the original goal."
        )
        try:
            response = await self._claude.messages.create(
                model=config.claude_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = response.content[0].text.strip()

            # ── Record usage ─────────────────────────────────────────────────
            usage = response.usage
            await self.tracker.record_usage(
                task_id=task_id,
                model=config.claude_model,
                model_provider="anthropic",
                tokens_in=usage.input_tokens,
                tokens_out=usage.output_tokens,
                task_description=f"[summary] {goal[:250]}",
            )

            return summary

        except anthropic.APIError as exc:
            logger.error("Summary synthesis failed: %s", exc)
            return f"(Summary unavailable: {exc})\n\nRaw results:\n{numbered}"
