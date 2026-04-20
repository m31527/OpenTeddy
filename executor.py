"""
OpenTeddy Executor
Qwen-powered agent that executes individual SubTasks.
Tries matching skills first; falls back to direct LLM inference.
Reports confidence so the Orchestrator can decide on escalation.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Dict, Optional, Tuple

import httpx

from config import config
from models import AgentRole, SubTask, TaskStatus
from skill_factory import SkillFactory
from tracker import Tracker

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are Teddy-Exec, a precise task executor powered by Qwen.
You receive a sub-task description and any relevant context.
Your job:
1. Attempt to complete the task directly using the provided context.
2. Output a JSON object with keys:
   - "result": string (your answer / action result)
   - "confidence": float 0.0-1.0 (how certain you are)
   - "skill_needed": string or null (name of a NEW skill that would help in future, \
     snake_case, or null if none needed)
   - "skill_description": string or null (one-sentence description of that skill)
Output ONLY the JSON object — no prose, no markdown.
"""


class Executor:
    """Qwen 3 executor agent."""

    def __init__(self, tracker: Tracker, skill_factory: SkillFactory) -> None:
        self.tracker = tracker
        self.skill_factory = skill_factory
        self._http = httpx.AsyncClient(timeout=120)

    async def close(self) -> None:
        await self._http.aclose()

    # ── Public API ────────────────────────────────────────────────────────────

    async def execute(self, subtask: SubTask, context: Dict) -> SubTask:
        """
        Execute a subtask.  Returns the updated subtask with result / status.
        """
        subtask.status = TaskStatus.RUNNING
        await self.tracker.update_subtask(subtask)

        # 1. Try a matching skill first
        skill_result = await self._try_skill(subtask, context)
        if skill_result is not None:
            success, output = skill_result
            subtask.result = output
            subtask.confidence = 0.95 if success else 0.2
            subtask.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
            subtask.completed_at = datetime.utcnow()
            await self.tracker.update_subtask(subtask)
            return subtask

        # 2. Fall back to Qwen inference
        result, confidence, skill_hint, skill_desc = await self._qwen_execute(
            subtask.description, context
        )
        subtask.result = result
        subtask.confidence = confidence
        subtask.skill_hint = skill_hint

        if confidence >= config.escalation_confidence_threshold:
            subtask.status = TaskStatus.COMPLETED
        else:
            subtask.status = TaskStatus.FAILED  # will be picked up for escalation

        subtask.completed_at = datetime.utcnow()
        await self.tracker.update_subtask(subtask)

        # 3. Background: request skill generation if Qwen signalled one is needed
        if skill_hint and skill_desc:
            await self._request_skill_creation(skill_hint, skill_desc)

        return subtask

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _try_skill(
        self, subtask: SubTask, context: Dict
    ) -> Optional[Tuple[bool, str]]:
        """Return (success, output) if a matching active skill exists, else None."""
        skills = await self.skill_factory.list_active_skills()
        best: Optional[str] = None

        # Prefer explicitly hinted skill
        if subtask.skill_hint:
            for s in skills:
                if s.name == subtask.skill_hint:
                    best = s.name
                    break

        # Simple keyword match fallback
        if not best:
            desc_lower = subtask.description.lower()
            for s in skills:
                if any(kw in desc_lower for kw in s.name.replace("_", " ").split()):
                    best = s.name
                    break

        if not best:
            return None

        logger.info("Executor invoking skill '%s' for subtask %s", best, subtask.id)
        return await self.skill_factory.invoke_skill(best, subtask.id, context)

    async def _qwen_execute(
        self,
        description: str,
        context: Dict,
    ) -> Tuple[str, float, Optional[str], Optional[str]]:
        """Call Qwen via Ollama API, parse structured JSON response."""
        prompt = (
            f"Task: {description}\n\n"
            f"Context: {json.dumps(context, ensure_ascii=False)[:2000]}"
        )
        payload = {
            "model": config.qwen_model,
            "prompt": prompt,
            "system": _SYSTEM_PROMPT,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 512},
        }
        try:
            resp = await self._http.post(
                f"{config.qwen_base_url}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
            raw_text = resp.json().get("response", "")
        except Exception as exc:  # noqa: BLE001
            logger.error("Qwen call failed: %s", exc)
            return f"Executor error: {exc}", 0.0, None, None

        return self._parse_qwen_response(raw_text)

    @staticmethod
    def _parse_qwen_response(
        text: str,
    ) -> Tuple[str, float, Optional[str], Optional[str]]:
        """Extract fields from Qwen's JSON output (tolerant of extra text)."""
        # Try to find JSON block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return text.strip(), 0.3, None, None
        try:
            data = json.loads(match.group())
            result = str(data.get("result", text.strip()))
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            skill_hint = data.get("skill_needed") or None
            skill_desc = data.get("skill_description") or None
            return result, confidence, skill_hint, skill_desc
        except (json.JSONDecodeError, ValueError):
            return text.strip(), 0.3, None, None

    async def _request_skill_creation(self, name: str, description: str) -> None:
        """Fire-and-forget: ask SkillFactory to build a new skill."""
        try:
            existing = await self.skill_factory.tracker.get_skill(name)
            if existing:
                return  # already exists
            logger.info("Requesting new skill creation: '%s'", name)
            await self.skill_factory.generate_skill(name, description)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skill creation for '%s' failed: %s", name, exc)
