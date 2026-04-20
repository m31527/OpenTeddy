"""
OpenTeddy Orchestrator
Gemma 4-powered planning agent.
Breaks a high-level goal into ordered SubTasks,
then drives Executor / Escalation to completion.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from config import config
from escalation import EscalationAgent
from executor import Executor
from models import AgentRole, SubTask, TaskRequest, TaskResult, TaskStatus
from skill_factory import SkillFactory
from tracker import Tracker

logger = logging.getLogger(__name__)

_PLAN_SYSTEM = """\
You are Teddy-Orch, the orchestrator of OpenTeddy — a self-growing multi-agent system.
Given a high-level goal, break it into an ordered list of concrete, atomic sub-tasks.
Output ONLY a JSON array.  Each element must have:
  - "description": string  (what to do)
  - "skill_hint": string or null  (name of an existing skill if you know one)
  - "order": integer  (execution order, starting at 0)
Keep the list to 10 or fewer sub-tasks.
"""


class Orchestrator:
    """Gemma-powered orchestrator that plans, dispatches, and aggregates."""

    def __init__(
        self,
        tracker: Tracker,
        executor: Executor,
        escalation: EscalationAgent,
        skill_factory: SkillFactory,
    ) -> None:
        self.tracker = tracker
        self.executor = executor
        self.escalation = escalation
        self.skill_factory = skill_factory
        self._http = httpx.AsyncClient(timeout=180)
        self._consecutive_failures: Dict[str, int] = {}

    async def close(self) -> None:
        await self._http.aclose()

    # ── Public API ────────────────────────────────────────────────────────────

    async def run(self, req: TaskRequest) -> TaskResult:
        """Full lifecycle: plan → execute → escalate → summarise."""
        logger.info("Orchestrator starting task %s: %s", req.id, req.goal)

        # Persist the task
        await self.tracker.create_task(req)
        await self.tracker.update_task_status(req.id, TaskStatus.RUNNING)

        try:
            # 1. Plan
            subtasks = await self._plan(req)
            for st in subtasks:
                await self.tracker.create_subtask(st)

            # 2. Execute subtasks sequentially (could be made parallel later)
            skills_used: list[str] = []
            new_skills: list[str] = []

            for st in subtasks:
                st = await self._run_subtask(st, req.context)

                if st.skill_hint:
                    skills_used.append(st.skill_hint)

                # Track consecutive failures per task
                if st.status == TaskStatus.FAILED:
                    self._consecutive_failures[req.id] = (
                        self._consecutive_failures.get(req.id, 0) + 1
                    )
                else:
                    self._consecutive_failures[req.id] = 0

                # Hard abort if too many sequential failures
                if (
                    self._consecutive_failures.get(req.id, 0)
                    >= config.escalation_failure_limit
                ):
                    logger.warning(
                        "Task %s hit failure limit — aborting remaining subtasks.",
                        req.id,
                    )
                    break

            # 3. Collect any newly created skills
            all_skills = await self.skill_factory.list_all_skills()
            new_skills = [s.name for s in all_skills if s.success_count == 0]

            # 4. Synthesise final summary
            results = [st.result or "" for st in subtasks if st.result]
            summary = await self.escalation.synthesize_summary(req.goal, results)

            overall_status = self._derive_status(subtasks)
            await self.tracker.update_task_status(req.id, overall_status, summary)

            return TaskResult(
                task_id=req.id,
                status=overall_status,
                summary=summary,
                subtasks=subtasks,
                skills_used=list(set(skills_used)),
                new_skills_created=new_skills,
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception("Orchestrator error for task %s: %s", req.id, exc)
            await self.tracker.update_task_status(
                req.id, TaskStatus.FAILED, str(exc)
            )
            raise

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _plan(self, req: TaskRequest) -> List[SubTask]:
        """Ask Gemma to decompose the goal into SubTasks."""
        active_skills = await self.skill_factory.list_active_skills()
        skill_names = [s.name for s in active_skills]

        prompt = (
            f"Goal: {req.goal}\n\n"
            f"Available skills: {json.dumps(skill_names)}\n\n"
            "Output the sub-task plan now."
        )
        raw_plan = await self._gemma_complete(prompt)
        subtasks = self._parse_plan(raw_plan, req.id)

        if not subtasks:
            # Fallback: single subtask = the whole goal
            subtasks = [
                SubTask(
                    parent_task_id=req.id,
                    description=req.goal,
                    agent=AgentRole.EXECUTOR,
                    order=0,
                )
            ]
        logger.info("Planned %d subtasks for task %s", len(subtasks), req.id)
        return subtasks

    async def _run_subtask(self, st: SubTask, context: Dict[str, Any]) -> SubTask:
        """Execute one subtask; escalate if confidence too low."""
        st = await self.executor.execute(st, context)

        should_escalate = (
            st.status == TaskStatus.FAILED
            or st.confidence < config.escalation_confidence_threshold
        )
        if should_escalate:
            logger.info(
                "Subtask %s needs escalation (status=%s, conf=%.2f)",
                st.id, st.status, st.confidence,
            )
            st = await self.escalation.resolve(st, context)
        return st

    async def _gemma_complete(self, prompt: str) -> str:
        payload = {
            "model": config.gemma_model,
            "prompt": prompt,
            "system": _PLAN_SYSTEM,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 1024},
        }
        try:
            resp = await self._http.post(
                f"{config.gemma_base_url}/api/generate",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as exc:  # noqa: BLE001
            logger.error("Gemma call failed: %s", exc)
            return "[]"

    @staticmethod
    def _parse_plan(raw: str, task_id: str) -> List[SubTask]:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            items = json.loads(match.group())
        except json.JSONDecodeError:
            return []

        subtasks = []
        for item in items[:10]:
            if not isinstance(item, dict):
                continue
            desc = item.get("description", "").strip()
            if not desc:
                continue
            subtasks.append(
                SubTask(
                    parent_task_id=task_id,
                    description=desc,
                    skill_hint=item.get("skill_hint") or None,
                    agent=AgentRole.EXECUTOR,
                    order=int(item.get("order", len(subtasks))),
                )
            )
        subtasks.sort(key=lambda s: s.order)
        return subtasks

    @staticmethod
    def _derive_status(subtasks: List[SubTask]) -> TaskStatus:
        statuses = {st.status for st in subtasks}
        if TaskStatus.FAILED in statuses:
            if all(s in (TaskStatus.FAILED,) for s in statuses):
                return TaskStatus.FAILED
            return TaskStatus.COMPLETED  # partial success
        if TaskStatus.ESCALATED in statuses:
            return TaskStatus.COMPLETED
        return TaskStatus.COMPLETED
