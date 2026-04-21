"""
OpenTeddy Executor
Qwen-powered agent that executes individual SubTasks.
Supports Ollama Function Calling for tool use (shell, file, GCP, DB, HTTP).
Falls back to direct LLM inference if no tools are needed.
Reports confidence so the Orchestrator can decide on escalation.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from config import config
from models import AgentRole, SubTask, TaskStatus
from skill_factory import SkillFactory
from tool_registry import ToolRegistry, tool_registry as _default_registry
from tracker import Tracker

logger = logging.getLogger(__name__)

_MAX_TOOL_ROUNDS = 10   # prevent infinite tool-call loops

_SYSTEM_PROMPT = """\
You are Teddy-Exec, a precise task executor powered by Qwen.
You receive a sub-task description and any relevant context.
You have access to tools — use them when needed.
After completing the task (or if no tools are needed), output a JSON object:
  {
    "result": "<string: your answer / action result>",
    "confidence": <float 0.0–1.0>,
    "skill_needed": "<string snake_case or null>",
    "skill_description": "<string or null>"
  }
Output ONLY the JSON object — no prose, no markdown.
"""


class Executor:
    """Qwen 3 executor agent with Ollama function-calling support."""

    def __init__(
        self,
        tracker: Tracker,
        skill_factory: SkillFactory,
        registry: Optional[ToolRegistry] = None,
        ws_callback: Optional[Callable] = None,
    ) -> None:
        self.tracker = tracker
        self.skill_factory = skill_factory
        self.registry: ToolRegistry = registry or _default_registry
        self.ws_callback = ws_callback   # async fn(event_dict) for UI pushes
        self._http = httpx.AsyncClient(timeout=120)

    async def close(self) -> None:
        await self._http.aclose()

    # ── Public API ────────────────────────────────────────────────────────────

    async def execute(self, subtask: SubTask, context: Dict) -> SubTask:
        """Execute a subtask. Returns the updated subtask with result / status."""
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

        # 2. Qwen with function calling
        result, confidence, skill_hint, skill_desc = await self._qwen_execute(
            subtask.description, context, task_id=subtask.parent_task_id
        )
        subtask.result = result
        subtask.confidence = confidence
        subtask.skill_hint = skill_hint

        if confidence >= config.escalation_confidence_threshold:
            subtask.status = TaskStatus.COMPLETED
        else:
            subtask.status = TaskStatus.FAILED

        subtask.completed_at = datetime.utcnow()
        await self.tracker.update_subtask(subtask)

        # 3. Background: request skill creation if signalled
        if skill_hint and skill_desc:
            await self._request_skill_creation(skill_hint, skill_desc)

        return subtask

    # ── Ollama Function Calling ───────────────────────────────────────────────

    async def _qwen_execute(
        self,
        description: str,
        context: Dict,
        task_id: str = "unknown",
    ) -> Tuple[str, float, Optional[str], Optional[str]]:
        """
        Multi-turn Qwen chat with Ollama function calling.
        Loop up to _MAX_TOOL_ROUNDS tool calls before forcing a final answer.
        """
        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"Task: {description}\n\n"
                    f"Context: {json.dumps(context, ensure_ascii=False)[:2000]}"
                ),
            }
        ]

        tools = self.registry.get_schemas()

        for round_idx in range(_MAX_TOOL_ROUNDS):
            payload: Dict[str, Any] = {
                "model": config.qwen_model,
                "messages": messages,
                "system": _SYSTEM_PROMPT,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 1024},
            }
            if tools:
                payload["tools"] = tools

            try:
                resp = await self._http.post(
                    f"{config.qwen_base_url}/api/chat",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:  # noqa: BLE001
                logger.error("Qwen chat call failed (round %d): %s", round_idx, exc)
                return f"Executor error: {exc}", 0.0, None, None

            message = data.get("message", {})
            tool_calls: List[Dict[str, Any]] = message.get("tool_calls") or []

            # ── No tool calls → final answer ──────────────────────────────────
            if not tool_calls:
                raw_text = message.get("content", "")
                return self._parse_qwen_response(raw_text)

            # ── Process tool calls ────────────────────────────────────────────
            # Add assistant's message (with tool_calls) to history
            messages.append({
                "role": "assistant",
                "content": message.get("content", ""),
                "tool_calls": tool_calls,
            })

            # Execute each tool and collect results
            for call in tool_calls:
                fn_info = call.get("function", {})
                tool_name: str = fn_info.get("name", "")
                raw_args = fn_info.get("arguments", {})
                args: Dict[str, Any] = (
                    raw_args if isinstance(raw_args, dict)
                    else json.loads(raw_args)
                )

                logger.info(
                    "Tool call round=%d tool=%s args=%s task=%s",
                    round_idx, tool_name, args, task_id,
                )

                # Push event to Web UI
                await self._push_event({
                    "event": "tool_call",
                    "round": round_idx,
                    "tool": tool_name,
                    "args": args,
                    "task_id": task_id,
                })

                tool_result = await self.registry.execute(
                    tool_name, args, task_id=task_id
                )

                logger.info(
                    "Tool result tool=%s success=%s duration_ms=%s",
                    tool_name,
                    tool_result.get("success"),
                    tool_result.get("duration_ms"),
                )

                # Push result event to Web UI
                await self._push_event({
                    "event": "tool_result",
                    "round": round_idx,
                    "tool": tool_name,
                    "success": tool_result.get("success"),
                    "error": tool_result.get("error"),
                    "task_id": task_id,
                })

                # Add tool result to conversation history
                messages.append({
                    "role": "tool",
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

        # Hit max rounds — ask for a final forced answer
        logger.warning(
            "Reached max tool rounds (%d) for task %s — forcing final answer.",
            _MAX_TOOL_ROUNDS, task_id,
        )
        messages.append({
            "role": "user",
            "content": (
                "You have used the maximum number of tool calls. "
                "Summarise what you've learned and output the final JSON now."
            ),
        })
        try:
            resp = await self._http.post(
                f"{config.qwen_base_url}/api/chat",
                json={
                    "model": config.qwen_model,
                    "messages": messages,
                    "system": _SYSTEM_PROMPT,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 512},
                },
            )
            resp.raise_for_status()
            raw_text = resp.json().get("message", {}).get("content", "")
        except Exception as exc:  # noqa: BLE001
            return f"Executor error (forced final): {exc}", 0.0, None, None

        return self._parse_qwen_response(raw_text)

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _push_event(self, event: Dict[str, Any]) -> None:
        """Push an event to the Web UI via ws_callback (if registered)."""
        if self.ws_callback:
            try:
                await self.ws_callback(event)
            except Exception as exc:  # noqa: BLE001
                logger.debug("ws_callback error: %s", exc)

    async def _try_skill(
        self, subtask: SubTask, context: Dict
    ) -> Optional[Tuple[bool, str]]:
        """Return (success, output) if a matching active skill exists, else None."""
        skills = await self.skill_factory.list_active_skills()
        best: Optional[str] = None

        if subtask.skill_hint:
            for s in skills:
                if s.name == subtask.skill_hint:
                    best = s.name
                    break

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

    @staticmethod
    def _parse_qwen_response(
        text: str,
    ) -> Tuple[str, float, Optional[str], Optional[str]]:
        """Extract fields from Qwen's JSON output (tolerant of extra text)."""
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
                return
            logger.info("Requesting new skill creation: '%s'", name)
            await self.skill_factory.generate_skill(name, description)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skill creation for '%s' failed: %s", name, exc)
