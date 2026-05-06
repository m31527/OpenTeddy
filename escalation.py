"""
OpenTeddy Escalation Agent
When Qwen's confidence is too low or repeated failures occur,
this agent calls the cloud LLM (today: Claude) to resolve the subtask.
All cloud LLM calls are recorded to the usage_records table.

The agent talks to whatever provider is wired through
:func:`llm_provider.get_default_provider` — Anthropic today,
OpenRouter / OpenAI / Gemini in the future. Nothing in this file
imports a specific SDK; only the provider layer does.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from config import config
from llm_provider import (
    LLMProvider,
    LLMProviderError,
    LLMToolResult,
    get_default_provider,
)
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

_SYSTEM_PROMPT_TIMEOUT = """\
You are Teddy-Prime, an expert AI assistant acting as the escalation layer in \
OpenTeddy — a self-growing multi-agent system.
The junior agent (Qwen) got STUCK and timed out while executing the task below — \
it did not return a result at all.
Your job: provide the correct, complete solution directly.
Do NOT ask "can you run this?" — just give the answer, the command, or the code.
If this involves Docker or shell operations, provide the exact commands to run.
"""


class EscalationAgent:
    """Cloud-LLM escalation handler.

    The model is selected by :func:`llm_provider.get_default_provider`,
    which today returns the Anthropic provider but will switch on a
    future ``config.llm_provider`` setting (OpenRouter / OpenAI /
    Gemini). All token usage is logged with the provider's stable
    ``provider_name`` so the Usage tab can attribute cost correctly.
    """

    def __init__(
        self,
        tracker: Tracker,
        provider: Optional[LLMProvider] = None,
    ) -> None:
        self.tracker = tracker
        # Provider is injectable for tests; production code uses the
        # default. Stored on the instance so a settings change that
        # swaps providers can be picked up by reassigning this field
        # without re-instantiating the agent.
        self.provider = provider or get_default_provider()

    @staticmethod
    def _disabled_message() -> str:
        return (
            "Cloud-LLM escalation is disabled. Toggle Settings → Model Settings → "
            "Allow Claude escalation back ON to let the cloud model resolve hard "
            "subtasks, or accept the local-only result."
        )

    def _escalation_blocked(self) -> str | None:
        """Return a friendly message if escalation should not run, else None."""
        if not config.escalation_enabled:
            return self._disabled_message()
        if not self.provider.is_configured():
            return self.provider.get_unconfigured_message()
        return None

    async def resolve(self, subtask: SubTask, context: Dict) -> SubTask:
        """
        Use the cloud LLM to resolve a failed/low-confidence subtask.
        Updates and returns the subtask.
        Records token usage to usage_records.
        """
        # Detect whether escalation was triggered by a timeout/hang
        error_text = subtask.error or ""
        is_timeout = "超時" in error_text or "timed out" in error_text.lower()

        logger.info(
            "Escalating subtask %s to %s (confidence was %.2f, timeout=%s)",
            subtask.id,
            self.provider.provider_name,
            subtask.confidence,
            is_timeout,
        )
        subtask.status = TaskStatus.ESCALATED
        await self.tracker.update_subtask(subtask)

        system_prompt = _SYSTEM_PROMPT_TIMEOUT if is_timeout else _SYSTEM_PROMPT

        prior_attempt = subtask.result or "(none)"
        timeout_note  = (
            f"\n\nNote: The junior agent timed out — it never returned a result. "
            f"Error: {error_text}"
        ) if is_timeout else ""

        user_message = (
            f"Task description: {subtask.description}\n\n"
            f"Context: {str(context)[:3000]}\n\n"
            f"Previous attempt result: {prior_attempt}"
            f"{timeout_note}\n\n"
            "Please provide the correct, complete answer."
        )

        # Pre-flight: bail with a friendly message if escalation is off
        # globally OR no API key is configured. Either way we fail fast
        # rather than letting the SDK error bubble up.
        block_msg = self._escalation_blocked()
        if block_msg:
            subtask.error = block_msg
            subtask.status = TaskStatus.FAILED
            subtask.completed_at = datetime.utcnow()
            await self.tracker.update_subtask(subtask)
            return subtask

        try:
            response = await self.provider.complete_text(
                user_message=user_message,
                system=system_prompt,
                max_tokens=2048,
            )
            subtask.result = response.text
            subtask.confidence = 1.0
            subtask.status = TaskStatus.COMPLETED

            # ── Record usage ─────────────────────────────────────────────────
            await self.tracker.record_usage(
                task_id=subtask.parent_task_id,
                model=self.provider.model_name,
                model_provider=self.provider.provider_name,
                tokens_in=response.usage.input_tokens,
                tokens_out=response.usage.output_tokens,
                task_description=subtask.description[:300],
            )

        except LLMProviderError as exc:
            logger.error("%s escalation failed: %s", self.provider.provider_name, exc)
            subtask.error = f"Escalation error: {exc}"
            subtask.status = TaskStatus.FAILED

        subtask.completed_at = datetime.utcnow()
        await self.tracker.update_subtask(subtask)
        return subtask

    async def resolve_whole_task(
        self,
        task: Dict[str, Any],
        subtasks: List[Dict[str, Any]],
        tool_registry,
        session_id: Optional[str] = None,
        user_hint: Optional[str] = None,
        ws_callback: Optional[Callable] = None,
        max_turns: int = 15,
    ) -> Dict[str, Any]:
        """User-triggered whole-task takeover. Given the full history of
        local-model attempts on a failed task, let the cloud model drive
        the remaining work end-to-end with full tool access.

        Differs from ``resolve()`` in three ways:

          1. **Scope** — sees the whole task + all subtask attempts, not
             one lonely subtask. That's what lets it fix the
             "Dockerfile COPY order" / "missing env var" kind of bugs
             a single-subtask escalation can't reason about.
          2. **Tool use** — model calls tools directly (shell, file,
             docker helpers, deploy tools) via the provider's tool-use
             loop. No hand-off back to Qwen.
          3. **User-gated** — only runs when the UI button is clicked,
             so the token spend is opt-in.

        Returns ``{success, summary, tools_used, turns}``.
        """
        registry = tool_registry
        task_id = task.get("id", "")

        # Build the prior-attempts digest. Cap each field so the total
        # context stays reasonable even on a 10-subtask task.
        prior_lines: List[str] = []
        for idx, st in enumerate(subtasks, 1):
            desc   = (st.get("description") or "")[:300]
            status = st.get("status", "?")
            conf   = float(st.get("confidence", 0) or 0)
            result = (st.get("result") or "")[:600]
            err    = (st.get("error")  or "")[:400]
            block = [f"#{idx} [{status} conf={conf:.2f}] {desc}"]
            if result:
                block.append(f"    result: {result}")
            if err:
                block.append(f"    error:  {err}")
            prior_lines.append("\n".join(block))
        prior_text = "\n\n".join(prior_lines) if prior_lines else "(no subtasks recorded)"

        system = (
            "You are Claude, taking over a task that OpenTeddy's local models "
            "(Gemma + Qwen) could not complete. You have full tool access — "
            "shell_exec_readonly/write, read_file/write_file, http_request, "
            "docker_project_detect, docker_diagnose, compose_validate, "
            "env_file_lint, compose_remap_port, port_probe, and any registered "
            "skills.\n\n"
            "Your job:\n"
            "  1. READ the actual project state with tools — don't trust the "
            "     prior attempts' summaries blindly, they may have been wrong.\n"
            "  2. Identify the ROOT CAUSE (e.g. Dockerfile COPY order, missing "
            "     env var, port conflict, bad healthcheck).\n"
            "  3. FIX it directly — write files, run commands, rebuild, verify.\n"
            "  4. When the task is complete and verified, write a concise final "
            "     summary (no tool call). Include what you changed and any URL "
            "     the user can visit.\n\n"
            "Constraints:\n"
            "  - Session workspace is the cwd for all shell commands.\n"
            "  - High-risk tools auto-queue for the user's approval — just call "
            "    them normally, the system handles the gate.\n"
            "  - Do NOT narrate commands the user should run — YOU run them.\n"
            "  - If truly stuck, explain what's blocking in the final summary "
            "    so the user knows what to do."
        )

        user_msg_parts = [
            f"Task goal:\n{task.get('goal', '(unknown)')}",
            "",
            f"Prior subtask attempts by local models ({len(subtasks)}):",
            prior_text,
        ]
        if user_hint:
            user_msg_parts += ["", f"User hint: {user_hint}"]
        user_msg_parts += [
            "",
            "Please investigate, fix the root cause, and complete the task. "
            "Use tools as needed. When done, reply with a final summary "
            "(no tool calls).",
        ]

        # Build a fresh provider-managed history. The provider stores
        # messages in its native shape; we only mutate it through the
        # three helper methods (add_user_message / add_assistant_turn /
        # add_tool_results), so this loop is provider-agnostic.
        history = self.provider.new_history()
        history.add_user_message("\n".join(user_msg_parts))

        tools_used: List[str] = []
        total_tokens_in = 0
        total_tokens_out = 0

        logger.info(
            "%s whole-task takeover starting for task=%s (%d prior subtasks)",
            self.provider.provider_name, task_id, len(subtasks),
        )

        block_msg = self._escalation_blocked()
        if block_msg:
            return {
                "success":    False,
                "summary":    block_msg,
                "tools_used": [],
                "turns":      0,
            }

        for turn in range(max_turns):
            try:
                response = await self.provider.complete_with_tools(
                    history=history,
                    tools=registry.get_schemas(),
                    system=system,
                    max_tokens=4096,
                )
            except LLMProviderError as exc:
                logger.error(
                    "%s API error on turn %d: %s",
                    self.provider.provider_name, turn, exc,
                )
                return {
                    "success":   False,
                    "summary":   f"{self.provider.provider_name} API error: {exc}",
                    "tools_used": tools_used,
                    "turns":     turn + 1,
                }

            # Record usage so the Usage tab reflects the opt-in cost.
            total_tokens_in  += response.usage.input_tokens
            total_tokens_out += response.usage.output_tokens
            try:
                await self.tracker.record_usage(
                    task_id=task_id,
                    model=self.provider.model_name,
                    model_provider=self.provider.provider_name,
                    tokens_in=response.usage.input_tokens,
                    tokens_out=response.usage.output_tokens,
                    task_description=f"[claude-fix turn {turn}] {task.get('goal', '')[:200]}",
                )
            except Exception:  # noqa: BLE001
                pass

            if not response.tool_uses:
                # Final answer — model is done.
                final_text = response.text
                logger.info(
                    "%s finished in %d turn(s), %d tokens in / %d out, "
                    "tools used: %s",
                    self.provider.provider_name,
                    turn + 1, total_tokens_in, total_tokens_out, tools_used,
                )
                return {
                    "success":   True,
                    "summary":   final_text or f"({self.provider.provider_name} returned no text)",
                    "tools_used": tools_used,
                    "turns":     turn + 1,
                    "tokens_in": total_tokens_in,
                    "tokens_out": total_tokens_out,
                }

            # Echo the assistant turn back into history so the next call
            # has full context.
            history.add_assistant_turn(response)

            # Execute every tool call in this turn.
            tool_results: List[LLMToolResult] = []
            for idx, tu in enumerate(response.tool_uses):
                tools_used.append(tu.name)

                logger.info(
                    "%s turn=%d idx=%d tool=%s args=%s",
                    self.provider.provider_name, turn, idx, tu.name, tu.input,
                )

                if ws_callback:
                    try:
                        await ws_callback({
                            "event":    "tool_call",
                            "round":    100 + turn,  # offset so UI keys don't clash with Qwen rounds
                            "call_idx": idx,
                            "tool":     tu.name,
                            "args":     tu.input,
                            "task_id":  task_id,
                            "source":   "claude-fix",
                        })
                    except Exception:  # noqa: BLE001
                        pass

                try:
                    result = await registry.execute(
                        tu.name, tu.input, task_id=task_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    result = {"success": False, "error": f"Tool execution crashed: {exc}"}

                # Truncate tool output so a huge stdout doesn't bloat the
                # next request's context window.
                output_text: str
                if isinstance(result.get("result"), dict):
                    output_text = (
                        result["result"].get("stdout")
                        or result["result"].get("stderr")
                        or str(result["result"])
                    )
                else:
                    output_text = str(result.get("result") or result.get("error", ""))
                output_text = output_text[:4000]

                if ws_callback:
                    try:
                        await ws_callback({
                            "event":    "tool_result",
                            "round":    100 + turn,
                            "call_idx": idx,
                            "tool":     tu.name,
                            "success":  bool(result.get("success")),
                            "output":   output_text[:500],
                            "error":    result.get("error") if not result.get("success") else "",
                            "task_id":  task_id,
                            "source":   "claude-fix",
                        })
                    except Exception:  # noqa: BLE001
                        pass

                tool_results.append(LLMToolResult(
                    tool_use_id=tu.id,
                    content=output_text,
                    is_error=not result.get("success"),
                ))

            history.add_tool_results(tool_results)

        # max_turns exhausted
        logger.warning(
            "%s whole-task takeover hit max_turns=%d for task %s",
            self.provider.provider_name, max_turns, task_id,
        )
        return {
            "success":   False,
            "summary": (
                f"{self.provider.provider_name} ran {max_turns} turns without "
                "converging on a final answer. The task may need manual "
                "attention, a bigger model, or a user hint passed via the "
                "'hint' parameter."
            ),
            "tools_used": tools_used,
            "turns":     max_turns,
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
        }

    async def synthesize_summary(
        self,
        goal: str,
        subtask_results: list[str],
        task_id: str = "",
    ) -> str:
        """
        Ask the cloud LLM to write a final coherent summary from all
        subtask outputs. Records token usage to usage_records.
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
        # Summary synthesis is best-effort — without a key (or with
        # escalation disabled) we just glue the sub-results together
        # rather than crashing the whole task.
        block_msg = self._escalation_blocked()
        if block_msg:
            return f"Summary:\n{numbered}\n\n({block_msg})"

        try:
            response = await self.provider.complete_text(
                user_message=prompt,
                max_tokens=1024,
            )

            # ── Record usage ─────────────────────────────────────────────────
            await self.tracker.record_usage(
                task_id=task_id,
                model=self.provider.model_name,
                model_provider=self.provider.provider_name,
                tokens_in=response.usage.input_tokens,
                tokens_out=response.usage.output_tokens,
                task_description=f"[summary] {goal[:250]}",
            )

            return response.text

        except LLMProviderError as exc:
            logger.error("Summary synthesis failed: %s", exc)
            return f"(Summary unavailable: {exc})\n\nRaw results:\n{numbered}"
