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

import json
import logging
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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

# Rides on the last tool result of a takeover that ran out of turns.
_WRAP_UP_NOTE = (
    "\n\n[OpenTeddy] That was your last tool turn. Do NOT call any more tools. "
    "Write the final answer for the person now, in the language they wrote in: "
    "give every concrete result you already found (numbers, prices, names, links, "
    "tables), then say plainly which parts are not done. Never say something "
    "doesn't exist just because you didn't finish looking."
)

_SYSTEM_PROMPT = """\
You are Teddy-Prime, an expert AI assistant acting as the escalation layer in \
OpenTeddy — a self-growing multi-agent system.
A junior agent (Qwen) attempted the task below but lacked confidence.
Your job: complete the task accurately and completely.
Be concise but thorough.  If you need to write code, write it.
"""

# A cloud model taking over one step gets the task's working folder: files,
# code, data and the web. Without tools it could only write code for nobody
# to run and make up the output: in the 2026-11-14 eval, 7 of qwen3.6:35b's
# 13 misses were such answers, accepted as done ("invoices.json written",
# "revenue 1,234,567.89"). The person's own browser, Google account, home
# devices, printer and messages stay with the local model.
_CLOUD_STEP_TOOLS = frozenset({
    "list_directory", "read_file", "write_file", "edit_file",
    "python_exec", "shell_exec_readonly", "shell_exec_write",
    "csv_describe", "doc_to_markdown", "pdf_extract_text",
    "db_list_connections", "db_list_tables", "db_describe_table",
    "db_query", "db_query_to_csv", "render_chart_report",
    "web_search", "http_get", "read_guide", "pip_list",
    "docker_project_detect", "docker_diagnose", "compose_validate", "env_file_lint",
})
_CLOUD_STEP_TURNS = 10

_STEP_SYSTEM = """\
You are finishing one step of a person's task for OpenTeddy; the local model \
couldn't. You have tools, and the task's working folder is the current directory.

Do the step yourself, with the tools:
- A file the step asks for exists only once you write it with write_file or \
edit_file. Showing its content in your reply creates nothing.
- Run code with python_exec; read or query data with a tool. Never hand over \
code for someone else to run.
- Every number, name and result in your answer must come from a tool result \
in this step. Never invent output.
- If something the step needs doesn't exist (a file, a column, a table), \
look with list_directory or read_file, then say plainly what is missing. \
Never make up a stand-in.
- Check your work before you finish: run the code, read the file back.

When the step is done, reply without a tool call: a short answer for the \
person, in the language they wrote in, with the concrete results.
"""

_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
_ANY_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str, inside_words: bool = False) -> Set[str]:
    """Numbers of two digits or more, as written without separators
    ("73,440" and "73440.0" are both "73440"). An answer's numbers are
    the free-standing ones; a source's include those inside names
    (report_2025.csv backs "the 2025 report")."""
    out: Set[str] = set()
    for m in (_ANY_NUMBER if inside_words else _NUMBER).finditer(text or ""):
        s = m.group().replace(",", "")
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        if sum(ch.isdigit() for ch in s) >= 2:
            out.add(s)
    return out


def ungrounded_numbers(answer: str, sources: List[str]) -> List[str]:
    """Numbers in the answer that appear in none of the sources (the
    request, the step and what the tools returned)."""
    return sorted(_numbers(answer) - _numbers("\n".join(sources), inside_words=True))


async def _emit(emit: Optional[Callable], event: Dict[str, Any]) -> None:
    """Best effort: a UI hiccup never stops the work."""
    if emit:
        try:
            await emit(event)
        except Exception:  # noqa: BLE001
            pass


def _tool_output_text(result: Dict[str, Any]) -> str:
    if isinstance(result.get("result"), dict):
        inner = result["result"]
        return str(inner.get("stdout") or inner.get("stderr") or inner)
    return str(result.get("result") or result.get("error", ""))


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
        registry: Any = None,
        push_event: Optional[Callable] = None,
    ) -> None:
        self.tracker = tracker
        # Provider is injectable for tests; production code uses the
        # default. Stored on the instance so a settings change that
        # swaps providers can be picked up by reassigning this field
        # without re-instantiating the agent.
        self.provider = provider or get_default_provider()
        # With a tool registry, a step taken over in code / analytic mode
        # is done with tools (_resolve_with_tools); without one, the cloud
        # model can only answer in text. push_event is the executor's, so
        # the cloud's tool calls land in the same process record.
        self.registry = registry
        self.push_event = push_event

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
        # Every cloud call here passes this gate (a step, a whole task, a
        # summary). A local-only session or a private task never reaches a
        # cloud model — the summary of an empty run did, before.
        from config import is_session_local_only
        import privacy
        if is_session_local_only() or privacy.is_private(privacy.current_task()):
            return "This task stays on this machine (local only), so no cloud model is used."
        if not self.provider.is_configured():
            return self.provider.get_unconfigured_message()
        return None

    async def resolve(self, subtask: SubTask, context: Dict, mode: str = "code") -> SubTask:
        """
        Use the cloud LLM to resolve a failed/low-confidence subtask.
        Updates and returns the subtask.
        Records token usage to usage_records.

        In code / analytic mode, with a registry, the cloud model does the
        step with workspace tools; otherwise it answers in text.
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

        tools = self._step_tools(context) if mode in ("code", "analytic") else []
        if tools and not self._escalation_blocked():
            return await self._resolve_with_tools(subtask, context, tools)

        system_prompt = _SYSTEM_PROMPT_TIMEOUT if is_timeout else _SYSTEM_PROMPT

        prior_attempt = subtask.result or "(none)"
        timeout_note  = (
            f"\n\nNote: The junior agent timed out — it never returned a result. "
            f"Error: {error_text}"
        ) if is_timeout else ""

        # Guides (agent_skills) tell an agent with tools to look, edit and
        # check; this model has none, and wrote essays in the guide's shape
        # with invented code instead of the answer (2026-11-08 eval).
        _ctx = {k: v for k, v in (context or {}).items() if k not in ("skill_guides", "skills")}
        user_message = (
            f"Task description: {subtask.description}\n\n"
            f"Context: {str(_ctx)[:3000]}\n\n"
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

    # ── One step, with tools ────────────────────────────────────────────

    def _step_tools(self, context: Optional[Dict]) -> List[str]:
        """The workspace tools the registry has, within the agent's tool
        scope when it declares one. Empty = answer in text."""
        if self.registry is None:
            return []
        try:
            have = {t["name"] for t in self.registry.list_tools()}
        except Exception:  # noqa: BLE001
            return []
        names = set(_CLOUD_STEP_TOOLS) & have
        scope = [str(x) for x in ((context or {}).get("allowed_tools") or [])]
        if scope:
            names &= set(scope)
        return sorted(names)

    async def _record_turn(self, task_id: str, response: Any, label: str) -> None:
        try:
            await self.tracker.record_usage(
                task_id=task_id,
                model=self.provider.model_name,
                model_provider=self.provider.provider_name,
                tokens_in=response.usage.input_tokens,
                tokens_out=response.usage.output_tokens,
                task_description=label[:300],
            )
        except Exception:  # noqa: BLE001
            pass

    async def _run_tool_uses(
        self,
        response: Any,
        registry: Any,
        task_id: str,
        *,
        round_no: int,
        emit: Optional[Callable] = None,
        allowed_tools: Optional[List[str]] = None,
        offered: Optional[List[str]] = None,
        tag: Optional[Dict[str, Any]] = None,
        used: Optional[List[str]] = None,
    ) -> Tuple[List[LLMToolResult], List[str]]:
        """Run every tool call of one model turn. Returns the results for
        the model and their texts, cut to 4,000 characters as the model
        sees them.

        allowed_tools is the agent's declared scope, passed on to the
        registry as the executor does: the registry reads a scope as leave
        to auto-approve unattended runs, so the cloud's own list must never
        be passed as one. That list (offered) is held here instead."""
        tag = tag or {}
        results: List[LLMToolResult] = []
        outputs: List[str] = []
        for idx, tu in enumerate(response.tool_uses):
            if used is not None:
                used.append(tu.name)
            logger.info(
                "%s round=%d idx=%d tool=%s args=%s",
                self.provider.provider_name, round_no, idx, tu.name, tu.input,
            )
            await _emit(emit, {
                "event": "tool_call", "round": round_no, "call_idx": idx,
                "tool": tu.name, "args": tu.input, "task_id": task_id, **tag,
            })
            if offered is not None and tu.name not in offered:
                result = {"success": False, "error": (
                    f"'{tu.name}' isn't available here. Use one of: {', '.join(offered)}.")}
            else:
                try:
                    result = await registry.execute(
                        tu.name, tu.input, task_id=task_id, allowed_tools=allowed_tools,
                    )
                except Exception as exc:  # noqa: BLE001
                    result = {"success": False, "error": f"Tool execution crashed: {exc}"}
            # Cut so a huge stdout doesn't bloat the next request.
            text = _tool_output_text(result)[:4000]
            await _emit(emit, {
                "event": "tool_result", "round": round_no, "call_idx": idx,
                "tool": tu.name, "success": bool(result.get("success")),
                "output": text[:500],
                "error": result.get("error") if not result.get("success") else "",
                "task_id": task_id, **tag,
            })
            results.append(LLMToolResult(
                tool_use_id=tu.id, content=text, is_error=not result.get("success"),
            ))
            outputs.append(text)
        return results, outputs

    async def _resolve_with_tools(
        self, subtask: SubTask, context: Dict, tools: List[str],
    ) -> SubTask:
        """The cloud model does the step in the working folder, with tools,
        and its answer has to rest on what the tools returned."""
        task_id = subtask.parent_task_id
        goal = ""
        try:
            goal = ((await self.tracker.get_task(task_id)) or {}).get("goal") or ""
        except Exception:  # noqa: BLE001
            pass
        scope = [str(x) for x in ((context or {}).get("allowed_tools") or [])] or None
        ctx = {k: v for k, v in (context or {}).items()
               if k not in ("skill_guides", "skills", "allowed_tools")}
        ctx_text = json.dumps(ctx, ensure_ascii=False, default=str)[:3000] if ctx else ""
        guides = str((context or {}).get("skill_guides") or "")[:3000]

        parts = []
        if goal:
            parts.append(f"The person's request, in their own words:\n{goal}")
        parts.append(f"This step:\n{subtask.description}")
        if ctx_text:
            parts.append(f"Context (including what earlier steps found):\n{ctx_text}")
        if (subtask.result or "").strip():
            parts.append(f"The local model's attempt, not verified:\n{subtask.result.strip()[:1500]}")
        if (subtask.error or "").strip():
            parts.append(f"Why that didn't count as done:\n{subtask.error.strip()[:600]}")
        system = _STEP_SYSTEM + (f"\nGuides for this kind of task:\n{guides}\n" if guides else "")

        history = self.provider.new_history()
        history.add_user_message("\n\n".join(parts))
        schemas = self.registry.get_schemas_by_names(tools)
        # What an answer's numbers may come from: the request, the step,
        # earlier steps, today's date, the names it gave files, and (below)
        # whatever the tools returned.
        sources = [goal, subtask.description or "", ctx_text,
                   datetime.now().strftime("%Y-%m-%d %H:%M")]
        tag = {"subtask_id": subtask.id, "source": "cloud-step"}
        label = f"[cloud-step] {(subtask.description or '')[:250]}"
        used: List[str] = []
        notes: List[str] = []
        final = ""
        finished = False
        nudged = False
        turn = 0
        logger.info("%s takes over step %s with %d tools", self.provider.provider_name, subtask.id, len(tools))

        while True:
            last = turn >= _CLOUD_STEP_TURNS
            try:
                response = await self.provider.complete_with_tools(
                    history=history, tools=schemas, system=system, max_tokens=4096,
                )
            except LLMProviderError as exc:
                logger.error("%s step takeover failed: %s", self.provider.provider_name, exc)
                subtask.error = f"Escalation error: {exc}"
                subtask.status = TaskStatus.FAILED
                subtask.completed_at = datetime.utcnow()
                await self.tracker.update_subtask(subtask)
                return subtask
            await self._record_turn(task_id, response, label)
            text = (response.text or "").strip()

            if not response.tool_uses:
                missing = ungrounded_numbers(text, sources)
                if missing and not nudged and not last:
                    # A number no tool returned was worked out in its head or
                    # made up — the "revenue 1,234,567.89" answers.
                    nudged = True
                    history.add_assistant_turn(response)
                    history.add_user_message(
                        "Check before you finish: these numbers in your answer appear in no "
                        f"tool result: {', '.join(missing[:12])}. Get them with a tool "
                        "(python_exec, db_query, read_file) and answer again, or leave out "
                        "any you can't back up.")
                    turn += 1
                    continue
                final, finished = text, True
                break
            if last:
                if text:
                    notes.append(text)
                break

            history.add_assistant_turn(response)
            if text:
                notes.append(text)
                await _emit(self.push_event, {"type": "process.note", "task_id": task_id,
                                              "text": text, **tag})
            results, outputs = await self._run_tool_uses(
                response, self.registry, task_id, round_no=200 + turn,
                emit=self.push_event, allowed_tools=scope, offered=tools, tag=tag, used=used,
            )
            sources.extend(outputs)
            sources.extend(str(v) for tu in response.tool_uses for k, v in (tu.input or {}).items()
                           if any(w in k.lower() for w in ("path", "name", "url")))
            if turn == _CLOUD_STEP_TURNS - 1 and results:
                results[-1].content = (results[-1].content or "") + _WRAP_UP_NOTE
            history.add_tool_results(results)
            turn += 1

        if not final:
            final = max(notes, key=len) if notes else ""
        missing = ungrounded_numbers(final, sources) if final else []
        if missing:
            final += ("\n\n（這些數字在這一步的工具結果裡找不到，沒有驗證過："
                      + "、".join(missing[:12]) + "）")
        if final:
            subtask.result = final
        if finished:
            subtask.status = TaskStatus.COMPLETED
            subtask.confidence = 0.6 if missing else 1.0
            subtask.error = None
        else:
            subtask.status = TaskStatus.FAILED
            subtask.error = f"雲端模型接手這一步，{_CLOUD_STEP_TURNS} 輪內還沒做完。"
        logger.info("%s step %s: %s after %d turn(s), tools %s%s",
                    self.provider.provider_name, subtask.id,
                    "done" if finished else "unfinished", turn + 1, used,
                    f", unverified numbers {missing[:5]}" if missing else "")
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
        notes: List[str] = []
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
            # What it says between tool calls goes into the task's process
            # record ("How it got there"), and is the fallback answer.
            if (response.text or "").strip():
                notes.append(response.text.strip())
                if ws_callback:
                    try:
                        await ws_callback({"type": "process.note", "task_id": task_id,
                                           "text": response.text.strip(), "source": "claude-fix"})
                    except Exception:  # noqa: BLE001
                        pass

            # Execute every tool call in this turn. Rounds are offset so UI
            # keys don't clash with the local model's.
            tool_results, _ = await self._run_tool_uses(
                response, registry, task_id, round_no=100 + turn,
                emit=ws_callback, tag={"source": "claude-fix"}, used=tools_used,
            )

            if turn == max_turns - 1 and tool_results:
                # Out of turns: the next call is the last. Without this the
                # takeover ended on "ran 15 turns without converging" and
                # dropped the prices it had already found (2026-11-11).
                last = tool_results[-1]
                last.content = (last.content or "") + _WRAP_UP_NOTE
            history.add_tool_results(tool_results)

        # max_turns exhausted — one more call to answer with what it found.
        final_text = ""
        try:
            response = await self.provider.complete_with_tools(
                history=history, tools=registry.get_schemas(), system=system, max_tokens=4096,
            )
            total_tokens_in += response.usage.input_tokens
            total_tokens_out += response.usage.output_tokens
            try:
                await self.tracker.record_usage(
                    task_id=task_id, model=self.provider.model_name,
                    model_provider=self.provider.provider_name,
                    tokens_in=response.usage.input_tokens, tokens_out=response.usage.output_tokens,
                    task_description=f"[claude-fix wrap-up] {task.get('goal', '')[:200]}",
                )
            except Exception:  # noqa: BLE001
                pass
            final_text = (response.text or "").strip()
        except LLMProviderError as exc:
            logger.warning("%s wrap-up call failed: %s", self.provider.provider_name, exc)
        if not final_text and notes:
            final_text = max(notes, key=len)      # the fullest thing it wrote on the way
        if final_text:
            logger.warning(
                "%s whole-task takeover hit max_turns=%d for task %s — partial answer kept",
                self.provider.provider_name, max_turns, task_id,
            )
            return {
                "success":    False,
                "partial":    True,
                "summary":    final_text,
                "tools_used": tools_used,
                "turns":      max_turns + 1,
                "tokens_in":  total_tokens_in,
                "tokens_out": total_tokens_out,
            }

        # Nothing to show at all
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
