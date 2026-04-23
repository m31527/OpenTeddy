"""
OpenTeddy Executor
Qwen-powered agent that executes individual SubTasks.
Supports Ollama Function Calling for tool use (shell, file, GCP, DB, HTTP).
Falls back to direct LLM inference if no tools are needed.
Reports confidence so the Orchestrator can decide on escalation.
"""

from __future__ import annotations

import asyncio
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

# Objective failure markers in tool results — override self-reported confidence
# when present, so Claude takes over instead of a false "completed".
# New additions cover compose/YAML failures that previously slipped through
# ("yaml: unmarshal errors" on compose up would leave Qwen cheerfully
# reporting 65% confidence while nothing had actually started).
_FAILURE_SIGNAL_RE = re.compile(
    r"\b(?:unhealthy|Restarting|Exited \(\d+\)|Dead|CrashLoopBackOff|"
    r"Error response from daemon|ERROR \d{4}(?:\s*\(\d+\))?|"
    r"WARNING: no compose file found|"
    # docker-compose YAML / parse failures
    r"yaml: unmarshal errors?|yaml: line \d+|"
    r"failed to (?:parse|load|read) compose|"
    r"services\.\w+\.(?:ports|volumes|environment).*?(?:is invalid|must be)|"
    r"mapping values are not allowed in this context|"
    r"required variable .* is not set|"
    # Image / build failures
    r"failed to solve with frontend|"
    r"pull access denied|manifest unknown|"
    r"(?:image|manifest) not found|"
    # Network failures that block startup
    r"bind:? address already in use|"
    r"port is already allocated|"
    # Filesystem / path failures — Qwen has been mis-reading these as
    # "nothing interesting happened" and returning 100% confidence on
    # what were actually total dead-ends (e.g. `docker compose -f
    # /nonexistent/path.yml` → "open X: no such file or directory").
    r"no such file or directory|"
    r"open [^:\n]+: no such file|"
    r"fatal: (?:not a git repository|could not read)|"
    r"cannot list |cannot access |"
    r"command not found|"
    # cd failures — the exact shape `sh` emits: "/bin/sh: N: cd: can't cd to X"
    r"cd: can'?t cd to |cd: no such file|"
    # Empty-but-success compose output flagged by shell_tool
    r"zero containers \(likely wrong cwd\))",
    re.IGNORECASE,
)
_FAILURE_CLAMP_CONFIDENCE = 0.3
_OUTPUT_PREVIEW_CHARS = 500

# Refusal patterns — Qwen/Gemma sometimes hallucinate "I'm just a language
# model, I can't run shell commands" even though the tools are exposed via
# function calling. When we see that pattern in the FINAL answer (not in a
# tool result), we clamp confidence so Claude escalation takes over instead
# of marking a useless "I can't help you" response as completed.
_REFUSAL_SIGNAL_RE = re.compile(
    r"(?:"
    r"我是一個 ?AI|我是.{0,6}語言模型|我沒有.{0,8}(?:能力|權限).{0,6}執行|"
    r"無法.{0,10}(?:直接)?執行|請(?:您|使用者).{0,6}自行執行|"
    r"I (?:am|'m) (?:just |only )?(?:an? )?(?:AI|language model)|"
    r"I (?:cannot|can't|don'?t have (?:the )?ability to) (?:execute|run|access)|"
    r"please (?:run|execute) (?:the )?(?:following )?command(?:s)? (?:yourself|on your own)"
    r")",
    re.IGNORECASE,
)


def _preview_tool_output(tool_result: Dict[str, Any]) -> str:
    """Produce a short string preview of a tool result for the Web UI.

    Falls back through stdout → stderr → stringified result → error so the
    collapsed tool card always shows something meaningful (not just "done").
    """
    if not tool_result:
        return ""
    result = tool_result.get("result")
    if isinstance(result, dict):
        text = (
            result.get("stdout")
            or result.get("stderr")
            or result.get("content")
            or ""
        )
        if not text:
            # Last resort: serialise the dict
            text = json.dumps(result, ensure_ascii=False)
    elif isinstance(result, str):
        text = result
    elif result is not None:
        text = str(result)
    else:
        text = tool_result.get("error") or ""
    text = (text or "").strip()
    return text[:_OUTPUT_PREVIEW_CHARS]

# ── Mode-specific executor prompts ────────────────────────────────────────────
# The orchestrator tells us which mode the user picked. Each mode gets its
# own system prompt + tool exposure so we don't have to squeeze every
# possible behavior into one general prompt.

_SYSTEM_PROMPT_CODE = """\
You are Teddy-Exec, a precise task executor running inside the OpenTeddy
multi-agent system. You are NOT a chatbot and you are NOT "just a language
model" — you are an executor with real tool access on a real machine.

AVAILABLE TOOLS (you MUST call these via function calling, not describe them):
  • shell_exec_readonly — run read-only shell commands: ls, cat, grep, find,
    pwd, env, docker ps, docker logs, git status, etc.
  • shell_exec_write   — run shell commands that change state: git clone,
    pip install, npm install, docker compose up/build/down, mkdir, mv, cp,
    file writes, service restarts. (High-risk; the system gates approval
    automatically — just call the tool.)
  • file_read / file_write — read or write a file.
  • http_request — make HTTP calls.

  DEPLOYMENT HELPERS (prefer these over raw shell for deploy workflows):
  • port_probe(port)            — is this port in use? returns PID/process,
                                  plus is_self, is_important, safe_to_kill_hint,
                                  and a recommendation string. ALWAYS READ
                                  safe_to_kill_hint before deciding what to do.
  • docker_project_detect(dir)  — scan for Dockerfile/compose; returns
                                  services, ports, and a suggested command.
                                  Call this FIRST for any deploy task.
  • compose_validate(dir)       — PRE-FLIGHT: `docker compose config --quiet`.
                                  Always run before `up`. Cheap ~1s check
                                  that catches YAML/env-substitution bugs
                                  which would otherwise surface 30s into a
                                  build as "yaml: unmarshal errors".
  • env_file_lint(path)         — scan .env for multi-line values, duplicate
                                  keys, unterminated quotes. Run when
                                  compose_validate reports a YAML unmarshal
                                  error — the culprit is almost always .env.
  • docker_diagnose(target)     — bundled inspect + logs + port, with a
                                  heuristic hint (OOM, port conflict, etc.).
                                  Use whenever a container is unhealthy
                                  (NOT when compose fails before creating
                                  containers — use compose_validate for that).
  • compose_remap_port(file, service, from, to) — edit a compose file to
                                  rebind a host port. THE PREFERRED fix
                                  when port_probe says safe_to_kill_hint=False.
  • port_free(port)             — HIGH RISK. Kill whatever holds a port.
                                  Only when safe_to_kill_hint=True.

  PORT-CONFLICT DECISION TREE (memorize this):
    port_probe says in_use=True →
      • safe_to_kill_hint=False (is_self OR is_important)
          → call compose_remap_port to move THE CONTAINER, do not touch host
      • safe_to_kill_hint=True (regular user process)
          → port_free is OK (will prompt user), or compose_remap_port to be safe
      • in_use=False
          → proceed with docker compose up

  • (Plus any task-specific skills the system has created.)

ABSOLUTE RULES — violating these is a task failure:

1. NEVER refuse with "I'm just a language model", "I can't run shell commands",
   "please run these commands yourself", or any equivalent. You CAN run them.
   If the sub-task asks you to deploy, install, or clone something, CALL
   shell_exec_write with the actual command. Do not list commands as text.

2. If a tool result contains "unhealthy", "Exited", "Restarting",
   "Error response", "ERROR <code>", or similar failure signals, you MUST:
     (a) Investigate with follow-up tool calls (docker logs --tail=100 <name>,
         docker inspect <name>, cat <file>).
     (b) NOT report completion — set confidence < 0.5 and describe the failure
         in "result". Let Claude escalation take over.

3. Always prefer calling a tool over narrating what you would do. If the
   sub-task is "執行 git clone https://github.com/foo/bar", you MUST call
   shell_exec_write with command="git clone https://github.com/foo/bar".

4. Only emit the final JSON (below) AFTER all tool work is done, or when the
   task is genuinely a pure-reasoning question that needs no tools.

FINAL OUTPUT FORMAT (emit exactly one JSON object, no prose, no markdown):
  {
    "result": "<string: your answer / action result>",
    "confidence": <float 0.0–1.0>,
    "skill_needed": "<string snake_case or null>",
    "skill_description": "<string or null>"
  }
"""


_SYSTEM_PROMPT_CHAT = """\
You are Teddy-Exec in **Chat mode**. The user wants pure text reasoning —
summarize, translate, explain, answer, write. No tools are available in
this mode and none are needed.

Just read the sub-task, think, and reply with the answer. Write in the same
language the user used. Format with markdown (headings, lists, bold) when it
improves readability — especially for summaries and structured explanations.

DO NOT mention tools, shell commands, files, or "I would need to…". Just
answer the question or produce the requested text.

FINAL OUTPUT FORMAT (emit exactly one JSON object, no prose outside it):
  {
    "result": "<string: your markdown-formatted answer>",
    "confidence": <float 0.0–1.0>,
    "skill_needed": null,
    "skill_description": null
  }
"""


# Back-compat alias — existing code might import _SYSTEM_PROMPT.
_SYSTEM_PROMPT = _SYSTEM_PROMPT_CODE


def _system_prompt_for_mode(mode: str) -> str:
    if mode == "chat": return _SYSTEM_PROMPT_CHAT
    # analytic currently uses the Code prompt (beta); see orchestrator comment.
    return _SYSTEM_PROMPT_CODE


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

    async def execute(
        self, subtask: SubTask, context: Dict, mode: str = "code",
    ) -> SubTask:
        """Execute a subtask. Returns the updated subtask with result / status.

        `mode` comes from the session (chat / code / analytic) and flips
        both the system prompt and the available tool set below.
        """
        subtask.status = TaskStatus.RUNNING
        await self.tracker.update_subtask(subtask)

        # 1. Try a matching skill first — skills are available in all modes
        # except pure chat (where tools make no sense).
        if mode != "chat":
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
            subtask.description, context,
            task_id=subtask.parent_task_id,
            subtask_id=subtask.id,
            mode=mode,
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
        subtask_id: str = "",
        mode: str = "code",
    ) -> Tuple[str, float, Optional[str], Optional[str]]:
        """
        Multi-turn Qwen chat with Ollama function calling.
        Loop up to _MAX_TOOL_ROUNDS tool calls before forcing a final answer.

        `mode` picks the system prompt and gates whether tools are exposed
        at all — in chat mode we send no `tools` field so the model is
        forced into a pure single-turn answer (no accidental shell calls).
        """
        system_prompt = _system_prompt_for_mode(mode)

        messages: List[Dict[str, Any]] = [
            {
                "role": "user",
                "content": (
                    f"Task: {description}\n\n"
                    f"Context: {json.dumps(context, ensure_ascii=False)[:2000]}"
                ),
            }
        ]

        # Chat mode deliberately hides tools from the model so it can't
        # accidentally write files / run shell commands while summarizing.
        tools = [] if mode == "chat" else self.registry.get_schemas()
        objective_failure_seen = False

        for round_idx in range(_MAX_TOOL_ROUNDS):
            payload: Dict[str, Any] = {
                "model": config.qwen_model,
                "messages": messages,
                "system": system_prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": config.qwen_max_tokens},
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

            # ── Record Qwen usage ─────────────────────────────────────────────
            _tokens_in  = data.get("prompt_eval_count", 0) or 0
            _tokens_out = data.get("eval_count", 0) or 0
            try:
                await self.tracker.record_usage(
                    task_id=task_id,
                    model=config.qwen_model,
                    model_provider="ollama",
                    tokens_in=_tokens_in,
                    tokens_out=_tokens_out,
                    task_description=description,
                    cost_usd=0.0,
                )
            except Exception:  # noqa: BLE001
                pass

            message = data.get("message", {})
            tool_calls: List[Dict[str, Any]] = message.get("tool_calls") or []

            # ── No tool calls → final answer ──────────────────────────────────
            if not tool_calls:
                raw_text = message.get("content", "")
                return self._finalize_response(raw_text, objective_failure_seen)

            # ── Process tool calls ────────────────────────────────────────────
            # Add assistant's message (with tool_calls) to history
            messages.append({
                "role": "assistant",
                "content": message.get("content", ""),
                "tool_calls": tool_calls,
            })

            # Execute each tool and collect results
            for call_idx, call in enumerate(tool_calls):
                fn_info = call.get("function", {})
                tool_name: str = fn_info.get("name", "")
                raw_args = fn_info.get("arguments", {})
                args: Dict[str, Any] = (
                    raw_args if isinstance(raw_args, dict)
                    else json.loads(raw_args)
                )

                logger.info(
                    "Tool call round=%d idx=%d tool=%s args=%s task=%s",
                    round_idx, call_idx, tool_name, args, task_id,
                )

                # Push event to Web UI. call_idx disambiguates multiple calls
                # of the same tool within a single round so each gets its own
                # card instead of sharing a DOM id.
                await self._push_event({
                    "event": "tool_call",
                    "round": round_idx,
                    "call_idx": call_idx,
                    "tool": tool_name,
                    "args": args,
                    "task_id": task_id,
                })

                try:
                    tool_result = await self.registry.execute(
                        tool_name, args, task_id=task_id
                    )
                except asyncio.CancelledError:
                    # Orchestrator cancelled us (usually subtask timeout).
                    # Fire-and-forget a cancellation event via create_task
                    # so awaiting it won't re-raise CancelledError, then
                    # re-raise so asyncio.wait_for completes cleanly.
                    try:
                        asyncio.create_task(self._push_event({
                            "event": "tool_result",
                            "round": round_idx,
                            "call_idx": call_idx,
                            "tool": tool_name,
                            "success": False,
                            "error": "Cancelled (subtask timeout / escalated)",
                            "output": "",
                            "task_id": task_id,
                        }))
                    except Exception:  # noqa: BLE001
                        pass
                    raise

                logger.info(
                    "Tool result tool=%s success=%s duration_ms=%s",
                    tool_name,
                    tool_result.get("success"),
                    tool_result.get("duration_ms"),
                )

                # ── Objective failure detection ───────────────────────────
                # Scan tool output for hard failure signals (unhealthy, Exited,
                # MySQL/Docker errors). If any match, we'll clamp confidence
                # at the end so Claude escalation kicks in even if Qwen
                # self-reports high confidence.
                if not objective_failure_seen:
                    tool_result_text = json.dumps(
                        tool_result, ensure_ascii=False
                    )
                    if _FAILURE_SIGNAL_RE.search(tool_result_text):
                        objective_failure_seen = True
                        logger.info(
                            "Objective failure signal detected in tool "
                            "result (tool=%s, task=%s) — will clamp "
                            "confidence to force escalation.",
                            tool_name, task_id,
                        )

                # Build a short, human-readable output preview for the UI so
                # collapsed cards show the real first line instead of "(done)".
                output_preview = _preview_tool_output(tool_result)

                # Push result event to Web UI
                await self._push_event({
                    "event": "tool_result",
                    "round": round_idx,
                    "call_idx": call_idx,
                    "tool": tool_name,
                    "success": tool_result.get("success"),
                    "error": tool_result.get("error"),
                    "output": output_preview,
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
                    "options": {"temperature": 0.2, "num_predict": config.qwen_max_tokens},
                },
            )
            resp.raise_for_status()
            _final_data = resp.json()
            _tokens_in  = _final_data.get("prompt_eval_count", 0) or 0
            _tokens_out = _final_data.get("eval_count", 0) or 0
            try:
                await self.tracker.record_usage(
                    task_id=task_id,
                    model=config.qwen_model,
                    model_provider="ollama",
                    tokens_in=_tokens_in,
                    tokens_out=_tokens_out,
                    task_description=description,
                    cost_usd=0.0,
                )
            except Exception:  # noqa: BLE001
                pass
            raw_text = _final_data.get("message", {}).get("content", "")
        except Exception as exc:  # noqa: BLE001
            return f"Executor error (forced final): {exc}", 0.0, None, None

        return self._finalize_response(raw_text, objective_failure_seen)

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

    @classmethod
    def _finalize_response(
        cls,
        text: str,
        objective_failure_seen: bool,
    ) -> Tuple[str, float, Optional[str], Optional[str]]:
        """Parse Qwen's JSON, then clamp confidence if tools saw hard failures
        or Qwen emitted a "I can't do this" refusal instead of calling tools."""
        result, confidence, skill_hint, skill_desc = cls._parse_qwen_response(text)
        if objective_failure_seen and confidence > _FAILURE_CLAMP_CONFIDENCE:
            logger.info(
                "Clamping self-reported confidence %.2f → %.2f "
                "(objective failure signal in tool output).",
                confidence, _FAILURE_CLAMP_CONFIDENCE,
            )
            confidence = _FAILURE_CLAMP_CONFIDENCE
        # Detect hallucinated refusals ("I'm just a language model, I can't
        # run shell commands") and force escalation — these are never
        # acceptable final answers for an executor that has tool access.
        if _REFUSAL_SIGNAL_RE.search(result) and confidence > _FAILURE_CLAMP_CONFIDENCE:
            logger.warning(
                "Qwen refused to use tools ('I am a language model…'). "
                "Clamping confidence %.2f → %.2f so Claude escalation runs.",
                confidence, _FAILURE_CLAMP_CONFIDENCE,
            )
            confidence = _FAILURE_CLAMP_CONFIDENCE
        return result, confidence, skill_hint, skill_desc

    @staticmethod
    def _parse_qwen_response(
        text: str,
    ) -> Tuple[str, float, Optional[str], Optional[str]]:
        """Extract fields from Qwen's JSON output (tolerant of extra text)."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return text.strip(), 0.65, None, None
        try:
            data = json.loads(match.group())
            result = str(data.get("result", text.strip()))
            confidence = float(data.get("confidence", 0.75))
            confidence = max(0.0, min(1.0, confidence))
            skill_hint = data.get("skill_needed") or None
            skill_desc = data.get("skill_description") or None
            return result, confidence, skill_hint, skill_desc
        except (json.JSONDecodeError, ValueError):
            return text.strip(), 0.65, None, None

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
