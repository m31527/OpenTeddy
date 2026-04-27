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
from model_profile import model_tier
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


def _format_tool_result_for_model(
    tool_name: str, tool_result: Dict[str, Any],
) -> str:
    """Render a tool result as a labeled plain-text block instead of a
    JSON dump.

    Why not json.dumps()?
      - Nested JSON with mid-string truncation markers (`... [truncated]`)
        makes bigger local models think they've received incomplete
        information, which pushes them to re-call the tool "to confirm".
      - Labeled plain text makes exit_code / stdout / stderr / error
        unambiguous at a glance, cutting the re-call reflex.

    Output shape:
        [tool_name] OK (32ms)
        exit_code: 0
        --- stdout ---
        <stdout text>
        --- stderr ---
        <stderr text>
        --- error ---
        <error text, if any>

    Missing sections are simply omitted (e.g. no stdout block on a
    write_file success). Arbitrary string results become a single
    'result:' section.
    """
    success = tool_result.get("success")
    duration = tool_result.get("duration_ms", 0)
    status = "OK" if success else "FAILED"
    lines: List[str] = [f"[{tool_name}] {status} ({duration}ms)"]

    inner = tool_result.get("result")
    if isinstance(inner, dict):
        if "exit_code" in inner:
            lines.append(f"exit_code: {inner['exit_code']}")
        stdout = (inner.get("stdout") or "").rstrip()
        stderr = (inner.get("stderr") or "").rstrip()
        if stdout:
            lines.append("--- stdout ---")
            lines.append(stdout)
        if stderr:
            lines.append("--- stderr ---")
            lines.append(stderr)
        # For non-shell tools (file_read, docker_project_detect) the
        # result dict carries domain-specific fields — surface whatever
        # else is there as a compact JSON line so the model can still
        # see structured data without us hiding it entirely.
        extra = {k: v for k, v in inner.items()
                 if k not in ("stdout", "stderr", "exit_code")}
        if extra:
            try:
                lines.append("--- result fields ---")
                lines.append(json.dumps(extra, ensure_ascii=False, default=str)[:2000])
            except Exception:  # noqa: BLE001
                pass
        if not stdout and not stderr and not extra:
            lines.append("(no output)")
    elif isinstance(inner, str):
        if inner.strip():
            lines.append("--- result ---")
            lines.append(inner)
    elif inner is not None:
        try:
            lines.append("--- result ---")
            lines.append(json.dumps(inner, ensure_ascii=False, default=str)[:2000])
        except Exception:  # noqa: BLE001
            lines.append(str(inner)[:2000])

    err = tool_result.get("error")
    if err:
        lines.append("--- error ---")
        lines.append(str(err))

    # Preserve any dedup notice attached when the inner result wasn't a dict.
    notice = tool_result.get("_dedup_notice")
    if notice:
        lines.append(str(notice).strip())

    return "\n".join(lines)


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

0. NEVER set `working_dir` to an absolute path outside the current session's
   workspace (e.g. NEVER pass working_dir="/home/user/OpenTeddy" — that is
   the agent's OWN source tree and will be HARD-BLOCKED by the shell tool,
   returning an error). Prefer relative subdir names like
   working_dir="worldmonitor" or omit working_dir to use the session default.
   Same rule for `cd` — never `cd` into OpenTeddy's own project directory.

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

4. DO NOT repeat identical tool calls. Before calling a tool, scan the
   recent messages — if you already called it with the same arguments
   within this subtask, DO NOT call it again. Use the previous output
   instead. The system will refuse a 3rd duplicate and force you to
   stop. Repeated identical calls waste GPU and produce no new information.

5. Only emit the final JSON (below) AFTER all tool work is done, or when the
   task is genuinely a pure-reasoning question that needs no tools.

6. Analytic mode — if the subtask is to produce a data-analysis report,
   put charts in fenced ```chart blocks inside the "result" field. The
   frontend renders them as interactive Chart.js v4 figures. Example:

     ```chart
     {
       "type": "bar",
       "data": {
         "labels": ["Jan","Feb","Mar"],
         "datasets": [{"label":"Revenue","data":[100,150,130],"backgroundColor":"#d97757"}]
       },
       "options": {"plugins":{"title":{"display":true,"text":"Q1 Revenue"}}}
     }
     ```

   Pick the chart type to match the data (line=trend, bar=compare, pie=share,
   scatter=correlation, radar=multi-dim). Include 2–5 charts + a short
   markdown summary with headings + bullet findings.

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


_STRICT_PREAMBLE = """\
[STRICT MODE — small model, follow these rules EXACTLY:]
1. To USE a tool: emit ONLY the tool call. No prose around it. No \
"I will call X" preamble.
2. To ANSWER directly: emit ONLY the answer text. No "Final answer:" \
prefix. No bullet-points-of-what-you-thought.
3. NEVER mix tool calls and prose answers in the same response.
4. Keep responses short. Aim for the minimum tokens that solve the task.

Examples:
User: list /tmp
You: (call shell_exec_readonly with command='ls -la /tmp')

User: what is 2 + 2?
You: 4

User: summarise this file: /etc/hosts
You: (call file_read with path='/etc/hosts')   # then on the next turn, after the result is given to you, emit the summary text

[End of strict-mode header. Below is the standard agent guide:]
"""

_OPEN_SUFFIX = """

[Capable model — extra freedom:]
You may take multiple reasoning steps before responding when the task is
genuinely complex. Keep the FINAL response concise; the user only sees the
last message, not your intermediate thinking.
"""


def _system_prompt_for_mode(mode: str, model_name: str = "") -> str:
    """Pick the base prompt for the given mode, then bend its strictness
    to match the model's capability tier.

    This is the core of "Adaptive Prompts" (#1) — small thinking models
    (qwen3.5:2b etc.) get the strict preamble + few-shot examples; large
    models get a brief openness suffix; mid-range stays as-is.
    """
    base = _SYSTEM_PROMPT_CHAT if mode == "chat" else _SYSTEM_PROMPT_CODE
    tier = model_tier(model_name)
    if tier == "strict":
        return _STRICT_PREAMBLE + "\n" + base
    if tier == "open":
        return base + _OPEN_SUFFIX
    return base


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
        # subtask_id → list of {path, relative_path, size_bytes, tool}.
        # Populated by _push_event when an artifact event flows; drained
        # by orchestrator's deliverable verifier (#2). See pop_subtask_artifacts.
        self._subtask_artifacts: Dict[str, List[Dict[str, Any]]] = {}

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
        system_prompt = _system_prompt_for_mode(mode, config.qwen_model)
        # One-line trace so it's obvious in the log which prompt tier the
        # current model is running under (strict / balanced / open).
        logger.info(
            "Executor prompt tier=%s mode=%s model=%s",
            model_tier(config.qwen_model), mode, config.qwen_model,
        )

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

        # Duplicate-call tracker. Small local models (Qwen 2.5 3B) routinely
        # re-call the same tool with the same args because their
        # short-term attention over tool results is weak. Each extra
        # round burns full GPU inference. We:
        #   - Nudge Qwen on the 2nd identical call (warning in the result)
        #   - Force-end the loop on the 3rd (inject a terminate message,
        #     don't execute the tool — just tell Qwen to emit its final
        #     JSON answer instead of burning another round).
        # Key = (tool_name, canonical args JSON). Stable across retries.
        call_counts: Dict[str, int] = {}

        # ── B: per-tool-name hard cap (separate from exact-args dedup) ──
        # The exact-args gate above only catches `read_file({path:'a'})`
        # called 3× with identical args. It can't stop a model from
        # firing 12 different `python_exec` calls at the same dataset
        # — exactly the failure mode that ate 8 hours on a real task.
        # Cap each tool NAME at MAX_PER_TOOL_NAME calls per subtask;
        # past that we synthesize a refusal so the model has to commit.
        tool_name_counts: Dict[str, int] = {}
        MAX_PER_TOOL_NAME = 5

        # ── D: stuck-loop circuit breaker ────────────────────────────
        # If a subtask accumulates this many tool failures in total, we
        # stop letting it spin and force the model to emit a final answer
        # with what it has. Catches "kept retrying python_exec with new
        # syntax errors forever" pattern.
        total_tool_failures = 0
        MAX_TOTAL_FAILURES  = 5

        def _call_key(tname: str, targs: Dict[str, Any]) -> str:
            try:
                return f"{tname}::{json.dumps(targs, sort_keys=True, ensure_ascii=False)}"
            except Exception:  # noqa: BLE001
                return f"{tname}::{targs!r}"

        # Context watchdog state. We track the most recent prompt_eval_count
        # from Ollama so the next round can decide if it needs to compress
        # earlier turns before they bust num_ctx and the model starts
        # truncating (or in extreme cases, simply silently dropping the
        # original task description). Reset to 0 after a successful compress.
        last_prompt_tokens = 0
        num_ctx       = int(getattr(config, "qwen_num_ctx", 16384))
        compress_at_f = float(getattr(config, "context_compress_at", 0.7))
        compress_threshold = max(1024, int(num_ctx * compress_at_f))

        for round_idx in range(_MAX_TOOL_ROUNDS):
            # ── Context watchdog (#4) ──────────────────────────────────────
            # If the previous round's prompt was already >= 70% of num_ctx,
            # we're one turn away from the model losing its grip. Compress
            # mid-conversation messages now while there's still headroom.
            if (
                last_prompt_tokens >= compress_threshold
                and len(messages) > 4
            ):
                logger.info(
                    "Context watchdog: prompt_tokens=%d >= %d (num_ctx=%d) "
                    "— compressing %d earlier messages.",
                    last_prompt_tokens, compress_threshold, num_ctx,
                    len(messages) - 4,
                )
                messages = await self._compress_messages(messages, system_prompt)
                last_prompt_tokens = 0   # next call will re-measure

            stream_on = bool(getattr(config, "streaming_enabled", True))
            payload: Dict[str, Any] = {
                "model": config.qwen_model,
                "messages": messages,
                "system": system_prompt,
                "stream": stream_on,
                "options": {
                    "temperature": float(getattr(config, "qwen_temperature", 0.2)),
                    "num_predict": config.qwen_max_tokens,
                    # Tell Ollama how big a context window we want — this
                    # is the real budget the watchdog is keeping us under.
                    "num_ctx":     num_ctx,
                },
            }
            if tools:
                payload["tools"] = tools

            try:
                if stream_on:
                    # Stream NDJSON chunks from Ollama, push each text
                    # delta onto the WebSocket so the chat bubble fills
                    # in word-by-word. We accumulate the full content +
                    # tool_calls so the rest of the loop can consume the
                    # response identically to the non-streaming path.
                    acc_content   = ""
                    acc_thinking  = ""
                    acc_tool_calls: List[Dict[str, Any]] = []
                    last_chunk: Dict[str, Any] = {}
                    async with self._http.stream(
                        "POST",
                        f"{config.qwen_base_url}/api/chat",
                        json=payload,
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line.strip():
                                continue
                            try:
                                chunk = json.loads(line)
                            except Exception:  # noqa: BLE001
                                continue
                            msg = chunk.get("message") or {}
                            d_content  = msg.get("content")  or ""
                            d_thinking = msg.get("thinking") or ""
                            if d_content:
                                acc_content += d_content
                                await self._push_event({
                                    "type":       "chat.stream.delta",
                                    "task_id":    task_id,
                                    "subtask_id": subtask_id,
                                    "text":       d_content,
                                })
                            if d_thinking:
                                acc_thinking += d_thinking
                            tcs = msg.get("tool_calls") or []
                            if tcs:
                                acc_tool_calls.extend(tcs)
                            if chunk.get("done"):
                                last_chunk = chunk
                    # Re-shape into the same structure non-streaming returned.
                    # Carry through Ollama's timing stats (eval_duration,
                    # total_duration) so #6 can compute t/s from them.
                    data = {
                        "message": {
                            "content":    acc_content,
                            "thinking":   acc_thinking,
                            "tool_calls": acc_tool_calls,
                        },
                        "prompt_eval_count":    last_chunk.get("prompt_eval_count", 0),
                        "eval_count":           last_chunk.get("eval_count", 0),
                        "eval_duration":        last_chunk.get("eval_duration", 0),
                        "total_duration":       last_chunk.get("total_duration", 0),
                        "prompt_eval_duration": last_chunk.get("prompt_eval_duration", 0),
                        "done_reason":          last_chunk.get("done_reason"),
                    }
                    # Mark the stream as complete so the UI can stop the
                    # word-by-word painter and lock in the final markdown.
                    await self._push_event({
                        "type":       "chat.stream.end",
                        "task_id":    task_id,
                        "subtask_id": subtask_id,
                    })
                else:
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
            # Feed the watchdog so the *next* round can decide if it
            # needs to compress before sending another giant prompt.
            last_prompt_tokens = _tokens_in
            # #6 Auto-benchmark: pull Ollama's own timing stats so the
            # Settings page can surface real per-model t/s. eval_duration
            # is in nanoseconds; tokens/sec = eval_count / (eval_dur / 1e9).
            _eval_dur_ns  = data.get("eval_duration", 0) or 0
            _total_dur_ns = data.get("total_duration", 0) or 0
            _tps = (
                (_tokens_out * 1_000_000_000 / _eval_dur_ns)
                if _eval_dur_ns > 0 and _tokens_out > 0 else 0.0
            )
            try:
                await self.tracker.record_usage(
                    task_id=task_id,
                    model=config.qwen_model,
                    model_provider="ollama",
                    tokens_in=_tokens_in,
                    tokens_out=_tokens_out,
                    task_description=description,
                    cost_usd=0.0,
                    duration_ms=int(_total_dur_ns / 1_000_000),
                    tokens_per_sec=_tps,
                )
            except Exception:  # noqa: BLE001
                pass

            message = data.get("message", {})
            tool_calls: List[Dict[str, Any]] = message.get("tool_calls") or []

            # ── No tool calls → final answer ──────────────────────────────────
            if not tool_calls:
                # `thinking` is the reasoning channel exposed by recent
                # qwen3.5 / gemma4 / DeepSeek-style "thinking" models.
                # When the model spends all its tokens reasoning before
                # emitting `content`, content is empty but thinking has
                # the answer. Fall back to thinking so we don't return
                # "" and trip the empty-response → low-confidence loop.
                raw_text = message.get("content") or message.get("thinking") or ""
                return self._finalize_response(raw_text, objective_failure_seen)

            # ── Process tool calls ────────────────────────────────────────────
            # Add assistant's message (with tool_calls) to history
            messages.append({
                "role": "assistant",
                "content": message.get("content", ""),
                "tool_calls": tool_calls,
            })

            # ── #5 Parallel tool calls ──────────────────────────────────
            # When the model emits multiple tool_calls in one round AND
            # every call is low-risk (read-only per registry taxonomy),
            # we launch them concurrently up-front via asyncio.gather.
            # The dedup / event-push / message-append loop below still
            # runs once per call, but each iteration consumes a
            # pre-completed result instead of awaiting registry.execute
            # serially — N reads/HTTP fetches now overlap on the network
            # / disk timeline instead of stacking.
            #
            # We deliberately bail out (and stay sequential) if ANY call
            # is medium/high risk: writes might depend on prior writes,
            # shell commands can have order-sensitive side effects, and
            # high-risk gating runs through the approval queue which
            # mustn't be batched.
            parallel_pre: Dict[int, Dict[str, Any]] = {}
            if len(tool_calls) >= 2 and all(
                self.registry.risk_of((c.get("function") or {}).get("name", "")) == "low"
                for c in tool_calls
            ):
                async def _exec_parallel(i: int, c: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
                    fn = c.get("function") or {}
                    a  = fn.get("arguments") or {}
                    if not isinstance(a, dict):
                        try:
                            a = json.loads(a)
                        except Exception:  # noqa: BLE001
                            a = {}
                    try:
                        r = await self.registry.execute(
                            fn.get("name", ""), a, task_id=task_id,
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:  # noqa: BLE001
                        from tool_registry import make_result as _mr
                        r = _mr(False, error=str(e))
                    return i, r

                logger.info(
                    "Round %d: %d low-risk tool_calls — running concurrently (#5)",
                    round_idx, len(tool_calls),
                )
                pre_results = await asyncio.gather(
                    *(_exec_parallel(i, c) for i, c in enumerate(tool_calls))
                )
                for idx, res in pre_results:
                    parallel_pre[idx] = res

            # Execute each tool and collect results
            force_finalize = False  # set True if we detect a runaway loop
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

                # Dedup gate: count how many times this exact (tool, args)
                # has been invoked in this subtask. On the 3rd attempt we
                # short-circuit — don't execute, don't burn GPU, just
                # synthesize a fail-fast tool_result that tells Qwen
                # "stop looping, emit your final answer".
                k = _call_key(tool_name, args)
                prev_count = call_counts.get(k, 0)
                call_counts[k] = prev_count + 1

                # ── B: per-tool-name hard cap (independent of args) ──
                # Catches "12 different python_exec calls trying random
                # variations" — the exact-args gate above misses that
                # because each call's code differs.
                tn_prev = tool_name_counts.get(tool_name, 0)
                tool_name_counts[tool_name] = tn_prev + 1

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

                if prev_count >= 2:
                    # 3rd+ identical call — refuse to execute, synthesize
                    # a nudge result, and schedule a hard break after this
                    # round. The nudge stays visible in the tool card so
                    # the user can see why the loop stopped.
                    logger.warning(
                        "Qwen loop detected: %s called %d times with same "
                        "args — refusing further executions this subtask.",
                        tool_name, prev_count + 1,
                    )
                    tool_result = {
                        "success": False,
                        "error": (
                            f"⛔ LOOP DETECTED: You have called `{tool_name}` "
                            f"{prev_count + 1} times with identical args in "
                            "this subtask. The output will not change. Stop "
                            "calling tools and emit your final JSON answer "
                            "RIGHT NOW using information you already have."
                        ),
                        "result": {"stdout": "", "stderr": "", "exit_code": -2},
                        "duration_ms": 0,
                    }
                    force_finalize = True
                    await self._push_event({
                        "event":    "tool_result",
                        "round":    round_idx,
                        "call_idx": call_idx,
                        "tool":     tool_name,
                        "success":  False,
                        "error":    "Loop detected — refusing to re-run (saves GPU)",
                        "output":   "",
                        "task_id":  task_id,
                    })
                    # Record the synthesized result as a tool_result message
                    # so Qwen sees the termination notice next round.
                    messages.append({
                        "role": "tool",
                        "content": tool_result["error"],
                        "tool_call_id": call.get("id", ""),
                    })
                    continue  # skip real execution

                # ── B cap check: per-tool-name limit ────────────────
                # Already-executed tn_prev = N means we've seen this
                # tool name N times before → this is the (N+1)-th call.
                # Refuse the 6th onwards (5 calls is the cap).
                if tn_prev >= MAX_PER_TOOL_NAME:
                    logger.warning(
                        "Tool-name cap (#B): %s called %d times in subtask "
                        "%s — refusing further executions, forcing finalize.",
                        tool_name, tn_prev + 1, subtask_id,
                    )
                    tool_result = {
                        "success": False,
                        "error": (
                            f"⛔ TOOL OVERUSE: You have called `{tool_name}` "
                            f"{tn_prev + 1} times in this subtask — that's "
                            f"way past the {MAX_PER_TOOL_NAME}-call limit. "
                            "Stop trying new variations and emit your final "
                            "answer NOW with whatever you've already learned. "
                            "If you genuinely cannot complete the task with "
                            "the existing tool results, say so explicitly so "
                            "the user knows to escalate."
                        ),
                        "result": {"stdout": "", "stderr": "", "exit_code": -3},
                        "duration_ms": 0,
                    }
                    force_finalize = True
                    await self._push_event({
                        "event":    "tool_result",
                        "round":    round_idx,
                        "call_idx": call_idx,
                        "tool":     tool_name,
                        "success":  False,
                        "error":    f"Tool name cap hit ({tn_prev + 1}× {tool_name})",
                        "output":   "",
                        "task_id":  task_id,
                    })
                    messages.append({
                        "role": "tool",
                        "content": tool_result["error"],
                        "tool_call_id": call.get("id", ""),
                    })
                    continue

                try:
                    if call_idx in parallel_pre:
                        # Already executed concurrently before the loop
                        # — just consume the cached result. Saves
                        # awaiting registry.execute serially again.
                        tool_result = parallel_pre[call_idx]
                    else:
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

                # Include a preview of the error / output so the server
                # log doesn't just say "success=False" — when a deploy
                # fails, we want the reason visible in the log without
                # re-running everything with debug on.
                result_preview = ""
                if not tool_result.get("success"):
                    result_preview = (
                        tool_result.get("error")
                        or _preview_tool_output(tool_result)
                        or ""
                    )
                else:
                    result_preview = _preview_tool_output(tool_result) or ""
                # Flatten whitespace + cap so multi-line stderr stays readable.
                result_preview = (
                    " ".join(result_preview.split())[:200]
                )
                logger.info(
                    "Tool result tool=%s success=%s duration_ms=%s -- %s",
                    tool_name,
                    tool_result.get("success"),
                    tool_result.get("duration_ms"),
                    result_preview,
                )

                # ── Objective failure detection ───────────────────────────
                # Scan tool output for hard failure signals (unhealthy, Exited,
                # MySQL/Docker errors). If any match, we'll clamp confidence
                # at the end so Claude escalation kicks in even if Qwen
                # self-reports high confidence.
                # IMPORTANT: skip pure file-read tools — their output is
                # *data*, not log lines, so a CSV row containing the words
                # "command not found" or "no such file" inside a customer
                # comment would otherwise spuriously trigger failure mode
                # (see the orders_export.csv repurchase-rate task that
                # kept escalating because Shopify exports include free-text
                # notes that sometimes match the regex).
                _DATA_ONLY_TOOLS = {
                    "read_file", "file_read", "csv_head", "json_read",
                    "fetch_url",
                }
                if not objective_failure_seen and tool_name not in _DATA_ONLY_TOOLS:
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

                # ── D: stuck-loop circuit breaker ────────────────────
                # Track total tool failures in this subtask. When we
                # hit MAX_TOTAL_FAILURES, force the model to give up
                # and emit a final answer with whatever it has — saves
                # us from "kept retrying python_exec with new syntax
                # errors for 8 hours" pathology.
                if not tool_result.get("success"):
                    total_tool_failures += 1
                    if total_tool_failures >= MAX_TOTAL_FAILURES and not force_finalize:
                        logger.warning(
                            "Circuit breaker (#D): %d tool failures in subtask "
                            "%s — force-finalising loop.",
                            total_tool_failures, subtask_id,
                        )
                        force_finalize = True
                        # Inject a strong nudge into the conversation so
                        # the model's NEXT (final) round knows it must
                        # stop trying tools and synthesize an answer.
                        messages.append({
                            "role": "tool",
                            "content": (
                                f"⛔ CIRCUIT BREAKER: This subtask has had "
                                f"{total_tool_failures} tool failures. The "
                                "system is forcing you to STOP calling tools. "
                                "On your next turn, emit ONLY a final-answer "
                                "summary of what you have learned, what failed, "
                                "and (if relevant) what the user should try "
                                "instead. Do NOT call any more tools."
                            ),
                            "tool_call_id": call.get("id", ""),
                        })
                        await self._push_event({
                            "event":   "circuit_breaker",
                            "task_id": task_id,
                            "subtask_id": subtask_id,
                            "failures": total_tool_failures,
                        })

                # ── Artifact event ───────────────────────────────────
                # When a tool produces a file on disk, surface a
                # dedicated `artifact` event so the UI can attach a
                # download chip to the task's result message. Without
                # this the user has to ssh in to find the file —
                # which they actively complained about on the
                # Analytic report flow.
                if tool_result.get("success"):
                    inner = tool_result.get("result")
                    if isinstance(inner, dict):
                        # render_chart_report + write_file both use the
                        # "path" key. delete_file also uses "deleted"
                        # which we intentionally skip (nothing to download).
                        file_path = inner.get("path")
                        if file_path and tool_name != "delete_file":
                            try:
                                import os as _os
                                if _os.path.isfile(file_path):
                                    await self._push_event({
                                        "event":     "artifact",
                                        "task_id":   task_id,
                                        # subtask_id lets the verifier
                                        # (#2) bucket artifacts per-subtask.
                                        "subtask_id": subtask_id,
                                        "tool":      tool_name,
                                        "path":      file_path,
                                        "relative_path": inner.get("relative_path", ""),
                                        "size_bytes":    inner.get("size_bytes")
                                                        or inner.get("bytes_written")
                                                        or _os.path.getsize(file_path),
                                        "name":      _os.path.basename(file_path),
                                    })
                            except Exception:  # noqa: BLE001
                                pass

                # Soft nudge: if this is the 2nd identical call, append a
                # visible warning to the tool result so Qwen sees "you're
                # about to loop — don't". Real output is preserved (state
                # might have legitimately changed between calls), just
                # prefixed with a comment.
                if prev_count == 1:
                    warn_note = (
                        f"\n\n[⚠️ OpenTeddy notice: This is the 2nd call to "
                        f"`{tool_name}` with identical args. If the output "
                        "looks the same as before, MOVE ON — do not call it "
                        "a 3rd time. The system will refuse a 3rd duplicate "
                        "and force you to emit your final answer.]"
                    )
                    # Attach to whichever field the model will actually read.
                    if isinstance(tool_result.get("result"), dict):
                        tool_result["result"]["stderr"] = (
                            (tool_result["result"].get("stderr") or "") + warn_note
                        )
                    else:
                        tool_result["_dedup_notice"] = warn_note

                # Add tool result to conversation history. We used to
                # just json.dumps() the entire result dict, but bigger
                # models (Qwen 3 MoE, Gemma 4) over-interpreted the
                # nested structure — seeing a `"stdout":"..."` truncated
                # mid-string would make them "want to double-check" by
                # re-calling the tool. A flat labeled plain-text view
                # is far easier for any model to consume and produces
                # fewer spurious repeat calls.
                messages.append({
                    "role": "tool",
                    "content": _format_tool_result_for_model(tool_name, tool_result),
                })

            # If any call in this round tripped the loop-detector, bail
            # out NOW — no more Qwen inference rounds, force the final
            # answer path below. Saves the GPU from another pointless
            # round while the synthesized "LOOP DETECTED" message is
            # already in the messages list.
            if force_finalize:
                logger.warning(
                    "Forcing final answer for task %s — loop detector fired.",
                    task_id,
                )
                break

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
                    "options": {
                    "temperature": float(getattr(config, "qwen_temperature", 0.2)),
                    "num_predict": config.qwen_max_tokens,
                },
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
            _final_msg = _final_data.get("message", {})
            raw_text = _final_msg.get("content") or _final_msg.get("thinking") or ""
        except Exception as exc:  # noqa: BLE001
            return f"Executor error (forced final): {exc}", 0.0, None, None

        return self._finalize_response(raw_text, objective_failure_seen)

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _push_event(self, event: Dict[str, Any]) -> None:
        """Push an event to the Web UI via ws_callback (if registered)."""
        # Side-channel: also collect artifact events into an in-memory
        # bucket keyed by subtask_id so the orchestrator can verify the
        # produced files (#2 Per-step verification) without us having
        # to thread file paths through the existing return tuple.
        if event.get("event") == "artifact":
            sid = event.get("subtask_id") or event.get("task_id") or ""
            path = event.get("path")
            if sid and path:
                self._subtask_artifacts.setdefault(sid, []).append({
                    "path":          path,
                    "relative_path": event.get("relative_path", ""),
                    "size_bytes":    event.get("size_bytes") or 0,
                    "tool":          event.get("tool", ""),
                })
        if self.ws_callback:
            try:
                await self.ws_callback(event)
            except Exception as exc:  # noqa: BLE001
                logger.debug("ws_callback error: %s", exc)

    def pop_subtask_artifacts(self, subtask_id: str) -> List[Dict[str, Any]]:
        """Return + clear the artifact list collected for a given subtask."""
        return self._subtask_artifacts.pop(subtask_id, [])

    # ── Context watchdog (#4) ────────────────────────────────────────────
    # When the running prompt size approaches the model's num_ctx, this
    # collapses the middle of the conversation into a single
    # "<earlier-rounds-summary>" assistant message. Keeps:
    #   • messages[0]   — the original task description (NEVER drop)
    #   • messages[-3:] — the most recent assistant turn + its tool
    #                     results, so the next call still has fresh
    #                     concrete context to act on
    # Drops the middle, summarised by Qwen itself with a tight prompt.

    _COMPRESS_PROMPT = (
        "You are compressing earlier turns of an AI agent's tool-use "
        "session so the conversation can continue without busting the "
        "model's context window. Read the dropped turns below and emit "
        "a SHORT bullet recap (≤ 200 words, plain text, no markdown "
        "fences) covering:\n"
        "  1. What the agent has TRIED so far (tools called, args used)\n"
        "  2. What WORKED — concrete results, file paths, values found\n"
        "  3. What FAILED — errors, dead-ends, things to NOT retry\n"
        "  4. Any open questions / next-step hints for the agent\n"
        "Write tersely; the agent will read this verbatim before its "
        "next turn. Output ONLY the recap text — no preamble.\n\n"
        "── Dropped turns ──\n"
    )

    async def _compress_messages(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
    ) -> List[Dict[str, Any]]:
        if len(messages) <= 4:
            return messages
        head = messages[:1]              # original user task
        tail = messages[-3:]             # last assistant turn + its tool results
        drop = messages[1:-3]            # what we summarise

        # Render dropped turns as plain text. Cap each tool result so a
        # 50-KB CSV dump doesn't poison the summary call's own context.
        drop_lines = []
        for m in drop:
            role = m.get("role", "?")
            content = m.get("content") or ""
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            content = content[:1200]
            tcs = m.get("tool_calls")
            if tcs:
                content = (content + "\n").lstrip() + "[tool_calls=" + json.dumps(tcs, ensure_ascii=False)[:500] + "]"
            drop_lines.append(f"{role.upper()}: {content}")
        prompt = self._COMPRESS_PROMPT + "\n\n".join(drop_lines)

        # Use a small num_predict so the recap stays compact even if the
        # model wants to ramble. num_ctx stays generous so the summary
        # call itself doesn't truncate the input we're trying to compress.
        try:
            resp = await self._http.post(
                f"{config.qwen_base_url}/api/chat",
                json={
                    "model":    config.qwen_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream":   False,
                    "options":  {
                        "temperature": 0.1,
                        "num_predict": 600,
                        "num_ctx":     int(getattr(config, "qwen_num_ctx", 16384)),
                    },
                },
                timeout=120,
            )
            resp.raise_for_status()
            msg = resp.json().get("message") or {}
            recap = (msg.get("content") or msg.get("thinking") or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Context compression failed: %s. Falling back to "
                "drop-middle without summary.", exc,
            )
            recap = f"(earlier {len(drop)} messages omitted; compression failed: {exc})"

        # Best-effort: emit a UI event so the chat shows a small
        # "🪶 compressed earlier rounds" chip — gives the user a hint
        # that the agent had to truncate (not silent failure mode).
        await self._push_event({
            "type":   "context.compressed",
            "dropped": len(drop),
            "recap_chars": len(recap),
        })

        summary_msg = {
            "role": "assistant",
            "content": (
                f"<earlier-rounds-summary dropped=\"{len(drop)}\">\n"
                f"{recap}\n"
                f"</earlier-rounds-summary>"
            ),
        }
        return head + [summary_msg] + tail

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
