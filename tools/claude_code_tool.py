"""
OpenTeddy → Claude Code: hand a coding task to Claude Code in a project
folder on this machine. Runs in the background; the result is sent to the
Telegram default chat. "用 Claude Code …" messages take the direct path
(claude_code.py); this tool is for the planner when coding is one step of
a bigger job. High risk: it edits files (never commits or pushes).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import claude_code as CC
from config import config
from tool_registry import RiskLevel, make_result


async def claude_code(task: str, directory: str = "") -> Dict[str, Any]:
    t0 = time.monotonic()
    try:
        if not CC.find_binary():
            raise CC.ClaudeCodeError("Claude Code is not installed on this machine.")
        where = CC.resolve_directory(directory or None)
    except CC.ClaudeCodeError as exc:
        return make_result(False, error=str(exc), duration_ms=int((time.monotonic() - t0) * 1000))
    chat = (getattr(config, "telegram_default_chat_id", "") or "").strip()

    async def notify(text: str) -> None:
        if chat and (getattr(config, "telegram_bot_token", "") or "").strip():
            from telegram_bridge import _send_rich
            await _send_rich(chat, text)

    job = CC.start(task, where, notify)
    return make_result(True, result={"job_id": job, "directory": where, "status": "running",
                                     "note": "Runs in the background; the result is sent to Telegram."},
                       duration_ms=int((time.monotonic() - t0) * 1000))


async def claude_code_jobs() -> Dict[str, Any]:
    return make_result(True, result=CC.jobs())


_SCHEMA = {"type": "function", "function": {
    "name": "claude_code",
    "description": (
        "Hand a coding task to Claude Code (a stronger coding agent) in a project folder on "
        "this machine: fix failing tests, implement a change, review code. Runs in the "
        "background (minutes); the result and changed files are sent to the user on Telegram. "
        "It edits files but never commits or pushes. Describe the task fully — Claude Code "
        "can't ask follow-up questions."),
    "parameters": {"type": "object", "properties": {
        "task": {"type": "string", "description": "What Claude Code should do, in full."},
        "directory": {"type": "string", "description": "Project folder, e.g. ~/projects/shop. Empty = the default folder from Settings."},
    }, "required": ["task"]}}}
_SCHEMA_JOBS = {"type": "function", "function": {
    "name": "claude_code_jobs",
    "description": "Status of recent Claude Code jobs (running / done / failed) with their summaries.",
    "parameters": {"type": "object", "properties": {}}}}

CLAUDE_CODE_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    (claude_code, _SCHEMA, "high"),
    (claude_code_jobs, _SCHEMA_JOBS, "low"),
]
