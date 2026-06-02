"""
OpenTeddy Tool Registry
Manages tool registration, JSON schema (Ollama function-calling format),
risk classification, and execution routing.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

from approval_store import ApprovalStore, approval_store as _default_store

logger = logging.getLogger(__name__)

RiskLevel = Literal["low", "high"]


# ── Telegram-driven safety policy ─────────────────────────────────────────────
# When a task is initiated from Telegram (we know because the bridge sets
# the `triggered_by` ContextVar before calling orchestrator.run), the
# user can't reach the web-UI approval prompt. Two design constraints:
#
#   1. Frictionless for legitimate work — don't make every shell_exec
#      hang waiting for an approval nobody will see.
#   2. Hard-block destructive actions — a malformed goal that leads the
#      model to "rm -rf /" must NOT proceed silently.
#
# The mechanism: auto-approve high-risk tools EXCEPT when the tool
# name or its arguments match one of the patterns below. A match
# returns a structured deny with a clear message, surfaced via the
# orchestrator's normal failure path → reaches the user in Telegram
# as "🚫 Blocked: ...".
#
# Add patterns conservatively. False positives lock out legitimate
# work; false negatives let destructive ops slip through. When in
# doubt, leave the pattern off the list — the user can always run
# the action from the web UI where approval is interactive.

# Tool-name matchers (exact-equal OR substring match against the
# tool's registered name). Apply to the *name*, not args.
_DENY_TOOL_NAMES: frozenset = frozenset({
    # Explicit file/DB delete operations
    "file_delete", "file_unlink", "file_rmtree",
    "db_delete", "db_drop_table", "db_truncate",
    # Anything Tauri / desktop-shell side that resets state
    "session_delete", "workspace_clear",
})

# Substring matchers — catches names like `mongo_delete_one`,
# `s3_object_remove`, future plugins, etc. Lowercase comparison.
_DENY_TOOL_NAME_SUBSTRINGS: Tuple[str, ...] = (
    "delete", "remove", "rmtree", "drop_table", "drop_database",
    "truncate", "wipe", "purge",
)

# Argument-content matchers — for tools whose name is generic
# (shell_exec_write, db_query, file_write) but the *contents* of the
# call would do something destructive. Each entry is a compiled regex
# applied to the JSON-serialised args, case-insensitive.
#
# Notes on the patterns:
#   - `\brm\b` matches "rm -rf foo" but not "merm" or "rmdir" (that's
#     handled separately so the message can name the right command).
#   - SQL patterns require trailing whitespace before the noun so
#     "DROPDOWN" or "TRUNCATED" inside a string literal aren't
#     spuriously flagged.
#   - `dd if=` covers the disk-wipe shape; we don't try to enumerate
#     every block-device path.
_DENY_ARG_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("rm",              re.compile(r"\brm\b", re.IGNORECASE)),
    ("rmdir",           re.compile(r"\brmdir\b", re.IGNORECASE)),
    ("unlink",          re.compile(r"\bunlink\b", re.IGNORECASE)),
    ("git rm",          re.compile(r"\bgit\s+rm\b", re.IGNORECASE)),
    ("shred",           re.compile(r"\bshred\b", re.IGNORECASE)),
    ("DROP TABLE",      re.compile(r"\bdrop\s+table\b", re.IGNORECASE)),
    ("DROP DATABASE",   re.compile(r"\bdrop\s+database\b", re.IGNORECASE)),
    ("DROP SCHEMA",     re.compile(r"\bdrop\s+schema\b", re.IGNORECASE)),
    ("TRUNCATE TABLE",  re.compile(r"\btruncate\s+table\b", re.IGNORECASE)),
    ("DELETE FROM",     re.compile(r"\bdelete\s+from\b", re.IGNORECASE)),
    ("mkfs",            re.compile(r"\bmkfs(\.\w+)?\b", re.IGNORECASE)),
    ("dd to device",    re.compile(r"\bdd\s+.*of=/dev/", re.IGNORECASE)),
    ("redirect to dev", re.compile(r">\s*/dev/sd[a-z]", re.IGNORECASE)),
    ("fdisk",           re.compile(r"\bfdisk\b", re.IGNORECASE)),
    ("format drive",    re.compile(r"\bformat\s+[A-Z]:", re.IGNORECASE)),
    ("chmod -R 000",    re.compile(r"\bchmod\s+(-R\s+)?0+\b", re.IGNORECASE)),
)


def _name_is_destructive(tool_name: str) -> Optional[str]:
    """Return a human-readable description of the deny match, or None
    if the tool name is allowed. Description is what gets surfaced to
    the user, so make it specific."""
    lname = tool_name.lower()
    if tool_name in _DENY_TOOL_NAMES:
        return f"tool name '{tool_name}' is on the destructive-action denylist"
    for needle in _DENY_TOOL_NAME_SUBSTRINGS:
        if needle in lname:
            return (
                f"tool name '{tool_name}' contains the destructive "
                f"keyword '{needle}'"
            )
    return None


def _args_are_destructive(args: Dict[str, Any]) -> Optional[str]:
    """Scan tool arguments for shell / SQL commands that would destroy
    data. Returns a description of the first match or None. The
    argument dict is serialised to JSON so nested values, list
    elements, etc. all get scanned in one pass — easier than walking
    the structure recursively and just as effective."""
    try:
        blob = json.dumps(args, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        # If we can't even serialise it, conservatively allow (the tool
        # will likely fail on its own). We log so this case is
        # diagnosable if it ever happens in practice.
        logger.debug("denylist: args not JSON-serialisable for %r", args)
        return None
    for label, pattern in _DENY_ARG_PATTERNS:
        if pattern.search(blob):
            return f"argument matched destructive pattern: {label}"
    return None


def check_destructive_denylist(
    tool_name: str, args: Dict[str, Any],
) -> Optional[str]:
    """Public entry point for the denylist. Returns the deny reason
    string if the call should be blocked, or None if it's allowed.
    Used by Telegram-initiated tasks to enforce safety without an
    interactive approval prompt."""
    return _name_is_destructive(tool_name) or _args_are_destructive(args)


# ── Tool result contract ───────────────────────────────────────────────────────

def make_result(
    success: bool,
    result: Any = None,
    error: Optional[str] = None,
    duration_ms: int = 0,
) -> Dict[str, Any]:
    """Unified tool result format."""
    return {
        "success": success,
        "result": result,
        "error": error,
        "duration_ms": duration_ms,
    }


# ── Registry ───────────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Central registry for all OpenTeddy tools.

    Each tool is registered with:
      - A callable (async function)
      - An Ollama-compatible JSON schema
      - A risk level: "low" (auto-execute) | "high" (requires human approval)

    Execution flow:
      LOW  → execute immediately, return result
      HIGH → create PendingApproval, suspend until resolved, then execute or reject
    """

    def __init__(self, store: Optional[ApprovalStore] = None) -> None:
        self._store: ApprovalStore = store or _default_store
        self._tools: Dict[str, Dict[str, Any]] = {}
        # { name: { "fn": callable, "schema": dict, "risk": RiskLevel } }

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        tool_fn: Callable,
        schema: Dict[str, Any],
        risk_level: RiskLevel = "low",
    ) -> None:
        """
        Register a tool.

        Args:
            tool_fn:    Async callable implementing the tool.
            schema:     Ollama function schema:
                        { "type": "function",
                          "function": { "name": ..., "description": ...,
                                        "parameters": { ... } } }
            risk_level: "low" | "high"
        """
        name: str = schema["function"]["name"]
        self._tools[name] = {
            "fn": tool_fn,
            "schema": schema,
            "risk": risk_level,
        }
        logger.debug("Registered tool '%s' (risk=%s)", name, risk_level)

    # ── Schema export ─────────────────────────────────────────────────────────

    def risk_of(self, tool_name: str) -> RiskLevel:
        """Return the registered risk level (or 'high' if the tool is
        unknown — fail-closed so we never accidentally parallelize
        something we don't know about)."""
        tool = self._tools.get(tool_name)
        if not tool:
            return "high"
        return tool["risk"]

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return all tool schemas in Ollama tools format."""
        return [t["schema"] for t in self._tools.values()]

    def get_schemas_by_names(self, names: List[str]) -> List[Dict[str, Any]]:
        """Return only the schemas for tools whose names match. Used by
        chat mode to expose a *specific* allow-list (just web_search,
        not the full toolbox) instead of all-or-nothing. Unknown names
        are silently dropped — the caller is usually a hardcoded list,
        and missing tools mean a feature wasn't built yet (e.g. the
        search API key wasn't registered), not a programming error."""
        out: List[Dict[str, Any]] = []
        for n in names:
            tool = self._tools.get(n)
            if tool:
                out.append(tool["schema"])
        return out

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return tool info including risk level (for /tools endpoint)."""
        return [
            {
                "name": name,
                "description": t["schema"]["function"].get("description", ""),
                "risk_level": t["risk"],
                "parameters": t["schema"]["function"].get("parameters", {}),
            }
            for name, t in self._tools.items()
        ]

    # ── Execution ─────────────────────────────────────────────────────────────

    async def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        task_id: str = "unknown",
        session_id: str = "",
    ) -> Dict[str, Any]:
        """
        Execute a tool by name.

        LOW risk  → call directly.
        HIGH risk → create PendingApproval, wait for human resolution.
                    On reject → return rejection error so agent can try alternative.

        session_id is bound to a ContextVar for the duration of the
        tool call so tools like ``db_query`` can look up the session's
        attached DB connection without the LLM having to pass session_id
        in its args (and without polluting the tool's JSON schema).
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return make_result(False, error=f"Unknown tool: {tool_name}")

        risk: RiskLevel = tool["risk"]
        fn: Callable = tool["fn"]

        # ── Telegram-driven safety policy ────────────────────────────────────
        # If the task was initiated from Telegram, the user can't see /
        # respond to the web-UI approval prompt. We split the policy:
        #   - Run the destructive-action denylist BEFORE the approval
        #     gate so even a "low-risk" tool with `rm -rf` in its args
        #     gets blocked. The denylist is a hard guard, never bypassed.
        #   - For tools that pass the denylist but ARE high-risk, skip
        #     the approval gate entirely (auto-approve). The user's
        #     whitelisted Telegram chat_id is itself the consent
        #     signal — the alternative is silent hangs the user can't
        #     resolve from inside Telegram.
        #
        # Web-UI / scheduled / other origins are unaffected — they fall
        # through to the existing approval flow below.
        try:
            from tools._context import get_triggered_by
            origin = get_triggered_by()
        except Exception:  # noqa: BLE001
            origin = ""

        if origin == "telegram":
            deny_reason = check_destructive_denylist(tool_name, args)
            if deny_reason:
                logger.warning(
                    "Telegram-driven task=%s blocked at tool=%s: %s "
                    "(args=%r)",
                    task_id, tool_name, deny_reason, args,
                )
                # Structured "blocked" result — the orchestrator surfaces
                # this through its normal failure handling, and the
                # Telegram bridge's reply formatter shows it to the user
                # verbatim. Includes a pointer to the web UI as the
                # escape hatch for cases where the user genuinely meant
                # the destructive action.
                return make_result(
                    False,
                    error=(
                        f"🚫 Blocked by Telegram safety policy: {deny_reason}. "
                        f"Destructive actions are not allowed when a task is "
                        f"driven from Telegram. If you really meant this, "
                        f"open the session in the web UI and run it there "
                        f"with interactive approval."
                    ),
                    duration_ms=0,
                )
            # Past the denylist — auto-approve any high-risk gate below
            # by skipping it. Low-risk tools fall through to execute
            # normally just like web-UI tasks.
            if risk == "high":
                logger.info(
                    "Telegram-driven task=%s auto-approving high-risk tool=%s "
                    "(denylist passed)",
                    task_id, tool_name,
                )
                # Skip the approval-store branch entirely; jump to execute.
                pass  # explicit no-op for readability

        # ── High-risk: gate on human approval ────────────────────────────────
        # Skipped above for Telegram-driven tasks that cleared the
        # denylist; runs as before for web-UI / scheduled / unknown
        # origins.
        elif risk == "high":
            approval_id = await self._store.create_approval(task_id, tool_name, args)
            logger.info(
                "High-risk tool '%s' queued for approval (id=%s, task=%s)",
                tool_name, approval_id, task_id,
            )
            # Read the auto-approve setting fresh on each call so toggling
            # it via Settings UI takes effect immediately (no agent
            # restart required). 0 = original behaviour (wait 5 min then
            # reject), > 0 = wait that many seconds then approve.
            try:
                from config import config as _cfg
                auto_after = float(getattr(_cfg, "approval_auto_approve_after", 0) or 0)
                wait_to = float(getattr(_cfg, "approval_wait_timeout", 1800) or 1800)
            except Exception:  # noqa: BLE001
                auto_after = 0.0
                wait_to = 1800.0
            approved = await self._store.wait_for_resolution(
                approval_id,
                timeout=wait_to,
                auto_approve_after=auto_after,
            )
            if not approved:
                return make_result(
                    False,
                    error="User rejected this action",
                    duration_ms=0,
                )

        # ── Execute ───────────────────────────────────────────────────────────
        # Bind session_id to the context var BEFORE invoking the tool
        # so any code path that needs it (db_tool's engine lookup,
        # future per-session sandboxing) can read it without an extra
        # function param. ONLY bind if session_id is non-empty —
        # orchestrator.run() already set the ContextVar at the top of
        # the task, and that binding propagates here through asyncio's
        # context inheritance. Overwriting with "" (the default for
        # callers that don't supply session_id) would erase the
        # orchestrator-level binding for this one tool call.
        from tools._context import set_session_id, reset_session_id
        ctx_token = set_session_id(session_id) if session_id else None
        start = time.monotonic()
        try:
            result = await fn(**args)
            duration_ms = int((time.monotonic() - start) * 1000)
            if isinstance(result, dict) and "success" in result:
                result["duration_ms"] = duration_ms
                return result
            return make_result(True, result=result, duration_ms=duration_ms)
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.error("Tool '%s' raised exception: %s", tool_name, exc)
            return make_result(False, error=str(exc), duration_ms=duration_ms)
        finally:
            if ctx_token is not None:
                reset_session_id(ctx_token)

    # ── Auto-registration ─────────────────────────────────────────────────────

    def auto_register_all(self) -> None:
        """
        Import and register every tool module under tools/.
        Called once at startup.
        """
        from tools.shell_tool import SHELL_TOOLS
        from tools.file_tool import FILE_TOOLS
        from tools.http_tool import HTTP_TOOLS
        from tools.package_tool import PACKAGE_TOOLS
        from tools.db_tool import DB_TOOLS
        from tools.gcp_tool import GCP_TOOLS
        from tools.deploy_tool import DEPLOY_TOOLS
        from tools.notify_tool import NOTIFY_TOOLS
        from tools.report_tool import REPORT_TOOLS
        from tools.analytic_tool import ANALYTIC_TOOLS
        from tools.search_tool import SEARCH_TOOLS
        # Browser tool sits next to http_tool semantically (both "fetch
        # a URL") but operates via headless Chromium so JS-rendered
        # pages actually surface their content. See browser_tool.py for
        # when to prefer it over fetch_url.
        from tools.browser_tool import BROWSER_TOOLS

        for fn, schema, risk in (
            SHELL_TOOLS + FILE_TOOLS + HTTP_TOOLS
            + PACKAGE_TOOLS + DB_TOOLS + GCP_TOOLS + DEPLOY_TOOLS
            + NOTIFY_TOOLS + REPORT_TOOLS + ANALYTIC_TOOLS
            + SEARCH_TOOLS + BROWSER_TOOLS
        ):
            self.register(fn, schema, risk)

        logger.info("Auto-registered %d tools.", len(self._tools))


# Module-level singleton
tool_registry = ToolRegistry()
