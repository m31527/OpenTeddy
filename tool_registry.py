"""
OpenTeddy Tool Registry
Manages tool registration, JSON schema (Ollama function-calling format),
risk classification, and execution routing.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Literal, Optional

from approval_store import ApprovalStore, approval_store as _default_store

logger = logging.getLogger(__name__)

RiskLevel = Literal["low", "high"]


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

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return all tool schemas in Ollama tools format."""
        return [t["schema"] for t in self._tools.values()]

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
    ) -> Dict[str, Any]:
        """
        Execute a tool by name.

        LOW risk  → call directly.
        HIGH risk → create PendingApproval, wait for human resolution.
                    On reject → return rejection error so agent can try alternative.
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return make_result(False, error=f"Unknown tool: {tool_name}")

        risk: RiskLevel = tool["risk"]
        fn: Callable = tool["fn"]

        # ── High-risk: gate on human approval ────────────────────────────────
        if risk == "high":
            approval_id = await self._store.create_approval(task_id, tool_name, args)
            logger.info(
                "High-risk tool '%s' queued for approval (id=%s, task=%s)",
                tool_name, approval_id, task_id,
            )
            approved = await self._store.wait_for_resolution(approval_id, timeout=300.0)
            if not approved:
                return make_result(
                    False,
                    error="User rejected this action",
                    duration_ms=0,
                )

        # ── Execute ───────────────────────────────────────────────────────────
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

        for fn, schema, risk in (
            SHELL_TOOLS + FILE_TOOLS + HTTP_TOOLS
            + PACKAGE_TOOLS + DB_TOOLS + GCP_TOOLS + DEPLOY_TOOLS
            + NOTIFY_TOOLS
        ):
            self.register(fn, schema, risk)

        logger.info("Auto-registered %d tools.", len(self._tools))


# Module-level singleton
tool_registry = ToolRegistry()
