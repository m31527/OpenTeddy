"""
OpenTeddy tool execution context
─────────────────────────────────────────────────────────────────────────────
Per-call context that needs to be visible to tool implementations without
threading it through every tool function signature.

Today there's one such variable: ``current_session_id``. It lets
:mod:`tools.db_tool` (which builds DB engines off the session's stored
connection URL) figure out *which* session is making the call without
the orchestrator having to pass session_id into every tool invocation.

Pattern:
  1. :class:`tool_registry.ToolRegistry.execute` calls
     :func:`set_session_id` immediately before invoking the tool
     function.
  2. The tool function reads :func:`get_session_id` and looks up the
     session's DB connection from tracker.
  3. Python's :mod:`contextvars` keeps the binding scoped to the
     current async task — concurrent tool calls across sessions don't
     leak across each other.

Why contextvars not threadlocal / not function args:
  - Async-safe (threadlocal isn't, in asyncio).
  - Doesn't pollute the public tool signature exposed to the LLM (a
    function arg would show up in the JSON schema and confuse the
    model).
  - Cleanly bounded: each registry.execute call resets the binding.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Optional


# Empty string = "no session in context" — happens when a tool is
# invoked outside the orchestrator (admin endpoints, smoke tests).
# Tools that NEED a session check for empty and surface a clear error
# rather than letting a downstream None propagate.
_current_session_id: ContextVar[str] = ContextVar(
    "openteddy_current_session_id", default="",
)

# What surface initiated this task? Used by `tool_registry.execute` to
# decide approval policy:
#   - "telegram" → auto-approve high-risk tools UNLESS the call hits
#     the destructive-action denylist (rm, DROP TABLE, etc.), in
#     which case it's hard-blocked rather than prompted (the user
#     isn't watching the web UI).
#   - "" (web UI, default) → original behaviour: high-risk tools wait
#     on the approval store.
# More origins (slack, schedule, …) can be added without changing the
# approval flow's shape.
_triggered_by: ContextVar[str] = ContextVar(
    "openteddy_triggered_by", default="",
)


def set_session_id(session_id: str) -> object:
    """Bind ``session_id`` for the current async task. Returns the
    token from :meth:`ContextVar.set` so the caller can later reset
    the binding via :func:`reset_session_id`."""
    return _current_session_id.set(session_id or "")


def get_session_id() -> str:
    """Return the session_id bound to the current async task, or
    an empty string if no binding is in place."""
    return _current_session_id.get()


def reset_session_id(token: object) -> None:
    """Reset the context var to its previous value. Pair with the
    token returned by :func:`set_session_id`. Safe to call on a
    token from a different task — contextvars handle that cleanly."""
    try:
        _current_session_id.reset(token)  # type: ignore[arg-type]
    except (ValueError, LookupError):
        # Token came from a different context (e.g. tool wrapped its
        # own task); reset isn't meaningful, so swallow.
        pass


def set_triggered_by(origin: str) -> object:
    """Bind the task origin (e.g. ``"telegram"``) for the current async
    task. Returns the reset token. Caller is responsible for resetting
    via :func:`reset_triggered_by` once the task completes — failing
    to reset leaks the binding into the next task in the same context."""
    return _triggered_by.set(origin or "")


def get_triggered_by() -> str:
    """Return the origin string bound to the current async task, or an
    empty string if not bound (default web-UI behaviour)."""
    return _triggered_by.get()


def reset_triggered_by(token: object) -> None:
    """Reset the origin binding. Safe to call on a token from a
    different context — contextvars handle that cleanly."""
    try:
        _triggered_by.reset(token)  # type: ignore[arg-type]
    except (ValueError, LookupError):
        pass
