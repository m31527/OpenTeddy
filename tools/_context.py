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


# ── Subtask deadline credit ───────────────────────────────────────────────────
#
# The orchestrator caps each subtask with a wall-clock timeout. That
# budget is meant to catch a stuck agent — a model looping, a tool
# hanging. It is NOT meant to cap work that is demonstrably progressing,
# but a plain wall-clock timer cannot tell the two apart, so anything
# genuinely slow gets killed and reported as "timeout" while succeeding.
#
# Two things legitimately consume wall-clock without the agent being
# stuck, and both are credited back here:
#
#   1. HUMAN TIME — the approval prompt blocks inside the tool call, so
#      seconds the user spends reading a high-risk call and deciding are
#      billed to the agent. That punishes exactly the careful review the
#      approval gate exists to encourage.
#   2. DECLARED LONG WORK — a tool that announces up front how long it
#      may take (http_post timeout=2700 against a video endpoint that
#      really does run 31 minutes). The tool's own timeout already bounds
#      it, so the subtask watchdog double-bounding it adds no safety,
#      only false cancellations.
#
# The effect is a timeout that tolerates slow work without going blind:
# the default stays tight (a hung agent is still caught in ~15 min) while
# a call that declares 45 minutes gets exactly 45 minutes, no global
# setting to raise and no other task made less responsive.
#
# Long work RESERVES its worst case before starting and REFUNDS whatever
# it didn't use, so the deadline covers the call while it is in flight
# but only the real elapsed time is charged once it returns.
#
# A mutable dict is used deliberately: asyncio copies the ContextVar into
# the child task by reference, so mutations made deep inside a tool call
# are visible to the watchdog running above it.
_deadline_credit: ContextVar[Optional[dict]] = ContextVar(
    "openteddy_deadline_credit", default=None,
)


def new_approval_budget() -> dict:
    """Start a fresh per-subtask credit budget. Call BEFORE spawning the
    executor task so the child inherits this dict."""
    budget: dict = {"paused": 0.0}
    _deadline_credit.set(budget)
    return budget


def add_deadline_credit(seconds: float) -> None:
    """Extend (or, with a negative value, refund) the subtask deadline.

    No-op outside a subtask. The running total is floored at zero so a
    mismatched refund can never shorten the original budget."""
    budget = _deadline_credit.get()
    if budget is not None:
        budget["paused"] = max(0.0, budget.get("paused", 0.0) + seconds)


def add_approval_pause(seconds: float) -> None:
    """Credit time spent waiting on a human."""
    add_deadline_credit(max(0.0, seconds))


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
