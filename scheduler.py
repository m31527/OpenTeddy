"""
OpenTeddy Scheduled Tasks runtime
─────────────────────────────────────────────────────────────────────────────
Cron-driven recurring tasks. Each schedule belongs to a session, and runs
the orchestrator with a stored goal at the configured cron time. If the
session is bound to a Telegram chat (via `sessions.telegram_chat_id`),
results are pushed back to that chat automatically — no extra wiring
needed in the schedule itself.

Why this exists
---------------
The user-facing pitch:

  "Every weekday 9:30, fetch GitHub trending top 10 and push the summary
   to my Telegram."

With this module wired up: one POST to /schedules + one cron expression
delivers that. The first time the schedule fires, OpenTeddy:

  1. Loads the bound session (memory / workspace / mode preserved)
  2. Sets `triggered_by=schedule` on the task — denylist still hard-blocks
     destructive ops, but auto-approve flows (same as Telegram-driven
     runs) skip web-UI approval since nobody's watching at 9:30 AM
  3. Calls orchestrator.run() exactly the way POST /run does
  4. On success: pushes the formatted summary to the session's bound
     Telegram chat (if any); resets the failure counter
  5. On failure: bumps `consecutive_failures`; if N reached, disables
     the schedule + pushes an alert to Telegram so it can't keep firing
     silently in the background

Why APScheduler specifically
----------------------------
Standard async cron solution in Python. Handles cron parsing, DST, missed
fires (we use the default "skip missed" policy — running 6 hours late
just confuses the user), and integrates cleanly with FastAPI's asyncio
event loop. Pure Python, no native deps, PyInstaller-bundles cleanly.

Lifecycle
---------
Started from main.lifespan() AFTER the orchestrator is constructed:

    from scheduler import start as start_scheduler
    await start_scheduler(tracker, orchestrator, ws_manager.broadcast)
    ...
    await stop_scheduler()

`start()` is idempotent — a second call when already running is a no-op.
Loads all `enabled=1` rows from `scheduled_tasks` and registers them with
APScheduler. Same job ids in the SQLite table and APScheduler so add /
update / delete operations map 1-to-1.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Module state ──────────────────────────────────────────────────────────────

# AsyncIOScheduler instance — lazy-created on first start() so importing
# the module doesn't pay APScheduler's setup cost on every server boot
# (only when scheduling is actually used).
_scheduler: Any = None

# Handles passed in from main.lifespan() — same pattern as the Telegram
# bridge. Kept module-global so the job callback closures don't have to
# thread them through every call.
_tracker: Any = None
_orchestrator: Any = None
_ws_broadcast: Optional[Callable] = None


# ── Lifecycle ─────────────────────────────────────────────────────────────────

async def start(
    tracker: Any, orchestrator: Any, ws_broadcast: Optional[Callable] = None,
) -> None:
    """Spawn the APScheduler instance + load all enabled schedules from
    SQLite. Called from main.lifespan(). Idempotent — calling twice is
    harmless."""
    global _scheduler, _tracker, _orchestrator, _ws_broadcast
    _tracker = tracker
    _orchestrator = orchestrator
    _ws_broadcast = ws_broadcast

    if _scheduler is not None and getattr(_scheduler, "running", False):
        logger.debug("Scheduler already running; ignoring start()")
        return

    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
    except ImportError:
        logger.warning(
            "APScheduler not installed — scheduled tasks disabled. "
            "Run `pip install -r requirements.txt` and restart."
        )
        return

    _scheduler = AsyncIOScheduler()
    _scheduler.start()

    rows = await tracker.list_scheduled_tasks(enabled_only=True)
    registered = 0
    for row in rows:
        try:
            _register_job(row)
            registered += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to register schedule %s on startup: %s",
                row.get("id"), exc,
            )
    logger.info(
        "Scheduler started — registered %d schedule(s) from SQLite.",
        registered,
    )


async def stop() -> None:
    """Shut down APScheduler. Called from main.lifespan() on app exit.
    Best-effort: any failure is logged, not raised."""
    global _scheduler
    if _scheduler is None:
        return
    try:
        _scheduler.shutdown(wait=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Scheduler shutdown raised (non-fatal): %s", exc)
    _scheduler = None
    logger.info("Scheduler stopped.")


# ── CRUD helpers used by the API + Telegram /cron command ────────────────────

async def add_schedule(
    session_id: str, cron: str, goal: str, max_failures: int = 3,
) -> Dict[str, Any]:
    """Persist a new schedule + register it with APScheduler. Returns the
    full row so the caller (API endpoint or Telegram handler) can echo
    the id back to the user.

    Validates the cron expression *before* writing to SQLite so a typo
    like `"30 9 * * *"` vs `"every day at 9:30"` surfaces as a clear
    error instead of silently saving an inert row.
    """
    _validate_cron(cron)
    schedule_id = str(uuid.uuid4())
    await _tracker.create_scheduled_task(
        schedule_id=schedule_id,
        session_id=session_id,
        cron=cron,
        goal=goal,
        max_failures=max_failures,
    )
    row = await _tracker.get_scheduled_task(schedule_id)
    if row:
        _register_job(row)
    return row or {}


async def delete_schedule(schedule_id: str) -> bool:
    """Remove a schedule from both SQLite and APScheduler. Returns True
    if anything was actually deleted, False if the id was unknown."""
    db_deleted = await _tracker.delete_scheduled_task(schedule_id)
    _unregister_job(schedule_id)
    return db_deleted


async def set_enabled(schedule_id: str, enabled: bool) -> bool:
    """Toggle enabled flag + add/remove from APScheduler accordingly.
    Returns True on success, False if the row doesn't exist."""
    row = await _tracker.get_scheduled_task(schedule_id)
    if not row:
        return False
    await _tracker.update_scheduled_task(schedule_id, enabled=enabled)
    if enabled:
        row["enabled"] = True
        _register_job(row)
    else:
        _unregister_job(schedule_id)
    return True


async def run_now(schedule_id: str) -> Optional[str]:
    """Trigger a schedule immediately (out-of-band), regardless of its
    cron. Useful for "test this schedule before tomorrow morning" via
    the API or Telegram command. Returns the new task_id, or None if
    the schedule wasn't found.

    Doesn't bump the cron — APScheduler's next-scheduled fire stays
    unchanged."""
    row = await _tracker.get_scheduled_task(schedule_id)
    if not row:
        return None
    # Fire the trigger handler directly with asyncio.create_task so we
    # don't block the HTTP / Telegram caller waiting for the run.
    asyncio.create_task(
        _on_trigger(schedule_id),
        name=f"sched_run_now:{schedule_id[:8]}",
    )
    return schedule_id


# ── APScheduler glue ──────────────────────────────────────────────────────────

def _register_job(row: Dict[str, Any]) -> None:
    """Take a scheduled_tasks row and (re)register it with APScheduler.
    Job id = SQLite id, so add+update+delete operations map 1-to-1
    without us having to maintain a separate mapping."""
    if _scheduler is None:
        return
    from apscheduler.triggers.cron import CronTrigger
    schedule_id = row["id"]
    cron = row["cron"]
    trigger = _cron_to_trigger(cron)

    # Replace any existing registration so update_schedule() can call
    # _register_job() without worrying about duplicates.
    try:
        _scheduler.remove_job(schedule_id)
    except Exception:  # noqa: BLE001
        pass

    _scheduler.add_job(
        _on_trigger,
        trigger=trigger,
        id=schedule_id,
        args=[schedule_id],
        replace_existing=True,
        # Drop missed fires older than 5 min — running yesterday's
        # GitHub trending job at noon today helps nobody.
        misfire_grace_time=300,
        coalesce=True,
    )
    # Write next-fire time back to the row so the UI / Telegram /cron
    # command can show "next run: 2026-06-04 09:30".
    job = _scheduler.get_job(schedule_id)
    next_run = job.next_run_time.isoformat() if job and job.next_run_time else None
    if next_run:
        asyncio.create_task(_persist_next_run(schedule_id, next_run))


async def _persist_next_run(schedule_id: str, next_run_at: str) -> None:
    try:
        await _tracker.db.execute(
            "UPDATE scheduled_tasks SET next_run_at=?, updated_at=? WHERE id=?",
            (next_run_at, datetime.utcnow().isoformat(), schedule_id),
        )
        await _tracker.db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.debug("persist next_run_at failed for %s: %s", schedule_id, exc)


def _unregister_job(schedule_id: str) -> None:
    if _scheduler is None:
        return
    try:
        _scheduler.remove_job(schedule_id)
    except Exception:  # noqa: BLE001
        pass


def _cron_to_trigger(cron: str):
    """Parse a 5-field cron string into an APScheduler CronTrigger.
    Wrapped so cron validation is centralised — _validate_cron and
    _register_job both go through here."""
    from apscheduler.triggers.cron import CronTrigger
    return CronTrigger.from_crontab(cron.strip())


def _validate_cron(cron: str) -> None:
    """Raise ValueError if the cron expression is malformed. Caller is
    expected to surface the error string to the user."""
    try:
        _cron_to_trigger(cron)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Invalid cron expression {cron!r}: {exc}. "
            "Expected 5 fields: 'minute hour day month weekday' "
            "(e.g. '30 9 * * *' for daily 9:30)"
        ) from exc


# ── Trigger handler — what happens when a schedule fires ──────────────────────

async def _on_trigger(schedule_id: str) -> None:
    """Called by APScheduler when a schedule's cron fires. Runs the
    orchestrator, records lifecycle, pushes results to Telegram if the
    session is bound, and circuit-breaks after N consecutive failures.

    Errors here must NEVER kill the scheduler — the loop owns the
    server's "always available" promise. We catch broadly + log."""
    try:
        await _execute_scheduled_run(schedule_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Scheduled run for %s raised at the top level: %s",
            schedule_id, exc,
        )


async def _execute_scheduled_run(schedule_id: str) -> None:
    """Inner body — the part that can raise without taking down the
    scheduler loop."""
    row = await _tracker.get_scheduled_task(schedule_id)
    if not row or not row.get("enabled"):
        logger.info(
            "Schedule %s fired but is disabled or missing — skipping",
            schedule_id,
        )
        return

    session_id = row["session_id"]
    goal = row["goal"]
    logger.info(
        "Scheduled run firing: schedule=%s session=%s goal=%r",
        schedule_id, session_id, goal[:80],
    )

    # Build a TaskRequest exactly like POST /run does, marked with
    # triggered_by=schedule so tool_registry's auto-approve / denylist
    # policy treats it identically to a Telegram-driven run (no UI
    # approval prompts, but destructive ops still hard-blocked).
    from models import TaskRequest, SessionMode
    from tools._context import set_triggered_by, reset_triggered_by
    task_id = str(uuid.uuid4())
    req = TaskRequest(
        id=task_id,
        goal=goal,
        context={
            "triggered_by": "schedule",
            "schedule_id":  schedule_id,
        },
        priority=1,
        session_id=session_id,
        mode=SessionMode.CODE,
    )

    origin_token = set_triggered_by("schedule")
    success = False
    error_str: Optional[str] = None
    result = None
    started = time.monotonic()
    try:
        # Hard cap consistent with the Telegram bridge's wait_for — a
        # stuck Ollama call shouldn't keep this schedule's "currently
        # running" slot indefinitely.
        result = await asyncio.wait_for(
            _orchestrator.run(req), timeout=600,
        )
        success = True
    except asyncio.TimeoutError:
        error_str = "Run exceeded 10 min and was cancelled"
        logger.warning("Schedule %s: %s", schedule_id, error_str)
    except Exception as exc:  # noqa: BLE001
        error_str = str(exc)[:500]
        logger.exception("Schedule %s orchestrator raised: %s", schedule_id, exc)
    finally:
        reset_triggered_by(origin_token)

    elapsed_s = time.monotonic() - started

    # Compute next fire time so the UI / Telegram can show "next: …"
    next_run_at: Optional[str] = None
    if _scheduler is not None:
        job = _scheduler.get_job(schedule_id)
        if job and job.next_run_time:
            next_run_at = job.next_run_time.isoformat()

    updated_row = await _tracker.record_scheduled_run(
        schedule_id=schedule_id,
        status="success" if success else "failure",
        task_id=task_id,
        error=error_str,
        next_run_at=next_run_at,
    )

    # Push result to the bound Telegram chat (if any). Best-effort: a
    # delivery failure mustn't propagate.
    try:
        await _maybe_push_to_telegram(updated_row, result, error_str, elapsed_s)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Scheduled-run Telegram delivery failed: %s", exc)

    # Circuit breaker: disable + alert after N consecutive failures so
    # the schedule doesn't keep firing into the void.
    fails = int(updated_row.get("consecutive_failures") or 0)
    cap = int(updated_row.get("max_consecutive_failures") or 3)
    if not success and fails >= cap:
        logger.warning(
            "Schedule %s disabled after %d consecutive failures (cap=%d)",
            schedule_id, fails, cap,
        )
        await _tracker.update_scheduled_task(schedule_id, enabled=False)
        _unregister_job(schedule_id)
        try:
            await _maybe_push_disable_alert(updated_row, fails, cap)
        except Exception as exc:  # noqa: BLE001
            logger.debug("disable-alert push failed: %s", exc)


# ── Telegram delivery helpers ─────────────────────────────────────────────────

async def _maybe_push_to_telegram(
    schedule_row: Dict[str, Any],
    result: Any,
    error_str: Optional[str],
    elapsed_s: float,
) -> None:
    """If the schedule's session is bound to a Telegram chat, push the
    result + artifacts there. Reuses telegram_bridge's formatters so
    the message shape matches a normal Telegram-driven run — same
    status header, same '📎 Files produced' block, same inline content
    + sendDocument for binaries."""
    session_id = schedule_row.get("session_id")
    if not session_id:
        return
    sess = await _tracker.get_session(session_id)
    if not sess:
        return
    chat_id = (sess.get("telegram_chat_id") or "").strip()
    if not chat_id:
        return

    # Late import to keep the cyclic risk out of module load time.
    from telegram_bridge import (
        _send_reply, _format_result_for_telegram,
        _format_artifacts_block, _push_artifact_contents,
    )

    if error_str:
        header = (
            f"❌ Scheduled run failed · {elapsed_s:.1f}s\n"
            f"\n"
            f"Goal: {schedule_row.get('goal','')[:120]}\n"
            f"Error: {error_str[:200]}\n"
            f"\n"
            f"schedule: {schedule_row.get('id','')[:8]}"
        )
        await _send_reply(chat_id, header)
        return

    # Success path — same formatter as Telegram-driven runs, prefixed
    # with a "⏰ Scheduled run" marker so the user can tell where it
    # came from at a glance.
    body = _format_result_for_telegram(result, session_id, elapsed_s)
    body = "⏰ Scheduled run\n\n" + body

    # Append artifact list if any
    artifacts = []
    try:
        task_row = await _tracker.get_task(getattr(result, "task_id", "") or "")
        if task_row:
            artifacts = task_row.get("artifacts") or []
    except Exception:  # noqa: BLE001
        pass
    if artifacts:
        body += _format_artifacts_block(artifacts)
    await _send_reply(chat_id, body)
    if artifacts:
        await _push_artifact_contents(chat_id, artifacts)


async def _maybe_push_disable_alert(
    schedule_row: Dict[str, Any], fails: int, cap: int,
) -> None:
    """Tell the bound Telegram chat that we tripped the circuit breaker.
    Without this the schedule would silently stop and the user wouldn't
    notice for days."""
    session_id = schedule_row.get("session_id")
    if not session_id:
        return
    sess = await _tracker.get_session(session_id)
    if not sess:
        return
    chat_id = (sess.get("telegram_chat_id") or "").strip()
    if not chat_id:
        return
    from telegram_bridge import _send_reply
    await _send_reply(
        chat_id,
        f"🛑 Scheduled task disabled — failed {fails} times in a row "
        f"(cap = {cap}).\n"
        f"\n"
        f"Goal: {schedule_row.get('goal', '')[:120]}\n"
        f"Schedule id: {schedule_row.get('id', '')[:8]}\n"
        f"\n"
        "Last error: " + (schedule_row.get('last_error') or '(none)')[:200] + "\n"
        f"\n"
        "Re-enable from the API once the root cause is fixed:\n"
        f"  curl -X PATCH http://<server>/schedules/{schedule_row.get('id','')} "
        '-d \'{"enabled": true}\'',
    )
