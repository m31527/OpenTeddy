"""
OpenTeddy Telegram Inbound Bridge
─────────────────────────────────────────────────────────────────────────────
Background task that long-polls Telegram's `getUpdates`, routes
whitelisted messages to a per-chat persistent OpenTeddy session, runs
them through the orchestrator, and pushes the result back via the
existing `telegram_send` tool (`tools/notify_tool.py`).

Why this exists
---------------
Outbound notifications (the agent calling `telegram_send` to push a
finished report) is half the loop. The other half is "send me a message
from anywhere → the agent picks it up → runs the task → replies".
Combined, OpenTeddy starts to feel like a personal assistant you can
talk to from the train, not a desktop app you have to open.

Why long-polling, not a webhook
-------------------------------
A webhook needs a publicly reachable HTTPS endpoint, which most of our
users (self-hosted server on a home LAN, Tailscale, Cloudflare Tunnel,
etc.) don't have set up. Long-polling works behind any NAT — we make
outbound requests only, Telegram holds them open for ~30 s waiting for
new messages, then returns whatever it has. Cost: one open TCP
connection at all times. Acceptable.

Lifecycle
---------
Called from `main.lifespan()` after the orchestrator is initialised:

    from telegram_bridge import start as start_telegram_bridge
    await start_telegram_bridge(tracker, orchestrator)
    ...
    await stop_telegram_bridge()  # on shutdown

`start()` is idempotent and a no-op when:
  - `config.telegram_bot_token` is empty
  - `config.telegram_inbound_enabled` is False
  - `config.telegram_inbound_chat_id_whitelist` is empty (we refuse to
    listen on an open bot — any stranger could otherwise drive the
    agent on your machine just by knowing your bot username)

Security
--------
Hard whitelist. Messages from chat_ids NOT in the whitelist are
**silently dropped** — no reply, no log line beyond debug — so the
bot's presence isn't a probe signal. If you've configured the bot but
forgotten to whitelist yourself, you'll see a "ignored chat_id %s" log
at DEBUG level; bump the log level to investigate.

Phase 1 scope (THIS FILE TODAY)
-------------------------------
  - Polling loop with long-polling + offset acknowledgement
  - Whitelist check
  - `/start`, `/help` commands → static replies
  - Any other text → echo back "received: <text>" so you can verify
    end-to-end connectivity before wiring the orchestrator

Phase 2 (next)
--------------
  - Resolve chat_id → session (auto-create on first message, bind in DB
    via `tracker.bind_session_to_telegram_chat`)
  - Run orchestrator with the text as goal
  - Push the resulting summary back via `telegram_send`
  - Concurrency guard: refuse new messages while the chat's session has
    a task running (reply "⏳ task running — /cancel to abort")

Phase 3 (later)
---------------
  - `/cancel`, `/new` commands
  - Approval-required gating (notify in chat, wait for web-UI ok)
  - Hot-reload on Settings save (currently needs server restart)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Set

import httpx

from config import config

logger = logging.getLogger(__name__)


# ── Tunables ──────────────────────────────────────────────────────────────────

_API_BASE = "https://api.telegram.org/bot{token}"

# Long-poll timeout. Telegram caps at 50 s; 30 s is the conventional
# default — long enough that getUpdates blocks efficiently when there's
# nothing happening, short enough that a config reload or shutdown
# isn't kept waiting half a minute.
_POLL_TIMEOUT_S = 30

# httpx client timeout = poll timeout + slack for TLS handshake and the
# server's own internal processing. If this is tighter than the poll
# timeout we'd cut off Telegram's reply just as it was about to land,
# producing a synthetic "always timing out" failure.
_HTTP_TIMEOUT_S = _POLL_TIMEOUT_S + 10

# Exponential backoff on transient failures (network down, 5xx, rate
# limited). Capped at 5 min so a long outage doesn't make recovery
# imperceptibly slow. Resets to the first value on the next success.
_BACKOFF_S = [5, 15, 30, 60, 120, 300]


# ── Module-level state ────────────────────────────────────────────────────────

# Single global task — at most one polling loop per process. Storing
# the asyncio.Task lets us .cancel() it cleanly on shutdown.
_task: Optional[asyncio.Task] = None

# Handles passed in from main.lifespan() — kept module-global so the
# message handler closures don't have to thread them through every call.
_tracker: Any = None        # type: tracker.Tracker
_orchestrator: Any = None   # type: orchestrator.Orchestrator

# Per-chat concurrency guard. Maps chat_id → the asyncio.Task currently
# running orchestrator.run() for that chat. While an entry exists, new
# free-text messages from the same chat are rejected with "⏳ task
# running" rather than queued — that's the cleanest UX (the user knows
# the previous request is still in flight) and avoids the corner cases
# of either silently dropping or piling up an unbounded queue.
_running_chats: Dict[str, asyncio.Task] = {}

# Diagnostics: last chat_id we silent-dropped because it wasn't on the
# whitelist. Surfaced by GET /admin/telegram/status so a user setting up
# inbound for the first time can see "I sent a message from chat X but
# my whitelist is Y" without having to grep server logs. Bounded — only
# the *most recent* drop is kept; we don't need a history, just the hint
# for whichever chat the user is currently trying to connect.
_last_dropped_chat_id: Optional[str] = None

# Telegram caps outbound messages at 4096 chars. Cap our payload at
# 3500 so there's room for the metadata header + trailer + truncation
# notice. Anything longer gets cut + a "...(truncated)" hint pointing
# to the web UI for the full summary.
_TELEGRAM_MAX_BODY_CHARS = 3500


# ── Entry / exit ──────────────────────────────────────────────────────────────

async def start(tracker: Any, orchestrator: Any) -> None:
    """Spawn the polling loop if config says we should. Idempotent —
    safe to call multiple times; second call when already running is a
    no-op. Called from main.lifespan() at startup."""
    global _task, _tracker, _orchestrator
    _tracker = tracker
    _orchestrator = orchestrator

    if _task and not _task.done():
        logger.debug("telegram bridge already running; ignoring start()")
        return

    reason = _why_we_should_not_start()
    if reason:
        logger.info("Telegram inbound bridge NOT started: %s", reason)
        return

    _task = asyncio.create_task(_poll_loop(), name="telegram_bridge")
    logger.info(
        "Telegram inbound bridge started — polling with %d-id whitelist.",
        len(_whitelisted_chat_ids()),
    )


async def stop() -> None:
    """Cancel the polling loop. Called from main.lifespan() on shutdown."""
    global _task
    if not _task:
        return
    _task.cancel()
    try:
        await _task
    except asyncio.CancelledError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("telegram bridge stop() saw unexpected: %s", exc)
    _task = None
    logger.info("Telegram inbound bridge stopped.")


def status() -> Dict[str, Any]:
    """Snapshot of the bridge's runtime state for diagnostics. Surfaced
    by GET /admin/telegram/status so the user setting this up can see
    "yes the bridge is running, my whitelist is [X, Y], you sent your
    last unauthorised message from chat Z" in one place — without
    having to grep server logs or guess from silence.

    Intentionally token-free (only `token_set: bool` flag) so the
    response is safe to glance at on a shared screen."""
    token_set = bool((getattr(config, "telegram_bot_token", "") or "").strip())
    return {
        "running":              _task is not None and not _task.done(),
        "inbound_enabled":      bool(getattr(config, "telegram_inbound_enabled", False)),
        "token_set":            token_set,
        "whitelist":            sorted(_whitelisted_chat_ids()),
        "blocked_reason":       _why_we_should_not_start(),
        "last_dropped_chat_id": _last_dropped_chat_id,
        "in_flight_chats":      sorted(_running_chats.keys()),
    }


async def reload_config() -> None:
    """Re-evaluate whether the polling loop should be running. Called
    when Settings → Telegram inbound is toggled at runtime so the user
    doesn't need to restart the server.

    Currently a stub — Phase 3 wires this into the settings POST
    handler. For now, settings changes take effect on next server
    restart."""
    # PHASE 3: implement as:
    #   reason = _why_we_should_not_start()
    #   if reason and _task and not _task.done(): await stop()
    #   elif not reason and (not _task or _task.done()):
    #       await start(_tracker, _orchestrator)
    logger.debug("telegram_bridge.reload_config() called — stub; restart server to apply")


# ── Config inspection helpers ─────────────────────────────────────────────────

def _why_we_should_not_start() -> Optional[str]:
    """Return a human-readable reason to skip starting, or None to
    proceed. Lets the startup log line spell out exactly which knob
    the user needs to flip — friendlier than a silent no-op."""
    if not (getattr(config, "telegram_inbound_enabled", False)):
        return "inbound disabled in Settings"
    token = (getattr(config, "telegram_bot_token", "") or "").strip()
    if not token:
        return "telegram_bot_token not set"
    whitelist = _whitelisted_chat_ids()
    if not whitelist:
        return (
            "chat_id whitelist is empty — refusing to run on an open bot "
            "(set Settings → Telegram inbound → Whitelist)"
        )
    return None


def _whitelisted_chat_ids() -> Set[str]:
    """Parse the comma-separated whitelist into a set of stripped
    strings. Accepts numeric ids and `@username` style for channels.
    Empty entries are dropped silently."""
    raw = getattr(config, "telegram_inbound_chat_id_whitelist", "") or ""
    return {p.strip() for p in raw.split(",") if p.strip()}


def _bot_url(path: str) -> str:
    token = (getattr(config, "telegram_bot_token", "") or "").strip()
    return _API_BASE.format(token=token) + "/" + path


# ── Human-friendly error translation ──────────────────────────────────────────

def friendly_telegram_error(description: str) -> str:
    """Translate Telegram's terse API error into an actionable, multi-
    line explanation aimed at someone setting up a bot for the first
    time. Returns the input unchanged if nothing matches — so a
    surprising error never gets silently lost; the raw text is always
    present at the end as `Raw: ...` for support / copy-paste.

    The patterns covered are the ones we've seen real users hit on
    first-time setup. Add more here as new failure shapes show up.
    """
    if not description:
        return "Telegram returned an empty error description."

    lower = description.lower()

    # The classic one: bot can't send to a chat that has never written
    # to it first. Standard fix is 3 steps — spell them out so the user
    # doesn't need to leave the app to figure it out.
    if "chat not found" in lower or "can't initiate conversation" in lower:
        return (
            "Telegram says 'chat not found' — usually this means your "
            "Telegram account has never started a conversation with this "
            "bot, so Telegram refuses to let the bot DM you out of the "
            "blue.\n"
            "\n"
            "Fix (30 seconds):\n"
            "  1. Open Telegram, search for your bot's @username "
            "(the one @BotFather gave you).\n"
            "  2. Tap Start (or send /start).\n"
            "  3. Come back here and click Test ping again.\n"
            "\n"
            f"Raw: {description}"
        )

    # Bad token — the 401 path usually surfaces here too, but if Telegram
    # sends it as a 400 with this phrase, catch it.
    if "unauthorized" in lower or "401" in lower:
        return (
            "Telegram says the bot token is invalid. Double-check you "
            "copied the whole token from @BotFather (it includes the "
            "colon and is ~46 chars). Regenerate with /token in "
            "@BotFather if you suspect the old one leaked.\n"
            "\n"
            f"Raw: {description}"
        )

    # User blocked the bot — happens after a previous setup attempt
    # where the user blocked instead of pressing Start.
    if "blocked" in lower:
        return (
            "Telegram says the user has blocked this bot. Open the bot's "
            "chat in Telegram, tap the menu (top right), and choose "
            "Unblock. Then click Test ping again.\n"
            "\n"
            f"Raw: {description}"
        )

    # Group chat without the bot being a member — common for the "send
    # the report to my team channel" use case.
    if ("bot is not a member" in lower
            or "need administrator rights" in lower
            or ("chat not found" in lower and description.lstrip("-").lstrip().startswith("-"))):
        return (
            "Telegram says the bot can't reach this group/channel. "
            "Either:\n"
            "  - Add the bot to the group (Group settings → Add members "
            "→ search the bot's @username), or\n"
            "  - For channels, make it an admin so it can post.\n"
            "\n"
            f"Raw: {description}"
        )

    # chat_id format / empty — usually means the user pasted a username
    # without the @ prefix, or left a stray character.
    if "chat_id is empty" in lower or "chat_id" in lower and "invalid" in lower:
        return (
            "Telegram says the chat_id is malformed. For a personal chat "
            "this should be a positive integer (e.g. 987654321). For a "
            "group it's a negative integer (e.g. -1001234567890). For a "
            "public channel it can be '@channelname'. Find your numeric "
            "chat_id by sending any message to @userinfobot.\n"
            "\n"
            f"Raw: {description}"
        )

    # Fall-through — surface the raw error verbatim so the user can
    # google it / paste it into a bug report.
    return f"Telegram returned: {description}"


# ── Polling loop ──────────────────────────────────────────────────────────────

async def _poll_loop() -> None:
    """Long-poll getUpdates forever (until cancelled). Each successful
    poll batch acks the highest-seen update_id via the `offset` param
    so Telegram garbage-collects delivered updates server-side."""
    offset: Optional[int] = None
    backoff_idx = 0

    while True:
        try:
            updates = await _fetch_updates(offset)
        except asyncio.CancelledError:
            raise
        except _FatalAuthError as exc:
            # 401 — bot token is bad. No amount of backoff fixes this.
            # Log loudly and stop the loop; user has to fix Settings.
            logger.error(
                "Telegram bridge: bot token rejected (%s). "
                "Fix Settings → Telegram Bot Token and restart.",
                exc,
            )
            return
        except _FatalConflictError as exc:
            # 409 — Telegram returns this when a webhook is already set
            # on the bot. Two pollers (or webhook + poller) can't
            # co-exist. Surface clearly and give up.
            logger.error(
                "Telegram bridge: conflict (%s). The bot already has a "
                "webhook set, or another OpenTeddy instance is polling. "
                "Stop the other process or delete the webhook before "
                "enabling inbound here.",
                exc,
            )
            return
        except Exception as exc:  # noqa: BLE001
            # Transient — back off and retry. Log at warning so it's
            # visible without being noisy (we expect occasional
            # network blips on long-lived connections).
            wait = _BACKOFF_S[min(backoff_idx, len(_BACKOFF_S) - 1)]
            logger.warning(
                "Telegram bridge poll failed: %s — backing off %ds",
                exc, wait,
            )
            backoff_idx += 1
            await asyncio.sleep(wait)
            continue

        # Success path — reset backoff and handle any updates.
        backoff_idx = 0
        for update in updates:
            try:
                await _dispatch(update)
            except Exception as exc:  # noqa: BLE001
                # Handler errors must never kill the poll loop. Log and
                # move on; the message will be lost but Telegram has
                # already acked it via the offset bump.
                logger.exception(
                    "Telegram bridge handler raised on update %s: %s",
                    update.get("update_id"), exc,
                )
            uid = update.get("update_id")
            if uid is not None:
                offset = uid + 1


# ── Telegram API I/O ──────────────────────────────────────────────────────────

class _FatalAuthError(Exception):
    """Raised on Telegram 401 — non-recoverable until user updates token."""


class _FatalConflictError(Exception):
    """Raised on Telegram 409 — webhook collision, can't be auto-fixed."""


async def _fetch_updates(offset: Optional[int]) -> List[Dict[str, Any]]:
    """One getUpdates round-trip. Returns the (possibly empty) list of
    update objects. Long-polls server-side for up to _POLL_TIMEOUT_S
    seconds."""
    params: Dict[str, Any] = {"timeout": _POLL_TIMEOUT_S}
    if offset is not None:
        params["offset"] = offset

    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
        resp = await client.get(_bot_url("getUpdates"), params=params)

    if resp.status_code == 401:
        raise _FatalAuthError(resp.text[:200])
    if resp.status_code == 409:
        raise _FatalConflictError(resp.text[:200])
    resp.raise_for_status()

    payload = resp.json()
    if not payload.get("ok"):
        # Telegram conveys logical errors with HTTP 200 + ok:false.
        raise RuntimeError(
            f"getUpdates returned ok=false: {payload.get('description')!r}"
        )
    return payload.get("result") or []


async def _send_reply(chat_id: str, text: str) -> None:
    """Best-effort outbound reply. Failures are logged, not raised —
    the dispatcher must keep processing whatever else came in this
    batch even if Telegram's outbound side is hiccuping."""
    if not text:
        return
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                _bot_url("sendMessage"),
                json={"chat_id": chat_id, "text": text},
            )
        if resp.status_code != 200:
            logger.warning(
                "Telegram sendMessage to %s returned %d: %s",
                chat_id, resp.status_code, resp.text[:200],
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telegram sendMessage to %s failed: %s", chat_id, exc)


# ── Native "thinking" indicator ───────────────────────────────────────────────
# Telegram clients show "Bot is typing…" for ~5 s after the bot calls
# sendChatAction. The indicator is much lighter-weight UX than a static
# text ack — a chat-mode question that finishes in 2 s shouldn't have
# its inbox cluttered by a separate "got it, this may take minutes"
# message. We refresh every ~4 s so the indicator stays alive through
# longer runs, and ONLY fall back to a textual "still working" hint if
# the task is actually long.

# Refresh interval — just under Telegram's ~5 s indicator timeout so
# the dots don't visibly blink off and on between sendChatAction calls.
_TYPING_REFRESH_S = 4

# After this many seconds without a result, send a one-shot text
# "still working" message so the user knows the bot hasn't silently
# crashed. ~5 s is the inflection point where the typing dots stop
# feeling "fast" and start feeling "stuck".
_LONG_TASK_HINT_S = 5

# Hard upper bound on a single Telegram-initiated task. If the
# orchestrator's subtask_timeout misbehaves, or the local model goes
# unresponsive (Ollama OOM, model unload mid-call), or some tool
# deadlocks waiting on an external resource — without this cap, the
# task would hold the chat's concurrency lock indefinitely and the
# user just sees a frozen bot. Caught by `asyncio.wait_for` below.
#
# 10 min is generous for any single Telegram-driven goal (the chat
# UX assumes a "send + wait" rhythm; nobody wants to babysit a 30 min
# task this way). Bumped here, not in Settings, because the right
# number depends on the Telegram UX, not on the user's preference —
# users genuinely waiting 10 min for a Telegram reply have a UX
# problem regardless of whether the underlying task could complete.
_TELEGRAM_RUN_TIMEOUT_S = 600


async def _send_chat_action(chat_id: str, action: str = "typing") -> None:
    """Fire-and-forget sendChatAction. Telegram-side this surfaces the
    native typing indicator. Errors are swallowed: an indicator failure
    must never derail the actual task reply."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                _bot_url("sendChatAction"),
                json={"chat_id": chat_id, "action": action},
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Telegram sendChatAction to %s failed: %s", chat_id, exc)


async def _keep_typing(chat_id: str) -> None:
    """Background loop that keeps the typing indicator alive for the
    duration of a task. Cancelled by the caller (via task.cancel())
    when the orchestrator returns."""
    try:
        while True:
            await _send_chat_action(chat_id, "typing")
            await asyncio.sleep(_TYPING_REFRESH_S)
    except asyncio.CancelledError:
        # Clean exit — caller cancelled us because the task finished.
        raise


async def _send_message_get_id(chat_id: str, text: str) -> Optional[int]:
    """sendMessage that returns the new message_id on success, None on
    failure. Used by progressive-update flows that want to EDIT this
    message later (editMessageText needs the id).

    Bypasses the existing `_send_reply` because that wrapper
    intentionally swallows the response — for fire-and-forget replies
    we don't care about the id, but for the progress message we do."""
    if not text:
        return None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                _bot_url("sendMessage"),
                json={"chat_id": chat_id, "text": text},
            )
        if resp.status_code == 200:
            payload = resp.json()
            if payload.get("ok"):
                return (payload.get("result") or {}).get("message_id")
        logger.debug(
            "_send_message_get_id non-OK for %s: %d %s",
            chat_id, resp.status_code, resp.text[:200],
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("_send_message_get_id error for %s: %s", chat_id, exc)
    return None


async def _edit_message(chat_id: str, message_id: int, text: str) -> None:
    """Best-effort editMessageText. Silently ignores rate-limit 429s
    (Telegram caps edits at ~1/s per chat; next progress event will
    catch up). Any other failure logs at debug and moves on — a
    stuck edit must never break the actual task pipeline."""
    if not message_id or not text:
        return
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                _bot_url("editMessageText"),
                json={
                    "chat_id":    chat_id,
                    "message_id": message_id,
                    "text":       text,
                },
            )
        if resp.status_code not in (200, 400, 429):
            # 400 = "message is not modified" (text unchanged — fine);
            # 429 = rate-limited (will catch up next round).
            logger.debug(
                "editMessageText non-OK for %s/%s: %d %s",
                chat_id, message_id, resp.status_code, resp.text[:200],
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("_edit_message error for %s/%s: %s",
                     chat_id, message_id, exc)


# ── Live progress relay ───────────────────────────────────────────────────────
# When the orchestrator runs a multi-subtask task, the executor
# broadcasts `subtask.progress` events through ws_manager. The web UI
# turns them into a live pill ("3/5 · 🎯 0.85 · 💰 $0.04"). For
# Telegram-driven tasks we mirror the same signal into a single message
# that gets edited in place — gives the user the same "I can see it
# working" feedback without spamming the chat with one bubble per
# subtask.

# Throttle. Telegram allows ~1 edit/sec to the same message; we leave
# headroom so a burst of fast subtasks doesn't trigger 429s.
_PROGRESS_EDIT_MIN_INTERVAL_S = 1.5


class _ProgressState:
    """Per-task state for the editable progress message in Telegram.

    One instance per Telegram-driven orchestrator run. Holds the
    message_id we're editing (None until first written), throttling
    state, and a lock so concurrent fire-and-forget edits don't race.
    """

    __slots__ = (
        "chat_id", "task_id", "short_goal",
        "message_id", "last_edit_at", "lock",
    )

    def __init__(self, chat_id: str, task_id: str, short_goal: str):
        self.chat_id    = chat_id
        self.task_id    = task_id
        self.short_goal = short_goal
        self.message_id: Optional[int] = None
        self.last_edit_at: float = 0.0
        self.lock = asyncio.Lock()


async def _write_progress(state: "_ProgressState", text: str) -> None:
    """Send-or-edit the progress message. Thread-safe via state.lock so
    a fast burst (long-task hint timer firing at the same time as the
    first subtask.progress event) can't double-send and end up with two
    separate threads claiming the same slot."""
    async with state.lock:
        if state.message_id is None:
            state.message_id = await _send_message_get_id(state.chat_id, text)
        else:
            await _edit_message(state.chat_id, state.message_id, text)
        state.last_edit_at = time.monotonic()


async def _long_task_hint(state: "_ProgressState", hint_task_ref=None) -> None:
    """If no progress event has populated the progress message within
    _LONG_TASK_HINT_S, send a generic 'still working' line so the chat
    doesn't look frozen. For chat-mode fast answers (which never emit
    subtask.progress events) this becomes the fallback signal.

    Cancelled by the progress listener as soon as a real subtask.
    progress event fires — that event's "Subtask 2/4 · 🎯 0.85 · 💰…"
    text is strictly more useful than the generic hint, and we don't
    want a 0.5s-later hint to clobber it."""
    try:
        await asyncio.sleep(_LONG_TASK_HINT_S)
        # If a listener already wrote to the progress message, skip the
        # hint — its generic wording would be a downgrade.
        if state.message_id is not None:
            return
        await _write_progress(
            state,
            f"🐻 Still working on: {state.short_goal}\n"
            "(local model is planning — first subtask should appear soon)",
        )
    except asyncio.CancelledError:
        # Cancellation is the happy path for fast tasks. No message
        # was sent; that's fine.
        raise


def _make_progress_listener(
    state: "_ProgressState", hint_task: "Optional[asyncio.Task]" = None,
):
    """Build a ws_manager listener that watches for subtask.progress
    events on the bound task and pushes them into the editable progress
    message. The listener also cancels the long-task hint (since the
    real progress info is now driving the message) so we don't get a
    generic "planning..." line clobbering a more specific "Subtask 3/5"
    update."""
    async def listener(event: Dict[str, Any]) -> None:
        # Filter to our task only — many tasks can run concurrently
        # across the server.
        if event.get("task_id") != state.task_id:
            return
        # We only care about subtask.progress for the editable
        # message. Other event types (tool_call, tool_result) fly past.
        if event.get("type") != "subtask.progress":
            return

        # Once we have real progress, the long-task hint becomes
        # redundant — its generic text would only clobber the more
        # informative subtask info.
        if hint_task is not None and not hint_task.done():
            hint_task.cancel()

        # Throttle. The first event always fires; subsequent edits
        # respect the Telegram rate-limit window.
        now = time.monotonic()
        if state.message_id is not None and (now - state.last_edit_at) < _PROGRESS_EDIT_MIN_INTERVAL_S:
            return

        order = int(event.get("order") or 0)
        total = int(event.get("total") or 0)
        confidence = float(event.get("confidence") or 0.0)
        cost = float(event.get("cost_usd") or 0.0)
        tokens_out = int(event.get("tokens_out") or 0)

        # Compose. Keep it dense — Telegram users skim. Confidence /
        # cost only shown when meaningful (skips the "$0.000" noise
        # for local-only chats).
        bits = [f"⚙️ Subtask {order}/{total}" if total else f"⚙️ Subtask {order}"]
        if confidence > 0:
            bits.append(f"🎯 {confidence:.2f}")
        if cost > 0:
            bits.append(f"💰 ${cost:.3f}")
        elif tokens_out > 0:
            bits.append(f"🔤 {tokens_out:,} tok")

        text = (
            f"🐻 Working on: {state.short_goal}\n"
            f"\n"
            f"{' · '.join(bits)}"
        )
        await _write_progress(state, text)
    return listener


# ── Update dispatch ───────────────────────────────────────────────────────────

async def _dispatch(update: Dict[str, Any]) -> None:
    """Route a single Telegram update. Handles whitelist auth, command
    parsing, and (Phase 2) hand-off to the orchestrator. Today this
    only handles /start, /help, and echoes other text so you can verify
    connectivity end-to-end before wiring the orchestrator."""
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        # Updates also cover callback_query, inline_query, etc. — we
        # don't expose any of those today.
        return

    chat = msg.get("chat") or {}
    chat_id_raw = chat.get("id")
    if chat_id_raw is None:
        return
    chat_id = str(chat_id_raw)

    # ── Whitelist gate ────────────────────────────────────────────────────────
    # Also accept @username form for public chats. Telegram surfaces it
    # in chat.username; build the candidate set first, then intersect.
    candidates = {chat_id}
    if chat.get("username"):
        candidates.add("@" + chat["username"])
    if not (candidates & _whitelisted_chat_ids()):
        # Bump to INFO so a user setting up inbound for the first time
        # can confirm "yes, the bot is receiving my message; no, my
        # chat_id isn't on the whitelist" in one log line. Also stash
        # the most-recent dropped id on the module so /admin/telegram/
        # status can show it without server-log access. Security note:
        # we still don't *reply* to non-whitelisted chats — the
        # behaviour shift is purely about local observability for the
        # owner, not about exposing anything to the probing chat.
        global _last_dropped_chat_id
        _last_dropped_chat_id = chat_id
        logger.info(
            "Telegram bridge: ignored chat_id %s (not in whitelist %s)",
            chat_id, sorted(_whitelisted_chat_ids()),
        )
        return

    text = (msg.get("text") or "").strip()
    if not text:
        # Photos, stickers, voice notes — out of scope for now. A
        # cheerful nudge is friendlier than silent ignore.
        await _send_reply(
            chat_id,
            "📎 Only text messages are supported for now. Try sending some words.",
        )
        return

    # ── Commands ──────────────────────────────────────────────────────────────
    if text.startswith("/"):
        await _handle_command(chat_id, text)
        return

    # ── Natural-language schedule cancel — must run BEFORE the
    # orchestrator-dispatch branch so a message like "取消我的排程" doesn't
    # get treated as a new goal and burn LLM tokens trying to interpret
    # it. Returns True iff it handled the message; False falls through.
    try:
        if await _maybe_handle_natural_cancel(chat_id, text):
            return
    except Exception as exc:  # noqa: BLE001
        logger.debug("natural-cancel guard raised, falling through: %s", exc)

    # ── Free-text → goal → orchestrator dispatch ──────────────────────────────
    # Concurrency guard: refuse new messages while this chat already has
    # an orchestrator run in flight. Queueing is intentionally not done —
    # if the user sent "fix bug X" 30s ago and is now typing "actually do
    # Y instead", we want them to know the first one is still chewing,
    # not silently stack them up.
    in_flight = _running_chats.get(chat_id)
    if in_flight and not in_flight.done():
        await _send_reply(
            chat_id,
            "⏳ A task is still running for this chat. Wait for it to "
            "finish, or send /cancel to abort it (Phase 3).",
        )
        return

    # Spawn the run as an asyncio.Task so we can register it for the
    # concurrency guard above + (Phase 3) /cancel from Telegram. The
    # outer try/except in _dispatch() still wraps any sync part of this,
    # and the task body itself owns its own error handling so a crashed
    # run doesn't poison _running_chats.
    task = asyncio.create_task(
        _run_goal_for_chat(chat_id, text),
        name=f"telegram_goal:{chat_id[:16]}",
    )
    _running_chats[chat_id] = task


async def _run_goal_for_chat(chat_id: str, goal_text: str) -> None:
    """Run one Telegram-initiated goal end-to-end:
      1. Resolve or create the persistent session bound to this chat
      2. Send an immediate "got it, working" ack so the user knows the
         message was received (orchestrator runs can take minutes)
      3. Build a TaskRequest and await orchestrator.run()
      4. Format the result for Telegram and send it back
      5. Clean up the _running_chats entry no matter what

    Errors at any step are caught + surfaced back to the user so a
    flaky run doesn't leave them staring at a silent chat. The poll
    loop never sees an exception from here."""
    start = time.monotonic()
    try:
        session_id = await _resolve_or_create_session(chat_id, goal_text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Telegram bridge: session resolution failed for chat=%s", chat_id)
        await _send_reply(
            chat_id,
            f"❌ Could not resolve a session for this chat: {exc}",
        )
        _running_chats.pop(chat_id, None)
        return

    # ── Natural-language scheduling shortcut ──────────────────────────────
    # Before the typing indicator + orchestrator dance, check whether the
    # user is asking for a recurring task ("每天早上 9 點 ...", "every
    # weekday at 18:00 ..."). The regex pre-screen makes this cheap on
    # the 99 %+ of messages that aren't schedules; only matches pay the
    # LLM round-trip. When a schedule is detected we persist it via
    # scheduler.add_schedule() and reply with the confirmation — no
    # orchestrator run for this turn. Failures here fall through to the
    # normal planner path so a flaky classifier never silences the bot.
    try:
        from scheduling_intent import detect_scheduling_intent
        from scheduler import add_schedule
        intent = await detect_scheduling_intent(goal_text)
        if intent is not None:
            row = await add_schedule(
                session_id=session_id, cron=intent.cron, goal=intent.task_goal,
            )
            short_id = (row.get("id") or "")[:8]
            next_at = (row.get("next_run_at") or "").replace("T", " ")[:16]
            await _send_reply(
                chat_id,
                f"⏰ 已排好：{intent.summary}\n"
                f"任務：{intent.task_goal}\n"
                f"下次執行：{next_at or '(scheduler 計算中)'} · id: `{short_id}`\n"
                f"取消請說「取消那個排程」或 `/cron cancel {short_id}`",
            )
            _running_chats.pop(chat_id, None)
            return
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Telegram: scheduling_intent detection failed for chat=%s, "
            "falling through: %s", chat_id, exc,
        )

    # Fire the native "typing…" indicator immediately and keep it alive
    # in the background. The previous version sent a static text ack
    # ("🐻 Got it. Working on: ... could take minutes") immediately —
    # visually mismatched for chat-mode questions that finish in 2-3 s.
    # With the indicator + editable progress message approach: fast
    # answers feel native ("typing…" → answer) and longer tasks get a
    # single "🐻 Subtask 3/5 · 🎯 0.85 · 💰 $0.04" message that edits
    # in place as the orchestrator works through its plan.
    short_goal = goal_text if len(goal_text) <= 80 else goal_text[:77] + "…"

    # Build TaskRequest first so the progress listener can filter by
    # task_id before the orchestrator starts emitting events. (Build
    # is cheap — just a Pydantic model instance.)
    try:
        from models import TaskRequest, SessionMode
        req = TaskRequest(
            id=str(uuid.uuid4()),
            goal=goal_text,
            # Mark the origin so logs / future analytics can break down
            # "where do my tasks come from". Surfaced in tasks.context.
            context={
                "triggered_by":      "telegram",
                "telegram_chat_id":  chat_id,
            },
            priority=1,
            session_id=session_id,
            mode=SessionMode.CODE,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Telegram bridge: failed to build TaskRequest")
        await _send_reply(chat_id, f"❌ Could not start task: {exc}")
        _running_chats.pop(chat_id, None)
        return

    # Live-progress state — shared between the long-task hint (sends
    # initial fallback message) and the ws_manager listener (edits the
    # same message with real subtask info). Whoever fires first writes
    # the message; the other side edits.
    progress_state = _ProgressState(
        chat_id=chat_id, task_id=req.id, short_goal=short_goal,
    )

    typing_task = asyncio.create_task(
        _keep_typing(chat_id), name=f"tg_typing:{chat_id[:16]}",
    )
    hint_task = asyncio.create_task(
        _long_task_hint(progress_state),
        name=f"tg_hint:{chat_id[:16]}",
    )

    # Subscribe to ws_manager so subtask.progress events filtered to
    # this task_id flow into the progress message. Late import (and
    # try/except) keeps this best-effort — if ws_manager is somehow
    # unavailable we still complete the task, just without live
    # progress feedback.
    ws_listener_token: Any = None
    try:
        from main import ws_manager as _wsm
        ws_listener_token = _wsm.subscribe(
            _make_progress_listener(progress_state, hint_task)
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Telegram bridge: ws_manager subscribe failed: %s", exc)

    # Run the orchestrator. Bind triggered_by so tool_registry can
    # apply the auto-approve + denylist policy. Wrap with wait_for so
    # a hung run doesn't freeze the chat. Reset / unsubscribe happen
    # in the finally to guarantee no state leaks across tasks.
    from tools._context import set_triggered_by, reset_triggered_by
    origin_token = set_triggered_by("telegram")
    try:
        try:
            result = await asyncio.wait_for(
                _orchestrator.run(req), timeout=_TELEGRAM_RUN_TIMEOUT_S,
            )
        finally:
            reset_triggered_by(origin_token)
            if ws_listener_token is not None:
                try:
                    from main import ws_manager as _wsm
                    _wsm.unsubscribe(ws_listener_token)
                except Exception:  # noqa: BLE001
                    pass
    except asyncio.TimeoutError:
        # Hard cap hit — log + tell the user clearly, don't leave them
        # staring at the typing dots forever. wait_for already cancelled
        # the inner coroutine on its way out, but it's worth noting that
        # the cancellation might still take a few seconds to actually
        # unwind through any in-flight HTTP call.
        logger.warning(
            "Telegram bridge: orchestrator.run() exceeded %ds for "
            "chat=%s session=%s — aborted.",
            _TELEGRAM_RUN_TIMEOUT_S, chat_id, session_id,
        )
        _cancel_indicator_tasks(typing_task, hint_task)
        await _send_reply(
            chat_id,
            f"⌛ Task ran longer than {_TELEGRAM_RUN_TIMEOUT_S // 60} min "
            "and was force-cancelled.\n"
            "Common causes: local model (Ollama) unresponsive, tool "
            "deadlock, or a plan loop. Try a simpler goal, or /new "
            "for a fresh session.",
        )
        _running_chats.pop(chat_id, None)
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "Telegram bridge: orchestrator.run() raised for chat=%s session=%s",
            chat_id, session_id,
        )
        # Stop the indicator / hint before surfacing the error so a slow
        # crash doesn't keep flashing "typing…" while the failure reply
        # is being read.
        _cancel_indicator_tasks(typing_task, hint_task)
        await _send_reply(
            chat_id,
            f"❌ Task crashed inside the orchestrator: {exc}\n"
            "Check the server logs for the full traceback.",
        )
        _running_chats.pop(chat_id, None)
        return

    # Happy path: cancel the indicator + hint so a fast answer doesn't
    # also get the "still working" line. For sub-_LONG_TASK_HINT_S runs
    # the hint never fired; for longer runs it fired once and that's
    # the right number of times.
    _cancel_indicator_tasks(typing_task, hint_task)

    elapsed_s = time.monotonic() - start
    reply = _format_result_for_telegram(result, session_id, elapsed_s)

    # Pull the persisted artifact list off the task row so the reply
    # can list "📎 github_trending_results.txt (1.2 KB)" and we can
    # also push small text files inline as follow-up messages. The
    # tracker's tasks.artifacts JSON column is the source of truth —
    # populated by the executor's producer-tool path AND (after this
    # commit) the post-subtask workspace scanner for shell-redirect
    # outputs.
    artifacts: List[Dict[str, Any]] = []
    try:
        task_row = await _tracker.get_task(req.id)
        if task_row:
            artifacts = task_row.get("artifacts") or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("artifact lookup failed for task %s: %s", req.id, exc)

    if artifacts:
        reply = reply + _format_artifacts_block(artifacts)

    await _send_reply(chat_id, reply)

    # For small text artifacts, push the file CONTENT as follow-up
    # messages so the user doesn't have to open the web UI to see
    # what was produced. Larger / binary files go via sendDocument so
    # the user can tap-to-download from inside Telegram.
    if artifacts:
        await _push_artifact_contents(chat_id, artifacts)

    _running_chats.pop(chat_id, None)


# ── Artifact rendering ────────────────────────────────────────────────────────
# Telegram bot file/media limits:
#   - sendMessage:  max 4096 chars per message (we cap at ~3500)
#   - sendDocument: up to 50 MB per file via standard Bot API
# Trade-off: inline message is one round-trip and immediately readable;
# sendDocument shows up as a "file attachment" card and is more
# discoverable but adds a click. We inline small text (< ~3 KB), send
# everything else as a document.

_INLINE_TEXT_SIZE_LIMIT = 3000   # bytes; below this we send content inline
_SEND_DOCUMENT_SIZE_LIMIT = 50 * 1024 * 1024  # 50 MB Telegram Bot API cap

# Tunable. Files we treat as text for inline display. Other extensions
# go through sendDocument (so the user gets a tap-to-download chip
# rather than a binary garbled into a message).
_TEXT_EXTENSIONS: frozenset = frozenset({
    ".txt", ".md", ".markdown", ".log", ".json", ".csv", ".tsv",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".rb", ".sh",
    ".bash", ".zsh", ".html", ".xml", ".css", ".scss", ".sql",
    ".java", ".kt", ".swift", ".c", ".h", ".cpp", ".hpp", ".cs",
})


def _format_artifacts_block(artifacts: List[Dict[str, Any]]) -> str:
    """Render the artifact list as a footer block appended to the
    main result reply. Compact: one line per artifact with name +
    human-readable size. Truncates to 10 entries because (a) Telegram
    has a 4096-char message cap and (b) a task producing >10 files is
    almost always a bug worth surfacing differently."""
    if not artifacts:
        return ""
    lines = ["", "📎 Files produced:"]
    for a in artifacts[:10]:
        name = a.get("name") or "(unnamed)"
        size = int(a.get("size_bytes") or 0)
        if size >= 1024 * 1024:
            size_str = f"{size / (1024 * 1024):.1f} MB"
        elif size >= 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        tool = a.get("tool") or "?"
        lines.append(f"  • {name} ({size_str}, via {tool})")
    if len(artifacts) > 10:
        lines.append(f"  …and {len(artifacts) - 10} more")
    return "\n".join(lines)


def _looks_like_text_file(path: str) -> bool:
    """Heuristic: file is "text-like" iff its extension matches one we
    inline-display. Not a perfect check (.dat could be text, .json could
    be a binary export) but covers the 95 % case the user actually
    produces — scrape outputs, scripts, configs, CSVs."""
    import os as _os
    _, ext = _os.path.splitext(path.lower())
    return ext in _TEXT_EXTENSIONS


async def _push_artifact_contents(
    chat_id: str, artifacts: List[Dict[str, Any]],
) -> None:
    """For each artifact: inline-display content if it's a small text
    file, otherwise send via sendDocument as a downloadable file.

    Silently skips files that don't exist on disk (could happen if the
    artifact path is stale — e.g. a session was deleted between the
    tool call and our scan). Per-file errors don't propagate; one
    unreadable file mustn't suppress the others."""
    import os as _os
    for a in artifacts[:10]:  # same 10-cap as the summary block
        fpath = a.get("path") or ""
        if not fpath or not _os.path.isfile(fpath):
            continue
        size = a.get("size_bytes") or 0
        name = a.get("name") or _os.path.basename(fpath)

        try:
            if size < _INLINE_TEXT_SIZE_LIMIT and _looks_like_text_file(fpath):
                # Inline display — read + format as a code-block-ish
                # message. Cap at ~3 KB so even a worst-case binary
                # masquerading as .txt can't blow the 4096 limit.
                with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read(_INLINE_TEXT_SIZE_LIMIT)
                preview = f"📄 {name}\n\n{content}"
                if len(preview) > 3800:
                    preview = preview[:3800] + "\n…(truncated)"
                await _send_reply(chat_id, preview)
            elif size <= _SEND_DOCUMENT_SIZE_LIMIT:
                # sendDocument — Telegram-native file attachment. Tap
                # to download / preview on phone.
                await _send_document(chat_id, fpath, caption=name)
            # Files larger than 50 MB get only the summary entry, no
            # content delivery — Telegram bot API rejects them and the
            # user can grab the file from the workspace directly.
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "artifact delivery failed for %s: %s", fpath, exc,
            )


async def _send_document(chat_id: str, file_path: str, caption: str = "") -> None:
    """Telegram sendDocument — uploads a file the user can tap to
    download. multipart/form-data because that's what the Telegram
    Bot API wants for binary uploads. Best-effort: any failure is
    logged at debug and ignored, so a flaky upload doesn't break
    the rest of the run."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            with open(file_path, "rb") as fh:
                files = {"document": (file_path.rsplit("/", 1)[-1], fh)}
                data = {"chat_id": chat_id}
                if caption:
                    data["caption"] = caption[:1024]  # Telegram caption cap
                resp = await client.post(
                    _bot_url("sendDocument"), data=data, files=files,
                )
        if resp.status_code != 200:
            logger.debug(
                "sendDocument non-OK for %s/%s: %d %s",
                chat_id, file_path, resp.status_code, resp.text[:200],
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("sendDocument error for %s/%s: %s",
                     chat_id, file_path, exc)


def _cancel_indicator_tasks(*tasks: asyncio.Task) -> None:
    """Cancel the typing-keep-alive + long-task-hint background tasks
    safely. Cancelling an already-finished task is a no-op; cancelling
    one that's mid-await produces the asyncio.CancelledError that those
    coroutines themselves let propagate. Both behaviours are fine here."""
    for t in tasks:
        if t and not t.done():
            t.cancel()


async def _resolve_or_create_session(chat_id: str, first_goal: str) -> str:
    """Find the session bound to this Telegram chat, or create + bind a
    new one. "One chat = one persistent agent" — the user's chat history
    in Telegram naturally mirrors the session's task history in OpenTeddy,
    so follow-ups continue with the same memory / workspace / mode.

    First-time-from-this-chat: we create a Session row titled with a
    short slice of the first goal so it shows up legibly in the
    sessions list ("Telegram: fix the deploy script" rather than "New
    session"). Subsequent calls just look up the binding."""
    existing = await _tracker.get_session_by_telegram_chat(chat_id)
    if existing:
        return existing["id"]

    # First message — provision a fresh session bound to this chat.
    session_id = str(uuid.uuid4())
    title = ("Telegram: " + first_goal.strip().splitlines()[0])[:60] or f"Telegram chat {chat_id}"
    await _tracker.create_session(session_id, title, mode="code")
    await _tracker.bind_session_to_telegram_chat(session_id, chat_id)
    logger.info(
        "Telegram bridge: provisioned new session %s for chat %s (%r)",
        session_id, chat_id, title,
    )
    return session_id


def _format_result_for_telegram(result: Any, session_id: str, elapsed_s: float) -> str:
    """Render a TaskResult into a Telegram-readable message. Aggressive
    on length (cap at _TELEGRAM_MAX_BODY_CHARS) since Telegram clamps
    messages at 4096 chars total and we want headroom for trailer
    metadata. Truncated summaries pointer the user at the web UI for
    the full text — exporting the session via the kebab menu is the
    "see everything" escape hatch."""
    # Late import — avoids pulling Pydantic models into module load.
    try:
        from models import TaskStatus
        status_val = result.status.value if hasattr(result.status, "value") else str(result.status)
    except Exception:  # noqa: BLE001
        status_val = str(getattr(result, "status", "unknown"))

    # Pick the right status emoji. Anything we haven't explicitly seen
    # falls through to a neutral marker rather than a misleading ✅/❌.
    icon = {
        "completed": "✅",
        "failed":    "❌",
        "escalated": "⚠️",
        "running":   "⏳",
    }.get(status_val.lower(), "ℹ️")

    summary = (getattr(result, "summary", "") or "").strip() or "(no summary)"
    truncated = False
    if len(summary) > _TELEGRAM_MAX_BODY_CHARS:
        summary = summary[:_TELEGRAM_MAX_BODY_CHARS]
        truncated = True

    subtask_count = len(getattr(result, "subtasks", []) or [])

    parts = [
        f"{icon} {status_val.title()} · {elapsed_s:.1f}s · {subtask_count} subtask"
        + ("s" if subtask_count != 1 else ""),
        "",
        summary,
    ]
    if truncated:
        parts.append(
            "\n…(truncated — open the session in the web UI or use the "
            "kebab menu's Export button for the full output)"
        )
    parts.append("")
    parts.append(f"session: {session_id}")
    return "\n".join(parts)


async def _handle_command(chat_id: str, text: str) -> None:
    """Static command dispatch. Phase 1 only knows /start and /help —
    /cancel and /new come in Phase 3."""
    cmd = text.split()[0].lower()
    # Strip @botname suffix Telegram appends in group chats
    # (/help@OpenTeddyBot → /help) so commands work regardless of
    # whether the user @-mentioned the bot.
    if "@" in cmd:
        cmd = cmd.split("@", 1)[0]

    if cmd == "/start":
        await _send_reply(
            chat_id,
            "🐻 Hi from OpenTeddy. You're on the whitelist — send any "
            "message and I'll treat it as a goal. Use /help for the "
            "command list.",
        )
        return

    if cmd == "/help":
        await _send_reply(
            chat_id,
            "OpenTeddy Telegram bridge\n"
            "\n"
            "Send any text message → I treat it as a goal and run it "
            "in a session bound to this chat.\n"
            "\n"
            "Commands:\n"
            "/start  — confirm you're connected\n"
            "/help   — this message\n"
            "/cancel — abort the currently-running task\n"
            "/new    — start a fresh session for this chat (old one stays in history)\n"
            "/cron   — list scheduled tasks for this chat\n"
            "          /cron cancel <id>  — cancel a schedule\n"
            "          /cron run <id>     — fire immediately (test)\n"
            "\n"
            "Natural cancel: a message like 「取消排程」or 'cancel my schedule' \n"
            "is recognised — if there's exactly one, it gets cancelled; otherwise \n"
            "I'll list them so you can pick.",
        )
        return

    if cmd == "/cancel":
        await _handle_cancel(chat_id)
        return

    if cmd == "/new":
        await _handle_new(chat_id)
        return

    if cmd == "/cron":
        await _handle_cron(chat_id, text)
        return

    await _send_reply(
        chat_id,
        f"❓ Unknown command: {cmd}. Try /help.",
    )


# ── /cron command + natural-language schedule management ─────────────────────

async def _handle_cron(chat_id: str, text: str) -> None:
    """Manage scheduled tasks bound to this chat's session.

    Usage:
      /cron                      — list all schedules for this chat
      /cron cancel <id_prefix>   — cancel a specific schedule (8-char prefix)
      /cron run <id_prefix>      — fire a schedule immediately (test before cron)

    The schedules listed are scoped to the session bound to this
    Telegram chat, so users can only see + manage their own. No
    cross-chat enumeration.
    """
    # Identify the chat's bound session — schedules are per-session.
    sess = await _tracker.get_session_by_telegram_chat(chat_id)
    if not sess:
        await _send_reply(
            chat_id,
            "ℹ️ This chat isn't bound to a session yet. Send any message "
            "to start one, then create schedules from there.",
        )
        return
    session_id = sess["id"]

    parts = text.strip().split(maxsplit=2)
    sub = parts[1].lower() if len(parts) >= 2 else ""

    if sub in ("", "list", "ls"):
        await _cron_list(chat_id, session_id)
        return
    if sub in ("cancel", "rm", "remove", "delete", "stop"):
        target = parts[2] if len(parts) >= 3 else ""
        await _cron_cancel(chat_id, session_id, target)
        return
    if sub in ("run", "trigger", "now"):
        target = parts[2] if len(parts) >= 3 else ""
        await _cron_run_now(chat_id, session_id, target)
        return

    await _send_reply(
        chat_id,
        "❓ Unknown /cron subcommand. Try:\n"
        "  /cron               — list schedules\n"
        "  /cron cancel <id>   — cancel a schedule\n"
        "  /cron run <id>      — trigger immediately\n",
    )


async def _cron_list(chat_id: str, session_id: str) -> None:
    """Render the schedule list with friendly formatting."""
    rows = await _tracker.list_scheduled_tasks(session_id=session_id)
    if not rows:
        await _send_reply(
            chat_id,
            "📭 No scheduled tasks for this chat yet.\n"
            "\n"
            "Create one via the API, e.g.:\n"
            "```\n"
            f"curl -X POST http://<server>/schedules -H 'Content-Type: application/json' \\\n"
            f'  -d \'{{"session_id":"{session_id}",'
            f'"cron":"30 9 * * *",'
            f'"goal":"用 browser_fetch 抓 https://github.com/trending 列 top 10"}}\'\n'
            "```\n"
            "Cron format: 'minute hour day month weekday' "
            "(e.g. `30 9 * * *` = daily 9:30)",
        )
        return

    lines = [f"⏰ {len(rows)} scheduled task(s):"]
    for r in rows:
        rid = (r.get("id") or "")[:8]
        enabled = "✅" if r.get("enabled") else "⏸"
        cron = r.get("cron") or "?"
        goal = (r.get("goal") or "")
        if len(goal) > 70:
            goal = goal[:67] + "…"
        next_at = (r.get("next_run_at") or "")[:16].replace("T", " ")
        last_status = r.get("last_status") or "—"
        fails = int(r.get("consecutive_failures") or 0)
        cap = int(r.get("max_consecutive_failures") or 3)
        fail_str = f" · ❌{fails}/{cap}" if fails > 0 else ""
        lines.append(
            f"\n{enabled} `{rid}`  cron: `{cron}`"
            f"\n   goal: {goal}"
            f"\n   next: {next_at or '—'} · last: {last_status}{fail_str}"
        )
    lines.append("\n\nTo cancel: `/cron cancel <id>` (use 8-char prefix).")
    await _send_reply(chat_id, "\n".join(lines))


async def _cron_cancel(chat_id: str, session_id: str, target: str) -> None:
    """Resolve `target` (id prefix or empty) and cancel.

    If the user wrote `/cron cancel` with no target, and the chat has
    exactly one schedule, cancel that — natural "I only have one, just
    kill it" intent.
    """
    rows = await _tracker.list_scheduled_tasks(session_id=session_id)
    if not rows:
        await _send_reply(chat_id, "ℹ️ Nothing to cancel — no schedules for this chat.")
        return

    target = (target or "").strip().lower()
    matches: List[Dict[str, Any]] = []
    if not target:
        # No target supplied — only resolves cleanly if there's exactly
        # one schedule. Anything more is ambiguous; refuse and list.
        if len(rows) == 1:
            matches = rows
        else:
            await _send_reply(
                chat_id,
                f"❓ {len(rows)} schedules in this chat — which one?\n"
                "Use `/cron cancel <8-char-id>`. Run `/cron` to see ids.",
            )
            return
    else:
        for r in rows:
            rid = (r.get("id") or "").lower()
            if rid.startswith(target):
                matches.append(r)

    if not matches:
        await _send_reply(
            chat_id,
            f"❓ No schedule id starting with `{target}` in this chat. "
            "Run `/cron` to see what's available.",
        )
        return
    if len(matches) > 1:
        ids = ", ".join(f"`{(r.get('id') or '')[:8]}`" for r in matches)
        await _send_reply(
            chat_id,
            f"❓ Prefix `{target}` matches {len(matches)} schedules: {ids}\n"
            "Use a longer prefix to disambiguate.",
        )
        return

    row = matches[0]
    from scheduler import delete_schedule as _del
    ok = await _del(row["id"])
    if ok:
        goal_preview = (row.get("goal") or "")[:60]
        await _send_reply(
            chat_id,
            f"🗑 Cancelled schedule `{row['id'][:8]}`\n"
            f"   was: {goal_preview}",
        )
    else:
        await _send_reply(
            chat_id, f"❌ Failed to cancel `{row['id'][:8]}` — try again.",
        )


async def _cron_run_now(chat_id: str, session_id: str, target: str) -> None:
    """Trigger one schedule immediately. Same prefix resolution as cancel."""
    rows = await _tracker.list_scheduled_tasks(session_id=session_id)
    if not rows:
        await _send_reply(chat_id, "ℹ️ Nothing to run — no schedules for this chat.")
        return

    target = (target or "").strip().lower()
    matches = [r for r in rows if (r.get("id") or "").lower().startswith(target)] if target else (rows if len(rows) == 1 else [])
    if not matches:
        await _send_reply(chat_id, "❓ Couldn't resolve which schedule to run. Try `/cron` first.")
        return
    if len(matches) > 1:
        await _send_reply(chat_id, "❓ Prefix matches multiple — try a longer id.")
        return

    row = matches[0]
    from scheduler import run_now as _run
    await _run(row["id"])
    await _send_reply(
        chat_id,
        f"▶️ Triggered `{row['id'][:8]}` — results will land here when done.",
    )


# Regex matching free-text "I want to cancel my schedule" — checked in
# _dispatch BEFORE handing the message to the orchestrator, so a natural
# request never gets misrouted into a new agent run.
_NL_CANCEL_RE = re.compile(
    r"(取消|停掉|關掉|cancel|stop|remove|delete)"
    r".{0,15}"
    r"(排程|定時|每天的|cron|schedule)",
    re.IGNORECASE,
)


async def _maybe_handle_natural_cancel(chat_id: str, text: str) -> bool:
    """If `text` reads like "please cancel my schedule" in natural
    language, route it to the /cron cancel flow + return True. Otherwise
    return False so the dispatcher continues to orchestrator.

    Disambiguation: with 1 schedule for the chat → cancel + confirm.
    With multiple → reply listing them + tell the user to use `/cron cancel
    <id>`. With 0 → reply "nothing scheduled".

    We deliberately don't try to AI-resolve "cancel the github one" vs
    "cancel the slack one" — too much ambiguity. Forcing the user to
    pick an id with `/cron cancel <id>` is safer than risking the
    wrong cancellation.
    """
    if not _NL_CANCEL_RE.search(text):
        return False
    sess = await _tracker.get_session_by_telegram_chat(chat_id)
    if not sess:
        return False
    await _cron_cancel(chat_id, sess["id"], target="")
    return True


async def _handle_cancel(chat_id: str) -> None:
    """Abort the orchestrator run currently in flight for this chat.
    Cancellation is cooperative — asyncio.Task.cancel() raises
    CancelledError inside the running coroutine on its next `await`,
    which propagates up through orchestrator.run() and unwinds the
    executor. The run_goal closure's outer try/except in
    _run_goal_for_chat catches the exception, but cancellation is a
    legitimate path: we explicitly send a different reply for it."""
    task = _running_chats.get(chat_id)
    if not task or task.done():
        await _send_reply(
            chat_id,
            "ℹ️ Nothing running for this chat — there's no task to cancel.",
        )
        return
    task.cancel()
    # The _run_goal_for_chat coroutine catches CancelledError implicitly
    # via the bare Exception handler and pops itself from _running_chats.
    # Confirm to the user separately so the cancellation reply isn't
    # gated on the orchestrator's cleanup actually completing (which
    # could take a few seconds if a tool is mid-flight).
    await _send_reply(chat_id, "⏹️ Cancelling — the running task will stop shortly.")


async def _handle_new(chat_id: str) -> None:
    """Detach this chat from its current session and start a fresh one.

    Refuses if a task is currently running — switching sessions
    mid-task would leave a confusing dangling run on the old session.
    Make the user /cancel first, then /new. Common UX would just
    /cancel automatically, but explicit is safer: a user might not
    realise the run is in-flight and discard real work.

    The old session is NOT deleted — it stays in the sessions list and
    history. Just the chat binding is moved to the new one.
    """
    in_flight = _running_chats.get(chat_id)
    if in_flight and not in_flight.done():
        await _send_reply(
            chat_id,
            "⏳ A task is still running. Send /cancel first, then /new.",
        )
        return

    try:
        existing = await _tracker.get_session_by_telegram_chat(chat_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Telegram bridge: lookup failed during /new for chat=%s", chat_id)
        await _send_reply(chat_id, f"❌ Could not look up your current session: {exc}")
        return

    # Create a fresh session and rebind. The atomic
    # bind_session_to_telegram_chat in tracker.py clears the chat_id off
    # the previous session in the same SQL transaction, so we don't end
    # up with two sessions claiming the same chat.
    try:
        new_id = str(uuid.uuid4())
        title = f"Telegram (fresh @ {time.strftime('%Y-%m-%d %H:%M')})"
        await _tracker.create_session(new_id, title, mode="code")
        await _tracker.bind_session_to_telegram_chat(new_id, chat_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Telegram bridge: failed to create fresh session for chat=%s", chat_id)
        await _send_reply(chat_id, f"❌ Could not create a fresh session: {exc}")
        return

    old_hint = f"\nPrevious session ({existing['id'][:8]}) stays in your history." if existing else ""
    await _send_reply(
        chat_id,
        f"🆕 Fresh session started.{old_hint}\n"
        "Send any text to kick off your first goal.",
    )
