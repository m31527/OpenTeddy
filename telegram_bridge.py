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


async def _long_task_hint(chat_id: str, short_goal: str) -> None:
    """One-shot 'still working on this' message that only fires if the
    task is taking long enough that the user might start wondering if
    the bot crashed. Cancelled before firing for fast tasks, so chat-
    mode quick-answer questions never see this."""
    try:
        await asyncio.sleep(_LONG_TASK_HINT_S)
        await _send_reply(
            chat_id,
            f"🐻 Still working on: {short_goal}\n"
            "(this is taking a moment — long tasks can run several minutes)",
        )
    except asyncio.CancelledError:
        # Task finished before the threshold — that's the *happy* path
        # for fast intent-classified chat answers. No reply was sent.
        raise


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

    # Fire the native "typing…" indicator immediately and keep it alive
    # in the background. The previous version sent a static text ack
    # ("🐻 Got it. Working on: ... could take minutes") immediately —
    # which was visually mismatched for chat-mode questions that finish
    # in 2-3 s and made the inbox look cluttered. With the indicator
    # approach, fast answers feel native ("typing…" → answer) and only
    # genuinely-slow tasks see the long-task hint below.
    short_goal = goal_text if len(goal_text) <= 80 else goal_text[:77] + "…"
    typing_task = asyncio.create_task(
        _keep_typing(chat_id), name=f"tg_typing:{chat_id[:16]}",
    )
    hint_task = asyncio.create_task(
        _long_task_hint(chat_id, short_goal),
        name=f"tg_hint:{chat_id[:16]}",
    )

    # Build TaskRequest exactly the way POST /run does — same shape, same
    # orchestrator.run() entry point, just bypassing HTTP.
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
        result = await _orchestrator.run(req)
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
    await _send_reply(chat_id, reply)
    _running_chats.pop(chat_id, None)


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
            "/new    — start a fresh session for this chat (old one stays in history)\n",
        )
        return

    if cmd == "/cancel":
        await _handle_cancel(chat_id)
        return

    if cmd == "/new":
        await _handle_new(chat_id)
        return

    await _send_reply(
        chat_id,
        f"❓ Unknown command: {cmd}. Try /help.",
    )


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
