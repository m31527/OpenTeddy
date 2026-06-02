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
# Phase 2 will use _orchestrator; Phase 1 reads _tracker only for
# bookkeeping / logging.
_tracker: Any = None        # type: tracker.Tracker
_orchestrator: Any = None   # type: orchestrator.Orchestrator


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
        logger.debug(
            "Telegram bridge: ignored chat_id %s (not in whitelist)",
            chat_id,
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

    # ── Free-text → goal (Phase 2 plug-in point) ──────────────────────────────
    # PHASE 2: resolve session for chat_id (create if missing, bind in
    # tracker), build TaskRequest, await orchestrator.run(req), push
    # result back via _send_reply. Concurrency guard: refuse if the
    # chat's session already has a running task.
    await _send_reply(
        chat_id,
        f"📨 Received: {text}\n\n"
        "(Phase 1 echo — orchestrator hand-off lands in the next commit.)",
    )


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
            "/cancel — stop the currently-running task (Phase 3)\n"
            "/new    — start a fresh session for this chat (Phase 3)\n",
        )
        return

    await _send_reply(
        chat_id,
        f"❓ Unknown command: {cmd}. Try /help.",
    )
