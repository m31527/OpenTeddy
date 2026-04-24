"""
OpenTeddy Notification Tools
Outbound-notification tools exposed to the executor model:

  telegram_send  — post a message to a Telegram chat via Bot API
  email_send     — send mail via SMTP (starttls auto-detected)

Both read credentials from ``config`` (which is filled from the
SettingsStore — editable via /settings POST or the UI's Settings tab).
No per-session setup required; once credentials are entered, ANY
session's agent can call these tools.

Risk level: both are LOW. They're opt-in (needs credentials), the
target (chat_id / to:) is in tool args and therefore visible in the
tool card, and blanket-approval for every notification would break
the point of using them in scheduled / webhook-triggered flows.
Destructive power is limited — worst case is spam to one's own chat.
"""

from __future__ import annotations

import asyncio
import logging
import smtplib
import ssl
import time
from email.message import EmailMessage
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)


# ── telegram_send ─────────────────────────────────────────────────────────────

_TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


async def telegram_send(
    text: str,
    chat_id: Optional[Union[str, int]] = None,
    parse_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a message to a Telegram chat.

    ``chat_id`` defaults to ``config.telegram_default_chat_id`` when
    not supplied — lets simple "notify me" flows omit it entirely
    once the user has set a default in Settings.

    ``parse_mode`` accepts "Markdown" / "HTML" / None (plain).
    """
    from config import config as _cfg
    start = time.monotonic()

    token = (getattr(_cfg, "telegram_bot_token", "") or "").strip()
    if not token:
        return make_result(
            False,
            error=(
                "Telegram bot token not configured. "
                "Set it in Settings → Notification Credentials → Telegram Bot Token."
            ),
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    target_chat = chat_id
    if not target_chat:
        target_chat = (getattr(_cfg, "telegram_default_chat_id", "") or "").strip()
    if not target_chat:
        return make_result(
            False,
            error=(
                "No chat_id supplied and no Telegram Default Chat ID set. "
                "Either pass chat_id explicitly, or set a default in Settings."
            ),
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    url = _TELEGRAM_API_BASE.format(token=token)
    payload: Dict[str, Any] = {"chat_id": target_chat, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json=payload)
        duration_ms = int((time.monotonic() - start) * 1000)
        # Telegram returns 200 with {ok: true, result: {...}} on success,
        # 400/401/403 with {ok: false, description: "..."} on failure.
        data: Dict[str, Any] = {}
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001
            pass
        if resp.status_code == 200 and data.get("ok"):
            # Don't echo the whole message back — can be long; just the IDs.
            message_id = (data.get("result") or {}).get("message_id")
            return make_result(
                True,
                result={
                    "chat_id":    str(target_chat),
                    "message_id": message_id,
                    "length":     len(text),
                    "parse_mode": parse_mode or "plain",
                },
                duration_ms=duration_ms,
            )
        # Bubble up Telegram's description for actionable errors like
        # "chat not found" or "bot was blocked".
        description = data.get("description") or f"HTTP {resp.status_code}"
        return make_result(
            False,
            error=f"Telegram API rejected the message: {description}",
            duration_ms=duration_ms,
        )
    except httpx.HTTPError as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        return make_result(
            False,
            error=f"Network error reaching Telegram: {exc}",
            duration_ms=duration_ms,
        )


_SCHEMA_TELEGRAM: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "telegram_send",
        "description": (
            "Send a message to a Telegram chat via the bot API. Use for "
            "real-time notifications: new orders, alerts, status updates. "
            "Credentials are configured once in Settings → Notification "
            "Credentials; this tool reads them automatically. Low risk — "
            "it only posts to chats your bot has been added to."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Message body. Plain text by default; pass "
                                   "parse_mode='Markdown' for bold/italic/code.",
                },
                "chat_id": {
                    "type": ["string", "integer"],
                    "description": "Target chat id (user / group / channel). "
                                   "Omit to use the Default Chat ID from "
                                   "Settings, if set.",
                },
                "parse_mode": {
                    "type": "string",
                    "enum": ["Markdown", "HTML"],
                    "description": "Optional formatting. Leave blank for plain text.",
                },
            },
            "required": ["text"],
        },
    },
}


# ── email_send ────────────────────────────────────────────────────────────────

def _send_smtp_sync(
    host: str, port: int, user: str, password: str,
    msg: EmailMessage, use_tls_explicit: bool,
) -> None:
    """Run the (blocking) SMTP conversation. Called via asyncio.to_thread.

    Heuristic for TLS:
      - port 465 → implicit SSL (SMTP_SSL)
      - port 587 (the default) or anything else → STARTTLS where available
    This covers 95% of real SMTP services (Gmail, Outlook, SES,
    Mailgun, Postmark, self-hosted Postfix).
    """
    if port == 465:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, context=ctx, timeout=20) as s:
            if user:
                s.login(user, password)
            s.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=20) as s:
            s.ehlo()
            if s.has_extn("STARTTLS") or use_tls_explicit:
                ctx = ssl.create_default_context()
                s.starttls(context=ctx)
                s.ehlo()
            if user:
                s.login(user, password)
            s.send_message(msg)


def _as_addr_list(value: Union[str, List[str], None]) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        # Accept "a@x.com, b@y.com" comma-separated shorthand.
        return [p.strip() for p in value.split(",") if p.strip()]
    return [str(v).strip() for v in value if str(v).strip()]


async def email_send(
    to: Union[str, List[str]],
    subject: str,
    body: str,
    html: bool = False,
    cc: Optional[Union[str, List[str]]] = None,
    bcc: Optional[Union[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Send an email via the configured SMTP server.

    Credentials come from the Settings → Notification Credentials
    block (smtp_host / smtp_port / smtp_user / smtp_password / smtp_from).
    Without them configured, returns a clear error pointing there.
    """
    from config import config as _cfg
    start = time.monotonic()

    host = (getattr(_cfg, "smtp_host", "") or "").strip()
    if not host:
        return make_result(
            False,
            error=(
                "SMTP server not configured. Set SMTP Host / Port / User / "
                "Password in Settings → Notification Credentials."
            ),
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    port = int(getattr(_cfg, "smtp_port", 587) or 587)
    user = (getattr(_cfg, "smtp_user", "") or "").strip()
    pwd  = getattr(_cfg, "smtp_password", "") or ""
    sender = (getattr(_cfg, "smtp_from", "") or user).strip()
    if not sender:
        return make_result(
            False,
            error="smtp_from / smtp_user empty — can't pick a From address.",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    to_list  = _as_addr_list(to)
    cc_list  = _as_addr_list(cc)
    bcc_list = _as_addr_list(bcc)
    if not to_list:
        return make_result(
            False,
            error="`to` is empty — nothing to send.",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"]   = ", ".join(to_list)
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)
    msg["Subject"] = subject or "(no subject)"
    if html:
        # Use set_content for a text fallback + add_alternative for HTML,
        # so clients that don't render HTML still see something sane.
        msg.set_content("(HTML-only message — please view in an HTML-capable client.)")
        msg.add_alternative(body, subtype="html")
    else:
        msg.set_content(body or "")

    # smtplib is blocking — run on a worker thread so we don't stall
    # the event loop during the SMTP handshake.
    try:
        await asyncio.to_thread(
            _send_smtp_sync, host, port, user, pwd, msg,
            False,  # use_tls_explicit
        )
    except smtplib.SMTPException as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        return make_result(
            False,
            error=f"SMTP error: {exc}",
            duration_ms=duration_ms,
        )
    except Exception as exc:  # noqa: BLE001
        duration_ms = int((time.monotonic() - start) * 1000)
        return make_result(
            False,
            error=f"email_send failed: {exc}",
            duration_ms=duration_ms,
        )

    duration_ms = int((time.monotonic() - start) * 1000)
    return make_result(
        True,
        result={
            "to":      to_list,
            "cc":      cc_list,
            "bcc":     bcc_list,
            "subject": subject,
            "bytes":   len(body.encode("utf-8")),
            "html":    bool(html),
        },
        duration_ms=duration_ms,
    )


_SCHEMA_EMAIL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "email_send",
        "description": (
            "Send an email via the configured SMTP server. Use for alerts "
            "/ reports / scheduled summaries. Supports plain text or HTML "
            "(set html=true for an HTML body). Credentials are configured "
            "once in Settings → Notification Credentials. Low risk — sends "
            "from your own account."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": ["string", "array"],
                    "description": "Recipient address(es). Can be a single "
                                   "email string, an array of strings, or a "
                                   "comma-separated string.",
                    "items": {"type": "string"},
                },
                "subject": {
                    "type": "string",
                    "description": "Subject line.",
                },
                "body": {
                    "type": "string",
                    "description": "Message body. Plain text by default; "
                                   "set html=true to send as HTML.",
                },
                "html": {
                    "type": "boolean",
                    "description": "Treat body as HTML. Default false.",
                    "default": False,
                },
                "cc":  {"type": ["string", "array"], "items": {"type": "string"}},
                "bcc": {"type": ["string", "array"], "items": {"type": "string"}},
            },
            "required": ["to", "subject", "body"],
        },
    },
}


# ── Export ────────────────────────────────────────────────────────────────────

NOTIFY_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    (telegram_send, _SCHEMA_TELEGRAM, "low"),
    (email_send,    _SCHEMA_EMAIL,    "low"),
]
