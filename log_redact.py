"""
Keep secrets out of logs.

httpx logs every request URL at INFO, and a Telegram Bot API URL carries
the bot token (https://api.telegram.org/bot<token>/sendMessage). Those
lines went to journald and to the in-memory buffer the diagnostics
endpoint serves. Exceptions leak the same way: httpx errors quote the URL.

install() quiets httpx/httpcore to WARNING (their per-request lines were
mostly getUpdates noise) and adds a filter to every root handler that
masks known secret shapes in the message and in the traceback.
"""
from __future__ import annotations

import logging
import re

_PATTERNS = [
    # Telegram Bot API and file download URLs
    (re.compile(r"((?:api\.telegram\.org/)(?:file/)?bot)\d+:[A-Za-z0-9_-]+"), r"\1<redacted>"),
    # ?key=… (Gemini), ?api_key=…, ?access_token=…, ?token=…
    (re.compile(r"([?&](?:key|api_key|apikey|access_token|token|secret)=)[^&\s'\"]+", re.I), r"\1<redacted>"),
    # Authorization headers that end up in a message
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]{8,}"), r"\1<redacted>"),
]


def redact(text: str) -> str:
    for pat, repl in _PATTERNS:
        text = pat.sub(repl, text)
    return text


class RedactSecrets(logging.Filter):
    """Handler filter: rewrites the record's message and traceback."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
            clean = redact(msg)
            if clean != msg:
                record.msg, record.args = clean, None
            if record.exc_info and not record.exc_text:
                # Formatter.format reuses exc_text when it is already set.
                record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
            elif record.exc_text:
                record.exc_text = redact(record.exc_text)
        except Exception:  # noqa: BLE001 — a filter must never drop or break a log line
            pass
        return True


_FILTER = RedactSecrets()


def install(*extra_handlers: logging.Handler) -> None:
    """Attach the filter to every root handler (and any extra ones).
    Idempotent; call again after adding handlers."""
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    for h in list(logging.getLogger().handlers) + list(extra_handlers):
        if _FILTER not in h.filters:
            h.addFilter(_FILTER)
