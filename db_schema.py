"""
Programmatic database schema snapshot.

Why this exists: an agent bound to a database used to LEARN the schema by
LLM exploration — db_list_tables, then db_describe_table one table at a
time, each round costing a full 35B inference (10-20s). A user watched
"how many factories does YCM have" burn ~9 exploration rounds (≈5 min)
describing sensor logs and firebase tables that had nothing to do with
the question.

The schema is static information. Dump it ONCE programmatically (pure
SQLAlchemy inspection — zero LLM calls, a few seconds), store the compact
summary on the agent row, and inject it into planning + execution. The
model then goes straight to db_query with the right table — exploration
rounds drop from ~9 to 0.

The summary contains table + column names/types only — never row data,
never credentials.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

_MAX_TABLES = 120        # very wide DBs get truncated with a note
_MAX_COLS_PER_TABLE = 30
_MAX_CHARS = 9000        # hard cap on the stored summary

# Bump when the snapshot FORMAT changes. Stored snapshots that don't
# carry the current marker are re-captured on next use — without this, an
# agent that stored a snapshot from a buggy older build keeps using it
# forever (v1 alphabetically truncated away the ycm_* entity tables, and
# the "capture only when empty" rule meant the fix never reached agents
# that already had one).
SCHEMA_FORMAT = "v2"
_FORMAT_MARKER = f"[schema-{SCHEMA_FORMAT}]"


def is_current_format(summary: str) -> bool:
    """True when a stored snapshot was produced by the current format."""
    return bool(summary) and summary.lstrip().startswith(_FORMAT_MARKER)

# Infrastructure / plumbing tables: real, but almost never what a business
# question is about. They get NAME-ONLY treatment so their (often very
# wide) column lists can't crowd out the entity tables.
#
# This exists because of a real failure: the first snapshot dumped all 63
# tables alphabetically with full columns, blew the char cap at
# "material_type", and silently truncated away ycm_brand / ycm_factory /
# ycm_warehouse — the exact tables the question was about. The model
# couldn't see them, so it answered from account.company_name LIKE '%YCM%'
# and reported user accounts instead of factories. Ordering a schema
# alphabetically and cutting the tail is how you lose the answer.
_INFRA_PREFIXES = (
    "auth_", "oauth_", "session_", "cache_", "job_", "queue_",
    "migration", "django_", "sys_", "firebase_", "mqtt_", "sms_",
    "email_", "firmware", "ota_", "ckeditor", "browse_",
)
_INFRA_SUFFIXES = (
    "_log", "_logs", "_history", "_audit", "_translation", "_i18n",
    "_seq", "_migrations",
)
_INFRA_CONTAINS = (
    "notification", "file_resource", "instruction_manual",
)


def _is_infra(table: str) -> bool:
    t = table.lower()
    return (
        t.startswith(_INFRA_PREFIXES)
        or t.endswith(_INFRA_SUFFIXES)
        or any(k in t for k in _INFRA_CONTAINS)
        or t in ("version", "version_address", "browse_record")
    )


async def snapshot_schema(db_url: str, timeout_s: float = 20.0) -> str:
    """Connect to db_url, inspect every table's columns, return a compact
    text summary. Raises on connection failure (caller decides whether
    that's fatal — agent save treats it as a warning, not an error).

    Budgeting is priority-ordered, never alphabetical: business/entity
    tables get their full column list first, infrastructure tables are
    listed by name only at the end. Even when the char cap bites, EVERY
    table name still appears — the model must never be unaware that a
    table exists.
    """
    from sqlalchemy import inspect
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(db_url, pool_pre_ping=True)
    try:
        async with engine.connect() as conn:
            def _dump(sync_conn):
                insp = inspect(sync_conn)
                tables = insp.get_table_names()[:_MAX_TABLES]
                core  = [t for t in tables if not _is_infra(t)]
                infra = [t for t in tables if _is_infra(t)]

                def _cols(t: str) -> str:
                    try:
                        cols = insp.get_columns(t)
                    except Exception as exc:  # noqa: BLE001
                        return f"- {t}: (columns unreadable: {exc})"
                    col_repr = ", ".join(
                        # "name(type)" with the type collapsed to its bare
                        # class ("VARCHAR(255) COLLATE x" → "varchar") —
                        # enough to write correct SQL, compact enough to
                        # fit a wide DB in one prompt.
                        f"{c['name']}({str(c['type']).split('(')[0].strip().lower() or '?'})"
                        for c in cols[:_MAX_COLS_PER_TABLE]
                    )
                    extra = "" if len(cols) <= _MAX_COLS_PER_TABLE else \
                        f", …+{len(cols) - _MAX_COLS_PER_TABLE} more"
                    return f"- {t}: {col_repr}{extra}"

                head = (f"{_FORMAT_MARKER} tables({len(tables)}) — "
                        "main tables with columns:")
                detail: list[str] = []
                used = len(head)
                dropped: list[str] = []
                for t in core:
                    line = _cols(t)
                    # Reserve room for the infra/name-only tail.
                    if used + len(line) > _MAX_CHARS - 600:
                        dropped.append(t)
                        continue
                    detail.append(line)
                    used += len(line) + 1

                out = [head] + detail
                if dropped:
                    out.append(
                        "(columns omitted for space — use db_describe_table "
                        "if needed): " + ", ".join(dropped)
                    )
                if infra:
                    out.append(
                        "system/log tables (rarely relevant to business "
                        "questions): " + ", ".join(infra)
                    )
                return "\n".join(out)

            summary = await asyncio.wait_for(
                conn.run_sync(_dump), timeout=timeout_s,
            )
    finally:
        await engine.dispose()

    if len(summary) > _MAX_CHARS:
        summary = summary[:_MAX_CHARS] + "\n…(schema truncated)"
    return summary
