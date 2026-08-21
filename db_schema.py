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
_MAX_COLS_PER_TABLE = 40
_MAX_CHARS = 8000        # hard cap on the stored summary


async def snapshot_schema(db_url: str, timeout_s: float = 20.0) -> str:
    """Connect to db_url, inspect every table's columns, return a compact
    text summary. Raises on connection failure (caller decides whether
    that's fatal — agent save treats it as a warning, not an error)."""
    from sqlalchemy import inspect
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(db_url, pool_pre_ping=True)
    try:
        async with engine.connect() as conn:
            def _dump(sync_conn):
                insp = inspect(sync_conn)
                tables = insp.get_table_names()
                lines = []
                for t in tables[:_MAX_TABLES]:
                    try:
                        cols = insp.get_columns(t)
                    except Exception as exc:  # noqa: BLE001
                        lines.append(f"- {t}: (columns unreadable: {exc})")
                        continue
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
                    lines.append(f"- {t}: {col_repr}{extra}")
                if len(tables) > _MAX_TABLES:
                    lines.append(f"…(+{len(tables) - _MAX_TABLES} more tables)")
                return f"tables({len(tables)}):\n" + "\n".join(lines)

            summary = await asyncio.wait_for(
                conn.run_sync(_dump), timeout=timeout_s,
            )
    finally:
        await engine.dispose()

    if len(summary) > _MAX_CHARS:
        summary = summary[:_MAX_CHARS] + "\n…(schema truncated)"
    return summary
