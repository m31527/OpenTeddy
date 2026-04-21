"""
OpenTeddy DB Tool
Async database access via SQLAlchemy.
Reads DATABASE_URL from environment.
SELECT queries → LOW risk.
INSERT / UPDATE / DELETE / DDL → HIGH risk.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)

# Lazy-imported to avoid hard dependency at startup
_engine: Optional[Any] = None


def _get_engine() -> Any:
    """Return (create if needed) the async SQLAlchemy engine."""
    global _engine
    if _engine is not None:
        return _engine
    try:
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError as exc:
        raise ImportError(
            "SQLAlchemy async support not installed. "
            "Run: pip install 'sqlalchemy[asyncio]' asyncpg aiosqlite"
        ) from exc

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        raise ValueError(
            "DATABASE_URL is not set. "
            "Example: postgresql+asyncpg://user:pass@host/dbname"
        )
    _engine = create_async_engine(db_url, echo=False, pool_pre_ping=True)
    return _engine


def _is_write_sql(sql: str) -> bool:
    """Return True if SQL is a write/DDL statement."""
    clean = sql.strip().upper()
    write_starters = ("INSERT", "UPDATE", "DELETE", "DROP", "CREATE",
                      "ALTER", "TRUNCATE", "REPLACE", "MERGE", "UPSERT")
    return any(clean.startswith(kw) for kw in write_starters)


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


# ── Implementations ────────────────────────────────────────────────────────────

async def db_query(sql: str) -> Dict[str, Any]:
    """Execute a SELECT query. LOW risk."""
    start = time.monotonic()
    if _is_write_sql(sql):
        return make_result(
            False,
            error="db_query only allows SELECT statements. Use db_execute for writes.",
            duration_ms=_ms(start),
        )
    try:
        from sqlalchemy import text
        engine = _get_engine()
        async with engine.connect() as conn:
            result = await conn.execute(text(sql))
            cols = list(result.keys())
            rows = [dict(zip(cols, row)) for row in result.fetchall()]
        return make_result(True, result={"columns": cols, "rows": rows,
                                          "row_count": len(rows)},
                           duration_ms=_ms(start))
    except (ImportError, ValueError) as exc:
        return make_result(False, error=str(exc), duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        logger.error("db_query error: %s", exc)
        return make_result(False, error=str(exc), duration_ms=_ms(start))


async def db_execute(sql: str) -> Dict[str, Any]:
    """Execute a write/DDL SQL statement. HIGH risk — requires approval."""
    start = time.monotonic()
    try:
        from sqlalchemy import text
        engine = _get_engine()
        async with engine.begin() as conn:
            result = await conn.execute(text(sql))
            rowcount = result.rowcount if hasattr(result, "rowcount") else -1
        return make_result(True, result={"rows_affected": rowcount},
                           duration_ms=_ms(start))
    except (ImportError, ValueError) as exc:
        return make_result(False, error=str(exc), duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        logger.error("db_execute error: %s", exc)
        return make_result(False, error=str(exc), duration_ms=_ms(start))


async def db_list_tables() -> Dict[str, Any]:
    """List all tables in the database. LOW risk."""
    start = time.monotonic()
    try:
        from sqlalchemy import inspect, text
        engine = _get_engine()
        async with engine.connect() as conn:
            # Works for Postgres, MySQL, SQLite
            insp = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).get_table_names()
            )
        return make_result(True, result={"tables": insp}, duration_ms=_ms(start))
    except (ImportError, ValueError) as exc:
        return make_result(False, error=str(exc), duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        logger.error("db_list_tables error: %s", exc)
        return make_result(False, error=str(exc), duration_ms=_ms(start))


async def db_describe_table(table: str) -> Dict[str, Any]:
    """Describe the columns of a table. LOW risk."""
    start = time.monotonic()
    try:
        from sqlalchemy import inspect
        engine = _get_engine()
        async with engine.connect() as conn:
            columns = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).get_columns(table)
            )
        col_info = [
            {
                "name": c["name"],
                "type": str(c["type"]),
                "nullable": c.get("nullable", True),
                "default": str(c.get("default", "")),
            }
            for c in columns
        ]
        return make_result(True, result={"table": table, "columns": col_info},
                           duration_ms=_ms(start))
    except (ImportError, ValueError) as exc:
        return make_result(False, error=str(exc), duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        logger.error("db_describe_table error: %s", exc)
        return make_result(False, error=str(exc), duration_ms=_ms(start))


# ── Schemas ───────────────────────────────────────────────────────────────────

_SCHEMA_QUERY: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "db_query",
        "description": "Execute a SQL SELECT query against the configured database.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL SELECT statement."},
            },
            "required": ["sql"],
        },
    },
}

_SCHEMA_EXECUTE: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "db_execute",
        "description": (
            "Execute a SQL write statement (INSERT, UPDATE, DELETE, DDL). "
            "Requires human approval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL statement to execute."},
            },
            "required": ["sql"],
        },
    },
}

_SCHEMA_LIST_TABLES: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "db_list_tables",
        "description": "List all tables in the connected database.",
        "parameters": {"type": "object", "properties": {}},
    },
}

_SCHEMA_DESCRIBE: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "db_describe_table",
        "description": "Show column names and types for a database table.",
        "parameters": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Table name."},
            },
            "required": ["table"],
        },
    },
}

# ── Export ─────────────────────────────────────────────────────────────────────

DB_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    (db_query,          _SCHEMA_QUERY,        "low"),
    (db_execute,        _SCHEMA_EXECUTE,      "high"),
    (db_list_tables,    _SCHEMA_LIST_TABLES,  "low"),
    (db_describe_table, _SCHEMA_DESCRIBE,     "low"),
]
