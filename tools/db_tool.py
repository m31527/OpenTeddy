"""
OpenTeddy DB Tool — per-session SQL database access.

Each session attaches at most one DB connection via the Analytic-mode
"Connect database" flow (UI + REST: POST /sessions/{id}/db_connect).
The connection URL lives on the session row in tracker.db (db_url
column, treated as a secret). Tools in this file all derive *which*
URL to use from the current task's session via the contextvar set
by tool_registry.execute().

The five tools shipped here:

  db_list_connections()
      Return whether the current session has a DB attached and, if so,
      its kind + label (NEVER the URL). The agent calls this first to
      decide whether to use DB tools or admit it can't query anything.

  db_list_tables()
      Inspect-mode schema introspection. SQLAlchemy's reflection layer
      handles all the supported backends (postgres / mysql / sqlite /
      mssql / oracle / duckdb) so the agent doesn't have to know the
      vendor's quirks.

  db_describe_table(table)
      Per-table column + type listing. The agent calls this on every
      table it sees in list_tables to understand the schema before
      writing SQL.

  db_query(sql)
      Run a SELECT. Hard-rejects anything that starts with a write
      keyword (INSERT / UPDATE / DELETE / DDL) — those go through
      db_execute. Result is capped at 1000 rows by default; we
      auto-append LIMIT 1000 to the SQL if the caller didn't.

  db_query_to_csv(sql, output_path)
      Same as db_query but writes results to a CSV file in the agent
      workspace. Use this in Analytic mode when the next step is
      pandas / matplotlib / Chart.js — it's much cheaper than
      shipping 50K rows back through the LLM context.

  db_execute(sql)
      INSERT / UPDATE / DELETE / DDL. HIGH risk — gated by the
      approval queue. User has to confirm in the UI before it runs.

Security:
  * URL stored as a secret on the session row. Redacted in
    /admin/diagnostics zips. Never returned by any API.
  * Read-only by default for db_query — write keywords rejected.
  * Write ops route through db_execute → approval queue.
  * Connection-pool lifetime = session lifetime. Cleared on session
    delete or explicit /sessions/{id}/db_connect DELETE.
"""

from __future__ import annotations

import csv
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)


# ── Engine cache (keyed by URL) ───────────────────────────────────────────────
# Multiple sessions pointing at the same DB share one engine — connection
# pooling is the whole reason engines exist, and SQLAlchemy's async engine
# is thread-safe across coroutines.
_engines: Dict[str, Any] = {}


def _is_write_sql(sql: str) -> bool:
    """Return True if SQL starts with a write/DDL keyword."""
    clean = sql.strip().upper()
    write_starters = (
        "INSERT", "UPDATE", "DELETE", "DROP", "CREATE",
        "ALTER", "TRUNCATE", "REPLACE", "MERGE", "UPSERT",
        "GRANT", "REVOKE", "VACUUM",
    )
    return any(clean.startswith(kw) for kw in write_starters)


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


async def _get_session_engine() -> Tuple[Optional[Any], Optional[str]]:
    """Resolve the current session's DB engine.

    Returns ``(engine, error_str)``. On success engine is set, error is
    None. On failure engine is None and error is a human-readable
    message the tool should bubble up unchanged.
    """
    # 1. Identify the session from the contextvar set by registry.execute
    from tools._context import get_session_id
    session_id = get_session_id()
    if not session_id:
        return None, (
            "No session in context — DB tools can only run inside an "
            "orchestrator-driven task. (If you're seeing this from a "
            "manual /run call, set session_id in the request.)"
        )

    # 2. Pull the attached connection from the session row
    from tracker import Tracker  # local import to avoid circular
    # NB: we reach into the global tracker instance through main.py.
    # tracker is created at startup and lives in main.<module>.tracker.
    import main as _main_module
    conn = await _main_module.tracker.get_session_db_connection(session_id)
    if not conn:
        return None, (
            "This session has no database attached. In Analytic mode, "
            "click the + button next to the chat input and pick "
            "'Connect database' to attach one before asking DB queries."
        )

    url = conn["url"]
    # 3. Build / reuse engine. Keyed by URL so two sessions on the same
    #    DB share a pool. Limited to a 5-connection pool — desktops
    #    don't need more, and it keeps remote DB load polite.
    if url in _engines:
        return _engines[url], None

    try:
        from sqlalchemy.ext.asyncio import create_async_engine
    except ImportError:
        return None, (
            "sqlalchemy[asyncio] is not installed in this build. "
            "(Frozen .app should include it — if you're seeing this, "
            "the PyInstaller bundle is missing the dep.)"
        )

    try:
        engine = create_async_engine(
            url,
            echo=False,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=5,
        )
        _engines[url] = engine
        return engine, None
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to build engine for session %s: %s", session_id, exc)
        return None, f"Could not connect to database: {exc}"


# ── Tool implementations ──────────────────────────────────────────────────────

async def db_list_connections() -> Dict[str, Any]:
    """Return the current session's DB connection metadata (kind +
    label only — never the URL). Use this first to know whether DB
    queries are possible at all in this session."""
    start = time.monotonic()
    from tools._context import get_session_id
    session_id = get_session_id()
    if not session_id:
        return make_result(False, error="no session in context",
                           duration_ms=_ms(start))
    import main as _main_module
    conn = await _main_module.tracker.get_session_db_connection(session_id)
    if not conn:
        return make_result(True, result={"connected": False},
                           duration_ms=_ms(start))
    return make_result(
        True,
        result={
            "connected": True,
            "kind":  conn["kind"],
            "label": conn["label"],
        },
        duration_ms=_ms(start),
    )


async def db_list_tables() -> Dict[str, Any]:
    """List every table in the connected database. Low risk."""
    start = time.monotonic()
    engine, err = await _get_session_engine()
    if err:
        return make_result(False, error=err, duration_ms=_ms(start))
    try:
        from sqlalchemy import inspect
        async with engine.connect() as conn:  # type: ignore[union-attr]
            names = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).get_table_names()
            )
        return make_result(True, result={"tables": names},
                           duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        logger.error("db_list_tables error: %s", exc)
        return make_result(False, error=str(exc), duration_ms=_ms(start))


async def db_describe_table(table: str) -> Dict[str, Any]:
    """Describe columns + types for a single table. Low risk."""
    start = time.monotonic()
    engine, err = await _get_session_engine()
    if err:
        return make_result(False, error=err, duration_ms=_ms(start))
    try:
        from sqlalchemy import inspect
        async with engine.connect() as conn:  # type: ignore[union-attr]
            columns = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).get_columns(table)
            )
        col_info = [
            {
                "name":     c["name"],
                "type":     str(c["type"]),
                "nullable": c.get("nullable", True),
                "default":  str(c.get("default", "")),
            }
            for c in columns
        ]
        return make_result(True, result={"table": table, "columns": col_info},
                           duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        logger.error("db_describe_table error: %s", exc)
        return make_result(False, error=str(exc), duration_ms=_ms(start))


async def db_query(sql: str, limit: int = 1000) -> Dict[str, Any]:
    """Run a SELECT against the session's database. Auto-LIMITs to
    1000 rows by default so a misjudged query can't return 10M rows
    into the LLM context. Pass ``limit=N`` to override.

    Hard-rejects write/DDL keywords — use db_execute for those.
    """
    start = time.monotonic()
    if _is_write_sql(sql):
        return make_result(
            False,
            error=(
                "db_query is SELECT-only. Use db_execute for INSERT / "
                "UPDATE / DELETE / DDL — those require user approval."
            ),
            duration_ms=_ms(start),
        )
    engine, err = await _get_session_engine()
    if err:
        return make_result(False, error=err, duration_ms=_ms(start))
    try:
        from sqlalchemy import text
        # Append a LIMIT if the SQL doesn't already have one. Cheap
        # safety net against runaway "SELECT * FROM events" calls
        # eating context tokens.
        sql_clean = sql.strip().rstrip(";")
        if "limit" not in sql_clean.lower() and limit > 0:
            sql_clean = f"{sql_clean} LIMIT {int(limit)}"
        async with engine.connect() as conn:  # type: ignore[union-attr]
            result = await conn.execute(text(sql_clean))
            cols = list(result.keys())
            # Coerce values to JSON-friendly types (Decimal → str,
            # datetime → ISO) so the orchestrator can serialise the
            # result without crashing on Pydantic-unfriendly types.
            rows = []
            for r in result.fetchall():
                row: Dict[str, Any] = {}
                for c, v in zip(cols, r):
                    row[c] = _to_jsonable(v)
                rows.append(row)
        return make_result(
            True,
            result={"columns": cols, "rows": rows, "row_count": len(rows)},
            duration_ms=_ms(start),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("db_query error: %s", exc)
        return make_result(False, error=str(exc), duration_ms=_ms(start))


async def db_query_to_csv(sql: str, output_path: str) -> Dict[str, Any]:
    """Run a SELECT and stream the rows to ``output_path`` (CSV).
    Returns row count + the file path. Designed for the Analytic-mode
    handoff: agent runs this, then feeds the CSV to python_exec for
    pandas / Chart.js.

    The output_path is resolved relative to the session workspace
    (config.agent_workspace_dir) — absolute paths and paths that
    escape the workspace are rejected to stop the model from writing
    SQL exports into /etc/.
    """
    start = time.monotonic()
    if _is_write_sql(sql):
        return make_result(
            False,
            error="db_query_to_csv is SELECT-only. Use db_execute for writes.",
            duration_ms=_ms(start),
        )
    # Workspace-relative path; reject anything that tries to escape.
    from config import config
    ws = os.path.abspath(config.agent_workspace_dir)
    abs_out = os.path.abspath(os.path.join(ws, output_path))
    if not abs_out.startswith(ws + os.sep) and abs_out != ws:
        return make_result(
            False,
            error=f"output_path must be inside the session workspace ({ws})",
            duration_ms=_ms(start),
        )

    engine, err = await _get_session_engine()
    if err:
        return make_result(False, error=err, duration_ms=_ms(start))

    try:
        from sqlalchemy import text
        os.makedirs(os.path.dirname(abs_out) or ws, exist_ok=True)
        sql_clean = sql.strip().rstrip(";")
        async with engine.connect() as conn:  # type: ignore[union-attr]
            result = await conn.execute(text(sql_clean))
            cols = list(result.keys())
            row_count = 0
            with open(abs_out, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(cols)
                for r in result.fetchall():
                    writer.writerow([_csv_cell(v) for v in r])
                    row_count += 1
        return make_result(
            True,
            result={
                "path":      abs_out,
                "row_count": row_count,
                "columns":   cols,
            },
            duration_ms=_ms(start),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("db_query_to_csv error: %s", exc)
        return make_result(False, error=str(exc), duration_ms=_ms(start))


async def db_execute(sql: str) -> Dict[str, Any]:
    """Execute INSERT / UPDATE / DELETE / DDL. HIGH risk — requires
    user approval via the popup before the SQL runs."""
    start = time.monotonic()
    engine, err = await _get_session_engine()
    if err:
        return make_result(False, error=err, duration_ms=_ms(start))
    try:
        from sqlalchemy import text
        async with engine.begin() as conn:  # type: ignore[union-attr]
            result = await conn.execute(text(sql))
            rowcount = result.rowcount if hasattr(result, "rowcount") else -1
        return make_result(True, result={"rows_affected": rowcount},
                           duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        logger.error("db_execute error: %s", exc)
        return make_result(False, error=str(exc), duration_ms=_ms(start))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_jsonable(v: Any) -> Any:
    """Coerce DB-driver-native types to something json.dumps handles.
    Covers Decimal, datetime, date, bytes — the three that bite us
    most often on Postgres + SQLite roundtrips."""
    import datetime
    from decimal import Decimal
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, Decimal):
        return str(v)
    if isinstance(v, (datetime.datetime, datetime.date, datetime.time)):
        return v.isoformat()
    if isinstance(v, bytes):
        return v.hex()
    return str(v)


def _csv_cell(v: Any) -> Any:
    """CSV-cell coercion. Same shape as _to_jsonable but None becomes
    empty string (which is the convention pandas / Excel expect)."""
    if v is None:
        return ""
    return _to_jsonable(v)


# ── Tool schemas ──────────────────────────────────────────────────────────────

_SCHEMA_LIST_CONNECTIONS: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "db_list_connections",
        "description": (
            "Return whether the current session has a database attached. "
            "Call this FIRST when the user asks anything that might need "
            "the DB — if connected=false, suggest the + → Connect "
            "database flow instead of guessing."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

_SCHEMA_LIST_TABLES: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "db_list_tables",
        "description": "List every table in the session's attached database.",
        "parameters": {"type": "object", "properties": {}},
    },
}

_SCHEMA_DESCRIBE: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "db_describe_table",
        "description": "Show column names + types + nullability for one table.",
        "parameters": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "Table name."},
            },
            "required": ["table"],
        },
    },
}

_SCHEMA_QUERY: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "db_query",
        "description": (
            "Run a SELECT against the session's database. Returns up to "
            "`limit` rows (default 1000). Write/DDL keywords are rejected — "
            "use db_execute for those (which goes through user approval)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql":   {"type": "string", "description": "SELECT statement."},
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return. Defaults to 1000.",
                    "default": 1000,
                },
            },
            "required": ["sql"],
        },
    },
}

_SCHEMA_QUERY_TO_CSV: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "db_query_to_csv",
        "description": (
            "Run a SELECT and write the rows to a CSV file in the session "
            "workspace. Use this in Analytic mode when the next step is "
            "pandas / Chart.js — much cheaper than dragging 50K rows "
            "through the LLM context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "SELECT statement (no LIMIT — write the whole result).",
                },
                "output_path": {
                    "type": "string",
                    "description": (
                        "Relative path inside the session workspace, e.g. "
                        "'q1_revenue.csv'. Absolute paths and '..' escapes "
                        "are rejected."
                    ),
                },
            },
            "required": ["sql", "output_path"],
        },
    },
}

_SCHEMA_EXECUTE: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "db_execute",
        "description": (
            "Run a write SQL statement (INSERT, UPDATE, DELETE, DDL). "
            "Requires user approval before execution."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "SQL statement."},
            },
            "required": ["sql"],
        },
    },
}


# ── Export to ToolRegistry ────────────────────────────────────────────────────

DB_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    (db_list_connections, _SCHEMA_LIST_CONNECTIONS, "low"),
    (db_list_tables,      _SCHEMA_LIST_TABLES,      "low"),
    (db_describe_table,   _SCHEMA_DESCRIBE,         "low"),
    (db_query,            _SCHEMA_QUERY,            "low"),
    (db_query_to_csv,     _SCHEMA_QUERY_TO_CSV,     "low"),
    (db_execute,          _SCHEMA_EXECUTE,          "high"),
]
