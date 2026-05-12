"""
OpenTeddy Tracker
Async SQLite persistence layer for tasks, subtasks, skills, invocations,
and commercial API usage records.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import aiosqlite

from config import config
from models import (
    SCHEMA_SQL,
    AgentRole,
    SkillInvocation,
    SkillMetadata,
    SkillStatus,
    SubTask,
    TaskRequest,
    TaskResult,
    TaskStatus,
)

logger = logging.getLogger(__name__)

# ── Claude pricing table ──────────────────────────────────────────────────────
# (input_price_per_token, output_price_per_token) in USD
_CLAUDE_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus":   (15.0  / 1_000_000, 75.0 / 1_000_000),
    "claude-sonnet": (3.0   / 1_000_000, 15.0 / 1_000_000),
    "claude-haiku":  (0.80  / 1_000_000, 4.0  / 1_000_000),
}


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Return an estimated USD cost for a model call."""
    model_lower = model.lower()
    for prefix, (p_in, p_out) in _CLAUDE_PRICING.items():
        if prefix in model_lower:
            return round(tokens_in * p_in + tokens_out * p_out, 8)
    return 0.0  # Ollama / unknown — local, no cost


class Tracker:
    """Async SQLite tracker.  Use as an async context manager or call open()/close()."""

    def __init__(self, db_path: str = "") -> None:
        self.db_path = db_path or config.db_path
        self._db: Optional[aiosqlite.Connection] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def open(self) -> None:
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()
        # Migrate existing DBs that pre-date the usage_records columns
        await self._migrate_usage_columns()
        logger.info("Tracker opened: %s", self.db_path)

    async def _migrate_usage_columns(self) -> None:
        """Add new columns for pre-existing databases.

        SQLite does not support ADD COLUMN IF NOT EXISTS, so we catch the
        OperationalError when a column already exists.
        """
        migrations = [
            "ALTER TABLE usage_records ADD COLUMN session_id TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE usage_records ADD COLUMN model_provider TEXT NOT NULL DEFAULT 'ollama'",
            "ALTER TABLE usage_records ADD COLUMN task_description TEXT DEFAULT ''",
            # #6 Auto-benchmark — capture wall time + measured t/s on
            # every Ollama call so Settings can surface "qwen3.5:2b
            # runs at 18 t/s on your machine, consider qwen3.5:0.8b".
            # tokens_per_sec is denormalised so the stats query stays a
            # single fast aggregate instead of needing per-row math.
            "ALTER TABLE usage_records ADD COLUMN duration_ms INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE usage_records ADD COLUMN tokens_per_sec REAL NOT NULL DEFAULT 0",
            "ALTER TABLE tasks ADD COLUMN session_id TEXT",
            # Session mode selector (chat / code / analytic). Default 'code'
            # so existing sessions keep their full-autonomy behavior.
            "ALTER TABLE sessions ADD COLUMN mode TEXT NOT NULL DEFAULT 'code'",
            # Per-session workspace override. Null = use global default.
            "ALTER TABLE sessions ADD COLUMN workspace_dir TEXT",
            # Privacy: when 1, this session refuses to call the
            # Anthropic (Claude) API. Auto-escalation + Let-Claude-fix
            # are both disabled. Intended for Analytic sessions
            # handling customer data / PII.
            "ALTER TABLE sessions ADD COLUMN local_only INTEGER NOT NULL DEFAULT 0",
            # Per-session DB connection (Analytic mode "Connect database"
            # flow). The URL is treated as a secret — masked in UI and
            # redacted in /admin/diagnostics zips. db_kind names the
            # SQLAlchemy driver family (postgres / mysql / sqlite /
            # mssql / oracle / duckdb). db_label is the friendly chip
            # text auto-derived from URL host+db on connect. All three
            # default to empty so existing rows match a fresh session.
            "ALTER TABLE sessions ADD COLUMN db_kind  TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE sessions ADD COLUMN db_url   TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE sessions ADD COLUMN db_label TEXT NOT NULL DEFAULT ''",
        ]
        for sql in migrations:
            try:
                await self.db.execute(sql)
                await self.db.commit()
            except Exception:  # noqa: BLE001
                pass  # Column already exists — safe to ignore

        # The session_id index has to live here (not in SCHEMA_SQL) because
        # on pre-existing DBs the column is only added by the ALTER above.
        try:
            await self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id)"
            )
            await self.db.commit()
        except Exception:  # noqa: BLE001
            pass

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> "Tracker":
        await self.open()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Tracker is not open. Call await tracker.open() first.")
        return self._db

    # ── Task CRUD ─────────────────────────────────────────────────────────────

    async def create_task(self, req: TaskRequest) -> None:
        await self.db.execute(
            "INSERT INTO tasks(id, goal, context, status, priority, created_at, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                req.id,
                req.goal,
                json.dumps(req.context),
                TaskStatus.PENDING.value,
                req.priority,
                req.created_at.isoformat(),
                req.session_id,
            ),
        )
        await self.db.commit()
        # Bump session updated_at so the session list can order by recency.
        if req.session_id:
            await self.db.execute(
                "UPDATE sessions SET updated_at=? WHERE id=?",
                (datetime.utcnow().isoformat(), req.session_id),
            )
            await self.db.commit()

    async def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        summary: Optional[str] = None,
    ) -> None:
        completed_at = datetime.utcnow().isoformat() if status in (
            TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.ESCALATED
        ) else None
        await self.db.execute(
            "UPDATE tasks SET status=?, summary=?, completed_at=? WHERE id=?",
            (status.value, summary, completed_at, task_id),
        )
        await self.db.commit()

    async def get_task(self, task_id: str) -> Optional[dict]:
        async with self.db.execute(
            "SELECT * FROM tasks WHERE id=?", (task_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def list_tasks(
        self, limit: int = 50, session_id: Optional[str] = None,
    ) -> List[dict]:
        if session_id:
            async with self.db.execute(
                "SELECT * FROM tasks WHERE session_id=? "
                "ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with self.db.execute(
                "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
            ) as cur:
                rows = await cur.fetchall()
        return [dict(r) for r in rows]

    # ── Session CRUD ──────────────────────────────────────────────────────────

    async def create_session(
        self, session_id: str, title: str, mode: str = "code",
    ) -> None:
        now = datetime.utcnow().isoformat()
        await self.db.execute(
            "INSERT OR IGNORE INTO sessions(id, title, mode, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, title, mode, now, now),
        )
        await self.db.commit()

    async def list_sessions(self, limit: int = 50) -> List[dict]:
        async with self.db.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

    async def get_session(self, session_id: str) -> Optional[dict]:
        async with self.db.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,)
        ) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None

    async def rename_session(self, session_id: str, title: str) -> None:
        await self.db.execute(
            "UPDATE sessions SET title=?, updated_at=? WHERE id=?",
            (title, datetime.utcnow().isoformat(), session_id),
        )
        await self.db.commit()

    async def update_session_mode(self, session_id: str, mode: str) -> None:
        """Change the mode on an existing session (Chat / Code / Analytic)."""
        await self.db.execute(
            "UPDATE sessions SET mode=?, updated_at=? WHERE id=?",
            (mode, datetime.utcnow().isoformat(), session_id),
        )
        await self.db.commit()

    async def update_session_workspace(
        self, session_id: str, workspace_dir: Optional[str],
    ) -> None:
        """Point this session at a specific workspace directory, or clear
        the override (pass None) to fall back to config.agent_workspace_dir.

        The caller is responsible for validating the path — we accept
        any string, including one that doesn't exist yet (so the user
        can pre-configure a session, then create the dir later)."""
        await self.db.execute(
            "UPDATE sessions SET workspace_dir=?, updated_at=? WHERE id=?",
            (workspace_dir, datetime.utcnow().isoformat(), session_id),
        )
        await self.db.commit()

    async def update_session_local_only(
        self, session_id: str, local_only: bool,
    ) -> None:
        """Flip the privacy guardrail on/off for a session. When True,
        orchestrator skips Claude escalation; Let-Claude-fix endpoint
        returns 403. Stored as 0/1 integer for SQLite portability."""
        await self.db.execute(
            "UPDATE sessions SET local_only=?, updated_at=? WHERE id=?",
            (1 if local_only else 0, datetime.utcnow().isoformat(), session_id),
        )
        await self.db.commit()

    # ── Per-session DB connection (Analytic mode "Connect database") ───────
    #
    # These three methods own the kind/url/label trio on sessions. The
    # URL is the only true secret here — kind is just a router (postgres
    # / mysql / sqlite / …), label is the friendly chip text shown in
    # the session UI. Callers retrieve the URL via get_session_db_connection
    # only at point of use (db_tool building an engine); the URL never
    # gets serialised back to the UI verbatim — only kind + label.

    async def set_session_db_connection(
        self,
        session_id: str,
        kind: str,
        url: str,
        label: str,
    ) -> None:
        """Attach a DB connection to the session. Overwrites any
        previously-set connection on the same session — switching
        databases mid-session means dropping the old engine and
        building a fresh one keyed off the new URL."""
        await self.db.execute(
            "UPDATE sessions SET db_kind=?, db_url=?, db_label=?, updated_at=? "
            "WHERE id=?",
            (kind, url, label, datetime.utcnow().isoformat(), session_id),
        )
        await self.db.commit()

    async def get_session_db_connection(
        self, session_id: str,
    ) -> Optional[dict]:
        """Return ``{kind, url, label}`` for the session, or None if
        no DB has been attached. The URL is returned verbatim — only
        call this from server-side code (tool execution, internal
        engine builder); never expose it to the UI."""
        async with self.db.execute(
            "SELECT db_kind, db_url, db_label FROM sessions WHERE id=?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        kind, url, label = row[0], row[1], row[2]
        if not (kind and url):
            return None
        return {"kind": kind, "url": url, "label": label}

    async def clear_session_db_connection(self, session_id: str) -> None:
        """Detach the DB. Same effect as set_session_db_connection with
        empty strings; named separately so call-sites read clearly."""
        await self.db.execute(
            "UPDATE sessions SET db_kind='', db_url='', db_label='', "
            "updated_at=? WHERE id=?",
            (datetime.utcnow().isoformat(), session_id),
        )
        await self.db.commit()

    async def delete_session(self, session_id: str) -> None:
        """Remove the session row. Tasks stay but their session_id dangles —
        they will no longer appear in the filtered task list."""
        await self.db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        await self.db.commit()

    # ── SubTask CRUD ──────────────────────────────────────────────────────────

    async def create_subtask(self, subtask: SubTask) -> None:
        await self.db.execute(
            "INSERT INTO subtasks(id, parent_task_id, description, skill_hint, "
            "agent, order_idx, status, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                subtask.id,
                subtask.parent_task_id,
                subtask.description,
                subtask.skill_hint,
                subtask.agent.value,
                subtask.order,
                subtask.status.value,
                subtask.confidence,
                subtask.created_at.isoformat(),
            ),
        )
        await self.db.commit()

    async def update_subtask(self, subtask: SubTask) -> None:
        completed_at = subtask.completed_at.isoformat() if subtask.completed_at else None
        await self.db.execute(
            "UPDATE subtasks SET status=?, result=?, confidence=?, error=?, "
            "completed_at=? WHERE id=?",
            (
                subtask.status.value,
                subtask.result,
                subtask.confidence,
                subtask.error,
                completed_at,
                subtask.id,
            ),
        )
        await self.db.commit()

    async def get_subtasks(self, task_id: str) -> List[SubTask]:
        async with self.db.execute(
            "SELECT * FROM subtasks WHERE parent_task_id=? ORDER BY order_idx",
            (task_id,),
        ) as cur:
            rows = await cur.fetchall()
        result = []
        for r in rows:
            result.append(
                SubTask(
                    id=r["id"],
                    parent_task_id=r["parent_task_id"],
                    description=r["description"],
                    skill_hint=r["skill_hint"],
                    agent=AgentRole(r["agent"]),
                    order=r["order_idx"],
                    status=TaskStatus(r["status"]),
                    result=r["result"],
                    confidence=r["confidence"],
                    error=r["error"],
                    created_at=datetime.fromisoformat(r["created_at"]),
                    completed_at=datetime.fromisoformat(r["completed_at"])
                    if r["completed_at"]
                    else None,
                )
            )
        return result

    # ── Skill CRUD ────────────────────────────────────────────────────────────

    async def upsert_skill(self, skill: SkillMetadata) -> None:
        now = datetime.utcnow().isoformat()
        await self.db.execute(
            "INSERT INTO skills(name, description, code, version, status, "
            "success_count, failure_count, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET "
            "description=excluded.description, code=excluded.code, "
            "version=excluded.version, status=excluded.status, "
            "success_count=excluded.success_count, "
            "failure_count=excluded.failure_count, "
            "updated_at=excluded.updated_at",
            (
                skill.name,
                skill.description,
                skill.code,
                skill.version,
                skill.status.value,
                skill.success_count,
                skill.failure_count,
                skill.created_at.isoformat(),
                now,
            ),
        )
        await self.db.commit()

    async def get_skill(self, name: str) -> Optional[SkillMetadata]:
        async with self.db.execute(
            "SELECT * FROM skills WHERE name=?", (name,)
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return None
        return SkillMetadata(
            name=row["name"],
            description=row["description"],
            code=row["code"],
            version=row["version"],
            status=SkillStatus(row["status"]),
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def list_skills(self, status: Optional[SkillStatus] = None) -> List[SkillMetadata]:
        if status:
            async with self.db.execute(
                "SELECT * FROM skills WHERE status=? ORDER BY success_count DESC",
                (status.value,),
            ) as cur:
                rows = await cur.fetchall()
        else:
            async with self.db.execute(
                "SELECT * FROM skills ORDER BY success_count DESC"
            ) as cur:
                rows = await cur.fetchall()
        return [
            SkillMetadata(
                name=r["name"],
                description=r["description"],
                code=r["code"],
                version=r["version"],
                status=SkillStatus(r["status"]),
                success_count=r["success_count"],
                failure_count=r["failure_count"],
                created_at=datetime.fromisoformat(r["created_at"]),
                updated_at=datetime.fromisoformat(r["updated_at"]),
            )
            for r in rows
        ]

    async def record_skill_invocation(self, inv: SkillInvocation) -> None:
        await self.db.execute(
            "INSERT INTO skill_invocations(id, skill_name, subtask_id, input_data, "
            "output_data, success, duration_ms, invoked_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                inv.id,
                inv.skill_name,
                inv.subtask_id,
                json.dumps(inv.input_data),
                inv.output_data,
                int(inv.success),
                inv.duration_ms,
                inv.invoked_at.isoformat(),
            ),
        )
        # Bump counters on the skill record
        if inv.success:
            await self.db.execute(
                "UPDATE skills SET success_count = success_count + 1 WHERE name=?",
                (inv.skill_name,),
            )
        else:
            await self.db.execute(
                "UPDATE skills SET failure_count = failure_count + 1 WHERE name=?",
                (inv.skill_name,),
            )
        await self.db.commit()

    async def promote_skill_if_ready(self, skill_name: str) -> bool:
        """Promote a TESTING skill to ACTIVE when success threshold is met."""
        skill = await self.get_skill(skill_name)
        if not skill or skill.status != SkillStatus.TESTING:
            return False
        if skill.success_count >= config.skill_promotion_threshold:
            await self.db.execute(
                "UPDATE skills SET status=? WHERE name=?",
                (SkillStatus.ACTIVE.value, skill_name),
            )
            await self.db.commit()
            logger.info("Skill '%s' promoted to ACTIVE.", skill_name)
            return True
        return False

    # ── Usage records ─────────────────────────────────────────────────────────

    async def record_usage(
        self,
        task_id: str,
        model: str,
        model_provider: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        task_description: str = "",
        session_id: str = "",
        cost_usd: Optional[float] = None,
        # #6 Auto-benchmark: duration of THIS call and the resulting
        # tokens/sec. Both default to 0 so legacy callers (or paths
        # without timing info) still record fine — the model_perf_stats
        # query just filters those rows out.
        duration_ms: int = 0,
        tokens_per_sec: float = 0.0,
    ) -> None:
        """
        Persist a single API call record to usage_records.
        If *cost_usd* is None the cost is estimated from the model pricing table.
        """
        if cost_usd is None:
            cost_usd = _estimate_cost(model, tokens_in, tokens_out)

        record_id   = str(uuid.uuid4())
        created_at  = datetime.utcnow().isoformat()

        try:
            await self.db.execute(
                "INSERT INTO usage_records "
                "(id, task_id, session_id, model, model_provider, "
                "tokens_in, tokens_out, cost_usd, task_description, "
                "duration_ms, tokens_per_sec, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record_id,
                    task_id,
                    session_id,
                    model,
                    model_provider,
                    tokens_in,
                    tokens_out,
                    cost_usd,
                    task_description[:500] if task_description else "",
                    int(duration_ms or 0),
                    float(tokens_per_sec or 0.0),
                    created_at,
                ),
            )
            await self.db.commit()
        except Exception as exc:  # noqa: BLE001
            logger.error("record_usage failed: %s", exc)

    async def get_model_perf_stats(self, limit_per_model: int = 50) -> List[dict]:
        """Aggregate per-model performance over the most recent N
        recorded calls. Used by the Settings page (#6) to surface
        "qwen3.5:2b · 18 t/s avg over 45 calls" + tier suggestions.

        Only counts rows where tokens_per_sec > 0 (we recorded perf
        data for them — older rows have it as the default 0).
        """
        # Simple SELECT + Python-side bucketing — avoids SQL window
        # functions which aren't available on every aiosqlite build.
        # 4000 rows is plenty even at heavy daily use; per-model cap
        # is enforced after the fact.
        async with self.db.execute(
            "SELECT model, model_provider, tokens_per_sec, duration_ms "
            "FROM usage_records "
            "WHERE tokens_per_sec > 0 "
            "ORDER BY created_at DESC LIMIT 4000"
        ) as cur:
            rows = await cur.fetchall()

        from collections import defaultdict
        buckets: dict[str, dict] = defaultdict(
            lambda: {"tps": [], "dur": [], "provider": ""}
        )
        for r in rows:
            cols = list(r)
            m, provider, tps, dur = cols[0], cols[1], float(cols[2] or 0), int(cols[3] or 0)
            if not m or tps <= 0:
                continue
            b = buckets[m]
            if len(b["tps"]) >= limit_per_model:
                continue
            b["tps"].append(tps)
            b["dur"].append(dur)
            if provider:
                b["provider"] = provider

        out: list[dict] = []
        for m, b in buckets.items():
            n = len(b["tps"])
            if n == 0:
                continue
            avg_tps = sum(b["tps"]) / n
            avg_dur = sum(b["dur"]) / n if b["dur"] else 0
            out.append({
                "model":          m,
                "model_provider": b["provider"],
                "samples":        n,
                "avg_tokens_per_sec": round(avg_tps, 2),
                "avg_duration_ms":    int(avg_dur),
            })
        out.sort(key=lambda x: x["avg_tokens_per_sec"])  # slowest first
        return out

    async def get_usage_paginated(
        self,
        page: int = 1,
        page_size: int = 20,
        provider_filter: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> Dict:
        """
        Paginated usage records, newest first.

        Returns:
          {items, total, page, page_size, total_pages}
        """
        conditions: list[str] = []
        params: list  = []

        if provider_filter:
            conditions.append("model_provider = ?")
            params.append(provider_filter)
        if date_from:
            conditions.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("created_at <= ?")
            params.append(date_to + "T23:59:59")

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        # Total count
        async with self.db.execute(
            f"SELECT COUNT(*) FROM usage_records {where}", params
        ) as cur:
            row = await cur.fetchone()
            total = row[0] if row else 0

        total_pages = max(1, (total + page_size - 1) // page_size)
        offset      = (page - 1) * page_size

        async with self.db.execute(
            f"SELECT * FROM usage_records {where} "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params + [page_size, offset],
        ) as cur:
            rows = await cur.fetchall()

        items = [dict(r) for r in rows]
        return {
            "items":       items,
            "total":       total,
            "page":        page,
            "page_size":   page_size,
            "total_pages": total_pages,
        }

    async def get_usage_summary(self) -> Dict:
        """
        Aggregate usage statistics across all records.

        Returns:
          {
            total_calls, commercial_calls,
            total_tokens_in, total_tokens_out,
            total_cost_usd, commercial_cost_usd,
            cost_by_model, cost_by_day,   # last 30 days
            escalation_count
          }
        """
        # ── Totals ────────────────────────────────────────────────────────────
        async with self.db.execute(
            "SELECT COUNT(*) as total_calls, "
            "SUM(tokens_in) as total_in, SUM(tokens_out) as total_out, "
            "SUM(cost_usd) as total_cost "
            "FROM usage_records"
        ) as cur:
            totals = dict(await cur.fetchone() or {})

        # ── Anthropic only ────────────────────────────────────────────────────
        async with self.db.execute(
            "SELECT COUNT(*) as comm_calls, SUM(cost_usd) as comm_cost "
            "FROM usage_records WHERE model_provider = 'anthropic'"
        ) as cur:
            commercial = dict(await cur.fetchone() or {})

        # ── Local (non-anthropic) token volume ────────────────────────────────
        # Any record not attributed to Anthropic is treated as "ran on a local
        # model" for the purposes of the GPT-4-baseline savings estimate.
        async with self.db.execute(
            "SELECT COUNT(*) as local_calls, "
            "SUM(tokens_in) as local_in, SUM(tokens_out) as local_out "
            "FROM usage_records "
            "WHERE model_provider IS NULL OR model_provider != 'anthropic'"
        ) as cur:
            local = dict(await cur.fetchone() or {})

        # ── Escalation count (usage records from anthropic == escalations) ────
        escalation_count = commercial.get("comm_calls") or 0

        # ── Cost by model ─────────────────────────────────────────────────────
        async with self.db.execute(
            "SELECT model, SUM(cost_usd) as cost FROM usage_records GROUP BY model"
        ) as cur:
            rows = await cur.fetchall()
        cost_by_model = {r["model"]: round(r["cost"] or 0.0, 6) for r in rows}

        # ── Cost by day (last 30 days) ────────────────────────────────────────
        async with self.db.execute(
            "SELECT substr(created_at, 1, 10) as day, SUM(cost_usd) as cost "
            "FROM usage_records "
            "WHERE created_at >= datetime('now', '-30 days') "
            "GROUP BY day ORDER BY day ASC"
        ) as cur:
            rows = await cur.fetchall()
        cost_by_day = [
            {"date": r["day"], "cost": round(r["cost"] or 0.0, 6)}
            for r in rows
        ]

        local_in  = local.get("local_in")  or 0
        local_out = local.get("local_out") or 0
        # GPT-4 (8k) public API pricing as of 2024: $30/1M input, $60/1M output.
        # Used as a "what would this have cost on GPT-4?" baseline for the
        # savings callout on the Usage page.
        gpt4_in_rate  = 30.0 / 1_000_000
        gpt4_out_rate = 60.0 / 1_000_000
        gpt4_equiv_cost = local_in * gpt4_in_rate + local_out * gpt4_out_rate

        return {
            "total_calls":        totals.get("total_calls") or 0,
            "commercial_calls":   commercial.get("comm_calls") or 0,
            "total_tokens_in":    totals.get("total_in") or 0,
            "total_tokens_out":   totals.get("total_out") or 0,
            "total_cost_usd":     round(totals.get("total_cost") or 0.0, 6),
            "commercial_cost_usd": round(commercial.get("comm_cost") or 0.0, 6),
            "cost_by_model":      cost_by_model,
            "cost_by_day":        cost_by_day,
            "escalation_count":   escalation_count,
            "local_calls":        local.get("local_calls") or 0,
            "local_tokens_in":    local_in,
            "local_tokens_out":   local_out,
            "gpt4_equiv_cost_usd": round(gpt4_equiv_cost, 6),
            "gpt4_price_in_per_mtok":  30.0,
            "gpt4_price_out_per_mtok": 60.0,
        }

    # ── Performance / time-to-result statistics ─────────────────────────────
    # User-facing observability so people can see "Chat: avg 2s, Code: 14s,
    # Analytic: 8s" right inside the Usage tab. Doubles as feedback data the
    # user can screenshot + send to dev when something feels slow.
    #
    # Numbers come from the tasks table only — created_at / completed_at
    # bracket the entire orchestrator.run() lifecycle (plan → execute →
    # escalate → summarise → memory write). No per-stage breakdown yet;
    # that would need additional timestamp columns and isn't worth the
    # schema churn until a user actually asks "why is the code phase slow".

    async def get_perf_stats(self, days: int = 30, slowest_n: int = 10) -> Dict:
        """Return per-mode duration aggregates + slowest N tasks.

        Returned shape:
          {
            "window_days":   30,
            "modes": {
                "chat":     {"count": N, "mean_ms": X, "p50_ms": X,
                             "p95_ms": X, "max_ms": X},
                "code":     {...same...},
                "analytic": {...same...},
            },
            "slowest": [
                {"id": "...", "goal": "first 80 chars…", "mode": "code",
                 "duration_ms": 38400, "completed_at": "2026-..."},
                ...
            ],
            "total_completed": N,   # tasks completed in the window
          }

        Percentiles computed in Python after fetching the durations —
        SQLite has no native percentile_cont(). For N up to a few
        thousand this is cheap (sorting one column).
        """
        # Date window. ISO-formatted strings comparable directly against
        # the TEXT created_at column.
        since = (datetime.utcnow() - timedelta(days=int(days))).isoformat()

        # Per-mode duration vectors. Pull (mode, duration_ms) only —
        # everything else is computed Python-side.
        async with self.db.execute(
            "SELECT COALESCE(s.mode, 'code') AS mode, "
            "       t.created_at AS created_at, "
            "       t.completed_at AS completed_at "
            "FROM tasks t "
            "LEFT JOIN sessions s ON s.id = t.session_id "
            "WHERE t.completed_at IS NOT NULL "
            "  AND t.completed_at >= ? "
            "  AND t.status = 'completed'",
            (since,),
        ) as cur:
            rows = await cur.fetchall()

        # Bucket durations by mode.
        per_mode: Dict[str, List[int]] = {
            "chat": [], "code": [], "analytic": [],
        }
        for r in rows:
            mode = (r["mode"] or "code").lower()
            if mode not in per_mode:
                per_mode[mode] = []
            try:
                t0 = datetime.fromisoformat(r["created_at"])
                t1 = datetime.fromisoformat(r["completed_at"])
                ms = int((t1 - t0).total_seconds() * 1000)
                if ms >= 0:
                    per_mode[mode].append(ms)
            except (TypeError, ValueError):
                # Malformed timestamps — skip rather than blow up the
                # whole stats endpoint over a single bad row.
                continue

        def _pct(sorted_xs: List[int], p: float) -> int:
            """Linear-interpolation percentile against a SORTED list.
            Returns 0 for empty input (so the UI doesn't blow up on
            modes the user has never tried)."""
            if not sorted_xs:
                return 0
            if len(sorted_xs) == 1:
                return sorted_xs[0]
            k = (len(sorted_xs) - 1) * (p / 100.0)
            f = int(k)
            c = min(f + 1, len(sorted_xs) - 1)
            if f == c:
                return sorted_xs[f]
            return int(sorted_xs[f] + (k - f) * (sorted_xs[c] - sorted_xs[f]))

        modes_out: Dict[str, Dict[str, int]] = {}
        for mode, vals in per_mode.items():
            if not vals:
                modes_out[mode] = {
                    "count": 0, "mean_ms": 0, "p50_ms": 0,
                    "p95_ms": 0, "max_ms": 0,
                }
                continue
            sorted_vals = sorted(vals)
            modes_out[mode] = {
                "count":   len(sorted_vals),
                "mean_ms": int(sum(sorted_vals) / len(sorted_vals)),
                "p50_ms":  _pct(sorted_vals, 50),
                "p95_ms":  _pct(sorted_vals, 95),
                "max_ms":  sorted_vals[-1],
            }

        # Slowest N tasks in the window. Order by duration desc on the
        # SQL side so we don't have to load every task. (Couldn't do this
        # in the same query above because we need the mode label from
        # the join AND we wanted the Python-side bucketing.)
        async with self.db.execute(
            "SELECT t.id, t.goal, "
            "       COALESCE(s.mode, 'code') AS mode, "
            "       t.created_at, t.completed_at, "
            "       CAST("
            "         (strftime('%s', t.completed_at) - strftime('%s', t.created_at)) "
            "         * 1000 AS INTEGER"
            "       ) AS duration_ms "
            "FROM tasks t "
            "LEFT JOIN sessions s ON s.id = t.session_id "
            "WHERE t.completed_at IS NOT NULL "
            "  AND t.completed_at >= ? "
            "  AND t.status = 'completed' "
            "ORDER BY duration_ms DESC "
            "LIMIT ?",
            (since, int(slowest_n)),
        ) as cur:
            slow_rows = await cur.fetchall()
        slowest: List[Dict] = []
        for r in slow_rows:
            slowest.append({
                "id":           r["id"],
                "goal":         (r["goal"] or "")[:120],
                "mode":         (r["mode"] or "code").lower(),
                "duration_ms":  int(r["duration_ms"] or 0),
                "completed_at": r["completed_at"],
            })

        total_completed = sum(m["count"] for m in modes_out.values())
        return {
            "window_days":     int(days),
            "modes":           modes_out,
            "slowest":         slowest,
            "total_completed": total_completed,
        }
