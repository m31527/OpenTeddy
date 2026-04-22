"""
OpenTeddy Tracker
Async SQLite persistence layer for tasks, subtasks, skills, invocations,
and commercial API usage records.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
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
            "ALTER TABLE tasks ADD COLUMN session_id TEXT",
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

    async def create_session(self, session_id: str, title: str) -> None:
        now = datetime.utcnow().isoformat()
        await self.db.execute(
            "INSERT OR IGNORE INTO sessions(id, title, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
        await self.db.commit()

    async def list_sessions(self, limit: int = 50) -> List[dict]:
        async with self.db.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

    async def rename_session(self, session_id: str, title: str) -> None:
        await self.db.execute(
            "UPDATE sessions SET title=?, updated_at=? WHERE id=?",
            (title, datetime.utcnow().isoformat(), session_id),
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
                "tokens_in, tokens_out, cost_usd, task_description, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    created_at,
                ),
            )
            await self.db.commit()
        except Exception as exc:  # noqa: BLE001
            logger.error("record_usage failed: %s", exc)

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
        }
