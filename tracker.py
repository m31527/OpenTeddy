"""
OpenTeddy Tracker
Async SQLite persistence layer for tasks, subtasks, skills, and invocations.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Optional

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
        logger.info("Tracker opened: %s", self.db_path)

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
            "INSERT INTO tasks(id, goal, context, status, priority, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                req.id,
                req.goal,
                json.dumps(req.context),
                TaskStatus.PENDING.value,
                req.priority,
                req.created_at.isoformat(),
            ),
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

    async def list_tasks(self, limit: int = 50) -> List[dict]:
        async with self.db.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

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
