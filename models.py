"""
OpenTeddy Data Models
Pydantic models and SQLite schema definitions shared across the system.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    ESCALATED = "escalated"


class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    EXECUTOR     = "executor"
    ESCALATION   = "escalation"
    SKILL_FACTORY = "skill_factory"


class SkillStatus(str, Enum):
    DRAFT    = "draft"
    TESTING  = "testing"
    ACTIVE   = "active"
    RETIRED  = "retired"


class SessionMode(str, Enum):
    """Per-session behavior selector (ChatGPT/Claude-style mode switch).

    Each mode ties to a different orchestrator plan prompt and a different
    executor tool exposure, so the user can declare intent explicitly
    instead of relying on the model to guess from the task description.
    """
    CHAT     = "chat"       # pure text reasoning, no tools exposed
    CODE     = "code"       # full autonomy: deploy/install/diagnose
    ANALYTIC = "analytic"   # coming-soon stub (csv/xlsx/json analysis)


# ── Core request / response ───────────────────────────────────────────────────

class TaskRequest(BaseModel):
    """Incoming task from a user or parent agent."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = Field(..., description="High-level goal in natural language.")
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    priority: int = Field(default=5, ge=1, le=10)
    session_id: Optional[str] = Field(
        default=None,
        description="Logical session / thread this task belongs to. Memory "
                    "retrieval is scoped to this session to prevent "
                    "cross-project contamination.",
    )
    mode: SessionMode = Field(
        default=SessionMode.CODE,
        description="Which behavior profile to run with. Controls the "
                    "orchestrator plan prompt and which tools the executor "
                    "exposes. Default is CODE (full autonomy) for back-compat.",
    )


class SubTask(BaseModel):
    """A single executable step produced by the Orchestrator."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_task_id: str
    description: str
    skill_hint: Optional[str] = None      # preferred skill name, if known
    agent: AgentRole = AgentRole.EXECUTOR
    order: int = 0
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class TaskResult(BaseModel):
    """Final aggregated result returned to the caller."""
    task_id: str
    status: TaskStatus
    summary: str
    subtasks: List[SubTask] = Field(default_factory=list)
    skills_used: List[str] = Field(default_factory=list)
    new_skills_created: List[str] = Field(default_factory=list)
    completed_at: datetime = Field(default_factory=datetime.utcnow)


# ── Skill models ──────────────────────────────────────────────────────────────

class SkillMetadata(BaseModel):
    """Metadata record stored in the DB for each skill."""
    name: str
    description: str
    code: str                             # Python source of the skill function
    version: int = 1
    status: SkillStatus = SkillStatus.DRAFT
    success_count: int = 0
    failure_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class SkillInvocation(BaseModel):
    """A single call to a skill — used for logging."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    skill_name: str
    subtask_id: str
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Optional[str] = None
    success: bool = False
    duration_ms: int = 0
    invoked_at: datetime = Field(default_factory=datetime.utcnow)


# ── Agent message passing ─────────────────────────────────────────────────────

class AgentMessage(BaseModel):
    """Internal message passed between agents."""
    sender: AgentRole
    recipient: AgentRole
    task_id: str
    subtask_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── API schemas ───────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    goal: str
    context: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5, ge=1, le=10)
    session_id: Optional[str] = None
    task_id: Optional[str] = Field(
        default=None,
        description="Client-supplied task id so the UI can cancel mid-flight "
                    "via POST /tasks/{id}/cancel. If omitted, the server "
                    "generates one.",
    )
    mode: Optional[SessionMode] = Field(
        default=None,
        description="Override the session's default mode just for this task. "
                    "Usually omitted — UI reads it from the active session.",
    )


class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "New session"
    mode: SessionMode = SessionMode.CODE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SessionListResponse(BaseModel):
    sessions: List[Session]


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None
    mode: Optional[SessionMode] = None


class RunResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str


class StatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    subtasks: List[SubTask]
    summary: Optional[str] = None


class SkillListResponse(BaseModel):
    skills: List[SkillMetadata]


# ── SQLite DDL (used by tracker.py) ──────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New session',
    mode        TEXT NOT NULL DEFAULT 'code',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
-- NOTE: for pre-existing DBs the `mode` column is added in
-- tracker._migrate_usage_columns, same pattern as tasks.session_id.

CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    goal        TEXT NOT NULL,
    context     TEXT DEFAULT '{}',
    status      TEXT NOT NULL DEFAULT 'pending',
    summary     TEXT,
    priority    INTEGER DEFAULT 5,
    created_at  TEXT NOT NULL,
    completed_at TEXT,
    session_id  TEXT
);

-- NOTE: the session_id index is created in tracker._migrate_usage_columns,
-- not here, because on a pre-existing DB the column is only added during
-- migration (ALTER TABLE runs after executescript).

CREATE TABLE IF NOT EXISTS subtasks (
    id             TEXT PRIMARY KEY,
    parent_task_id TEXT NOT NULL REFERENCES tasks(id),
    description    TEXT NOT NULL,
    skill_hint     TEXT,
    agent          TEXT NOT NULL DEFAULT 'executor',
    order_idx      INTEGER DEFAULT 0,
    status         TEXT NOT NULL DEFAULT 'pending',
    result         TEXT,
    confidence     REAL DEFAULT 1.0,
    error          TEXT,
    created_at     TEXT NOT NULL,
    completed_at   TEXT
);

CREATE TABLE IF NOT EXISTS skills (
    name          TEXT PRIMARY KEY,
    description   TEXT NOT NULL,
    code          TEXT NOT NULL,
    version       INTEGER DEFAULT 1,
    status        TEXT NOT NULL DEFAULT 'draft',
    success_count INTEGER DEFAULT 0,
    failure_count INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skill_invocations (
    id          TEXT PRIMARY KEY,
    skill_name  TEXT NOT NULL,
    subtask_id  TEXT NOT NULL,
    input_data  TEXT DEFAULT '{}',
    output_data TEXT,
    success     INTEGER DEFAULT 0,
    duration_ms INTEGER DEFAULT 0,
    invoked_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS usage_records (
    id               TEXT PRIMARY KEY,
    task_id          TEXT NOT NULL DEFAULT '',
    session_id       TEXT NOT NULL DEFAULT '',
    model            TEXT NOT NULL DEFAULT '',
    model_provider   TEXT NOT NULL DEFAULT 'ollama',
    tokens_in        INTEGER DEFAULT 0,
    tokens_out       INTEGER DEFAULT 0,
    cost_usd         REAL DEFAULT 0.0,
    task_description TEXT DEFAULT '',
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_usage_created_at
    ON usage_records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_usage_provider
    ON usage_records(model_provider);
"""
