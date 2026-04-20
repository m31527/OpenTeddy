"""
OpenTeddy Main Entry Point
FastAPI server exposing the multi-agent system over HTTP.
Run with: uvicorn main:app --reload
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from config import config
from escalation import EscalationAgent
from executor import Executor
from models import (
    RunRequest,
    RunResponse,
    SkillListResponse,
    StatusResponse,
    TaskRequest,
    TaskStatus,
)
from orchestrator import Orchestrator
from skill_factory import SkillFactory
from tracker import Tracker

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("openteddy")

# ── Application state ─────────────────────────────────────────────────────────
tracker: Tracker
skill_factory: SkillFactory
executor: Executor
escalation_agent: EscalationAgent
orchestrator: Orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:  # type: ignore[type-arg]
    """Open / close shared resources around the app's lifetime."""
    global tracker, skill_factory, executor, escalation_agent, orchestrator

    # Validate required config
    try:
        config.validate()
    except ValueError as exc:
        logger.warning("Config warning: %s", exc)

    # Initialise components
    tracker = Tracker()
    await tracker.open()

    skill_factory = SkillFactory(tracker)
    executor = Executor(tracker, skill_factory)
    escalation_agent = EscalationAgent(tracker)
    orchestrator = Orchestrator(tracker, executor, escalation_agent, skill_factory)

    logger.info("OpenTeddy is ready 🐻")
    yield

    # Teardown
    await executor.close()
    await orchestrator.close()
    await tracker.close()
    logger.info("OpenTeddy shut down cleanly.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="OpenTeddy",
    description="Self-growing multi-agent system: Gemma Orchestrator + Qwen Executor + Claude Escalation",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static UI
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    """Serve the dashboard."""
    index = os.path.join(_static_dir, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return FileResponse(__file__)  # fallback


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/run", response_model=RunResponse, status_code=202)
async def run_task(body: RunRequest) -> RunResponse:
    """
    Submit a new task to OpenTeddy.
    The task is executed synchronously (for simplicity); use background tasks
    for fire-and-forget production use.
    """
    req = TaskRequest(goal=body.goal, context=body.context, priority=body.priority)
    result = await orchestrator.run(req)
    return RunResponse(
        task_id=result.task_id,
        status=result.status,
        message=result.summary,
    )


@app.get("/tasks/{task_id}", response_model=StatusResponse)
async def get_task_status(task_id: str) -> StatusResponse:
    """Get the status and subtask breakdown for a task."""
    task = await tracker.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    subtasks = await tracker.get_subtasks(task_id)
    return StatusResponse(
        task_id=task_id,
        status=TaskStatus(task["status"]),
        subtasks=subtasks,
        summary=task.get("summary"),
    )


@app.get("/tasks", response_model=list)
async def list_tasks(limit: int = 20) -> list:
    """List recent tasks."""
    return await tracker.list_tasks(limit=limit)


@app.get("/skills", response_model=SkillListResponse)
async def list_skills() -> SkillListResponse:
    """List all known skills."""
    skills = await skill_factory.list_all_skills()
    return SkillListResponse(skills=skills)


@app.post("/skills/generate")
async def generate_skill(name: str, description: str) -> dict:
    """
    Manually trigger skill generation via Claude.
    Useful for bootstrapping the skill library.
    """
    try:
        skill = await skill_factory.generate_skill(name, description)
        return {
            "name": skill.name,
            "status": skill.status,
            "message": f"Skill '{name}' generated and saved.",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.api_host,
        port=config.api_port,
        reload=True,
        log_level="info",
    )
