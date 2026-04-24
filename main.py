"""
OpenTeddy Main Entry Point
FastAPI server exposing the multi-agent system over HTTP.
Run with: uvicorn main:app --reload
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from approval_store import approval_store
from config import config
from escalation import EscalationAgent
from executor import Executor
from memory import MemoryManager
from models import (
    CreateSessionRequest,
    RunRequest,
    RunResponse,
    Session,
    SessionListResponse,
    SessionMode,
    SkillListResponse,
    StatusResponse,
    TaskRequest,
    TaskStatus,
)
from orchestrator import Orchestrator
from settings_store import SETTINGS_META, settings_store
from skill_factory import SkillFactory
from tool_registry import tool_registry
from tracker import Tracker

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("openteddy")

# ── Application state ─────────────────────────────────────────────────────────
tracker:          Tracker
skill_factory:    SkillFactory
executor:         Executor
escalation_agent: EscalationAgent
orchestrator:     Orchestrator
memory_manager:   MemoryManager

# ── WebSocket connection manager ──────────────────────────────────────────────

class _WSManager:
    """Tracks all active WebSocket connections and broadcasts events."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)

    async def broadcast(self, data: dict) -> None:
        """Send a JSON event to every connected client (fire-and-forget per client)."""
        import json as _json
        dead: set[WebSocket] = set()
        msg = _json.dumps(data, ensure_ascii=False)
        for ws in list(self._connections):
            try:
                await ws.send_text(msg)
            except Exception:  # noqa: BLE001
                dead.add(ws)
        self._connections -= dead


ws_manager = _WSManager()


# ── In-flight task registry ───────────────────────────────────────────────────
# Maps task_id → asyncio.Task so POST /tasks/{id}/cancel can interrupt a
# running orchestrator.run() coroutine (Stop button in the UI).
_running_tasks: dict[str, asyncio.Task] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:  # type: ignore[type-arg]
    """Open / close shared resources around the app's lifetime."""
    global tracker, skill_factory, executor, escalation_agent, orchestrator, memory_manager

    try:
        config.validate()
    except ValueError as exc:
        logger.warning("Config warning: %s", exc)

    # ── Settings store (must init before config.reload_from_store) ─────────────
    try:
        await settings_store.init()
        await config.reload_from_store(settings_store)
        logger.info("Settings loaded from DB.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("SettingsStore init warning: %s", exc)

    # Auto-register all tools
    try:
        tool_registry.auto_register_all()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tool auto-registration warning: %s", exc)

    # Ensure the agent workspace exists — Code / Analytic modes default
    # shell commands to run here. Created once at startup so the first
    # shell_exec_* call doesn't race to mkdir.
    #
    # Print the absolute path BIG AND LOUD so users can immediately see
    # where their clones / builds are landing, regardless of which dir
    # they started uvicorn from. Most deploy confusions trace back to
    # "I thought it was in a different agent-workspace".
    try:
        ws = os.path.abspath(config.agent_workspace_dir)
        os.makedirs(ws, exist_ok=True)
        logger.info("=" * 72)
        logger.info("OpenTeddy agent workspace: %s", ws)
        logger.info("  All git clones, builds, file writes default to this dir.")
        logger.info("  Override with AGENT_WORKSPACE_DIR env var or Settings → Agent Workspace.")
        logger.info("=" * 72)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not create agent workspace %s: %s",
                       config.agent_workspace_dir, exc)

    tracker = Tracker()
    await tracker.open()

    # Long-term memory (ChromaDB — gracefully degrades if not installed)
    memory_manager = MemoryManager(db_path=config.memory_db_path)
    await memory_manager.open()

    skill_factory    = SkillFactory(tracker)
    executor         = Executor(
        tracker,
        skill_factory,
        registry=tool_registry,
        ws_callback=ws_manager.broadcast,   # ← 接線 WebSocket 廣播
    )
    escalation_agent = EscalationAgent(tracker)
    orchestrator     = Orchestrator(
        tracker, executor, escalation_agent, skill_factory,
        memory=memory_manager,
    )

    logger.info(
        "OpenTeddy is ready 🐻  (%d tools registered, memory=%s)",
        len(tool_registry.list_tools()),
        "enabled" if memory_manager.is_available else "disabled",
    )
    yield

    await executor.close()
    await orchestrator.close()
    await memory_manager.close()
    await tracker.close()
    logger.info("OpenTeddy shut down cleanly.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="OpenTeddy",
    description="Self-growing multi-agent system: Gemma Orchestrator + Qwen Executor + Claude Escalation",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """WebSocket endpoint — streams tool_call / tool_result events to the UI."""
    await ws_manager.connect(ws)
    try:
        # Keep the connection alive; client sends pings if needed
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception:  # noqa: BLE001
        ws_manager.disconnect(ws)


@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    index = os.path.join(_static_dir, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return FileResponse(__file__)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    """Convenience route — browsers hit /favicon.ico by convention even when
    the HTML declares the icon elsewhere. Serve the one under /static so it
    matches the <link rel="icon"> tags in index.html."""
    path = os.path.join(_static_dir, "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/x-icon")
    raise HTTPException(status_code=404, detail="favicon not found")


@app.get("/health")
async def health() -> dict:
    return {
        "status":  "ok",
        "version": "0.3.0",
        "memory":  memory_manager.is_available,
    }


@app.get("/files")
async def download_file(path: str, session_id: Optional[str] = None) -> FileResponse:
    """Serve a file produced by the agent (Analytic HTML reports,
    written files, etc.) so the UI can show a download link instead of
    the user having to SSH in.

    Security model: the file MUST live under either
      - the global agent_workspace_dir, OR
      - the session's workspace_dir (when session_id is supplied and that
        session has an override)
    Any path that resolves outside those boundaries is rejected — we
    don't want /files turning into an arbitrary-file-read.
    """
    if not path:
        raise HTTPException(status_code=400, detail="`path` required")

    target = os.path.abspath(path)

    # Collect allowed root directories.
    allowed_roots: list[str] = [os.path.abspath(config.agent_workspace_dir)]
    if session_id:
        try:
            sess = await tracker.get_session(session_id)
            if sess and sess.get("workspace_dir"):
                allowed_roots.append(os.path.abspath(sess["workspace_dir"]))
        except Exception:  # noqa: BLE001
            pass

    # Path must be INSIDE one of the allowed roots.
    is_inside = any(
        target == root or target.startswith(root + os.sep)
        for root in allowed_roots
    )
    if not is_inside:
        logger.warning(
            "Refused /files download — path %s outside allowed roots %s",
            target, allowed_roots,
        )
        raise HTTPException(
            status_code=403,
            detail="Path is outside the agent workspace. Refusing to serve.",
        )

    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail="File not found")

    # Content-Disposition: pick inline for browsers to render .html
    # in a new tab, attachment for binary types.
    filename = os.path.basename(target)
    ext = os.path.splitext(filename)[1].lower()
    inline_exts = {".html", ".htm", ".txt", ".md", ".json", ".csv", ".log",
                   ".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf"}
    disposition = "inline" if ext in inline_exts else "attachment"
    return FileResponse(
        target,
        filename=filename,
        headers={"Content-Disposition": f'{disposition}; filename="{filename}"'},
    )


@app.get("/debug/workspace")
async def debug_workspace(session_id: Optional[str] = None) -> dict:
    """Return absolute path state so the user can verify where clones go.

    When ``session_id`` is supplied, also returns that session's
    workspace_dir override (or null if it inherits the global default)
    plus a listing of ITS effective workspace.

    Everything returned here is safe to display — absolute paths only,
    no secrets.
    """
    global_ws = os.path.abspath(config.agent_workspace_dir)

    # Figure out the effective workspace (session override vs global).
    effective_ws = global_ws
    session_override: Optional[str] = None
    if session_id:
        try:
            sess = await tracker.get_session(session_id)
            if sess and sess.get("workspace_dir"):
                session_override = sess["workspace_dir"]
                effective_ws = os.path.abspath(session_override)
        except Exception:  # noqa: BLE001
            pass

    exists = os.path.isdir(effective_ws)
    entries: list[dict] = []
    if exists:
        try:
            for name in sorted(os.listdir(effective_ws)):
                full = os.path.join(effective_ws, name)
                entries.append({
                    "name": name,
                    "is_dir": os.path.isdir(full),
                    "size": os.path.getsize(full) if os.path.isfile(full) else None,
                })
        except Exception:  # noqa: BLE001
            pass
    return {
        "agent_workspace_dir":   global_ws,
        "session_workspace_dir": session_override,
        "effective_workspace":   effective_ws,
        "exists": exists,
        "uvicorn_cwd":  os.getcwd(),
        "project_root": os.path.dirname(os.path.abspath(__file__)),
        "entry_count":  len(entries),
        "entries":      entries[:50],  # cap to keep the response small
    }


@app.post("/webhooks/{session_id}", status_code=202)
async def webhook_trigger(
    session_id: str,
    request: Request,
) -> dict:
    """External-trigger entrypoint for the agent-as-service pattern.

    POST a JSON payload here to fire a task in the given session —
    cron / Stripe / GitHub / your own backend can all hit this. The
    body shape is flexible:

      {"goal": "check for new orders"}          → uses that goal literally
      {"order_id": 1234, "amount": 50000}       → no `goal` key, so we
                                                   auto-wrap as "process
                                                   webhook payload: ..."
                                                   and stash the payload
                                                   in context.webhook_payload

    Security: if `webhook_secret` is configured (Settings → Notification
    Credentials), requests MUST include it either as an
    `X-OpenTeddy-Webhook-Secret` header or a `?secret=` query param, else
    401. If no secret is configured, the endpoint is open — the UI warns
    about this explicitly so it's an informed choice for localhost-only
    deployments.

    Returns 202 immediately with the spawned task_id; the task runs in
    the background. Progress is visible in the session's chat via the
    usual WebSocket stream.
    """
    # Header / query-param auth
    configured = (getattr(config, "webhook_secret", "") or "").strip()
    if configured:
        provided = (
            request.headers.get("x-openteddy-webhook-secret")
            or request.query_params.get("secret")
            or ""
        )
        if provided != configured:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Invalid or missing webhook secret. Supply it via "
                    "`X-OpenTeddy-Webhook-Secret` header or `?secret=` query."
                ),
            )

    # Session must exist — refuse to auto-create on webhook path, since
    # that would let a mis-sent webhook spawn random sessions.
    sess = await tracker.get_session(session_id)
    if not sess:
        raise HTTPException(
            status_code=404,
            detail=(
                "Session not found. Create it first in the UI (so the user "
                "has explicitly decided what workspace / mode / local_only "
                "settings apply), then point the webhook at its id."
            ),
        )

    # Parse body (JSON). If not JSON, treat as empty.
    body: dict = {}
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {"payload": body}
    except Exception:  # noqa: BLE001
        body = {}

    # Pick the goal: explicit `goal` field wins; otherwise auto-wrap
    # so Gemma has something concrete to plan from.
    goal = body.get("goal") if isinstance(body, dict) else None
    if not goal:
        summary = json.dumps(
            {k: v for k, v in body.items() if k != "goal"},
            ensure_ascii=False, default=str,
        )[:600]
        goal = (
            f"[Webhook trigger] Process this incoming payload:\n{summary}\n\n"
            "Decide what action is appropriate based on the session's purpose. "
            "If a notification is warranted, use telegram_send / email_send."
        )

    task_id = str(uuid.uuid4())
    session_mode = sess.get("mode", "code")
    try:
        from models import SessionMode
        mode_enum = SessionMode(session_mode)
    except Exception:  # noqa: BLE001
        from models import SessionMode
        mode_enum = SessionMode.CODE

    req = TaskRequest(
        id=task_id,
        goal=goal,
        context={"webhook_payload": body} if body else {},
        priority=5,
        session_id=session_id,
        mode=mode_enum,
    )

    # Fire-and-forget — respond 202 so the caller (cron / stripe / etc.)
    # doesn't sit waiting for the full task to finish. The usual task
    # lifecycle (tracker, WebSocket events, memory) all kicks in
    # automatically since we route through orchestrator.run().
    async def _run_bg() -> None:
        try:
            await orchestrator.run(req)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Webhook-triggered task %s crashed: %s", task_id, exc)

    asyncio.create_task(_run_bg())
    logger.info(
        "Webhook triggered session=%s task=%s (payload keys: %s)",
        session_id, task_id, list(body.keys()),
    )
    return {
        "ok": True,
        "task_id": task_id,
        "session_id": session_id,
        "message": "Task queued; watch progress in the UI or poll /tasks/{task_id}.",
    }


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
) -> dict:
    """Upload a file into the session's workspace under ``uploads/``.

    Used by Analytic mode (CSV/XLSX/JSON ingestion) but general-purpose.
    The file lands where shell tools already run from, so the agent can
    read it by just referring to ``uploads/<name>`` without any extra
    plumbing.

    Filenames are sanitized (basename only, no path traversal) and
    collisions get a timestamp suffix so a second upload of the same
    name doesn't overwrite the first.
    """
    # Resolve which workspace we're uploading into — session override
    # wins if one is configured, otherwise the global default.
    from config import config as _cfg, set_session_workspace
    workspace = _cfg.agent_workspace_dir
    if session_id:
        try:
            sess = await tracker.get_session(session_id)
            if sess and sess.get("workspace_dir"):
                workspace = os.path.abspath(sess["workspace_dir"])
                # Also prime the context var so any parallel /run
                # requests in the same event-loop task see the right ws.
                set_session_workspace(sess["workspace_dir"])
        except Exception:  # noqa: BLE001
            pass

    uploads_dir = os.path.join(workspace, "uploads")
    try:
        os.makedirs(uploads_dir, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Cannot create uploads dir: {exc}")

    # Sanitize: basename only, no path traversal, no null bytes.
    safe_name = os.path.basename(file.filename or "upload.bin")
    safe_name = safe_name.replace("\x00", "").strip() or "upload.bin"

    dest = os.path.join(uploads_dir, safe_name)
    # Collision → append timestamp before the extension.
    if os.path.exists(dest):
        base, ext = os.path.splitext(safe_name)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"{base}_{ts}{ext}"
        dest = os.path.join(uploads_dir, safe_name)

    size_bytes = 0
    try:
        with open(dest, "wb") as f:
            while True:
                chunk = await file.read(1 << 20)  # 1 MiB chunks
                if not chunk:
                    break
                f.write(chunk)
                size_bytes += len(chunk)
    except Exception as exc:  # noqa: BLE001
        # Best-effort cleanup on partial write
        try: os.remove(dest)
        except Exception: pass
        raise HTTPException(500, f"Upload write failed: {exc}")

    # Return the path relative to the workspace — that's what agents
    # see as the working cwd, so the path is portable.
    rel_path = os.path.relpath(dest, workspace)
    logger.info(
        "Uploaded %s (%d bytes, session=%s) → %s",
        safe_name, size_bytes, session_id or "(global)", dest,
    )
    return {
        "ok": True,
        "name": safe_name,
        "rel_path": rel_path,          # e.g. "uploads/data.csv"
        "abs_path": dest,
        "size_bytes": size_bytes,
        "content_type": file.content_type or "application/octet-stream",
    }


@app.post("/run", response_model=RunResponse, status_code=202)
async def run_task(body: RunRequest) -> RunResponse:
    # Use a client-supplied task_id when present so the UI can call
    # POST /tasks/{id}/cancel (Stop button) on the *same* id it sent.
    task_id = body.task_id or str(uuid.uuid4())

    # Resolve the mode. Priority:
    #   1. Explicit override in body.mode (rare — UI usually omits it)
    #   2. The mode stored on the session row (what the user picked in the UI)
    #   3. Default to CODE for back-compat with sessions that pre-date modes
    resolved_mode = body.mode or SessionMode.CODE
    if body.session_id and not body.mode:
        try:
            sess = await tracker.get_session(body.session_id)
            if sess and sess.get("mode"):
                resolved_mode = SessionMode(sess["mode"])
        except Exception:  # noqa: BLE001
            pass

    req = TaskRequest(
        id=task_id,
        goal=body.goal,
        context=body.context,
        priority=body.priority,
        session_id=body.session_id,
        mode=resolved_mode,
    )
    # Auto-create the session row if the client gave us a new id so the
    # sessions list picks it up on refresh.
    if body.session_id:
        try:
            await tracker.create_session(
                body.session_id,
                body.goal[:60] or "New session",
                mode=resolved_mode.value,
            )
        except Exception:  # noqa: BLE001
            pass

    # Run the orchestrator as a tracked asyncio.Task so /tasks/{id}/cancel
    # can call .cancel() on it mid-flight. Without this wrapper we'd have
    # no handle to interrupt the coroutine from another HTTP handler.
    run_task_obj = asyncio.create_task(orchestrator.run(req))
    _running_tasks[task_id] = run_task_obj
    try:
        result = await run_task_obj
    except asyncio.CancelledError:
        logger.info("Task %s was cancelled by user (Stop button)", task_id)
        cancelled_msg = "⏹️ 任務已被使用者中斷"
        # Best-effort: mark the task failed so the history reflects the
        # interrupt (swallow errors — DB state isn't essential here).
        try:
            await tracker.update_task_status(task_id, TaskStatus.FAILED, cancelled_msg)
        except Exception:  # noqa: BLE001
            pass
        return RunResponse(
            task_id=task_id,
            status=TaskStatus.FAILED,
            message=cancelled_msg,
        )
    finally:
        _running_tasks.pop(task_id, None)

    return RunResponse(
        task_id=result.task_id,
        status=result.status,
        message=result.summary,
    )


@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str) -> dict:
    """Cancel an in-flight task (Stop button from the UI)."""
    t = _running_tasks.get(task_id)
    if not t:
        raise HTTPException(
            status_code=404,
            detail="No running task with that id (already finished?)",
        )
    t.cancel()
    return {"cancelled": True, "task_id": task_id}


@app.post("/tasks/{task_id}/claude-fix", status_code=202)
async def claude_fix_task(task_id: str, body: Optional[dict] = None) -> dict:
    """User-triggered "Let Claude fix this" button.

    Loads the failed task + all subtask attempts, hands them to Claude
    with full tool access, and lets Claude drive the remaining work
    end-to-end. Responds immediately (202) and does the heavy lifting
    in the background; progress flows through the usual WebSocket
    tool_call / tool_result events so the UI can show what Claude's
    doing in real time.

    Optional body: {"hint": "...extra user guidance for Claude..."}.
    """
    task = await tracker.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    # Privacy guardrail: refuse the whole flow if the session is
    # local-only. This is the point at which task history (including
    # tool outputs that may contain uploaded file contents) would
    # otherwise be sent to Anthropic's API.
    session_id = task.get("session_id")
    if session_id:
        try:
            sess = await tracker.get_session(session_id)
            if sess and bool(sess.get("local_only")):
                raise HTTPException(
                    status_code=403,
                    detail=(
                        "This session is marked local-only. Claude escalation "
                        "is disabled to keep task data on this machine. "
                        "Turn off local-only in the chat header (🔒 → 🔓) if "
                        "you want to dispatch this task to Claude."
                    ),
                )
        except HTTPException:
            raise
        except Exception:  # noqa: BLE001
            pass

    subtasks_raw = await tracker.get_subtasks(task_id)
    # Tracker returns SubTask objects; normalise to plain dicts for the
    # escalation agent so it doesn't need to know the SubTask shape.
    subtasks = [s.model_dump() if hasattr(s, "model_dump") else dict(s) for s in subtasks_raw]
    user_hint = (body or {}).get("hint") if body else None

    # Apply the session's workspace context var so any shell commands
    # Claude runs land in the same dir as the original task.
    if session_id:
        try:
            sess = await tracker.get_session(session_id)
            if sess and sess.get("workspace_dir"):
                from config import set_session_workspace
                set_session_workspace(sess["workspace_dir"])
        except Exception:  # noqa: BLE001
            pass

    async def _run_claude_fix() -> None:
        try:
            result = await escalation_agent.resolve_whole_task(
                task=task,
                subtasks=subtasks,
                tool_registry=tool_registry,
                session_id=session_id,
                user_hint=user_hint,
                ws_callback=ws_manager.broadcast,
            )
            final_status = TaskStatus.COMPLETED if result.get("success") else TaskStatus.ESCALATED
            summary = result.get("summary") or "(Claude returned no summary)"
            # Prepend a marker so the user can tell which summary came
            # from a Claude fix vs the original local run.
            marked = f"🛠️ **Claude Fix**\n\n{summary}"
            try:
                await tracker.update_task_status(task_id, final_status, marked)
            except Exception:  # noqa: BLE001
                pass
            await ws_manager.broadcast({
                "event":    "claude_fix_done",
                "task_id":  task_id,
                "status":   final_status.value,
                "summary":  marked,
                "turns":    result.get("turns"),
                "tools_used": result.get("tools_used", []),
            })
        except Exception as exc:  # noqa: BLE001
            logger.exception("claude-fix crashed for %s: %s", task_id, exc)
            await ws_manager.broadcast({
                "event":   "claude_fix_done",
                "task_id": task_id,
                "status":  "failed",
                "summary": f"🛠️ Claude Fix crashed: {exc}",
            })

    asyncio.create_task(_run_claude_fix())
    return {"ok": True, "task_id": task_id, "message": "Claude is taking over..."}


@app.get("/tasks/{task_id}", response_model=StatusResponse)
async def get_task_status(task_id: str) -> StatusResponse:
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
async def list_tasks(
    limit: int = 20, session_id: Optional[str] = None,
) -> list:
    return await tracker.list_tasks(limit=limit, session_id=session_id)


# ── Session endpoints ─────────────────────────────────────────────────────────

@app.get("/sessions", response_model=SessionListResponse)
async def list_sessions(limit: int = 50) -> SessionListResponse:
    rows = await tracker.list_sessions(limit=limit)
    for r in rows:
        if "local_only" in r:
            r["local_only"] = bool(r["local_only"])
    return SessionListResponse(sessions=[Session(**r) for r in rows])


@app.post("/sessions", response_model=Session)
async def create_session(body: CreateSessionRequest) -> Session:
    s = Session(
        title=body.title or "New session",
        mode=body.mode or SessionMode.CODE,
        workspace_dir=body.workspace_dir,
        local_only=bool(body.local_only) if body.local_only is not None else False,
    )
    await tracker.create_session(s.id, s.title, mode=s.mode.value)
    if s.workspace_dir:
        await tracker.update_session_workspace(s.id, s.workspace_dir)
    if s.local_only:
        await tracker.update_session_local_only(s.id, True)
    return s


@app.patch("/sessions/{session_id}", response_model=Session)
async def update_session(session_id: str, body: CreateSessionRequest) -> Session:
    # Update whichever fields the client sent. All are optional —
    # if none is provided we just return the current row.
    if body.title is not None:
        await tracker.rename_session(session_id, body.title or "Session")
    if body.mode is not None:
        await tracker.update_session_mode(session_id, body.mode.value)
    if body.workspace_dir is not None:
        # Empty string ⇒ clear the override (fall back to global default).
        ws = body.workspace_dir.strip() or None
        # Resolve relative user input against the global workspace so the
        # UI can stay terse ("my-project" → <global>/my-project), while
        # still accepting absolute paths verbatim (e.g. /home/me/repo).
        if ws and not os.path.isabs(ws):
            ws = os.path.abspath(os.path.join(config.agent_workspace_dir, ws))
        await tracker.update_session_workspace(session_id, ws)
    if body.local_only is not None:
        await tracker.update_session_local_only(session_id, bool(body.local_only))
    row = await tracker.get_session(session_id)
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    # SQLite returns local_only as 0/1 int — coerce to bool so the
    # Pydantic model serialises cleanly.
    if "local_only" in row:
        row["local_only"] = bool(row["local_only"])
    return Session(**row)


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> dict:
    await tracker.delete_session(session_id)
    # Also wipe any long-term memory tagged with this session so the
    # user doesn't see it resurface if they reuse the same id elsewhere.
    try:
        await memory_manager.clear_session(session_id)
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True}


@app.get("/sessions/{session_id}/memory")
async def session_memory_stats(session_id: str) -> dict:
    """Count how many long-term memories are tagged with this session.

    Used by the chat header to show a small badge like "🧠 12" so the user
    can see how much accumulated context Gemma is pulling in — high
    numbers from an old project are the usual cause of cross-project
    contamination complaints.
    """
    count = await memory_manager.count_for_session(session_id)
    return {"session_id": session_id, "count": count}


@app.delete("/sessions/{session_id}/memory")
async def clear_session_memory(session_id: str) -> dict:
    """Clear just this session's long-term memory — the chat history in
    SQLite is preserved. Fixes "new project, same session" contamination
    without forcing the user to spin up a fresh session.
    """
    deleted = await memory_manager.clear_session(session_id)
    return {"ok": True, "deleted_count": deleted, "session_id": session_id}


@app.get("/skills", response_model=SkillListResponse)
async def list_skills() -> SkillListResponse:
    skills = await skill_factory.list_all_skills()
    return SkillListResponse(skills=skills)


@app.post("/skills/generate")
async def generate_skill(name: str, description: str) -> dict:
    try:
        skill = await skill_factory.generate_skill(name, description)
        return {
            "name":    skill.name,
            "status":  skill.status,
            "message": f"Skill '{name}' generated and saved.",
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Tool endpoints ────────────────────────────────────────────────────────────

@app.get("/tools")
async def list_tools() -> dict:
    """List all registered tools with their risk level and parameter schemas."""
    return {"tools": tool_registry.list_tools()}


# ── Approval endpoints ────────────────────────────────────────────────────────

@app.get("/approvals")
async def list_approvals() -> dict:
    """Return all pending tool approvals awaiting human review."""
    pending = await approval_store.get_pending()
    return {"approvals": [a.to_dict() for a in pending]}


@app.post("/approvals/{approval_id}/approve")
async def approve_tool(approval_id: str) -> dict:
    """Approve a pending high-risk tool call. The waiting task will resume."""
    resolved = await approval_store.resolve(approval_id, approved=True)
    if not resolved:
        raise HTTPException(
            status_code=404,
            detail="Approval not found or already resolved.",
        )
    return {"status": "approved", "approval_id": approval_id}


@app.post("/approvals/{approval_id}/reject")
async def reject_tool(approval_id: str) -> dict:
    """Reject a pending high-risk tool call. The agent will receive an error and try an alternative."""
    resolved = await approval_store.resolve(approval_id, approved=False)
    if not resolved:
        raise HTTPException(
            status_code=404,
            detail="Approval not found or already resolved.",
        )
    return {"status": "rejected", "approval_id": approval_id}


# ── Memory endpoints ──────────────────────────────────────────────────────────

@app.get("/memory")
async def list_memory(
    page:      int = Query(default=1,  ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict:
    """Paginated list of long-term memories."""
    return await memory_manager.list_memories(page=page, page_size=page_size)


@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str) -> dict:
    """Delete a single memory entry by ID."""
    success = await memory_manager.delete_memory(memory_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Memory not found or could not be deleted.",
        )
    return {"status": "deleted", "id": memory_id}


@app.delete("/memory")
async def clear_memory() -> dict:
    """Delete ALL memories. This action is irreversible."""
    deleted = await memory_manager.clear_all()
    return {"status": "cleared", "deleted_count": deleted}


# ── Usage endpoints ───────────────────────────────────────────────────────────

@app.get("/usage")
async def get_usage(
    page:          int           = Query(default=1,    ge=1),
    page_size:     int           = Query(default=20,   ge=1, le=100),
    provider:      Optional[str] = Query(default=None),
    date_from:     Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    date_to:       Optional[str] = Query(default=None, description="YYYY-MM-DD"),
) -> dict:
    """Paginated API usage records with optional filters."""
    return await tracker.get_usage_paginated(
        page=page,
        page_size=page_size,
        provider_filter=provider,
        date_from=date_from,
        date_to=date_to,
    )


@app.get("/usage/summary")
async def get_usage_summary() -> dict:
    """Aggregated usage statistics (totals, cost by model, cost by day)."""
    return await tracker.get_usage_summary()


# ── Settings endpoints ────────────────────────────────────────────────────────

@app.get("/settings")
async def get_settings() -> dict:
    """Return all settings values, metadata, and current config snapshot."""
    try:
        values = await settings_store.get_all()
        result = {}
        for key, meta in SETTINGS_META.items():
            result[key] = {
                **meta,
                "value": values.get(key, ""),
            }
        return {"success": True, "data": result, "error": None}
    except Exception as exc:  # noqa: BLE001
        logger.error("GET /settings error: %s", exc)
        return {"success": False, "data": None, "error": str(exc)}


@app.post("/settings")
async def update_settings(body: dict) -> dict:
    """Persist new setting values and hot-reload config — no restart needed."""
    try:
        # Validate keys
        unknown = [k for k in body if k not in SETTINGS_META]
        if unknown:
            return {
                "success": False,
                "data": None,
                "error": f"Unknown setting keys: {unknown}",
            }

        await settings_store.update_many({k: str(v) for k, v in body.items()})
        await config.reload_from_store(settings_store)

        # Return fresh values
        values = await settings_store.get_all()
        result = {}
        for key, meta in SETTINGS_META.items():
            result[key] = {**meta, "value": values.get(key, "")}

        logger.info("Settings updated: %s", list(body.keys()))
        return {"success": True, "data": result, "error": None}
    except Exception as exc:  # noqa: BLE001
        logger.error("POST /settings error: %s", exc)
        return {"success": False, "data": None, "error": str(exc)}


# ── Ollama management endpoints ───────────────────────────────────────────────

def _ollama_url() -> str:
    """Use the currently configured Ollama base URL."""
    return config.gemma_base_url.rstrip("/")


@app.get("/settings/ollama/status")
async def ollama_status() -> dict:
    """Ping Ollama to check whether it is reachable."""
    import httpx as _httpx

    url = _ollama_url()
    try:
        async with _httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{url}/api/tags")
            online = resp.status_code == 200
    except Exception:  # noqa: BLE001
        online = False

    return {"success": True, "data": {"online": online, "url": url}, "error": None}


@app.get("/settings/ollama/models")
async def list_ollama_models() -> dict:
    """Return the list of locally installed Ollama models with size and family info."""
    import httpx as _httpx

    url = _ollama_url()
    try:
        async with _httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{url}/api/tags")
            resp.raise_for_status()
            raw = resp.json()

        models = []
        for m in raw.get("models", []):
            details = m.get("details", {})
            size_bytes = m.get("size", 0)
            size_gb = round(size_bytes / 1_073_741_824, 2) if size_bytes else None
            models.append({
                "name":   m.get("name", ""),
                "size":   size_gb,
                "family": details.get("family", ""),
                "format": details.get("format", ""),
                "modified_at": m.get("modified_at", ""),
            })

        return {"success": True, "data": {"models": models}, "error": None}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ollama model list error: %s", exc)
        return {"success": False, "data": {"models": []}, "error": str(exc)}


@app.post("/settings/ollama/pull")
async def pull_ollama_model(body: dict) -> dict:
    """Trigger an Ollama model pull; streams progress to WebSocket clients."""
    import asyncio as _asyncio
    import httpx as _httpx
    import json as _json

    model_name: str = (body.get("model") or "").strip()
    if not model_name:
        return {"success": False, "data": None, "error": "model name required"}

    url = _ollama_url()

    async def _stream_pull() -> None:
        try:
            async with _httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{url}/api/pull",
                    json={"name": model_name, "stream": True},
                ) as resp:
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            chunk = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue

                        completed = chunk.get("completed", 0) or 0
                        total     = chunk.get("total", 0) or 0
                        pct       = int(completed / total * 100) if total else 0

                        await ws_manager.broadcast({
                            "type":      "pull_progress",
                            "model":     model_name,
                            "status":    chunk.get("status", ""),
                            "completed": completed,
                            "total":     total,
                            "percent":   pct,
                        })

                        if chunk.get("status") == "success":
                            break

            # Signal completion
            await ws_manager.broadcast({
                "type":    "pull_progress",
                "model":   model_name,
                "status":  "success",
                "percent": 100,
            })
        except Exception as exc:  # noqa: BLE001
            logger.error("Ollama pull error for '%s': %s", model_name, exc)
            await ws_manager.broadcast({
                "type":   "pull_progress",
                "model":  model_name,
                "status": "error",
                "error":  str(exc),
            })

    # Fire-and-forget; client tracks progress via WebSocket
    import asyncio as _asyncio
    _asyncio.create_task(_stream_pull())

    return {
        "success": True,
        "data":    {"model": model_name, "message": "Pull started, track via WebSocket"},
        "error":   None,
    }


@app.delete("/settings/ollama/models/{name:path}")
async def delete_ollama_model(name: str) -> dict:
    """Delete a locally installed Ollama model."""
    import httpx as _httpx

    url = _ollama_url()
    try:
        async with _httpx.AsyncClient(timeout=30) as client:
            resp = await client.request(
                "DELETE",
                f"{url}/api/delete",
                json={"name": name},
            )
            if resp.status_code not in (200, 204):
                return {
                    "success": False,
                    "data":    None,
                    "error":   f"Ollama returned {resp.status_code}: {resp.text}",
                }
        return {
            "success": True,
            "data":    {"deleted": name},
            "error":   None,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("Ollama delete error: %s", exc)
        return {"success": False, "data": None, "error": str(exc)}


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
