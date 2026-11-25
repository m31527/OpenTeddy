"""
The /v1 API: what an app with an API key (api_keys.py) can use, shaped
like OpenAI's, so its SDKs work with base_url=<OpenTeddy>/v1 and the key.

  GET  /v1/models                          one model, "openteddy"
  GET  /v1/datasets                        the knowledge-base datasets `datasets` can name
  GET  /v1/voices                          the voices `voice` can name (speech, realtime, LiveKit)
  POST /v1/chat/completions                a turn with OpenTeddy (stream too; `datasets`)
  GET  /v1/tasks/{id}                      a turn that took longer than the wait
  POST /v1/tasks/{id}/approve|reject|cancel
  POST /v1/audio/transcriptions            speech → text (the voice service)
  WS   /v1/audio/stt-stream                speech → text as it is spoken (partial / final)
  POST /v1/audio/speech                    text → speech (wav or raw pcm)
  WS   /v1/realtime                        a spoken conversation (voice_bridge.py's protocol)
  POST /v1/livekit/token                   a LiveKit room with OpenTeddy's agent in it, and the
                                           app's token to join it (contrib/livekit/)

Every route needs a key; errors come back the OpenAI way ({"error": {…}}).
A turn runs exactly like one typed in the web UI (gate, knowledge base,
memory, tools), as the key's own person in the key's own sessions.
docs/api.md is the reference.

OPENTEDDY_API_PORT=8100 starts a second listener that serves /v1 and
nothing else (on OPENTEDDY_API_HOST, default 127.0.0.1): the address to
put behind a tunnel, so the web UI is never reachable from outside.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import re
import shutil
import time
import uuid
import wave
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

import api_keys
import voice_bridge

logger = logging.getLogger(__name__)

MODEL = "openteddy"
WAIT_S = 90.0              # a plain reply waits this long: under a tunnel's 100 s for the first byte
MAX_WAIT_S = 600.0
STREAM_MAX_S = 1800.0      # a streamed reply keeps the line open (keep-alives) this long
KEEPALIVE_S = 15.0
MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_AUDIO_S = 120          # one transcription; split longer recordings
MAX_SPEECH_CHARS = 4096
ASR_RATE = voice_bridge.ASR_RATE
HISTORY_MAX = 12           # earlier messages carried into a new session

router = APIRouter(prefix="/v1")


class _Hooks:
    """main.py's pieces, handed over by install() so this file needs no import of it."""
    start_task: Callable[..., Awaitable[Any]]
    await_task: Callable[..., Awaitable[Any]]
    tracker: Callable[[], Any]                  # → the tracker (made at start-up, so asked for each time)
    approvals: Callable[[str], Awaitable[dict]]
    approve: Callable[[str], Awaitable[dict]]
    reject: Callable[[str], Awaitable[dict]]
    cancel: Callable[[str], Awaitable[dict]]


H = _Hooks()


# ── Errors, the OpenAI way ───────────────────────────────────────────────────

class ApiError(Exception):
    def __init__(self, status: int, message: str, type_: str = "invalid_request_error",
                 code: Optional[str] = None, param: Optional[str] = None) -> None:
        super().__init__(message)
        self.status, self.message, self.type, self.code, self.param = status, message, type_, code, param


async def _on_api_error(_request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse({"error": {"message": exc.message, "type": exc.type, "param": exc.param, "code": exc.code}},
                        status_code=exc.status)


async def require_key(request: Request) -> Dict[str, Any]:
    token = api_keys.token_from(request.headers)
    if not token:
        raise ApiError(401, "Missing API key: send Authorization: Bearer ot-… (Settings → API 金鑰).",
                       code="missing_api_key")
    key = await api_keys.verify(token)
    if key is None:
        raise ApiError(401, "Incorrect API key provided, or it was revoked.", code="invalid_api_key")
    return key


async def _key_or_grant(token: str) -> Optional[Dict[str, Any]]:
    """A key, or a grant (api_keys.issue_grant) acting as one: a LiveKit
    room's agent listening, speaking and answering in its one conversation."""
    if token.startswith(api_keys.GRANT_PREFIX):
        return await api_keys.verify_grant(token)
    return await api_keys.verify(token)


async def require_key_or_grant(request: Request) -> Dict[str, Any]:
    token = api_keys.token_from(request.headers)
    if not token:
        return await require_key(request)               # the missing-key error
    key = await _key_or_grant(token)
    if key is None:
        raise ApiError(401, "Incorrect API key provided, or it was revoked.", code="invalid_api_key")
    return key


# ── Sessions ─────────────────────────────────────────────────────────────────

async def _new_session(key: Dict[str, Any], first: str, sid: str = "",
                       agent: Optional[Dict[str, Any]] = None) -> str:
    """A session for this key, titled so the owner's web UI shows whose it is;
    built from `agent` (its persona, mode and bindings) when one is named."""
    sid = sid or str(uuid.uuid4())
    title = (f"🔑 {key['name']} · " + (f"{agent.get('name', '')} · " if agent else "")
             + (' '.join((first or '').split())[:40] or 'API'))
    if agent:
        await H.tracker().create_session_from_agent(sid, agent, title=title)
    else:
        await H.tracker().create_session(sid, title, mode="code")
    await H.tracker().rename_session(sid, title, by="user")     # session_titler leaves it as it is
    await api_keys.claim_session(key["id"], sid)
    return sid


async def _own_task(key: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    row = await H.tracker().get_task(task_id)
    if not row or not await api_keys.owns_session(key["id"], row.get("session_id") or ""):
        raise ApiError(404, f"No task {task_id} for this key.", code="task_not_found")
    return row


def _status(s: Any) -> str:
    return str(getattr(s, "value", s) or "")


# ── Knowledge-base datasets ──────────────────────────────────────────────────
# A session answers from the datasets it picked (knowledge.py). An app picks
# them with `datasets` — on a chat call, in the realtime hello, or with a
# {"type":"datasets"} message mid-conversation — and they stay the session's
# pick until it names others. Each turn then reads exactly those. `[]` is a
# pick too: no knowledge at all, for the rest of the conversation (a role
# that plays a customer). A session that never picked reads its Agent's
# datasets when the Agent sets some, else the owner's "always" datasets.

async def _agent_datasets(session_id: str, agent: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    """The datasets the conversation's Agent brings (`agent`, or the session's
    own), existing ones only; None when it has no Agent or the Agent sets none."""
    import knowledge
    if agent is None and session_id:
        sess = await H.tracker().get_session(session_id)
        aid = (sess or {}).get("agent_id") or ""
        agent = await H.tracker().get_agent(aid) if aid else None
    ids = knowledge.agent_datasets(agent)
    if ids is None:
        return None
    known = {d["id"] for d in await knowledge.list_datasets()}
    return [i for i in ids if i in known]


async def _pick_datasets(session_id: str, requested: Any,
                         agent: Optional[Dict[str, Any]] = None) -> Optional[List[str]]:
    """Save `requested` (names or ids; None = keep the session's pick) as the
    session's datasets; return the ids each turn passes on ([] = none), or
    None to leave the choice to knowledge.scope_for (no pick and no Agent
    datasets, or the knowledge base off). `agent`: a new session's Agent,
    before the session exists."""
    import knowledge
    if requested is not None:
        if not isinstance(requested, list) or not all(isinstance(x, str) and x.strip() for x in requested):
            raise ApiError(400, "datasets: a list of dataset names or ids (GET /v1/datasets).", param="datasets")
        if requested and not knowledge.enabled():
            raise ApiError(400, "The knowledge base is off on this OpenTeddy (Settings → 知識庫).",
                           code="knowledge_off", param="datasets")
        if not knowledge.enabled():
            await api_keys.set_no_datasets(session_id, True)
            return None
        try:
            ids = await knowledge.set_session_datasets(session_id, requested)
        except knowledge.KnowledgeError as exc:            # an app lists them at /v1/datasets, not /knowledge
            raise ApiError(400, str(exc).replace("GET /knowledge ", "GET /v1/datasets "),
                           code="dataset_not_found", param="datasets")
        await api_keys.set_no_datasets(session_id, not ids)
        return ids
    if not knowledge.enabled():
        return None
    try:
        picked = await knowledge.session_datasets(session_id)
    except RuntimeError:                          # knowledge base not started
        return None
    if picked:
        return picked
    if await api_keys.no_datasets(session_id):
        return []
    return await _agent_datasets(session_id, agent)


async def _dataset_names(ids: Optional[List[str]]) -> List[str]:
    if not ids:
        return []
    import knowledge
    by_id = {d["id"]: d["name"] for d in await knowledge.list_datasets()}
    return [by_id.get(i, i) for i in ids]


# ── The role a conversation plays ────────────────────────────────────────────
# `agent` builds the conversation from an Agent defined in OpenTeddy (its
# persona, kept and edited there); `instructions` is a role the app writes
# itself, kept for the conversation's every turn. Both together: the agent's
# persona first, then the instructions. gate.with_role puts it in the chat
# lane's prompt, so a trainee talking to "a brand's customer" gets the
# customer, not the assistant's 「請問您對我們哪一項服務比較感興趣呢」.

MAX_INSTRUCTIONS = 4000


async def _resolve_agent(ref: Any) -> Dict[str, Any]:
    name = str(ref or "").strip()
    if not name:
        raise ApiError(400, "agent: the name or id of an Agent (OpenTeddy → Agent).", param="agent")
    tracker = H.tracker()
    agent = await tracker.get_agent(name)
    if not agent:
        agent = next((a for a in await tracker.list_agents()
                      if (a.get("name") or "").strip().lower() == name.lower()), None)
    if not agent:
        raise ApiError(400, f"No agent named {name!r} (OpenTeddy → Agent).", code="agent_not_found", param="agent")
    return agent


def _instructions(value: Any) -> Optional[str]:
    """The `instructions` an app sent: None when absent, "" to clear."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ApiError(400, "instructions: a string.", param="instructions")
    if len(value) > MAX_INSTRUCTIONS:
        raise ApiError(400, f"instructions: at most {MAX_INSTRUCTIONS} characters.", param="instructions")
    return value.strip()


# ── Models ───────────────────────────────────────────────────────────────────

@router.get("/models")
async def models(key: Dict[str, Any] = Depends(require_key)) -> dict:
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "created": 0, "owned_by": "openteddy"}]}


@router.get("/datasets")
async def datasets(key: Dict[str, Any] = Depends(require_key)) -> dict:
    """The knowledge-base datasets an app can name in `datasets` (no contents)."""
    import knowledge
    if not knowledge.enabled():
        return {"object": "list", "enabled": False, "data": []}
    rows = await knowledge.list_datasets()
    return {"object": "list", "enabled": True, "data": [
        {"id": d["id"], "name": d["name"], "description": d.get("description") or "",
         "items": d.get("items", 0), "always": bool(d.get("always"))} for d in rows]}


# ── Chat ─────────────────────────────────────────────────────────────────────

def _text_of(content: Any) -> str:
    """A message's text: a string, or the text parts of a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text") or "") for p in content
                         if isinstance(p, dict) and p.get("type") in ("text", "input_text"))
    return ""


def _wait_s(value: Any) -> float:
    try:
        return max(0.0, min(MAX_WAIT_S, float(value)))
    except (TypeError, ValueError):
        return WAIT_S


def _still_working(task_id: str, approvals: List[dict], en: bool) -> str:
    if approvals:
        tools = "、".join(sorted({str(a.get("tool_name") or "?") for a in approvals}))
        if en:
            return (f"This needs approval before it can go on ({tools}). Approve it with "
                    f"POST /v1/tasks/{task_id}/approve, or in OpenTeddy's web UI.")
        return f"這件事要先核准才能繼續（{tools}）。用 POST /v1/tasks/{task_id}/approve 核准，或在 OpenTeddy 網頁上核准。"
    if en:
        return f"Still working on it. Get the result with GET /v1/tasks/{task_id} when it's done."
    return f"還在處理中。完成後用 GET /v1/tasks/{task_id} 取回結果。"


async def _pending(task_id: str) -> List[dict]:
    try:
        return list((await H.approvals(task_id)).get("approvals") or [])
    except Exception:  # noqa: BLE001
        return []


@router.post("/chat/completions")
async def chat_completions(request: Request, key: Dict[str, Any] = Depends(require_key_or_grant)) -> Response:
    try:
        body = await request.json()
    except ValueError:
        raise ApiError(400, "The body must be JSON.")
    if not isinstance(body, dict):
        raise ApiError(400, "The body must be a JSON object.")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ApiError(400, "messages is required.", param="messages")
    last = next((i for i in range(len(messages) - 1, -1, -1)
                 if isinstance(messages[i], dict) and messages[i].get("role") == "user"), None)
    goal = _text_of(messages[last].get("content")).strip() if last is not None else ""
    if not goal:
        raise ApiError(400, "The last user message has no text.", param="messages")
    model = str(body.get("model") or MODEL)

    # OpenTeddy keeps the conversation itself: name a session to go on in
    # it, or leave it out and the earlier messages come along this once.
    session_id = str(body.get("session_id") or request.headers.get("x-openteddy-session") or "").strip()
    if key.get("grant_session") is not None and session_id != key["grant_session"]:
        raise ApiError(403, "This grant is for one conversation: send its session_id.", code="grant_scope",
                       param="session_id")
    context: Dict[str, Any] = {"api": True}
    if body.get("voice"):
        # The answer will be read aloud (a LiveKit agent, an app's own TTS): the
        # chat lane then answers in short spoken sentences, no markdown or links.
        context["voice"] = True
    role = _instructions(body.get("instructions"))
    if session_id:
        if not await api_keys.owns_session(key["id"], session_id):
            raise ApiError(404, f"No session {session_id} for this key.", code="session_not_found",
                           param="session_id")
        if body.get("agent") is not None:
            raise ApiError(400, "agent sets the role of a new conversation: leave session_id out to start one.",
                           param="agent")
        ds = await _pick_datasets(session_id, body.get("datasets"))
    else:
        # The agent and datasets are checked before the session exists, so a
        # wrong name leaves no empty session behind.
        agent = await _resolve_agent(body["agent"]) if body.get("agent") is not None else None
        new_id = str(uuid.uuid4())
        ds = await _pick_datasets(new_id, body.get("datasets"), agent)
        session_id = await _new_session(key, goal, sid=new_id, agent=agent)
        history = [{"role": m.get("role"), "content": _text_of(m.get("content"))}
                   for m in messages[:last] if isinstance(m, dict)]
        history = [h for h in history if h["role"] in ("system", "user", "assistant") and h["content"].strip()]
        if history:
            context["history"] = history[-HISTORY_MAX:]
    if role is not None:
        await api_keys.set_instructions(session_id, role)
    role = await api_keys.get_instructions(session_id)
    if role:
        context["agent_persona"] = role                # with the agent's own persona, if any (orchestrator)

    # A streamed answer listens for the model's words before the task starts,
    # so not one piece is missed (answer_stream.py).
    import answer_stream
    words, new_task = None, str(uuid.uuid4())
    if body.get("stream"):
        words = answer_stream.listen(new_task)
    try:
        task_id, session_id, job, shortcut = await H.start_task(
            goal=goal, context=context, session_id=session_id, auto_lane=True, api_key_id=key["id"],
            task_id=new_task, **({"datasets": ds} if ds is not None else {}))
    except HTTPException as exc:
        if words is not None:
            answer_stream.close(new_task, words)
        raise ApiError(exc.status_code, str(exc.detail))
    en = voice_bridge.english(goal)
    created = int(time.time())
    head = {"X-OpenTeddy-Session": session_id or "", "X-OpenTeddy-Task": task_id}
    ids = {"session_id": session_id, "task_id": task_id,
           **({"datasets": await _dataset_names(ds)} if ds is not None else {})}

    if body.get("stream"):
        return StreamingResponse(
            _stream(task_id, job, shortcut, model, created, ids, en, words),
            media_type="text/event-stream", headers={**head, "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if shortcut is not None:
        text, status = shortcut, "completed"
    else:
        waiter = asyncio.ensure_future(H.await_task(task_id, job))
        done, _ = await asyncio.wait({waiter}, timeout=_wait_s(body.get("wait", WAIT_S)))
        if done:
            st, text = waiter.result()
            status = _status(st)
        else:
            waiter.cancel()                       # only this wait: the task runs on
            pending = await _pending(task_id)
            text, status = _still_working(task_id, pending, en), ("waiting_approval" if pending else "running")
    return JSONResponse({
        "id": f"chatcmpl-{task_id}", "object": "chat.completion", "created": created, "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text or ""}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "status": status, **ids,
    }, headers=head)


def _rest(streamed: str, final: str) -> str:
    """What the final answer adds to the words already streamed: nothing when
    they are the answer; the remainder when it goes on; all of it, after a
    blank line, when the turn went to the agent after a few words."""
    s, f = streamed.strip(), (final or "").strip()
    if not f or f == s:
        return ""
    if not s:
        return f
    if f.startswith(s):
        return f[len(s):]
    return "\n\n" + f


async def _stream(task_id: str, job: Any, shortcut: Optional[str], model: str, created: int,
                  ids: Dict[str, Any], en: bool, words: Optional["asyncio.Queue"] = None):
    """Server-sent chunks: the role first, then the answer as the model
    writes it (answer_stream.py) — one chunk per piece — with keep-alives
    while the agent works on a turn that needs tools, whose result comes
    when it is done."""
    import answer_stream
    base = {"id": f"chatcmpl-{task_id}", "object": "chat.completion.chunk", "created": created, "model": model}

    def sse(obj: dict) -> str:
        return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"

    def piece(text: str) -> str:
        return sse({**base, "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]})

    yield sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                    "finish_reason": None}], **ids})
    waiter = None
    sink = words
    streamed: List[str] = []
    try:
        if shortcut is not None:
            text, status = shortcut, "completed"
        else:
            waiter = asyncio.ensure_future(H.await_task(task_id, job))
            started = last = time.monotonic()
            getter: Optional[asyncio.Future] = None
            while True:
                if words is not None and getter is None:
                    getter = asyncio.ensure_future(words.get())
                wait_on = {waiter} | ({getter} if getter is not None else set())
                done, _ = await asyncio.wait(wait_on, timeout=KEEPALIVE_S, return_when=asyncio.FIRST_COMPLETED)
                if getter is not None and getter in done:
                    got = getter.result()
                    getter = None
                    if got is answer_stream.END:
                        words = None                  # the model's text is complete; the task finishes next
                    else:
                        streamed.append(got)
                        last = time.monotonic()
                        yield piece(got)
                    continue
                if waiter in done:
                    while words is not None and not words.empty():   # pieces that came with the end
                        got = words.get_nowait()
                        if got is not answer_stream.END:
                            streamed.append(got)
                            yield piece(got)
                    st, final = waiter.result()
                    status = _status(st)
                    text = _rest("".join(streamed), final)
                    break
                if time.monotonic() - started > STREAM_MAX_S:
                    pending = await _pending(task_id)
                    text = ("\n\n" if streamed else "") + _still_working(task_id, pending, en)
                    status = "waiting_approval" if pending else "running"
                    break
                if time.monotonic() - last >= KEEPALIVE_S:
                    last = time.monotonic()
                    yield ": keep-alive\n\n"
            if getter is not None:
                getter.cancel()
        if text:
            yield piece(text)
        yield sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "status": status, **ids})
        yield "data: [DONE]\n\n"
    finally:
        if sink is not None:
            answer_stream.close(task_id, sink)
        if waiter is not None and not waiter.done():
            waiter.cancel()                       # the app hung up: the task runs on


# ── Tasks a key started ──────────────────────────────────────────────────────

_TERMINAL = {"completed", "failed", "escalated", "cancelled"}


@router.get("/tasks/{task_id}")
async def task(task_id: str, key: Dict[str, Any] = Depends(require_key)) -> dict:
    row = await _own_task(key, task_id)
    status = str(row.get("status") or "")
    return {"task_id": task_id, "session_id": row.get("session_id"), "status": status,
            "result": (row.get("summary") or "") if status in _TERMINAL else None,
            "approvals": await _pending(task_id),
            "created_at": row.get("created_at"), "updated_at": row.get("updated_at") or row.get("completed_at")}


@router.post("/tasks/{task_id}/approve")
async def approve(task_id: str, key: Dict[str, Any] = Depends(require_key)) -> dict:
    await _own_task(key, task_id)
    return await H.approve(task_id)


@router.post("/tasks/{task_id}/reject")
async def reject(task_id: str, key: Dict[str, Any] = Depends(require_key)) -> dict:
    await _own_task(key, task_id)
    return await H.reject(task_id)


@router.post("/tasks/{task_id}/cancel")
async def cancel(task_id: str, key: Dict[str, Any] = Depends(require_key)) -> dict:
    await _own_task(key, task_id)
    return await H.cancel(task_id)


# ── Audio ────────────────────────────────────────────────────────────────────

def _voice_ready() -> None:
    if not voice_bridge.configured():
        raise ApiError(503, "This OpenTeddy has no voice service set (Settings → 語音服務網址).",
                       "server_error", "voice_not_configured")
    if not voice_bridge.enabled():
        raise ApiError(503, "Voice is turned off on this OpenTeddy (Settings → 語音對話).",
                       "server_error", "voice_off")


def _resample(samples: Any, rate: int) -> bytes:
    """Float or int16 mono samples at `rate` → PCM16 at 16 kHz."""
    import numpy as np
    x = np.asarray(samples, dtype=np.float32)
    if rate != ASR_RATE and len(x):
        n = int(round(len(x) * ASR_RATE / rate))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return np.clip(np.round(x), -32768, 32767).astype("<i2").tobytes()


def _wav_to_pcm(data: bytes) -> bytes:
    import numpy as np
    try:
        with wave.open(io.BytesIO(data)) as w:
            ch, width, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
            frames = w.readframes(w.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ApiError(400, f"Couldn't read that WAV file ({exc}); send 16-bit PCM WAV.", code="unsupported_audio")
    if width == 2:
        x = np.frombuffer(frames, "<i2").astype(np.float32)
    elif width == 1:
        x = (np.frombuffer(frames, np.uint8).astype(np.float32) - 128.0) * 256.0
    elif width == 4:
        x = np.frombuffer(frames, "<i4").astype(np.float32) / 65536.0
    else:
        raise ApiError(400, f"{width * 8}-bit WAV isn't supported; send 16-bit PCM WAV.", code="unsupported_audio")
    if ch > 1:
        x = x[: len(x) // ch * ch].reshape(-1, ch).mean(axis=1)
    return _resample(x, rate)


async def _to_pcm(data: bytes, content_type: str, filename: str) -> bytes:
    """Uploaded audio → PCM16 mono 16 kHz, what the voice service hears."""
    import numpy as np
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return _wav_to_pcm(data)
    ct = (content_type or "").lower()
    if ct.startswith(("audio/l16", "audio/pcm")) or filename.lower().endswith(".pcm"):
        m = re.search(r"rate=(\d+)", ct)
        return _resample(np.frombuffer(data[: len(data) // 2 * 2], "<i2"), int(m.group(1)) if m else ASR_RATE)
    exe = shutil.which("ffmpeg")
    if exe:
        proc = await asyncio.create_subprocess_exec(
            exe, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ac", "1",
            "-ar", str(ASR_RATE), "pipe:1",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await asyncio.wait_for(proc.communicate(data), 120)
        if proc.returncode == 0 and out:
            return out
        raise ApiError(400, f"Couldn't decode that audio: {err.decode(errors='replace')[-300:]}",
                       code="unsupported_audio")
    raise ApiError(400, "Send WAV, or raw 16-bit PCM (Content-Type audio/L16; rate=16000). mp3, m4a, webm "
                        "and ogg need ffmpeg on the machine OpenTeddy runs on.", code="unsupported_audio")


@router.post("/audio/transcriptions")
async def transcriptions(file: UploadFile = File(...), model: str = Form("sensevoice"),
                         language: Optional[str] = Form(None), response_format: str = Form("json"),
                         key: Dict[str, Any] = Depends(require_key_or_grant)) -> Response:
    fmt = (response_format or "json").lower()
    if fmt not in ("json", "text", "verbose_json"):
        raise ApiError(400, "response_format: json, text or verbose_json.", param="response_format")
    _voice_ready()
    data = await file.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise ApiError(413, "The file is over 25 MB.", param="file")
    pcm = await _to_pcm(data, file.content_type or "", file.filename or "")
    seconds = len(pcm) / 2 / ASR_RATE
    if seconds > MAX_AUDIO_S:
        raise ApiError(400, f"That is {seconds:.0f} s of audio; send at most {MAX_AUDIO_S} s at a time.",
                       param="file")
    try:
        heard = await voice_bridge.transcribe(pcm)
    except Exception as exc:  # noqa: BLE001
        raise ApiError(502, f"The voice service didn't answer: {exc}", "server_error", "voice_unreachable")
    text = (heard.get("text") or "").strip()
    if not re.search(r"\w", text):                # no speech, or a noise heard as 「。」
        text = ""
    if fmt == "text":
        return PlainTextResponse(text)
    if fmt == "verbose_json":
        return JSONResponse({"task": "transcribe", "language": heard.get("language") or "",
                             "duration": round(seconds, 2), "text": text})
    return JSONResponse({"text": text})


async def _voice_health() -> Dict[str, Any]:
    _voice_ready()
    try:
        return await voice_bridge.health()
    except Exception as exc:  # noqa: BLE001
        raise ApiError(502, f"The voice service didn't answer: {exc}", "server_error", "voice_unreachable")


async def _named_voice(name: Any) -> str:
    """A voice an app named for a call → that name; 400 for one the voice
    service doesn't have (unlike audio/speech, which takes OpenAI's names)."""
    if not isinstance(name, str) or not name.strip():
        raise ApiError(400, "voice: a name from GET /v1/voices.", param="voice")
    voices = [str(v) for v in ((await _voice_health()).get("voices") or [])]
    if name.strip() not in voices:
        raise ApiError(400, f"No voice named {name.strip()!r}; this OpenTeddy has: {', '.join(voices) or 'none'}.",
                       code="voice_not_found", param="voice")
    return name.strip()


@router.get("/voices")
async def voices(key: Dict[str, Any] = Depends(require_key)) -> dict:
    """The voices an app can name in `voice` (audio/speech, the realtime hello, livekit/token)."""
    names = [str(v) for v in ((await _voice_health()).get("voices") or [])]
    default = voice_bridge.default_voice(names)
    return {"object": "list", "default": default,
            "data": [{"id": v, "object": "voice", "default": v == default} for v in names]}


@router.post("/audio/speech")
async def speech(request: Request, key: Dict[str, Any] = Depends(require_key_or_grant)) -> Response:
    try:
        body = await request.json()
    except ValueError:
        raise ApiError(400, "The body must be JSON.")
    text = voice_bridge.spoken(str((body or {}).get("input") or ""))
    if not text:
        raise ApiError(400, "input is required.", param="input")
    if len(text) > MAX_SPEECH_CHARS:
        raise ApiError(400, f"input is over {MAX_SPEECH_CHARS} characters.", param="input")
    fmt = str(body.get("response_format") or "wav").lower()
    if fmt not in ("wav", "pcm"):
        raise ApiError(400, "response_format: wav or pcm.", param="response_format")
    h = await _voice_health()
    rate = int(h.get("tts_sample_rate") or 24000)
    voice = str(body.get("voice") or "")
    voice = voice if voice in (h.get("voices") or []) else ""     # OpenAI's names ("alloy"…) → the set voice

    async def pcm_chunks():
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as c:
            for piece in voice_bridge.sentences(text):
                async for chunk in voice_bridge.speak(piece, c, voice=voice):
                    yield chunk

    if fmt == "pcm":
        return StreamingResponse(pcm_chunks(), media_type=f"audio/L16; rate={rate}",
                                 headers={"X-Sample-Rate": str(rate)})
    try:
        pcm = b"".join([c async for c in pcm_chunks()])
    except httpx.HTTPError as exc:
        raise ApiError(502, f"The voice service didn't answer: {exc}", "server_error", "voice_unreachable")
    return Response(voice_bridge.wav(pcm, rate), media_type="audio/wav")


# ── Streaming speech to text ─────────────────────────────────────────────────
# SenseVoice transcribes a whole clip, fast (30–100 ms for a sentence), and
# has no streaming mode. So the utterance so far is transcribed again every
# PARTIAL_MS while the person talks (a "partial"), and once more when it
# ends (the "final"): when the app says so ({"type":"flush"} — LiveKit's
# VAD does) or, with endpoint_ms, after that much silence.

STT_CHUNK_MAX = 64000             # one message: at most 2 s of audio
PARTIAL_MS = 300
PREROLL_BYTES = ASR_RATE * 2 // 2  # 0.5 s kept before speech starts, so its start isn't cut
SPEECH_RMS_MIN = 600.0


def _rms(chunk: bytes) -> float:
    import numpy as np
    x = np.frombuffer(chunk[: len(chunk) // 2 * 2], "<i2").astype(np.float32)
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


@router.websocket("/audio/stt-stream")
async def stt_stream(ws: WebSocket) -> None:
    """→ binary PCM16 mono 16 kHz in small chunks (20–40 ms), and optionally
    {"type":"config","endpoint_ms":600,"interim":true}, {"type":"flush"},
    {"type":"reset"}. ← {"type":"speech_started"}, {"type":"partial","text"},
    {"type":"final","text"}. The key as for /v1/realtime (or a room's grant)."""
    token = api_keys.token_from(ws.headers)
    key = await _key_or_grant(token) if token else None
    if key is None:
        await ws.close(code=4401)
        return
    offered = [p.strip() for p in (ws.headers.get("sec-websocket-protocol") or "").split(",") if p.strip()]
    await ws.accept(subprotocol=api_keys.SUBPROTOCOL if api_keys.SUBPROTOCOL in offered else (offered[0] if offered else None))
    try:
        _voice_ready()
    except ApiError as exc:
        await ws.send_json({"type": "error", "code": exc.code, "text": exc.message})
        await ws.close()
        return

    cfg = {"endpoint_ms": 0, "interim": True}
    buf = bytearray()
    speech, silence_ms, floor = False, 0.0, 300.0
    last_partial, last_text = 0.0, ""
    inflight: Optional[asyncio.Task] = None
    client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))
    send_lock = asyncio.Lock()

    async def send(obj: Dict[str, Any]) -> None:
        async with send_lock:
            await ws.send_json(obj)

    async def hear(pcm: bytes) -> str:
        res = await voice_bridge.transcribe(pcm, client)
        text = (res.get("text") or "").strip()
        return text if re.search(r"\w", text) else ""

    async def partial(snapshot: bytes) -> None:
        nonlocal last_text
        try:
            text = await hear(snapshot)
        except Exception:  # noqa: BLE001 — a missed partial is fine; the final counts
            return
        if text and text != last_text and speech:
            last_text = text
            await send({"type": "partial", "text": text})

    async def final(always: bool) -> None:
        nonlocal buf, speech, silence_ms, last_text, inflight
        if inflight is not None and not inflight.done():
            inflight.cancel()
        pcm, had_speech = bytes(buf), speech
        buf, speech, silence_ms, last_text = bytearray(), False, 0.0, ""
        text = ""
        if pcm and (had_speech or always):
            try:
                text = await hear(pcm[-MAX_AUDIO_S * ASR_RATE * 2:])
            except Exception as exc:  # noqa: BLE001
                await send({"type": "error", "code": "voice_unreachable", "text": str(exc)})
                return
        if text or always:
            await send({"type": "final", "text": text})

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                chunk = msg["bytes"][:STT_CHUNK_MAX]
                buf += chunk
                ms = len(chunk) / 2 / ASR_RATE * 1000
                level = _rms(chunk)
                if level > max(SPEECH_RMS_MIN, floor * 3):
                    if not speech:
                        speech = True
                        await send({"type": "speech_started"})
                    silence_ms = 0.0
                else:
                    floor = floor * 0.95 + level * 0.05 if not speech else floor
                    if speech:
                        silence_ms += ms
                        if cfg["endpoint_ms"] and silence_ms >= cfg["endpoint_ms"]:
                            await final(False)
                            continue
                if not speech and len(buf) > PREROLL_BYTES:
                    del buf[: len(buf) - PREROLL_BYTES]       # only the last 0.5 s before speech
                now = time.monotonic()
                if (cfg["interim"] and speech and now - last_partial >= PARTIAL_MS / 1000
                        and (inflight is None or inflight.done())):
                    last_partial = now
                    inflight = asyncio.create_task(partial(bytes(buf[-MAX_AUDIO_S * ASR_RATE * 2:])))
            elif msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                except ValueError:
                    continue
                kind = data.get("type")
                if kind == "flush":
                    await final(True)
                elif kind == "reset":
                    buf, speech, silence_ms, last_text = bytearray(), False, 0.0, ""
                elif kind == "config":
                    if "endpoint_ms" in data:
                        cfg["endpoint_ms"] = max(0, min(5000, int(data.get("endpoint_ms") or 0)))
                    if "interim" in data:
                        cfg["interim"] = bool(data["interim"])
    except Exception as exc:  # noqa: BLE001 — the app went away
        logger.info("stt-stream ended (%s)", type(exc).__name__)
    finally:
        if inflight is not None and not inflight.done():
            inflight.cancel()
        await client.aclose()
        with contextlib.suppress(Exception):
            await ws.close()


# ── Realtime: a spoken conversation ──────────────────────────────────────────

async def _bind_conversation(key: Dict[str, Any], sid: str, opts: Dict[str, Any], first: str) -> Dict[str, Any]:
    """A spoken conversation's session: `sid` (the key's own) or a new one —
    from `agent` when one is named — with its datasets and role
    (`instructions`). What the app is told back: session_id, and datasets /
    agent / instructions when set. Everything is checked before a new
    session is made."""
    if sid and not await api_keys.owns_session(key["id"], sid):
        raise ApiError(404, f"No session {sid} for this key.", code="session_not_found", param="session_id")
    if sid and opts.get("agent") is not None:
        raise ApiError(400, "agent sets the role of a new conversation: leave session_id out to start one.",
                       param="agent")
    new_id = sid or str(uuid.uuid4())
    agent = await _resolve_agent(opts["agent"]) if opts.get("agent") is not None and not sid else None
    role = _instructions(opts.get("instructions"))
    ds = await _pick_datasets(new_id, opts.get("datasets"), agent)
    if not sid:
        await _new_session(key, first, sid=new_id, agent=agent)
    if role is not None:
        await api_keys.set_instructions(new_id, role)
    return {"session_id": new_id, **({"datasets": await _dataset_names(ds)} if ds is not None else {}),
            **({"agent": agent.get("name", "")} if agent else {}),
            **({"instructions": True} if await api_keys.get_instructions(new_id) else {})}


@router.websocket("/realtime")
async def realtime(ws: WebSocket) -> None:
    """voice_bridge.py's conversation for an app. The key comes as
    `Authorization: Bearer …`, or (a browser can't set headers) as the
    subprotocol `openteddy-key.<key>` next to `openteddy`."""
    token = api_keys.token_from(ws.headers)
    key = await api_keys.verify(token) if token else None
    if key is None:
        await ws.close(code=4401)                 # → HTTP 403 at the handshake
        return
    offered = [p.strip() for p in (ws.headers.get("sec-websocket-protocol") or "").split(",") if p.strip()]
    chosen = api_keys.SUBPROTOCOL if api_keys.SUBPROTOCOL in offered else (offered[0] if offered else None)
    await ws.accept(subprotocol=chosen)

    def refuse(exc: ApiError) -> ValueError:
        err = ValueError(exc.message)
        err.code = exc.code or "bad_request"      # type: ignore[attr-defined]
        return err

    async def bind(sid: str, hello: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return await _bind_conversation(key, sid, hello, "語音對話")
        except ApiError as exc:
            if exc.code == "session_not_found":
                raise ValueError(exc.message)
            raise refuse(exc)

    async def control(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        kind = data.get("type")
        try:
            if kind == "datasets":
                ds = await _pick_datasets(conv.session_id, data.get("datasets") if "datasets" in data else [])
                return {"type": "datasets", "datasets": await _dataset_names(ds)}
            if kind == "instructions":                    # a new role from the next turn on ("" clears it)
                await api_keys.set_instructions(conv.session_id, _instructions(data.get("instructions") or ""))
                return {"type": "instructions", "set": bool(await api_keys.get_instructions(conv.session_id))}
        except ApiError as exc:
            raise refuse(exc)
        return None

    async def start(**kw: Any) -> Any:
        sid = kw.get("session_id") or ""
        ds = await _pick_datasets(sid, None) if sid else None
        role = await api_keys.get_instructions(sid)
        if role:
            kw["context"] = {**(kw.get("context") or {}), "agent_persona": role}
        return await H.start_task(**kw, api_key_id=key["id"], **({"datasets": ds} if ds is not None else {}))

    conv = voice_bridge.Conversation(ws, start_task=start, await_task=H.await_task,
                                     bind_session=bind, on_control=control)
    await conv.run()
    with contextlib.suppress(Exception):
        await ws.close()


# ── LiveKit: a room with OpenTeddy's agent in it ─────────────────────────────
#
# An app with its key asks for a call; OpenTeddy makes the conversation (as
# for /v1/realtime: agent, instructions, datasets), opens a LiveKit room,
# sends its agent (contrib/livekit/agent.py) into it with a grant for that
# one conversation, and hands the app a token to join. The app needs no
# LiveKit secret and no server of its own for it.
#
#   LIVEKIT_URL         what the app connects to (wss://lk.example.com)
#   LIVEKIT_API_URL     where OpenTeddy reaches LiveKit's server API
#                       (default: LIVEKIT_URL as http)
#   LIVEKIT_API_KEY, LIVEKIT_API_SECRET   the LiveKit server's key pair
#   LIVEKIT_AGENT_NAME  the agent's name (its AGENT_NAME), default "openteddy"

LK_TOKEN_TTL_S = 900          # time to join; LiveKit refreshes it while the app stays connected
GRANT_TTL_S = 6 * 3600        # the longest call
LK_EMPTY_TIMEOUT_S = 120      # a room nobody joins closes after this


def _livekit_config() -> Dict[str, str]:
    url = os.environ.get("LIVEKIT_URL", "").strip()
    key, secret = os.environ.get("LIVEKIT_API_KEY", "").strip(), os.environ.get("LIVEKIT_API_SECRET", "").strip()
    if not (url and key and secret):
        raise ApiError(503, "LiveKit isn't set up on this OpenTeddy (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET).",
                       type_="api_error", code="not_configured")
    api = os.environ.get("LIVEKIT_API_URL", "").strip() or re.sub(r"^ws", "http", url)
    return {"url": url, "api": api.rstrip("/"), "key": key, "secret": secret,
            "agent": os.environ.get("LIVEKIT_AGENT_NAME", "").strip() or "openteddy"}


def _jwt(secret: str, claims: Dict[str, Any]) -> str:
    """An HS256 JWT, the kind LiveKit takes for joining and for its server API."""
    import base64, hashlib, hmac  # noqa: E401
    b64 = lambda raw: base64.urlsafe_b64encode(raw).rstrip(b"=").decode()  # noqa: E731
    head = b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = b64(json.dumps(claims, separators=(",", ":"), ensure_ascii=False).encode())
    sig = b64(hmac.new(secret.encode(), f"{head}.{body}".encode(), hashlib.sha256).digest())
    return f"{head}.{body}.{sig}"


async def _livekit(cfg: Dict[str, str], method: str, body: Dict[str, Any], video: Dict[str, Any]) -> Dict[str, Any]:
    """One call to LiveKit's server API (Twirp, JSON), e.g. "RoomService/CreateRoom"."""
    now = int(time.time())
    token = _jwt(cfg["secret"], {"iss": cfg["key"], "nbf": now - 5, "exp": now + 60, "video": video})
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=3.0)) as c:
            r = await c.post(f"{cfg['api']}/twirp/livekit.{method}", json=body,
                             headers={"Authorization": f"Bearer {token}"})
    except httpx.HTTPError as exc:
        raise ApiError(502, f"LiveKit is unreachable ({type(exc).__name__}).", type_="api_error",
                       code="livekit_unreachable")
    if r.status_code >= 400:
        raise ApiError(502, f"LiveKit refused {method}: {r.text[:200]}", type_="api_error", code="livekit_error")
    return r.json() if r.content else {}


@router.post("/livekit/token")
async def livekit_token(request: Request, key: Dict[str, Any] = Depends(require_key)) -> dict:
    """{session_id?, agent?, instructions?, datasets?, voice?, identity?, name?} →
    {url, token, room, identity, expires_in, session_id, datasets?, agent?, instructions?, voice?}"""
    cfg = _livekit_config()
    try:
        body = await request.json() if (await request.body()).strip() else {}
    except ValueError:
        raise ApiError(400, "The body must be JSON.")
    if not isinstance(body, dict):
        raise ApiError(400, "The body must be a JSON object.")
    identity = str(body.get("identity") or f"app-{uuid.uuid4().hex[:8]}")
    if not re.fullmatch(r"[\w.@:-]{1,64}", identity):
        raise ApiError(400, "identity: up to 64 letters, digits, and . _ - @ :", param="identity")
    name = " ".join(str(body.get("name") or "").split())[:64]
    voice = await _named_voice(body["voice"]) if body.get("voice") is not None else ""

    conv = await _bind_conversation(key, str(body.get("session_id") or "").strip(), body, "語音通話")
    room = f"ot-{uuid.uuid4().hex[:12]}"
    await _livekit(cfg, "RoomService/CreateRoom",
                   {"name": room, "empty_timeout": LK_EMPTY_TIMEOUT_S, "departure_timeout": 20, "max_participants": 4},
                   {"roomCreate": True})
    grant = await api_keys.issue_grant(key["id"], conv["session_id"], GRANT_TTL_S)
    # The grant rides in the dispatch, which only the agent sees — not in the
    # room's metadata, which every participant can read.
    await _livekit(cfg, "AgentDispatchService/CreateDispatch",
                   {"agent_name": cfg["agent"], "room": room,
                    "metadata": json.dumps({"session_id": conv["session_id"], "grant": grant,
                                            **({"voice": voice} if voice else {})})},
                   {"roomAdmin": True, "room": room})
    now = int(time.time())
    token = _jwt(cfg["secret"], {"iss": cfg["key"], "sub": identity, "nbf": now - 5, "exp": now + LK_TOKEN_TTL_S,
                                 **({"name": name} if name else {}),
                                 "video": {"room": room, "roomJoin": True, "canPublish": True,
                                           "canSubscribe": True, "canPublishData": True}})
    logger.info("livekit: room %s for key %s, conversation %s", room, key["id"], conv["session_id"])
    return {"url": cfg["url"], "token": token, "room": room, "identity": identity,
            "expires_in": LK_TOKEN_TTL_S, **conv, **({"voice": voice} if voice else {})}


# ── Wiring ───────────────────────────────────────────────────────────────────

def install(app: Any, **hooks: Any) -> None:
    """Add /v1 to `app`; `hooks` are main.py's start_task, await_task,
    tracker, approvals, approve, reject, cancel."""
    for name, fn in hooks.items():
        setattr(H, name, fn)
    app.include_router(router)
    app.add_exception_handler(ApiError, _on_api_error)


class OnlyV1:
    """The app as the API port serves it: /v1/… and nothing else."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") in ("http", "websocket") and not str(scope.get("path") or "").startswith("/v1/"):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 4404})
                return
            body = json.dumps({"error": {"message": "Not found: this port serves /v1 only.",
                                         "type": "invalid_request_error", "param": None, "code": "not_found"}})
            await send({"type": "http.response.start", "status": 404,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": body.encode()})
            return
        await self.app(scope, receive, send)


_server: Any = None
_task: Optional[asyncio.Task] = None


def port() -> int:
    try:
        return int(os.environ.get("OPENTEDDY_API_PORT", "") or 0)
    except ValueError:
        return 0


async def start(app: Any) -> bool:
    """Serve /v1 alone on OPENTEDDY_API_PORT, when it is set."""
    global _server, _task
    p = port()
    if not p or _task is not None:
        return False
    import uvicorn
    host = os.environ.get("OPENTEDDY_API_HOST", "127.0.0.1")
    cfg = uvicorn.Config(OnlyV1(app), host=host, port=p, log_level="warning", lifespan="off", ws="auto")
    _server = uvicorn.Server(cfg)
    _server.capture_signals = contextlib.nullcontext          # type: ignore[method-assign]
    _task = asyncio.create_task(_server.serve())
    logger.info("api: /v1 also on http://%s:%d (that port serves nothing else)", host, p)
    return True


async def stop() -> None:
    global _server, _task
    if _server is not None:
        _server.should_exit = True
    if _task is not None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(_task, 5)
    _server, _task = None, None
