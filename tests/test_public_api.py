"""
API keys and the /v1 API (api_keys.py, public_api.py).

  A. Keys: made once, only a hash kept, verified, counted, revoked; the
     three ways a key travels (Bearer, X-API-Key, WebSocket subprotocol).
  B. Every /v1 route needs a key; errors come back the OpenAI way.
  C. Chat: a new session per key (titled 🔑, owned), earlier messages carried
     in, the key's person on the task; another key's session refused; long
     work answered "running" while it goes on; approvals; streaming; the
     earlier messages read as 【最近的對話】 (orchestrator._history_block).
  D. Tasks: only the key's own; approve goes through.
  E. The API-only port serves /v1 and nothing else.
  F. Audio: WAV / raw PCM to 16 kHz, other formats refused without ffmpeg,
     transcription and speech through the voice service, voice off → 503.
  G. Realtime: key by header or subprotocol, a session made and named in
     "ready", turns run as the key, another key's session refused.
  H. privacy.resolve: the key's person only through the api_key door.
  L. LiveKit: /v1/livekit/token makes the key's conversation and a room, sends
     the agent with a grant, signs the app's join token; a grant acts as its
     key in that one conversation only, and dies with expiry or the key.

    .venv/bin/python tests/test_public_api.py
"""
from __future__ import annotations

import asyncio, io, json, logging, math, os, sqlite3, struct, sys, tempfile, types, wave
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

from config import config

TMP = tempfile.mkdtemp()
config.db_path = os.path.join(TMP, "t.db")

import api_keys
import public_api
import voice_bridge
from fastapi import FastAPI
from fastapi.testclient import TestClient


def run(coro):
    return asyncio.run(coro)


# ── fakes for main.py's pieces ───────────────────────────────────────────────

class FakeTracker:
    def __init__(self):
        self.sessions, self.tasks = {}, {}
        self.agents = {"ag1": {"id": "ag1", "name": "品牌客戶・王經理", "mode": "chat",
                               "system_prompt": "你是品牌客戶王經理，在意價格與交期。"},
                       "ag2": {"id": "ag2", "name": "客服・小美", "mode": "chat",
                               "system_prompt": "你是品牌客服小美。", "knowledge_datasets": '["ds1"]'}}

    async def create_session(self, sid, title, mode="code"):
        self.sessions[sid] = {"title": title, "mode": mode, "state": "pending"}

    async def get_agent(self, aid):
        return self.agents.get(aid)

    async def get_session(self, sid):
        return self.sessions.get(sid)

    async def list_agents(self):
        return list(self.agents.values())

    async def create_session_from_agent(self, sid, agent, title=""):
        self.sessions[sid] = {"title": title, "mode": agent["mode"], "state": "pending", "agent_id": agent["id"]}

    async def rename_session(self, sid, title, by="user"):
        self.sessions[sid].update(title=title, state=by)

    async def get_task(self, task_id):
        return self.tasks.get(task_id)


TR = FakeTracker()
CALLS = []
PLAN = {"delay": 0.0, "message": "答案", "shortcut": None, "approvals": [], "words": None}
APPROVED = []


async def fake_start_task(**kw):
    CALLS.append(kw)
    tid = kw.get("task_id") or f"t{len(CALLS)}"
    TR.tasks[tid] = {"id": tid, "session_id": kw["session_id"], "status": "running", "summary": ""}
    if PLAN["shortcut"]:
        return tid, kw["session_id"], None, PLAN["shortcut"]
    words = PLAN["words"]

    async def job():
        import answer_stream
        for w in words or []:                     # the gate streaming its answer (answer_stream.py)
            await asyncio.sleep(0.05)
            answer_stream.push(tid, w)
        if words is not None:
            answer_stream.end(tid)
        await asyncio.sleep(PLAN["delay"])
        TR.tasks[tid].update(status="completed", summary=PLAN["message"])
        return types.SimpleNamespace(status="completed", summary=PLAN["message"])
    return tid, kw["session_id"], asyncio.ensure_future(job()), None


async def fake_await_task(task_id, job):
    r = await asyncio.shield(job)
    return r.status, r.summary


async def fake_approvals(task_id):
    return {"approvals": [{**a, "task_id": task_id} for a in PLAN["approvals"] if a["task_id"] in (task_id, "*")]}


async def fake_approve(task_id):
    APPROVED.append(task_id)
    return {"approved": 1}


app = FastAPI()
public_api.install(app, start_task=fake_start_task, await_task=fake_await_task, tracker=lambda: TR,
                   approvals=fake_approvals, approve=fake_approve, reject=fake_approve, cancel=fake_approve)


# ── A. keys ──────────────────────────────────────────────────────────────────

def test_keys():
    row, token = run(api_keys.create("  客服   App  "))
    assert token.startswith("ot-") and len(token) > 40 and row["name"] == "客服 App"
    assert row["hint"].startswith(token[:6]) and row["hint"].endswith(token[-4:])
    with sqlite3.connect(config.db_path) as db:
        dump = "\n".join(str(r) for r in db.execute("SELECT * FROM api_keys"))
    assert token not in dump and api_keys._hash(token) in dump
    print("  ✓ a key is shown once: only its hash and a hint are stored")

    assert run(api_keys.verify(token))["id"] == row["id"]
    assert run(api_keys.verify(token + "x")) is None and run(api_keys.verify("sk-abc")) is None
    assert run(api_keys.list_keys())[0]["uses"] == 1
    try:
        run(api_keys.create("   ")); raise AssertionError
    except ValueError:
        pass
    _, other = run(api_keys.create("tmp"))
    tmp_id = run(api_keys.verify(other))["id"]
    assert run(api_keys.revoke(tmp_id)) and not run(api_keys.revoke(tmp_id))
    assert run(api_keys.verify(other)) is None
    print("  ✓ verified and counted; wrong / unnamed / revoked keys refused")

    assert api_keys.token_from({"authorization": "Bearer ot-1"}) == "ot-1"
    assert api_keys.token_from({"x-api-key": "ot-2"}) == "ot-2"
    assert api_keys.token_from({"sec-websocket-protocol": "openteddy, openteddy-key.ot-3"}) == "ot-3"
    assert api_keys.token_from({}) == ""
    print("  ✓ the key travels as Bearer, X-API-Key, or a WebSocket subprotocol")
    return row, token


# ── B–D. HTTP ────────────────────────────────────────────────────────────────

def test_http(c, token, token2):
    H = {"Authorization": f"Bearer {token}"}
    r = c.get("/v1/models")
    assert r.status_code == 401 and r.json()["error"]["code"] == "missing_api_key", r.text
    r = c.get("/v1/models", headers={"Authorization": "Bearer ot-nope"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
    assert set(r.json()["error"]) == {"message", "type", "param", "code"}
    assert c.get("/v1/models", headers=H).json()["data"][0]["id"] == "openteddy"
    print("  ✓ no key / a wrong key → 401 in OpenAI's error shape; /v1/models with one")

    msgs = [{"role": "system", "content": "你是客服"}, {"role": "user", "content": "我叫小明"},
            {"role": "assistant", "content": "你好小明"},
            {"role": "user", "content": [{"type": "text", "text": "我叫什麼？"}, {"type": "image_url"}]}]
    r = c.post("/v1/chat/completions", headers=H, json={"model": "openteddy", "messages": msgs})
    d = r.json()
    assert r.status_code == 200 and d["object"] == "chat.completion", d
    assert d["choices"][0]["message"] == {"role": "assistant", "content": "答案"} and d["status"] == "completed"
    call = CALLS[-1]
    assert call["goal"] == "我叫什麼？" and call["auto_lane"] and call["api_key_id"] == KEY1["id"]
    assert [h["role"] for h in call["context"]["history"]] == ["system", "user", "assistant"]
    sid = d["session_id"]
    assert r.headers["X-OpenTeddy-Session"] == sid and call["session_id"] == sid
    assert TR.sessions[sid]["title"].startswith("🔑 客服 App · ") and TR.sessions[sid]["state"] == "user"
    assert run(api_keys.owns_session(KEY1["id"], sid))
    print("  ✓ a turn: new session for the key (🔑 title, kept), earlier messages carried, key on the task")

    from orchestrator import Orchestrator
    block = Orchestrator._history_block(call["context"]["history"], 2400)
    assert block.startswith("【最近的對話") and "（App 的說明）你是客服" in block
    assert "使用者：我叫小明\n助理：你好小明" in block
    print("  ✓ earlier messages read as the 【最近的對話】 block")

    n = len(TR.sessions)
    r = c.post("/v1/chat/completions", headers=H,
               json={"messages": [{"role": "user", "content": "再一次"}], "session_id": sid})
    assert r.json()["session_id"] == sid and len(TR.sessions) == n and "history" not in CALLS[-1]["context"]
    r = c.post("/v1/chat/completions", headers={"Authorization": f"Bearer {token2}"},
               json={"messages": [{"role": "user", "content": "偷看"}], "session_id": sid})
    assert r.status_code == 404 and r.json()["error"]["code"] == "session_not_found"
    r = c.post("/v1/chat/completions", headers={**H, "X-OpenTeddy-Session": sid},
               json={"messages": [{"role": "user", "content": "header 也行"}]})
    assert r.json()["session_id"] == sid
    assert c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "assistant", "content": "x"}]}).status_code == 400
    print("  ✓ going on in the key's session; another key's session → 404; no user text → 400")
    c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "說話"}], "voice": True})
    assert CALLS[-1]["context"].get("voice") is True
    c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "打字"}]})
    assert "voice" not in CALLS[-1]["context"]
    print("  ✓ \"voice\": true → the chat lane answers for speaking (context voice)")

    PLAN.update(delay=1.5, message="慢慢做完了")
    r = c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "做件大事"}], "wait": 0.2})
    d = r.json()
    assert d["status"] == "running" and f"/v1/tasks/{d['task_id']}" in d["choices"][0]["message"]["content"], d
    slow = d["task_id"]
    PLAN["approvals"] = [{"task_id": "t-next", "tool_name": "shell"}]
    import time as _t
    _t.sleep(1.6)
    t = c.get(f"/v1/tasks/{slow}", headers=H).json()
    assert t["status"] == "completed" and t["result"] == "慢慢做完了", t
    print("  ✓ long work → 'running' + where to fetch it; it ran on and GET /v1/tasks has the result")

    PLAN.update(delay=1.0)
    PLAN["approvals"] = [{"task_id": "*", "tool_name": "shell"}]           # pending on whichever task asks
    d = c.post("/v1/chat/completions", headers=H,
               json={"messages": [{"role": "user", "content": "rm stuff"}], "wait": 0.1}).json()
    assert d["status"] == "waiting_approval" and "shell" in d["choices"][0]["message"]["content"]
    assert "approve" in d["choices"][0]["message"]["content"]
    assert c.get(f"/v1/tasks/{d['task_id']}", headers=H).json()["approvals"][0]["tool_name"] == "shell"
    assert c.post(f"/v1/tasks/{d['task_id']}/approve", headers=H).status_code == 200 and APPROVED[-1] == d["task_id"]
    other = c.get(f"/v1/tasks/{d['task_id']}", headers={"Authorization": f"Bearer {token2}"})
    assert other.status_code == 404 and other.json()["error"]["code"] == "task_not_found"
    assert c.post(f"/v1/tasks/{d['task_id']}/approve", headers={"Authorization": f"Bearer {token2}"}).status_code == 404
    PLAN.update(delay=0.0, approvals=[])
    _t.sleep(1.1)
    print("  ✓ waiting on an approval says so; the key approves its own task, never another's")

    PLAN["message"] = "串流的答案"
    with c.stream("POST", "/v1/chat/completions", headers=H,
                  json={"stream": True, "messages": [{"role": "user", "content": "串流"}]}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [ln for ln in r.iter_lines() if ln.startswith("data: ")]
    chunks = [json.loads(ln[6:]) for ln in lines[:-1]]
    assert lines[-1] == "data: [DONE]" and chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(ch["choices"][0]["delta"].get("content") or "" for ch in chunks) == "串流的答案"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop" and chunks[-1]["object"] == "chat.completion.chunk"
    def stream_pieces(text):
        with c.stream("POST", "/v1/chat/completions", headers=H,
                      json={"stream": True, "messages": [{"role": "user", "content": text}]}) as r:
            return [json.loads(ln[6:])["choices"][0]["delta"].get("content") for ln in r.iter_lines()
                    if ln.startswith("data: ") and ln != "data: [DONE]"]
    PLAN.update(words=["退貨", "期限是", "七天。"], message="退貨期限是七天。")
    got = stream_pieces("退貨期限")
    assert got[1:4] == ["退貨", "期限是", "七天。"] and "".join(x or "" for x in got) == "退貨期限是七天。", got
    PLAN.update(words=["我幫你查。"], message="台北今天晴。")
    got = stream_pieces("台北天氣")
    assert "".join(x or "" for x in got) == "我幫你查。\n\n台北今天晴。", got
    PLAN.update(words=[], message="工作完成。")
    assert "".join(x or "" for x in stream_pieces("做件事")) == "工作完成。"
    import answer_stream as _as
    assert not _as._queues, "every listener closed"
    PLAN.update(words=None, message="答")
    print("  ✓ streaming relays each piece as its own chunk; a hand-off's result follows the words; listeners closed")
    PLAN["shortcut"] = "✓ 已排好"
    d = c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "每天九點提醒我"}]}).json()
    assert d["choices"][0]["message"]["content"] == "✓ 已排好" and d["status"] == "completed"
    PLAN["shortcut"] = None
    print("  ✓ streaming: role, answer, stop, [DONE]; a scheduling shortcut answers at once")


# ── E. the API-only port ─────────────────────────────────────────────────────

def test_only_v1(token):
    from fastapi import FastAPI as F2
    whole = F2()

    @whole.get("/")
    async def ui():
        return {"ui": True}

    @whole.get("/settings")
    async def settings():
        return {"secret": True}
    public_api.install(whole, start_task=fake_start_task, await_task=fake_await_task, tracker=lambda: TR,
                       approvals=fake_approvals, approve=fake_approve, reject=fake_approve, cancel=fake_approve)
    with TestClient(public_api.OnlyV1(whole)) as c:
        for path in ("/", "/settings", "/api-keys", "/v1", "/static/index.html"):
            r = c.get(path)
            assert r.status_code == 404 and r.json()["error"]["code"] == "not_found", (path, r.text)
        assert c.get("/v1/models", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    print("  ✓ the API port answers /v1 and 404s the web UI, Settings, everything else")


# ── F. audio ─────────────────────────────────────────────────────────────────

def _wav_bytes(rate, channels, seconds=0.5, width=2):
    n = int(rate * seconds)
    frames = b"".join(struct.pack("<" + "h" * channels, *([int(8000 * math.sin(i / 10))] * channels)) for i in range(n))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels); w.setsampwidth(width); w.setframerate(rate); w.writeframes(frames)
    return buf.getvalue()


def test_audio(c, token):
    H = {"Authorization": f"Bearer {token}"}
    pcm = public_api._wav_to_pcm(_wav_bytes(48000, 2))
    assert abs(len(pcm) / 2 - 8000) <= 2, len(pcm)
    raw = struct.pack("<" + "h" * 8000, *([1000] * 8000))
    assert len(run(public_api._to_pcm(raw, "audio/L16; rate=8000", "x"))) == 32000
    assert run(public_api._to_pcm(raw, "audio/L16; rate=16000", "x")) == raw
    real_which = public_api.shutil.which
    public_api.shutil.which = lambda name: None
    try:
        run(public_api._to_pcm(b"ID3\x04fake-mp3", "audio/mpeg", "a.mp3")); raise AssertionError
    except public_api.ApiError as exc:
        assert exc.code == "unsupported_audio" and "ffmpeg" in exc.message
    public_api.shutil.which = real_which
    print("  ✓ WAV (48 kHz stereo) and raw PCM → 16 kHz mono; mp3 without ffmpeg → a clear 400")

    vb = voice_bridge
    saved = (vb.configured, vb.enabled, vb.transcribe, vb.health, vb.speak)
    vb.configured, vb.enabled = (lambda: True), (lambda: False)
    r = c.post("/v1/audio/transcriptions", headers=H, files={"file": ("q.wav", _wav_bytes(16000, 1), "audio/wav")})
    assert r.status_code == 503 and r.json()["error"]["code"] == "voice_off"
    vb.enabled = lambda: True
    heard = []

    async def transcribe(p, client=None):
        heard.append(len(p))
        return {"text": "今天幾號" if len(heard) == 1 else "。", "language": "zh"}
    vb.transcribe = transcribe
    r = c.post("/v1/audio/transcriptions", headers=H, files={"file": ("q.wav", _wav_bytes(16000, 1), "audio/wav")})
    assert r.json() == {"text": "今天幾號"} and heard[0] == 16000, (r.text, heard)
    r = c.post("/v1/audio/transcriptions", headers=H, data={"response_format": "verbose_json"},
               files={"file": ("q.wav", _wav_bytes(16000, 1), "audio/wav")})
    assert r.json()["text"] == "" and r.json()["duration"] == 0.5
    assert c.post("/v1/audio/transcriptions", files={"file": ("q.wav", b"x", "audio/wav")}).status_code == 401
    print("  ✓ transcription through the voice service; noise → empty text; voice off → 503; key needed")

    spoke = []

    async def health(timeout=5.0):
        return {"ok": True, "tts_sample_rate": 24000, "voices": ["小美"]}

    async def speak(text, client, voice=""):
        spoke.append((text, voice))
        yield b"\x01\x00" * 100
    vb.health, vb.speak = health, speak
    r = c.post("/v1/audio/speech", headers=H, json={"input": "你好！**粗體** https://x.y 😀", "voice": "alloy"})
    assert r.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(r.content)) as w:
        assert w.getframerate() == 24000 and w.getnchannels() == 1 and w.getnframes() == 100 * len(spoke)
    assert spoke[0][1] == "" and "**" not in spoke[0][0] and "http" not in spoke[0][0] and "😀" not in spoke[0][0]
    r = c.post("/v1/audio/speech", headers=H, json={"input": "嗨", "voice": "小美", "response_format": "pcm"})
    assert r.headers["x-sample-rate"] == "24000" and len(r.content) == 200 and spoke[-1][1] == "小美"
    assert c.post("/v1/audio/speech", headers=H, json={"input": "x", "response_format": "mp3"}).status_code == 400
    assert c.post("/v1/audio/speech", headers=H, json={"input": ""}).status_code == 400
    print("  ✓ speech: WAV at the service's rate, markdown/links/emoji dropped, OpenAI voice names → the set voice, pcm too")

    saved_name = getattr(vb.config, "voice_name", "")
    try:
        vb.config.voice_name = ""
        r = c.get("/v1/voices", headers=H).json()
        assert r["default"] == "小美" and r["data"] == [{"id": "小美", "object": "voice", "default": True}], r
        vb.config.voice_name = "已刪除的聲音"                       # set, then gone from the service
        assert c.get("/v1/voices", headers=H).json()["default"] == "小美"
        assert c.get("/v1/voices").status_code == 401
        vb.enabled = lambda: False
        assert c.get("/v1/voices", headers=H).json()["error"]["code"] == "voice_off"
    finally:
        vb.config.voice_name = saved_name
        vb.configured, vb.enabled, vb.transcribe, vb.health, vb.speak = saved
    print("  ✓ GET /v1/voices: the service's voices and which is used when none is named")


# ── G. realtime ──────────────────────────────────────────────────────────────

def test_realtime(c, token, token2):
    from starlette.websockets import WebSocketDisconnect
    vb = voice_bridge
    saved = (vb.configured, vb.enabled, vb.health, vb.speak)
    vb.configured, vb.enabled = (lambda: True), (lambda: True)

    async def health(timeout=5.0):
        return {"ok": True, "tts_sample_rate": 24000, "voices": [], "asr": "sv", "tts": "cv"}

    async def speak(text, client, voice=""):
        yield b"\x00\x00" * 10
    vb.health, vb.speak = health, speak

    try:
        with c.websocket_connect("/v1/realtime"):
            raise AssertionError("no key must not connect")
    except WebSocketDisconnect:
        pass
    n = len(CALLS)
    PLAN.update(message="語音的答案", delay=0.0)
    with c.websocket_connect("/v1/realtime", subprotocols=["openteddy", f"openteddy-key.{token}"]) as ws:
        assert ws.accepted_subprotocol == "openteddy"
        ws.send_text(json.dumps({"type": "hello"}))
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["session_id"] and ready["sample_rate"] == 24000, ready
        sid = ready["session_id"]
        assert TR.sessions[sid]["title"].startswith("🔑 客服 App · ")
        ws.send_text(json.dumps({"type": "text", "text": "你好"}))
        got = []
        while not any(m.get("type") == "reply" for m in got if isinstance(m, dict)):
            got.append(ws.receive_json())
        assert next(m for m in got if m.get("type") == "reply")["text"] == "語音的答案"
    call = CALLS[n]
    assert call["api_key_id"] == KEY1["id"] and call["session_id"] == sid and call["context"] == {"voice": True}
    print("  ✓ realtime by subprotocol: a session made for the key, named in ready, turns run as the key")

    with c.websocket_connect("/v1/realtime", headers={"Authorization": f"Bearer {token2}"}) as ws:
        ws.send_text(json.dumps({"type": "hello", "session_id": sid}))
        err = ws.receive_json()
        assert err["type"] == "error" and err["code"] == "bad_session", err
    print("  ✓ by header too; another key's session → bad_session")

    spoke = []

    async def health2(timeout=5.0):
        return {"ok": True, "tts_sample_rate": 24000, "voices": ["male1", "sample"], "asr": "sv", "tts": "cv"}

    async def speak2(text, client, voice=""):
        spoke.append(voice)
        yield b"\x00\x00" * 10
    vb.health, vb.speak = health2, speak2

    def until(ws, *types):
        while True:
            m = ws.receive()
            if m.get("text"):
                d = json.loads(m["text"])
                if d.get("type") in types:
                    return d
    H = {"Authorization": f"Bearer {token}"}
    saved_name = getattr(vb.config, "voice_name", "")
    try:
        vb.config.voice_name = ""
        with c.websocket_connect("/v1/realtime", headers=H) as ws:
            ws.send_text(json.dumps({"type": "hello"}))
            ready = until(ws, "ready")
            assert ready["voice"] == "male1" and ready["voices"] == ["male1", "sample"], ready
        with c.websocket_connect("/v1/realtime", headers=H) as ws:
            ws.send_text(json.dumps({"type": "hello", "voice": "sample"}))
            assert until(ws, "ready")["voice"] == "sample"
            ws.send_text(json.dumps({"type": "text", "text": "你好"}))
            until(ws, "audio_end")
            assert spoke and set(spoke) == {"sample"}, spoke
            ws.send_text(json.dumps({"type": "voice", "voice": "male1"}))
            assert until(ws, "voice", "error") == {"type": "voice", "voice": "male1"}
            spoke.clear()
            ws.send_text(json.dumps({"type": "text", "text": "再說一次"}))
            until(ws, "audio_end")
            assert spoke and set(spoke) == {"male1"}, spoke
            ws.send_text(json.dumps({"type": "voice", "voice": "alloy"}))
            err = until(ws, "voice", "error")
            assert err["code"] == "voice_not_found" and "male1" in err["text"], err
        with c.websocket_connect("/v1/realtime", headers=H) as ws:
            ws.send_text(json.dumps({"type": "hello", "voice": "不存在"}))
            assert until(ws, "error", "ready")["code"] == "voice_not_found"
        print("  ✓ realtime voice: named in the hello (ready says which), changed mid-call, unknown → voice_not_found")
    finally:
        vb.config.voice_name = saved_name
        vb.configured, vb.enabled, vb.health, vb.speak = saved


# ── J. streaming speech to text ──────────────────────────────────────────────

def test_stt_stream(c, token):
    import math, struct, time as _time
    from starlette.websockets import WebSocketDisconnect
    vb = voice_bridge
    saved = (vb.configured, vb.enabled, vb.transcribe)
    vb.configured, vb.enabled = (lambda: True), (lambda: True)
    heard = []

    async def transcribe(pcm, client=None):
        heard.append(len(pcm))
        if not any(pcm):                       # the voice service's own VAD: silence → nothing
            return {"text": ""}
        return {"text": "退貨" if len(pcm) < 30000 else "退貨期限是幾天"}
    vb.transcribe = transcribe
    speech = struct.pack("<640h", *[int(8000 * math.sin(i / 3)) for i in range(640)])   # 40 ms, loud
    quiet = b"\x00\x00" * 640                                                          # 40 ms, silence
    H = {"Authorization": f"Bearer {token}"}

    def until(ws, *types):
        while True:
            d = ws.receive_json()
            if d["type"] in types:
                return d
    try:
        with c.websocket_connect("/v1/audio/stt-stream", headers=H) as ws:
            for _ in range(10):
                ws.send_bytes(quiet)                       # silence first: nothing happens
            seen = []
            for i in range(30):                            # 1.2 s of speech at real-time pace
                ws.send_bytes(speech)
                _time.sleep(0.04)
            ws.send_json({"type": "flush"})
            while True:
                d = ws.receive_json()
                seen.append(d)
                if d["type"] == "final":
                    break
            kinds = [d["type"] for d in seen]
            assert kinds[0] == "speech_started" and "partial" in kinds, kinds
            assert seen[-1] == {"type": "final", "text": "退貨期限是幾天"}, seen[-1]
            assert max(heard) <= 30 * 1280 + 16000, "only 0.5 s kept from before the speech"
            print(f"  ✓ partials while speaking ({kinds.count('partial')}), final on flush; silence before speech trimmed")

            ws.send_json({"type": "config", "endpoint_ms": 400})
            for _ in range(15):
                ws.send_bytes(speech)
            for _ in range(15):
                ws.send_bytes(quiet)
            assert until(ws, "final")["text"], "400 ms of silence ends the utterance by itself"
            ws.send_json({"type": "flush"})
            assert until(ws, "final") == {"type": "final", "text": ""}, "a flush with nothing said → empty final"
            print("  ✓ endpoint_ms: silence ends an utterance without a flush; an empty flush says so")

        with c.websocket_connect("/v1/audio/stt-stream", subprotocols=["openteddy", f"openteddy-key.{token}"]) as ws:
            assert ws.accepted_subprotocol == "openteddy"
        try:
            with c.websocket_connect("/v1/audio/stt-stream"):
                raise AssertionError("no key must not connect")
        except WebSocketDisconnect:
            pass
        vb.enabled = lambda: False
        with c.websocket_connect("/v1/audio/stt-stream", headers=H) as ws:
            assert ws.receive_json()["code"] == "voice_off"
        print("  ✓ key by header or subprotocol, none refused; voice off → said so")
    finally:
        vb.configured, vb.enabled, vb.transcribe = saved


# ── K. the role a conversation plays ─────────────────────────────────────────

def test_roles(c, token):
    H = {"Authorization": f"Bearer {token}"}
    PLAN.update(words=None, message="嗯", delay=0.0, shortcut=None)
    role = "你是品牌客戶王經理，正在比較三家供應商，說話直接，不推銷、不反問服務。"
    r = c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "您好"}],
                                                        "instructions": role}).json()
    sid = r["session_id"]
    assert CALLS[-1]["context"]["agent_persona"] == role
    c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "報價如下"}], "session_id": sid})
    assert CALLS[-1]["context"]["agent_persona"] == role, "the role stays for the conversation"
    c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "結束"}], "session_id": sid,
                                                    "instructions": ""})
    assert "agent_persona" not in CALLS[-1]["context"], "\"\" clears it"
    bad = c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "x"}], "instructions": 3})
    assert bad.status_code == 400 and bad.json()["error"]["param"] == "instructions"
    print("  ✓ instructions: the conversation's role on every turn until changed; \"\" clears; not a string → 400")

    r = c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "您好"}],
                                                        "agent": "品牌客戶・王經理"}).json()
    s2 = r["session_id"]
    assert TR.sessions[s2]["agent_id"] == "ag1" and "品牌客戶・王經理" in TR.sessions[s2]["title"]
    assert "agent_persona" not in CALLS[-1]["context"], "the agent's persona is the orchestrator's to add (live)"
    r = c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "x"}], "agent": "ag1",
                                                        "instructions": "這次客戶想比價。"}).json()
    assert TR.sessions[r["session_id"]]["agent_id"] == "ag1" and CALLS[-1]["context"]["agent_persona"] == "這次客戶想比價。"
    n = len(TR.sessions)
    r = c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "x"}], "agent": "不存在"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "agent_not_found" and len(TR.sessions) == n
    r = c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "x"}], "agent": "ag1",
                                                        "session_id": s2})
    assert r.status_code == 400 and r.json()["error"]["param"] == "agent"
    print("  ✓ agent by name or id → a session built from it; with instructions on top; unknown / on an existing session → 400")

    vb = voice_bridge
    saved = (vb.configured, vb.enabled, vb.health, vb.speak)
    vb.configured, vb.enabled = (lambda: True), (lambda: True)

    async def health(timeout=5.0):
        return {"ok": True, "tts_sample_rate": 24000}

    async def speak(text, client, voice=""):
        yield b"\x00\x00"
    vb.health, vb.speak = health, speak

    def until(ws, *types):
        while True:
            m = ws.receive()
            if m.get("text"):
                d = json.loads(m["text"])
                if d.get("type") in types:
                    return d
    try:
        with c.websocket_connect("/v1/realtime", headers=H) as ws:
            ws.send_text(json.dumps({"type": "hello", "agent": "品牌客戶・王經理", "instructions": "客戶今天很急。"}))
            ready = until(ws, "ready")
            assert ready["agent"] == "品牌客戶・王經理" and ready["instructions"] is True, ready
            assert TR.sessions[ready["session_id"]]["agent_id"] == "ag1"
            ws.send_text(json.dumps({"type": "text", "text": "您好，我是業務小林"}))
            until(ws, "reply")
            assert CALLS[-1]["context"] == {"voice": True, "agent_persona": "客戶今天很急。"}, CALLS[-1]["context"]
            ws.send_text(json.dumps({"type": "instructions", "instructions": "客戶改成很有耐心。"}))
            assert until(ws, "instructions")["set"] is True
            ws.send_text(json.dumps({"type": "text", "text": "那我們談談交期"}))
            until(ws, "reply")
            assert CALLS[-1]["context"]["agent_persona"] == "客戶改成很有耐心。"
        with c.websocket_connect("/v1/realtime", headers=H) as ws:
            ws.send_text(json.dumps({"type": "hello", "agent": "不存在"}))
            assert until(ws, "error", "ready")["code"] == "agent_not_found"
        print("  ✓ realtime: agent + instructions in the hello (named in ready), each spoken turn plays it, changed mid-call")
    finally:
        vb.configured, vb.enabled, vb.health, vb.speak = saved


# ── I. datasets ──────────────────────────────────────────────────────────────

class FakeKB:
    """knowledge.py's surface that public_api uses."""
    class KnowledgeError(Exception):
        pass

    def __init__(self):
        self.on = True
        self.ds = [{"id": "ds1", "name": "客服 FAQ", "description": "退換貨", "items": 12, "always": False},
                   {"id": "ds2", "name": "產品手冊", "description": "", "items": 40, "always": True}]
        self.picks = {}

    def enabled(self):
        return self.on

    async def list_datasets(self):
        return [dict(d) for d in self.ds]

    async def set_session_datasets(self, sid, refs):
        ids = []
        for r in refs:
            d = next((d for d in self.ds if r in (d["id"], d["name"])), None)
            if not d:
                raise FakeKB.KnowledgeError(f"找不到這些資料集：{r}")
            ids.append(d["id"])
        self.picks[sid] = ids
        return ids

    async def session_datasets(self, sid):
        return list(self.picks.get(sid, []))


def test_datasets(c, token, token2):
    import knowledge
    kb = FakeKB()
    saved = {n: getattr(knowledge, n) for n in ("enabled", "list_datasets", "set_session_datasets",
                                                  "session_datasets", "KnowledgeError")}
    for n in saved:
        setattr(knowledge, n, getattr(kb, n) if n != "KnowledgeError" else FakeKB.KnowledgeError)
    H = {"Authorization": f"Bearer {token}"}
    try:
        d = c.get("/v1/datasets", headers=H).json()
        assert d["enabled"] and [x["name"] for x in d["data"]] == ["客服 FAQ", "產品手冊"] and d["data"][0]["items"] == 12
        assert "sources" not in d["data"][0]
        print("  ✓ GET /v1/datasets: names, ids, sizes — no contents")

        PLAN.update(message="答", delay=0.0, shortcut=None)
        r = c.post("/v1/chat/completions", headers=H,
                   json={"messages": [{"role": "user", "content": "可以退貨嗎"}], "datasets": ["客服 FAQ"]}).json()
        sid = r["session_id"]
        assert CALLS[-1]["datasets"] == ["ds1"] and r["datasets"] == ["客服 FAQ"] and kb.picks[sid] == ["ds1"], r
        c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "那運費呢"}], "session_id": sid})
        assert CALLS[-1]["datasets"] == ["ds1"], "the session keeps its pick"
        r = c.post("/v1/chat/completions", headers=H,
                   json={"messages": [{"role": "user", "content": "換手冊"}], "session_id": sid, "datasets": ["ds2", "客服 FAQ"]}).json()
        assert CALLS[-1]["datasets"] == ["ds2", "ds1"] and r["datasets"] == ["產品手冊", "客服 FAQ"]
        r = c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "無"}]}).json()
        assert "datasets" not in CALLS[-1] and "datasets" not in r, "no pick → the owner's defaults (scope_for)"
        print("  ✓ chat: datasets by name or id, kept by the session, changed by naming others; none → defaults")

        r = c.post("/v1/chat/completions", headers=H,
                   json={"messages": [{"role": "user", "content": "我是客戶"}], "datasets": []}).json()
        sid = r["session_id"]
        assert CALLS[-1]["datasets"] == [] and r["datasets"] == [], r
        for _ in range(2):
            r = c.post("/v1/chat/completions", headers=H,
                       json={"messages": [{"role": "user", "content": "再問"}], "session_id": sid}).json()
            assert CALLS[-1]["datasets"] == [] and r["datasets"] == [], "[] holds for the whole conversation"
        r = c.post("/v1/chat/completions", headers=H,
                   json={"messages": [{"role": "user", "content": "查 FAQ"}], "session_id": sid, "datasets": ["客服 FAQ"]}).json()
        assert CALLS[-1]["datasets"] == ["ds1"]
        c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "再"}], "session_id": sid})
        assert CALLS[-1]["datasets"] == ["ds1"], "naming datasets again ends the [] choice"
        print("  ✓ chat: [] = no knowledge base, \"always\" datasets included, for the rest of the conversation")

        r = c.post("/v1/chat/completions", headers=H,
                   json={"messages": [{"role": "user", "content": "退貨"}], "agent": "客服・小美"}).json()
        sid = r["session_id"]
        assert CALLS[-1]["datasets"] == ["ds1"] and r["datasets"] == ["客服 FAQ"], r
        r = c.post("/v1/chat/completions", headers=H,
                   json={"messages": [{"role": "user", "content": "運費"}], "session_id": sid}).json()
        assert CALLS[-1]["datasets"] == ["ds1"] and r["datasets"] == ["客服 FAQ"], "later turns: the session's agent"
        r = c.post("/v1/chat/completions", headers=H,
                   json={"messages": [{"role": "user", "content": "x"}], "agent": "ag2", "datasets": ["產品手冊"]}).json()
        assert CALLS[-1]["datasets"] == ["ds2"], "the app's own datasets come before the agent's"
        r = c.post("/v1/chat/completions", headers=H,
                   json={"messages": [{"role": "user", "content": "x"}], "agent": "ag2", "datasets": []}).json()
        c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "y"}], "session_id": r["session_id"]})
        assert CALLS[-1]["datasets"] == [], "[] too, for the whole conversation"
        r = c.post("/v1/chat/completions", headers=H,
                   json={"messages": [{"role": "user", "content": "x"}], "agent": "品牌客戶・王經理"}).json()
        assert "datasets" not in CALLS[-1] and "datasets" not in r, "an agent that sets none: the owner's defaults"
        print("  ✓ chat: an agent brings its datasets; the app's datasets (or []) come first")

        n = len(TR.sessions)
        r = c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "x"}], "datasets": ["不存在"]})
        assert r.status_code == 400 and r.json()["error"]["code"] == "dataset_not_found" and len(TR.sessions) == n
        r = c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "x"}], "datasets": "客服 FAQ"})
        assert r.status_code == 400 and r.json()["error"]["param"] == "datasets"
        kb.on = False
        r = c.post("/v1/chat/completions", headers=H, json={"messages": [{"role": "user", "content": "x"}], "datasets": ["ds1"]})
        assert r.status_code == 400 and r.json()["error"]["code"] == "knowledge_off"
        assert c.get("/v1/datasets", headers=H).json() == {"object": "list", "enabled": False, "data": []}
        kb.on = True
        print("  ✓ an unknown dataset → 400 and no empty session left; not a list → 400; knowledge base off → knowledge_off")

        vb = voice_bridge
        vsaved = (vb.configured, vb.enabled, vb.health, vb.speak)
        vb.configured, vb.enabled = (lambda: True), (lambda: True)

        async def health(timeout=5.0):
            return {"ok": True, "tts_sample_rate": 24000}

        async def speak(text, client, voice=""):
            yield b"\x00\x00"
        vb.health, vb.speak = health, speak
        def until(ws, *types):
            """The next JSON message of one of these types; the spoken audio (binary) is skipped."""
            while True:
                m = ws.receive()
                if m.get("text"):
                    d = json.loads(m["text"])
                    if d.get("type") in types:
                        return d

        with c.websocket_connect("/v1/realtime", headers=H) as ws:
            ws.send_text(json.dumps({"type": "hello", "datasets": ["客服 FAQ"]}))
            ready = until(ws, "ready")
            assert ready["datasets"] == ["客服 FAQ"] and kb.picks[ready["session_id"]] == ["ds1"], ready
            ws.send_text(json.dumps({"type": "text", "text": "退貨要多久"}))
            until(ws, "reply")
            assert CALLS[-1]["datasets"] == ["ds1"] and CALLS[-1]["context"] == {"voice": True}
            ws.send_text(json.dumps({"type": "datasets", "datasets": ["產品手冊"]}))
            assert until(ws, "datasets")["datasets"] == ["產品手冊"]
            ws.send_text(json.dumps({"type": "text", "text": "保固多久"}))
            until(ws, "reply")
            assert CALLS[-1]["datasets"] == ["ds2"]
            ws.send_text(json.dumps({"type": "datasets", "datasets": ["沒有這個"]}))
            got = until(ws, "error", "datasets")
            assert got["type"] == "error" and got["code"] == "dataset_not_found", got
        with c.websocket_connect("/v1/realtime", headers=H) as ws:
            ws.send_text(json.dumps({"type": "hello", "agent": "客服・小美"}))
            assert until(ws, "ready")["datasets"] == ["客服 FAQ"]
            ws.send_text(json.dumps({"type": "text", "text": "退貨"}))
            until(ws, "reply")
            assert CALLS[-1]["datasets"] == ["ds1"], "realtime: the agent's datasets each turn"
        with c.websocket_connect("/v1/realtime", headers=H) as ws:
            ws.send_text(json.dumps({"type": "hello", "datasets": []}))
            assert until(ws, "ready")["datasets"] == []
            for say in ("你好", "多少錢"):
                ws.send_text(json.dumps({"type": "text", "text": say}))
                until(ws, "reply")
                assert CALLS[-1]["datasets"] == [], "[] in the hello holds for every turn"
            ws.send_text(json.dumps({"type": "datasets", "datasets": ["客服 FAQ"]}))
            until(ws, "datasets")
            ws.send_text(json.dumps({"type": "datasets", "datasets": []}))
            assert until(ws, "datasets")["datasets"] == []
            ws.send_text(json.dumps({"type": "text", "text": "再問"}))
            until(ws, "reply")
            assert CALLS[-1]["datasets"] == []
        with c.websocket_connect("/v1/realtime", headers=H) as ws:
            ws.send_text(json.dumps({"type": "hello", "datasets": ["不存在"]}))
            err = until(ws, "error", "ready")
            assert err["type"] == "error" and err["code"] == "dataset_not_found", err
        vb.configured, vb.enabled, vb.health, vb.speak = vsaved
        print("  ✓ realtime: datasets in the hello (named in ready), each turn reads them, switched mid-talk, unknown → error")
    finally:
        for n, v in saved.items():
            setattr(knowledge, n, v)


# ── L. LiveKit ───────────────────────────────────────────────────────────────

def _jwt_claims(tok, secret):
    import base64, hashlib, hmac
    head, body, sig = tok.split(".")
    good = base64.urlsafe_b64encode(hmac.new(secret.encode(), f"{head}.{body}".encode(),
                                             hashlib.sha256).digest()).rstrip(b"=").decode()
    assert sig == good, "signed with the LiveKit secret"
    assert json.loads(base64.urlsafe_b64decode(head + "==")) == {"alg": "HS256", "typ": "JWT"}
    return json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))


def test_livekit(c, token, token2):
    import time
    H = {"Authorization": f"Bearer {token}"}
    CHAT = "/v1/chat/completions"
    kid = run(api_keys.verify(token))["id"]
    r = c.post("/v1/livekit/token", headers=H, json={})
    assert r.status_code == 503 and r.json()["error"]["code"] == "not_configured", r.text
    env = {"LIVEKIT_URL": "wss://lk.example.com", "LIVEKIT_API_URL": "http://lk.internal:7880",
           "LIVEKIT_API_KEY": "APIk", "LIVEKIT_API_SECRET": "s" * 40}
    saved_env = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    calls, real = [], public_api._livekit

    async def fake(cfg, method, body, video):
        calls.append((cfg["api"], method, body, video))
        return {}
    public_api._livekit = fake
    try:
        r = c.post("/v1/livekit/token", headers=H,
                   json={"agent": "品牌客戶・王經理", "instructions": "很趕交期", "identity": "trainee-1", "name": "小林"})
        assert r.status_code == 200, r.text
        d = r.json()
        sid, room = d["session_id"], d["room"]
        assert d["url"] == "wss://lk.example.com" and d["agent"] == "品牌客戶・王經理" and d["instructions"] is True
        assert d["identity"] == "trainee-1" and d["expires_in"] == 900 and TR.sessions[sid]["agent_id"] == "ag1"
        claims = _jwt_claims(d["token"], env["LIVEKIT_API_SECRET"])
        assert claims["iss"] == "APIk" and claims["sub"] == "trainee-1" and claims["name"] == "小林"
        assert claims["video"] == {"room": room, "roomJoin": True, "canPublish": True, "canSubscribe": True,
                                   "canPublishData": True}, claims
        assert 0 < claims["exp"] - time.time() <= 900
        (api1, m1, b1, v1), (api2, m2, b2, v2) = calls
        assert api1 == "http://lk.internal:7880" and m1 == "RoomService/CreateRoom" and b1["name"] == room
        assert v1 == {"roomCreate": True}
        assert m2 == "AgentDispatchService/CreateDispatch" and b2["agent_name"] == "openteddy" and b2["room"] == room
        assert v2 == {"roomAdmin": True, "room": room}
        meta = json.loads(b2["metadata"])
        grant = meta["grant"]
        assert meta["session_id"] == sid and grant.startswith("otg-") and grant not in d["token"] + json.dumps(b1)
        print("  ✓ token: the key's conversation (agent, instructions), a room, the agent sent with a grant only it sees, a signed join token")

        G = {"Authorization": f"Bearer {grant}"}
        PLAN.update(message="交期能多快？", delay=0.0, shortcut=None, words=None)
        r = c.post(CHAT, headers=G, json={"messages": [{"role": "user", "content": "您好"}], "session_id": sid})
        assert r.status_code == 200 and r.json()["session_id"] == sid, r.text
        assert CALLS[-1]["api_key_id"] == kid and CALLS[-1]["context"]["agent_persona"] == "很趕交期"
        r = c.post(CHAT, headers=G, json={"messages": [{"role": "user", "content": "x"}]})
        assert r.status_code == 403 and r.json()["error"]["code"] == "grant_scope"
        other = c.post(CHAT, headers=H, json={"messages": [{"role": "user", "content": "別的"}]}).json()["session_id"]
        r = c.post(CHAT, headers=G, json={"messages": [{"role": "user", "content": "x"}], "session_id": other})
        assert r.status_code == 403, "a grant is for its one conversation, not the key's others"
        assert c.get("/v1/models", headers=G).status_code == 401, "a grant is no key"
        assert c.post("/v1/livekit/token", headers=G, json={}).status_code == 401
        assert run(public_api._key_or_grant(grant))["grant_session"] == sid
        print("  ✓ a grant acts as the key in its conversation only; anywhere else it is refused")

        bad = grant[:-3] + ("AAA" if not grant.endswith("AAA") else "BBB")
        assert c.post(CHAT, headers={"Authorization": f"Bearer {bad}"},
                      json={"messages": [{"role": "user", "content": "x"}], "session_id": sid}).status_code == 401
        old = run(api_keys.issue_grant(kid, sid, -1))
        assert c.post(CHAT, headers={"Authorization": f"Bearer {old}"},
                      json={"messages": [{"role": "user", "content": "x"}], "session_id": sid}).status_code == 401
        k2 = run(api_keys.verify(token2))["id"]
        stray = run(api_keys.issue_grant(k2, sid, 60))                # a grant of another key, for this session
        r = c.post(CHAT, headers={"Authorization": f"Bearer {stray}"},
                   json={"messages": [{"role": "user", "content": "x"}], "session_id": sid})
        assert r.status_code == 404, "a grant never reaches past its own key"
        tmp_row, tmp = run(api_keys.create("暫時 App"))
        t = c.post("/v1/livekit/token", headers={"Authorization": f"Bearer {tmp}"}, json={}).json()
        tg = json.loads(calls[-1][2]["metadata"])["grant"]
        ok = c.post(CHAT, headers={"Authorization": f"Bearer {tg}"},
                    json={"messages": [{"role": "user", "content": "x"}], "session_id": t["session_id"]})
        assert ok.status_code == 200
        run(api_keys.revoke(tmp_row["id"]))
        r = c.post(CHAT, headers={"Authorization": f"Bearer {tg}"},
                   json={"messages": [{"role": "user", "content": "x"}], "session_id": t["session_id"]})
        assert r.status_code == 401, "revoking the key ends its calls"
        print("  ✓ a tampered or expired grant, another key's grant, a revoked key's grant: refused")

        n = len(calls)
        r = c.post("/v1/livekit/token", headers=H, json={"session_id": sid}).json()
        assert r["session_id"] == sid and json.loads(calls[-1][2]["metadata"])["session_id"] == sid
        r = c.post("/v1/livekit/token", headers=H, json={"session_id": sid, "agent": "ag1"})
        assert r.status_code == 400 and r.json()["error"]["param"] == "agent"
        r = c.post("/v1/livekit/token", headers={"Authorization": f"Bearer {token2}"}, json={"session_id": sid})
        assert r.status_code == 404
        r = c.post("/v1/livekit/token", headers=H, json={"identity": "a b"})
        assert r.status_code == 400 and r.json()["error"]["param"] == "identity"
        assert len(calls) == n + 2, "nothing opened for a refused request"

        vb = voice_bridge
        vsaved = (vb.configured, vb.enabled, vb.health)
        vb.configured, vb.enabled = (lambda: True), (lambda: True)

        async def vhealth(timeout=5.0):
            return {"ok": True, "voices": ["male1", "sample"]}
        vb.health = vhealth
        try:
            r = c.post("/v1/livekit/token", headers=H, json={"voice": "male1"}).json()
            assert r["voice"] == "male1" and json.loads(calls[-1][2]["metadata"])["voice"] == "male1"
            n = len(calls)
            r = c.post("/v1/livekit/token", headers=H, json={"voice": "alloy"})
            assert r.status_code == 400 and r.json()["error"]["code"] == "voice_not_found", r.text
            assert c.post("/v1/livekit/token", headers=H, json={"voice": 3}).status_code == 400
            assert len(calls) == n, "no room for an unknown voice"
            c.post("/v1/livekit/token", headers=H, json={})
            assert "voice" not in json.loads(calls[-1][2]["metadata"]), "no voice named → the agent uses the set one"
        finally:
            vb.configured, vb.enabled, vb.health = vsaved
        print("  ✓ token with a voice: checked against the voice service, sent to the agent; unknown → 400, no room")

        async def down(cfg, method, body, video):
            raise public_api.ApiError(502, "LiveKit is unreachable (ConnectError).", type_="api_error",
                                      code="livekit_unreachable")
        public_api._livekit = down
        r = c.post("/v1/livekit/token", headers=H, json={})
        assert r.status_code == 502 and r.json()["error"]["code"] == "livekit_unreachable"
        print("  ✓ an earlier conversation goes on in a new room; another key's / agent on it / bad identity → refused; LiveKit down → 502")
    finally:
        public_api._livekit = real
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── H. privacy ───────────────────────────────────────────────────────────────

def test_privacy():
    import privacy
    from tools._context import reset_triggered_by, set_triggered_by
    req = types.SimpleNamespace(id="x", session_id=None, context={"api_key_id": "abc"})
    assert run(privacy.resolve(req, None)) == privacy.WEB_PERSON, "context alone must not make a key's person"
    tok = set_triggered_by("api_key")
    try:
        assert run(privacy.resolve(req, None)) == "api:abc"
        assert not privacy.has_account(None) and privacy.has_account("api:abc") is True
    finally:
        reset_triggered_by(tok)
    print("  ✓ api:<id> only through the api_key door, never from a caller's context")


if __name__ == "__main__":
    print("A. keys"); KEY1, TOKEN = test_keys()
    _k2, TOKEN2 = run(api_keys.create("另一個 App"))
    with TestClient(app) as client:
        print("B–D. http"); test_http(client, TOKEN, TOKEN2)
        print("F. audio"); test_audio(client, TOKEN)
        print("G. realtime"); test_realtime(client, TOKEN, TOKEN2)
        print("I. datasets"); test_datasets(client, TOKEN, TOKEN2)
        print("J. streaming speech to text"); test_stt_stream(client, TOKEN)
        print("K. roles"); test_roles(client, TOKEN)
        print("L. livekit"); test_livekit(client, TOKEN, TOKEN2)
    print("E. api-only port"); test_only_v1(TOKEN)
    print("H. privacy"); test_privacy()
    print("\nALL PUBLIC API TESTS PASS")
