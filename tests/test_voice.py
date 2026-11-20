"""
The 🎙 voice mode (voice_bridge.py) against a fake voice service.

  A. what is read aloud: markdown, tables, links, emoji and knowledge-base
     citations left out; a long work result cut to its gist; sentences
     for streaming
  B. a conversation over /voice/ws: not set up → said so; hello → ready
     with the service's sample rate; an utterance → transcript, the task
     started like POST /tasks auto_lane (voice context, the session), the
     answer's text and its audio; silence → nothing; talking over it stops
     the audio; slow work → "I'm on it" now and the result when it's done
  C. only OpenTeddy's own page may open /voice/ws; the chat lane is told
     the answer is read aloud

    .venv/bin/python tests/test_voice.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

import uvicorn
import websockets
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse

from config import config
import voice_bridge as V

RATE = 22050
HEARD: list = []
SPOKEN: list = []


def fake_voice_service() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"ok": True, "tts": "fake-tts", "asr": "fake-asr", "tts_sample_rate": RATE, "voices": ["me"]}

    @app.post("/asr")
    async def asr(request: Request):
        body = await request.body()
        HEARD.append(len(body))
        # 1 s of audio → a question; anything shorter is silence
        return JSONResponse({"text": "退貨期限是幾天？" if len(body) >= 32000 else "", "ms": 12})

    @app.post("/tts")
    async def tts(request: Request):
        text = (await request.json())["text"]
        SPOKEN.append(text)

        async def chunks():
            for _ in range(4 if "慢" not in text else 40):
                await asyncio.sleep(0.02 if "慢" not in text else 0.05)
                yield b"\x01\x00" * 480           # 480 samples
        return StreamingResponse(chunks(), media_type=f"audio/L16; rate={RATE}")
    return app


STARTED: list = []


def openteddy(slow: dict) -> FastAPI:
    app = FastAPI()

    async def start_task(**kw):
        STARTED.append(kw)

        async def job():
            await asyncio.sleep(slow.get("s", 0.05))
            return slow.get("answer", "收到商品後 7 天內可以申請退貨。（知識庫：faq.csv 第 2 列）")
        return "task-1", kw.get("session_id"), asyncio.ensure_future(job()), None

    async def await_task(task_id, job):
        return "completed", await job

    @app.websocket("/voice/ws")
    async def ws(w: WebSocket):
        await w.accept()
        await V.Conversation(w, start_task=start_task, await_task=await_task).run()
    return app


async def serve(app: FastAPI) -> tuple:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
    return server, task, port


async def recv_until(w, kind: str, audio: list, seen: list, secs: float = 5.0) -> dict:
    async def go():
        while True:
            m = await w.recv()
            if isinstance(m, bytes):
                audio.append(m)
                continue
            d = json.loads(m)
            seen.append(d)
            if d.get("type") == kind:
                return d
    return await asyncio.wait_for(go(), secs)


# ── A ────────────────────────────────────────────────────────────────────────

def test_spoken() -> None:
    md = ("## 退貨說明\n\n**退貨期限**：收到商品後 7 天內。（知識庫：faq.csv 第 2 列）\n"
          "- 運費：滿 NT$990 免運 🚚\n| 欄 | 值 |\n|---|---|\n| a | b |\n詳見 https://shop.example/faq")
    s = V.spoken(md)
    assert s == "退貨說明，退貨期限：收到商品後 7 天內。運費：滿 NT$990 免運，詳見 連結", s
    long = "第一句話說明了重點。" + "接著是很多細節的說明。" * 40
    cut = V.spoken(long, V.SPOKEN_MAX_CHARS)
    assert cut.startswith("第一句話說明了重點。") and cut.endswith(V.DONE_MORE) and len(cut) <= V.SPOKEN_MAX_CHARS + 20
    assert V.sentences("你好。今天要做什麼？好的！") == ["你好。今天要做什麼？", "好的！"], V.sentences("你好。今天要做什麼？好的！")
    assert V.sentences("收到商品後七天內可以申請退貨，商品須保持全新。") == ["收到商品後七天內可以申請退貨，商品須保持全新。"], \
        "one call for a sentence: each call waits for its first chunk again"
    parts = V.sentences("這是一句很長很長的話，" * 20)
    assert all(len(p) <= 120 for p in parts) and "".join(parts) == "這是一句很長很長的話，" * 20
    print("  ✓ read aloud: no markdown, tables, links, emoji or citations; long results cut to the gist; sentences")


# ── B ────────────────────────────────────────────────────────────────────────

async def test_conversation() -> None:
    svc, svc_task, svc_port = await serve(fake_voice_service())
    slow: dict = {}
    ot, ot_task, ot_port = await serve(openteddy(slow))
    url = f"ws://127.0.0.1:{ot_port}/voice/ws"
    try:
        config.voice_service_url = ""
        async with websockets.connect(url) as w:
            await w.send(json.dumps({"type": "hello", "session_id": "s1"}))
            m = json.loads(await w.recv())
            assert m["type"] == "error" and m["code"] == "not_configured", m
        print("  ✓ no voice service set: said so")

        config.voice_service_url = f"http://127.0.0.1:{svc_port}"
        V.WORK_ACK_S = 1.0
        async with websockets.connect(url) as w:
            audio: list = []
            seen: list = []
            await w.send(json.dumps({"type": "hello", "session_id": "s1"}))
            ready = await recv_until(w, "ready", audio, seen)
            assert ready["sample_rate"] == RATE and ready["asr_rate"] == 16000 and ready["voice"] == "me", ready

            await w.send(b"\x00\x00" * 8000)        # half a second: nothing said
            while not (seen[-1:] and seen[-1].get("value") == "listening" and len(seen) > 2):
                await recv_until(w, "state", audio, seen)
            await asyncio.sleep(0.2)
            assert not STARTED, "silence starts nothing"

            seen.clear()
            await w.send(b"\x00\x00" * 16000)       # a sentence
            t = await recv_until(w, "transcript", audio, seen)
            assert t["text"] == "退貨期限是幾天？"
            r = await recv_until(w, "reply", audio, seen)
            assert "7 天" in r["text"] and "faq.csv" in r["text"], "the screen keeps the citation"
            await recv_until(w, "audio_end", audio, seen)
            await recv_until(w, "state", audio, seen)
            assert STARTED[-1]["auto_lane"] is True and STARTED[-1]["session_id"] == "s1" \
                and STARTED[-1]["context"] == {"voice": True}, STARTED[-1]
            assert SPOKEN[-1] == "收到商品後 7 天內可以申請退貨。", "the citation isn't read aloud"
            assert sum(len(a) for a in audio) == 4 * 960
            states = [d["value"] for d in seen if d["type"] == "state"]
            assert states[0] == "thinking" and states.index("speaking") > 0 and states[-1] == "listening", states
            print("  ✓ an utterance: transcript, the task as auto_lane in the session, the answer shown and spoken")

            # Talking over it stops the audio.
            slow["answer"] = "慢慢說的一段很長的回答。"
            seen.clear()
            await w.send(b"\x00\x00" * 16000)
            await recv_until(w, "state", audio, seen)
            while not any(d.get("value") == "speaking" for d in seen if d["type"] == "state"):
                await recv_until(w, "state", audio, seen)
            await asyncio.sleep(0.2)
            await w.send(json.dumps({"type": "interrupt"}))
            end = await recv_until(w, "audio_end", audio, seen)
            assert end.get("interrupted") is True
            print("  ✓ talking over the answer stops it")

            # Slow work: "I'm on it" now, the result later.
            slow.update(s=1.6, answer="報告已經做好了，放在 report.html。")
            seen.clear()
            await w.send(b"\x00\x00" * 16000)
            working = await recv_until(w, "working", audio, seen, secs=4)
            assert working["text"] == V.ACK
            done = await recv_until(w, "done", audio, seen, secs=5)
            assert "report.html" in done["text"]
            while sum(d["type"] == "audio_end" for d in seen) < 2:     # the "on it", then the result
                await recv_until(w, "audio_end", audio, seen)
            said = "".join(SPOKEN)
            assert V.ACK in said and said.endswith("報告已經做好了，放在 report.html。"), SPOKEN[-4:]
            print("  ✓ slow work: \"I'm on it\" at once, the result spoken when it's done")
    finally:
        for s, t in ((ot, ot_task), (svc, svc_task)):
            s.should_exit = True
            await t


# ── C ────────────────────────────────────────────────────────────────────────

async def test_guards() -> None:
    import main as M
    app = FastAPI()
    app.router.routes.extend(r for r in M.app.routes if getattr(r, "path", "") == "/voice/ws")
    srv, task, port = await serve(app)
    config.voice_service_url = ""
    try:
        try:
            async with websockets.connect(f"ws://127.0.0.1:{port}/voice/ws", origin="https://evil.example") as w:
                await w.recv()
            raise AssertionError("another site's page got in")
        except (websockets.exceptions.InvalidStatus, websockets.exceptions.ConnectionClosed):
            pass
        async with websockets.connect(f"ws://127.0.0.1:{port}/voice/ws", origin=f"http://127.0.0.1:{port}") as w:
            await w.send(json.dumps({"type": "hello"}))
            assert json.loads(await w.recv())["code"] == "not_configured"
    finally:
        srv.should_exit = True
        await task
    print("  ✓ /voice/ws: OpenTeddy's own page only")

    # The https door (https_listener.py): same app, a certificate made here.
    import ssl
    import tempfile
    import httpx
    import https_listener as H
    config.db_path = os.path.join(tempfile.mkdtemp(), "t.db")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    hport = s.getsockname()[1]
    s.close()
    os.environ.update(OPENTEDDY_HTTPS_PORT=str(hport), OPENTEDDY_HOST="127.0.0.1", OPENTEDDY_HTTPS_NAMES="teddy.local")
    ok_app = FastAPI()
    ok_app.router.routes.extend(r for r in M.app.routes if getattr(r, "path", "") == "/voice/ws")

    @ok_app.get("/ping")
    async def ping():
        return {"pong": True}
    try:
        assert await H.start(ok_app)
        for _ in range(100):
            if H._server and H._server.started:
                break
            await asyncio.sleep(0.05)
        async with httpx.AsyncClient(verify=False) as c:
            assert (await c.get(f"https://127.0.0.1:{hport}/ping")).json() == {"pong": True}
        from cryptography import x509
        crt, key = H._paths()
        cert = x509.load_pem_x509_certificate(open(crt, "rb").read())
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        assert "teddy.local" in san.get_values_for_type(x509.DNSName)
        assert any(str(i) == "127.0.0.1" for i in san.get_values_for_type(x509.IPAddress))
        assert oct(os.stat(key).st_mode & 0o777) == "0o600", "the key is readable only by its owner"
        ctx = ssl.create_default_context()
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
        async with websockets.connect(f"wss://127.0.0.1:{hport}/voice/ws", ssl=ctx,
                                      origin=f"https://127.0.0.1:{hport}") as w:
            await w.send(json.dumps({"type": "hello"}))
            assert json.loads(await w.recv())["code"] == "not_configured"
        mtime = os.stat(crt).st_mtime
        H.ensure_cert()
        assert os.stat(crt).st_mtime == mtime, "made once, then kept"
    finally:
        await H.stop()
        for k in ("OPENTEDDY_HTTPS_PORT", "OPENTEDDY_HTTPS_NAMES"):
            os.environ.pop(k, None)
    print("  ✓ https door: same app over https and wss, a certificate for this machine's names, made once")

    from orchestrator import Orchestrator
    from models import TaskRequest
    seen: dict = {}

    async def recent(req):
        return ""

    async def complete(prompt, system, **k):
        seen["p"] = prompt
        raise RuntimeError("stop")
    o = types.SimpleNamespace(memory=None, _recent_turns_block=recent, _orchestrator_complete=complete)
    for ctx, has in (({"voice": True}, True), ({}, False)):
        try:
            await Orchestrator._fast_chat_response(o, TaskRequest(goal="你好", session_id="s", context=ctx), "chat", {})
        except RuntimeError:
            pass
        assert ("念出來" in seen["p"]) == has, seen["p"]
    print("  ✓ the chat lane is told when its answer is read aloud")


async def main() -> None:
    print("A. read aloud"); test_spoken()
    print("B. a conversation"); await test_conversation()
    print("C. guards"); await test_guards()
    print("\nALL VOICE TESTS PASS")


asyncio.run(main())
