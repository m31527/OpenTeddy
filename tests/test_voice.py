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
  D. turning it on (voice_setup.py): what this machine can do, and the
     install — build when missing or older than contrib/voice, start or
     replace the container, follow the download, turn voice mode on at the
     end; failures said with their reason; turning it off stops the service
  E. speaking while the answer is being written
  F. the voices in Settings: the list with which one speaks, hearing one,
     the words in a recording, adding a voice (only with the person's word
     that it is theirs to use; the voice that spoke until then keeps
     speaking), removing one (the choice falls back), a voice service from
     before voices could be added

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
HEALTH: dict = {"ok": True}


def fake_voice_service() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"tts": "fake-tts", "asr": "fake-asr", "tts_sample_rate": RATE, "voices": ["me"], **HEALTH}

    @app.post("/asr")
    async def asr(request: Request):
        body = await request.body()
        HEARD.append(len(body))
        # 1 s of audio → a question; 0.75 s → a noise heard as 「。」; shorter → silence
        n = len(body)
        return JSONResponse({"text": "退貨期限是幾天？" if n >= 32000 else "。" if n >= 24000 else "", "ms": 12})

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
    # English: ". " ends a sentence (not "3.5"), lines join with ", ", a long result
    # ends in English, and OpenTeddy's own phrases follow the person's language.
    assert V.sentences("Hello there. How are you? Version 3.5 is out.") == \
        ["Hello there.", "How are you?", "Version 3.5 is out."], V.sentences("Hello there. How are you? Version 3.5 is out.")
    assert V.spoken("## Returns\n- **Deadline**: 7 days (source: faq.csv row 2)\n- Shipping: free over $30") == \
        "Returns, Deadline: 7 days, Shipping: free over $30", V.spoken("## Returns\n- **Deadline**: 7 days (source: faq.csv row 2)\n- Shipping: free over $30")
    long_en = V.spoken("The key point is first. " + "More detail follows here. " * 20, V.SPOKEN_MAX_CHARS)
    assert long_en.startswith("The key point is first.") and long_en.endswith(V.EN[V.DONE_MORE]), long_en
    # A citation said as a clause is dropped (the owner's examples, 11/24); a product code isn't one.
    assert V.spoken("尺寸需要依您的包裝空間來評估，這是來自編號 M-016 的資訊。") == "尺寸需要依您的包裝空間來評估。"
    assert V.spoken("我們可以提供到廠稽核、教育訓練，這些內容都可以在知識庫的編號 M-130 中找到詳細說明喔。") == \
        "我們可以提供到廠稽核、教育訓練。"
    assert V.spoken("根據知識庫，退貨期限是七天。") == "退貨期限是七天。"
    assert V.spoken("型號 A-100 的價格是 1200 元。") == "型號 A-100 的價格是 1200 元。"
    assert V.spoken("The deadline is 7 days, according to entry M-016. Shipping is free.") == \
        "The deadline is 7 days. Shipping is free."
    assert V.english("What can you do for me?") and not V.english("退貨幾天？") and not V.english("OpenTeddy 是什麼")
    assert V.phrase(V.ACK, True) == V.EN[V.ACK] and V.phrase(V.ACK, False) == V.ACK
    print("  ✓ read aloud: no markdown, tables, links, emoji or citations; long results cut to the gist; sentences "
          "(Chinese and English); OpenTeddy's own phrases in the person's language")


# ── B ────────────────────────────────────────────────────────────────────────

async def test_conversation() -> None:
    svc, svc_task, svc_port = await serve(fake_voice_service())
    slow: dict = {}
    ot, ot_task, ot_port = await serve(openteddy(slow))
    url = f"ws://127.0.0.1:{ot_port}/voice/ws"
    try:
        config.voice_enabled = False
        config.voice_service_url = f"http://127.0.0.1:{svc_port}"
        async with websockets.connect(url) as w:
            await w.send(json.dumps({"type": "hello", "session_id": "s1"}))
            m = json.loads(await w.recv())
            assert m["type"] == "error" and m["code"] == "not_enabled", m
        config.voice_enabled = True
        config.voice_service_url = ""
        async with websockets.connect(url) as w:
            await w.send(json.dumps({"type": "hello", "session_id": "s1"}))
            m = json.loads(await w.recv())
            assert m["type"] == "error" and m["code"] == "not_configured", m
        config.voice_service_url = f"http://127.0.0.1:{svc_port}"
        HEALTH.update(ok=False, stage="downloading")
        async with websockets.connect(url) as w:
            await w.send(json.dumps({"type": "hello", "session_id": "s1"}))
            m = json.loads(await w.recv())
            assert m["type"] == "error" and m["code"] == "loading", m
        HEALTH.clear()
        HEALTH["ok"] = True
        print("  ✓ voice mode off, no voice service set, or the service still downloading: said so")

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
            await w.send(b"\x00\x00" * 12000)       # a tap or a breath, heard as 「。」
            while not (seen[-1:] and seen[-1].get("value") == "listening" and len(seen) > 1):
                await recv_until(w, "state", audio, seen)
            await asyncio.sleep(0.2)
            assert not STARTED and not any(d["type"] == "transcript" for d in seen), "punctuation alone starts nothing"

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
    config.voice_enabled = True
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
        # Through a tunnel to an https domain: the page is the forwarded host.
        async with websockets.connect(f"ws://127.0.0.1:{port}/voice/ws", origin="https://teddy.example.com",
                                      additional_headers={"X-Forwarded-Host": "teddy.example.com"}) as w:
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

    # The start-up warm-up loads each model at the context size its calls use;
    # at Ollama's default the first real call loaded it again (5.7 s, demo2 11/21).
    sent: list = []

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json=None, **k):
            sent.append(json)
            return types.SimpleNamespace(status_code=200, text="")
    import httpx as _httpx
    real_client, saved = _httpx.AsyncClient, (config.gemma_model, config.qwen_model, config.vllm_models)
    try:
        _httpx.AsyncClient = _Client
        config.gemma_model, config.qwen_model, config.vllm_models = "gemma-x", "qwen-x", ""
        await M._warmup_ollama_models()
        ctx = {p["model"]: p["options"]["num_ctx"] for p in sent}
        assert ctx == {"gemma-x": int(config.gemma_num_ctx), "qwen-x": int(config.qwen_num_ctx)}, ctx
        sent.clear()
        config.vllm_models = "qwen-x=http://127.0.0.1:8001"
        import local_engine
        real_system = local_engine.platform.system
        local_engine.platform.system = lambda: "Linux"          # routes are off on macOS by design
        try:
            await M._warmup_ollama_models()
        finally:
            local_engine.platform.system = real_system
        assert [p["model"] for p in sent] == ["gemma-x"], "a model on vLLM isn't loaded into Ollama"
    finally:
        _httpx.AsyncClient = real_client
        config.gemma_model, config.qwen_model, config.vllm_models = saved
    print("  ✓ start-up warm-up at the context size the real calls use (no second load)")

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
    # Asked in English, the note is English and asks for English: a Chinese
    # note made the answer Chinese (demo2, 11/21).
    try:
        await Orchestrator._fast_chat_response(
            o, TaskRequest(goal="What can you help me with today?", session_id="s", context={"voice": True}), "chat", {})
    except RuntimeError:
        pass
    assert "reply in English" in seen["p"] and "念出來" not in seen["p"], seen["p"]
    print("  ✓ the chat lane is told when its answer is read aloud, in the language the person spoke")

    # "Yeah." twice is not a question asked again (it sent a noise to the
    # planner, which looked up the date: demo2, 11/22); a real question is.
    from datetime import datetime
    past: list = []

    async def list_tasks(limit=4, session_id=None):
        return [{"id": "old", "status": "completed", "goal": past[0], "created_at": datetime.utcnow().isoformat()}]
    o = types.SimpleNamespace(tracker=types.SimpleNamespace(list_tasks=list_tasks))
    for said, again in (("Yeah.", False), ("謝謝", False), ("好。", False), ("Thank you", False),
                        ("今天幾月幾號？", True), ("幾點了", True), ("what time is it", True)):
        past[:] = [said]
        got = await Orchestrator._asked_again(o, TaskRequest(goal=said, session_id="s"))
        assert got is again, (said, got)
    print("  ✓ a word or two said twice isn't a question asked again; a question is")


# ── D ────────────────────────────────────────────────────────────────────────

class FakeDocker:
    """docker as voice_setup sees it: an image, a container, a base."""

    def __init__(self, image_rev=None, container=None, base=True, build_rc=0, crash=False):
        self.image_rev, self.container, self.base = image_rev, container, base
        self.build_rc, self.crash, self.calls = build_rc, crash, []

    async def __call__(self, *args, timeout=30.0):
        import voice_setup as S
        self.calls.append(args)
        a = list(args)
        if a[:1] == ["version"]:
            return 0, "28.0.1"
        if a[:2] == ["image", "inspect"]:
            if a[2] == S.IMAGE:
                return (0, self.image_rev or "<no value>") if self.image_rev is not None else (1, "No such image")
            return (0, "sha256:x") if self.base else (1, "No such image")
        if a[:2] == ["container", "inspect"]:
            if not self.container:
                return 1, "No such container"
            status, rev = self.container
            if "{{.RestartCount}}" in a[-1]:
                return 0, f"{'exited' if self.crash else status}|0"
            if "|" in a[-1]:
                return 0, f"{status}|{rev or '<no value>'}"
            return 0, status
        if a[0] == "run":
            self.container = ("running", self.image_rev)
            return 0, "abc123"
        if a[0] == "start":
            self.container = ("running", self.container[1])
            return 0, S.NAME
        if a[:2] == ["rm", "-f"]:
            self.container = None
            return 0, S.NAME
        if a[0] == "stop":
            self.container = ("exited", self.container[1])
            return 0, S.NAME
        if a[0] == "logs":
            return 0, "Traceback: CUDA out of memory"
        return 0, ""


async def test_setup() -> None:
    import shutil
    import tempfile
    import voice_setup as S
    real = (S.platform.system, S.platform.machine, S.shutil.which, S._docker, S._build, S.URL, S.MODELS, S.VOICES,
            S.asyncio.sleep, S.shutil.disk_usage)
    tmp = tempfile.mkdtemp()
    try:
        S.platform.system = lambda: "Darwin"
        c = await S.check()
        assert not c["installable"] and c["reason"] == "not_linux", c
        S.platform.system, S.platform.machine = (lambda: "Linux"), (lambda: "aarch64")
        S.shutil.which = lambda name: None if name == "nvidia-smi" else f"/usr/bin/{name}"
        S._docker = FakeDocker()
        assert (await S.check())["reason"] == "no_gpu"
        S.shutil.which = lambda name: f"/usr/bin/{name}"
        S.MODELS, S.VOICES = __import__("pathlib").Path(tmp, "models"), __import__("pathlib").Path(tmp, "voices")
        S.shutil.disk_usage = lambda p: types.SimpleNamespace(free=500e9)
        assert S.base_image() == "vllm/vllm-openai:v0.33.0-aarch64"
        c = await S.check()
        assert c["installable"] and not c["image"] and not c["container"] and c["download_gb"] == S.MODEL_GB, c
        S._docker = FakeDocker(base=False)
        c = await S.check()
        assert c["download_gb"] == S.MODEL_GB + S.BASE_GB, "no vLLM image here: the runtime is downloaded too"
        S.shutil.disk_usage = lambda p: types.SimpleNamespace(free=20e9)
        c = await S.check()
        assert not c["installable"] and c["reason"] == "low_disk", c
        S.shutil.disk_usage = lambda p: types.SimpleNamespace(free=500e9)
        rev = S.source_rev()
        S._docker = FakeDocker(image_rev=rev, container=("exited", rev))
        c = await S.check()
        assert c["installable"] and c["container"] == "exited" and not c["outdated"], c
        S._docker = FakeDocker(image_rev="", container=("running", ""))
        assert (await S.check())["outdated"], "built before the rev label (demo2's first one): an update"
        print("  ✓ what this machine can do: not Linux / no GPU / not enough disk said; the runtime counted only "
              "when missing; an older build is an update")

        # A fake voice service whose health walks through the stages.
        stages = iter([{"stage": "starting"}, {"stage": "downloading", "detail": "FunAudioLLM/SenseVoiceSmall"},
                       {"stage": "loading"}, {"ok": True, "stage": "ready"}])
        app = FastAPI()
        latest: dict = {}

        @app.get("/health")
        async def health():
            latest.clear()
            latest.update(next(stages, {"ok": True, "stage": "ready"}))
            return latest
        srv, task, port = await serve(app)
        S.URL = f"http://127.0.0.1:{port}"

        built: list = []
        phases: list = []

        async def quick(_s):                       # each wait while it gets ready: what it said it was doing
            phases.append(S.JOB.phase)
            await real[8](0)
        S.asyncio.sleep = quick

        async def build():
            built.append(rev)
            S._docker.image_rev = rev
            S._on_build_line("#7 [3/6] RUN pip install funasr")
            return 0, ""
        S._build = build
        turned_on: list = []

        async def on_ready(url):
            turned_on.append(url)

        async def run(docker):
            S._docker = docker
            S.begin(on_ready)
            while S.running():
                await real[8](0.005)
            return S.job()

        try:
            d = FakeDocker()
            j = await run(d)
            assert j["state"] == "ready" and turned_on == [S.URL] and built == [rev], j
            runs = [c for c in d.calls if c[0] == "run"]
            assert len(runs) == 1 and "--gpus" in runs[0] and f"127.0.0.1:{S.PORT}:{S.PORT}" in runs[0] \
                and f"{S.MODELS}:/models" in runs[0] and "--restart" in runs[0], runs
            assert "download" in phases and "load" in phases, phases
            print("  ✓ nothing installed: built, started (GPU, this machine only, models mounted), download "
                  "followed, voice mode turned on")

            built.clear(); turned_on.clear()
            stages = iter([{"ok": True, "stage": "ready"}])
            d = FakeDocker(image_rev=rev, container=("exited", rev))
            j = await run(d)
            assert j["state"] == "ready" and not built and [c[0] for c in d.calls if c[0] in ("run", "start", "rm")] == ["start"]
            built.clear(); turned_on.clear()
            stages = iter([{"ok": True, "stage": "ready"}])
            d = FakeDocker(image_rev="", container=("running", ""))
            j = await run(d)
            assert j["state"] == "ready" and built == [rev] and \
                [c[0] for c in d.calls if c[0] in ("run", "start", "rm")] == ["rm", "run"], d.calls
            print("  ✓ already installed: just started; an older build: rebuilt and replaced (models kept)")

            async def bad_build():
                S._on_build_line("#9 ERROR: pip failed")
                return 1, "#9 ERROR: pip failed"
            S._build = bad_build
            turned_on.clear()
            j = await run(FakeDocker())
            assert j["state"] == "error" and j["error"] == "build_failed" and "pip failed" in j["detail"] and not turned_on, j
            S._build = build
            stages = iter([{"stage": "loading"}] * 50)
            j = await run(FakeDocker(crash=True))
            assert j["state"] == "error" and j["error"] == "crashed" and "out of memory" in j["detail"], j
            stages = iter([{"stage": "error", "detail": "OSError: download failed"}])
            j = await run(FakeDocker())
            assert j["state"] == "error" and j["error"] == "load_failed" and "download failed" in j["detail"], j
            assert not turned_on
            print("  ✓ a failed build, a crash, a failed download: said, with the reason; voice mode stays off")

            d = FakeDocker(image_rev=rev, container=("running", rev))
            S._docker = d
            assert await S.stop() and d.container[0] == "exited"
            assert not await S.stop(), "already stopped"
            print("  ✓ turning it off stops the service (models stay)")

            # The endpoints, on a settings store of their own.
            import main as M
            M.settings_store._db_path = os.path.join(tmp, "settings.db")
            await M.settings_store.init()
            S._docker = FakeDocker()
            config.voice_enabled, config.voice_service_url = False, ""
            st = await M.voice_status()
            assert st["enabled"] is False and st["setup"]["installable"] and st["setup"]["job"]["state"] in ("ready", "error")
            svc, svc_task, svc_port = await serve(fake_voice_service())
            try:
                r = await M.voice_enable(M._VoiceEnableBody(url=f"http://127.0.0.1:{svc_port}"))
                assert r["ok"] and config.voice_enabled and config.voice_service_url.endswith(str(svc_port)), r
                st = await M.voice_status()
                assert st["enabled"] and st["ok"] and "setup" not in st, st
                d = FakeDocker(image_rev=rev, container=("running", rev))
                S._docker = d
                r = await M.voice_disable()
                assert r == {"ok": True, "stopped": False} and not config.voice_enabled, \
                    "a service on another machine isn't ours to stop"
                S.URL = f"http://127.0.0.1:{S.PORT}"
                await M.settings_store.update_many({"voice_service_url": S.URL})
                r = await M.voice_disable()
                assert r["stopped"] and d.container[0] == "exited", r
                r = await M.voice_enable(M._VoiceEnableBody(url="http://127.0.0.1:1"))
                assert not r["ok"] and not config.voice_enabled
            finally:
                svc.should_exit = True
                await svc_task
            print("  ✓ /voice/status, /voice/enable (a service elsewhere), /voice/disable (stops only its own)")
        finally:
            srv.should_exit = True
            await task
    finally:
        (S.platform.system, S.platform.machine, S.shutil.which, S._docker, S._build, S.URL, S.MODELS, S.VOICES,
         S.asyncio.sleep, S.shutil.disk_usage) = real
        shutil.rmtree(tmp, ignore_errors=True)


# ── E. speaking while the answer is being written ────────────────────────────

def test_next_sentence() -> None:
    ns = V.next_sentence
    assert ns("你好，今天") is None and ns("你好，今天", final=True) == ("你好，今天", "")
    assert ns("好。今天天氣") is None, "「好。」 alone waits for more"
    assert ns("好。今天天氣很好。明天") == ("好。今天天氣很好。", "明天")
    assert ns("Version 3.5 is") is None and ns("Hello there. How") == ("Hello there.", " How")
    assert ns("第一點\n第二點") == ("第一點\n", "第二點") or ns("第一點\n第二點") is None
    long = "這是一段很長的說明，" * 8
    s, rest = ns(long)
    assert s.endswith("，") and len(s) <= V.LONG_RUN and s + rest == long
    print("  ✓ the first sentence of text still being written: not 「好。」 alone, not inside 3.5, long runs at a comma")


def openteddy_streaming(plan: dict) -> FastAPI:
    """start_task's answer arrives in pieces (as gate.py streams it) before the task is done."""
    import answer_stream
    app = FastAPI()

    async def start_task(**kw):
        STARTED.append(kw)
        tid = kw["task_id"]

        async def job():
            for piece in plan["pieces"]:
                await asyncio.sleep(plan.get("gap", 0.15))
                plan.setdefault("pushed_at", []).append(asyncio.get_running_loop().time())
                answer_stream.push(tid, piece)
            answer_stream.end(tid)
            plan["ended_at"] = asyncio.get_running_loop().time()
            await asyncio.sleep(plan.get("tail", 0.3))           # tracker + memory writes
            return plan.get("final", "".join(plan["pieces"]))
        return tid, kw.get("session_id"), asyncio.ensure_future(job()), None

    async def await_task(task_id, job):
        return "completed", await job

    @app.websocket("/voice/ws")
    async def ws(w: WebSocket):
        await w.accept()
        await V.Conversation(w, start_task=start_task, await_task=await_task).run()
    return app


async def test_streamed_speech() -> None:
    svc, svc_task, svc_port = await serve(fake_voice_service())
    plan: dict = {"pieces": ["收到商品", "後七天內", "可以申請退貨。", "商品需保持", "全新包裝完整。", "運費由我們負擔。"]}
    ot, ot_task, ot_port = await serve(openteddy_streaming(plan))
    try:
        config.voice_enabled = True
        config.voice_service_url = f"http://127.0.0.1:{svc_port}"
        async with websockets.connect(f"ws://127.0.0.1:{ot_port}/voice/ws") as w:
            await w.send(json.dumps({"type": "hello", "session_id": "s9"}))
            assert json.loads(await w.recv())["type"] == "ready"
            SPOKEN.clear()
            await w.send(json.dumps({"type": "text", "text": "退貨怎麼算"}))
            loop = asyncio.get_running_loop()
            audio, seen, first_audio = [], [], None

            async def collect():
                nonlocal first_audio
                while True:
                    m = await w.recv()
                    if isinstance(m, bytes):
                        first_audio = first_audio or loop.time()
                        audio.append(m)
                        continue
                    d = json.loads(m)
                    seen.append(d)
                    if d.get("type") == "audio_end":
                        return
            await asyncio.wait_for(collect(), 10)
            assert first_audio < plan["ended_at"], "the first sentence is heard before the answer is fully written"
            assert SPOKEN[0] == "收到商品後七天內可以申請退貨。", SPOKEN
            assert SPOKEN == ["收到商品後七天內可以申請退貨。", "商品需保持全新包裝完整。", "運費由我們負擔。"], SPOKEN
            reply = next(d for d in seen if d["type"] == "reply")
            assert reply["text"] == "".join(plan["pieces"]) and seen[-1]["type"] == "audio_end"
            assert STARTED[-1]["task_id"] and STARTED[-1]["context"] == {"voice": True}
            await asyncio.sleep(0.5)
            assert not any(d["type"] == "done" for d in seen) and len(SPOKEN) == 3, "the same answer isn't spoken twice"
            print(f"  ✓ spoken sentence by sentence while being written: first sound "
                  f"{(first_audio - plan['pushed_at'][0]) * 1000:.0f} ms after the first words, "
                  f"{(plan['ended_at'] - first_audio) * 1000:.0f} ms before the answer was complete; reply = the whole text")

            # A few words, then the turn went to the agent: the result comes as "done", spoken.
            plan.update(pieces=["我幫你查一下最新的資料。"], final="台北今天晴，最高 28 度。", tail=0.6)
            SPOKEN.clear()
            await w.send(json.dumps({"type": "text", "text": "台北天氣"}))
            done = await recv_until(w, "done", [], [], 10)
            assert done["text"] == "台北今天晴，最高 28 度。"
            await asyncio.sleep(0.6)
            assert SPOKEN == ["我幫你查一下最新的資料。", "台北今天晴，最高 28 度。"], SPOKEN
            print("  ✓ handed to the agent after a few words: those are said, the result follows as done")
    finally:
        for s, t in ((ot, ot_task), (svc, svc_task)):
            s.should_exit = True
            await t


# ── F. the voices in Settings ────────────────────────────────────────────────

def fake_voice_library(state: dict) -> FastAPI:
    """A voice service with voices that can be added and removed (POST/DELETE /voices)."""
    import base64
    from fastapi import HTTPException
    app = FastAPI()

    def names():
        return sorted(state["voices"]) + ["sample"]

    @app.get("/health")
    async def health():
        return {"ok": True, "tts_sample_rate": RATE, "voices": names()}

    @app.get("/voices")
    async def voices():
        out = {"voices": names()}
        if not state.get("old"):
            out["details"] = [{"name": n, "builtin": n == "sample", "seconds": 7.0,
                               "text": state["voices"].get(n, "希望你以后能够做的比我还好呦。")} for n in names()]
        return out

    @app.post("/voices")
    async def add(request: Request):
        body = await request.json()
        state["got"] = {**body, "wav": base64.b64decode(body["wav"])}
        if body["name"] in names():
            raise HTTPException(409, f"there is already a voice named {body['name']}")
        state["voices"][body["name"]] = body["text"]
        return {"voice": body["name"], "voices": names()}

    @app.delete("/voices/{name}")
    async def remove(name: str):
        if name == "sample":
            raise HTTPException(400, "sample is built in and can't be removed")
        if name not in state["voices"]:
            raise HTTPException(404, f"no voice named {name}")
        del state["voices"][name]
        return {"voices": names()}

    @app.post("/asr")
    async def asr(request: Request):
        state["asr"] = (request.headers.get("content-type"), await request.body())
        return {"text": "來自日本東京生物科技。", "language": "zh"}

    @app.post("/tts")
    async def tts(request: Request):
        state.setdefault("spoken", []).append(await request.json())

        async def chunks():
            for _ in range(3):
                yield b"\x01\x00" * 480
        return StreamingResponse(chunks(), media_type=f"audio/L16; rate={RATE}")
    return app


async def test_voice_library() -> None:
    import tempfile
    import httpx
    import main as M
    M.settings_store._db_path = os.path.join(tempfile.mkdtemp(), "settings.db")
    await M.settings_store.init()
    app = FastAPI()
    app.router.routes.extend(r for r in M.app.routes if getattr(r, "path", "").startswith(
        ("/voice/voices", "/voice/transcribe", "/voice/preview")))
    state: dict = {"voices": {"female1": "來自日本東京生物科技。", "male1": "例如王經理用男聲。"}}
    svc, svc_task, svc_port = await serve(fake_voice_library(state))
    saved = (config.voice_service_url, config.voice_enabled, getattr(config, "voice_name", ""))
    await M.settings_store.update_many({"voice_service_url": f"http://127.0.0.1:{svc_port}", "voice_enabled": "true",
                                        "voice_name": ""})
    await config.reload_from_store(M.settings_store)
    wav = V.wav(b"\x00\x10" * RATE * 5, RATE)                      # 5 s
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://ot") as c:
            d = (await c.get("/voice/voices")).json()
            assert [v["name"] for v in d["voices"]] == ["female1", "male1", "sample"], d
            assert d["selected"] == "" and d["default"] == "female1" and d["can_add"], d
            assert d["voices"][0]["text"] == "來自日本東京生物科技。" and d["voices"][2]["builtin"]
            print("  ✓ the list: each voice's words and length, which one speaks (none chosen: the first)")

            r = await c.post("/voice/preview", json={"voice": "male1", "text": "你好。"})
            assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF", r
            assert state["spoken"][-1] == {"text": "你好。", "voice": "male1"}, state["spoken"]
            await c.post("/voice/preview", json={"voice": "sample"})
            assert state["spoken"][-1]["text"] == M.VOICE_PREVIEW_TEXT
            assert (await c.post("/voice/preview", json={"voice": "nobody"})).status_code == 404
            print("  ✓ ▶ a sentence in the voice asked for, as a WAV; an unknown voice → 404")

            r = await c.post("/voice/transcribe", content=wav, headers={"Content-Type": "audio/wav"})
            assert r.json()["text"] == "來自日本東京生物科技。" and state["asr"] == ("audio/wav", wav), r.json()
            assert (await c.post("/voice/transcribe", content=b"")).status_code == 400
            print("  ✓ the words in a recording, from the voice service's speech recognition")

            form = {"name": "aaa", "text": "來自日本東京生物科技。"}
            files = {"file": ("voice.wav", wav, "audio/wav")}
            r = await c.post("/voice/voices", data=form, files=files)
            assert r.status_code == 400 and "consent" in r.json()["detail"] and "aaa" not in state["voices"], r.text
            r = await c.post("/voice/voices", data={**form, "consent": "true"}, files=files)
            assert r.status_code == 200, r.text
            d = r.json()
            assert state["got"] == {"name": "aaa", "text": "來自日本東京生物科技。", "wav": wav}, "the recording as sent"
            assert [v["name"] for v in d["voices"]] == ["aaa", "female1", "male1", "sample"], d
            assert d["selected"] == d["default"] == "female1" and config.voice_name == "female1", \
                "a new voice that sorts first doesn't take over: the one that spoke keeps speaking"
            r = await c.post("/voice/voices", data={**form, "consent": "true"}, files=files)
            assert r.status_code == 409 and "already" in r.json()["detail"], r.text
            r = await c.post("/voice/voices", data={**form, "consent": "true"},
                             files={"file": ("big.wav", b"0" * (M.MAX_VOICE_UPLOAD + 1), "audio/wav")})
            assert r.status_code == 413, r.text
            print("  ✓ adding: only with the person's word; the service's refusal passed on; too large → 413")

            await M.settings_store.update_many({"voice_name": "aaa"})
            await config.reload_from_store(M.settings_store)
            d = (await c.delete("/voice/voices/aaa")).json()
            assert "aaa" not in state["voices"] and d["selected"] == "" and d["default"] == "female1", d
            assert config.voice_name == "", "the chosen voice is gone: back to the first"
            r = await c.delete("/voice/voices/sample")
            assert r.status_code == 400 and "built in" in r.json()["detail"], r.text
            assert (await c.delete("/voice/voices/nobody")).status_code == 404
            print("  ✓ removing: the choice falls back when its voice goes; built-in ones stay")

            state["old"] = True
            d = (await c.get("/voice/voices")).json()
            assert not d["can_add"] and [v["name"] for v in d["voices"]] == ["female1", "male1", "sample"], d
            state["old"] = False
            await M.settings_store.update_many({"voice_service_url": "http://127.0.0.1:1"})
            await config.reload_from_store(M.settings_store)
            r = await c.get("/voice/voices")
            assert r.status_code == 502 and "didn't answer" in r.json()["detail"], r.text
            print("  ✓ a voice service from before can't add (said so); one that doesn't answer → 502")
    finally:
        svc.should_exit = True
        await svc_task
        config.voice_service_url, config.voice_enabled, config.voice_name = saved


async def main() -> None:
    print("A. read aloud"); test_spoken()
    print("B. a conversation"); await test_conversation()
    print("C. guards"); await test_guards()
    print("D. turning it on"); await test_setup()
    print("E. speaking while the answer is written"); test_next_sentence(); await test_streamed_speech()
    print("F. the voices in Settings"); await test_voice_library()
    print("\nALL VOICE TESTS PASS")


asyncio.run(main())
