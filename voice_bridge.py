"""
Talking with OpenTeddy, like a voice assistant: the web UI's 🎙 mode.

The browser listens (its own echo cancelling and a simple voice detector),
sends each thing the person said as one piece of 16 kHz PCM over /voice/ws,
and plays back what comes down. In between:

  speech → text     the voice service's /asr (SenseVoice), contrib/voice
  text → answer     the same path as POST /tasks with auto_lane: the gate
                    answers small things in one call, hands real work to the
                    agent; the session's knowledge base and memory apply
  answer → speech   the voice service's /tts (CosyVoice), sentence by
                    sentence, streamed, so the first words play while the
                    rest is still being made

Work that takes longer than WORK_ACK_S gets a spoken "I'm on it" and the
conversation carries on; the result is spoken when it's done. Speaking
while OpenTeddy talks stops it (the browser sends "interrupt").

Off unless voice_service_url is set (Settings). The voice service runs on a
machine of its own (contrib/voice/README.md).

Protocol (JSON text frames, audio as binary frames):
  → {"type":"hello","session_id"}        ← {"type":"ready","sample_rate",…}
  → <PCM16 mono 16 kHz: one utterance>   ← {"type":"transcript","text"}
  → {"type":"text","text"}  (typed)      ← {"type":"reply","text","task_id"}
  → {"type":"interrupt"}                 ← <PCM16 at sample_rate>… {"type":"audio_end"}
                                         ← {"type":"state","value": listening|thinking|speaking}
                                         ← {"type":"working","task_id","text"} … {"type":"done",…}
                                         ← {"type":"error","text"}
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx

from config import config

logger = logging.getLogger(__name__)

ASR_RATE = 16000
MAX_UTTERANCE_BYTES = ASR_RATE * 2 * 45          # 45 s of speech
WORK_ACK_S = 8.0                                  # longer than this: say "I'm on it" and carry on
SPOKEN_MAX_CHARS = 220                            # a work result read aloud: the gist, the rest is on screen

ACK = "好，我正在處理，完成後告訴你。"
DONE_MORE = "詳細內容我放在對話裡了。"
NOT_UNDERSTOOD = "抱歉，我沒聽清楚，可以再說一次嗎？"


def configured() -> bool:
    return bool((getattr(config, "voice_service_url", "") or "").strip())


def _base() -> str:
    return (getattr(config, "voice_service_url", "") or "").strip().rstrip("/")


# ── The voice service ────────────────────────────────────────────────────────

async def health(timeout: float = 5.0) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.get(f"{_base()}/health")
        r.raise_for_status()
        return r.json()


async def transcribe(pcm: bytes, client: Optional[httpx.AsyncClient] = None) -> Dict[str, Any]:
    own = client is None
    c = client or httpx.AsyncClient(timeout=30.0)
    try:
        r = await c.post(f"{_base()}/asr", content=pcm, headers={"Content-Type": f"audio/L16; rate={ASR_RATE}"})
        r.raise_for_status()
        return r.json()
    finally:
        if own:
            await c.aclose()


async def speak(text: str, client: httpx.AsyncClient) -> AsyncIterator[bytes]:
    """PCM16 chunks as the voice service makes them."""
    body = {"text": text, "voice": getattr(config, "voice_name", "") or ""}
    async with client.stream("POST", f"{_base()}/tts", json=body) as r:
        r.raise_for_status()
        carry = b""
        async for chunk in r.aiter_bytes():
            chunk = carry + chunk
            if len(chunk) % 2:                    # a sample split across reads
                carry, chunk = chunk[-1:], chunk[:-1]
            else:
                carry = b""
            if chunk:
                yield chunk


# ── Text that is read aloud ──────────────────────────────────────────────────

_CITE = re.compile(r"[（(](?:[^（）()]*?(?:知識庫|來源|參考|資料來源|source)[^（）()]*?)[）)]", re.I)
_URL = re.compile(r"https?://\S+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")


def spoken(text: str, limit: int = 0) -> str:
    """What a TTS engine should say: no markdown, tables, links, emoji or
    「（知識庫：faq.csv 第 6 列）」 — those stay in the transcript on screen.
    With `limit`, the first sentences up to about that many characters."""
    lines: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or set(line) <= set("-=*_|:` ") or line.count("|") >= 2 or line.startswith("```"):
            continue
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"^(?:[-*•>]+|\d+[.)、])\s*", "", line)
        line = _CITE.sub("", line).strip()
        if line:
            lines.append(line)
    t = ""
    for line in lines:                         # lines become one flow: a comma only where nothing ends them
        t += ("" if not t or re.search(r"[。！？!?；;：:，,、]$", t) else "，") + line
    t = _CITE.sub("", t)
    t = _URL.sub("連結", t)
    t = re.sub(r"[*_`#~]+", "", t)
    t = _EMOJI.sub("", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([，。！？、；：])", r"\1", t)
    t = re.sub(r"，+([。！？!?，])", r"\1", t).strip(" ，")
    if limit and len(t) > limit:
        cut = [m.end() for m in re.finditer(r"[。！？!?]", t) if m.end() <= limit]
        t = (t[:cut[-1]] if cut else t[:limit].rstrip("，、 ") + "。") + DONE_MORE
    return t


def sentences(text: str, max_chars: int = 120) -> List[str]:
    """Pieces to synthesise one after another: sentence ends, then commas
    for long ones; a short piece rides with the next."""
    parts = [p.strip() for p in re.split(r"(?<=[。！？!?；;])", text or "") if p.strip()]
    out: List[str] = []
    for p in parts:
        while len(p) > max_chars:
            cut = max(p.rfind("，", 0, max_chars), p.rfind(",", 0, max_chars), p.rfind("、", 0, max_chars))
            cut = cut + 1 if cut > 10 else max_chars
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            if out and len(out[-1]) < 8:
                out[-1] += p
            else:
                out.append(p)
    return out


# ── One conversation over /voice/ws ──────────────────────────────────────────

StartTask = Callable[..., Awaitable[Tuple[str, Optional[str], Optional["asyncio.Task"], Optional[str]]]]
AwaitTask = Callable[[str, Any], Awaitable[Tuple[Any, str]]]


class Conversation:
    """The person's turns, OpenTeddy's answers, spoken. `start_task` and
    `await_task` are main.py's, passed in so this file needs no import of it."""

    def __init__(self, ws: Any, start_task: StartTask, await_task: AwaitTask) -> None:
        self.ws, self.start_task, self.await_task = ws, start_task, await_task
        self.session_id = ""
        self.sample_rate = 24000
        self.turn = 0                              # the latest thing the person said
        self.speaking: Optional[asyncio.Task] = None
        self.turn_task: Optional[asyncio.Task] = None
        self.background: List[asyncio.Task] = []
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
        self._send_lock = asyncio.Lock()

    async def send(self, obj: Any) -> None:
        async with self._send_lock:
            if isinstance(obj, (bytes, bytearray)):
                await self.ws.send_bytes(bytes(obj))
            else:
                await self.ws.send_text(json.dumps(obj, ensure_ascii=False))

    async def state(self, value: str) -> None:
        await self.send({"type": "state", "value": value})

    async def run(self) -> None:
        try:
            first = await asyncio.wait_for(self.ws.receive(), 10.0)
            hello = json.loads(first.get("text") or "{}")
        except Exception:  # noqa: BLE001
            hello = {}
        self.session_id = str(hello.get("session_id") or "")
        if not configured():
            await self.send({"type": "error", "code": "not_configured",
                             "text": "語音服務還沒設定（設定 → 語音服務網址）。"})
            return
        try:
            h = await health()
            self.sample_rate = int(h.get("tts_sample_rate") or self.sample_rate)
        except Exception as exc:  # noqa: BLE001
            await self.send({"type": "error", "code": "unreachable", "text": f"連不到語音服務（{_base()}）：{exc}"})
            return
        await self.send({"type": "ready", "sample_rate": self.sample_rate, "asr_rate": ASR_RATE,
                         "voice": getattr(config, "voice_name", "") or (h.get("voices") or [""])[0],
                         "asr": h.get("asr"), "tts": h.get("tts")})
        await self.state("listening")
        try:
            while True:
                msg = await self.ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    await self.on_utterance(msg["bytes"])
                elif msg.get("text"):
                    data = json.loads(msg["text"])
                    kind = data.get("type")
                    if kind == "interrupt":
                        await self.stop_speaking()
                        await self.state("listening")
                    elif kind == "text" and str(data.get("text") or "").strip():
                        await self.new_turn(str(data["text"]).strip(), heard=None)
        except Exception as exc:  # noqa: BLE001 — the browser went away
            logger.info("voice: conversation ended (%s)", type(exc).__name__)
        finally:
            await self.stop_speaking()
            for t in self.background + ([self.turn_task] if self.turn_task else []):
                t.cancel()
            await self.http.aclose()

    async def stop_speaking(self) -> None:
        if self.speaking and not self.speaking.done():
            self.speaking.cancel()
            try:
                await self.speaking
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self.speaking = None

    async def on_utterance(self, pcm: bytes) -> None:
        if len(pcm) > MAX_UTTERANCE_BYTES:
            pcm = pcm[:MAX_UTTERANCE_BYTES]
        await self.stop_speaking()
        await self.state("thinking")
        t0 = time.monotonic()
        try:
            heard = await transcribe(pcm, self.http)
        except Exception as exc:  # noqa: BLE001
            await self.send({"type": "error", "text": f"語音辨識失敗：{exc}"})
            await self.state("listening")
            return
        text = (heard.get("text") or "").strip()
        if not text:
            await self.state("listening")
            return
        await self.new_turn(text, heard={**heard, "ms_total": int((time.monotonic() - t0) * 1000)})

    async def new_turn(self, text: str, heard: Optional[Dict[str, Any]]) -> None:
        self.turn += 1
        turn = self.turn
        await self.send({"type": "transcript", "text": text, **({"ms": heard.get("ms")} if heard else {})})
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()                # the person said something new: drop the old wait
        self.turn_task = asyncio.create_task(self.answer(text, turn))

    async def answer(self, text: str, turn: int) -> None:
        await self.state("thinking")
        try:
            task_id, _sid, job, shortcut = await self.start_task(
                goal=text, context={"voice": True}, session_id=self.session_id or None, auto_lane=True)
        except Exception as exc:  # noqa: BLE001
            detail = getattr(exc, "detail", None) or str(exc)
            await self.send({"type": "error", "text": f"沒辦法開始：{detail}"})
            await self.state("listening")
            return
        if shortcut is not None:
            await self.reply(shortcut, task_id, turn)
            return
        waiter = asyncio.ensure_future(self.await_task(task_id, job))
        try:
            done, _ = await asyncio.wait({waiter}, timeout=WORK_ACK_S)
        except asyncio.CancelledError:
            # The person said something new before this was answered: it isn't
            # spoken now, but real work still gets its "done" later.
            self.background.append(asyncio.create_task(self.when_done(waiter, task_id, quiet_if_quick=True)))
            raise
        if done:
            _status, message = waiter.result()
            await self.reply(message, task_id, turn)
            return
        # Real work: say so, keep listening, speak the result when it's in.
        await self.send({"type": "working", "task_id": task_id, "text": ACK})
        await self.say(ACK, turn)
        self.background.append(asyncio.create_task(self.when_done(waiter, task_id)))

    async def when_done(self, waiter: "asyncio.Future", task_id: str, quiet_if_quick: bool = False) -> None:
        started = time.monotonic()
        try:
            _status, message = await waiter
        except asyncio.CancelledError:
            return
        except Exception as exc:  # noqa: BLE001
            message = f"那件事沒有完成：{exc}"
        if quiet_if_quick and time.monotonic() - started < WORK_ACK_S:
            return                                 # a quick answer the person talked over: on screen only
        await self.send({"type": "done", "task_id": task_id, "text": message})
        while self.speaking and not self.speaking.done():   # don't talk over the current answer
            await asyncio.sleep(0.3)
        await self.say(spoken(message, SPOKEN_MAX_CHARS), self.turn)

    async def reply(self, message: str, task_id: str, turn: int) -> None:
        if turn != self.turn:                      # the person has moved on
            return
        await self.send({"type": "reply", "text": message or "", "task_id": task_id})
        await self.say(spoken(message or NOT_UNDERSTOOD, SPOKEN_MAX_CHARS), turn)

    async def say(self, text: str, turn: int) -> None:
        if not text or turn != self.turn:
            await self.state("listening")
            return
        await self.stop_speaking()
        self.speaking = asyncio.create_task(self._stream(text))
        try:
            await self.speaking
        except asyncio.CancelledError:
            pass

    async def _stream(self, text: str) -> None:
        await self.state("speaking")
        try:
            for piece in sentences(text):
                async for chunk in speak(piece, self.http):
                    await self.send(chunk)
        except asyncio.CancelledError:
            await self.send({"type": "audio_end", "interrupted": True})
            raise
        except Exception as exc:  # noqa: BLE001
            await self.send({"type": "error", "text": f"語音合成失敗：{exc}"})
        await self.send({"type": "audio_end"})
        await self.state("listening")
