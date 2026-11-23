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
# What OpenTeddy says by itself follows the language the person spoke.
EN = {ACK: "Okay, I'm working on it. I'll tell you when it's done.",
      DONE_MORE: "The details are in the chat.",
      NOT_UNDERSTOOD: "Sorry, I didn't catch that. Could you say it again?"}


def english(text: str) -> bool:
    """Said in English: Latin words and no Chinese."""
    return not _CJK.search(text or "") and len(re.findall(r"[A-Za-z]{2,}", text or "")) >= 1


def phrase(zh: str, in_english: bool) -> str:
    return EN.get(zh, zh) if in_english else zh


def configured() -> bool:
    return bool((getattr(config, "voice_service_url", "") or "").strip())


def enabled() -> bool:
    """Turned on by the person (Settings, or the 🎙 prompt) — off by default."""
    return bool(getattr(config, "voice_enabled", False))


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


async def speak(text: str, client: httpx.AsyncClient, voice: str = "") -> AsyncIterator[bytes]:
    """PCM16 chunks as the voice service makes them (`voice`: one of its
    voices; "" = the one set in Settings)."""
    body = {"text": text, "voice": voice or getattr(config, "voice_name", "") or ""}
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

_CJK = re.compile(r"[㐀-鿿豈-﫿　-〿＀-￯]")
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
        if t and not re.search(r"[。！？!?；;：:，,、.]$", t):
            t += "，" if _CJK.search(t[-1]) or _CJK.search(line[0]) else ","
        t = _join(t, line)
    t = _CITE.sub("", t)
    t = _URL.sub("連結" if _CJK.search(t) else "a link", t)
    t = re.sub(r"[*_`#~]+", "", t)
    t = _EMOJI.sub("", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([，。！？、；：])", r"\1", t)
    t = re.sub(r"，+([。！？!?，])", r"\1", t).strip(" ，")
    if limit and len(t) > limit:
        en = english(t)
        ends = r"[。！？!?]|\.(?=\s|$)" if en else r"[。！？!?]"
        cut = [m.end() for m in re.finditer(ends, t) if m.end() <= limit]
        t = _join(t[:cut[-1]] if cut else t[:limit].rstrip("，、, ") + ("." if en else "。"), phrase(DONE_MORE, en))
    return t


def sentences(text: str, max_chars: int = 120) -> List[str]:
    """Pieces to synthesise one after another: sentence ends, then commas
    for long ones; a short piece rides with the next.

    Not split smaller for a quicker first sound: CosyVoice streams its first
    chunk after a fixed number of speech tokens however long the piece is,
    and every call pays that wait again — cutting the first clause off made
    the first sound later (4.1 s vs 2.9 s, demo2 11/20)."""
    return [p for p in _pieces(text, max_chars) if p]


_SENTENCE_END = re.compile(r"[。！？!?；;\n]|\.(?=\s)")
MIN_SENTENCE = 6                  # 「好。」 alone rides with the next sentence
LONG_RUN = 60                     # this long with no sentence end: cut at the last comma


def next_sentence(buf: str, final: bool = False) -> Optional[Tuple[str, str]]:
    """(the first complete sentence of text still being written, what follows),
    or None until one is complete — with `final`, the rest is the last one.

    Speaking starts with the first sentence, so it should end as soon as the
    sentence does; but not on 「好。」 alone (every call to CosyVoice pays its
    first-chunk wait, see sentences()), nor inside "3.5"."""
    for m in _SENTENCE_END.finditer(buf):
        if len(buf[:m.end()].strip()) >= MIN_SENTENCE:
            return buf[:m.end()], buf[m.end():]
    if len(buf) > LONG_RUN:
        cut = max(buf.rfind("，", 0, LONG_RUN), buf.rfind(",", 0, LONG_RUN), buf.rfind("、", 0, LONG_RUN))
        if cut > 10:
            return buf[:cut + 1], buf[cut + 1:]
    if final and buf.strip():
        return buf, ""
    return None


def _join(a: str, b: str) -> str:
    """Two pieces of speech side by side: nothing between Chinese, a space in English."""
    if not a or not b:
        return a + b
    return a + ("" if _CJK.search(a[-1]) or _CJK.search(b[0]) else " ") + b


def _pieces(text: str, max_chars: int) -> List[str]:
    # Sentence ends: Chinese marks, and an English ". " before the next sentence
    # (not "3.5" or "e.g." inside one).
    parts = [p.strip() for p in re.split(r"(?<=[。！？!?；;])|(?<=\.)\s+(?=[A-Z0-9\"'(])", text or "") if p and p.strip()]
    out: List[str] = []
    for p in parts:
        while len(p) > max_chars:
            cut = max(p.rfind("，", 0, max_chars), p.rfind(",", 0, max_chars), p.rfind("、", 0, max_chars))
            if cut <= 10:
                cut = p.rfind(" ", 0, max_chars)               # English without a comma: a word boundary
            cut = cut + 1 if cut > 10 else max_chars
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            if out and len(out[-1]) < 8:
                out[-1] = _join(out[-1], p)
            else:
                out.append(p)
    return out


# ── One conversation over /voice/ws ──────────────────────────────────────────

StartTask = Callable[..., Awaitable[Tuple[str, Optional[str], Optional["asyncio.Task"], Optional[str]]]]
AwaitTask = Callable[[str, Any], Awaitable[Tuple[Any, str]]]


class Conversation:
    """The person's turns, OpenTeddy's answers, spoken. `start_task` and
    `await_task` are main.py's, passed in so this file needs no import of it."""

    def __init__(self, ws: Any, start_task: StartTask, await_task: AwaitTask,
                 bind_session: Optional[Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]] = None,
                 on_control: Optional[Callable[[Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]] = None) -> None:
        """For /v1/realtime (public_api.py):
        `bind_session(session_id, hello)`: the session to talk in, from the one
        the hello named ("" = none) → {"session_id", …more for "ready"}; raises
        ValueError to refuse (another key's session, an unknown dataset).
        `on_control(message)`: a JSON message of another type (e.g. "datasets")
        → the answer to send, or None; ValueError → an error message."""
        self.ws, self.start_task, self.await_task = ws, start_task, await_task
        self.bind_session, self.on_control = bind_session, on_control
        self.ready_extra: Dict[str, Any] = {}
        self.session_id = ""
        self.sample_rate = 24000
        self.turn = 0                              # the latest thing the person said
        self.en = False                            # ... and whether they said it in English
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
        if self.bind_session is not None:
            try:
                bound = await self.bind_session(self.session_id, hello)
            except ValueError as exc:
                await self.send({"type": "error", "code": getattr(exc, "code", "") or "bad_session", "text": str(exc)})
                return
            self.session_id = str(bound.get("session_id") or self.session_id)
            self.ready_extra = {k: v for k, v in bound.items() if k != "session_id"}
        if not enabled():
            await self.send({"type": "error", "code": "not_enabled",
                             "text": "語音對話還沒開啟（設定 → 語音對話）。"})
            return
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
        if not h.get("ok", True):
            await self.send({"type": "error", "code": "loading",
                             "text": f"語音服務還在準備（{h.get('stage') or '啟動中'}）。"})
            return
        await self.send({"type": "ready", "sample_rate": self.sample_rate, "asr_rate": ASR_RATE,
                         "voice": getattr(config, "voice_name", "") or (h.get("voices") or [""])[0],
                         "asr": h.get("asr"), "tts": h.get("tts"),
                         **({"session_id": self.session_id} if self.session_id else {}), **self.ready_extra})
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
                    elif self.on_control is not None:
                        try:
                            answer = await self.on_control(data)
                        except ValueError as exc:
                            answer = {"type": "error", "code": getattr(exc, "code", "") or "bad_request",
                                      "text": str(exc)}
                        if answer:
                            await self.send(answer)
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
        if not re.search(r"\w", text):             # nothing, or a noise heard as 「。」 / "."
            await self.state("listening")
            return
        await self.new_turn(text, heard={**heard, "ms_total": int((time.monotonic() - t0) * 1000)})

    async def new_turn(self, text: str, heard: Optional[Dict[str, Any]]) -> None:
        self.en = english(text)                    # OpenTeddy's own phrases follow the person's language
        self.turn += 1
        turn = self.turn
        await self.send({"type": "transcript", "text": text, **({"ms": heard.get("ms")} if heard else {})})
        if self.turn_task and not self.turn_task.done():
            self.turn_task.cancel()                # the person said something new: drop the old wait
        self.turn_task = asyncio.create_task(self.answer(text, turn))

    async def answer(self, text: str, turn: int) -> None:
        await self.state("thinking")
        # Listen for the answer's words before the task starts (answer_stream.py):
        # the first sentence is spoken while the model is still writing the rest.
        import answer_stream
        import uuid
        tid = str(uuid.uuid4())
        words = answer_stream.listen(tid)
        try:
            await self._answer(text, turn, tid, words)
        finally:
            answer_stream.close(tid, words)

    async def _answer(self, text: str, turn: int, tid: str, words: "asyncio.Queue") -> None:
        import answer_stream
        started = time.monotonic()
        try:
            task_id, _sid, job, shortcut = await self.start_task(
                goal=text, context={"voice": True}, session_id=self.session_id or None, auto_lane=True,
                task_id=tid)
        except Exception as exc:  # noqa: BLE001
            detail = getattr(exc, "detail", None) or str(exc)
            await self.send({"type": "error", "text": f"沒辦法開始：{detail}"})
            await self.state("listening")
            return
        if shortcut is not None:
            await self.reply(shortcut, task_id, turn)
            return
        waiter = asyncio.ensure_future(self.await_task(task_id, job))
        getter = asyncio.ensure_future(words.get())
        try:
            # The first of: the answer's first words, the task done (a shortcut,
            # a hand-off finished quickly), or the work-acknowledgement deadline.
            done, _ = await asyncio.wait({waiter, getter}, timeout=WORK_ACK_S,
                                         return_when=asyncio.FIRST_COMPLETED)
            first = getter.result() if getter in done else None
            if getter in done and first is answer_stream.END and waiter not in done:
                # The model handed the turn to the agent: no words; wait as before.
                done, _ = await asyncio.wait({waiter}, timeout=max(0.0, WORK_ACK_S - (time.monotonic() - started)))
                first = None
        except asyncio.CancelledError:
            # The person said something new before this was answered: it isn't
            # spoken now, but real work still gets its "done" later.
            getter.cancel()
            self.background.append(asyncio.create_task(self.when_done(waiter, task_id, quiet_if_quick=True)))
            raise
        if not getter.done():
            getter.cancel()
        if first is not None and first is not answer_stream.END:
            await self.say_streamed(first, words, task_id, turn, waiter)
            return
        if waiter in done:
            _status, message = waiter.result()
            await self.reply(message, task_id, turn)
            return
        # Real work: say so, keep listening, speak the result when it's in.
        ack = phrase(ACK, self.en)
        await self.send({"type": "working", "task_id": task_id, "text": ack})
        await self.say(ack, turn)
        self.background.append(asyncio.create_task(self.when_done(waiter, task_id)))

    async def say_streamed(self, first: str, words: "asyncio.Queue", task_id: str, turn: int,
                           waiter: "asyncio.Future") -> None:
        """Speak the answer sentence by sentence as the model writes it; the
        text goes out as `reply` once it is complete. A turn that went on to
        the agent after a few words gets its result later, as `done`."""
        if turn != self.turn:
            return
        await self.stop_speaking()
        written: List[str] = [first]
        self.speaking = asyncio.create_task(self._stream_words(first, words, written, task_id, turn))
        try:
            await self.speaking
        except asyncio.CancelledError:
            pass
        self.background.append(asyncio.create_task(self._after_streamed(waiter, task_id, "".join(written))))

    async def _after_streamed(self, waiter: "asyncio.Future", task_id: str, streamed: str) -> None:
        """The task's own result, once in: nothing more when it is the words
        already spoken; otherwise (handed to the agent after a few words) the
        result, as a work result is."""
        try:
            _status, message = await waiter
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            return
        m, s = (message or "").strip(), streamed.strip()
        if not m or m == s or m.startswith(s) and len(m) - len(s) < 4:
            return
        await self.send({"type": "done", "task_id": task_id, "text": message})
        while self.speaking and not self.speaking.done():
            await asyncio.sleep(0.3)
        await self.say(spoken(message, SPOKEN_MAX_CHARS), self.turn)

    async def _stream_words(self, first: str, words: "asyncio.Queue", written: List[str],
                            task_id: str, turn: int) -> None:
        import answer_stream
        buf, ended, said, sent_reply = first, False, 0, False
        speaking = False
        try:
            while True:
                cut = next_sentence(buf, final=ended)
                if cut is None:
                    if ended:
                        break
                    got = await words.get()
                    if got is answer_stream.END:
                        ended = True
                        if not sent_reply:
                            sent_reply = True
                            await self.send({"type": "reply", "text": "".join(written), "task_id": task_id})
                    else:
                        written.append(got)
                        buf += got
                    continue
                sentence, buf = cut
                line = spoken(sentence)
                if not line:
                    continue
                if said + len(line) > SPOKEN_MAX_CHARS and said:
                    line = phrase(DONE_MORE, self.en)          # the gist was said; the rest is on screen
                    buf, ended = "", True
                said += len(line)
                if not speaking:
                    speaking = True
                    await self.state("speaking")
                async for chunk in speak(line, self.http):
                    await self.send(chunk)
            if not sent_reply:
                # Stopped early (the gist said): the screen still gets the whole answer.
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    try:
                        got = await asyncio.wait_for(words.get(), max(0.1, deadline - time.monotonic()))
                    except asyncio.TimeoutError:
                        break
                    if got is answer_stream.END:
                        break
                    written.append(got)
                await self.send({"type": "reply", "text": "".join(written), "task_id": task_id})
        except asyncio.CancelledError:
            await self.send({"type": "audio_end", "interrupted": True})
            raise
        except Exception as exc:  # noqa: BLE001
            await self.send({"type": "error", "text": f"語音合成失敗：{exc}"})
        await self.send({"type": "audio_end"})
        await self.state("listening")

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
        await self.say(spoken(message or phrase(NOT_UNDERSTOOD, self.en), SPOKEN_MAX_CHARS), turn)

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
