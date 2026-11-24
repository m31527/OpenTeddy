"""
OpenTeddy as a LiveKit voice agent.

LiveKit carries the audio: WebRTC to the app, echo cancellation, jitter
buffering, turn-taking and barge-in. OpenTeddy listens, thinks and speaks
through the three streaming interfaces of its /v1 API (docs/api.md):

  STT  WS   /v1/audio/stt-stream      SenseVoice: partials while the person
                                       talks, the final as soon as LiveKit's
                                       VAD says they stopped (a flush)
  LLM  POST /v1/chat/completions      stream: OpenTeddy's chat lane word by
                                       word, with its knowledge base, memory,
                                       session and datasets
  TTS  POST /v1/audio/speech (pcm)    CosyVoice, sentence by sentence while
                                       the LLM is still writing

Run (README.md):

    OPENTEDDY_URL=http://192.168.100.1:8000/v1
    LIVEKIT_URL=ws://127.0.0.1:7880  LIVEKIT_API_KEY=…  LIVEKIT_API_SECRET=…
    python agent.py start

The agent goes where it is sent (explicit dispatch, AGENT_NAME, default
"openteddy"). Usually OpenTeddy sends it: an app asks POST /v1/livekit/token
with its key, OpenTeddy makes the conversation and the room, and dispatches
the agent with {"session_id", "grant"} — a grant that acts as the app's key
in that one conversation, so the call is the app's own, with its agent,
instructions and datasets. One room = one OpenTeddy conversation.

A room made some other way (your own server, with the LiveKit secret, and a
dispatch without a grant) uses OPENTEDDY_KEY, and its metadata (JSON), or the
caller's attributes of the same names, sets the conversation up:
    {"datasets": ["客服 FAQ"], "agent": "品牌客戶・王經理", "instructions": "…"}
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from typing import Any, List, Optional

import httpx
import websockets
from livekit.agents import (DEFAULT_API_CONNECT_OPTIONS, Agent, AgentSession, APIConnectOptions, JobContext,
                            JobProcess, WorkerOptions, cli, llm, stt, tts)
from livekit.agents import tokenize
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.plugins import silero

logger = logging.getLogger("openteddy-agent")

OT_URL = os.environ.get("OPENTEDDY_URL", "http://127.0.0.1:8000/v1").rstrip("/")
OT_KEY = os.environ.get("OPENTEDDY_KEY", "")
TTS_RATE = int(os.environ.get("OPENTEDDY_TTS_RATE", "24000"))       # CosyVoice 3
LANG = os.environ.get("OPENTEDDY_LANGUAGE", "zh")

# Turn-taking, tuned for a quick answer (LiveKit's defaults wait ~1 s after
# the person stops: 0.55 s of silence + 0.5 s endpointing delay).
VAD_SILENCE_S = float(os.environ.get("VAD_SILENCE_S", "0.3"))
ENDPOINT_MIN_S = float(os.environ.get("ENDPOINT_MIN_S", "0.05"))


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ── STT: WS /v1/audio/stt-stream ─────────────────────────────────────────────

class OpenTeddySTT(stt.STT):
    """SenseVoice through OpenTeddy. manual_flush: at the end of speech LiveKit
    flushes the stream, and OpenTeddy answers with the final (~40 ms)."""

    def __init__(self, token: str) -> None:
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True, manual_flush=True))
        self.token = token                          # a key, or the room's grant

    @property
    def model(self) -> str:
        return "sensevoice"

    @property
    def provider(self) -> str:
        return "openteddy"

    async def _recognize_impl(self, buffer: Any, *, language: NotGivenOr[str] = NOT_GIVEN,
                              conn_options: APIConnectOptions) -> stt.SpeechEvent:
        from livekit import rtc
        frame = rtc.combine_audio_frames(buffer) if isinstance(buffer, list) else buffer
        async with httpx.AsyncClient(timeout=conn_options.timeout) as c:
            r = await c.post(f"{OT_URL}/audio/transcriptions", headers=_headers(self.token),
                             files={"file": ("speech.wav", frame.to_wav_bytes(), "audio/wav")})
            r.raise_for_status()
        return _event(stt.SpeechEventType.FINAL_TRANSCRIPT, r.json().get("text", ""))

    def stream(self, *, language: NotGivenOr[str] = NOT_GIVEN,
               conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> "OpenTeddySpeechStream":
        return OpenTeddySpeechStream(stt=self, conn_options=conn_options)


def _event(kind: stt.SpeechEventType, text: str) -> stt.SpeechEvent:
    return stt.SpeechEvent(type=kind, alternatives=[stt.SpeechData(language=LANG, text=text)])


class OpenTeddySpeechStream(stt.SpeechStream):
    def __init__(self, *, stt: OpenTeddySTT, conn_options: APIConnectOptions) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=16000)   # frames arrive as 16 kHz

    async def _run(self) -> None:
        url = OT_URL.replace("http", "ws", 1) + "/audio/stt-stream"
        token = self._stt.token                                         # type: ignore[attr-defined]
        async with websockets.connect(url, additional_headers=_headers(token), max_size=None) as ws:
            await ws.send(json.dumps({"type": "config", "endpoint_ms": 0, "interim": True}))

            async def send() -> None:
                async for data in self._input_ch:
                    if isinstance(data, self._FlushSentinel):
                        logger.info("timeline %.3f stt flush", time.time())
                        await ws.send(json.dumps({"type": "flush"}))
                    else:
                        await ws.send(bytes(data.data))                      # PCM16 mono 16 kHz
                await ws.close()

            async def receive() -> None:
                async for message in ws:
                    d = json.loads(message)
                    if d.get("type") == "partial" and d.get("text"):
                        self._event_ch.send_nowait(_event(stt.SpeechEventType.INTERIM_TRANSCRIPT, d["text"]))
                    elif d.get("type") == "final":
                        logger.info("timeline %.3f stt final (%d chars)", time.time(), len(d.get("text", "")))
                        # Sent even when empty: LiveKit waits for a final after its flush.
                        self._event_ch.send_nowait(_event(stt.SpeechEventType.FINAL_TRANSCRIPT, d.get("text", "")))
                    elif d.get("type") == "error":
                        logger.warning("stt-stream: %s", d)

            await asyncio.gather(send(), receive())


# ── LLM: POST /v1/chat/completions, streamed ─────────────────────────────────

class OpenTeddyLLM(llm.LLM):
    """OpenTeddy keeps the conversation (its session: memory, knowledge base,
    datasets), so only the newest user message is sent each turn."""

    def __init__(self, token: str, session_id: Optional[str] = None, datasets: Optional[List[str]] = None,
                 agent: Optional[str] = None, instructions: Optional[str] = None) -> None:
        super().__init__()
        self.token = token
        self.session_id: Optional[str] = session_id         # set up by OpenTeddy already, or made on the first turn
        self.datasets, self.agent, self.instructions = datasets, agent, instructions

    @property
    def model(self) -> str:
        return "openteddy"

    @property
    def provider(self) -> str:
        return "openteddy"

    def chat(self, *, chat_ctx: llm.ChatContext, tools: Optional[list] = None,
             conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
             parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN, tool_choice: Any = NOT_GIVEN,
             extra_kwargs: NotGivenOr[dict] = NOT_GIVEN) -> "OpenTeddyLLMStream":
        return OpenTeddyLLMStream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class OpenTeddyLLMStream(llm.LLMStream):
    async def _run(self) -> None:
        ot: OpenTeddyLLM = self._llm                                    # type: ignore[assignment]
        said = next((m.text_content for m in reversed(self._chat_ctx.items)
                     if getattr(m, "type", "") == "message" and getattr(m, "role", "") == "user"
                     and (m.text_content or "").strip()), "")
        if not said:
            return
        body: dict = {"model": "openteddy", "stream": True, "voice": True,
                      "messages": [{"role": "user", "content": said}]}
        if ot.session_id:
            body["session_id"] = ot.session_id
        else:                                                           # the session keeps all three
            if ot.datasets is not None:
                body["datasets"] = ot.datasets
            if ot.agent:
                body["agent"] = ot.agent
            if ot.instructions:
                body["instructions"] = ot.instructions
        rid = str(uuid.uuid4())
        logger.info("timeline %.3f llm request (%d chars)", time.time(), len(said))
        first = True
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=5.0)) as c:
            async with c.stream("POST", f"{OT_URL}/chat/completions", headers=_headers(ot.token), json=body) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    d = json.loads(line[6:])
                    ot.session_id = ot.session_id or d.get("session_id")
                    for ch in d.get("choices") or []:
                        piece = (ch.get("delta") or {}).get("content")
                        if piece:
                            if first:
                                first = False
                                logger.info("timeline %.3f llm first piece", time.time())
                            self._event_ch.send_nowait(
                                llm.ChatChunk(id=rid, delta=llm.ChoiceDelta(role="assistant", content=piece)))


# ── TTS: POST /v1/audio/speech, pcm ──────────────────────────────────────────

class OpenTeddyTTS(tts.TTS):
    """One sentence per call; StreamAdapter feeds the sentences as the LLM writes them."""

    def __init__(self, token: str, voice: str = "") -> None:
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=TTS_RATE, num_channels=1)
        self.token = token
        self.voice = voice                                              # "" = the one set in OpenTeddy

    @property
    def model(self) -> str:
        return "cosyvoice"

    @property
    def provider(self) -> str:
        return "openteddy"

    def synthesize(self, text: str, *,
                   conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> "OpenTeddyChunkedStream":
        return OpenTeddyChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class OpenTeddyChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        # 20 ms frames: the default 200 ms held the first sound back by that much.
        output_emitter.initialize(request_id=str(uuid.uuid4()), sample_rate=TTS_RATE, num_channels=1,
                                  mime_type="audio/pcm", frame_size_ms=20)
        logger.info("timeline %.3f tts request (%d chars)", time.time(), len(self.input_text))
        first = True
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0)) as c:
            token, voice = self._tts.token, self._tts.voice             # type: ignore[attr-defined]
            async with c.stream("POST", f"{OT_URL}/audio/speech", headers=_headers(token),
                                json={"model": "cosyvoice", "input": self.input_text, "response_format": "pcm",
                                      **({"voice": voice} if voice else {})}) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes():
                    if first:
                        first = False
                        logger.info("timeline %.3f tts first audio", time.time())
                    output_emitter.push(chunk)
        output_emitter.flush()


# ── What goes to TTS, and when ───────────────────────────────────────────────

_CLAUSE = re.compile(r"[，,、。！？!?；;：:\n]")
_SENTENCE = re.compile(r"(?<=[。！？!?；;\n])")
FIRST_CLAUSE_MIN = 6


class ClauseFirstTokenizer(tokenize.SentenceTokenizer):
    """Pieces for TTS while the answer is still being written: the first one
    ends at the first comma or stop past 6 characters, the rest at sentence
    ends. Speaking starts with 「您好，我是 OpenTeddy，」 instead of waiting for
    a 50-character first sentence (0.67 s on spark B, 11/24); synthesis runs
    at about half real time, so the next piece is ready before this one has
    been said. LiveKit's own tokenizers wait for 20 characters and for 10
    more past a sentence end."""

    def tokenize(self, text: str, *, language: Optional[str] = None) -> List[str]:
        return self._splitter()(text)

    def stream(self, *, language: Optional[str] = None) -> tokenize.SentenceStream:
        return tokenize.BufferedSentenceStream(tokenizer=self._splitter(), min_token_len=FIRST_CLAUSE_MIN,
                                               min_ctx_len=1)

    @staticmethod
    def _splitter():
        state = {"first": True}

        def split(text: str) -> List[str]:
            head: List[str] = []
            rest = text
            if state["first"]:
                cut = next((m.end() for m in _CLAUSE.finditer(text)
                            if len(text[:m.end()].strip()) >= FIRST_CLAUSE_MIN), None)
                if cut is None:
                    return [text]
                head, rest = [text[:cut]], text[cut:]
            tokens = head + [p for p in _SENTENCE.split(rest) if p.strip()]
            if state["first"] and len(tokens) >= 2:
                state["first"] = False                 # the first piece goes out now
            return tokens
        return split


# ── The worker ───────────────────────────────────────────────────────────────

def _room_settings(ctx: JobContext, participant: Any) -> dict:
    """{"datasets", "agent", "instructions", "voice"} from the room's metadata
    (JSON), else the caller's attributes of the same names (datasets as a JSON list)."""
    out: dict = {}
    try:
        meta = json.loads(ctx.room.metadata or "{}")
    except ValueError:
        meta = {}
    attrs = getattr(participant, "attributes", {}) or {}
    for name in ("datasets", "agent", "instructions", "voice"):
        value = meta.get(name) if isinstance(meta, dict) else None
        if value is None and attrs.get(name):
            value = attrs[name]
            if name == "datasets":
                try:
                    value = json.loads(value)
                except ValueError:
                    value = None
        if value is not None:
            out[name] = value
    if isinstance(meta, list):                                          # metadata that is just the datasets
        out.setdefault("datasets", meta)
    if not isinstance(out.get("datasets", []), list):
        out.pop("datasets")
    return out


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=VAD_SILENCE_S)


def _dispatch(ctx: JobContext) -> dict:
    """What OpenTeddy sent with this job: {"session_id", "grant", "voice"?}, or {}."""
    try:
        meta = json.loads(getattr(ctx.job, "metadata", "") or "{}")
    except ValueError:
        return {}
    return meta if isinstance(meta, dict) and meta.get("grant") and meta.get("session_id") else {}


JOIN_WAIT_S = float(os.environ.get("AGENT_JOIN_WAIT_S", "120"))


async def entrypoint(ctx: JobContext) -> None:
    job = _dispatch(ctx)
    token = job.get("grant") or OT_KEY
    if not token:
        logger.error("room %s: no grant in the dispatch and no OPENTEDDY_KEY — leaving", ctx.job.room.name)
        return
    await ctx.connect()
    try:
        participant = await asyncio.wait_for(ctx.wait_for_participant(), JOIN_WAIT_S)
    except asyncio.TimeoutError:
        logger.info("room %s: nobody joined in %.0f s — leaving", ctx.room.name, JOIN_WAIT_S)
        ctx.shutdown("nobody joined")
        return
    if job:
        # OpenTeddy made the conversation for the app (POST /v1/livekit/token):
        # its agent, instructions and datasets are already on it.
        llm_ = OpenTeddyLLM(token, session_id=job["session_id"])
        voice = str(job.get("voice") or "")
        logger.info("room %s: a caller joined, conversation %s (from OpenTeddy), voice %s", ctx.room.name,
                    job["session_id"], voice or "(default)")
    else:
        settings = _room_settings(ctx, participant)
        llm_ = OpenTeddyLLM(token, datasets=settings.get("datasets"), agent=settings.get("agent"),
                            instructions=settings.get("instructions"))
        voice = str(settings.get("voice") or "")
        logger.info("room %s: a caller joined, datasets %s, agent %s, instructions %s", ctx.room.name,
                    settings.get("datasets"), settings.get("agent"), "yes" if settings.get("instructions") else "no")
    session = AgentSession(
        vad=ctx.proc.userdata.get("vad") or silero.VAD.load(min_silence_duration=VAD_SILENCE_S),
        stt=OpenTeddySTT(token),
        llm=llm_,
        # Speak the first clause as soon as it is written (ClauseFirstTokenizer).
        tts=tts.StreamAdapter(tts=OpenTeddyTTS(token, voice), sentence_tokenizer=ClauseFirstTokenizer()),
        turn_handling={"endpointing": {"min_delay": ENDPOINT_MIN_S, "max_delay": 2.0},
                       # Off: a half-finished question would become a turn in OpenTeddy's session.
                       "preemptive_generation": {"enabled": False}},
    )
    @session.on("metrics_collected")
    def _metrics(ev: Any) -> None:
        # Where the time goes, per turn: end of utterance (VAD + endpointing +
        # STT final), LLM first token, TTS first audio.
        m = ev.metrics
        nums = {k: round(v, 3) for k, v in m.model_dump().items()
                if isinstance(v, float) and k not in ("timestamp",) and v}
        logger.info("metrics %s %s", type(m).__name__, nums)

    await session.start(agent=Agent(instructions="OpenTeddy answers; its own prompts apply."), room=ctx.room)


if __name__ == "__main__":
    if not OT_KEY:
        logger.warning("OPENTEDDY_KEY is not set: only rooms OpenTeddy opens (with a grant) will work")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm,
                              # Explicit dispatch: it joins the rooms it is sent to, not every room.
                              agent_name=os.environ.get("AGENT_NAME", "openteddy"),
                              num_idle_processes=int(os.environ.get("AGENT_IDLE_PROCESSES", "1")),
                              host=os.environ.get("AGENT_HTTP_HOST", "127.0.0.1")))     # its health check, local only
