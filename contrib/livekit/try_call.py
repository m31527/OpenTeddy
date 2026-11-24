"""
A test call through LiveKit, end to end and the way an app makes one: ask
OpenTeddy for a room with the key (POST /v1/livekit/token), join it as a
person, say a question into the microphone track, and time the agent's
voice coming back.

    python try_call.py [--question "請問你可以幫我做什麼？"] [--datasets 客服FAQ]
                       [--agent 品牌客戶・王經理] [--instructions "…"] [--voice male1] [--runs 3]

Env: OPENTEDDY_URL + OPENTEDDY_KEY. LIVEKIT_URL, when set, is joined instead
of the address OpenTeddy hands out (e.g. ws://192.168.100.2:7880 on the LAN).

"First sound" is from the end of the question's speech (trailing silence
trimmed) to the first audible frame of the agent's answer, as received
here: VAD end-of-speech + STT final + LLM first sentence + TTS first audio
+ WebRTC, everything the spec's < 1.5 s target covers.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import os
import statistics
import time
import wave
from typing import List, Optional

import httpx
import numpy as np
from livekit import rtc

RATE, FRAME = 48000, 480                      # 10 ms frames
LK_URL = os.environ.get("LIVEKIT_URL", "")
OT_URL = os.environ.get("OPENTEDDY_URL", "http://127.0.0.1:8000/v1").rstrip("/")
AUTH = {"Authorization": f"Bearer {os.environ.get('OPENTEDDY_KEY', '')}"}


async def question_audio(text: str) -> np.ndarray:
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{OT_URL}/audio/speech", json={"input": text}, headers=AUTH)
        r.raise_for_status()
    with wave.open(io.BytesIO(r.content)) as w:
        rate, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32)
    x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * RATE / rate)), np.arange(len(x)), x)
    loud = np.nonzero(np.abs(x) > 500)[0]
    return x[: loud[-1] + 1].astype("<i2") if len(loud) else x.astype("<i2")       # speech ends where sound does


async def one_call(q: np.ndarray, setup: dict, hear: float = 3.0) -> Optional[float]:
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{OT_URL}/livekit/token", headers=AUTH,
                         json={"identity": "tester", "name": "Tester", **setup})
        if r.status_code >= 400:
            raise SystemExit(f"/v1/livekit/token: {r.status_code} {r.text[:300]}")
        call = r.json()
    print(f"  room {call['room']}, conversation {call['session_id']}"
          + "".join(f", {k} {call[k]}" for k in ("agent", "datasets", "voice") if k in call))
    room = rtc.Room()
    state = {"end": None, "first": None, "any": None, "agent": asyncio.Event(), "said": []}

    async def listen(track: rtc.Track) -> None:
        async for ev in rtc.AudioStream(track, sample_rate=16000, num_channels=1):
            if state["end"] is None:
                continue
            level = np.abs(np.frombuffer(ev.frame.data, np.int16)).mean()
            if state["any"] is None and level > 20:              # the first sound of any kind
                state["any"] = time.perf_counter()
            if state["first"] is None and level > 300:           # clearly speech
                state["first"] = time.perf_counter()

    @room.on("track_subscribed")
    def _sub(track: rtc.Track, pub: rtc.RemoteTrackPublication, p: rtc.RemoteParticipant) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            asyncio.ensure_future(listen(track))

    @room.on("participant_connected")
    def _joined(p: rtc.RemoteParticipant) -> None:
        state["agent"].set()

    async def transcript(reader, participant_identity) -> None:
        try:
            text = await reader.read_all()
        except Exception:  # noqa: BLE001 — cut off by hanging up
            return
        state["said"].append(("you: " if participant_identity == "tester" else "agent: ") + text)
    try:
        room.register_text_stream_handler("lk.transcription", lambda r, pid: asyncio.ensure_future(transcript(r, pid)))
    except Exception:  # noqa: BLE001 — transcripts are a nicety
        pass

    await room.connect(LK_URL or call["url"], call["token"])
    source = rtc.AudioSource(RATE, 1)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    await room.local_participant.publish_track(track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    if room.remote_participants:
        state["agent"].set()
    await asyncio.wait_for(state["agent"].wait(), 30)

    def frame(samples: np.ndarray) -> rtc.AudioFrame:
        return rtc.AudioFrame(data=samples.astype("<i2").tobytes(), sample_rate=RATE, num_channels=1,
                              samples_per_channel=len(samples))
    silence = np.zeros(FRAME, dtype="<i2")
    for _ in range(150):                                    # 1.5 s of quiet: the agent settles in
        await source.capture_frame(frame(silence))
    for i in range(0, len(q), FRAME):
        chunk = q[i:i + FRAME]
        await source.capture_frame(frame(np.pad(chunk, (0, FRAME - len(chunk)))))
    await source.wait_for_playout()                         # the question has been sent
    state["end"] = time.perf_counter()
    deadline = state["end"] + 15
    while state["first"] is None and time.perf_counter() < deadline:
        await source.capture_frame(frame(silence))          # a microphone keeps sending, quiet now
    for _ in range(int(hear * 100)):                        # hear the answer out
        await source.capture_frame(frame(silence))
    await room.disconnect()                                 # the room closes itself once empty
    took = state["first"] - state["end"] if state["first"] else None
    onset = state["any"] - state["end"] if state["any"] else None
    wall = time.time() - time.perf_counter()                 # perf_counter → wall clock, for the agent's timeline
    print(f"  end of speech @{state['end'] + wall:.3f}, first audio @{state['any'] + wall:.3f}" if state["any"] else "")
    print(f"  first sound {took:.2f}s (first audio of any level {onset:.2f}s)" if took else "  no answer heard in 15 s",
          "|", " / ".join(s for s in state["said"] if s.split(": ", 1)[-1])[-200:])
    return took


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="請問你可以幫我做什麼？")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--agent")
    ap.add_argument("--instructions")
    ap.add_argument("--voice", help="one of GET /v1/voices; default: the one set in OpenTeddy")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--hear", type=float, default=3.0, help="seconds to keep listening after the first sound")
    a = ap.parse_args()
    q = await question_audio(a.question)
    setup = {k: v for k, v in (("datasets", a.datasets), ("agent", a.agent), ("instructions", a.instructions),
                               ("voice", a.voice)) if v is not None}
    print(f"«{a.question}» ({len(q) / RATE:.1f}s of speech), {a.runs} calls")
    times = [t for t in [await one_call(q, setup, a.hear) for _ in range(a.runs)] if t]
    if times:
        print(f"end of speech → first sound: median {statistics.median(times):.2f}s, best {min(times):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
