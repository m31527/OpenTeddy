"""
A test call through LiveKit, end to end: join a room as a person, say a
question into the microphone track, and time the agent's voice coming back.

    python try_call.py [--question "請問你可以幫我做什麼？"] [--datasets 客服FAQ] [--runs 3]

Env: LIVEKIT_URL (e.g. ws://192.168.100.2:7880), LIVEKIT_API_KEY, LIVEKIT_API_SECRET,
     OPENTEDDY_URL + OPENTEDDY_KEY (the question's audio is made with /v1/audio/speech).

"First sound" is from the end of the question's speech (trailing silence
trimmed) to the first audible frame of the agent's answer, as received
here: VAD end-of-speech + STT final + LLM first sentence + TTS first audio
+ WebRTC, everything the spec's < 1.5 s target covers.
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import statistics
import time
import uuid
import wave
from typing import List, Optional

import httpx
import numpy as np
from livekit import api, rtc

RATE, FRAME = 48000, 480                      # 10 ms frames
LK_URL = os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880")
OT_URL = os.environ.get("OPENTEDDY_URL", "http://127.0.0.1:8000/v1").rstrip("/")


async def question_audio(text: str) -> np.ndarray:
    async with httpx.AsyncClient(timeout=120) as c:
        r = await c.post(f"{OT_URL}/audio/speech", json={"input": text},
                         headers={"Authorization": f"Bearer {os.environ['OPENTEDDY_KEY']}"})
        r.raise_for_status()
    with wave.open(io.BytesIO(r.content)) as w:
        rate, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32)
    x = np.interp(np.linspace(0, len(x) - 1, int(len(x) * RATE / rate)), np.arange(len(x)), x)
    loud = np.nonzero(np.abs(x) > 500)[0]
    return x[: loud[-1] + 1].astype("<i2") if len(loud) else x.astype("<i2")       # speech ends where sound does


async def one_call(q: np.ndarray, datasets: Optional[List[str]], hear: float = 3.0) -> Optional[float]:
    name = f"ot-test-{uuid.uuid4().hex[:8]}"
    key, secret = os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"]
    lk = api.LiveKitAPI(url=LK_URL.replace("ws", "http", 1), api_key=key, api_secret=secret)
    await lk.room.create_room(api.CreateRoomRequest(
        name=name, empty_timeout=20, metadata=json.dumps({"datasets": datasets}) if datasets else ""))
    token = (api.AccessToken(key, secret).with_identity("tester").with_name("Tester")
             .with_grants(api.VideoGrants(room_join=True, room=name)).to_jwt())
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
        text = await reader.read_all()
        state["said"].append(("you: " if participant_identity == "tester" else "agent: ") + text)
    try:
        room.register_text_stream_handler("lk.transcription", lambda r, pid: asyncio.ensure_future(transcript(r, pid)))
    except Exception:  # noqa: BLE001 — transcripts are a nicety
        pass

    await room.connect(LK_URL, token)
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
    await room.disconnect()
    await lk.room.delete_room(api.DeleteRoomRequest(room=name))
    await lk.aclose()
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
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--hear", type=float, default=3.0, help="seconds to keep listening after the first sound")
    a = ap.parse_args()
    q = await question_audio(a.question)
    print(f"«{a.question}» ({len(q) / RATE:.1f}s of speech), {a.runs} calls through {LK_URL}")
    times = [t for t in [await one_call(q, a.datasets, a.hear) for _ in range(a.runs)] if t]
    if times:
        print(f"end of speech → first sound: median {statistics.median(times):.2f}s, best {min(times):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
