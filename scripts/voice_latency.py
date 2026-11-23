#!/usr/bin/env python3
"""
Voice latency, stage by stage, through OpenTeddy's /v1 API — the numbers
docs/livekit-voice.md asks for (target: end of speech → first sound < 1.5 s).

    scripts/voice_latency.py --base http://127.0.0.1:8100/v1 --key-file ~/.ot-key [--datasets 客服FAQ]

  tts       POST /v1/audio/speech (pcm): time to the first audio bytes
  stt       WS /v1/audio/stt-stream: audio sent in 40 ms chunks at real-time
            pace; time from the last chunk to "final" (skipped if absent)
  llm       POST /v1/chat/completions stream: time to the first content token
            (TTFT) and to the end
  realtime  WS /v1/realtime: one spoken question → transcript, first sound

Every number is the median of --runs runs (default 3).
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import statistics
import sys
import time
import wave
from typing import Dict, List, Optional

import httpx

QUESTIONS = ["你好，簡單介紹一下你自己", "用一句話解釋什麼是雲端運算", "給我一個今天晚餐的建議"]
SPOKEN = "請問你可以幫我做什麼？"


def pcm16k(wav_bytes: bytes) -> bytes:
    """A WAV from /v1/audio/speech (24 kHz) → PCM16 16 kHz mono."""
    import numpy as np
    with wave.open(io.BytesIO(wav_bytes)) as w:
        rate, frames = w.getframerate(), w.readframes(w.getnframes())
    x = np.frombuffer(frames, "<i2").astype(np.float32)
    n = int(len(x) * 16000 / rate)
    y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return np.clip(y, -32768, 32767).astype("<i2").tobytes()


def med(xs: List[float]) -> str:
    xs = [x for x in xs if x is not None]
    return f"{statistics.median(xs):.2f}s" if xs else "—"


async def bench_tts(c: httpx.AsyncClient, base: str, h: Dict[str, str], runs: int) -> None:
    first, total = [], []
    for i in range(runs):
        t = time.perf_counter()
        async with c.stream("POST", f"{base}/audio/speech", headers=h,
                            json={"model": "cosyvoice", "input": QUESTIONS[i % len(QUESTIONS)], "response_format": "pcm"}) as r:
            got = None
            async for chunk in r.aiter_bytes():
                if chunk and got is None:
                    got = time.perf_counter() - t
            first.append(got)
            total.append(time.perf_counter() - t)
    print(f"tts       first audio {med(first)}   whole sentence {med(total)}")


async def bench_llm(c: httpx.AsyncClient, base: str, h: Dict[str, str], runs: int, datasets: Optional[List[str]]) -> None:
    ttft, total = [], []
    for i in range(runs):
        body = {"model": "openteddy", "stream": True,
                "messages": [{"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]}]}
        if datasets:
            body["datasets"] = datasets
        t, first = time.perf_counter(), None
        async with c.stream("POST", f"{base}/chat/completions", headers=h, json=body) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
                if delta and first is None:
                    first = time.perf_counter() - t
        ttft.append(first)
        total.append(time.perf_counter() - t)
    print(f"llm       first token {med(ttft)}   whole answer {med(total)}")


async def bench_stt(base: str, h: Dict[str, str], pcm: bytes, runs: int) -> None:
    import websockets
    url = base.replace("http", "ws", 1) + "/audio/stt-stream"
    lat, partials = [], []
    for _ in range(runs):
        try:
            async with websockets.connect(url, additional_headers=h) as ws:
                n_partial, step = 0, 1280                      # 40 ms of 16 kHz PCM16
                for i in range(0, len(pcm), step):
                    await ws.send(pcm[i:i + step])
                    await asyncio.sleep(0.04)
                    try:
                        while True:
                            m = json.loads(await asyncio.wait_for(ws.recv(), 0.001))
                            n_partial += m.get("type") == "partial"
                    except (asyncio.TimeoutError, TimeoutError):
                        pass
                t = time.perf_counter()
                await ws.send(json.dumps({"type": "flush"}))
                while True:
                    m = json.loads(await asyncio.wait_for(ws.recv(), 30))
                    if m.get("type") == "final":
                        lat.append(time.perf_counter() - t)
                        partials.append(n_partial)
                        text = m.get("text")
                        break
        except Exception as exc:  # noqa: BLE001
            print(f"stt       not available ({type(exc).__name__}: {exc})")
            return
    print(f"stt       final after end of speech {med(lat)}   partials while speaking {partials}   «{text}»")


async def bench_realtime(base: str, h: Dict[str, str], pcm: bytes, runs: int, datasets: Optional[List[str]]) -> None:
    import websockets
    url = base.replace("http", "ws", 1) + "/realtime"
    heard, first_sound = [], []
    for _ in range(runs):
        async with websockets.connect(url, additional_headers=h) as ws:
            await ws.send(json.dumps({"type": "hello", **({"datasets": datasets} if datasets else {})}))
            ready = json.loads(await ws.recv())
            if ready.get("type") != "ready":
                print(f"realtime  {ready}")
                return
            t = time.perf_counter()
            await ws.send(pcm)
            h_t = s_t = None
            while True:
                m = await asyncio.wait_for(ws.recv(), 120)
                if isinstance(m, bytes):
                    if s_t is None:
                        s_t = time.perf_counter() - t
                    continue
                d = json.loads(m)
                if d["type"] == "transcript" and h_t is None:
                    h_t = time.perf_counter() - t
                if d["type"] in ("audio_end", "error"):
                    break
            heard.append(h_t)
            first_sound.append(s_t)
    print(f"realtime  heard {med(heard)}   first sound {med(first_sound)}   (from the end of the utterance)")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8100/v1")
    ap.add_argument("--key-file", default=os.path.expanduser("~/.ot-key"))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--only", nargs="*", default=["tts", "stt", "llm", "realtime"])
    a = ap.parse_args()
    h = {"Authorization": f"Bearer {open(a.key_file).read().strip()}"}
    async with httpx.AsyncClient(timeout=180) as c:
        wav = (await c.post(f"{a.base}/audio/speech", headers=h, json={"input": SPOKEN})).content
        pcm = pcm16k(wav)
        print(f"question: «{SPOKEN}» ({len(pcm) / 32000:.1f}s of speech), {a.runs} runs each, medians")
        if "tts" in a.only:
            await bench_tts(c, a.base, h, a.runs)
        if "stt" in a.only:
            await bench_stt(a.base, h, pcm, a.runs)
        if "llm" in a.only:
            await bench_llm(c, a.base, h, a.runs, a.datasets)
    if "realtime" in a.only:
        await bench_realtime(a.base, h, pcm, a.runs, a.datasets)


if __name__ == "__main__":
    asyncio.run(main())
