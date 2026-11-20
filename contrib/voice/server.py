"""
OpenTeddy voice service — speech in, speech out, on a machine of its own.

  POST /asr     PCM16 mono 16 kHz (or a WAV) → {"text", "language", "emotion", "ms"}
                SenseVoiceSmall (FunASR); Chinese comes back in Traditional
                characters (OpenCC s2twp) unless ZH_VARIANT=cn
  POST /tts     {"text", "voice"} → raw PCM16 mono at the model's rate,
                streamed as it is generated (Fun-CosyVoice3-0.5B by default)
  GET  /voices  the reference voices in VOICES_DIR (name.wav + name.txt)
  GET  /health  models, device, sample rates

OpenTeddy (voice_bridge.py) calls this from its /voice/ws; nothing here
knows about sessions, tools or the knowledge base. It serves one person's
conversation at a time: one lock per model.

Licences: CosyVoice code and the Fun-CosyVoice3 / CosyVoice2 weights are
Apache-2.0 (model cards); SenseVoiceSmall is under the FunASR Model
License (use, modify, share with attribution — read it before a commercial
product). A reference voice is someone's voice: use your own, or one you
have the rights to. See README.md.

    python server.py            # env: VOICE_PORT, VOICE_HOST, MODELS_DIR, VOICES_DIR, TTS_MODEL, ASR_MODEL
"""
from __future__ import annotations

import io
import logging
import os
import queue
import re
import sys
import threading
import time
from typing import Any, Dict, Iterator, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

COSY_DIR = os.environ.get("COSYVOICE_DIR", "/opt/CosyVoice")
sys.path[:0] = [COSY_DIR, os.path.join(COSY_DIR, "third_party", "Matcha-TTS")]

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
VOICES_DIR = os.environ.get("VOICES_DIR", "/voices")
TTS_MODEL = os.environ.get("TTS_MODEL", "FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
ASR_MODEL = os.environ.get("ASR_MODEL", "FunAudioLLM/SenseVoiceSmall")
ZH_VARIANT = os.environ.get("ZH_VARIANT", "tw")          # tw → Traditional (Taiwan phrasing), cn → as is
ASR_RATE = 16000
CV3_PREFIX = "You are a helpful assistant.<|endofprompt|>"   # Fun-CosyVoice3's prompt text starts with this

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("voice")
app = FastAPI(title="OpenTeddy voice service")

_tts: Any = None
_asr: Any = None
_tts_lock = threading.Lock()
_asr_lock = threading.Lock()
_voices: Dict[str, str] = {}            # name → CosyVoice speaker id, prepared once
_cc: Any = None
# CosyVoice2/3 double their stream chunk after each chunk (25 → 50 → 100
# speech tokens) and never set it back, so from the second sentence on the
# first sound waited for 100 tokens (~4 s of audio). Set back before each one.
_STREAM_HOP = [25]
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")   # SenseVoice marks emotions with them


# ── Loading ──────────────────────────────────────────────────────────────────

def _local(repo: str) -> str:
    """The model's folder under MODELS_DIR, downloaded from Hugging Face the first time."""
    path = os.path.join(MODELS_DIR, repo.split("/")[-1])
    if not os.path.isdir(path) or not os.listdir(path):
        from huggingface_hub import snapshot_download
        log.info("downloading %s → %s", repo, path)
        snapshot_download(repo, local_dir=path)
    return path


def _soundfile_load(path: str, *a: Any, **k: Any):
    """torchaudio ≥ 2.9 loads through torchcodec (FFmpeg); a WAV prompt only needs soundfile."""
    import soundfile as sf
    import torch
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr


def _stub_training_only() -> None:
    """CosyVoice's model config imports its dataset code, which imports
    pyworld (pitch for training). Inference never calls it, and on aarch64 it
    only installs by compiling — so a stand-in that says so if it is ever used."""
    import importlib.util
    import types
    if importlib.util.find_spec("pyworld") is None:
        stub = types.ModuleType("pyworld")

        def _no(*a: Any, **k: Any) -> Any:
            raise RuntimeError("pyworld is only needed for CosyVoice training")
        stub.harvest = stub.dio = stub.stonemask = _no            # type: ignore[attr-defined]
        sys.modules["pyworld"] = stub


def load() -> None:
    global _tts, _asr, _cc
    import torch
    import torchaudio
    torchaudio.load = _soundfile_load                     # see _soundfile_load
    _stub_training_only()
    from cosyvoice.cli.cosyvoice import AutoModel as CosyModel
    t0 = time.time()
    _tts = CosyModel(model_dir=_local(TTS_MODEL))
    # Newer transformers load the Qwen2 inside CosyVoice in the checkpoint's
    # bfloat16 while CosyVoice feeds it float32 ("mat1 and mat2 must have the
    # same dtype", demo2 11/20): run it in float32 as CosyVoice expects.
    llm = getattr(getattr(_tts, "model", None), "llm", None)
    if llm is not None:
        llm.float()
    _STREAM_HOP[0] = int(getattr(_tts.model, "token_hop_len", 25) or 25)
    log.info("TTS %s ready in %.1fs (rate %d)", TTS_MODEL, time.time() - t0, _tts.sample_rate)
    from funasr import AutoModel as AsrModel
    t0 = time.time()
    _asr = AsrModel(model=_local(ASR_MODEL), trust_remote_code=False, disable_update=True,
                    device="cuda:0" if torch.cuda.is_available() else "cpu")
    log.info("ASR %s ready in %.1fs", ASR_MODEL, time.time() - t0)
    if ZH_VARIANT == "tw":
        from opencc import OpenCC
        _cc = OpenCC("s2twp")
    _prepare_voices()
    # The first synthesis after loading took 3 s (demo2, 11/20), later ones
    # 1.5 s: pay it now, not in the person's first sentence.
    if _voices:
        t0 = time.time()
        for _ in _speak("你好。", "", 1.0):
            pass
        log.info("TTS warmed up in %.1fs", time.time() - t0)


def _voice_files() -> Dict[str, tuple]:
    """name → (wav, prompt text). VOICES_DIR first; the CosyVoice sample as "sample"."""
    out: Dict[str, tuple] = {}
    if os.path.isdir(VOICES_DIR):
        for f in sorted(os.listdir(VOICES_DIR)):
            if f.endswith(".wav"):
                name = f[:-4]
                txt = os.path.join(VOICES_DIR, name + ".txt")
                if os.path.exists(txt):
                    out[name] = (os.path.join(VOICES_DIR, f), open(txt, encoding="utf-8").read().strip())
    sample = os.path.join(COSY_DIR, "asset", "zero_shot_prompt.wav")
    if os.path.exists(sample):
        out.setdefault("sample", (sample, "希望你以后能够做的比我还好呦。"))
    return out


def _prepare_voices() -> None:
    for name, (wav, text) in _voice_files().items():
        prompt = text if text.startswith(CV3_PREFIX) or "CosyVoice3" not in TTS_MODEL else CV3_PREFIX + text
        spk = f"ot_{name}"
        with _tts_lock:
            if _tts.add_zero_shot_spk(prompt, wav, spk):
                _voices[name] = spk
    log.info("voices: %s", ", ".join(_voices) or "(none)")


# ── Speech → text ────────────────────────────────────────────────────────────

_TAG = re.compile(r"<\|([^|]+)\|>")


def _pcm(request_body: bytes, content_type: str) -> np.ndarray:
    if "wav" in content_type or request_body[:4] == b"RIFF":
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(request_body), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if sr != ASR_RATE:
            import librosa
            mono = librosa.resample(mono, orig_sr=sr, target_sr=ASR_RATE)
        return mono.astype(np.float32)
    return np.frombuffer(request_body, dtype="<i2").astype(np.float32) / 32768.0


@app.post("/asr")
async def asr(request: Request) -> JSONResponse:
    body = await request.body()
    if not body:
        raise HTTPException(400, "no audio")
    audio = _pcm(body, request.headers.get("content-type", ""))
    if audio.size < ASR_RATE // 10:
        return JSONResponse({"text": "", "language": "", "emotion": "", "ms": 0})
    t0 = time.time()

    def run() -> list:
        with _asr_lock:
            return _asr.generate(input=audio, cache={}, language="auto", use_itn=True, batch_size_s=60)
    import asyncio
    res = await asyncio.get_running_loop().run_in_executor(None, run)
    raw = (res[0].get("text") if res else "") or ""
    tags = [t.lower() for t in _TAG.findall(raw)]
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    text = _EMOJI.sub("", rich_transcription_postprocess(raw)).strip()
    lang = next((t for t in tags if t in {"zh", "en", "yue", "ja", "ko", "nospeech"}), "")
    emotion = next((t for t in tags if t in {"happy", "sad", "angry", "neutral", "fearful", "disgusted", "surprised"}), "")
    if _cc is not None and lang in {"zh", "yue", ""}:
        text = _cc.convert(text)
    return JSONResponse({"text": text, "language": lang, "emotion": emotion, "ms": int((time.time() - t0) * 1000)})


# ── Text → speech, streamed ──────────────────────────────────────────────────

def _speak(text: str, voice: str, speed: float) -> Iterator[bytes]:
    """PCM16 chunks as CosyVoice produces them; runs in a worker thread."""
    q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=64)
    spk = _voices.get(voice) or next(iter(_voices.values()), None)
    if spk is None:
        raise HTTPException(503, "no reference voice: put name.wav + name.txt in VOICES_DIR")

    def work() -> None:
        try:
            with _tts_lock:
                _tts.model.token_hop_len = _STREAM_HOP[0]          # see _STREAM_HOP
                for out in _tts.inference_zero_shot(text, "", "", zero_shot_spk_id=spk, stream=True, speed=speed):
                    wav = out["tts_speech"].squeeze(0).float().cpu().numpy()
                    q.put((np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
        except Exception as exc:  # noqa: BLE001
            log.exception("tts failed: %s", exc)
        finally:
            q.put(None)
    threading.Thread(target=work, daemon=True).start()
    while True:
        chunk = q.get()
        if chunk is None:
            return
        yield chunk


@app.post("/tts")
async def tts(request: Request) -> StreamingResponse:
    body = await request.json()
    text = str(body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "no text")
    speed = float(body.get("speed") or 1.0)
    rate = int(_tts.sample_rate)
    return StreamingResponse(_speak(text[:2000], str(body.get("voice") or ""), speed),
                             media_type=f"audio/L16; rate={rate}; channels=1",
                             headers={"X-Sample-Rate": str(rate)})


@app.get("/voices")
async def voices() -> Dict[str, Any]:
    return {"voices": list(_voices)}


@app.get("/health")
async def health() -> Dict[str, Any]:
    import torch
    return {"ok": _tts is not None and _asr is not None, "tts": TTS_MODEL, "asr": ASR_MODEL,
            "tts_sample_rate": int(_tts.sample_rate) if _tts else 0, "asr_sample_rate": ASR_RATE,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "voices": list(_voices), "zh_variant": ZH_VARIANT}


if __name__ == "__main__":
    import uvicorn
    load()
    uvicorn.run(app, host=os.environ.get("VOICE_HOST", "127.0.0.1"), port=int(os.environ.get("VOICE_PORT", "8095")))
