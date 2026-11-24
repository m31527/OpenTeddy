"""
OpenTeddy voice service — speech in, speech out, on a machine of its own.

  POST /asr     PCM16 mono 16 kHz (or a WAV) → {"text", "language", "emotion", "ms"}
                SenseVoiceSmall (FunASR); Chinese comes back in Traditional
                characters (OpenCC s2twp) unless ZH_VARIANT=cn. A sound with
                no speech in it (fsmn-vad) → empty text and "speech": false
  POST /tts     {"text", "voice"} → raw PCM16 mono at the model's rate,
                streamed as it is generated (Fun-CosyVoice3-0.5B by default)
  GET  /voices  the reference voices in VOICES_DIR (name.wav + name.txt)
  GET  /health  models, device, sample rates; while it is still getting
                ready, "stage" says what it is doing (downloading a model,
                loading, warming up) — it answers from the first second

OpenTeddy (voice_bridge.py) calls this from its /voice/ws; nothing here
knows about sessions, tools or the knowledge base. It serves one person's
conversation at a time: one lock per model.

Licences: CosyVoice code and the Fun-CosyVoice3 / CosyVoice2 weights are
Apache-2.0 (model cards), as is fsmn-vad; SenseVoiceSmall is under the FunASR Model
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
# SenseVoice writes words for any sound: a breath came back "Yeah.", a tap
# ".", a squeal 「在。」 (demo2, 11/22: a child nearby started conversations).
# FunASR's speech detector (3.9 MB, Apache-2.0) found no speech in any of
# those and still caught 0.3 s of real speech. VAD_MODEL= (empty) turns it off.
VAD_MODEL = os.environ.get("VAD_MODEL", "funasr/fsmn-vad")
ZH_VARIANT = os.environ.get("ZH_VARIANT", "tw")          # tw → Traditional (Taiwan phrasing), cn → as is
# The flow's ODE steps (CosyVoice: 10). First sound on demo2 (11/21): 10 steps
# 1.21 s, 6 steps 0.91 s, 4 steps 0.84 s — speech recognised right at all of them.
FLOW_STEPS = int(os.environ.get("FLOW_STEPS", "4") or 0)        # the owner chose 4 by ear (11/21)
# How soon the first sound comes. CosyVoice streams a chunk once it has
# TTS_FIRST_HOP speech tokens (25 = 1 s of audio) and doubles the chunk each
# time (TTS_HOP_SCALE). Its speech model makes ~40 tokens/s in float32 — not
# far above the 25/s audio needs — so a smaller first chunk only pays off
# with a faster model: TTS_FP16=1 (half precision) or TTS_VLLM=1 (CosyVoice's
# vLLM path). Unset, all four keep CosyVoice's own behaviour.
TTS_FIRST_HOP = int(os.environ.get("TTS_FIRST_HOP", "0") or 0)
TTS_HOP_SCALE = int(os.environ.get("TTS_HOP_SCALE", "0") or 0)
TTS_FP16 = os.environ.get("TTS_FP16", "").strip() in {"1", "true", "yes"}
TTS_VLLM = os.environ.get("TTS_VLLM", "").strip() in {"1", "true", "yes"}
ASR_RATE = 16000
CV3_PREFIX = "You are a helpful assistant.<|endofprompt|>"   # Fun-CosyVoice3's prompt text starts with this

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("voice")
app = FastAPI(title="OpenTeddy voice service")

_tts: Any = None
_asr: Any = None
_vad: Any = None
_tts_lock = threading.Lock()
_asr_lock = threading.Lock()
_voices: Dict[str, str] = {}            # name → CosyVoice speaker id, prepared once
_cc: Any = None
# CosyVoice2/3 double their stream chunk after each chunk (25 → 50 → 100
# speech tokens) and never set it back, so from the second sentence on the
# first sound waited for 100 tokens (~4 s of audio). Set back before each one.
_STREAM_HOP = [25]
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")   # SenseVoice marks emotions with them
# What it is doing before it can serve: OpenTeddy's installer (voice_setup.py)
# shows this while the models download the first time (~10 GB).
_stage: Dict[str, str] = {"stage": "starting", "detail": ""}


def _at(stage: str, detail: str = "") -> None:
    _stage.update(stage=stage, detail=detail)


# ── Loading ──────────────────────────────────────────────────────────────────

def _local(repo: str) -> str:
    """The model's folder under MODELS_DIR, downloaded from Hugging Face the first time."""
    path = os.path.join(MODELS_DIR, repo.split("/")[-1])
    if not os.path.isfile(os.path.join(path, ".complete")):
        from huggingface_hub import snapshot_download
        _at("downloading", repo)
        log.info("downloading %s → %s", repo, path)
        had = os.path.isdir(path) and bool(os.listdir(path))
        try:
            snapshot_download(repo, local_dir=path)    # picks up where a broken download stopped
            open(os.path.join(path, ".complete"), "w").close()
        except Exception as exc:  # noqa: BLE001
            if not had:
                raise
            # Folders from before the marker existed, on a machine offline now.
            log.warning("could not check %s (%s); using what is in %s", repo, exc, path)
    _at("loading", repo)
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
    global _tts, _asr, _vad, _cc
    import torch
    import torchaudio
    torchaudio.load = _soundfile_load                     # see _soundfile_load
    _stub_training_only()
    from cosyvoice.cli.cosyvoice import AutoModel as CosyModel
    t0 = time.time()
    _tts = CosyModel(model_dir=_local(TTS_MODEL), **({"fp16": True} if TTS_FP16 else {}),
                     **({"load_vllm": True} if TTS_VLLM else {}))
    # Newer transformers load the Qwen2 inside CosyVoice in the checkpoint's
    # bfloat16 while CosyVoice feeds it float32 ("mat1 and mat2 must have the
    # same dtype", demo2 11/20): run it in float32 as CosyVoice expects —
    # or in half precision throughout with TTS_FP16.
    llm = getattr(getattr(_tts, "model", None), "llm", None)
    if llm is not None and not TTS_VLLM:
        llm.half() if TTS_FP16 else llm.float()
    if TTS_FIRST_HOP:
        _tts.model.token_hop_len = TTS_FIRST_HOP
    if TTS_HOP_SCALE:
        _tts.model.stream_scale_factor = TTS_HOP_SCALE
    _STREAM_HOP[0] = int(getattr(_tts.model, "token_hop_len", 25) or 25)
    _quicker()
    log.info("TTS %s ready in %.1fs (rate %d, flow steps %s, first chunk %d tokens, ×%s, %s)", TTS_MODEL,
             time.time() - t0, _tts.sample_rate, FLOW_STEPS or "default", _STREAM_HOP[0],
             getattr(_tts.model, "stream_scale_factor", "?"),
             "vLLM" if TTS_VLLM else "fp16" if TTS_FP16 else "float32")
    from funasr import AutoModel as AsrModel
    t0 = time.time()
    _asr = AsrModel(model=_local(ASR_MODEL), trust_remote_code=False, disable_update=True,
                    device="cuda:0" if torch.cuda.is_available() else "cpu")
    log.info("ASR %s ready in %.1fs", ASR_MODEL, time.time() - t0)
    if VAD_MODEL:
        try:                                  # ~8 ms a sentence on the CPU
            _vad = AsrModel(model=_local(VAD_MODEL), disable_update=True, device="cpu")
            log.info("VAD %s ready", VAD_MODEL)
        except Exception as exc:  # noqa: BLE001 — without it, every sound is transcribed as before
            log.warning("VAD %s unavailable (%s): every sound goes to ASR", VAD_MODEL, exc)
    if ZH_VARIANT == "tw":
        from opencc import OpenCC
        _cc = OpenCC("s2twp")
    _prepare_voices()
    # The first synthesis after loading took 3 s (demo2, 11/20), later ones
    # 1.5 s: pay it now, not in the person's first sentence.
    if _voices:
        _at("warming", TTS_MODEL)
        t0 = time.time()
        for _ in _speak("你好。", "", 1.0):
            pass
        log.info("TTS warmed up in %.1fs", time.time() - t0)
    _at("ready")


def _load_in_background() -> None:
    try:
        load()
    except Exception as exc:  # noqa: BLE001
        log.exception("loading failed")
        _at("error", f"{type(exc).__name__}: {exc}"[:500])


def _ready() -> None:
    if _stage["stage"] != "ready":
        raise HTTPException(503, f"not ready: {_stage['stage']} {_stage['detail']}".strip())


class _QuickPoll:
    """What cosyvoice.cli.model sees as `time`: its stream loop checks for new
    speech tokens every 0.1 s; every 0.02 s gets each chunk out sooner."""

    def sleep(self, s: float) -> None:
        time.sleep(min(s, 0.02))

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


def _quicker() -> None:
    import cosyvoice.cli.model as cosy_model
    cosy_model.time = _QuickPoll()
    if FLOW_STEPS:
        dec = _tts.model.flow.decoder
        forward = dec.forward

        def fewer_steps(*a: Any, **k: Any) -> Any:
            if "n_timesteps" in k:
                k["n_timesteps"] = FLOW_STEPS
            return forward(*a, **k)
        dec.forward = fewer_steps


def _align_prompt(spk: str) -> None:
    """The first streamed chunk waits for enough speech tokens to line the
    reference voice's own tokens up with the chunk size (25): with the
    sample's 87 that was 13 more. Starting the flow's copy of the reference
    a few tokens in (the speech model keeps all of it) makes the first sound
    come ~0.25 s sooner (demo2, 11/21)."""
    import torch
    info = _tts.frontend.spk2info[spk]
    hop = _STREAM_HOP[0]
    tok = info["flow_prompt_speech_token"]
    n = tok.shape[1]
    k = n % hop
    ratio = int(getattr(_tts.model.flow, "token_mel_ratio", 2) or 2)
    if not k or n - k < hop or info["prompt_speech_feat"].shape[1] < ratio * n:
        return
    info["flow_prompt_speech_token"] = tok[:, k:]
    info["flow_prompt_speech_token_len"] = torch.tensor([n - k], dtype=torch.int32)
    feat = info["prompt_speech_feat"][:, ratio * k:]
    info["prompt_speech_feat"] = feat
    info["prompt_speech_feat_len"] = torch.tensor([feat.shape[1]], dtype=torch.int32)


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
                _align_prompt(spk)
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
    _ready()
    body = await request.body()
    if not body:
        raise HTTPException(400, "no audio")
    audio = _pcm(body, request.headers.get("content-type", ""))
    if audio.size < ASR_RATE // 10:
        return JSONResponse({"text": "", "language": "", "emotion": "", "ms": 0})
    t0 = time.time()

    def run() -> Optional[list]:
        with _asr_lock:
            if _vad is not None and not (_vad.generate(input=audio, cache={}) or [{}])[0].get("value"):
                return None                   # no one spoke: a breath, a tap, a squeal
            return _asr.generate(input=audio, cache={}, language="auto", use_itn=True, batch_size_s=60)
    import asyncio
    res = await asyncio.get_running_loop().run_in_executor(None, run)
    if res is None:
        return JSONResponse({"text": "", "language": "", "emotion": "", "speech": False,
                             "ms": int((time.time() - t0) * 1000)})
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
        t0, first, samples, chunks = time.time(), 0.0, 0, 0
        try:
            with _tts_lock:
                t0 = time.time()
                _tts.model.token_hop_len = _STREAM_HOP[0]          # see _STREAM_HOP
                for out in _tts.inference_zero_shot(text, "", "", zero_shot_spk_id=spk, stream=True, speed=speed):
                    wav = out["tts_speech"].squeeze(0).float().cpu().numpy()
                    first = first or time.time() - t0
                    samples += len(wav)
                    chunks += 1
                    q.put((np.clip(wav, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
            took, audio = time.time() - t0, samples / float(_tts.sample_rate)
            log.info("tts: first sound %.2fs, %d chunks, %.2fs for %.2fs of audio (RTF %.2f): %s",
                     first, chunks, took, audio, took / audio if audio else 0.0, text[:40])
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
    _ready()
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
    return {"ok": _stage["stage"] == "ready", "stage": _stage["stage"], "detail": _stage["detail"],
            "tts": TTS_MODEL, "asr": ASR_MODEL, "vad": VAD_MODEL if _vad is not None else "",
            "tts_sample_rate": int(_tts.sample_rate) if _tts else 0, "asr_sample_rate": ASR_RATE,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "voices": list(_voices), "zh_variant": ZH_VARIANT}


if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=_load_in_background, name="load", daemon=True).start()
    uvicorn.run(app, host=os.environ.get("VOICE_HOST", "127.0.0.1"), port=int(os.environ.get("VOICE_PORT", "8095")))
