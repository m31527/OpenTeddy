# OpenTeddy voice service

Speech recognition and speech synthesis for OpenTeddy's 🎙 voice mode, on a
machine of its own (a DGX Spark, or the same box as OpenTeddy).

**Languages: Chinese and English.** Ask in either and OpenTeddy answers in
it, out loud. (The models know a few more; only these two are tried and
supported.)

| | Model | Licence |
|---|---|---|
| Speech → text | [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) (FunASR) | [FunASR Model License](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE): use, modify, share with attribution. It also says the models are "for reference and learning purposes" — read it before building a paid product on it. |
| Is anyone speaking? | [fsmn-vad](https://huggingface.co/funasr/fsmn-vad) (FunASR, 3.9 MB) | Apache-2.0 (model card) |
| Text → speech | [Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) (default), or [CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) | Apache-2.0 (code and model cards). The authors haven't answered a question about commercial use yet ([#1964](https://github.com/FunAudioLLM/CosyVoice/issues/1964)). |

OpenTeddy never downloads or starts this by itself. Voice mode is off until
you turn it on: the 🎙 button (or Settings → 🎙 Voice conversation) asks
first, and on a Linux machine with Docker and an NVIDIA GPU it can then do
the steps below for you — build the image, start the container, download the
models (about 10 GB) — and show how far it is. Turning voice mode off stops
the container; the models stay. Or install it yourself as below and point
OpenTeddy at it (`voice_service_url`).

## The voice it speaks with

CosyVoice imitates a short recording of a voice: a WAV of 3 to 10 seconds
plus the exact words said in it. **A voice belongs to a person.** Use your
own, or one you have the right to use. The bundled `sample` (CosyVoice's
demo prompt) is for trying it out only.

Put each voice in `VOICES_DIR` as `name.wav` + `name.txt`:

```
voices/
  me.wav      # 16 kHz or more, mono, quiet room, 3–10 s
  me.txt      # 「大家好，我是…」 — exactly what is said in me.wav
```

Then set `voice_name` in OpenTeddy (Settings) to `me`.

## Run it on a DGX Spark (aarch64, GB10)

The image is built on `vllm/vllm-openai:v0.33.0-aarch64`, whose torch and
torchaudio are built for the GB10. pip has no aarch64 CUDA wheels to mix in.

```bash
cd contrib/voice
docker build -t openteddy-voice .
mkdir -p ~/voice-models ~/voice-voices
docker run -d --name openteddy-voice --restart unless-stopped --gpus all \
  -p 127.0.0.1:8095:8095 -e VOICE_HOST=0.0.0.0 \
  -v ~/voice-models:/models -v ~/voice-voices:/voices openteddy-voice
curl -s 127.0.0.1:8095/health
```

The first start downloads the models into `~/voice-models` (about 10 GB);
`/health` says how far it is (`stage`: downloading, loading, warming,
ready). They use a few GB of GPU memory.

On an x86-64 machine with an NVIDIA GPU, build with
`--build-arg BASE=vllm/vllm-openai:v0.33.0` (not tried yet).

If the voice service runs on another machine than OpenTeddy, publish the
port on an address OpenTeddy can reach (`-p 10.x.y.z:8095:8095`) instead of
`127.0.0.1`. There is no login on it, so keep it on your own network.

## Point OpenTeddy at it

Settings → `voice_service_url` = `http://127.0.0.1:8095`, or the 🎙 button
asks for it.

Browsers let a page use the microphone only over **https** or on
**localhost**. To talk from another computer or a phone, give OpenTeddy an
https address:

- **A domain of your own** (for example a Cloudflare tunnel to
  `http://127.0.0.1:8000`) is the smooth way: a real certificate, and no
  warning on phones. WebSockets go through. `/voice/ws` accepts the page
  when its Origin matches the `Host` or the `X-Forwarded-Host` the tunnel
  passes on. The audio then travels through the tunnel provider.
- **No domain:** start OpenTeddy with an https door as well:

  ```bash
  OPENTEDDY_HTTPS_PORT=8443 ./run.sh --host 0.0.0.0
  ```

  Then open `https://<its address>:8443`. The certificate is made on that
  machine, so the browser warns once; accept it.

For a quick try from your own computer, an SSH tunnel to `localhost` is
enough: `ssh -L 8000:127.0.0.1:8000 <server>`, then open
`http://localhost:8000`.

## Environment

| Variable | Default | |
|---|---|---|
| `VOICE_HOST` / `VOICE_PORT` | `127.0.0.1` / `8095` | where it listens |
| `MODELS_DIR` | `/models` | models, downloaded the first time |
| `VOICES_DIR` | `/voices` | reference voices |
| `TTS_MODEL` | `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` | or `FunAudioLLM/CosyVoice2-0.5B` |
| `ASR_MODEL` | `FunAudioLLM/SenseVoiceSmall` | |
| `VAD_MODEL` | `funasr/fsmn-vad` | checks a sound has speech in it before ASR (SenseVoice turns a breath into "Yeah."); empty turns it off |
| `ZH_VARIANT` | `tw` | `tw` converts recognised Chinese to Traditional (Taiwan); `cn` leaves it as is |
| `FLOW_STEPS` | `4` | the flow's ODE steps (CosyVoice's own: 10); fewer is a sooner first sound |
| `TTS_FP16` | off | `1` runs CosyVoice in half precision: first sound 0.72 → 0.52 s on a DGX Spark, the same words heard back |
| `TTS_FIRST_HOP` | 25 | speech tokens in the first chunk; `12` with `TTS_FP16=1` gave 0.44 s and no stalls (without fp16 it stalls) |
| `TTS_HOP_SCALE` / `TTS_VLLM` | 2 / off | how fast later chunks grow; CosyVoice's vLLM path (doesn't load with this image's transformers) |

spark B runs `TTS_FP16=1 TTS_FIRST_HOP=12` (11/24, the owner listened and chose it).
