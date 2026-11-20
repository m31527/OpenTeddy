# OpenTeddy voice service

Speech recognition and speech synthesis for OpenTeddy's 🎙 voice mode, on a
machine of its own (a DGX Spark, or the same box as OpenTeddy).

| | Model | Licence |
|---|---|---|
| Speech → text | [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall) (FunASR) | [FunASR Model License](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE): use, modify, share with attribution. It also says the models are "for reference and learning purposes" — read it before building a paid product on it. |
| Text → speech | [Fun-CosyVoice3-0.5B-2512](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) (default), or [CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) | Apache-2.0 (code and model cards). The authors haven't answered a question about commercial use yet ([#1964](https://github.com/FunAudioLLM/CosyVoice/issues/1964)). |

OpenTeddy never downloads or starts this by itself. You install it, and you
point OpenTeddy at it (`voice_service_url`).

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

The first start downloads the models into `~/voice-models` (about 9 GB).
They use a few GB of GPU memory.

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
| `ZH_VARIANT` | `tw` | `tw` converts recognised Chinese to Traditional (Taiwan); `cn` leaves it as is |
