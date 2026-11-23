# OpenTeddy's /v1 API: for apps with an API key

The `/v1` API lets an app talk to OpenTeddy. An app can:

- chat,
- turn speech into text and text into speech,
- hold a spoken conversation in real time.

It works the way OpenAI's API does: the key goes in `Authorization: Bearer ot-…`. OpenAI's SDKs work if you set `base_url` to `<OpenTeddy>/v1`.

A turn runs exactly like one typed in OpenTeddy's web UI: the knowledge base, memory and tools all take part. Code: `public_api.py` and `api_keys.py`.

## Keys

To make a key, go to **Settings → 🔑 API keys**, type a name and press **Create key**. The key is shown once; copy it then.

Only a hash of the key is stored. The list shows each key's hint (`ot-AbC…wxyz`), when it was last used, and how many calls it has made. **Revoke** stops a key immediately.

Each key is its own user:

- **Its sessions are its own.** Another key can't read or continue them.
- **What it learns stays with it.** Memory is kept per session.
- **It can't reach your personal things.** No paired Chrome, no linked accounts.
- **You can still see its conversations.** They appear in your web UI's session list as `🔑 <name> · …`.

A tool that needs approval waits for it, exactly as it does in the web UI. You can approve it in the web UI, or the app can approve it with `POST /v1/tasks/{id}/approve`.

## Where to reach it

`/v1` is served on OpenTeddy's usual address, for example `http://<host>:8000/v1`.

To open it to the internet, start OpenTeddy with a port that serves **only** `/v1` and put a tunnel in front of that port:

```bash
# .env, next to OpenTeddy
OPENTEDDY_API_PORT=8100          # serves /v1 and nothing else
OPENTEDDY_API_HOST=127.0.0.1     # the default: only the tunnel on this machine reaches it
```

Then point the tunnel at `http://127.0.0.1:8100`. For example, in a Cloudflare tunnel, a public hostname `api.example.com` → `http://127.0.0.1:8100`.

On that port the web UI, Settings and every other route answer 404. The tunnel provides HTTPS.

## Errors

Errors come back with an HTTP status, shaped as OpenAI shapes them:

```json
{"error": {"message": "Incorrect API key provided, or it was revoked.", "type": "invalid_request_error", "param": null, "code": "invalid_api_key"}}
```

| Status | `code` | When |
|---|---|---|
| 401 | `missing_api_key` / `invalid_api_key` | No key, a wrong key, or a revoked key |
| 404 | `session_not_found` / `task_not_found` | The session or task isn't this key's |
| 400 | `unsupported_audio` and others | The request itself is wrong |
| 503 | `voice_not_configured` / `voice_off` | There is no voice service, or voice is turned off in Settings |
| 502 | `voice_unreachable` | The voice service didn't answer |

## Chat: `POST /v1/chat/completions`

The request is OpenAI-shaped. OpenTeddy answers the **last user message**.

```bash
curl https://api.example.com/v1/chat/completions \
  -H "Authorization: Bearer $OPENTEDDY_KEY" -H "Content-Type: application/json" \
  -d '{"model": "openteddy", "messages": [{"role": "user", "content": "今天台北天氣如何？"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="https://api.example.com/v1", api_key="ot-…")
r = client.chat.completions.create(model="openteddy",
                                   messages=[{"role": "user", "content": "Hi, who are you?"}])
print(r.choices[0].message.content)
session = r.model_extra["session_id"]          # to go on in the same conversation
```

The answer is OpenAI's `chat.completion` object. It also carries the fields `session_id`, `task_id` and `status` (`completed`, `failed`, `running` or `waiting_approval`). The headers `X-OpenTeddy-Session` and `X-OpenTeddy-Task` carry the same IDs. The `usage` counts are 0.

### Conversations

There are two ways to keep a conversation going:

- **OpenTeddy keeps it.** Send the `session_id` from the previous answer, either as a body field or as the `X-OpenTeddy-Session` header. With the OpenAI SDK, put it in `extra_body={"session_id": …}`. Only the last user message is needed; OpenTeddy remembers the rest.
- **The app keeps it.** Leave `session_id` out and send the whole history as usual. Each such call starts a new session. Its earlier messages (system, user, assistant; the last 12) are given to OpenTeddy as the conversation so far.

### Waiting, and long work

A plain answer waits up to **90 seconds**. That is under a tunnel's 100-second limit for the first byte. Change it with the body field `"wait": <seconds>`, from 0 to 600.

When the work takes longer, the answer has `status: "running"` and a short message saying it is still working. Fetch the result later with `GET /v1/tasks/{task_id}`.

If the work waits for an approval, the status is `waiting_approval`. The message names the tool.

### Streaming

With `"stream": true` the answer comes as server-sent `chat.completion.chunk` events:

1. The role arrives first.
2. A `: keep-alive` comment follows every 15 s while OpenTeddy works.
3. The answer then arrives as **one** chunk, followed by `finish_reason: "stop"` and `data: [DONE]`.

The answer isn't split token by token, because not every path produces it that way. A stream waits up to 30 minutes.

Text only: image parts in `content` are ignored. `temperature`, `max_tokens` and similar fields are accepted and ignored; OpenTeddy picks its own models.

## `GET /v1/models`

Returns a list with one model, `openteddy`.

## Tasks: `GET /v1/tasks/{id}`

```json
{"task_id": "…", "session_id": "…", "status": "completed", "result": "…", "approvals": [],
 "created_at": "…", "updated_at": "…"}
```

`result` is filled in once the task is done (`completed`, `failed` or `escalated`). `approvals` lists the tool calls waiting for a yes or no.

- `POST /v1/tasks/{id}/approve` lets the waiting tool calls run.
- `POST /v1/tasks/{id}/reject` refuses them.
- `POST /v1/tasks/{id}/cancel` stops the task.

## Speech to text: `POST /v1/audio/transcriptions`

The request is OpenAI-shaped multipart: `file`, plus optional `model`, `language` and `response_format`. `response_format` can be `json` (the default), `text` or `verbose_json`.

Speech recognition is SenseVoice, running in the voice service. It handles Chinese, English, Cantonese, Japanese and Korean.

```bash
curl https://api.example.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENTEDDY_KEY" -F file=@question.wav
# {"text": "今天幾月幾號"}
```

What it accepts:

- **WAV** (PCM, any sample rate, mono or stereo), or raw 16-bit PCM sent as `audio/L16; rate=16000`.
- **mp3, m4a, webm and ogg** only when `ffmpeg` is installed where OpenTeddy runs.
- At most 25 MB and 120 seconds per call.

Silence or a noise gives `{"text": ""}`.

## Text to speech: `POST /v1/audio/speech`

```bash
curl https://api.example.com/v1/audio/speech \
  -H "Authorization: Bearer $OPENTEDDY_KEY" -H "Content-Type: application/json" \
  -d '{"model": "cosyvoice", "input": "你好，我是 OpenTeddy。", "response_format": "wav"}' -o hello.wav
```

`response_format` is `wav` (the default) or `pcm`. `pcm` is raw 16-bit mono, streamed as it is made; the `X-Sample-Rate` header gives the rate, 24000 for CosyVoice 3.

`voice` is one of the voice service's voices. Any other name, such as OpenAI's `alloy`, uses the voice set in Settings. Markdown, links and emoji are dropped before speaking. The limit is 4096 characters.

## Real-time voice: `WS /v1/realtime`

This is a spoken conversation, the same one the 🎙 button in the web UI uses (`voice_bridge.py`). Its protocol is OpenTeddy's own, not OpenAI's Realtime protocol.

The voice service must be set up and voice turned on in Settings.

**The key.** A native app sends it as `Authorization: Bearer ot-…`. A browser can't set headers on a WebSocket, so it offers two subprotocols: `openteddy` and `openteddy-key.<key>`.

```js
const ws = new WebSocket("wss://api.example.com/v1/realtime", ["openteddy", "openteddy-key." + key]);
ws.binaryType = "arraybuffer";
```

If the key is wrong, the handshake is refused with HTTP 403.

**Messages.** → is from the app, ← is from OpenTeddy.

| Direction | Message |
|---|---|
| → first | `{"type": "hello", "session_id": "…"}`. Send `session_id: ""` (or leave it out) for a new conversation. |
| ← | `{"type": "ready", "session_id", "sample_rate", "asr_rate": 16000, "voice", "asr", "tts"}` |
| → | **binary**: one utterance, PCM16 mono at 16 kHz. The app detects the end of speech; the web UI waits for 0.5 s of silence. |
| → | `{"type": "text", "text": "…"}`: a typed turn instead of a spoken one |
| → | `{"type": "interrupt"}`: the person started talking, so stop speaking |
| ← | `{"type": "state", "value": "listening" \| "thinking" \| "speaking"}` |
| ← | `{"type": "transcript", "text", "ms"}`: what was heard |
| ← | `{"type": "reply", "text", "task_id"}`: the answer as text |
| ← | **binary**: the spoken answer, PCM16 mono at `sample_rate`, in chunks as it is made. `{"type": "audio_end"}` follows. |
| ← | `{"type": "working", "task_id", "text"}`: work that takes longer than 8 s is acknowledged out loud … |
| ← | `{"type": "done", "task_id", "text"}`: … and then spoken when it is done |
| ← | `{"type": "error", "code", "text"}`: `code` is `not_enabled`, `not_configured`, `unreachable`, `loading` or `bad_session` |

Noise that isn't speech starts nothing: the voice service checks for a voice first.

The audio of the conversation travels between the app and OpenTeddy over the tunnel. Between OpenTeddy and its voice service it stays on the local network.
