# OpenTeddy's /v1 API: for apps with an API key

The `/v1` API lets an app talk to OpenTeddy. An app can:

- chat,
- turn speech into text and text into speech,
- hold a spoken conversation in real time,
- answer from the knowledge-base datasets it names.

Building a voice app? Read [Building a voice app](#building-a-voice-app) at the end.

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
| 400 | `dataset_not_found` / `knowledge_off` | A dataset name or id that doesn't exist, or the knowledge base is off |
| 400 | `agent_not_found` | No Agent with that name or id |
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
2. Then the answer **as the model writes it**, one chunk per piece. On spark-a the first piece arrives about 0.15–0.25 s after the request.
3. `finish_reason: "stop"` and `data: [DONE]` close the stream.

Some turns aren't written by the chat model at all. A turn that needs tools goes to the agent; a shortcut (weather, a reminder) answers directly. Such a turn sends `: keep-alive` comments every 15 s while it works, and then its result arrives in one chunk. If the model wrote a few words before handing a turn to the agent, the result follows those words after a blank line. A stream waits up to 30 minutes.

**`"voice": true`** says the answer will be read aloud, by an app's own TTS or the LiveKit agent. OpenTeddy then answers in one to three plain spoken sentences in the person's language, with no lists, markdown, emoji or links. Such a turn also skips long-term memory lookup, which saves 120 ms of the first word; the conversation's recent turns still come along.

Text only: image parts in `content` are ignored. `temperature`, `max_tokens` and similar fields are accepted and ignored; OpenTeddy picks its own models.

## `GET /v1/models`

Returns a list with one model, `openteddy`.

## Roles: who OpenTeddy plays

By default OpenTeddy answers as a helpful assistant. A conversation can be given a **role** instead, and then every reply speaks as that role, in the first person.

The main use is training. In new-staff training, OpenTeddy plays a brand's customer: it asks what a real customer would ask, pushes back on price, and answers seriously. It does not answer the way an assistant would, with lines like 「很高興您覺得不錯，請問您對我們哪一項服務比較感興趣呢」.

There are two ways to give a role, and they can be combined:

- **`agent`**: the name or id of an **Agent** defined in OpenTeddy's web UI (Agent tab). Its persona is the role, and its 知識庫 setting picks the datasets it answers from (see [Datasets](#datasets-which-knowledge-to-answer-from)). This suits fixed roles used again and again, such as five typical customers. Edit the Agent and its conversations change, with no app release.
- **`instructions`**: a role the app writes itself, for example a scenario made up for this one session. It is kept for the whole conversation until the app changes it; `""` removes it.

With both, the Agent's persona comes first and the instructions add to it ("this time the customer wants a discount").

| Where | How |
|---|---|
| Chat | Body fields `agent` and `instructions`. `agent` only applies when starting a conversation (no `session_id`); `instructions` can be sent with any turn. With the OpenAI SDK, put them in `extra_body`. |
| Realtime voice | Put `agent` and `instructions` in the hello; `ready` names the agent and says `"instructions": true`. To change the role mid-call, send `{"type": "instructions", "instructions": "…"}`. |
| LiveKit | Room metadata `{"agent": "…", "instructions": "…"}`, next to `datasets`. |

Writing a good role:

- Say **who** it is, **what** it wants, and **how** it talks. For example: 「你是連鎖烘焙品牌的採購王經理，正在比較三家包材供應商。你在意單價和交期，說話直接、不客套。你是客戶，不是業務：不要推銷、不要問對方想了解哪項服務；就你的立場認真回應，該追問就追問。」
- A role that plays a **customer** shouldn't answer from the brand's knowledge base. Set its Agent's 知識庫 to 不使用知識庫, or send `"datasets": []`; either way the whole conversation stays off the knowledge base. A role that speaks for the brand answers from the brand's datasets instead. See [Datasets](#datasets-which-knowledge-to-answer-from).
- The length and plain style of a spoken answer still apply with `"voice": true`.

```python
# The Agent 品牌客戶・王經理 is set to 不使用知識庫 in the Agent tab, so "datasets" isn't needed.
r = client.chat.completions.create(model="openteddy", messages=[{"role": "user", "content": "王經理您好，我是業務小林。"}],
    extra_body={"agent": "品牌客戶・王經理", "instructions": "這次他很趕交期。", "voice": True})
```

An unknown agent gives `agent_not_found`, and no conversation is started.

## Datasets: which knowledge to answer from

The knowledge base holds the owner's reference material: FAQs, product notes, policies. It is organised in named **datasets**, made in OpenTeddy's 知識庫 tab, and it must be switched on there.

When a conversation names datasets, OpenTeddy answers from what they say and says so when they don't cover the question.

**List them** so the app can offer a choice:

```bash
curl https://api.example.com/v1/datasets -H "Authorization: Bearer $OPENTEDDY_KEY"
```

```json
{"object": "list", "enabled": true, "data": [
  {"id": "k3f9…", "name": "客服 FAQ", "description": "退換貨、運費", "items": 120, "always": false},
  {"id": "p81c…", "name": "產品手冊", "description": "", "items": 640, "always": true}]}
```

The list gives names and sizes, not contents. `enabled: false` means the knowledge base is off.

**Name them** with `datasets`, a list of names or ids:

- **In chat:** `datasets` in the body. With the OpenAI SDK, use `extra_body={"datasets": ["客服 FAQ"]}`.
- **In realtime voice:** `datasets` in the hello. To change them mid-conversation, send `{"type": "datasets", "datasets": [...]}`.
- **Through LiveKit:** `datasets` in the room's metadata (`contrib/livekit/README.md`).

A conversation **keeps** the datasets it named. Later turns use them until the app names others; there is no need to send them every time.

| The app sends | The conversation answers from |
|---|---|
| `["客服 FAQ", "產品手冊"]` | exactly these datasets, from now on |
| `[]` | **no** knowledge base at all, from now on, "always" datasets included |
| nothing, ever, with an `agent` whose 知識庫 is set | the Agent's datasets (or none, when the Agent is set to 不使用知識庫) |
| nothing, ever | the datasets the owner marked "always" (`always: true`) |

The app's own `datasets` always win over the Agent's. The answer (chat) or `ready` (voice) lists the datasets in effect, by name; `[]` there means none.

Datasets and [roles](#roles-who-openteddy-plays) go together: a role that **speaks for the brand** (a support agent, a sales rep) answers from the brand's datasets, while a role that **plays a customer** answers from none, because a customer doesn't know the brand's FAQ. The simplest way is to set this on the Agent, in the Agent tab under 知識庫:

- 預設: the datasets marked "always", as in any conversation;
- 只用下面勾選的資料集: exactly the ones ticked;
- 不使用知識庫: none.

Then the app only sends `agent`. An app that builds its roles with `instructions` sends `datasets` itself.

```python
r = client.chat.completions.create(
    model="openteddy",
    messages=[{"role": "user", "content": "可以退貨嗎？運費誰出？"}],
    extra_body={"datasets": ["客服 FAQ"]})
print(r.choices[0].message.content, r.model_extra["datasets"])   # … ['客服 FAQ']
```

A name that doesn't exist gives `dataset_not_found`, and the conversation isn't started. Naming datasets while the knowledge base is off gives `knowledge_off`.

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

## Speech to text as it is spoken: `WS /v1/audio/stt-stream`

The app or agent sends audio as it records. It gets back the words heard so far (`partial`), and the whole utterance (`final`) once it ends. The key works as for `/v1/realtime`: a Bearer header, or the subprotocols `openteddy` and `openteddy-key.<key>`.

| Direction | Message |
|---|---|
| → | **binary**: PCM16 mono 16 kHz, in small chunks (20–40 ms) |
| → | `{"type": "config", "endpoint_ms": 600, "interim": true}`. Optional; see below. |
| → | `{"type": "flush"}`: the utterance has ended, send the final now |
| → | `{"type": "reset"}`: drop what was heard so far |
| ← | `{"type": "speech_started"}`: sound loud enough to be speech |
| ← | `{"type": "partial", "text": "請問你可以"}`: the utterance so far, about every 300 ms while the person talks |
| ← | `{"type": "final", "text": "請問你可以幫我做什麼？"}` |

An utterance ends in one of two ways:

- **The app says so with `flush`.** This is what LiveKit's VAD does, and the final comes about 0.04 s later. A flush always gets a final; it is empty if nothing was said.
- **OpenTeddy hears enough silence.** Set `endpoint_ms`, for example 600 ms. The default is 0, which means only `flush` ends an utterance.

SenseVoice has no streaming mode of its own. A partial is the utterance so far transcribed again; the final is the whole utterance transcribed once more. Silence before speech starts isn't kept, except the last 0.5 s.

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
| → first | `{"type": "hello", "session_id": "…", "datasets": [...], "agent": "…", "instructions": "…"}`. Leave `session_id` out (or `""`) for a new conversation. Everything except `type` is optional; see [Roles](#roles-who-openteddy-plays). |
| ← | `{"type": "ready", "session_id", "sample_rate", "asr_rate": 16000, "voice", "asr", "tts", "datasets", "agent", "instructions"}` |
| → | **binary**: one utterance, PCM16 mono at 16 kHz, at most 45 s. The app decides where speech ends; the web UI waits for 0.5 s of silence. |
| → | `{"type": "text", "text": "…"}`: a typed turn instead of a spoken one |
| → | `{"type": "datasets", "datasets": [...]}`: answer from these from now on. The reply is ← `{"type": "datasets", "datasets": [names]}`. |
| → | `{"type": "instructions", "instructions": "…"}`: play this role from the next turn on (`""` removes it). The reply is ← `{"type": "instructions", "set": true}`. |
| → | `{"type": "interrupt"}`: the person started talking, so stop speaking |
| ← | `{"type": "state", "value": "listening" \| "thinking" \| "speaking"}` |
| ← | `{"type": "transcript", "text", "ms"}`: what was heard |
| ← | `{"type": "reply", "text", "task_id"}`: the answer as text, once it is all written (usually while its first sentence is being spoken) |
| ← | **binary**: the spoken answer, PCM16 mono at `sample_rate`, in chunks as it is made. Speaking starts with the answer's first sentence while the rest is still being written. `{"type": "audio_end"}` follows (`"interrupted": true` when it was cut short). |
| ← | `{"type": "working", "task_id", "text"}`: work that takes longer than 8 s is acknowledged out loud … |
| ← | `{"type": "done", "task_id", "text"}`: … and then spoken when it is done |
| ← | `{"type": "error", "code", "text"}`. `code` is one of `not_enabled`, `not_configured`, `unreachable`, `loading`, `bad_session`, `dataset_not_found`, `knowledge_off` or `agent_not_found`. |

Noise that isn't speech starts nothing: the voice service checks for a voice first.

The audio of the conversation travels between the app and OpenTeddy over the tunnel. Between OpenTeddy and its voice service it stays on the local network.

## Building a voice app

**For the quickest answers, use LiveKit.** LiveKit carries the audio over WebRTC, with echo cancellation and barge-in. OpenTeddy's agent in `contrib/livekit/` connects the three streams above to it:

- `stt-stream` for listening,
- chat with `stream` and `"voice": true` for the answer,
- `audio/speech` for speaking, one clause at a time.

On the DGX Spark pair, the first sound comes about 1.6 s after the person stops talking (median of 5 calls, 11/24). The app only needs LiveKit's SDK and a room token; see `contrib/livekit/README.md`.

Without LiveKit there are two ways to build it. Pick one.

| | **A. Realtime** (`/v1/realtime`) | **B. Three HTTP calls** |
|---|---|---|
| Flow | One WebSocket carries the whole conversation | transcriptions → chat/completions → speech |
| First sound | About 1 s after the utterance arrives (0.97 s on spark-a); speaking starts at the answer's first sentence | Later: the whole answer is written, then synthesised, before it plays |
| Interrupting OpenTeddy | Yes (`interrupt`) | Stop playback yourself |
| Effort | Audio in and out as PCM | Plain HTTP with files |

**A** is what the web UI's 🎙 uses. **B** is simpler when an app already records files and only needs question → spoken answer.

### A. Realtime, step by step

1. **Connect** to `wss://<host>/v1/realtime` with the key. Send the hello with any `datasets`. Wait for `ready`.
2. **Keep** `ready.session_id` to resume the same conversation later (put it in the next hello), and `ready.sample_rate` for playback.
3. **Record** 16 kHz mono 16-bit PCM. Android `AudioRecord` gives this directly. On iOS, `AVAudioEngine` and a converter to 16 kHz Int16 do it. A browser can create an `AudioContext({sampleRate: 16000})`.
4. **Decide where an utterance ends.** **Push-to-talk** is the easiest: record while a button is held. **Hands-free** listening ends an utterance after about 0.5 s below a noise floor. Send each utterance as **one binary message**.
5. **Show** `transcript`, then `reply`. **Play** the binary chunks back to back at `sample_rate` until `audio_end`. The audio starts with the answer's first sentence, usually **before** `reply` arrives, so don't wait for `reply` to start playing.
6. When the person starts talking over the answer, send `interrupt` and drop the queued audio.
7. A `working` message means longer work is under way. Keep listening; `done` arrives with the result later, and it is spoken too.

**Browser (JavaScript), push-to-talk:**

```js
const KEY = "ot-…";              // see "Keeping the key safe" below
const ws = new WebSocket("wss://api.example.com/v1/realtime", ["openteddy", "openteddy-key." + KEY]);
ws.binaryType = "arraybuffer";
const out = new AudioContext();
let rate = 24000, playAt = 0, playing = [];

ws.onopen = () => ws.send(JSON.stringify({ type: "hello", datasets: ["客服 FAQ"] }));
ws.onmessage = (ev) => {
  if (typeof ev.data !== "string") return play(new Int16Array(ev.data));
  const m = JSON.parse(ev.data);
  if (m.type === "ready") rate = m.sample_rate;          // keep m.session_id to resume later
  else if (m.type === "transcript") console.log("you:", m.text);
  else if (m.type === "reply" || m.type === "done") console.log("OpenTeddy:", m.text);
  else if (m.type === "error") console.warn(m.code, m.text);
};

function play(pcm) {                                       // PCM16 → queued audio
  const buf = out.createBuffer(1, pcm.length, rate), ch = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
  const src = out.createBufferSource();
  src.buffer = buf; src.connect(out.destination);
  playAt = Math.max(playAt, out.currentTime); src.start(playAt); playAt += buf.duration;
  playing.push(src); src.onended = () => playing = playing.filter(s => s !== src);
}

let rec;
async function startTalking() {                            // button pressed
  ws.send(JSON.stringify({ type: "interrupt" }));          // OpenTeddy stops talking …
  playing.forEach(s => s.stop()); playing = []; playAt = 0; // … and so does the queue
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const ctx = new AudioContext({ sampleRate: 16000 });     // the browser resamples to 16 kHz
  const node = ctx.createScriptProcessor(4096, 1, 1), chunks = [];
  node.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  ctx.createMediaStreamSource(stream).connect(node); node.connect(ctx.destination);
  rec = { stream, ctx, chunks };
}
function stopTalking() {                                   // button released: one utterance
  rec.stream.getTracks().forEach(t => t.stop()); rec.ctx.close();
  const n = rec.chunks.reduce((a, c) => a + c.length, 0), pcm = new Int16Array(n);
  let i = 0;
  for (const c of rec.chunks) for (const s of c) pcm[i++] = Math.max(-1, Math.min(1, s)) * 32767;
  ws.send(pcm.buffer);
}
```

The page must be served over https (or from localhost), or the browser won't give it the microphone.

**Python, one question from a WAV file** (16 kHz mono 16-bit), useful for testing:

```python
import asyncio, json, wave
import websockets                     # pip install websockets

KEY, URL = "ot-…", "wss://api.example.com/v1/realtime"

async def ask(path):
    async with websockets.connect(URL, additional_headers={"Authorization": f"Bearer {KEY}"}) as ws:
        await ws.send(json.dumps({"type": "hello", "datasets": ["客服 FAQ"]}))
        ready = json.loads(await ws.recv())
        with wave.open(path) as w:
            await ws.send(w.readframes(w.getnframes()))
        audio = bytearray()
        while True:
            m = await ws.recv()
            if isinstance(m, bytes):
                audio += m
                continue
            d = json.loads(m)
            if d["type"] in ("transcript", "reply", "error"):
                print(d["type"], d.get("text"))
            if d["type"] in ("audio_end", "error"):
                break
        with wave.open("answer.wav", "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(ready["sample_rate"]); w.writeframes(audio)

asyncio.run(ask("question.wav"))
```

### B. Three HTTP calls

```python
from openai import OpenAI
client = OpenAI(base_url="https://api.example.com/v1", api_key="ot-…")

heard = client.audio.transcriptions.create(model="sensevoice", file=open("question.wav", "rb")).text
r = client.chat.completions.create(model="openteddy", messages=[{"role": "user", "content": heard}],
                                   extra_body={"datasets": ["客服 FAQ"]})     # + "session_id" to go on
answer, session = r.choices[0].message.content, r.model_extra["session_id"]
with client.audio.speech.with_streaming_response.create(model="cosyvoice", voice="default",
                                                         input=answer, response_format="wav") as s:
    s.stream_to_file("answer.wav")
```

Send WAV to transcriptions: other formats need ffmpeg on the OpenTeddy machine. A long answer can be read only in part: the app decides how much of `answer` to speak.

### Keeping the key safe

Anyone who has the key can use OpenTeddy as the app does.

- An app that only **your own team** uses (an internal tool, a kiosk) can carry a key. Give each app its own key, so revoking one doesn't stop the others.
- An app **customers install**, or a public web page, should not carry the key: it can be read out of the app. Put a small server of your own in between. The app logs in to your server, and your server calls OpenTeddy with the key, forwarding the WebSocket for voice.
