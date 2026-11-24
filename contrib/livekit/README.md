# OpenTeddy voice over LiveKit

This is the LiveKit voice integration spec (`docs/LiveKit 語音整合規格.md`) built on top of OpenTeddy.

LiveKit carries the audio between the app and the agent: WebRTC, echo cancellation, jitter buffering, turn-taking and barge-in. OpenTeddy listens, answers and speaks. It keeps the CosyVoice voice, SenseVoice and the knowledge base.

```
App (livekit_client) ──WebRTC──▶ LiveKit Server ◀──▶ agent.py ──▶ OpenTeddy /v1
                                                     Silero VAD     ├ WS   /audio/stt-stream   SenseVoice: partial / final
                                                     turn-taking    ├ POST /chat/completions   stream, voice: word by word, KB + session + datasets
                                                     barge-in       └ POST /audio/speech       CosyVoice: one clause at a time
```

## What it measures (DGX Spark pair, 11/24)

These are times from the end of the person's speech to the first sound of the answer. They were measured with `try_call.py` (an agent-side timeline in brackets).

| | first sound |
|---|---|
| First working version | 2.82 s |
| + 20 ms audio frames, VAD 0.3 s, endpointing 0.05 s | 2.5 s |
| + the first clause goes to TTS as soon as it is written | 1.9–2.3 s |
| + spoken turns skip long-term memory (LLM first token 0.30 → 0.15 s) | **1.8–2.0 s** |
| + CosyVoice in fp16 with a 12-token first chunk (needs your OK on the voice) | about 1.6–1.7 s |

Here is where a turn's time goes on the agent side, with float32 TTS:

| Step | Time |
|---|---|
| VAD end of speech → flush | ~0.43 s |
| STT final | 0.05 s |
| LLM first token | 0.15 s |
| First clause written | 0.15–0.25 s |
| TTS first audio | 0.75 s (0.53 s with fp16) |
| WebRTC to the app | ~0.2 s |

The spec's target is under 1.5 s. What is left to win:

- **A semantic turn detector** (`livekit-plugins-turn-detector`, multilingual). It would allow a shorter VAD silence without cutting people off.
- **A faster CosyVoice.** Its vLLM path, or TensorRT, for the flow.

## Run it

### 1. LiveKit Server (spark B, Docker)

`~/livekit/livekit.yaml` holds the API key and secret, mode 600:

```yaml
port: 7880
bind_addresses: ["127.0.0.1", "192.168.100.2", "10.255.128.99"]   # local, the Spark link, NetBird
rtc:
  tcp_port: 7881
  port_range_start: 50000
  port_range_end: 50200
  use_external_ip: false
  interfaces:
    includes: ["enp1s0f0np0", "wt0", "lo"]
keys:
  <API key>: <secret>
```

```bash
docker run -d --name livekit --restart unless-stopped --network host \
  -v ~/livekit/livekit.yaml:/etc/livekit.yaml:ro livekit/livekit-server:v1.13.11 --config /etc/livekit.yaml
```

### 2. The agent (spark B, Docker)

Create `~/livekit-agent/agent.env` with mode 600:

```
LIVEKIT_URL=ws://127.0.0.1:7880
LIVEKIT_API_KEY=…
LIVEKIT_API_SECRET=…
OPENTEDDY_URL=http://192.168.100.1:8000/v1     # spark A over the link
OPENTEDDY_KEY=ot-…                              # its own key: Settings → 🔑 API 金鑰
```

```bash
docker build -t openteddy-livekit-agent contrib/livekit
docker run -d --name openteddy-livekit-agent --restart unless-stopped --network host \
  --env-file ~/livekit-agent/agent.env openteddy-livekit-agent
```

The agent joins every new room. Each room is one OpenTeddy session, so OpenTeddy remembers the conversation for as long as the room lasts. The agent's conversations show in OpenTeddy's session list as `🔑 <key name> · 語音對話`.

You can tune it with these environment variables:

| Variable | Default | What it controls |
|---|---|---|
| `VAD_SILENCE_S` | 0.3 | Silence that ends a turn. Lower is quicker, but cuts more pauses. |
| `ENDPOINT_MIN_S` | 0.05 | Extra wait after the VAD before the turn is committed. |
| `OPENTEDDY_TTS_RATE` | 24000 | CosyVoice 3's output sample rate. |
| `AGENT_IDLE_PROCESSES` | 1 | Warm worker processes. Add one for each call you expect at the same time. |

Each turn logs a timeline: flush, final, LLM request and first piece, TTS request and first audio. It logs character counts, never the words.

### 3. Try a call

```bash
set -a; . ~/livekit-agent/agent.env; set +a
python try_call.py --runs 3 [--datasets 客服FAQ]
```

`try_call.py` joins a room as a person, says a question into the microphone track, and times the first sound of the answer.

## For the app team (Flutter, `livekit_client`)

1. Get a room token from your backend (next section), never from the app itself.
2. Connect: `Room().connect(url, token)`.
3. Publish the microphone: `room.localParticipant.setMicrophoneEnabled(true)`.
4. Play the agent's audio track. The SDK subscribes automatically.
5. Show the transcripts, if you want them. The person's words arrive on the `lk.transcription` text stream as they are recognised; this was seen in `try_call.py`. The agent's own answer text did **not** arrive there in our tests. That is open: the agent can publish it on a topic of its own if the app needs it.

Nothing else is needed. There is no OpenTeddy protocol to speak and no PCM to handle. Barge-in works by itself: talk over the agent and it stops.

## For the backend: room tokens

The app logs in with Firebase as it does now. Your backend checks the ID token, then signs a LiveKit token, for example with `livekit-api` (Python):

```python
from livekit import api
token = (api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
         .with_identity(firebase_uid)
         .with_grants(api.VideoGrants(room_join=True, room=f"call-{firebase_uid}-{uuid4().hex[:6]}"))
         .to_jwt())
```

**Datasets.** Create the room with metadata `{"datasets": ["客服 FAQ"]}` (`RoomService.CreateRoom`), or give the participant an attribute `datasets` holding a JSON list. The agent passes these datasets to OpenTeddy, and the room's conversation answers from them. The knowledge base must be turned on in OpenTeddy (知識庫).

The LiveKit secret and the OpenTeddy key stay on servers.

## For DevOps: reaching it from outside

WebRTC media is UDP, so it **can't go through the Cloudflare tunnel** that carries `lulu.openteddy.net`. For apps outside the office there are two options:

- **Expose LiveKit.** Give it a public address, either through UniFi port forwarding (TCP 7880 behind TLS, TCP 7881, UDP 50000–50200) or with its built-in TURN/TLS on 443. Set `rtc.use_external_ip: true`, and put a domain with a certificate on the signalling port.
- **Use LiveKit Cloud for the media** while the agent and OpenTeddy stay on the Sparks. The agent connects outbound, so nothing has to be opened. This is the quickest way to try a proof of concept from anywhere.
