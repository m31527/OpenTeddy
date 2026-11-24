# OpenTeddy voice over LiveKit

This is the LiveKit voice integration spec (`docs/LiveKit 語音整合規格.md`) built on top of OpenTeddy.

LiveKit carries the audio between the app and the agent: WebRTC, echo cancellation, jitter buffering, turn-taking and barge-in. OpenTeddy listens, answers and speaks. It keeps the CosyVoice voice, SenseVoice and the knowledge base.

```
App ──POST /v1/livekit/token (its ot- key)──▶ OpenTeddy: conversation + room + agent dispatched → {url, token}
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
| + spoken turns skip long-term memory (LLM first token 0.30 → 0.15 s) | 1.8–2.0 s |
| + CosyVoice in fp16 with a 12-token first chunk (`TTS_FP16=1 TTS_FIRST_HOP=12`) | **1.54–1.86 s, median 1.58 s** |

Here is where a turn's time goes on the agent side:

| Step | Time |
|---|---|
| VAD end of speech → flush | ~0.43 s |
| STT final | 0.05 s |
| LLM first token | 0.15 s |
| First clause written | 0.15–0.25 s |
| TTS first audio | 0.75 s float32 → 0.38–0.44 s with fp16 (live on spark B since 11/24) |
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
OPENTEDDY_KEY=ot-…                              # optional: only for rooms made without OpenTeddy (below)
```

```bash
docker build -t openteddy-livekit-agent contrib/livekit
docker run -d --name openteddy-livekit-agent --restart unless-stopped --network host \
  --env-file ~/livekit-agent/agent.env openteddy-livekit-agent
```

The agent joins the rooms it is **sent** to (LiveKit's explicit dispatch, under the name `AGENT_NAME`, default `openteddy`), not every room. Each room is one OpenTeddy conversation.

You can tune it with these environment variables:

| Variable | Default | What it controls |
|---|---|---|
| `VAD_SILENCE_S` | 0.3 | Silence that ends a turn. Lower is quicker, but cuts more pauses. |
| `ENDPOINT_MIN_S` | 0.05 | Extra wait after the VAD before the turn is committed. |
| `OPENTEDDY_TTS_RATE` | 24000 | CosyVoice 3's output sample rate. |
| `AGENT_IDLE_PROCESSES` | 1 | Warm worker processes. Add one for each call you expect at the same time. |
| `AGENT_NAME` | openteddy | The name OpenTeddy dispatches it by (`LIVEKIT_AGENT_NAME` on OpenTeddy's side). |
| `AGENT_JOIN_WAIT_S` | 120 | How long it waits in a room for the caller before it leaves. |

Each turn logs a timeline: flush, final, LLM request and first piece, TTS request and first audio. It logs character counts, never the words.

### 3. OpenTeddy (spark A)

OpenTeddy opens the rooms. Add these to its `.env` and restart it:

```
LIVEKIT_URL=wss://lk.example.com              # what apps connect to (the Spark pair: ws://10.255.128.99:7880, NetBird)
LIVEKIT_API_URL=http://192.168.100.2:7880     # where OpenTeddy reaches LiveKit's server API
LIVEKIT_API_KEY=…                              # the same pair as in livekit.yaml
LIVEKIT_API_SECRET=…
```

### 4. Try a call

```bash
OPENTEDDY_URL=http://127.0.0.1:8100/v1 OPENTEDDY_KEY=ot-… \
  python try_call.py --runs 3 [--agent 品牌客戶・王經理] [--datasets 客服FAQ]
```

`try_call.py` asks for a room the way an app does, joins it as a person, says a question into the microphone track, and times the first sound of the answer. Set `LIVEKIT_URL` to join a different address than the one OpenTeddy hands out.

## For the app team (Flutter, `livekit_client`)

1. **Ask OpenTeddy for a call** with the app's OpenTeddy key, the same `ot-…` key as for the rest of the API:

   ```
   POST /v1/livekit/token
   Authorization: Bearer ot-…
   {"agent": "品牌客戶・王經理", "instructions": "這次他很趕交期。", "datasets": [], "voice": "male1", "identity": "trainee-17"}
   → {"url": "wss://…", "token": "…", "room": "ot-…", "session_id": "…", "agent": "品牌客戶・王經理", …}
   ```

   Everything in the body is optional (`docs/api.md`, "Voice through LiveKit"). Send `session_id` to go on with an earlier conversation.
2. Connect: `Room().connect(url, token)`. Join within 15 minutes.
3. Publish the microphone: `room.localParticipant.setMicrophoneEnabled(true)`.
4. Play the agent's audio track. The SDK subscribes automatically.
5. Show the transcripts, if you want them. The person's words arrive on the `lk.transcription` text stream as they are recognised; this was seen in `try_call.py`. The agent's own answer text did **not** arrive there in our tests. That is open: the agent can publish it on a topic of its own if the app needs it.

Nothing else is needed. There is no OpenTeddy protocol to speak and no PCM to handle, and no LiveKit secret anywhere near the app. Barge-in works by itself: talk over the agent and it stops.

The call is the app's own: it shows in OpenTeddy's session list as `🔑 <key name> · …`, the key's own conversation, and `session_id` continues it later in chat or in another call. The agent never holds the key. OpenTeddy hands it a grant that acts as the key in that one conversation, for at most six hours, and that ends when the key is revoked.

**The key in a customer's app.** As for the rest of the API, an app that customers install shouldn't carry the key (`docs/api.md`, "Keeping the key safe"). Its server calls `/v1/livekit/token` and passes `url` and `token` on. The token is only good for joining that one room.

## Rooms of your own

A backend that holds the LiveKit secret can still open rooms itself: create the room, then dispatch the agent (`AgentDispatchService.CreateDispatch`, agent name `openteddy`) without a grant. The agent then uses its own `OPENTEDDY_KEY`, and the room's metadata sets the conversation up, for example `{"datasets": ["客服 FAQ"], "agent": "品牌客戶・王經理", "instructions": "這次他很趕交期。", "voice": "male1"}`. Participant attributes of the same names also work; for `datasets` the attribute holds a JSON list.

## For DevOps: reaching it from outside

WebRTC media is UDP, so it **can't go through the Cloudflare tunnel** that carries `lulu.openteddy.net`. For apps outside the office there are two options:

- **Expose LiveKit.** Give it a public address, either through UniFi port forwarding (TCP 7880 behind TLS, TCP 7881, UDP 50000–50200) or with its built-in TURN/TLS on 443. Set `rtc.use_external_ip: true`, and put a domain with a certificate on the signalling port.
- **Use LiveKit Cloud for the media** while the agent and OpenTeddy stay on the Sparks. The agent connects outbound, so nothing has to be opened. This is the quickest way to try a proof of concept from anywhere.
