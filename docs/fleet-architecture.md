# OpenTeddy Fleet Architecture (RFC)

> Status: **draft / building** — MVP milestone 1 (central dispatch).
> Branch: `feat/fleet-orchestration`.

## What this is

A thin coordination layer that lets multiple OpenTeddy nodes act as one
distributed "AI brain cluster": a central node dispatches goals to worker
nodes, each worker runs the goal on its own existing single-machine
orchestrator, and results flow back. Built for the 5–10 node NVIDIA DGX
Spark fleet case, but works for any set of OpenTeddy installs that can
reach each other on a network.

## The non-negotiable: single-machine installs are unaffected

Fleet is an **opt-in additive layer, off by default**. The same
philosophy as Telegram bridge / vLLM / chrome_attached — a feature that
sleeps until explicitly enabled.

```
OPENTEDDY_FLEET_ROLE unset (default)  → "none"
    Single-machine mode. NOTHING in fleet/ is imported. Desktop .dmg,
    personal web install, every existing user — runs exactly as today.

OPENTEDDY_FLEET_ROLE=worker
    DGX worker node. Dials the central orchestrator, receives tasks.

OPENTEDDY_FLEET_ROLE=orchestrator
    DGX central node. Opens a WS server, dispatches tasks, aggregates
    status.
```

Guarantees:

- **No import when off.** main.py gates the fleet import on
  `OPENTEDDY_FLEET_ROLE`. With role=none the `fleet/` package is never
  imported — zero memory, zero startup cost, zero new ports.
- **Existing orchestrator untouched.** Fleet never modifies
  `orchestrator.run()`. A worker that receives a task calls the SAME
  `orchestrator.run()` a local user would. Fleet is a wrapper around the
  mature single-machine agent, not a fork of it.
- **Self-contained.** All fleet code lives under `fleet/`. Deleting that
  directory + the one gate in main.py fully removes the feature.
- **UI hidden off-fleet.** Fleet settings/controls only render when
  role != none, gated like the vLLM radio is on macOS.

## Topology: hub-and-spoke

For 5–10 nodes, a central hub with N worker spokes is the right shape
(mesh / broker is over-engineering at this scale).

```
            ┌──────────────────────────────┐
   operator │   Orchestrator (1 node)      │
   ───────► │   - WS server :8770          │
            │   - node registry + heartbeat│
            │   - dispatch + result collect│
            │   - status aggregation       │
            └───────┬──────────────────────┘
                    │  WS (token-authed), persistent
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   ┌────────┐  ┌────────┐  ┌────────┐
   │worker A│  │worker B│  │worker C│   … up to ~9 spokes
   │ role:  │  │ role:  │  │ role:  │
   │finance │  │external│  │ secops │
   │ own    │  │ own    │  │ own    │
   │orchestr│  │orchestr│  │orchestr│
   └────────┘  └────────┘  └────────┘
   each = a full OpenTeddy install with its own model + tools + data
```

## Transport: WebSocket, behind an abstraction seam

WS chosen because:
- OpenTeddy already runs a WS stack (reuse, not reinvent).
- Real-time matters for heartbeat + (later) proactive alerts; HTTP
  polling adds latency + waste.
- A persistent connection makes the worker→central push path (alerts)
  trivial later — same socket, reverse direction.

The message layer is abstracted (`fleet/protocol.py`) so a future swap
to NATS / gRPC for larger scale doesn't touch orchestration logic.

## Auth: shared bearer token (MVP)

Each node carries a token (`OPENTEDDY_FLEET_TOKEN`); the orchestrator
validates it on connect. Rationale:
- mTLS is the right end-state but heavy for MVP (cert gen / rotation /
  distribution × N nodes).
- "No auth" is unsafe even on a LAN — anyone who reaches the WS port
  could dispatch shell-capable tasks.
- Shared token is the 80/20: blocks casual access, trivial to ship, and
  the auth check is abstracted so upgrading to mTLS later is localized.
- Composes with the WireGuard / Tailscale private-mesh future: token
  over an already-encrypted overlay is a sane layering.

## Message protocol (v0)

All messages are JSON objects with a `type` discriminator. Token travels
in the initial `register` only; subsequent frames are trusted on the
authed connection.

Worker → Orchestrator:
```
{type:"register",  node_id, role, token, capabilities:{tools:[...], model}}
{type:"heartbeat", node_id, ts, status:"idle"|"busy", load:{...}}
{type:"result",    task_id, node_id, status:"completed"|"failed", summary, artifacts:[...]}
{type:"alert",     node_id, severity, observation, confidence}   # milestone 2
```

Orchestrator → Worker:
```
{type:"register_ack", assigned, server_version}
{type:"dispatch",     task_id, goal, mode, context:{...}}
{type:"cancel",       task_id}
```

## Node lifecycle

```
worker boot → dial central WS → send register(token)
   ↓ register_ack
worker idle → heartbeat every 15s
   ↓ dispatch arrives
worker busy → orchestrator.run(goal) → send result
   ↓ back to idle
central loses 3 consecutive heartbeats → mark node offline
```

## Milestones

| # | Deliverable | Status |
|---|---|---|
| 1 | Central dispatches a goal to one worker; result returns. Includes the connection layer (register / heartbeat / token). | **building** |
| 2 | Worker watcher loop → proactive alerts pushed to central. | planned |
| 3 | Central status dashboard (fleet health, per-node load, live task feed). | planned |
| 4 | Multi-node routing (central LLM picks which worker(s) by role). | planned |
| 5 | mTLS, operator auth + audit log, WireGuard onboarding. | planned |

## File layout

```
fleet/
├── __init__.py        # empty; only imported when role != none
├── protocol.py        # message schema + token validation + (de)serialise
├── orchestrator.py    # central: WS server, registry, dispatch, collect
├── worker.py          # worker: dial central, receive, run, report
└── README.md          # operator-facing setup
docs/
└── fleet-architecture.md   # this file
```

## What fleet explicitly does NOT do

- Does not change `orchestrator.run()` / `executor` / planner.
- Does not run on single-machine installs (role=none → never imported).
- Does not share models across nodes — each node owns its inference.
- Does not centralise data — workers hold their own data; central holds
  only routing + audit + operator conversation state.
