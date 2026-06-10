# OpenTeddy Fleet — operator guide

Distributed-orchestration layer: one central node dispatches goals to
worker nodes; each worker runs the goal on its own OpenTeddy
orchestrator and reports back. Off by default — single-machine installs
never touch this. Full design: [docs/fleet-architecture.md](../docs/fleet-architecture.md).

## Try it first — single-machine demo (no second machine needed)

Before wiring up real nodes, prove the fleet plumbing works on one box:

```bash
bash scripts/fleet-demo.sh
```

It spins up an in-process central + worker over a real localhost
WebSocket, registers, dispatches a goal, and prints the result. A
`🎉 fleet demo PASSED` line means register → dispatch → run → result all
work — then real multi-node is just the `.env` files below.

## Enabling a real fleet

Every node is a normal OpenTeddy install. You turn it into a fleet node
by adding env vars to its `.env` (read at startup) — nothing else
changes. There are **ready-made templates** so you don't hand-write them:

| Node | Command |
|---|---|
| Central (exactly one) | `cat fleet/env.orchestrator.example >> .env` |
| Each worker | `cat fleet/env.worker.example >> .env` |

### Step 1 — generate ONE shared token

On any machine, once:

```bash
openssl rand -hex 32
```

Copy the output. It goes into `OPENTEDDY_FLEET_TOKEN` on **every** node —
identical value. It's how nodes trust each other; without it the central
refuses to start and workers are rejected (fail-closed).

### Step 2 — central node (exactly one)

```bash
cat fleet/env.orchestrator.example >> .env
# then edit .env: paste the token into OPENTEDDY_FLEET_TOKEN
```

Restart OpenTeddy. The `.env` block sets:
`OPENTEDDY_FLEET_ROLE=orchestrator` + `OPENTEDDY_FLEET_PORT=8770`.

### Step 3 — each worker node

```bash
cat fleet/env.worker.example >> .env
# then edit .env:
#   OPENTEDDY_FLEET_TOKEN     ← the same token from step 1
#   OPENTEDDY_FLEET_CENTRAL   ← ws://<central-host-or-ip>:8770
#   OPENTEDDY_FLEET_NODE_ID   ← a name for this node, e.g. dgx-02
#   OPENTEDDY_FLEET_NODE_ROLE ← this node's job, e.g. finance
```

Restart OpenTeddy. The worker auto-dials the central + waits for tasks.

### Step 4 — restart + confirm

After editing each `.env`, restart that node's backend:

```bash
pkill -f "uvicorn.*main:app"; sleep 1
nohup .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/openteddy.log 2>&1 &
```

Then on the **central**, check the fleet came together:

```bash
curl -s http://localhost:8000/fleet/nodes | python3 -m json.tool
```

## Operating the fleet (from the central node)

List connected nodes:

```bash
curl -s http://localhost:8000/fleet/nodes | python3 -m json.tool
```

```json
{
  "fleet_enabled": true,
  "nodes": [
    {"node_id": "dgx-02", "role": "finance", "online": true,
     "model": "qwen2.5:7b", "status": "idle", "last_seen_s": 3.1, ...}
  ]
}
```

Dispatch a goal to a specific node and get its result:

```bash
curl -s -X POST http://localhost:8000/fleet/dispatch \
  -H 'Content-Type: application/json' \
  -d '{"node_id":"dgx-02","goal":"整理本月財務異常付款 top 10","mode":"code"}' \
  | python3 -m json.tool
```

On a non-orchestrator node these endpoints return `fleet_enabled:false`
(for `/fleet/nodes`) or HTTP 409 (for `/fleet/dispatch`).

## Network + security notes

- The central's WS port (`OPENTEDDY_FLEET_PORT`) binds `0.0.0.0` so
  workers across the LAN can reach it. Put the fleet behind a private
  mesh (WireGuard / Tailscale) or a firewall — the shared token is
  access control, not transport encryption.
- The token authenticates the `register` handshake; subsequent frames on
  an authed connection are trusted. mTLS is the planned upgrade
  (milestone 5) and the token check is funnelled through one function
  (`protocol.verify_token`) to keep that change localized.
- Dispatched goals run with full executor capability on the worker (the
  same `shell_exec_write` / `python_exec` a local user has). Only
  enable workers on machines + a network you trust.

## What runs where

| | Central (orchestrator) | Worker |
|---|---|---|
| Opens WS server | ✓ :8770 | — |
| Dials central | — | ✓ |
| Runs dispatched goals on local `orchestrator.run()` | — | ✓ |
| Holds node registry + routes dispatch | ✓ | — |
| Owns its own model / tools / data | ✓ (its own) | ✓ (its own) |

Each node is a full, independent OpenTeddy. Fleet only adds the thin
"who runs what" coordination on top.

## Disabling

Unset `OPENTEDDY_FLEET_ROLE` (or set it to `none`) and restart. The
`fleet/` package stops being imported entirely — back to a plain
single-machine install.
