# OpenTeddy Fleet — operator guide

Distributed-orchestration layer: one central node dispatches goals to
worker nodes; each worker runs the goal on its own OpenTeddy
orchestrator and reports back. Off by default — single-machine installs
never touch this. Full design: [docs/fleet-architecture.md](../docs/fleet-architecture.md).

## Enabling a fleet

Every node is a normal OpenTeddy install. You turn it into a fleet node
with environment variables — nothing else changes.

### Shared token (set on EVERY node, same value)

```bash
export OPENTEDDY_FLEET_TOKEN="<a long random shared secret>"
```

Without this, the orchestrator refuses to start and workers are rejected
(fail-closed). Generate one with `openssl rand -hex 32`.

### Central node (exactly one)

```bash
export OPENTEDDY_FLEET_ROLE=orchestrator
export OPENTEDDY_FLEET_PORT=8770        # WS port workers dial (default 8770)
# start OpenTeddy normally — the orchestrator starts with it
```

### Worker nodes (the rest)

```bash
export OPENTEDDY_FLEET_ROLE=worker
export OPENTEDDY_FLEET_CENTRAL=ws://<central-host>:8770
export OPENTEDDY_FLEET_NODE_ID=dgx-02            # defaults to hostname
export OPENTEDDY_FLEET_NODE_ROLE=finance         # free-form role label
# start OpenTeddy normally — the worker dials the central + waits for tasks
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
