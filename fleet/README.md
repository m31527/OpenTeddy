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

There's also a web console at `http://<central>:8000/fleet` with three
tabs: **Workers** (live node status), **Playground** (type a goal → it
auto-picks an idle worker, no manual selection), and **Alerts** (the
proactive reports below).

## Proactive alerts (the watcher)

Dispatch is reactive — you ask, a node answers. The watcher is the
proactive half: each worker periodically self-checks its area of
responsibility and pushes an **alert** to the central *without being
asked*. Alerts land in the console's Alerts tab and (if Telegram is
configured) get pushed to your phone.

Enable per worker via its `.env`:

```bash
OPENTEDDY_FLEET_WATCH_ENABLED=1
OPENTEDDY_FLEET_WATCH_INTERVAL=900          # seconds; 120 while testing
OPENTEDDY_FLEET_WATCH_PROMPT=<what THIS node should check>
```

The prompt is the node-specific part — describe what "abnormal" means
for this node's data + tools. The watcher appends a machine-readable
`ANOMALY / SEVERITY / CONFIDENCE` block automatically, so you only write
the "what to look at" half. Role examples:

| Role | Example `OPENTEDDY_FLEET_WATCH_PROMPT` |
|---|---|
| finance | 檢查 /data/finance 最近一小時是否有異常大額付款、首次出現的收款方、批次資料竄改 |
| secops | 檢查 auth.log 最近一小時是否有暴力登入、異常地理位置登入、權限提升 |
| sys-health | 檢查磁碟 / 記憶體 / docker / ollama 是否接近滿載或已停止 |
| external-intel | 查最近一小時與本公司/產業相關的重大新聞、資安通報、輿情變化 |

How a cycle works:

```
每隔 INTERVAL 秒（±15% jitter）
  → worker 用自己的 orchestrator.run() 跑 WATCH_PROMPT
    → 模型回答，結尾帶 ANOMALY: yes/no · SEVERITY · CONFIDENCE
      → ANOMALY: yes  → push alert 給中控 → Alerts tab + Telegram
      → ANOMALY: no   → 靜默，不打擾
```

Leaving `OPENTEDDY_FLEET_WATCH_PROMPT` unset uses a generic placeholder
that mostly reports "no anomaly" — useful to prove the pipeline, useless
for real monitoring. Write the real prompt before relying on it.

See alerts via the console Alerts tab or:

```bash
curl -s http://localhost:8000/fleet/alerts | python3 -m json.tool
```

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
