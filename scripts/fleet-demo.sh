#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# OpenTeddy Fleet — single-machine demo
# ─────────────────────────────────────────────────────────────────────────────
#
# Proves the fleet end-to-end WITHOUT needing two physical machines. Spins up
# an in-process central orchestrator + one worker (both inside one Python
# process, talking over a real localhost WebSocket), registers the worker,
# dispatches a goal, and prints the result.
#
# The worker's local orchestrator.run() is mocked here so the demo doesn't
# need Ollama / a model loaded — it isolates the FLEET plumbing (register →
# dispatch → run → result) so you can confirm that layer works before
# committing two real DGX nodes.
#
# Usage:
#   bash scripts/fleet-demo.sh
#
# Expected: a "🎉 fleet demo PASSED" line. If you see it, the protocol +
# registry + dispatch loop all work; real multi-node is then just a matter
# of the per-node .env files (fleet/env.*.example).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
[ -x "$PY" ] || PY=python3

OPENTEDDY_FLEET_TOKEN="demo-token-$(date +%s 2>/dev/null || echo fixed)" \
OPENTEDDY_FLEET_PORT="8779" \
OPENTEDDY_FLEET_CENTRAL="ws://127.0.0.1:8779" \
OPENTEDDY_FLEET_NODE_ID="demo-worker" \
OPENTEDDY_FLEET_NODE_ROLE="finance" \
"$PY" - <<'PYEOF'
import sys; sys.path.insert(0, ".")
import asyncio

import fleet.worker as W

# Mock the worker's local orchestrator so the demo needs no model.
class _Result:
    status = "completed"
    summary = "✅ (demo) worker ran the dispatched goal on its local orchestrator"
class _Orch:
    async def run(self, req):
        print(f"   [worker] running goal: {req.goal!r}")
        await asyncio.sleep(0.2)
        return _Result()
W._get_local_orchestrator = lambda: _Orch()

from fleet.orchestrator import FleetOrchestrator
from fleet.worker import FleetWorker


async def main():
    print("▶ starting central orchestrator…")
    central = FleetOrchestrator()
    await central.start()
    await asyncio.sleep(0.3)

    print("▶ starting worker (dials central)…")
    worker = FleetWorker()
    worker.start()

    # Wait for registration.
    for _ in range(50):
        if central.registry():
            break
        await asyncio.sleep(0.1)
    reg = central.registry()
    if not reg:
        print("✗ worker never registered"); sys.exit(1)
    n = reg[0]
    print(f"▶ central sees node: id={n['node_id']} role={n['role']} online={n['online']}")

    print("▶ dispatching a goal to the worker…")
    result = await central.dispatch(
        n["node_id"], "整理本月財務異常付款 top 10", timeout_s=10)
    print(f"▶ result: status={result['status']}")
    print(f"          summary={result['summary']}")

    await worker.stop()
    await central.stop()
    await asyncio.sleep(0.2)

    if result["status"] == "completed":
        print("\n🎉 fleet demo PASSED — register → dispatch → run → result all work.")
        print("   Real multi-node next: copy fleet/env.orchestrator.example to the")
        print("   central's .env and fleet/env.worker.example to each worker's .env.")
    else:
        print("\n✗ fleet demo FAILED"); sys.exit(1)


asyncio.run(main())
PYEOF
