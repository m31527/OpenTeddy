"""
OpenTeddy Fleet — optional distributed-orchestration layer.

This package is imported ONLY when OPENTEDDY_FLEET_ROLE is set to
"worker" or "orchestrator". On a single-machine install (the default,
role="none") nothing here is imported — desktop / personal web users
pay zero cost and see zero behaviour change.

See docs/fleet-architecture.md for the full design.
"""

import os


def fleet_role() -> str:
    """Resolve the node's fleet role from the environment.

    Returns one of "none" (default — single-machine), "worker", or
    "orchestrator". Anything unrecognised is treated as "none" so a
    typo can never accidentally open a network listener.
    """
    role = (os.getenv("OPENTEDDY_FLEET_ROLE", "none") or "none").strip().lower()
    return role if role in ("none", "worker", "orchestrator") else "none"


def is_fleet_enabled() -> bool:
    return fleet_role() != "none"
