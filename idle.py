"""
How long OpenTeddy has had nothing to do.

Every task goes through orchestrator.run(), which marks itself here.
Background chores (naming sessions, and later others) wait for a quiet
stretch, so they don't take the model from someone who is waiting on it.
"""

from __future__ import annotations

import time

_in_flight = 0
_last_active = time.monotonic()


def begin() -> None:
    global _in_flight
    _in_flight += 1


def end() -> None:
    global _in_flight, _last_active
    _in_flight = max(0, _in_flight - 1)
    _last_active = time.monotonic()


def quiet_for() -> float:
    """Seconds since the last task finished; 0 while one is running."""
    if _in_flight:
        return 0.0
    return time.monotonic() - _last_active
