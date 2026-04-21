"""
OpenTeddy Approval Store
Manages pending high-risk tool approvals with asyncio event-based waiting.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PendingApproval:
    """A single tool invocation awaiting human approval."""
    id: str
    task_id: str
    tool_name: str
    args: Dict[str, Any]
    created_at: datetime
    status: str = "pending"          # pending | approved | rejected
    result: Optional[Any] = None
    _event: asyncio.Event = field(default_factory=asyncio.Event, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "tool_name": self.tool_name,
            "args": self.args,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
        }


# ── Store ─────────────────────────────────────────────────────────────────────

class ApprovalStore:
    """
    Thread-safe in-memory store for pending tool approvals.

    Usage pattern:
        approval_id = await store.create_approval(task_id, tool_name, args)
        approved = await store.wait_for_resolution(approval_id, timeout=300)
    """

    def __init__(self) -> None:
        self._pending: Dict[str, PendingApproval] = {}
        self._lock = asyncio.Lock()

    # ── Write API ─────────────────────────────────────────────────────────────

    async def create_approval(
        self, task_id: str, tool_name: str, args: Dict[str, Any]
    ) -> str:
        """Create a new pending approval entry and return its ID."""
        approval_id = str(uuid.uuid4())
        approval = PendingApproval(
            id=approval_id,
            task_id=task_id,
            tool_name=tool_name,
            args=args,
            created_at=datetime.utcnow(),
        )
        async with self._lock:
            self._pending[approval_id] = approval
        return approval_id

    async def resolve(self, approval_id: str, approved: bool) -> bool:
        """
        Resolve a pending approval.
        Notifies the waiting coroutine via asyncio.Event.
        Returns False if the approval ID is not found.
        """
        async with self._lock:
            approval = self._pending.get(approval_id)
            if not approval or approval.status != "pending":
                return False
            approval.status = "approved" if approved else "rejected"

        # Signal the waiting coroutine outside the lock
        approval._event.set()
        return True

    async def wait_for_resolution(
        self, approval_id: str, timeout: float = 300.0
    ) -> bool:
        """
        Block until the approval is resolved or timeout expires.
        Returns True if approved, False if rejected or timed out.
        """
        async with self._lock:
            approval = self._pending.get(approval_id)
        if not approval:
            return False

        try:
            await asyncio.wait_for(approval._event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            async with self._lock:
                if approval.status == "pending":
                    approval.status = "rejected"
            return False

        return approval.status == "approved"

    # ── Read API ──────────────────────────────────────────────────────────────

    async def get_pending(self) -> List[PendingApproval]:
        """Return all approvals with status 'pending'."""
        async with self._lock:
            return [a for a in self._pending.values() if a.status == "pending"]

    async def get_all(self) -> List[PendingApproval]:
        """Return all approvals (any status)."""
        async with self._lock:
            return list(self._pending.values())

    async def get(self, approval_id: str) -> Optional[PendingApproval]:
        """Get a single approval by ID."""
        async with self._lock:
            return self._pending.get(approval_id)

    async def cleanup_resolved(self, max_age_seconds: float = 3600.0) -> int:
        """Remove resolved approvals older than max_age_seconds. Returns count removed."""
        now = datetime.utcnow()
        to_remove = []
        async with self._lock:
            for aid, approval in self._pending.items():
                if approval.status != "pending":
                    age = (now - approval.created_at).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(aid)
            for aid in to_remove:
                del self._pending[aid]
        return len(to_remove)


# Module-level singleton
approval_store = ApprovalStore()
