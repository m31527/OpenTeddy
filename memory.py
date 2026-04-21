"""
OpenTeddy Memory Manager
Long-term semantic memory using ChromaDB local vector database.
No external services required — ChromaDB runs fully embedded.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from config import config

if TYPE_CHECKING:
    from models import TaskRequest, TaskResult

logger = logging.getLogger(__name__)

# ── Optional ChromaDB import ──────────────────────────────────────────────────
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    _CHROMADB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CHROMADB_AVAILABLE = False
    logger.warning(
        "chromadb package not installed — long-term memory is disabled. "
        "Run: pip install 'chromadb>=0.5.0'"
    )


# ── Pricing stub for cost calculation (mirrors tracker.py) ───────────────────
_CLAUDE_PRICES: dict[str, tuple[float, float]] = {
    # model_prefix: (price_per_input_token, price_per_output_token)
    "claude-opus":    (15.0 / 1_000_000, 75.0 / 1_000_000),
    "claude-sonnet":  (3.0  / 1_000_000, 15.0 / 1_000_000),
    "claude-haiku":   (0.80 / 1_000_000, 4.0  / 1_000_000),
}


class MemoryManager:
    """
    Semantic long-term memory backed by a local ChromaDB PersistentClient.

    Usage:
        memory = MemoryManager()
        await memory.open()
        ctx = await memory.get_context_for_task("analyse sales data")
        ...
        await memory.close()
    """

    COLLECTION_NAME = "openteddy_memory"

    def __init__(self, db_path: str = "") -> None:
        self.db_path = db_path or getattr(config, "memory_db_path", "./memory_db")
        self._client: Optional[object] = None
        self._collection: Optional[object] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def open(self) -> None:
        if not _CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available — memory features disabled.")
            return
        try:
            self._client = chromadb.PersistentClient(
                path=self.db_path,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(  # type: ignore[union-attr]
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "MemoryManager opened: %s (%d memories stored)",
                self.db_path,
                self._collection.count(),  # type: ignore[union-attr]
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("MemoryManager failed to open: %s", exc)
            self._client = None
            self._collection = None

    async def close(self) -> None:
        self._client = None
        self._collection = None

    @property
    def is_available(self) -> bool:
        return self._collection is not None

    # ── Write ─────────────────────────────────────────────────────────────────

    async def add_memory(
        self,
        content: str,
        memory_type: str,
        task_id: str,
        importance: float = 0.5,
    ) -> Optional[str]:
        """
        Embed *content* and store it in the collection.

        Args:
            content:     The text to remember.
            memory_type: One of "task_result" | "user_preference" |
                         "system_context" | "conversation".
            task_id:     Parent task ID for provenance.
            importance:  0.0–1.0 float used for future weighting.

        Returns:
            The generated memory ID, or None on failure.
        """
        if not self.is_available:
            return None
        if not content or not content.strip():
            return None

        memory_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        try:
            self._collection.add(  # type: ignore[union-attr]
                documents=[content.strip()],
                metadatas=[{
                    "type": memory_type,
                    "task_id": task_id,
                    "timestamp": timestamp,
                    "importance": float(importance),
                }],
                ids=[memory_id],
            )
            logger.debug(
                "Memory stored id=%s type=%s task=%s",
                memory_id[:8], memory_type, task_id,
            )
            return memory_id
        except Exception as exc:  # noqa: BLE001
            logger.error("add_memory failed: %s", exc)
            return None

    # ── Read ──────────────────────────────────────────────────────────────────

    async def search_memory(
        self, query: str, n_results: int = 5
    ) -> list[dict]:
        """
        Semantic similarity search.

        Returns a list of dicts:
          {id, content, type, task_id, timestamp, importance, relevance_score}
        Most relevant first.
        """
        if not self.is_available or not query:
            return []

        try:
            total = self._collection.count()  # type: ignore[union-attr]
            if total == 0:
                return []

            actual_n = min(n_results, total)
            results = self._collection.query(  # type: ignore[union-attr]
                query_texts=[query],
                n_results=actual_n,
            )

            docs      = results.get("documents", [[]])[0]
            metas     = results.get("metadatas",  [[]])[0]
            ids       = results.get("ids",        [[]])[0]
            distances = results.get("distances",  [[]])[0]

            memories = []
            for doc, meta, mem_id, dist in zip(docs, metas, ids, distances):
                memories.append({
                    "id":              mem_id,
                    "content":         doc,
                    "type":            meta.get("type", "unknown"),
                    "task_id":         meta.get("task_id", ""),
                    "timestamp":       meta.get("timestamp", ""),
                    "importance":      float(meta.get("importance", 0.5)),
                    "relevance_score": round(max(0.0, 1.0 - float(dist)), 4),
                })
            return memories

        except Exception as exc:  # noqa: BLE001
            logger.error("search_memory failed: %s", exc)
            return []

    async def get_context_for_task(self, task_description: str) -> str:
        """
        Retrieve the top-5 most relevant memories and format them as a
        context block ready for injection into an LLM system prompt.

        Returns empty string if no memories exist or ChromaDB is unavailable.
        """
        memories = await self.search_memory(task_description, n_results=5)
        if not memories:
            return ""

        _LABELS = {
            "task_result":    "Past task result",
            "user_preference": "User preference",
            "system_context": "System context",
            "conversation":   "Conversation",
        }

        lines = ["=== Relevant Memory ==="]
        for mem in memories:
            label   = _LABELS.get(mem["type"], mem["type"].replace("_", " ").title())
            snippet = mem["content"][:400]
            lines.append(f"[{label}]: {snippet}")

        return "\n".join(lines)

    # ── Post-task storage ─────────────────────────────────────────────────────

    async def summarize_and_store(
        self,
        task_id: str,
        goal: str,
        final_output: str,
    ) -> None:
        """
        Called automatically after a task completes.
        Stores a task-result summary and any detected user preferences.
        """
        if not self.is_available:
            return

        # 1. Store the task result summary
        summary = f"Task: {goal}\nResult: {final_output[:800]}"
        await self.add_memory(
            content=summary,
            memory_type="task_result",
            task_id=task_id,
            importance=0.7,
        )

        # 2. Extract and store detected user preferences
        for pref in _extract_preferences(goal, final_output):
            await self.add_memory(
                content=pref,
                memory_type="user_preference",
                task_id=task_id,
                importance=0.8,
            )

    # ── Pagination / management ───────────────────────────────────────────────

    async def list_memories(
        self, page: int = 1, page_size: int = 20
    ) -> dict:
        """
        Return a paginated list of all stored memories (newest first).

        Response shape:
          {items, total, page, page_size, total_pages}
        """
        empty = {
            "items": [],
            "total": 0,
            "page": page,
            "page_size": page_size,
            "total_pages": 0,
        }
        if not self.is_available:
            return empty

        try:
            total = self._collection.count()  # type: ignore[union-attr]
            if total == 0:
                return empty

            # ChromaDB get() returns all records
            raw = self._collection.get(  # type: ignore[union-attr]
                include=["documents", "metadatas"],
            )
            docs  = raw.get("documents", [])
            metas = raw.get("metadatas", [])
            ids   = raw.get("ids", [])

            items = [
                {
                    "id":        mem_id,
                    "content":   doc,
                    "type":      meta.get("type", "unknown"),
                    "task_id":   meta.get("task_id", ""),
                    "timestamp": meta.get("timestamp", ""),
                    "importance": float(meta.get("importance", 0.5)),
                }
                for doc, meta, mem_id in zip(docs, metas, ids)
            ]

            # Sort newest-first
            items.sort(key=lambda x: x["timestamp"], reverse=True)

            total_pages = max(1, (total + page_size - 1) // page_size)
            offset      = (page - 1) * page_size
            page_items  = items[offset : offset + page_size]

            return {
                "items":       page_items,
                "total":       total,
                "page":        page,
                "page_size":   page_size,
                "total_pages": total_pages,
            }

        except Exception as exc:  # noqa: BLE001
            logger.error("list_memories failed: %s", exc)
            return empty

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a single memory entry by ID. Returns True on success."""
        if not self.is_available:
            return False
        try:
            self._collection.delete(ids=[memory_id])  # type: ignore[union-attr]
            logger.debug("Memory deleted: %s", memory_id[:8])
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("delete_memory failed for %s: %s", memory_id, exc)
            return False

    async def clear_all(self) -> int:
        """
        Delete every memory in the collection.
        Returns the number of records that were deleted.
        """
        if not self.is_available:
            return 0
        try:
            count = self._collection.count()  # type: ignore[union-attr]
            if count > 0:
                # Easiest full-wipe: delete & recreate the collection
                self._client.delete_collection(self.COLLECTION_NAME)  # type: ignore[union-attr]
                self._collection = self._client.get_or_create_collection(  # type: ignore[union-attr]
                    name=self.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info("MemoryManager cleared %d memories.", count)
            return count
        except Exception as exc:  # noqa: BLE001
            logger.error("clear_all failed: %s", exc)
            return 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_preferences(goal: str, output: str) -> list[str]:
    """
    Heuristically extract user-preference hints from a completed task.
    Deliberately conservative — only stores high-confidence signals.
    """
    prefs: list[str] = []
    goal_lower = goal.lower()

    # Language preference
    if any(kw in goal_lower for kw in ("繁體中文", "traditional chinese", "zh-tw")):
        prefs.append("User prefers responses in Traditional Chinese (繁體中文)")
    elif any(kw in goal_lower for kw in ("english only", "reply in english", "英文回答")):
        prefs.append("User prefers responses in English")

    # Output format preferences
    if any(kw in goal_lower for kw in ("markdown", "表格", "table format")):
        prefs.append(f"User requested structured/table output for goal: {goal[:120]}")

    # Brevity / verbosity signals
    if any(kw in goal_lower for kw in ("簡短", "brief", "concise", "tl;dr")):
        prefs.append("User prefers concise, brief answers")
    elif any(kw in goal_lower for kw in ("詳細", "detailed", "comprehensive", "in depth")):
        prefs.append("User prefers detailed, comprehensive answers")

    return prefs
