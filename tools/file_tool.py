"""
OpenTeddy File Tool
Read, write, list and delete files on the local filesystem.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)


# ── Implementations ────────────────────────────────────────────────────────────

async def read_file(path: str) -> Dict[str, Any]:
    """Read a file and return its contents as a string. LOW risk."""
    start = time.monotonic()
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return make_result(False, error=f"File not found: {path}",
                               duration_ms=_ms(start))
        if not p.is_file():
            return make_result(False, error=f"Path is not a file: {path}",
                               duration_ms=_ms(start))
        content = p.read_text(encoding="utf-8", errors="replace")
        return make_result(True, result={"path": str(p), "content": content,
                                         "size_bytes": p.stat().st_size},
                           duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=str(exc), duration_ms=_ms(start))


async def write_file(path: str, content: str) -> Dict[str, Any]:
    """Write content to a file (creates parent dirs). HIGH risk."""
    start = time.monotonic()
    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return make_result(True, result={"path": str(p),
                                          "bytes_written": len(content.encode())},
                           duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=str(exc), duration_ms=_ms(start))


async def list_directory(path: str) -> Dict[str, Any]:
    """List files and directories at a given path. LOW risk."""
    start = time.monotonic()
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return make_result(False, error=f"Path not found: {path}",
                               duration_ms=_ms(start))
        if not p.is_dir():
            return make_result(False, error=f"Not a directory: {path}",
                               duration_ms=_ms(start))
        entries = []
        for entry in sorted(p.iterdir()):
            stat = entry.stat()
            entries.append({
                "name": entry.name,
                "type": "dir" if entry.is_dir() else "file",
                "size_bytes": stat.st_size if entry.is_file() else None,
            })
        return make_result(True, result={"path": str(p), "entries": entries},
                           duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=str(exc), duration_ms=_ms(start))


async def delete_file(path: str) -> Dict[str, Any]:
    """Delete a file. HIGH risk — requires approval."""
    start = time.monotonic()
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return make_result(False, error=f"File not found: {path}",
                               duration_ms=_ms(start))
        if p.is_dir():
            return make_result(False,
                               error="Use shell_exec_write to delete directories.",
                               duration_ms=_ms(start))
        p.unlink()
        return make_result(True, result={"deleted": str(p)},
                           duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=str(exc), duration_ms=_ms(start))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


# ── Schemas ───────────────────────────────────────────────────────────────────

_SCHEMA_READ: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path."},
            },
            "required": ["path"],
        },
    },
}

_SCHEMA_WRITE: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to a file, creating it if it doesn't exist. Requires approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write to."},
                "content": {"type": "string", "description": "Text content to write."},
            },
            "required": ["path", "content"],
        },
    },
}

_SCHEMA_LIST: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "list_directory",
        "description": "List files and subdirectories in a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list."},
            },
            "required": ["path"],
        },
    },
}

_SCHEMA_DELETE: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "delete_file",
        "description": "Delete a file. Requires approval. Cannot delete directories.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to delete."},
            },
            "required": ["path"],
        },
    },
}

# ── Export ─────────────────────────────────────────────────────────────────────

FILE_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    (read_file,      _SCHEMA_READ,   "low"),
    (write_file,     _SCHEMA_WRITE,  "high"),
    (list_directory, _SCHEMA_LIST,   "low"),
    (delete_file,    _SCHEMA_DELETE, "high"),
]
