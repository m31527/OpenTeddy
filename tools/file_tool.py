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

def _resolve_path(path: str) -> Path:
    """Resolve ``path`` against the session's effective workspace when
    relative, against the user's home with ~, otherwise verbatim.

    Before this fix, ``Path(path).resolve()`` anchored relative paths to
    the Python process cwd (uvicorn's cwd — usually the OpenTeddy repo
    root). That meant ``read_file("Dockerfile.relay")`` looked in the
    wrong place and failed, even though ``cat Dockerfile.relay`` via
    shell_tool worked — because shell_tool anchors at the workspace
    while file_tool did not. The two tools disagreeing on cwd was
    causing Qwen to loop: retry the same read_file that it couldn't
    see, fall back to cat, waste rounds.

    Now every file tool uses the same anchor as shell_tool and
    deploy_tool: the session's effective workspace.
    """
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return Path(expanded).resolve()
    # Relative — anchor to the session workspace.
    try:
        from config import effective_workspace_dir
        ws = effective_workspace_dir()
    except Exception:  # noqa: BLE001
        ws = os.getcwd()
    return Path(os.path.join(ws, expanded)).resolve()


async def read_file(path: str) -> Dict[str, Any]:
    """Read a file and return its contents as a string. LOW risk."""
    start = time.monotonic()
    try:
        p = _resolve_path(path)
        if not p.exists():
            return make_result(False, error=f"File not found: {p}",
                               duration_ms=_ms(start))
        if not p.is_file():
            return make_result(False, error=f"Path is not a file: {p}",
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
        p = _resolve_path(path)
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
        p = _resolve_path(path)
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


async def pdf_extract_text(
    path: str,
    max_pages: int = 50,
    start_page: int = 0,
) -> Dict[str, Any]:
    """Extract text from a PDF, page by page, and return one big string.

    The agent calls this when the user attaches a PDF and asks a question
    about it. Once the text is in the tool-result the executor can
    reason over it like any other text file.

    Args:
        path:       PDF location, relative to the session workspace or
                    absolute (same rules as read_file).
        max_pages:  Hard cap on pages extracted in one call. Default 50
                    keeps a typical 30-page report in scope while
                    protecting against a 1000-page tome blowing the
                    executor's context window. Bump it explicitly when
                    you know the PDF is bigger.
        start_page: 0-indexed page to start at. Combine with max_pages
                    to page through a large PDF in chunks (e.g. call
                    with start_page=50, max_pages=50 for pages 50-99).

    Returns (in result):
        path, total_pages, extracted_pages (= count actually read),
        start_page, end_page, metadata (title/author/etc if present),
        text (single string, pages joined with "--- Page N ---" markers
        so the model can cite page numbers in its answer),
        truncated (True if we hit the char_cap below).

    Image-only PDFs (no embedded text layer — e.g. scanned docs) return
    success=True but text="" or near-empty; we surface a hint in that
    case so the model can tell the user it needs an OCR pre-pass
    instead of silently answering nothing.
    """
    start = time.monotonic()
    # ~300K chars ≈ 75K tokens at 4 chars/token. Beyond that, dumping the
    # whole thing into a single tool result risks blowing the executor's
    # context. The agent can call again with a higher start_page for
    # later sections.
    CHAR_CAP = 300_000

    try:
        # Local import so the rest of file_tool.py stays usable even if
        # pypdf is somehow missing from the install (e.g. sidecar shipped
        # without the dep). Returns a clear, actionable error instead of
        # a confusing ImportError at registration time.
        try:
            from pypdf import PdfReader
        except ImportError:
            return make_result(
                False,
                error=("pypdf not installed. Run: pip install pypdf>=4.0.0 "
                       "(or rebuild the sidecar with the updated "
                       "requirements.txt)."),
                duration_ms=_ms(start),
            )

        p = _resolve_path(path)
        if not p.exists():
            return make_result(False, error=f"File not found: {p}",
                               duration_ms=_ms(start))
        if not p.is_file():
            return make_result(False, error=f"Path is not a file: {p}",
                               duration_ms=_ms(start))
        if p.suffix.lower() != ".pdf":
            # Soft check — pypdf would fail with a less helpful message
            # on a mis-renamed file. Better to fail fast with a hint.
            return make_result(
                False,
                error=f"Not a .pdf file: {p.name}. Use read_file for "
                       f"plain-text formats.",
                duration_ms=_ms(start),
            )

        # strict=False: tolerate slightly-malformed PDFs (Acrobat's own
        # output is shockingly often non-spec-compliant). pypdf still
        # raises on truly broken files, which we catch below.
        reader = PdfReader(str(p), strict=False)

        total_pages = len(reader.pages)
        # Defensive arg clamping — callers can hand us garbage.
        start_page = max(0, int(start_page))
        max_pages  = max(1, min(int(max_pages), 500))  # 500 abs upper bound
        end_page   = min(total_pages, start_page + max_pages)

        if start_page >= total_pages:
            return make_result(
                False,
                error=(f"start_page={start_page} is past end of document "
                       f"(total_pages={total_pages}). Use start_page=0 to "
                       f"begin from the first page."),
                duration_ms=_ms(start),
            )

        # Page-by-page extract with a per-page try so one malformed page
        # doesn't kill the whole call. The agent gets whatever we could
        # read, plus a note about which pages failed.
        chunks: list[str] = []
        failed_pages: list[int] = []
        total_chars = 0
        truncated = False
        for i in range(start_page, end_page):
            try:
                page_text = reader.pages[i].extract_text() or ""
            except Exception as exc:  # noqa: BLE001
                failed_pages.append(i + 1)
                page_text = f"[extraction failed: {exc}]"
            # 1-indexed in the marker for human readability — matches
            # what PDF viewers show.
            chunks.append(f"--- Page {i + 1} ---\n{page_text.strip()}")
            total_chars += len(page_text)
            if total_chars >= CHAR_CAP:
                truncated = True
                break

        # Metadata is genuinely useful for the agent (title, author,
        # creation date) when answering "who wrote this" / "when was
        # this published" questions. pypdf returns None / strings; we
        # coerce to a flat dict of strs and drop empties so the result
        # stays tidy.
        meta: dict[str, str] = {}
        try:
            info = reader.metadata or {}
            for raw_key, raw_val in info.items():
                if raw_val is None:
                    continue
                # Keys come back as "/Title", "/Author", etc. Strip the
                # leading slash + lowercase so downstream JSON is clean.
                key = str(raw_key).lstrip("/").lower()
                meta[key] = str(raw_val)
        except Exception:  # noqa: BLE001
            # Some PDFs have malformed metadata that pypdf rejects.
            # That's not fatal — just skip metadata.
            pass

        joined = "\n\n".join(chunks)
        # Heuristic: if we read pages but got essentially no text, it's
        # almost certainly an image-only PDF. Tell the agent so it can
        # tell the user instead of fabricating an answer from nothing.
        stripped_text = joined.replace("--- Page", "").strip()
        # Subtract roughly the marker overhead.
        approx_text_chars = max(0, len(stripped_text) - 30 * len(chunks))
        likely_image_only = approx_text_chars < 50 and (end_page - start_page) >= 1

        result = {
            "path":            str(p),
            "total_pages":     total_pages,
            "extracted_pages": end_page - start_page,
            "start_page":      start_page,
            "end_page":        end_page,
            "metadata":        meta,
            "text":            joined,
            "truncated":       truncated,
        }
        if failed_pages:
            result["failed_pages"] = failed_pages
        if likely_image_only:
            result["hint"] = (
                "This PDF appears to be image-only (scanned, no text "
                "layer). Tell the user the file needs OCR — pdf_extract_text "
                "can only read embedded text."
            )

        return make_result(True, result=result, duration_ms=_ms(start))
    except Exception as exc:  # noqa: BLE001
        return make_result(False, error=f"PDF read failed: {exc}",
                           duration_ms=_ms(start))


async def delete_file(path: str) -> Dict[str, Any]:
    """Delete a file. HIGH risk — requires approval."""
    start = time.monotonic()
    try:
        p = _resolve_path(path)
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

_SCHEMA_PDF_EXTRACT: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "pdf_extract_text",
        "description": (
            "Extract text from a PDF file and return it as a single "
            "string with '--- Page N ---' markers between pages. Use "
            "this when the user uploads a PDF and asks a question "
            "about its contents. For multi-hundred-page PDFs, call "
            "repeatedly with increasing start_page to page through. "
            "Image-only PDFs (no embedded text layer) return empty "
            "text plus a hint; tell the user OCR is needed instead "
            "of fabricating an answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "PDF path. Relative paths resolve "
                                   "against the session workspace "
                                   "(e.g. 'uploads/report.pdf').",
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Max pages to read in one call (default 50, "
                                   "hard cap 500). Lower this for huge PDFs "
                                   "to keep the result under the context "
                                   "window.",
                    "default": 50,
                },
                "start_page": {
                    "type": "integer",
                    "description": "0-indexed first page to extract. Use "
                                   "to page through a large PDF (e.g. "
                                   "0,50,100…).",
                    "default": 0,
                },
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
    (read_file,        _SCHEMA_READ,        "low"),
    (write_file,       _SCHEMA_WRITE,       "high"),
    (list_directory,   _SCHEMA_LIST,        "low"),
    (delete_file,      _SCHEMA_DELETE,      "high"),
    # Read-only — same risk class as read_file. Lets the agent answer
    # "what does this PDF say about X" without an approval round-trip.
    (pdf_extract_text, _SCHEMA_PDF_EXTRACT, "low"),
]
