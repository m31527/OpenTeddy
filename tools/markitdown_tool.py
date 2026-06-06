"""
OpenTeddy Markitdown Tool
─────────────────────────────────────────────────────────────────────────────
Universal "any document → clean markdown" tool, backed by Microsoft's
`markitdown` library.

Why this exists
---------------
Before this commit OpenTeddy had `pdf_extract_text` (pypdf-backed) and
nothing else for document extraction. So:

  - Reading a .docx required the agent to plan a `python_exec` round
    with python-docx
  - Reading a .pptx wasn't really possible without similar plumbing
  - Reading an .xlsx left the agent to `csv_describe` after a manual
    conversion
  - Reading audio / EPUB / YouTube URLs simply wasn't on the menu

markitdown collapses all of those into one entry point. PDF / PPTX /
DOCX / XLSX / images (with EXIF) / audio (with EXIF) / HTML / CSV /
JSON / XML / ZIP / YouTube URLs / EPUB all become markdown that the
LLM can chew on directly.

Read-only by design — never writes to the workspace. The output goes
back to the LLM as the tool result, the LLM decides whether to
summarise it / save a derived file / pull specific sections.

Trade-offs vs `pdf_extract_text` — complementary, not a swap
------------------------------------------------------------
A real user A/B test on a recruitment-tracking PDF revealed that
markitdown isn't a strict upgrade for every PDF shape:

  - Form-style PDFs (resumes, application forms, recruitment trackers,
    contracts with section labels): markitdown's structure-first
    extraction CAN lose the spatial pairing between a label and its
    value. The actual example case: an HR sheet had two columns
    [現任狀態 | 待業中] [前任公司 | 北京字節跳動]. pypdf reads each
    cell in space-order, the LLM sees "現任狀態 待業中 前任公司
    北京字節跳動" and pairs correctly. markitdown tried to be smart
    about the structure, dropped the "前任" prefix during
    markdownification, and the LLM ended up answering "current
    company: ByteDance" — wrong.

  - Structured documents (reports, specs, whitepapers, slides, .docx,
    .xlsx, .epub, .html): markitdown wins clearly because it
    preserves headings + tables + lists. pypdf flattens these into
    space-separated tokens and the LLM loses chapter / table
    structure.

So neither tool deprecates the other. The planner prompt
(_PLAN_SYSTEM_CODE in orchestrator.py) routes between them based on
PDF shape; the executor honours whichever the planner picked. New
formats markitdown alone reaches (.pptx / .docx / .xlsx / .epub /
images / audio / YouTube URLs) all unambiguously go through
doc_to_markdown — there's no pypdf alternative for those.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)


# ── Tunables ──────────────────────────────────────────────────────────────────

# Cap the markdown returned to the LLM. markitdown can output tens of
# thousands of chars on a long PDF — we don't want one tool call to
# blow the executor's context budget. The agent can ask for a page
# range / specific section in a follow-up if it needs more.
_MAX_MARKDOWN_CHARS = 40000

# Per-call timeout. markitdown invokes synchronous parsing internally
# (pypdf / python-docx / openpyxl / etc.) — most files complete in
# < 5 s but a corrupted 100-page PDF can take a while. Beyond 90 s
# something's wrong and we'd rather fail loudly than tie up the executor.
_PARSE_TIMEOUT_S = 90

# Extensions we explicitly advertise as "this tool handles them".
# markitdown actually supports more (audio formats, YouTube URLs, EPUB,
# …) — we just don't put those in the schema's primary list to keep
# the description tight. The tool DOES handle them when given the path;
# we list them in the schema's notes string instead.
_PRIMARY_EXTENSIONS: Tuple[str, ...] = (
    ".pdf", ".pptx", ".docx", ".xlsx", ".xls",
    ".html", ".htm", ".csv", ".tsv", ".json", ".xml",
    ".md", ".txt",
)


# ── Implementation ────────────────────────────────────────────────────────────

async def doc_to_markdown(
    path: str,
    max_chars: Optional[int] = None,
) -> Dict[str, Any]:
    """Convert a document at `path` to markdown.

    Supported formats (best-effort, depends on markitdown extras
    installed): PDF, PowerPoint, Word, Excel, images (with EXIF),
    audio (with EXIF + optional speech transcription), HTML, CSV,
    JSON, XML, ZIP, YouTube URLs, EPUB, plain text.

    Returns ``{success, result: {format, length_chars, markdown,
    truncated}, error, duration_ms}`` — same shape as every other
    OpenTeddy tool.

    Read-only; risk level "low".
    """
    start = time.monotonic()

    path_str = (path or "").strip()
    if not path_str:
        return make_result(False, error="path is empty", duration_ms=0)

    is_url = path_str.startswith(("http://", "https://"))
    if not is_url and not os.path.isfile(path_str):
        return make_result(
            False,
            error=f"file not found: {path_str}",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    # Cap requested length but don't let the model ask for unbounded
    # output — that's how the executor's context blows up. If the
    # caller wants more they can call the tool again on a specific
    # section / page range (a future enhancement).
    if max_chars is None:
        max_chars = _MAX_MARKDOWN_CHARS
    max_chars = max(500, min(int(max_chars), _MAX_MARKDOWN_CHARS))

    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        return make_result(
            False,
            error=(
                f"markitdown is not installed in this Python ({exc}). "
                "Run `pip install 'markitdown[all]'` and restart."
            ),
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    # markitdown's main entry is synchronous — wrap with run_in_executor
    # so we don't pin the event loop for a multi-second PDF parse.
    loop = asyncio.get_event_loop()

    def _convert() -> Any:
        mid = MarkItDown()
        return mid.convert(path_str)

    try:
        res = await asyncio.wait_for(
            loop.run_in_executor(None, _convert),
            timeout=_PARSE_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        return make_result(
            False,
            error=(
                f"markitdown parse exceeded {_PARSE_TIMEOUT_S}s — file may "
                "be corrupted or unusually complex. Try a smaller portion."
            ),
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("markitdown convert failed for %s", path_str)
        return make_result(
            False,
            error=f"markitdown error: {exc}",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    # markitdown.convert() returns a DocumentConverterResult-ish object
    # with .text_content (markdown body) and .title attrs across all
    # converters. Defensive getattr — a future version could rename.
    markdown = (
        getattr(res, "text_content", None)
        or getattr(res, "markdown", None)
        or ""
    )
    title = getattr(res, "title", "") or ""
    truncated = False
    if len(markdown) > max_chars:
        markdown = markdown[:max_chars] + (
            "\n\n…(truncated — file produced more markdown than the "
            f"{max_chars}-char cap; ask for a specific section if needed)"
        )
        truncated = True

    # Best-effort format hint from the extension. Doesn't drive any
    # behaviour, just tells the agent "you got markdown out of a pptx"
    # so the model knows whether tables / slide breaks are meaningful.
    _, ext = os.path.splitext(path_str.lower())
    format_hint = (ext or "").lstrip(".") or "unknown"
    if is_url:
        if "youtube.com" in path_str or "youtu.be" in path_str:
            format_hint = "youtube"
        else:
            format_hint = "url"

    return make_result(
        True,
        result={
            "format":      format_hint,
            "title":       title[:200],
            "length_chars": len(markdown),
            "truncated":   truncated,
            "markdown":    markdown,
        },
        duration_ms=int((time.monotonic() - start) * 1000),
    )


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA_DOC_TO_MARKDOWN: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "doc_to_markdown",
        "description": (
            "Convert a document into clean markdown for the LLM to read. "
            "Use this as the FIRST step whenever you need to understand "
            "the content of a file the user uploaded or named. Covers "
            "PDF, PowerPoint (.pptx), Word (.docx), Excel (.xlsx/.xls), "
            "HTML, CSV/TSV, JSON, XML, plain text, EPUB, images (EXIF + "
            "OCR), audio (EXIF + transcription), ZIP archives, and "
            "YouTube URLs. Read-only — does not write to the workspace. "
            "Output is capped at ~40000 chars; ask for a specific section "
            "if the file is long."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Absolute or workspace-relative path to a file, OR "
                        "a public http/https URL (the tool will fetch it). "
                        "YouTube URLs supported — markitdown extracts "
                        "transcript + metadata."
                    ),
                },
                "max_chars": {
                    "type": "integer",
                    "description": (
                        "Optional cap on returned markdown size. Clamped "
                        "to the range [500, 40000]. Default 40000."
                    ),
                },
            },
            "required": ["path"],
        },
    },
}


# ── Export ────────────────────────────────────────────────────────────────────

MARKITDOWN_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    (doc_to_markdown, _SCHEMA_DOC_TO_MARKDOWN, "low"),
]
