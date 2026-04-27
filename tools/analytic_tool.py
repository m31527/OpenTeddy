"""
OpenTeddy Analytic Tool
─────────────────────────────────────────────────────────────────────────────
Adds first-class CSV / dataframe handling so the agent can answer
"compute X over this CSV" questions (repurchase rate, frequency, basic
stats) without inventing it inside Qwen's head.

The two tools shipped here:

  csv_describe(path)
      Quick survey: shape, dtypes, head(5), and per-column stats.
      Always call this BEFORE writing pandas code — gives the model the
      column names and types so it doesn't hallucinate.

  python_exec(code, [path])
      Run Python in a subprocess with pandas / numpy preloaded. Scoped
      to the session's agent workspace; the optional ``path`` argument
      lands a ``df`` already loaded with that CSV / Parquet / JSON.

Both tools are sandboxed by:
  • Running in a SUBPROCESS — process crash can't take the server down.
  • Hard 60 s timeout — kills runaway pandas pipelines.
  • cwd = agent workspace — file access is scoped to the user's data.
  • No network preloaded; agent can still ``import requests`` if needed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)


# ── Workspace resolution ───────────────────────────────────────────────────────

def _workspace_root(working_dir: Optional[str]) -> Path:
    """Mirror file_tool's resolution: explicit working_dir wins, else
    the global agent_workspace_dir from config."""
    if working_dir:
        return Path(working_dir).expanduser().resolve()
    from config import config
    return Path(config.agent_workspace_dir).expanduser().resolve()


def _resolve(path: str, working_dir: Optional[str]) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (_workspace_root(working_dir) / p).resolve()


# ── csv_describe ──────────────────────────────────────────────────────────────

async def csv_describe(path: str, working_dir: Optional[str] = None) -> Dict[str, Any]:
    """Quick CSV survey: shape, dtypes, head(5), describe().

    Always cheap to call (subprocess + pandas import is ~500 ms). Use
    this BEFORE writing python_exec code so the model sees real column
    names and dtypes instead of guessing.
    """
    target = _resolve(path, working_dir)
    if not target.exists():
        return make_result(False, error=f"file not found: {target}")
    if not target.is_file():
        return make_result(False, error=f"not a regular file: {target}")

    # Pick a reader based on extension. xlsx / xls go through pandas'
    # Excel reader (openpyxl / xlrd) instead of read_csv — without this
    # the agent saw "Unicode error" / "no header" and gave up, telling
    # the user to "convert to CSV", which is the wrong UX.
    ext = target.suffix.lower()
    if ext in {".xlsx", ".xlsm"}:
        reader_call = f"pd.read_excel({str(target)!r}, engine='openpyxl')"
    elif ext == ".xls":
        reader_call = f"pd.read_excel({str(target)!r}, engine='xlrd')"
    elif ext == ".parquet":
        reader_call = f"pd.read_parquet({str(target)!r})"
    elif ext == ".json":
        reader_call = f"pd.read_json({str(target)!r})"
    elif ext in {".tsv", ".tab"}:
        reader_call = f"pd.read_csv({str(target)!r}, sep='\\t', low_memory=False)"
    else:
        reader_call = f"pd.read_csv({str(target)!r}, low_memory=False)"

    code = textwrap.dedent(f"""\
        import json, sys
        import pandas as pd
        df = {reader_call}
        out = {{
            "rows":    int(len(df)),
            "cols":    int(df.shape[1]),
            "columns": list(df.columns),
            "dtypes":  {{c: str(t) for c, t in df.dtypes.items()}},
            "head":    df.head(5).to_dict(orient="records"),
        }}
        try:
            desc = df.describe(include="all", datetime_is_numeric=True)
            out["describe"] = json.loads(desc.to_json(orient="index"))
        except TypeError:
            # Older pandas without datetime_is_numeric
            out["describe"] = json.loads(df.describe(include="all").to_json(orient="index"))
        print(json.dumps(out, default=str, ensure_ascii=False))
    """)

    rc, stdout, stderr = await _run_python(code, cwd=str(_workspace_root(working_dir)), timeout=30)
    if rc != 0:
        hint = _missing_package_hint(stderr)
        return make_result(False, error=f"pandas describe failed: {hint or stderr.strip() or stdout.strip()}")
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return make_result(False, error=f"could not parse describe output: {exc}")
    return make_result(True, result=data)


# ── python_exec ───────────────────────────────────────────────────────────────

async def python_exec(
    code: str,
    path: Optional[str] = None,
    working_dir: Optional[str] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Run Python (with pandas + numpy preloaded) in a subprocess.

    If ``path`` is supplied, a DataFrame ``df`` is already loaded from
    the CSV / Parquet / JSON file before the user's code runs — this is
    the common case for the agent ("load this CSV and compute X").

    The agent should ``print(...)`` whatever it wants to return — stdout
    becomes the tool result. Tracebacks land in stderr and surface as
    a tool error so the next turn can correct itself.
    """
    if not code or not code.strip():
        return make_result(False, error="code argument is required")

    cwd = str(_workspace_root(working_dir))

    # Preamble loads the dataframe if a path was supplied. We pick a
    # reader by extension; unknown extensions fall back to read_csv.
    preamble_lines: List[str] = [
        "import json, sys, math",
        "import pandas as pd",
        "import numpy as np",
    ]
    if path:
        target = _resolve(path, working_dir)
        ext = target.suffix.lower()
        # Map extension → full reader expression. We build the entire
        # call inline (rather than just the function name) because Excel
        # readers need the openpyxl/xlrd engine kwarg and read_csv needs
        # the low_memory flag — they don't share a uniform signature.
        if ext in {".xlsx", ".xlsm"}:
            reader_call = f"pd.read_excel({str(target)!r}, engine='openpyxl')"
        elif ext == ".xls":
            reader_call = f"pd.read_excel({str(target)!r}, engine='xlrd')"
        elif ext == ".parquet":
            reader_call = f"pd.read_parquet({str(target)!r})"
        elif ext == ".json":
            reader_call = f"pd.read_json({str(target)!r})"
        elif ext in {".tsv", ".tab"}:
            reader_call = f"pd.read_csv({str(target)!r}, sep='\\t', low_memory=False)"
        else:
            reader_call = f"pd.read_csv({str(target)!r}, low_memory=False)"
        preamble_lines.append(f"df = {reader_call}")
        preamble_lines.append("# df is the loaded dataframe — go.")

    full_code = "\n".join(preamble_lines) + "\n\n" + textwrap.dedent(code)

    rc, stdout, stderr = await _run_python(full_code, cwd=cwd, timeout=timeout)
    out = {
        "stdout":     stdout[-20_000:],   # cap to ~20 KB so the LLM context doesn't explode
        "stderr":     stderr[-5_000:],
        "returncode": rc,
        "cwd":        cwd,
    }
    if rc != 0:
        hint = _missing_package_hint(stderr)
        if hint:
            err = hint
        else:
            err = f"python exited with code {rc}: {stderr.strip()[-500:] or '(no stderr)'}"
        return make_result(False, result=out, error=err)
    return make_result(True, result=out)


# ── Error post-processing (#C: Python error translation) ─────────────────────
# Convert raw subprocess tracebacks into one-line, actionable hints the
# small executor model can actually parse and act on. Without this the
# model sees a 30-line traceback, doesn't know which line matters, and
# typically tries the same broken code with a tweak — burning rounds.
import re as _re

# Patterns are tried in order; first match wins. Each entry is
# (regex, hint_fn) where hint_fn(match) -> short actionable string.
_PY_ERROR_PATTERNS = [
    # ModuleNotFoundError: No module named 'X' or 'X.y'
    (
        _re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]"),
        lambda m: (
            f"missing Python package '{m.group(1).split('.')[0]}' in the "
            f"agent's venv — install with: pip install {m.group(1).split('.')[0]}"
        ),
    ),
    # ImportError variant — covers "cannot import name 'X' from 'Y'"
    (
        _re.compile(r"ImportError: cannot import name ['\"]([^'\"]+)['\"] from ['\"]([^'\"]+)['\"]"),
        lambda m: (
            f"can't import '{m.group(1)}' from '{m.group(2)}' — package version mismatch "
            f"or wrong name. Try a different import path or upgrade the package."
        ),
    ),
    # FileNotFoundError: ... 'path'
    (
        _re.compile(r"FileNotFoundError: \[Errno 2\] No such file or directory: ['\"]([^'\"]+)['\"]"),
        lambda m: (
            f"file not found: '{m.group(1)}'. Check the path is correct and "
            f"relative to the workspace (use csv_describe / list_directory to find it first)."
        ),
    ),
    # KeyError on dataframe / dict access
    (
        _re.compile(r"KeyError: ['\"]([^'\"]+)['\"]"),
        lambda m: (
            f"key/column '{m.group(1)}' not found in DataFrame. Run csv_describe "
            f"FIRST to see the exact column names — they are case-sensitive and "
            f"may include spaces or different language characters than you expect."
        ),
    ),
    # IndentationError — show the actual issue line
    (
        _re.compile(r"IndentationError: (.+?)(?:\n|$)"),
        lambda m: (
            f"Python indentation error: {m.group(1).strip()}. Use 4 spaces "
            f"consistently. Mixed tabs/spaces will also trigger this."
        ),
    ),
    # SyntaxError
    (
        _re.compile(r"SyntaxError: (.+?)(?:\n|$)"),
        lambda m: (
            f"Python syntax error: {m.group(1).strip()}. Re-check quotes, "
            f"parentheses, colons, and f-string braces."
        ),
    ),
    # AttributeError: 'X' object has no attribute 'Y'
    (
        _re.compile(r"AttributeError: ['\"]?(\w+)['\"]? object has no attribute ['\"]([^'\"]+)['\"]"),
        lambda m: (
            f"`{m.group(1)}` object has no attribute `{m.group(2)}` — likely a "
            f"typo or a method name that doesn't exist on this type. Check the "
            f"docs for {m.group(1)} or use dir() to list available attributes."
        ),
    ),
    # NameError: name 'X' is not defined
    (
        _re.compile(r"NameError: name ['\"]?(\w+)['\"]? is not defined"),
        lambda m: (
            f"undefined name `{m.group(1)}` — define the variable / import the "
            f"module before use, or check spelling. Note: pandas is `pd`, "
            f"numpy is `np` (already imported in your runtime)."
        ),
    ),
    # ValueError — common pandas mistakes
    (
        _re.compile(r"ValueError: (.+?)(?:\n|$)"),
        lambda m: (
            f"value error: {m.group(1).strip()[:160]}. Check input data shape / "
            f"types match what the function expects."
        ),
    ),
    # TypeError — likely wrong argument
    (
        _re.compile(r"TypeError: (.+?)(?:\n|$)"),
        lambda m: (
            f"type error: {m.group(1).strip()[:160]}. Verify argument types "
            f"and signatures."
        ),
    ),
    # Permission denied
    (
        _re.compile(r"PermissionError: \[Errno 13\] (.+?)(?:\n|$)"),
        lambda m: (
            f"permission denied: {m.group(1).strip()}. The agent runs in the "
            f"workspace dir; writing outside it usually needs explicit setup."
        ),
    ),
    # Connection refused (httpx / requests)
    (
        _re.compile(r"(?:ConnectionError|ConnectionRefused|HTTPError).*?\b(?:Refused|refused|Connection refused|Name or service not known)\b"),
        lambda m: (
            "network connection refused — check the URL is reachable and the "
            "service is running. Local-only tasks shouldn't hit external URLs."
        ),
    ),
]


def _python_error_hint(stderr: str) -> Optional[str]:
    """Convert a raw subprocess traceback into a one-line, actionable hint.
    Returns None if no pattern matched — caller falls back to raw stderr."""
    if not stderr:
        return None
    for pattern, hint_fn in _PY_ERROR_PATTERNS:
        m = pattern.search(stderr)
        if m:
            try:
                return hint_fn(m)
            except Exception:  # noqa: BLE001
                continue
    return None


# Back-compat alias — the older call site used _missing_package_hint.
# Keep it pointing at the broader translator so downstream code still
# benefits from the wider pattern coverage.
_missing_package_hint = _python_error_hint


# ── Subprocess runner ────────────────────────────────────────────────────────

async def _run_python(code: str, cwd: str, timeout: int) -> Tuple[int, str, str]:
    """Run a Python snippet via the same interpreter the server uses.

    We invoke ``sys.executable -c '<code>'`` so the agent inherits any
    packages installed in the server's venv (pandas, numpy, etc.).
    """
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c", code,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -1, "", f"timeout after {timeout}s"
    return (
        proc.returncode if proc.returncode is not None else -1,
        stdout_b.decode("utf-8", errors="replace"),
        stderr_b.decode("utf-8", errors="replace"),
    )


# ── Schemas ───────────────────────────────────────────────────────────────────

_SCHEMA_DESCRIBE: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "csv_describe",
        "description": (
            "Quick tabular-file survey: shape, dtypes, head(5), and describe() "
            "per column. Supports CSV, TSV, Excel (.xlsx/.xls/.xlsm), Parquet, "
            "and JSON. Always call this FIRST when handed a data file — never "
            "guess column names."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Workspace-relative or absolute path to the CSV file.",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional override for the workspace root.",
                },
            },
            "required": ["path"],
        },
    },
}

_SCHEMA_PYTHON_EXEC: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "python_exec",
        "description": (
            "Run Python (with pandas, numpy preloaded) in a sandboxed subprocess. "
            "If `path` is supplied, the file is auto-loaded as DataFrame `df` "
            "(supports .csv, .tsv, .json, .parquet, .xlsx, .xls, .xlsm). "
            "Use `print(...)` to return "
            "results — only stdout is surfaced. Common use: compute repurchase "
            "rate, group-by aggregations, time-series resampling, basic charts "
            "(emit JSON the chat layer renders)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python source. `df` is pre-loaded if `path` is given. "
                        "Imports already in scope: json, sys, math, pandas as pd, "
                        "numpy as np. Print whatever you want returned."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": "Optional CSV/Parquet/JSON to pre-load as `df`.",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Optional workspace root override.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Hard subprocess timeout in seconds (default 60).",
                },
            },
            "required": ["code"],
        },
    },
}


ANALYTIC_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    (csv_describe, _SCHEMA_DESCRIBE,    "low"),
    (python_exec,  _SCHEMA_PYTHON_EXEC, "medium"),
]
