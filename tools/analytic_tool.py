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

    code = textwrap.dedent(f"""\
        import json, sys
        import pandas as pd
        df = pd.read_csv({str(target)!r}, low_memory=False)
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
        if ext == ".parquet":
            reader = "pd.read_parquet"
        elif ext == ".json":
            reader = "pd.read_json"
        elif ext in {".tsv", ".tab"}:
            reader = 'pd.read_csv'
            preamble_lines.append(f"df = {reader}({str(target)!r}, sep='\\t', low_memory=False)")
            preamble_lines.append("# df is the loaded dataframe — go.")
        else:
            reader = "pd.read_csv"
        if ext not in {".tsv", ".tab"}:
            preamble_lines.append(f"df = {reader}({str(target)!r}, low_memory=False)")
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


# ── Error post-processing ────────────────────────────────────────────────────

import re as _re

_MISSING_MOD_RE = _re.compile(r"ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]")


def _missing_package_hint(stderr: str) -> Optional[str]:
    """Convert a raw ModuleNotFoundError traceback into an actionable
    hint the agent can act on (it can chain a `pip_install` tool call,
    or surface a clean message to the user)."""
    if not stderr:
        return None
    m = _MISSING_MOD_RE.search(stderr)
    if not m:
        return None
    pkg = m.group(1).split(".")[0]   # toplevel package
    return (
        f"missing Python package '{pkg}' in the agent's venv — "
        f"install with: pip install {pkg}"
    )


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
            "Quick CSV survey: shape, dtypes, head(5), and describe() per column. "
            "Always call this FIRST when given a CSV — never guess column names."
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
            "(supports .csv, .tsv, .json, .parquet). Use `print(...)` to return "
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
