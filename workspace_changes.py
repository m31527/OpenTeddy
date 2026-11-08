"""
Which files in the workspace a task or a step actually changed.

A fact the models can't talk around. The 2026-11-08 eval: the executor
named both causes of a display bug, changed nothing, and the summary
said "已調整…主角已可見". Comparing the workspace before and after
catches every way of writing — write_file, edit_file, sed, a script.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".page_look",
              ".mypy_cache", ".pytest_cache", "dist", "build", ".next"}
_MAX_FILES = 5000

State = Dict[str, Tuple[int, int]]


def snapshot(root: Optional[str] = None) -> Optional[State]:
    """path → (mtime_ns, size) for files under the workspace; None when
    there are too many to watch cheaply or the folder is unreadable."""
    if root is None:
        from config import effective_workspace_dir
        root = effective_workspace_dir()
    state: State = {}
    try:
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if x not in _SKIP_DIRS]
            for f in files:
                p = os.path.join(d, f)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                state[os.path.relpath(p, root)] = (st.st_mtime_ns, st.st_size)
                if len(state) > _MAX_FILES:
                    return None
    except OSError:
        return None
    return state


def changed_since(before: Optional[State], root: Optional[str] = None) -> Optional[List[str]]:
    """Files added or modified since `before`, sorted; None if unknown."""
    if before is None:
        return None
    after = snapshot(root)
    if after is None:
        return None
    return sorted(p for p, sig in after.items() if before.get(p) != sig)


def describe(changed: Optional[List[str]]) -> str:
    """One line for a prompt."""
    if changed is None:
        return "實際改動或新增的檔案：（無法得知）"
    if not changed:
        return "實際改動或新增的檔案：（沒有 — 這次沒有任何檔案被改動或新增）"
    shown = ", ".join(changed[:15]) + (f" …等 {len(changed)} 個" if len(changed) > 15 else "")
    return f"實際改動或新增的檔案：{shown}"
