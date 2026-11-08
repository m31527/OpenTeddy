"""
read_guide — read a skill (a how-to guide in agent_skills/) mid-task.

Called a "guide" to the model: there "skill" means a generated Python
function (skill_needed → skill_factory).

The skills a task needs are usually picked before planning and already
in the prompt (agent_skills.pick). This lets the agent read another one
when the work turns out to need it, or a reference file a guide points
to. The description lists every skill, built once at startup, so it
stays in the cached prompt prefix.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Tuple

import agent_skills
from tool_registry import RiskLevel, make_result


async def read_guide(name: str, file: str = "") -> Dict[str, Any]:
    start = time.monotonic()
    ms = lambda: int((time.monotonic() - start) * 1000)  # noqa: E731
    skills = agent_skills.load()
    s = skills.get((name or "").strip())
    if not s:
        return make_result(False, error=f"No guide named {name!r}. Guides: {', '.join(skills) or 'none'}",
                           duration_ms=ms())
    if file:
        path = os.path.realpath(os.path.join(s.folder, file))
        if not path.startswith(os.path.realpath(s.folder) + os.sep) or not os.path.isfile(path):
            return make_result(False, error=f"{file!r} is not in guide {s.name}. Files: {', '.join(s.files()) or 'none'}",
                               duration_ms=ms())
        with open(path, encoding="utf-8", errors="replace") as fh:
            return make_result(True, {"guide": s.name, "file": file, "content": fh.read()[:20000]},
                               duration_ms=ms())
    return make_result(True, {"guide": s.name, "text": s.body, "files": s.files()}, duration_ms=ms())


def _schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "read_guide",
            "description": (
                "Read a how-to guide for a kind of work before doing that kind of work, and follow "
                "it. Guides already in your instructions don't need reading again. "
                "Guides:\n" + (agent_skills.index_text() or "(none)")
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The guide's name."},
                    "file": {"type": "string",
                             "description": "Optional: a file in the guide's folder that it mentions."},
                },
                "required": ["name"],
            },
        },
    }


READ_GUIDE_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    (read_guide, _schema(), "low"),
]
