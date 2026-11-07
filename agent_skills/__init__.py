"""
Skills: how-to guides the model reads when a task calls for them.

Each skill is a folder here with a SKILL.md — the same format as Claude's
skills (agentskills.io): YAML front matter with `name` and `description`,
then the guide. Other files in the folder (references, examples) are
read on demand with the use_skill tool. A new skill is a new folder; no
code changes.

How a skill reaches the model (2026-11-08):
  * Before planning, the fast model sees every skill's one-line
    description and picks the ones this request needs (pick) — the
    model's judgment, not keyword rules.
  * The picked guides go into the planner's prompt and onto the end of
    every subtask's system prompt (after the cached prefix).
  * Mid-task the executor can read any other skill with use_skill; its
    tool description lists them all.

Python "skills" (skill_factory.py, skills/) are a different thing:
generated functions that run. These are know-how the model follows.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SKILLS_DIR = os.path.dirname(os.path.abspath(__file__))
_MAX_PICK = 3


@dataclass
class Skill:
    name: str
    description: str
    body: str
    folder: str

    def files(self) -> List[str]:
        """Other files in the skill's folder, relative to it."""
        out = []
        for root, _dirs, names in os.walk(self.folder):
            for n in sorted(names):
                rel = os.path.relpath(os.path.join(root, n), self.folder)
                if rel != "SKILL.md" and not n.startswith("."):
                    out.append(rel)
        return out


def parse(text: str, folder: str = "") -> Optional[Skill]:
    """SKILL.md → Skill; None without a name and description."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if not m:
        return None
    meta: Dict[str, str] = {}
    for line in m.group(1).splitlines():
        k, sep, v = line.partition(":")
        if sep and k.strip() and not line.startswith((" ", "\t")):
            meta[k.strip().lower()] = v.strip().strip("\"'")
    name, desc = meta.get("name", ""), meta.get("description", "")
    if not name or not desc:
        return None
    return Skill(name=name, description=desc, body=m.group(2).strip(), folder=folder)


_cache: Dict[str, Any] = {"key": None, "skills": {}}


def load() -> Dict[str, Skill]:
    """Every skill under SKILLS_DIR, by name. Re-read when a SKILL.md changes."""
    paths = []
    for entry in sorted(os.listdir(SKILLS_DIR)):
        p = os.path.join(SKILLS_DIR, entry, "SKILL.md")
        if os.path.isfile(p):
            paths.append(p)
    key = tuple((p, os.path.getmtime(p)) for p in paths)
    if key != _cache["key"]:
        skills: Dict[str, Skill] = {}
        for p in paths:
            try:
                with open(p, encoding="utf-8") as fh:
                    s = parse(fh.read(), os.path.dirname(p))
            except OSError as exc:
                logger.warning("Skill %s unreadable: %s", p, exc)
                continue
            if s:
                skills[s.name] = s
            else:
                logger.warning("Skill %s has no name/description front matter — skipped", p)
        _cache.update(key=key, skills=skills)
    return _cache["skills"]


def index_text() -> str:
    return "\n".join(f"- {s.name}: {s.description}" for s in load().values())


def guide_text(names: List[str]) -> str:
    """The picked skills, ready to append to a prompt."""
    skills = load()
    parts = []
    for n in names:
        s = skills.get(n)
        if s:
            parts.append(f"[Skill: {s.name} — follow this guide]\n{s.body}")
    return "\n\n".join(parts)


PICK_SYSTEM = """You choose which how-to guides (skills) an AI agent should read before doing a request.
Read the request and the skills' descriptions. Pick a skill only when its description fits what this request needs; most requests need none.
Reply with JSON only: {"skills": ["name", ...]} — an empty list when none fits.

Skills:
"""


def parse_pick(text: str) -> List[str]:
    """Names the model picked that exist; [] for anything unusable."""
    names = load()
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.S)
        try:
            data = json.loads(m.group(0)) if m else {}
        except ValueError:
            data = {}
    picked = data.get("skills") if isinstance(data, dict) else data
    if not isinstance(picked, list):
        return []
    out: List[str] = []
    for n in picked:
        n = str(n).strip()
        if n in names and n not in out:
            out.append(n)
    return out[:_MAX_PICK]


async def pick(orch: Any, goal: str, task_id: str = "") -> List[str]:
    """The skills this request needs, chosen by the fast model."""
    if not load():
        return []
    import fast_model
    reply = await fast_model.ask(orch, PICK_SYSTEM + index_text(), goal[:2000],
                                 max_tokens=80, label="[skills]", json_mode=True,
                                 task_id=task_id)
    return parse_pick(reply)
