"""
OpenTeddy Cyber-Skills Lookup Tool
─────────────────────────────────────────────────────────────────────────────
A search tool over the 754-entry cybersecurity skill index sourced from
mukul975/Anthropic-Cybersecurity-Skills (Apache 2.0), mapped to MITRE
ATT&CK / NIST CSF / D3FEND / ATLAS / AI RMF.

Design philosophy
-----------------
These are not executable Python skills — they're documentation-style
workflows in agentskills.io's SKILL.md format. Trying to auto-execute
them would be wrong: many call out shell commands the user's system may
not have (Volatility, Autopsy, MITRE Navigator binaries…), reference
forensic tools that need licences, or assume evidence files in fixed
paths.

So the integration is a LOOKUP tool, not a runtime. The agent decides
when a goal could benefit from cyber-skill guidance ("walk through
analysing an email header for phishing", "what's the standard incident
response for a ransomware case", "MITRE technique T1566.001 — what's
the standard analysis workflow") and calls `cyber_skill_lookup(query)`.
The tool returns the top matching skills with their full Workflow
markdown — the LLM reads the steps and adapts them to the user's
actual scenario.

Index lifecycle
---------------
The index lives at `cyber_skills/index.json` and is built/refreshed by
`cyber_skills/update.py` (manual or scheduled). The tool loads it
lazily on first call and caches in-memory across invocations of the
same Python process.

Read-only by design — never writes to the workspace. Risk: low.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)

# ── Index loading ─────────────────────────────────────────────────────────────

# Lazy: read from disk on first lookup, hold in memory for the process.
# Re-running `cyber_skills/update.py` regenerates the JSON; users restart
# the server to pick up the new index (the file is small enough — ~ a
# few MB — that auto-reload would be cheap, but explicit restart matches
# the rest of OpenTeddy's "settings change → restart" pattern).
_INDEX_CACHE: Optional[Dict[str, Any]] = None
_INDEX_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "cyber_skills",
    "index.json",
)


def _load_index() -> Dict[str, Any]:
    """Read index.json into memory once per process. Returns an empty
    structure (not an exception) if the file is missing — the tool
    surfaces a clear 'run update.py' error rather than crashing."""
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
    try:
        with open(_INDEX_PATH, "r", encoding="utf-8") as fh:
            _INDEX_CACHE = json.load(fh)
    except FileNotFoundError:
        _INDEX_CACHE = {"_meta": {"skill_count": 0}, "skills": []}
    except Exception as exc:  # noqa: BLE001
        logger.warning("cyber_skills index failed to load: %s", exc)
        _INDEX_CACHE = {"_meta": {"skill_count": 0}, "skills": []}
    return _INDEX_CACHE


# ── Search ────────────────────────────────────────────────────────────────────
# Simple keyword scoring — no embedding dependency. The index has
# clean fields (description, tags, MITRE/NIST mappings, domain/
# subdomain) so token matching is precise enough for the agent's
# "find me the phishing investigation workflow" use case. If embeddings
# turn out to matter later we can swap in ChromaDB; for now skip the
# 754-row indexing cost on every server boot.

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9.+_-]{1,}")


def _tokenise(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _score(skill: Dict[str, Any], query_tokens: List[str]) -> float:
    """Per-skill score. Weighted because matches in (description,
    tags, framework ids) are stronger signals than matches buried in
    the workflow body."""
    if not query_tokens:
        return 0.0
    fields: List[Tuple[float, List[str]]] = [
        (5.0, _tokenise(skill.get("name", ""))),
        (4.0, _tokenise(skill.get("description", ""))),
        (3.0, [t.lower() for t in (skill.get("tags") or [])]),
        (3.0, [m.lower() for m in (skill.get("mitre_attack") or [])]),
        (2.5, [m.lower() for m in (skill.get("nist_csf") or [])]),
        (2.5, [m.lower() for m in (skill.get("atlas") or [])]),
        (2.0, _tokenise(skill.get("subdomain", ""))),
        (1.5, _tokenise(skill.get("domain", ""))),
        (0.5, _tokenise(skill.get("body", ""))),
    ]
    score = 0.0
    for weight, field_tokens in fields:
        if not field_tokens:
            continue
        ftset = set(field_tokens)
        for qt in query_tokens:
            if qt in ftset:
                score += weight
    return score


# ── Public tool ───────────────────────────────────────────────────────────────

# Cap returned body so even multi-result queries don't blow the
# executor context budget. 4 000 chars per skill × top 3 = 12 000 chars
# upper bound, manageable on small models.
_MAX_BODY_CHARS_PER_SKILL = 4000
_DEFAULT_TOP_N = 3
_MAX_TOP_N = 8


async def cyber_skill_lookup(
    query: str,
    top_n: Optional[int] = None,
    include_body: bool = True,
) -> Dict[str, Any]:
    """Search the cyber-skills index and return the top matches.

    Args:
        query: Natural-language description of what you're trying to do
               OR a specific MITRE/NIST/D3FEND identifier (e.g.
               "T1566.001", "RS.MA-01"). Both work — the scorer
               weights framework matches higher than body matches.
        top_n: How many skills to return. Default 3, max 8.
        include_body: When True (default), each match includes the
               full Workflow markdown. When False, only metadata —
               useful for "list relevant skills" without paying the
               context cost.

    Returns the standard make_result shape with result =
        {"query": ..., "match_count": N, "skills": [...]}.

    Read-only; risk: low.
    """
    start = time.monotonic()
    q = (query or "").strip()
    if not q:
        return make_result(False, error="query is empty", duration_ms=0)

    n = max(1, min(int(top_n or _DEFAULT_TOP_N), _MAX_TOP_N))
    idx = _load_index()
    skills = idx.get("skills") or []

    if not skills:
        return make_result(
            False,
            error=(
                "Cyber-skill index is empty. Run "
                "`python cyber_skills/update.py` from the repo root to "
                "build it (one-off ~12 min)."
            ),
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    query_tokens = _tokenise(q)
    # Also tokenise raw identifiers (T1566.001) which _TOKEN_RE catches.
    scored: List[Tuple[float, Dict[str, Any]]] = [
        (_score(s, query_tokens), s) for s in skills
    ]
    scored = [(sc, s) for sc, s in scored if sc > 0]
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:n]

    matches: List[Dict[str, Any]] = []
    for sc, s in top:
        entry = {
            "name":         s.get("name"),
            "description":  s.get("description"),
            "domain":       s.get("domain"),
            "subdomain":    s.get("subdomain"),
            "tags":         s.get("tags") or [],
            "mitre_attack": s.get("mitre_attack") or [],
            "nist_csf":     s.get("nist_csf") or [],
            "atlas":        s.get("atlas") or [],
            "score":        round(sc, 2),
            "upstream_path": s.get("upstream_path"),
        }
        if include_body:
            body = s.get("body") or ""
            if len(body) > _MAX_BODY_CHARS_PER_SKILL:
                body = body[:_MAX_BODY_CHARS_PER_SKILL] + (
                    f"\n\n…(truncated — full skill is at "
                    f"https://github.com/mukul975/Anthropic-Cybersecurity-Skills/blob/main/{s.get('upstream_path','')})"
                )
            entry["body"] = body
        matches.append(entry)

    return make_result(
        True,
        result={
            "query":        q,
            "total_skills_in_index": len(skills),
            "match_count":  len(matches),
            "skills":       matches,
        },
        duration_ms=int((time.monotonic() - start) * 1000),
    )


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA_LOOKUP: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "cyber_skill_lookup",
        "description": (
            "Search the indexed expert-skill catalogue and return the "
            "top matching workflows. Current sources:\n"
            "  • mukul975/Anthropic-Cybersecurity-Skills (754 entries) — "
            "incident response, digital forensics, malware analysis, "
            "threat hunting, red/blue team, cloud security, AppSec, "
            "AI/ML threats. Maps to MITRE ATT&CK / NIST CSF / D3FEND / "
            "ATLAS / NIST AI RMF.\n"
            "  • mvanhorn/last30days-skill (1 entry) — multi-platform "
            "trend research across Reddit / X / YouTube / HN / Polymarket "
            "with engagement-weighted ranking.\n"
            "Call this when the user's goal touches any of those domains. "
            "Also accepts framework identifiers directly (e.g. "
            "'T1566.001', 'RS.MA-01'). Read-only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Natural-language description of the task OR a "
                        "framework ID. e.g. 'phishing email header "
                        "analysis', 'analyzing Cobalt Strike beacon', "
                        "'T1566.001', 'NIST CSF RS.AN-01'."
                    ),
                },
                "top_n": {
                    "type": "integer",
                    "description": (
                        "How many top matches to return. Default 3, "
                        "max 8."
                    ),
                },
                "include_body": {
                    "type": "boolean",
                    "description": (
                        "When true (default) returns the full Workflow "
                        "markdown for each match. Set false for a "
                        "lightweight 'list relevant skills' query that "
                        "skips the body."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


# ── Export ────────────────────────────────────────────────────────────────────

CYBER_SKILLS_TOOLS = [
    (cyber_skill_lookup, _SCHEMA_LOOKUP, "low"),  # type: RiskLevel
]
