"""
OpenTeddy skill matcher — "do I already know how to do this?"

Replaces the executor's old naive substring check with a scored,
deterministic matcher (skill-plus spec §4). Deliberately pragmatic:
token overlap across name / capabilities / description, weighted by
signal strength. No vector DB, no LLM call — matching runs on every
task, so it must be instant and free; the LLM's opinion already arrives
via `skill_hint` (the executor's model names the skill it wants, which
short-circuits to confidence 1.0 here).

Threshold comes from config.skill_match_threshold (existing setting) so
operators tune reuse-vs-rebuild without code changes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from models import SkillMetadata, SkillStatus

logger = logging.getLogger(__name__)

# Words too generic to indicate a capability on their own.
_STOPWORDS = {
    "the", "a", "an", "to", "for", "of", "and", "or", "in", "on", "with",
    "this", "that", "my", "me", "please", "use", "using", "then", "into",
    "from", "it", "is", "are", "be", "do", "does", "how", "what", "your",
    "幫", "我", "請", "把", "的", "了", "一", "個", "用", "並", "與", "和",
}


def _tokens(text: str) -> set:
    """Lowercased word tokens; CJK is additionally split per character so
    Chinese goals still overlap with Chinese descriptions."""
    text = (text or "").lower()
    words = set(re.findall(r"[a-z0-9]+", text))
    cjk = set(re.findall(r"[一-鿿]", text))
    # 2-gram CJK tokens carry much more meaning than single characters
    cjk2 = set(a + b for a, b in zip(*(lambda s: (s, s[1:]))("".join(
        re.findall(r"[一-鿿]+", text)))))
    return {t for t in (words | cjk | cjk2) if t not in _STOPWORDS}


@dataclass
class MatchResult:
    matched: bool
    skill_name: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""


def match(
    task_text: str,
    skills: List[SkillMetadata],
    threshold: float,
    skill_hint: Optional[str] = None,
) -> MatchResult:
    """Pick the best existing skill for a task, or report no match.

    Only ACTIVE + enabled skills are considered — a disabled or retired
    skill must never be silently resurrected by fuzzy matching.
    """
    candidates = [
        s for s in skills
        if s.status == SkillStatus.ACTIVE and getattr(s, "enabled", True)
    ]

    # The executor's model explicitly asked for a skill by name — trust it
    # when that skill actually exists and is usable.
    if skill_hint:
        for s in candidates:
            if s.name == skill_hint:
                logger.info("skill.search.matched skill=%s confidence=1.00 "
                            "reason=explicit-hint", s.name)
                return MatchResult(True, s.name, 1.0, "explicit skill_hint")

    task_toks = _tokens(task_text)
    if not task_toks or not candidates:
        logger.info("skill.search.no_match candidates=%d", len(candidates))
        return MatchResult(False, reason="no candidates or empty task")

    best: Optional[SkillMetadata] = None
    best_score = 0.0
    for s in candidates:
        name_toks = _tokens(s.name.replace("_", " "))
        cap_toks = _tokens(" ".join(s.capabilities))
        desc_toks = _tokens(s.description)

        def overlap(toks: set) -> float:
            return len(task_toks & toks) / len(toks) if toks else 0.0

        # Name identity is the strongest signal, declared capabilities
        # next, prose description weakest (wordy, noisy).
        score = min(1.0,
                    0.6 * overlap(name_toks)
                    + 0.3 * overlap(cap_toks)
                    + 0.2 * overlap(desc_toks))
        if score > best_score:
            best, best_score = s, score

    if best is not None and best_score >= threshold:
        logger.info("skill.search.matched skill=%s confidence=%.2f",
                    best.name, best_score)
        return MatchResult(True, best.name, round(best_score, 4),
                           f"token overlap {best_score:.2f} >= {threshold}")
    logger.info("skill.search.no_match best=%s score=%.2f threshold=%.2f",
                best.name if best else None, best_score, threshold)
    return MatchResult(False, None, round(best_score, 4),
                       f"best score {best_score:.2f} < threshold {threshold}")
