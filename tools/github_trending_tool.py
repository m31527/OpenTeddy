"""
OpenTeddy GitHub Trending Tool
─────────────────────────────────────────────────────────────────────────────
A dedicated, *tested-working* tool for fetching the GitHub Trending list.

Why this exists as a first-class tool instead of "just ask python_exec":

We tried "use python_exec with these CSS selectors" first. Small planner
models (Gemma 4b, qwen2.5:4b) couldn't reliably generate the scraping code
even with a step-by-step recipe — they'd substitute wrong selectors,
use BeautifulSoup methods that don't exist, or treat the HTML as Markdown
and `markdown.parse()` it (which silently returns nothing). End result:
the user's "list top 10 trending repos" came back with rows of
``[N/A](#) Description`` — the script ran, the data extraction failed,
the output template filled in placeholders.

So this tool moves the parsing into well-tested Python code that the
planner *calls* rather than *writes*. The planner just emits
``github_trending(top_n=10)`` and the tool guarantees correct output.

Parses ``github.com/trending`` directly (server-rendered HTML; no JS
needed). Polite to GitHub: caches results for 5 min per (since, language)
key so back-to-back lookups don't hammer the page.

Read-only. Risk: low.
"""

from __future__ import annotations

import logging
import re
import ssl
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)


# ── HTTP + SSL helpers ───────────────────────────────────────────────────────


def _ssl_ctx() -> ssl.SSLContext:
    """macOS system Python 3.12 ships without a configured cert store.
    Use certifi's bundle when available — it's a transitive dep of the
    OpenTeddy backend so it's always there in production. Falls back
    to the default context if certifi isn't around (won't happen in
    practice, but defensive)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


_UA = (
    # Looks like a real browser so GitHub doesn't return the trimmed
    # bot-detection variant. Updated 2026-06; bump if the page structure
    # ever changes per UA again.
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15"
)


def _fetch(url: str, timeout: int = 15) -> str:
    """One HTTP GET, decoded UTF-8. Errors bubble up — caller wraps in
    make_result(False, error=...)."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
        body = resp.read()
    return body.decode("utf-8", errors="replace")


# ── Parsing ──────────────────────────────────────────────────────────────────
# Confirmed working as of 2026-06 against live github.com/trending.
# CSS selectors used:
#   - article.Box-row              → one per repo (15 on the page; we slice top_n)
#   - h2 a                          → "{owner}/{name}" link
#   - p                             → description (may be missing)
#   - span[itemprop=programmingLanguage] → primary language (may be missing)
#   - .d-inline-block.float-sm-right    → "N stars today" (may be missing if 0)
#   - a[href$="/stargazers"]        → total star count (often present)
#
# If GitHub redesigns the page (last did in 2021, stable since), update
# the selectors here in ONE place and the tool's contract stays the same.


def _parse_trending(html: str, top_n: int) -> List[Dict[str, Any]]:
    """Return a list of dicts ordered by trending rank."""
    # Lazy-import bs4 so a missing dep surfaces a clear runtime error
    # rather than a startup crash that prevents OpenTeddy from booting
    # entirely.
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover — defensive
        raise RuntimeError(
            "BeautifulSoup4 (bs4) is required by github_trending_tool. "
            "Install with: pip install beautifulsoup4"
        ) from exc

    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("article.Box-row")
    out: List[Dict[str, Any]] = []
    for rank, r in enumerate(rows[:top_n], start=1):
        # Name + URL — clean whitespace; the inner <a> wraps the
        # "{owner}/{name}" with newlines and extra spaces.
        name_el = r.select_one("h2 a")
        href = (name_el.get("href") or "").strip() if name_el else ""
        name = re.sub(r"\s+", "", name_el.get_text()) if name_el else ""
        # Strip leading slash → "owner/repo".
        repo_full = href.lstrip("/") or name or "?"
        repo_url = f"https://github.com/{repo_full}" if "/" in repo_full else ""

        # Description (may be missing for trending repos with empty repo
        # descriptions — uncommon but happens).
        desc_el = r.select_one("p")
        description = desc_el.get_text(strip=True) if desc_el else ""

        # Primary programming language.
        lang_el = r.select_one("span[itemprop=programmingLanguage]")
        language = lang_el.get_text(strip=True) if lang_el else None

        # "N stars today" — note the comma in 4-digit star counts.
        stars_today_raw = ""
        stars_today_count: Optional[int] = None
        stars_today_el = r.select_one(".d-inline-block.float-sm-right")
        if stars_today_el:
            stars_today_raw = stars_today_el.get_text(strip=True)
            m = re.search(r"([\d,]+)", stars_today_raw)
            if m:
                try:
                    stars_today_count = int(m.group(1).replace(",", ""))
                except ValueError:
                    pass

        # Total stars (separate from "today") — first <a> ending in
        # /stargazers usually carries the number.
        total_stars: Optional[int] = None
        star_a = r.select_one('a[href$="/stargazers"]')
        if star_a:
            txt = re.sub(r"\s+", "", star_a.get_text())
            # Could be "1,234" or "1.2k". Best-effort parse — only the
            # comma form needs to be exact for sorting later.
            m = re.match(r"([\d,]+)", txt)
            if m:
                try:
                    total_stars = int(m.group(1).replace(",", ""))
                except ValueError:
                    pass

        out.append({
            "rank":         rank,
            "name":         repo_full,
            "url":          repo_url,
            "description":  description,
            "language":     language,
            "stars_today":  stars_today_count,
            "stars_today_raw": stars_today_raw,
            "total_stars":  total_stars,
        })
    return out


# ── Caching ──────────────────────────────────────────────────────────────────
# 5-minute TTL — back-to-back calls in the same agent session reuse the
# previous fetch. Independent of the cyber_skills cache (different file,
# different lifetime).

_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL_S = 300


def _cache_key(since: str, language: Optional[str]) -> str:
    return f"{since}::{(language or '').lower()}"


# ── Public tool ──────────────────────────────────────────────────────────────


async def github_trending(
    since: str = "daily",
    language: Optional[str] = None,
    top_n: int = 10,
) -> Dict[str, Any]:
    """Return the top trending repos from github.com/trending.

    Args:
        since: Window the trending list covers. Accepted values:
               ``"daily"`` (default), ``"weekly"``, ``"monthly"``.
               Anything else falls back to daily.
        language: Optional slug to filter by language, matching GitHub's
               own filter. e.g. ``"python"``, ``"rust"``, ``"go"``,
               ``"typescript"``. ``None`` = all languages.
        top_n: How many to return. Default 10, max 25 (GitHub only
               renders ~25 on the page).

    Returns the standard ``make_result`` shape with
    ``result = {"since": ..., "language": ..., "repos": [{rank, name,
    url, description, language, stars_today, ...}, ...]}``.

    Use this instead of ``python_exec`` + ``browser_fetch`` when the
    goal mentions GitHub trending — the planner generates the
    scraping code wrong roughly 50% of the time, this tool's parsing
    is tested.
    """
    start = time.monotonic()
    since_clean = (since or "daily").strip().lower()
    if since_clean not in ("daily", "weekly", "monthly"):
        since_clean = "daily"
    lang_clean = (language or "").strip().lower() or None
    n = max(1, min(int(top_n or 10), 25))

    # Cache hit?
    key = _cache_key(since_clean, lang_clean)
    now = time.monotonic()
    cached = _CACHE.get(key)
    if cached and (now - cached["fetched_at"]) < _CACHE_TTL_S:
        repos = cached["repos"][:n]
        return make_result(
            True,
            result={
                "since":     since_clean,
                "language":  lang_clean,
                "cached":    True,
                "age_s":     round(now - cached["fetched_at"], 1),
                "repos":     repos,
            },
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    # Build URL. GitHub's trending page accepts ?since= and a
    # language path segment (e.g. /trending/python).
    base = "https://github.com/trending"
    if lang_clean:
        base += "/" + urllib.parse.quote(lang_clean)
    qs = urllib.parse.urlencode({"since": since_clean})
    url = f"{base}?{qs}"

    try:
        html = _fetch(url)
    except Exception as exc:  # noqa: BLE001
        return make_result(
            False,
            error=f"Failed to fetch {url}: {exc}",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    try:
        # Always parse up to 25 so a later call with a bigger top_n
        # can use the cache without re-fetching.
        repos_all = _parse_trending(html, top_n=25)
    except Exception as exc:  # noqa: BLE001
        return make_result(
            False,
            error=f"Parsed 0 repos. Page structure may have changed: {exc}",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    if not repos_all:
        return make_result(
            False,
            error=(
                "Parsed 0 repos from the trending page. GitHub may have "
                "redesigned the layout — update CSS selectors in "
                "tools/github_trending_tool.py::_parse_trending."
            ),
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    _CACHE[key] = {"repos": repos_all, "fetched_at": now}
    return make_result(
        True,
        result={
            "since":     since_clean,
            "language":  lang_clean,
            "cached":    False,
            "repos":     repos_all[:n],
        },
        duration_ms=int((time.monotonic() - start) * 1000),
    )


# ── Schema ───────────────────────────────────────────────────────────────────


_SCHEMA_TRENDING: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "github_trending",
        "description": (
            "Fetch the top trending repos from github.com/trending. "
            "Returns a structured list with name, url, description, "
            "language, today's star gain, and total stars per entry. "
            "Call this when the user's goal mentions GitHub trending / "
            "今日熱門 / top trending repos / weekly trending. "
            "PREFER this over `python_exec` or `browser_fetch` for "
            "trending tasks — the scraping is tested-working, whereas "
            "letting a small model write the scraping code typically "
            "yields N/A placeholders or wrong selectors."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "since": {
                    "type": "string",
                    "description": (
                        "Trending window. One of 'daily' (default), "
                        "'weekly', 'monthly'."
                    ),
                    "enum": ["daily", "weekly", "monthly"],
                },
                "language": {
                    "type": "string",
                    "description": (
                        "Optional language filter slug (e.g. 'python', "
                        "'rust', 'go', 'typescript'). Omit to include "
                        "all languages."
                    ),
                },
                "top_n": {
                    "type": "integer",
                    "description": (
                        "How many top trending repos to return. "
                        "Default 10, max 25."
                    ),
                },
            },
            "required": [],
        },
    },
}


# ── Export ───────────────────────────────────────────────────────────────────


GITHUB_TRENDING_TOOLS = [
    (github_trending, _SCHEMA_TRENDING, "low"),  # type: RiskLevel
]
