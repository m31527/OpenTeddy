"""
OpenTeddy Search Tool

Web search for the agent — closes the "local model has no idea what
happened after its training cutoff" gap. Powered by Brave Search API.

Why this matters for the OpenTeddy positioning: a small local LLM is
useless for current events / version numbers / today's news without
augmentation. Adding `web_search` to chat mode means the local model
can confidently say "let me look that up" and ground its answer in
real sources, instead of hallucinating plausible-sounding facts. This
is exactly the kind of tooling that turns a 2B/4B local model from a
toy into a real assistant.

Tool gracefully no-ops when ``BRAVE_SEARCH_API_KEY`` is not set so
local-only / privacy-strict installs aren't forced to enable an
external API. The tool registry exposes the schema either way; the
tool just returns an explanatory error if the key is missing.

Pricing (as of 2026): free tier 2,000 queries/month, Data for Search
plan $5/month for 20,000 queries. See https://brave.com/search/api/.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

import httpx

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)


# Brave Search API endpoint. Documented at
# https://api-dashboard.search.brave.com/app/documentation/web-search/get-started
_BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

# Hard cap on results so we don't blow the executor's context budget.
# 5 is the sweet spot — enough for the model to triangulate facts,
# small enough that even 2B-class models can read all of them.
_MAX_RESULTS_HARD_CAP = 8

# Per-result snippet truncation. Brave returns up to ~250 chars per
# `description` field; trimming further keeps the total tool result
# under ~3 KB even with 8 results.
_SNIPPET_MAX_CHARS = 240

# Default timeout — search APIs are usually <1 s but we give them
# headroom in case Brave is having a bad afternoon. Tool runs inside
# an asyncio.gather so a stuck request blocks the whole round.
_HTTP_TIMEOUT_S = 8.0


def _api_key() -> str:
    """Read the API key live each call so /settings hot-reloads work
    without a server restart. Falls back to the env var if the user
    hasn't set it via the UI."""
    # Late import — avoids a circular config <-> tools dependency at
    # module-load time.
    from config import config as cfg
    return (
        getattr(cfg, "brave_search_api_key", "")
        or os.getenv("BRAVE_SEARCH_API_KEY", "")
    ).strip()


async def web_search(
    query: str,
    max_results: int = 5,
    freshness: str = "",
) -> str:
    """Search the web via Brave Search and return the top hits as a
    markdown-formatted block: title, URL, snippet per result.

    Args mirror Brave's public params plus a sane local default. The
    return string is intentionally markdown so the chat-mode model can
    quote it directly back to the user with working hyperlinks.
    """
    q = (query or "").strip()
    if not q:
        return "ERROR: search query is empty."

    key = _api_key()
    if not key:
        # Soft failure — the schema is registered but the API isn't
        # configured. Tell the model exactly what's missing so it can
        # surface a clean message instead of pretending the search
        # worked.
        return (
            "ERROR: web_search is not configured. Set BRAVE_SEARCH_API_KEY "
            "in .env or in Settings → Search to enable web search. "
            "Without it, answer from your training data and explicitly "
            "warn the user that the result may be out of date."
        )

    n = max(1, min(int(max_results or 5), _MAX_RESULTS_HARD_CAP))
    params: Dict[str, Any] = {
        "q":     q,
        "count": n,
        # Default to global English-leaning results — the agent can ask
        # for a specific country with a separate param if needed later.
        "country": "us",
    }
    # Brave's freshness filter: pd (past day), pw (past week),
    # pm (past month), py (past year). Pass-through if the model
    # set it; ignore otherwise.
    if freshness in {"pd", "pw", "pm", "py"}:
        params["freshness"] = freshness

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": key,
    }

    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
            resp = await client.get(
                _BRAVE_SEARCH_ENDPOINT,
                params=params,
                headers=headers,
            )
    except httpx.TimeoutException:
        return f"ERROR: web_search timed out after {_HTTP_TIMEOUT_S}s"
    except Exception as exc:  # noqa: BLE001
        logger.exception("web_search HTTP error: %s", exc)
        return f"ERROR: web_search HTTP error: {exc}"

    if resp.status_code == 401:
        return "ERROR: BRAVE_SEARCH_API_KEY is invalid or expired."
    if resp.status_code == 429:
        return (
            "ERROR: Brave Search rate-limited the request "
            "(too many queries this minute / month). "
            "Wait a few seconds and try again."
        )
    if resp.status_code >= 400:
        return (
            f"ERROR: web_search returned HTTP {resp.status_code}: "
            f"{resp.text[:200]}"
        )

    try:
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: web_search returned non-JSON: {exc}"

    web_block = (payload.get("web") or {}).get("results") or []
    if not web_block:
        return f"No results for: {q}"

    # Render markdown so the chat-mode model can quote it verbatim and
    # the UI's existing markdown renderer makes the URLs clickable
    # (which then route through openInBrowser inside Tauri).
    lines: List[str] = [f'### Search results for "{q}"', ""]
    for i, hit in enumerate(web_block[:n], 1):
        title = (hit.get("title") or "Untitled").strip()
        url   = (hit.get("url")   or "").strip()
        desc  = (hit.get("description") or "").strip()
        # Brave description fields contain inline `<strong>` markup
        # for query-term highlighting. Strip it — the chat-mode model
        # would otherwise have to know HTML to make sense of it.
        for tag in ("<strong>", "</strong>", "<b>", "</b>"):
            desc = desc.replace(tag, "")
        if len(desc) > _SNIPPET_MAX_CHARS:
            desc = desc[: _SNIPPET_MAX_CHARS - 1] + "…"
        lines.append(f"{i}. **[{title}]({url})**")
        if desc:
            lines.append(f"   {desc}")
        lines.append("")
    lines.append(
        "_Use these as ground truth — cite the URLs inline when you quote facts._"
    )
    return "\n".join(lines)


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA_WEB_SEARCH: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the public web for current information. USE THIS when "
            "the user's question depends on facts your training data may "
            "not have — current events, today's news, recent product / "
            "library version numbers, current prices, sports scores, "
            "weather, schedules, anything time-sensitive. Do NOT use this "
            "for general explanations, math, code reasoning, or "
            "well-known historical facts — those don't need fresh data "
            "and search adds latency. Returns a markdown list of top hits "
            "with title, URL, and snippet — quote the URLs back when you "
            "report facts to the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Write it the way a human would "
                        "type it into Google — keywords, not full "
                        "sentences. Include the year for time-sensitive "
                        "topics ('Pydantic 2.x changes 2026')."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        "How many results to return (1–8, default 5). "
                        "Higher = more facts to triangulate but more "
                        "context burned."
                    ),
                    "default": 5,
                    "minimum": 1,
                    "maximum": _MAX_RESULTS_HARD_CAP,
                },
                "freshness": {
                    "type": "string",
                    "description": (
                        "Optional time filter. 'pd' = past day (today's "
                        "news), 'pw' = past week, 'pm' = past month, "
                        "'py' = past year. Leave empty for no filter."
                    ),
                    "enum": ["", "pd", "pw", "pm", "py"],
                    "default": "",
                },
            },
            "required": ["query"],
        },
    },
}


# ── Export ────────────────────────────────────────────────────────────────────

# Risk = "low" — read-only, no side effects on the user's machine. The
# external API call is no riskier than the http_get tool that already
# exists.
SEARCH_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    (web_search, _SCHEMA_WEB_SEARCH, "low"),
]
