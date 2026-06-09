"""
OpenTeddy Threads Search Tool
─────────────────────────────────────────────────────────────────────────────
Sister tool to x_search — but for Meta's Threads (threads.net).

Why this exists as a first-class tool:

The user's first attempt at "整理 threads.com 上最近討論黴菌的熱門推文"
revealed three failure modes stacked on top of each other:
  1. threads.com is NOT the Meta Threads domain (it's threads.NET).
     Anonymous browser_fetch on threads.com hits an unrelated US
     insurance company's redirect → 0 useful content.
  2. No dedicated Threads tool existed, so the planner fell back to
     python_exec to write a scraper. Small models can't reliably
     generate the right selectors, so the result was either empty
     or hallucinated content.
  3. The synthesizer claimed a downloadable HTML file was produced
     without verifying — so the next "let me download that file"
     query found nothing.

This tool fixes #1 and #2: hard-coded canonical URL (threads.net),
hard-coded selectors that match Threads' current DOM, and the same
v1.1.6 anti-bot hardening as x_search (stealth init, rate limit,
sleep window, real mouse-wheel scrolling).

Authentication:
  Threads logins go through Instagram (Meta-owned). Operator logs in
  ONCE via scripts/login-helper.sh — open the Brave window, navigate
  to threads.net, click "Continue with Instagram", complete the
  Instagram login flow, and the session persists in the same Brave
  profile that x_search uses. No separate cookie management.

Risk: medium — same as x_search. Threads has Meta's bot detection
behind it, so use a throwaway Instagram account
(see scripts/setup-throwaway-account.md).
"""

from __future__ import annotations

import asyncio
import logging
import time
import urllib.parse
from typing import Any, Dict, List, Optional

from tool_registry import RiskLevel, make_result

# Reuse the heavy lifting from chrome_attached_tool — attach helper,
# stealth, cookies, rate limit, sleep window all live there.
from tools.chrome_attached_tool import (
    _open_attached_page,
    _rate_limit_check,
    _in_sleep_window,
    _human_pause,
    _SLEEP_WINDOW_START_HOUR,
    _SLEEP_WINDOW_END_HOUR,
)

logger = logging.getLogger(__name__)


# Register threads_search in the rate-limit map at import time so
# chrome_attach_check reports it alongside x_search.
from tools.chrome_attached_tool import _RATE_LIMITS as _CHROME_RATE_LIMITS
_CHROME_RATE_LIMITS.setdefault("threads_search", 20)


# ── DOM extractor ────────────────────────────────────────────────────────────
# Threads' DOM is React-rendered, similar shape to Instagram. Selectors
# pinned to data-testid / role attributes where possible — Meta tends to
# preserve those across UI refreshes more reliably than CSS class hashes.
# Confirmed selectors as of 2026-06; update this single block when
# Meta ships a redesign.

_THREADS_EXTRACTOR_JS = r"""
(() => {
    const out = [];

    // Post containers — Threads wraps each thread in <article>; the
    // search results page has them in the main feed.
    const articles = document.querySelectorAll('article, div[role="article"]');

    for (const art of articles) {
        // Author handle + display name — usually in the first <a>
        // that points at a user profile (/<handle>).
        let handle = "";
        let displayName = "";
        const authorAnchors = art.querySelectorAll('a[href^="/"]');
        for (const a of authorAnchors) {
            const href = a.getAttribute('href') || "";
            // Profile URLs are /@handle (or /@handle/). Skip post
            // permalinks (which are /@handle/post/...) here.
            const m = href.match(/^\/(@[\w.]+)\/?$/);
            if (m) {
                handle = m[1];
                displayName = a.innerText.trim() || handle;
                break;
            }
        }

        // Permalink + timestamp — find the first <a> that points at
        // /@handle/post/...
        let postUrl = null;
        let postedAt = null;
        for (const a of art.querySelectorAll('a[href*="/post/"]')) {
            const href = a.getAttribute('href') || "";
            if (/^\/@[\w.]+\/post\//.test(href)) {
                postUrl = new URL(href, location.origin).href;
                const t = a.querySelector('time');
                if (t) postedAt = t.getAttribute('datetime');
                break;
            }
        }

        // Body text — Threads bundles the actual post body in a
        // span with dir="auto" inside the article. Concatenate to
        // handle multi-paragraph posts.
        const bodyParts = [];
        for (const span of art.querySelectorAll('span[dir="auto"]')) {
            // Skip spans that are clearly just timestamps or author
            // text (the author block also uses dir=auto, but it's
            // wrapped in the author <a> we already captured).
            if (span.closest('a[href^="/@"]')) continue;
            const t = (span.innerText || "").trim();
            if (!t) continue;
            // Avoid grabbing the same text twice if Threads renders
            // it in nested spans.
            if (bodyParts.length && bodyParts[bodyParts.length - 1] === t) continue;
            bodyParts.push(t);
        }
        const body = bodyParts.join("\n").trim();

        // Engagement metrics — Threads uses aria-label like
        // "12 replies" / "84 likes" / "3 reposts".
        function getCount(labelSubstring) {
            for (const el of art.querySelectorAll('[aria-label]')) {
                const lbl = el.getAttribute('aria-label') || "";
                if (lbl.toLowerCase().includes(labelSubstring)) {
                    const m = lbl.match(/([\d,\.]+)/);
                    if (m) return parseFloat(m[1].replace(/,/g, '')) || null;
                }
            }
            return null;
        }
        const replies = getCount('repl');
        const likes   = getCount('like');
        const reposts = getCount('repost');

        // Skip articles with no body — those are usually "follow
        // suggestion" cards Threads sprinkles into search results.
        if (!body || (!postUrl && !handle)) continue;

        out.push({
            handle:      handle,
            displayName: displayName,
            text:        body,
            url:         postUrl,
            posted_at:   postedAt,
            replies:     replies,
            reposts:     reposts,
            likes:       likes,
        });
    }
    return out;
})()
"""


# ── Tool ─────────────────────────────────────────────────────────────────────


async def threads_search(
    query: str,
    top_n: int = 10,
) -> Dict[str, Any]:
    """Search Meta Threads (https://www.threads.net) using the user's
    Instagram-authenticated browser session.

    Args:
        query: Free-form search query (URL-encoded internally).
        top_n: How many posts to return. Default 10, max 50.
               Values > 10 trigger auto-scroll to lazy-load more posts.

    Returns the standard make_result shape with:
        result = {
            "query":  ...,
            "url":    canonical Threads search URL used,
            "total":  N,
            "posts":  [{handle, displayName, text, url, posted_at,
                        replies, reposts, likes}, ...],
        }

    Use this instead of python_exec / browser_fetch when the goal
    mentions Threads / threads.net / Meta 推文. Anonymous Threads is
    heavily restricted; this tool relies on the operator having
    completed the one-time Instagram login via login-helper.sh.

    The query goes to the canonical Meta URL — https://www.threads.net
    — NOT threads.com (which is unrelated to Meta's product). Common
    operator mistake.
    """
    start = time.monotonic()
    q = (query or "").strip()
    if not q:
        return make_result(False, error="query is empty", duration_ms=0)

    # Same anti-bot pre-flight as x_search.
    if _in_sleep_window():
        return make_result(
            False,
            error=(
                f"Inside the sleep window ({_SLEEP_WINDOW_START_HOUR:02d}:00-"
                f"{_SLEEP_WINDOW_END_HOUR:02d}:00 local). Real users don't "
                "search at 3am; running scheduled scrapes overnight is a "
                "fast way to get the throwaway account flagged. Re-run "
                "after the window, or set OPENTEDDY_NO_SLEEP=1 to override."
            ),
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    allowed, count, cap = _rate_limit_check("threads_search")
    if not allowed:
        return make_result(
            False,
            error=(
                f"Rate-limited: {count}/{cap} threads_search calls in the "
                f"last hour. Wait, or raise the cap with "
                f"OPENTEDDY_RATE_LIMIT_THREADS_SEARCH=<n>."
            ),
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    n = max(1, min(int(top_n or 10), 50))
    url = (
        "https://www.threads.net/search?q="
        + urllib.parse.quote(q)
        + "&serp_type=default"
    )

    page = browser = p = None
    try:
        page, browser, p = await _open_attached_page(
            reuse_tab_url_contains="threads.net",
        )
    except RuntimeError as exc:
        return make_result(
            False, error=str(exc),
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except Exception as exc:  # noqa: BLE001
        return make_result(
            False, error=f"Failed to attach to Chrome: {exc}",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=25000)
        # Threads' results lazy-load after the initial DOM is ready; wait
        # a little for the first article to appear, OR for a known
        # logged-out marker. 15s budget covers slow first-paint on
        # Threads' web app.
        try:
            await page.wait_for_selector(
                'article, [data-testid="login-button"], a[href^="/login"]',
                timeout=15000,
            )
        except Exception:
            pass

        # Login check — Threads bounces unauthed users to /login.
        current_url = page.url or ""
        login_signals: List[str] = []
        if "/login" in current_url:
            login_signals.append(f"redirected to {current_url}")
        for sel in (
            '[data-testid="login-button"]',
            'a[href="/login"]',
            'a[href*="/accounts/login"]',
        ):
            if await page.query_selector(sel):
                login_signals.append(f"found {sel}")
                break
        if login_signals:
            return make_result(
                False,
                error=(
                    "Not logged in to Threads. Signals: "
                    + "; ".join(login_signals)
                    + ". Fix: run scripts/login-helper.sh, navigate to "
                    "https://www.threads.net in the Brave window that "
                    "opens, click 'Continue with Instagram', complete "
                    "the Instagram login flow, close the browser. Then "
                    "retry."
                ),
                duration_ms=int((time.monotonic() - start) * 1000),
            )

        # Scroll to lazy-load more posts. Same real-mouse-wheel pattern
        # as x_search; Threads' infinite scroll is triggered by real
        # WheelEvents, not synthetic scrollBy.
        SCROLL_ATTEMPTS = max(0, (n - 8 + 7) // 8)  # initial ~8, each scroll +8
        viewport = page.viewport_size or {"height": 900}
        for _ in range(SCROLL_ATTEMPTS):
            import random
            delta_y = int(viewport["height"] * (1.4 + random.random() * 0.3))
            try:
                await page.mouse.wheel(0, delta_y)
            except Exception:
                await page.evaluate(f"window.scrollBy(0, {delta_y})")
            await _human_pause(0.7, 1.8)

        # Extract.
        raw = await page.evaluate(_THREADS_EXTRACTOR_JS)
        posts = list(raw or [])[:n]
        return make_result(
            True,
            result={
                "query":  q,
                "url":    url,
                "total":  len(posts),
                "posts":  posts,
            },
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except Exception as exc:  # noqa: BLE001
        return make_result(
            False, error=f"threads_search failed: {exc}",
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    finally:
        try:
            if browser is not None:
                await browser.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if p is not None:
                await p.stop()
        except Exception:  # noqa: BLE001
            pass


# ── Schema ───────────────────────────────────────────────────────────────────


_SCHEMA_THREADS_SEARCH: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "threads_search",
        "description": (
            "Search Meta Threads (https://www.threads.net — NOT "
            "threads.com, a common typo for an unrelated site) using "
            "the user's Instagram-authenticated Chrome session. Returns "
            "a structured list of {handle, displayName, text, url, "
            "posted_at, replies, reposts, likes} per post. "
            "PREFER this over python_exec / browser_fetch / "
            "chrome_attached_browse when the goal mentions Threads / "
            "Meta 推文 / threads.net — the dedicated DOM selectors are "
            "tested-working, whereas letting a small model write the "
            "scraping code gives wrong selectors and hallucinated "
            "content. Requires login (via scripts/login-helper.sh)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Free-form search query. Routed to "
                        "https://www.threads.net/search?q=<query>."
                    ),
                },
                "top_n": {
                    "type": "integer",
                    "description": (
                        "How many posts to return. Default 10, max 50. "
                        "Values > 10 trigger auto-scroll inside the "
                        "page to lazy-load more results."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}


# ── Export ───────────────────────────────────────────────────────────────────


THREADS_TOOLS = [
    (threads_search, _SCHEMA_THREADS_SEARCH, "medium"),  # type: RiskLevel
]
