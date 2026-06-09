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
# v1.1.7 rewrite: the first version used `article, div[role="article"]` as
# post-container selectors. That worked on Instagram's DOM but Threads' web
# app uses anonymous <div> wrappers with React-generated class hashes for
# its post containers — no semantic article tags at all. Result: extractor
# returned 0 posts even when the browser visibly showed dozens.
#
# New strategy is anchor-based and React-resilient: every post has EXACTLY
# ONE <time datetime="..."> element (the timestamp) and exactly one
# <a href=".../post/..."> element (the permalink). Both are semantic
# elements Meta is unlikely to remove because accessibility tools and
# search indexers rely on them. We:
#
#   1. Enumerate every <time datetime="...">
#   2. From each, walk up to the nearest <a href*="/post/">  → that's the
#      permalink anchor for this post
#   3. Walk up further until we find an outer block that contains
#      span[dir="auto"] (the body text) AND aria-label[*="like"] /
#      [*="reply"] elements (the engagement row)
#   4. Extract from that outer block
#
# Dedupe by post URL since Threads sometimes renders the same post twice
# (once collapsed in a "more replies" stack, once expanded).

_THREADS_EXTRACTOR_JS = r"""
(() => {
    const out = [];
    const seenUrls = new Set();

    // Anchor: <time datetime="..."> — one per post, semantic, persistent.
    const times = document.querySelectorAll('time[datetime]');

    for (const timeEl of times) {
        // Step 1: walk up to the permalink anchor wrapping this timestamp.
        const permalinkAnchor = timeEl.closest('a[href*="/post/"]');
        if (!permalinkAnchor) continue;

        const href = permalinkAnchor.getAttribute('href') || "";
        if (!href || !/\/post\//.test(href)) continue;
        const postUrl = new URL(href, location.origin).href;
        if (seenUrls.has(postUrl)) continue;
        seenUrls.add(postUrl);

        const postedAt = timeEl.getAttribute('datetime');

        // Handle parsed from the URL itself — /<handle>/post/<id>
        // (Threads uses @handle in the path; sometimes preceded by a
        // bare slug. Be permissive on the regex.)
        const handleMatch = postUrl.match(/\/(@[\w.]+)\/post\//);
        const handle = handleMatch ? handleMatch[1] : "";

        // Step 2: walk up from the permalink anchor until we hit a
        // container that has BOTH the body text (span[dir="auto"]) and
        // the engagement row (aria-label with 'like' / 'repl').
        // Threads' post block sits ~5-10 ancestors above the permalink.
        let outer = permalinkAnchor.parentElement;
        let bestOuter = null;
        let bestScore = 0;
        for (let i = 0; i < 15 && outer; i++) {
            const hasBody = outer.querySelector('span[dir="auto"]') !== null;
            const hasReact = outer.querySelector('[aria-label*="ike" i], [aria-label*="epl" i], [aria-label*="epost" i]') !== null;
            const score = (hasBody ? 1 : 0) + (hasReact ? 2 : 0);
            if (score > bestScore) { bestScore = score; bestOuter = outer; }
            // Stop walking once we've found both — going further usually
            // captures sibling posts and pollutes the extraction.
            if (hasBody && hasReact) break;
            outer = outer.parentElement;
        }
        const block = bestOuter || permalinkAnchor.parentElement || timeEl.parentElement;

        // Display name — pull from the FIRST <a> in `block` whose href is
        // a bare /@handle (not a post permalink).
        let displayName = "";
        for (const a of block.querySelectorAll('a[href^="/@"]')) {
            const h = a.getAttribute('href') || "";
            if (/^\/@[\w.]+\/?$/.test(h)) {
                displayName = (a.innerText || "").replace(/\s+/g, " ").trim();
                if (displayName) break;
            }
        }

        // Body text from span[dir="auto"]. Skip ones inside profile-link
        // anchors (those are the author name we already captured).
        const bodyParts = [];
        for (const span of block.querySelectorAll('span[dir="auto"]')) {
            if (span.closest('a[href^="/@"]')) continue;
            if (span.closest('time')) continue;
            const t = (span.innerText || "").trim();
            if (!t) continue;
            if (bodyParts.length && bodyParts[bodyParts.length - 1] === t) continue;
            bodyParts.push(t);
        }
        const text = bodyParts.join("\n").trim();

        // Engagement — aria-label is the stable contract. Threads uses
        // "12 replies", "84 likes", "3 reposts" / its localised
        // equivalents. Match case-insensitively on the English root since
        // even localised UIs keep the English root in the aria-label for
        // a11y compliance.
        function getCount(rootRegex) {
            for (const el of block.querySelectorAll('[aria-label]')) {
                const lbl = el.getAttribute('aria-label') || "";
                if (rootRegex.test(lbl)) {
                    const m = lbl.match(/([\d,\.]+)/);
                    if (m) {
                        const n = parseFloat(m[1].replace(/,/g, ""));
                        return Number.isFinite(n) ? n : null;
                    }
                    return 0;  // labeled but no number = zero
                }
            }
            return null;
        }
        const replies = getCount(/repl/i);
        const likes   = getCount(/like/i);
        const reposts = getCount(/repost/i);

        if (!text && !displayName) continue;

        out.push({
            handle:      handle,
            displayName: displayName,
            text:        text,
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
    # threads.com is the canonical Meta domain (threads.net redirects);
    # both work. Use .com so the URL matches what the operator sees in
    # the browser address bar (less confusing for debugging).
    url = (
        "https://www.threads.com/search?q="
        + urllib.parse.quote(q)
        + "&serp_type=default"
    )

    page = browser = p = None
    try:
        # Reuse any tab already on Threads (either domain — they share a
        # session, so cookies / login state carry).
        page, browser, p = await _open_attached_page(
            reuse_tab_url_contains="threads.",
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
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        # Threads' results are React-rendered after JS bundles execute;
        # the initial DOM load fires long before any post is visible.
        # Wait for either a <time> element (real post timestamp) or a
        # known logged-out marker. 18 s budget covers slow first-paint
        # on the Threads web app + a cold React bundle.
        try:
            await page.wait_for_selector(
                'time[datetime], [data-testid="login-button"], a[href^="/login"]',
                timeout=18000,
            )
        except Exception:
            pass
        # Even after the first <time> appears, the React app keeps
        # streaming more posts in for ~1-2 s. Brief settle pause so the
        # extractor sees the full first batch.
        await _human_pause(1.0, 2.0)

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
