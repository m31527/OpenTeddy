"""
OpenTeddy Chrome-Attached Tools
─────────────────────────────────────────────────────────────────────────────
Tools that attach to the user's already-running Chrome (via Chrome DevTools
Protocol on port 9222) instead of spinning up our own headless browser.

Why this exists:

The existing `browser_fetch` tool starts a fresh headless Chromium and
fetches anonymously. That's fine for public, anonymously-readable
sites (GitHub Trending, blogs, news). It falls over for:

  - X / Twitter — most content is gated behind login; anonymous view
    is heavily restricted and tripping anti-bot triggers in 2025+.
  - LinkedIn — same story, even harder.
  - Sites where the user is genuinely the authenticated party (their
    bank statements, internal corp dashboards, paid SaaS).

For those, the cleanest approach is to "borrow" the user's already-
authenticated Chrome session. The user starts Chrome once with the
remote-debugging port open, we connect via CDP, drive the existing
session, and read the rendered DOM. No login credentials touch our
process, no separate cookie store, no anti-bot to fight — we look
exactly like the user, because we are.

How the user enables this (one-time setup):

  # macOS
  open -na "Google Chrome" --args --remote-debugging-port=9222
  # or via a script alias they keep in their dotfiles

  # Linux
  google-chrome --remote-debugging-port=9222 &

  # Windows
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" --remote-debugging-port=9222

After that, OpenTeddy can ``connect_over_cdp("http://localhost:9222")``
on demand. Chrome stays running with the user's profile + cookies +
session storage intact.

Tools exposed:

  - ``chrome_attach_check()`` — diagnostic: is Chrome running with the
    remote-debugging port? Returns connection metadata and a helpful
    error message with the exact command to fix it if not.

  - ``x_search(query, top_n, since)`` — search X / Twitter via the
    user's logged-in tab. Extracts tweet text, author, timestamp,
    engagement metrics for the top N matching posts.

  - ``chrome_attached_browse(url, extract_query, top_n)`` — generic:
    navigate to any URL inside the user's Chrome and return the
    rendered visible text. Caller's LLM does the extraction from
    the returned text. Use when you need authenticated access to a
    page that doesn't have a dedicated tool.

Risk: medium — this attaches to the user's real browser, can see
anything they can see, and runs JavaScript in pages they're logged
into. Auto-approve OFF in Telegram by default; user explicitly
confirms before each call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)


# ── CDP endpoint discovery ───────────────────────────────────────────────────


_CDP_PORT = 9222
_CDP_URL = f"http://127.0.0.1:{_CDP_PORT}"


def _cdp_ws_endpoint() -> Optional[str]:
    """Probe Chrome's /json/version endpoint. Returns the WebSocket
    debugger URL when Chrome is up, ``None`` otherwise. Connection
    failures get swallowed — caller surfaces a friendly hint."""
    try:
        with urllib.request.urlopen(f"{_CDP_URL}/json/version", timeout=2) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data.get("webSocketDebuggerUrl")
    except Exception as exc:  # noqa: BLE001
        logger.debug("CDP probe failed: %s", exc)
        return None


_NOT_RUNNING_HINT = (
    "Chrome isn't reachable on the debugging port. Quit any running "
    "Chrome window first, then start it with the debug port open:\n\n"
    "  macOS:\n"
    '    open -na "Google Chrome" --args --remote-debugging-port=9222 '
    "--user-data-dir=$HOME/Library/Application\\ Support/Google/Chrome\n\n"
    "  Linux:\n"
    "    google-chrome --remote-debugging-port=9222 &\n\n"
    "  Windows (PowerShell):\n"
    '    & "$env:ProgramFiles\\Google\\Chrome\\Application\\chrome.exe" '
    "--remote-debugging-port=9222\n\n"
    "After Chrome is up, log in to X / Twitter (or whatever site) once "
    "manually, then run the OpenTeddy task again. The --user-data-dir "
    "flag keeps your normal profile + cookies; without it Chrome will "
    "boot a blank, signed-out session."
)


# ── Tool 1: connectivity check ───────────────────────────────────────────────


async def chrome_attach_check() -> Dict[str, Any]:
    """Verify that Chrome is reachable via the remote-debugging port.

    Returns the CDP version metadata when reachable (browser version,
    user-agent, WebSocket URL) so the planner can confirm the session
    is alive before paying for a full ``x_search`` round-trip.
    """
    start = time.monotonic()
    ws = _cdp_ws_endpoint()
    if ws is None:
        return make_result(
            False,
            error=_NOT_RUNNING_HINT,
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    # Pull the full /json/version payload for diagnostics.
    try:
        with urllib.request.urlopen(f"{_CDP_URL}/json/version", timeout=2) as r:
            meta = json.loads(r.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        meta = {"error": str(exc)}
    return make_result(
        True,
        result={
            "ws_endpoint": ws,
            "browser":     meta.get("Browser", "Chrome"),
            "user_agent":  meta.get("User-Agent", ""),
            "cdp_url":     _CDP_URL,
        },
        duration_ms=int((time.monotonic() - start) * 1000),
    )


# ── Playwright helpers ───────────────────────────────────────────────────────


async def _open_attached_page(reuse_tab_url_contains: Optional[str] = None):
    """Attach to the user's Chrome, return a (page, browser, p)
    tuple. Caller MUST close ``p`` (the playwright context) — not the
    browser — in a finally block, otherwise Playwright leaks a child
    process.

    If ``reuse_tab_url_contains`` is given and any existing tab's URL
    matches that substring, we re-use it instead of opening a new one.
    Critical for X / Twitter: opening a fresh tab each call forces a
    cold session load + triggers their rate-limited "loading…" state.
    """
    ws = _cdp_ws_endpoint()
    if ws is None:
        raise RuntimeError(_NOT_RUNNING_HINT)

    # Lazy import — Playwright is ~200 ms cold-start, no point paying
    # it for the connectivity check.
    from playwright.async_api import async_playwright

    p = await async_playwright().start()
    try:
        browser = await p.chromium.connect_over_cdp(_CDP_URL)
    except Exception:
        await p.stop()
        raise

    # Find or create a page. connect_over_cdp returns a BrowserContext
    # with the existing tabs as pages.
    contexts = browser.contexts
    if not contexts:
        # Shouldn't happen, but defensive.
        ctx = await browser.new_context()
    else:
        ctx = contexts[0]

    page = None
    if reuse_tab_url_contains:
        for existing in ctx.pages:
            try:
                if reuse_tab_url_contains in (existing.url or ""):
                    page = existing
                    break
            except Exception:
                continue
    if page is None:
        page = await ctx.new_page()

    return page, browser, p


# ── Tool 2: X / Twitter search ───────────────────────────────────────────────


_X_SEARCH_EXTRACTOR_JS = r"""
// Extract tweets visible on the page. Run after the search results
// have rendered. Returns a list of plain objects, no DOM nodes.
(() => {
    const out = [];
    const articles = document.querySelectorAll('article[data-testid="tweet"]');
    for (const art of articles) {
        const textEl   = art.querySelector('[data-testid="tweetText"]');
        const text     = textEl ? textEl.innerText.trim() : "";

        // Author handle ("@something") + display name. The User-Name
        // block contains both joined as innerText "Name @handle · 2h".
        const nameEl   = art.querySelector('[data-testid="User-Name"]');
        const fullName = nameEl ? nameEl.innerText.replace(/\n/g, " ").trim() : "";

        // Permalink — the inner timestamp's parent <a> carries href
        // /<handle>/status/<id>.
        const timeEl   = art.querySelector('time');
        const linkEl   = timeEl ? timeEl.closest('a') : null;
        const href     = linkEl ? linkEl.getAttribute('href') : null;
        const url      = href ? new URL(href, location.origin).href : null;
        const ts       = timeEl ? timeEl.getAttribute('datetime') : null;

        // Engagement counts — the aria-label on the action buttons
        // carries the count as e.g. "1,234 replies".
        function getCount(testid) {
            const btn = art.querySelector('[data-testid="' + testid + '"]');
            if (!btn) return null;
            const label = btn.getAttribute('aria-label') || "";
            const m = label.match(/([\d,\.]+)/);
            if (!m) return null;
            const n = parseFloat(m[1].replace(/,/g, ''));
            return Number.isFinite(n) ? n : null;
        }
        const replies   = getCount('reply');
        const reposts   = getCount('retweet');
        const likes     = getCount('like');
        const views     = getCount('analytics');

        out.push({
            full_name_block: fullName,
            text:            text,
            url:             url,
            posted_at:       ts,
            replies:         replies,
            reposts:         reposts,
            likes:           likes,
            views:           views,
        });
    }
    return out;
})()
"""


async def x_search(
    query: str,
    top_n: int = 10,
    since: str = "live",
    require_login: bool = True,
) -> Dict[str, Any]:
    """Search X / Twitter using the user's already-authenticated Chrome
    session. Returns the top N matching posts with text, author,
    timestamp, and engagement metrics.

    Args:
        query: Search query. Free-form text; gets URL-encoded.
        top_n: How many tweets to return. Default 10, max 50. Higher
               values trigger scroll-to-load-more inside the page.
        since: ``"live"`` for latest (default), ``"top"`` for the top
               results algorithm, ``"people"`` / ``"media"`` for
               filters. Maps to X's ``&f=`` URL param.
        require_login: When True (default), we verify the user is
               logged in before searching — saves a wasted round-trip
               when Chrome dropped the session. Set False to allow
               anonymous browsing (results will be heavily limited).

    Returns the standard make_result shape with:
        result = {
            "query":   ...,
            "since":   ...,
            "total":   N,
            "posts":   [{full_name_block, text, url, posted_at,
                         replies, reposts, likes, views}, ...]
        }

    Failure modes:
      - Chrome not running with debug port → make_result(False) with
        an exact-command-to-run hint in `error`.
      - User not logged in → make_result(False, error="Not logged in
        to X — log in once in your Chrome, then retry").
      - Search renders 0 articles → make_result(True) but empty
        ``posts`` array (the caller can decide if this is a problem).
    """
    start = time.monotonic()
    q = (query or "").strip()
    if not q:
        return make_result(False, error="query is empty", duration_ms=0)

    n = max(1, min(int(top_n or 10), 50))
    since_clean = (since or "live").strip().lower()
    if since_clean not in ("live", "top", "people", "media", "lists"):
        since_clean = "live"

    url = (
        "https://x.com/search?q=" + urllib.parse.quote(q)
        + "&f=" + since_clean
    )

    page = browser = p = None
    try:
        page, browser, p = await _open_attached_page(
            reuse_tab_url_contains="x.com" if require_login else None,
        )
    except RuntimeError as exc:
        return make_result(
            False,
            error=str(exc),
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except Exception as exc:  # noqa: BLE001
        return make_result(
            False,
            error=f"Failed to attach to Chrome: {exc}",
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
        # Wait for at least one tweet article OR a clear "login required"
        # marker, whichever comes first. 12 s budget — X's first paint
        # is usually 1-3 s on a warm session.
        try:
            await page.wait_for_selector(
                'article[data-testid="tweet"], [data-testid="loginButton"]',
                timeout=12000,
            )
        except Exception:
            # Selector didn't appear in time — still try extracting in
            # case the page is just slow. We'll catch zero results below.
            pass

        # Login check.
        if require_login:
            login_btn = await page.query_selector('[data-testid="loginButton"]')
            if login_btn is not None:
                return make_result(
                    False,
                    error=(
                        "Not logged in to X in this Chrome session. Open "
                        "https://x.com/login in the same Chrome window, "
                        "log in once, then re-run the OpenTeddy task. "
                        "Your session persists between OpenTeddy calls."
                    ),
                    duration_ms=int((time.monotonic() - start) * 1000),
                )

        # Scroll to load more if the user wants more than the initial
        # render (~10 tweets). Each scroll triggers another ~10 to load.
        SCROLL_ATTEMPTS = max(0, (n - 10 + 9) // 10)  # 11→1, 20→1, 25→2 …
        for _ in range(SCROLL_ATTEMPTS):
            await page.evaluate("window.scrollBy(0, window.innerHeight * 1.5)")
            # Small wait so new articles can render before the next
            # scroll fires. 800 ms balances "load complete" vs "user is
            # waiting forever".
            await asyncio.sleep(0.8)

        # Extract.
        raw = await page.evaluate(_X_SEARCH_EXTRACTOR_JS)
        posts: List[Dict[str, Any]] = list(raw or [])[:n]
        return make_result(
            True,
            result={
                "query":  q,
                "since":  since_clean,
                "url":    url,
                "total":  len(posts),
                "posts":  posts,
            },
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except Exception as exc:  # noqa: BLE001
        return make_result(
            False,
            error=f"X search failed: {exc}",
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    finally:
        # Don't close the page — the user wants to keep their browser
        # state. Just disconnect Playwright cleanly.
        try:
            if browser is not None:
                await browser.close()  # close = disconnect (CDP attach)
        except Exception:  # noqa: BLE001
            pass
        try:
            if p is not None:
                await p.stop()
        except Exception:  # noqa: BLE001
            pass


# ── Tool 3: generic browse (any URL via user's Chrome) ───────────────────────


async def chrome_attached_browse(
    url: str,
    selector: Optional[str] = None,
    timeout_s: int = 15,
) -> Dict[str, Any]:
    """Navigate to ``url`` inside the user's already-running Chrome and
    return the rendered visible text + the first 5 link hrefs.

    Use when you need authenticated browsing for a site that doesn't
    have a dedicated tool — e.g. a paid SaaS dashboard, a corp wiki,
    an internal bug tracker the user is logged into. For X / Twitter,
    prefer ``x_search`` which knows the right selectors.

    Args:
        url: Page to fetch.
        selector: Optional CSS selector to wait for before extracting
            (e.g. ``"article"``). Skipped if None.
        timeout_s: Seconds before giving up on the navigation.

    Returns:
        result = {
            "url":      final URL after redirects,
            "title":    page title,
            "text":     visible innerText, capped at 8000 chars,
            "links":    [{href, text}, ...] capped at 30 entries,
        }
    """
    start = time.monotonic()
    if not url or "://" not in url:
        return make_result(
            False, error="url must be an http(s) URL", duration_ms=0,
        )

    page = browser = p = None
    try:
        page, browser, p = await _open_attached_page(
            reuse_tab_url_contains=None,
        )
    except RuntimeError as exc:
        return make_result(
            False, error=str(exc),
            duration_ms=int((time.monotonic() - start) * 1000),
        )

    try:
        await page.goto(url, wait_until="domcontentloaded",
                        timeout=timeout_s * 1000)
        if selector:
            try:
                await page.wait_for_selector(selector, timeout=timeout_s * 1000)
            except Exception:
                pass

        title = await page.title()
        text = await page.evaluate(
            "() => document.body ? document.body.innerText : ''"
        )
        # Cap to keep tool_log_text manageable downstream.
        text = (text or "")[:8000]

        links = await page.evaluate(
            r"""() => {
                const ls = [];
                const anchors = document.querySelectorAll('a[href]');
                let i = 0;
                for (const a of anchors) {
                    const href = a.getAttribute('href');
                    if (!href) continue;
                    const txt = (a.innerText || a.textContent || '').trim();
                    if (!txt) continue;
                    try {
                        ls.push({ href: new URL(href, location.origin).href,
                                  text: txt.slice(0, 80) });
                    } catch (_) {}
                    i++;
                    if (i >= 30) break;
                }
                return ls;
            }"""
        )
        final_url = page.url
        return make_result(
            True,
            result={
                "url":    final_url,
                "title":  title,
                "text":   text,
                "links":  list(links or []),
            },
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    except Exception as exc:  # noqa: BLE001
        return make_result(
            False, error=f"chrome_attached_browse failed: {exc}",
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


# ── Schemas ──────────────────────────────────────────────────────────────────


_SCHEMA_CHECK: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "chrome_attach_check",
        "description": (
            "Verify that the user's Chrome is running with the remote-"
            "debugging port open (so OpenTeddy can attach via CDP). "
            "Returns the browser version + WebSocket URL on success, or "
            "an exact 'how to start Chrome correctly' hint on failure. "
            "Call this FIRST before x_search / chrome_attached_browse "
            "if you're unsure whether the user has done the one-time "
            "Chrome setup. Read-only, zero side effects."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_SCHEMA_X_SEARCH: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "x_search",
        "description": (
            "Search X / Twitter using the user's already-authenticated "
            "Chrome session and return the top N matching posts with "
            "text, author, timestamp, and engagement (replies / reposts "
            "/ likes / views). "
            "PREFER this over python_exec / browser_fetch when the goal "
            "mentions Twitter / X / 推文 / 推特, because X anti-bot "
            "blocks anonymous scraping. Requires the user to have "
            "started Chrome with --remote-debugging-port=9222 and "
            "logged in to x.com manually once. Call "
            "chrome_attach_check FIRST if you're not sure the setup "
            "is in place."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Free-form search query. Supports X's standard "
                        "operators (since:, until:, lang:, from:, etc.)."
                    ),
                },
                "top_n": {
                    "type": "integer",
                    "description": (
                        "How many tweets to return. Default 10, max 50. "
                        "Values >10 trigger auto-scroll inside the page."
                    ),
                },
                "since": {
                    "type": "string",
                    "description": (
                        "Result type. 'live' (newest, default), 'top' "
                        "(algorithm-ranked), 'people', 'media', 'lists'."
                    ),
                    "enum": ["live", "top", "people", "media", "lists"],
                },
            },
            "required": ["query"],
        },
    },
}

_SCHEMA_BROWSE: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "chrome_attached_browse",
        "description": (
            "Navigate to any URL inside the user's already-running "
            "Chrome (attached via CDP on port 9222) and return the "
            "rendered visible text + first 30 link hrefs. Use this "
            "for authenticated browsing of paid SaaS, corp wikis, "
            "bug trackers, or any site where the user's login state "
            "is required and there isn't a dedicated tool. For X / "
            "Twitter use x_search instead — it knows the right "
            "selectors and engagement metrics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "HTTP(S) URL to fetch.",
                },
                "selector": {
                    "type": "string",
                    "description": (
                        "Optional CSS selector to wait for before "
                        "extracting (e.g. 'article', '.main-content'). "
                        "Use when content is JS-rendered."
                    ),
                },
                "timeout_s": {
                    "type": "integer",
                    "description": (
                        "Per-navigation timeout in seconds. Default 15."
                    ),
                },
            },
            "required": ["url"],
        },
    },
}


# ── Export ───────────────────────────────────────────────────────────────────


CHROME_ATTACHED_TOOLS = [
    (chrome_attach_check,    _SCHEMA_CHECK,    "low"),  # type: RiskLevel
    (x_search,               _SCHEMA_X_SEARCH, "medium"),
    (chrome_attached_browse, _SCHEMA_BROWSE,   "medium"),
]
