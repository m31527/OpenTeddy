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
import datetime
import json
import logging
import os
import random
import time
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)


# ── Anti-bot hardening (v1.1.6) ──────────────────────────────────────────────
# Three layers of "look more like a human, less like a scraper":
#
#   1. Stealth init script — flips the Chromium tells (navigator.webdriver,
#      missing plugins, headless UA fingerprint) that platforms like X /
#      LinkedIn check first when deciding "is this a bot?". Injected into
#      every new context BEFORE the first navigation so the page's
#      bot-detection JS sees a normal browser from request one.
#
#   2. Per-tool rate limit — process-local sliding-window counter that
#      caps calls per hour. Default values are deliberately conservative
#      (20 /hr for x_search) — even an enthusiastic operator stays under
#      X's "this account hits search like a human" threshold. Override
#      with OPENTEDDY_RATE_LIMIT_<TOOL>=N env var when you need more.
#
#   3. Sleep window — refuse calls between 02:00-06:00 local time by
#      default. Real humans don't reliably search at 3am; an account
#      that does is a strong bot signal. Off when OPENTEDDY_NO_SLEEP=1
#      so power users can override during legitimate burst work.
#
# All three can be bypassed per-call by passing _hardening_override=True
# to the underlying tool — used by tests, never by the planner LLM.

_RATE_LIMITS = {
    # tool_name → max calls per rolling hour
    "x_search":               int(os.environ.get("OPENTEDDY_RATE_LIMIT_X_SEARCH",       "20")),
    "chrome_attached_browse": int(os.environ.get("OPENTEDDY_RATE_LIMIT_BROWSE",          "30")),
    # chrome_attach_check is a cheap localhost probe — no limit.
}
_RATE_WINDOW_S = 3600.0
_CALL_TIMESTAMPS: Dict[str, deque] = defaultdict(deque)

_SLEEP_WINDOW_START_HOUR = int(os.environ.get("OPENTEDDY_SLEEP_START", "2"))
_SLEEP_WINDOW_END_HOUR = int(os.environ.get("OPENTEDDY_SLEEP_END", "6"))
_SLEEP_DISABLED = os.environ.get("OPENTEDDY_NO_SLEEP", "0").lower() in ("1", "true", "yes")


def _rate_limit_check(tool_name: str) -> tuple[bool, int, int]:
    """Process-local sliding-window rate limit. Returns
    (allowed, count_in_window, cap). Caller emits a friendly error when
    not allowed."""
    cap = _RATE_LIMITS.get(tool_name)
    if cap is None or cap <= 0:
        return True, 0, 0
    now = time.monotonic()
    q = _CALL_TIMESTAMPS[tool_name]
    # Drop timestamps that fell out of the rolling window.
    while q and q[0] < now - _RATE_WINDOW_S:
        q.popleft()
    if len(q) >= cap:
        return False, len(q), cap
    q.append(now)
    return True, len(q), cap


def _in_sleep_window() -> bool:
    """Return True when local clock is inside the configured sleep window.
    Disabled by OPENTEDDY_NO_SLEEP=1."""
    if _SLEEP_DISABLED:
        return False
    h = datetime.datetime.now().hour
    if _SLEEP_WINDOW_START_HOUR <= _SLEEP_WINDOW_END_HOUR:
        return _SLEEP_WINDOW_START_HOUR <= h < _SLEEP_WINDOW_END_HOUR
    # Wraps midnight (e.g. 22 → 6); rare but support it.
    return h >= _SLEEP_WINDOW_START_HOUR or h < _SLEEP_WINDOW_END_HOUR


# Stealth init script injected into every context. Hides the obvious
# Chromium automation signals. Battle-tested copy from puppeteer-extra-
# plugin-stealth — same techniques, distilled to the high-value pieces
# without bringing in the whole plugin tree.
_STEALTH_INIT_JS = r"""
// 1. Hide the webdriver flag — the #1 bot signal. Real browsers leave
//    navigator.webdriver as undefined; Chromium-with-CDP defaults to true.
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

// 2. Normalise plugins / mimeTypes — headless Chrome has [], real browsers
//    have at least a few.
Object.defineProperty(navigator, 'plugins', {
    get: () => [
        { name: 'Chrome PDF Plugin' }, { name: 'Chrome PDF Viewer' },
        { name: 'Native Client' }
    ],
});
Object.defineProperty(navigator, 'languages', {
    get: () => ['en-US', 'en', 'zh-TW', 'zh'],
});

// 3. Chrome runtime object — present in real Chrome, missing in stripped
//    headless builds. Anti-bot scripts check for it explicitly.
window.chrome = window.chrome || { runtime: {} };

// 4. Permissions query patch — headless Chrome reports
//    Notification permission as 'default'; real Chrome reports 'denied'
//    until the user grants it. Anti-bot scripts use the mismatch.
const _origQuery = window.navigator.permissions ?
    window.navigator.permissions.query : null;
if (_origQuery) {
    window.navigator.permissions.query = (p) =>
        p.name === 'notifications'
            ? Promise.resolve({ state: Notification.permission })
            : _origQuery(p);
}

// 5. WebGL vendor / renderer — headless reports SwiftShader, real Chrome
//    reports the GPU. Spoof to a common Intel one.
const _getParam = WebGLRenderingContext.prototype.getParameter;
WebGLRenderingContext.prototype.getParameter = function(p) {
    if (p === 37445) return 'Intel Inc.';
    if (p === 37446) return 'Intel Iris OpenGL Engine';
    return _getParam.call(this, p);
};
"""


async def _human_pause(min_s: float = 1.5, max_s: float = 4.0) -> None:
    """Short randomised delay to break the "back-to-back identical
    timing" pattern that synthetic clients produce. Cheap; default
    1.5-4s is invisible at the user's scale but breaks behavioural
    fingerprints used by X / LinkedIn rate analyses."""
    await asyncio.sleep(min_s + random.random() * (max_s - min_s))


# ── storage_state.json loading ───────────────────────────────────────────────
# Look-up order for a Playwright-format storage_state.json. Found cookies
# get injected into the attached context on every CDP attach. Lifecycle:
#   - Operator captures the state once on a workstation with a display
#     (see scripts/capture-edge-state.md).
#   - scp'd to each fleet node's /var/lib/openteddy/storage_state.json.
#   - Edge / Chromium boots fresh (via systemd) and OpenTeddy injects
#     these cookies before every x_search / chrome_attached_browse call.
#
# This means every fleet node scrapes X / LinkedIn / wherever as the
# operator who captured the state, with no per-node login dance.
_STATE_SEARCH_PATHS = [
    # Highest priority — explicit override.
    lambda: Path(os.environ["OPENTEDDY_CDP_STATE"])
            if os.environ.get("OPENTEDDY_CDP_STATE") else None,
    # Standard fleet location written by scripts/setup-edge-cdp.sh.
    lambda: Path("/var/lib/openteddy/storage_state.json"),
    # User-scoped fallback for single-user dev installs.
    lambda: Path.home() / ".config" / "openteddy" / "storage_state.json",
]


def _find_state_file() -> Optional[Path]:
    """Resolve the first existing storage_state.json from the search
    path. Returns None when nothing is found — the tool still works,
    cookies just aren't injected (so any login-gated site stays
    logged out)."""
    for resolver in _STATE_SEARCH_PATHS:
        try:
            p = resolver()
        except Exception:  # noqa: BLE001
            continue
        if p and p.exists():
            return p
    return None


async def _inject_state_if_present(context) -> int:
    """If a storage_state.json exists, push its cookies into the
    attached Playwright context. Returns the number of cookies
    injected (0 on any failure — failures are logged but never
    raised; the caller should still try to scrape, the worst case
    is a logged-out page).

    Why we re-inject on every attach instead of once per process:
    the systemd-managed Edge instance can outlive OpenTeddy
    restarts; cookies set in a previous attach do persist in the
    profile dir, but cheap re-injection is the safest way to
    guarantee freshness across restarts of either side.
    """
    state_path = _find_state_file()
    if state_path is None:
        return 0
    try:
        with open(state_path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "storage_state at %s failed to parse: %s", state_path, exc,
        )
        return 0
    cookies = state.get("cookies") or []
    if not cookies:
        return 0
    try:
        await context.add_cookies(cookies)
        logger.info(
            "Injected %d cookies from %s into CDP context",
            len(cookies), state_path,
        )
        return len(cookies)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Cookie injection failed: %s (cookies=%d, path=%s)",
            exc, len(cookies), state_path,
        )
        return 0


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
    "Chrome / Chromium isn't reachable on the debugging port. Quit any "
    "running browser window first, then start it with the debug port "
    "open. Any Chromium-based browser works — the CDP wire protocol is "
    "the same.\n\n"
    "  macOS (Apple Silicon or Intel):\n"
    '    open -na "Google Chrome" --args --remote-debugging-port=9222 '
    "--user-data-dir=$HOME/Library/Application\\ Support/Google/Chrome\n\n"
    "  Linux x86_64 (Chrome):\n"
    "    google-chrome --remote-debugging-port=9222 \\\n"
    "                  --user-data-dir=$HOME/.config/google-chrome &\n\n"
    "  Linux ARM64 (DGX Spark / Raspberry Pi / NVIDIA Jetson / etc.) —\n"
    "  Google Chrome has no official ARM64 build, use Chromium instead:\n"
    "    sudo apt install chromium-browser   # Ubuntu/Debian\n"
    "    chromium-browser --remote-debugging-port=9222 \\\n"
    "                     --user-data-dir=$HOME/.config/chromium &\n"
    "    # OR Microsoft Edge (officially supports Linux ARM64):\n"
    "    microsoft-edge --remote-debugging-port=9222 \\\n"
    "                   --user-data-dir=$HOME/.config/microsoft-edge &\n\n"
    "  Windows (PowerShell):\n"
    '    & "$env:ProgramFiles\\Google\\Chrome\\Application\\chrome.exe" '
    "--remote-debugging-port=9222\n\n"
    "After the browser is up, log in to X / Twitter (or whatever site) "
    "once manually, then run the OpenTeddy task again. The "
    "--user-data-dir flag keeps your normal profile + cookies; without "
    "it the browser will boot a blank, signed-out session."
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
    # Report storage_state status so the operator sees "yep, cookies will
    # be injected" without having to read the source. Critical for fleet
    # ops: if state_path is None on a node that's supposed to scrape X,
    # somebody forgot to scp the file.
    state_path = _find_state_file()
    state_info: Dict[str, Any] = {"path": None, "cookie_count": 0}
    if state_path is not None:
        state_info["path"] = str(state_path)
        try:
            with open(state_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            state_info["cookie_count"] = len(data.get("cookies") or [])
        except Exception as exc:  # noqa: BLE001
            state_info["parse_error"] = str(exc)
    # Hardening status — surfaces "you have N/cap calls left this hour"
    # and whether we're inside the sleep window. Helps operators decide
    # "should I run that scheduled task now or wait" without grepping
    # logs.
    now = time.monotonic()
    hardening: Dict[str, Any] = {
        "stealth_init_script": True,
        "sleep_window": {
            "start_hour":  _SLEEP_WINDOW_START_HOUR,
            "end_hour":    _SLEEP_WINDOW_END_HOUR,
            "currently_in": _in_sleep_window(),
            "disabled":    _SLEEP_DISABLED,
        },
        "rate_limits": {},
    }
    for tool, cap in _RATE_LIMITS.items():
        q = _CALL_TIMESTAMPS.get(tool, deque())
        recent = sum(1 for ts in q if ts > now - _RATE_WINDOW_S)
        hardening["rate_limits"][tool] = {
            "used_in_last_hour": recent,
            "cap_per_hour":      cap,
            "remaining":         max(0, cap - recent),
        }
    return make_result(
        True,
        result={
            "ws_endpoint":   ws,
            "browser":       meta.get("Browser", "Chrome"),
            "user_agent":    meta.get("User-Agent", ""),
            "cdp_url":       _CDP_URL,
            "storage_state": state_info,
            "hardening":     hardening,
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

    # Inject cookies from storage_state.json (if present) BEFORE we open
    # / re-use a tab. Doing it after page.goto() works too but the first
    # navigation can fire login-detection JS before our cookies land,
    # producing a flicker of "Sign in" UI that messes up screenshots /
    # accessibility checks.
    await _inject_state_if_present(ctx)

    # Inject the stealth init script BEFORE any page navigation in this
    # context, so anti-bot JS on the page sees a normal-looking browser
    # from request one. add_init_script applies to every existing page
    # AND every future new_page() call — safe to call multiple times
    # across attach cycles; idempotent at the JS level.
    try:
        await ctx.add_init_script(_STEALTH_INIT_JS)
    except Exception as exc:  # noqa: BLE001
        # Older Playwright versions sometimes throw here on already-
        # initialised contexts; log + continue, the stealth flags are
        # nice-to-have, not load-bearing.
        logger.debug("stealth init script attach failed: %s", exc)

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

    # Anti-bot hardening pre-flight (v1.1.6). Both checks return a
    # cooperative error rather than raising — the planner reads
    # `success: False` and stops, the user sees a clear "wait an
    # hour" / "wait until 6am" message.
    if _in_sleep_window():
        return make_result(
            False,
            error=(
                f"Inside the sleep window ({_SLEEP_WINDOW_START_HOUR:02d}:00-"
                f"{_SLEEP_WINDOW_END_HOUR:02d}:00 local). Automated "
                "searches during typical sleep hours are a strong bot "
                "signal for X. Re-run after the window, or set "
                "OPENTEDDY_NO_SLEEP=1 in your environment if this is "
                "intentional burst work."
            ),
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    allowed, count, cap = _rate_limit_check("x_search")
    if not allowed:
        return make_result(
            False,
            error=(
                f"Rate-limited: {count}/{cap} x_search calls in the last "
                f"hour. Hitting X search faster than human pace is the #1 "
                f"way to get the account flagged. Wait ~{int(60 - (count - cap + 1) * 3)} "
                f"minutes, or raise the cap with "
                f"OPENTEDDY_RATE_LIMIT_X_SEARCH=<n> (and accept the "
                f"increased lockout risk)."
            ),
            duration_ms=int((time.monotonic() - start) * 1000),
        )

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

        # Login check — multi-signal because X changes its UI markup
        # often. Confirmed logged-out signals (any one triggers):
        #   1. URL redirected to /i/flow/login
        #   2. data-testid="loginButton" present (old UI)
        #   3. data-testid="login" link present (current UI)
        #   4. Page contains the X "sign in" CTA card
        if require_login:
            current_url = page.url or ""
            logged_out_signals = []
            if "/i/flow/login" in current_url or "/login" in current_url:
                logged_out_signals.append(f"redirected to {current_url}")
            for sel in (
                '[data-testid="loginButton"]',
                '[data-testid="login"]',
                'a[href="/login"]',
                'a[href="/i/flow/login"]',
            ):
                if await page.query_selector(sel):
                    logged_out_signals.append(f"found {sel}")
                    break
            # Last resort: zero tweet articles AND we see one of X's
            # "Don't miss what's happening" / "Sign in to X" upsell
            # texts. That's the most common state right now.
            if not logged_out_signals:
                article_count = await page.evaluate(
                    'document.querySelectorAll(\'article[data-testid="tweet"]\').length'
                )
                if article_count == 0:
                    body_text = await page.evaluate(
                        '() => document.body ? document.body.innerText.slice(0, 4000) : ""'
                    )
                    for phrase in (
                        "Sign in to X",
                        "Don't miss what's happening",
                        "登入 X",
                        "登錄 X",
                        "登录 X",
                        "Create account",
                    ):
                        if phrase in (body_text or ""):
                            logged_out_signals.append(f"page text contains '{phrase}'")
                            break

            if logged_out_signals:
                return make_result(
                    False,
                    error=(
                        "Not logged in to X. Signals detected: "
                        + "; ".join(logged_out_signals)
                        + ". Fix: capture a storage_state.json on a "
                        "workstation with a display (see scripts/"
                        "capture-edge-state.md), scp it to "
                        "/var/lib/openteddy/storage_state.json on this "
                        "host, restart openteddy-cdp.service, then "
                        "retry. The headless browser cannot accept an "
                        "interactive login."
                    ),
                    duration_ms=int((time.monotonic() - start) * 1000),
                )

        # Scroll to load more if the user wants more than the initial
        # render (~10 tweets). Each scroll triggers another ~10 to load.
        #
        # Anti-bot: use page.mouse.wheel() instead of window.scrollBy.
        # The synthetic scrollBy() fires no MouseEvent / WheelEvent —
        # only a scroll event — so X's bot detector can tell the
        # difference. mouse.wheel() generates real wheel events with a
        # plausible velocity profile.
        SCROLL_ATTEMPTS = max(0, (n - 10 + 9) // 10)  # 11→1, 20→1, 25→2 …
        viewport = page.viewport_size or {"height": 900}
        for _ in range(SCROLL_ATTEMPTS):
            # Scroll roughly one-and-a-half viewport heights, with a
            # ±15% jitter so consecutive scrolls don't look identical.
            delta_y = int(viewport["height"] * (1.4 + random.random() * 0.3))
            try:
                await page.mouse.wheel(0, delta_y)
            except Exception:
                # Older Playwright / weird embed cases — fall back to
                # synthetic, accepting the lower stealth quality.
                await page.evaluate(f"window.scrollBy(0, {delta_y})")
            # Randomised wait so consecutive scrolls don't fire on a
            # 0.8s metronome. 0.6-1.6s mimics "reading a couple
            # tweets" pacing.
            await _human_pause(0.6, 1.6)

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

    # Same hardening as x_search — sleep window + rate limit. Slightly
    # higher rate cap (30/hr default) since chrome_attached_browse is
    # used across a wider variety of sites, each less aggressive than X.
    if _in_sleep_window():
        return make_result(
            False,
            error=(
                f"Inside the sleep window ({_SLEEP_WINDOW_START_HOUR:02d}:00-"
                f"{_SLEEP_WINDOW_END_HOUR:02d}:00 local). Set "
                "OPENTEDDY_NO_SLEEP=1 to override."
            ),
            duration_ms=int((time.monotonic() - start) * 1000),
        )
    allowed, count, cap = _rate_limit_check("chrome_attached_browse")
    if not allowed:
        return make_result(
            False,
            error=(
                f"Rate-limited: {count}/{cap} browse calls in the last "
                f"hour. Override with OPENTEDDY_RATE_LIMIT_BROWSE=<n>."
            ),
            duration_ms=int((time.monotonic() - start) * 1000),
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
        # Brief human-like pause before extracting — gives JS time to
        # finish settling and breaks "instant-extract" timing pattern.
        await _human_pause(0.8, 2.2)
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
