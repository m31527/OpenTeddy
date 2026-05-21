"""
OpenTeddy Browser Tool
─────────────────────────────────────────────────────────────────────────────
Headless Chromium fetcher via Playwright.

Why this exists
---------------
`tools/http_tool.py::http_get` is a pure HTTP GET via httpx — it can't
execute JavaScript. That's fine for static / server-rendered pages but
returns empty bodies for:

  - Single-page apps (React/Vue/Svelte sites where the main DOM is
    hydrated client-side — tixcraft, KKTIX event detail pages, most
    modern e-commerce listings)
  - Cloudflare's JS challenge ("Verifying you are human… one moment")
  - Lazy-loaded / infinite-scroll lists where the visible items are
    fetched as the user scrolls

`browser_fetch` spins up a real headless Chromium, lets the page render,
optionally waits for a specific element to appear, optionally scrolls
to trigger lazy-loaded content, then returns the rendered DOM converted
to clean markdown.

Trade-offs vs `fetch_url`
-------------------------
  - Slow: ~3-5 s per page (browser startup + nav + wait). Use `fetch_url`
    first; only fall back to `browser_fetch` when you actually see an
    empty/SSR-only response.
  - Memory: one headless Chromium ≈ 300-400 MB RAM. We serialise via
    `_BROWSER_LOCK` so parallel goals don't fork three chromiums at
    once on a 16 GB MacBook.
  - First-run delay: Chromium isn't bundled with OpenTeddy. The first
    `browser_fetch` call downloads it (~150 MB, ~30 s on a decent
    connection) into Playwright's cache dir. Subsequent calls reuse.

Design notes
------------
  - Playwright is imported *lazily* inside `browser_fetch` so a missing
    install doesn't break the whole tool registry at startup.
  - `_ensure_chromium()` shells out to `python -m playwright install
    chromium` on first use. We broadcast a `browser_tool.status` WS
    event so the front-end Tools tab can show "Downloading browser
    engine…" instead of the user staring at a frozen UI for 30 s.
  - Risk is "medium": read-only (no clicks/form submits) but heavier
    than `fetch_url` and can be abused to hammer external sites if the
    model loops. Subject to the same approval policy as other medium-
    risk tools.
  - Output is truncated to `_MAX_CONTENT_CHARS` to keep LLM context
    manageable. A 20 KB page is plenty for the kinds of summarisation
    questions the agent actually asks; anything bigger should be paged
    via `wait_for_selector` to extract just the relevant region.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)


# ── Tunables ──────────────────────────────────────────────────────────────────

# One chromium at a time. Without this lock, three parallel subtasks
# each fetching a different URL would each spin up a 400 MB chromium
# → 1.2 GB resident, easy OOM on low-RAM machines.
_BROWSER_LOCK = asyncio.Lock()

# Per-call timeouts. Aggressive on purpose — a slow page should fail
# fast so the orchestrator can try a different approach, not block
# the whole subtask for 60+ s.
_DEFAULT_NAV_TIMEOUT_MS = 15000     # page.goto upper bound
_DEFAULT_WAIT_TIMEOUT_MS = 8000     # wait_for_selector / networkidle

# Output cap. The LLM's context window matters more than completeness —
# 20 KB ≈ 5 K tokens is plenty for "summarise this page" and short
# enough that 8B-class local models don't lose track.
_MAX_CONTENT_CHARS = 20000

# Realistic Chrome UA. Some sites (Cloudflare's basic protection, a few
# news outlets) reject obvious "HeadlessChrome" UAs. Match a current
# Chrome stable on macOS — the most common user-fingerprint.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# ── First-run Chromium install ────────────────────────────────────────────────

# Set once after a successful install probe so we don't re-check on
# every browser_fetch call. Module-global state is intentional: the
# install is genuinely process-wide.
_chromium_ready = False
_chromium_install_lock = asyncio.Lock()


def _playwright_cache_dir() -> Path:
    """Where Playwright stores downloaded browser binaries. Used to
    cheaply probe "is chromium already installed?" without paying the
    cost of importing playwright (~200 ms cold-start)."""
    custom = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    if custom:
        return Path(custom)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ms-playwright"
    if sys.platform.startswith("linux"):
        return Path.home() / ".cache" / "ms-playwright"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", "")) / "ms-playwright"
    return Path.home() / ".cache" / "ms-playwright"


def _chromium_already_installed() -> bool:
    """Filesystem probe — returns True iff a `chromium-*` directory
    exists under Playwright's cache. False positives are nearly
    impossible (no other tool uses this directory layout); false
    negatives just trigger one extra install attempt which is
    idempotent."""
    cache = _playwright_cache_dir()
    if not cache.exists():
        return False
    try:
        return any(
            p.is_dir() and p.name.startswith("chromium-")
            for p in cache.iterdir()
        )
    except (PermissionError, OSError):
        # Unreadable cache dir → assume not installed; install will
        # surface the real permission error.
        return False


async def _broadcast_status(stage: str, message: str = "") -> None:
    """Push a WebSocket event so the front-end can show install /
    fetch progress. Best-effort — running outside the server context
    (smoke tests, CLI use) silently no-ops."""
    try:
        # Late import to avoid a circular main ↔ tools dependency at
        # module load time.
        from main import ws_manager
        await ws_manager.broadcast({
            "event": "browser_tool.status",
            "stage": stage,
            "message": message,
            "_ts": time.time(),
        })
    except Exception:  # noqa: BLE001
        pass


async def _ensure_chromium() -> Optional[str]:
    """Make sure Chromium is downloaded before the first browser_fetch.

    Returns None on success, or an error string the tool can surface.
    Idempotent — concurrent callers all serialise on the same install,
    second caller through the lock sees _chromium_ready=True and
    returns immediately."""
    global _chromium_ready
    if _chromium_ready:
        return None
    async with _chromium_install_lock:
        if _chromium_ready:
            return None
        if _chromium_already_installed():
            _chromium_ready = True
            return None

        await _broadcast_status(
            "downloading",
            message="Downloading browser engine (~150 MB, first run only)…",
        )
        logger.info(
            "Chromium not found in %s — running `playwright install chromium`",
            _playwright_cache_dir(),
        )
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "playwright", "install", "chromium",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
        except FileNotFoundError as exc:
            # `python -m playwright` not available — playwright not
            # installed. Surface a clear message.
            err = f"playwright is not installed in this Python: {exc}"
            await _broadcast_status("error", message=err)
            return err
        except Exception as exc:  # noqa: BLE001
            err = f"playwright install subprocess error: {exc}"
            await _broadcast_status("error", message=err)
            return err

        if proc.returncode != 0:
            tail = (
                stderr.decode(errors="replace")[-500:]
                or stdout.decode(errors="replace")[-500:]
            )
            err = (
                f"`playwright install chromium` failed "
                f"(exit {proc.returncode}): {tail}"
            )
            await _broadcast_status("error", message=err)
            return err

        _chromium_ready = True
        await _broadcast_status("ready", message="Browser engine ready.")
        logger.info("Chromium download complete.")
        return None


# ── Public tool ───────────────────────────────────────────────────────────────

async def browser_fetch(
    url: str,
    wait_for_selector: Optional[str] = None,
    wait_ms: int = _DEFAULT_WAIT_TIMEOUT_MS,
    scroll_to_bottom: bool = False,
    return_format: str = "markdown",
) -> Dict[str, Any]:
    """Render a URL in a headless Chromium and return its content.

    Prefer `fetch_url` for static pages — `browser_fetch` is ~10× slower
    and consumes far more memory. Use this only when `fetch_url` returns
    a near-empty body or you've confirmed the target is a JS-rendered
    SPA. Read-only: does not click, fill forms, or otherwise interact
    with the page.
    """
    start = time.monotonic()

    # ── Validate ──────────────────────────────────────────────────────────────
    url = (url or "").strip()
    if not url:
        return make_result(False, error="url is empty", duration_ms=0)
    if not url.startswith(("http://", "https://")):
        return make_result(
            False,
            error=f"url must start with http:// or https:// (got {url!r})",
            duration_ms=0,
        )
    if return_format not in {"markdown", "text", "html"}:
        return make_result(
            False,
            error=(
                f"invalid return_format {return_format!r} — "
                "must be 'markdown', 'text', or 'html'"
            ),
            duration_ms=0,
        )

    # Clamp wait_ms into a sensible range. Models occasionally pass
    # absurd values ("wait 600000 ms"); cap so a bad call can't hang
    # the orchestrator.
    wait_ms = max(500, min(int(wait_ms or _DEFAULT_WAIT_TIMEOUT_MS), 30000))

    # ── First-run install ────────────────────────────────────────────────────
    install_err = await _ensure_chromium()
    if install_err:
        return make_result(False, error=install_err, duration_ms=_ms(start))

    # ── Lazy-import playwright ────────────────────────────────────────────────
    try:
        from playwright.async_api import async_playwright, Error as PlaywrightError
    except ImportError as exc:
        return make_result(
            False,
            error=(
                f"playwright not installed in this Python "
                f"(`pip install playwright`): {exc}"
            ),
            duration_ms=_ms(start),
        )

    # ── Fetch ─────────────────────────────────────────────────────────────────
    # Serialise — one chromium at a time, see _BROWSER_LOCK comment.
    async with _BROWSER_LOCK:
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                try:
                    context = await browser.new_context(
                        user_agent=_USER_AGENT,
                        viewport={"width": 1280, "height": 800},
                        # Sensible locale defaults — match a real
                        # Taiwan / SE-Asia user since OpenTeddy's
                        # primary audience is bilingual.
                        locale="zh-TW",
                    )
                    page = await context.new_page()
                    page.set_default_navigation_timeout(_DEFAULT_NAV_TIMEOUT_MS)

                    # Navigate. `domcontentloaded` is enough to start
                    # interacting; `wait_for_load_state("networkidle")`
                    # below covers SPA hydration without forcing every
                    # `goto` to wait for every analytics beacon to land.
                    try:
                        await page.goto(url, wait_until="domcontentloaded")
                    except PlaywrightError as exc:
                        return make_result(
                            False,
                            error=f"navigation failed: {exc}",
                            duration_ms=_ms(start),
                        )

                    # Optional: wait for a specific element. Models can
                    # use this to target SPA-rendered content (e.g.
                    # ".event-card" on a listing page). On timeout we
                    # log but proceed — partial content beats no
                    # content for "summarise this page" tasks.
                    if wait_for_selector:
                        try:
                            await page.wait_for_selector(
                                wait_for_selector, timeout=wait_ms,
                            )
                        except PlaywrightError as exc:
                            logger.warning(
                                "wait_for_selector %r timed out on %s: %s",
                                wait_for_selector, url, exc,
                            )
                    else:
                        # Default: wait for the network to go idle for
                        # 500 ms (Playwright's "networkidle" threshold).
                        # Catches most SPA hydration without forcing a
                        # selector contract on the caller.
                        try:
                            await page.wait_for_load_state(
                                "networkidle", timeout=wait_ms,
                            )
                        except PlaywrightError:
                            # Some sites (live dashboards, analytics-
                            # heavy pages) never reach networkidle.
                            # That's fine — proceed with what loaded.
                            pass

                    # Optional: trigger lazy-load by scrolling. Bounded
                    # to ~10 s and ~20000 px of travel to avoid hanging
                    # on infinite-scroll feeds.
                    if scroll_to_bottom:
                        await _autoscroll(page)

                    title = await page.title()
                    final_url = page.url
                    html = await page.content()

                    if return_format == "html":
                        content = html
                    elif return_format == "text":
                        # body innerText — flat dump, no structure but
                        # cheapest path for "is this the right page?"
                        content = await page.inner_text("body")
                    else:  # markdown (default)
                        content = _html_to_markdown(html)

                    if len(content) > _MAX_CONTENT_CHARS:
                        truncated = len(content) - _MAX_CONTENT_CHARS
                        content = (
                            content[:_MAX_CONTENT_CHARS]
                            + f"\n\n…(truncated {truncated} chars — "
                            "narrow the wait_for_selector to target "
                            "a smaller region)…"
                        )

                    return make_result(
                        True,
                        result={
                            "title": title,
                            "url": final_url,
                            "content": content,
                            "format": return_format,
                            "length": len(content),
                        },
                        duration_ms=_ms(start),
                    )
                finally:
                    await browser.close()
        except Exception as exc:  # noqa: BLE001
            logger.exception("browser_fetch failed for %s", url)
            return make_result(
                False,
                error=f"browser_fetch error: {exc}",
                duration_ms=_ms(start),
            )


# ── Helpers ────────────────────────────────────────────────────────────────────

async def _autoscroll(page) -> None:
    """Scroll to the bottom in steps to trigger lazy-loaded content.
    Bounded so we don't get stuck on infinite-scroll feeds (Twitter-
    style timelines, Pinterest, etc.) — we stop at either the document
    bottom or 20 000 px, whichever comes first."""
    await page.evaluate(
        """
        async () => {
            await new Promise((resolve) => {
                let total = 0;
                const step = 400;
                const maxPx = 20000;
                const timer = setInterval(() => {
                    window.scrollBy(0, step);
                    total += step;
                    if (total >= document.body.scrollHeight
                        || total > maxPx) {
                        clearInterval(timer);
                        resolve();
                    }
                }, 200);
            });
        }
        """
    )
    # Beat for any fetch the last scroll triggered to land before we
    # snapshot the DOM.
    await asyncio.sleep(0.5)


def _html_to_markdown(html: str) -> str:
    """Convert rendered HTML to clean markdown.

    Two-stage strip:

      1. BeautifulSoup `.decompose()` — removes script / style / svg /
         iframe **elements AND their inner content**. This matters
         because markdownify's `strip=[...]` param removes only the
         outer tags but keeps the inner text, so a 5 KB `<style>` block
         of CSS still leaks into the markdown as raw text. Pre-
         removing the entire element fixes that.
      2. markdownify the cleaned soup — produces ATX-style headings
         and proper link/list/blockquote handling.

    Falls back to a basic regex strip when BeautifulSoup / markdownify
    aren't importable — the tool still returns *something* useful
    instead of erroring on a missing optional dep.
    """
    try:
        from bs4 import BeautifulSoup
        from markdownify import markdownify as md
    except ImportError:
        # Regex fallback. Inferior — won't handle nested tags cleanly —
        # but never hard-fails.
        import re
        text = re.sub(
            r"<script.*?</script>", "", html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(
            r"<style.*?</style>", "", text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    soup = BeautifulSoup(html, "html.parser")
    # Drop noise elements WITH their inner content — see docstring.
    for tag_name in (
        "script", "style", "noscript", "iframe", "svg",
        "meta", "link", "head",
    ):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # markdownify wants a string — re-serialise the cleaned soup.
    return md(str(soup), heading_style="ATX").strip()


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA_BROWSER_FETCH: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "browser_fetch",
        "description": (
            "Render a URL in a headless Chromium browser and return its content "
            "as markdown (default), plain text, or raw HTML. Use this for "
            "JavaScript-heavy pages where `fetch_url` returns near-empty HTML: "
            "single-page apps (React/Vue/Svelte), sites behind Cloudflare's JS "
            "challenge, ticket sites like tixcraft / KKTIX, news outlets that "
            "lazy-load comments / event lists. ~3-5 s per page including browser "
            "startup. Prefer `fetch_url` for static / server-rendered pages — "
            "it's ~10× faster. Read-only: does not click, fill forms, or "
            "otherwise interact with the page. First call downloads Chromium "
            "(~150 MB one-time)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": (
                        "Full URL to fetch. Must include scheme "
                        "(http:// or https://)."
                    ),
                },
                "wait_for_selector": {
                    "type": "string",
                    "description": (
                        "Optional CSS selector to wait for before extracting "
                        "content. Use when you know the data you want is "
                        "injected by JavaScript into a specific element "
                        "(e.g. '#event-list', '.product-grid .card'). If "
                        "unset, waits for network idle (good default for "
                        "most SPAs)."
                    ),
                },
                "wait_ms": {
                    "type": "integer",
                    "description": (
                        "Max milliseconds to wait for the selector or for "
                        "network idle. Clamped to [500, 30000]. Default 8000."
                    ),
                },
                "scroll_to_bottom": {
                    "type": "boolean",
                    "description": (
                        "If true, scroll the page to the bottom in steps to "
                        "trigger lazy-loaded items (paginated or infinite-"
                        "scroll lists). Adds ~2-5 s. Default false."
                    ),
                },
                "return_format": {
                    "type": "string",
                    "enum": ["markdown", "text", "html"],
                    "description": (
                        "Output format. 'markdown' (default): structured "
                        "text suitable for LLM consumption. 'text': body "
                        "innerText only, flat. 'html': raw rendered DOM "
                        "(verbose — only use when you need attributes "
                        "like href / data-id)."
                    ),
                },
            },
            "required": ["url"],
        },
    },
}


# ── Export ────────────────────────────────────────────────────────────────────

BROWSER_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    # Medium risk: read-only, but heavier than http_get (spins a real
    # browser, ~400 MB RAM, ~3-5 s per call) and able to hammer
    # external sites if the model loops. Same approval policy as other
    # medium-risk tools.
    (browser_fetch, _SCHEMA_BROWSER_FETCH, "medium"),
]
