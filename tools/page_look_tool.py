"""
page_look — open a page OpenTeddy made, do what the person would do, and look.

Why (2026-11-08): a Christmas game OpenTeddy wrote showed only falling
snow — a full-screen snow canvas painted an opaque background over the
game, and the emoji were drawn at 3% opacity. Asked twice to fix it, the
agent simulated the game logic in Python (which was fine), changed
nothing, and reported it fixed. It had no way to see the page.

This opens the page in a headless Chromium on this machine (no need for
the person's own Chrome, works from Telegram), does the actions given
(click 開始遊戲, press ArrowRight, …) and reports:
  * covered — elements something else is painted over, and what
  * errors  — JS errors, failed loads
  * seen    — the vision model's account of the screenshot
              (config.vision_model, the one the browser extension uses)
  * the screenshot's path

Only pages OpenTeddy made or serves nearby: a workspace file, or a
localhost / LAN address. Websites go through browser_fetch / browser_*.
"""

from __future__ import annotations

import base64
import ipaddress
import logging
import os
import re
import time
import urllib.parse
from typing import Any, Dict, List, Optional, Tuple

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)

_VIEWPORT = {"width": 1280, "height": 800}
_CGNAT = ipaddress.ip_network("100.64.0.0/10")   # Tailscale / NetBird style addresses


def _nearby(host: str) -> bool:
    host = (host or "").strip("[]").lower()
    if host in ("localhost", "0.0.0.0") or host.endswith((".local", ".localhost", ".lan")):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip in _CGNAT


def target_url(target: str) -> Tuple[Optional[str], Optional[str]]:
    """(url, None) or (None, why not)."""
    t = (target or "").strip()
    if not t:
        return None, "target is empty — give a workspace html file or a localhost URL"
    if t.startswith(("http://", "https://")):
        if not _nearby(urllib.parse.urlparse(t).hostname or ""):
            return None, ("page_look opens pages OpenTeddy made or serves nearby: a workspace "
                          "file, or a localhost / LAN address. For websites use browser_fetch "
                          "or the browser_* tools.")
        return t, None
    path = urllib.parse.unquote(t[7:]) if t.startswith("file://") else t
    from tools.file_tool import _resolve_path
    p = _resolve_path(path)
    if not p.is_file():
        return None, f"File not found: {p}"
    return p.as_uri(), None


# "#start-btn", ".play", "[data-x]", "button.primary", "canvas" — else it's text on the page.
_CSS = re.compile(r"^([#.\[]|(button|canvas|a|input|select|textarea|img|svg|div|span|body)([#.\[:]|$))", re.I)


async def _act(page: Any, action: str) -> None:
    verb, _, arg = action.strip().partition(" ")
    verb, arg = verb.lower(), arg.strip()
    if verb == "click":
        if not arg:
            raise ValueError("click what? e.g. 'click 開始遊戲' or 'click #start-btn'")
        cands = ([page.locator(arg)] if _CSS.match(arg)
                 else [page.get_by_role("button", name=arg), page.get_by_text(arg)])
        for loc in cands:
            if await loc.count():
                await loc.first.click(timeout=3000)
                return
        raise ValueError(f"nothing on the page matches {arg!r}")
    if verb == "press":
        await page.keyboard.press(arg or "Enter")
    elif verb == "type":
        await page.keyboard.type(arg)
    elif verb == "wait":
        await page.wait_for_timeout(max(0, min(int(re.sub(r"\D", "", arg) or 500), 10000)))
    else:
        raise ValueError(f"unknown action {action!r} — use click / press / type / wait")


# Which visible elements have something else painted over their centre.
# Hit-testing follows paint order, but skips pointer-events:none layers —
# exactly the decorative overlays (snow, vignettes) that cover content —
# so they're switched back on for the test. A cover only counts if it
# actually paints there: visible, not transparent, and (for a canvas)
# not a transparent pixel at that point.
_COVER_JS = r"""
() => {
  const name = el => el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') +
    (typeof el.className === 'string' && el.className.trim()
      ? '.' + el.className.trim().split(/\s+/).slice(0, 2).join('.') : '');
  const alpha = c => { const m = c.match(/rgba?\(([^)]+)\)/); if (!m) return 0;
    const p = m[1].split(',').map(s => parseFloat(s)); return p.length > 3 ? p[3] : 1; };
  const shown = el => { let o = 1; for (let e = el; e && e.nodeType === 1; e = e.parentElement) {
      const cs = getComputedStyle(e); if (cs.visibility === 'hidden' || cs.display === 'none') return 0;
      o *= parseFloat(cs.opacity); } return o; };
  const paints = (el, x, y) => {
    if (shown(el) < 0.1) return false;
    const cs = getComputedStyle(el);
    if (alpha(cs.backgroundColor) > 0.3 || cs.backgroundImage !== 'none') return true;
    const tag = el.tagName.toLowerCase();
    if (tag === 'canvas') {
      try { const r = el.getBoundingClientRect();
        const px = el.getContext('2d').getImageData(Math.floor((x - r.left) * el.width / r.width),
                                                    Math.floor((y - r.top) * el.height / r.height), 1, 1).data;
        return px[3] > 76; } catch (e) { return true; }
    }
    return ['img', 'video', 'iframe'].includes(tag);
  };
  // Decorations (pointer-events:none by the author — snow, glows) are meant
  // to sit behind or over things; only content is checked for being covered.
  const want = [...document.querySelectorAll('canvas, svg, img, video, iframe, button, input, h1, h2, h3, main, [id]')]
    .filter(el => { const r = el.getBoundingClientRect(); return r.width >= 16 && r.height >= 16 &&
                    shown(el) >= 0.1 && getComputedStyle(el).pointerEvents !== 'none'; })
    .slice(0, 60);
  const style = document.createElement('style');
  style.textContent = '*{pointer-events:auto !important}';
  document.head.appendChild(style);
  const found = [];
  for (const el of want) {
    const r = el.getBoundingClientRect();
    const x = r.left + r.width / 2, y = r.top + r.height / 2;
    if (x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) continue;
    for (const top of document.elementsFromPoint(x, y)) {
      if (top === el || el.contains(top) || top.contains(el)) break;
      if (paints(top, x, y)) { found.push([el, top]); break; }
    }
  }
  style.remove();
  return found.map(([el, top]) => { const cs = getComputedStyle(top);
    return `${name(el)} is covered by ${name(top)} (position: ${cs.position}; z-index: ${cs.zIndex}; ` +
           `background-color: ${cs.backgroundColor}; pointer-events: ${cs.pointerEvents})`; });
}
"""


def _vision_prompt(actions: List[str], question: str) -> str:
    after = f", taken after: {'; '.join(actions)}" if actions else ""
    return ("This is a screenshot of a web page" + after + ". Describe concretely what is visible: "
            "the main content, text, images, game pieces or characters, and anything that looks "
            "empty, covered, cut off, too faint or broken. Then answer: "
            + (question or "Does the page look complete and working?")
            + " Describe only what you can see; don't guess what should be there.")


async def page_look(target: str, actions: Optional[List[str]] = None, question: str = "") -> Dict[str, Any]:
    """Open a workspace page or a nearby URL, do `actions`, and look."""
    start = time.monotonic()
    ms = lambda: int((time.monotonic() - start) * 1000)  # noqa: E731
    url, why = target_url(target)
    if not url:
        return make_result(False, error=why, duration_ms=ms())
    actions = [str(a) for a in (actions or []) if str(a).strip()][:20]

    from tools.browser_tool import _BROWSER_LOCK, _ensure_chromium
    err = await _ensure_chromium()
    if err:
        return make_result(False, error=err, duration_ms=ms())
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        return make_result(False, error=f"playwright not installed: {exc}", duration_ms=ms())

    errors: List[str] = []
    done: List[str] = []
    async with _BROWSER_LOCK:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            try:
                page = await browser.new_page(viewport=_VIEWPORT, locale="zh-TW")
                page.on("pageerror", lambda e: errors.append(f"JS error: {e}"))
                page.on("console", lambda m: errors.append(f"console.error: {m.text}")
                        if m.type == "error" else None)
                page.on("requestfailed", lambda r: errors.append(f"failed to load {r.url}"))
                page.on("response", lambda r: errors.append(f"HTTP {r.status} for {r.url}")
                        if r.status >= 400 else None)
                try:
                    await page.goto(url, wait_until="load", timeout=15000)
                except Exception as exc:  # noqa: BLE001
                    return make_result(False, error=f"could not open {url}: {exc}", duration_ms=ms())
                await page.wait_for_timeout(500)
                for a in actions:
                    try:
                        await _act(page, a)
                        done.append(a)
                    except Exception as exc:  # noqa: BLE001 — say what failed, keep looking
                        errors.append(f"action {a!r} failed: {str(exc).splitlines()[0][:300]}")
                await page.wait_for_timeout(400)
                covered = await page.evaluate(_COVER_JS)
                title = await page.title()
                text = (await page.evaluate("() => document.body ? document.body.innerText : ''"))[:600]
                png = await page.screenshot()
            finally:
                await browser.close()

    from config import effective_workspace_dir
    shot_dir = os.path.join(effective_workspace_dir(), ".page_look")
    os.makedirs(shot_dir, exist_ok=True)
    shot = os.path.join(shot_dir, f"look-{time.strftime('%H%M%S')}.png")
    with open(shot, "wb") as fh:
        fh.write(png)

    seen = ""
    try:
        from tools.browser_ext_tool import _vision
        seen = await _vision(_vision_prompt(done, question), base64.b64encode(png).decode(), num_predict=350)
    except Exception as exc:  # noqa: BLE001 — the other findings still stand
        seen = f"(no vision model: {exc})"

    return make_result(True, {
        "url": url, "title": title, "actions_done": done,
        "covered": covered or "nothing is covered",
        "errors": errors[:20] or "none",
        "seen": seen, "text": text, "screenshot": shot,
    }, duration_ms=ms())


_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "page_look",
        "description": (
            "Open a web page, HTML report or browser game you made — a workspace file or a "
            "localhost / LAN URL — in a headless browser, do what the person would (click, press "
            "keys, wait), and look at it: which elements are covered by something else, JS errors "
            "and failed loads, a vision model's description of the screen, and a screenshot. Use it "
            "to check a page works before saying it is done, and to see a reported display problem "
            "(blank, missing, covered, faint, not moving) before fixing it — and again after."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string",
                           "description": "Workspace file (e.g. game.html) or URL like http://localhost:8080/."},
                "actions": {"type": "array", "items": {"type": "string"},
                            "description": "In order: 'click <button text or CSS selector>', "
                                           "'press <Key>' (ArrowRight, Space, Enter…), 'type <text>', "
                                           "'wait <ms>'. E.g. [\"click 開始遊戲\", \"wait 1000\"]."},
                "question": {"type": "string",
                             "description": "What to check on screen, e.g. 看得到主角嗎？"},
            },
            "required": ["target"],
        },
    },
}

PAGE_LOOK_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    (page_look, _SCHEMA, "low"),
]
