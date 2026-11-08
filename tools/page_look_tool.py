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
  * answer         — the vision model's YES / NO to the question, first
  * covered        — elements something else is painted over, and what
  * canvas_drawing — text / images drawn too faintly or off the canvas
  * errors         — JS errors, failed loads
  * seen           — the vision model's account of the screenshot
                     (config.vision_model, the one the browser extension uses)
  * the screenshot's path

The first eval run (2026-11-08) is why the answer comes first and the
canvas is watched: the agent fixed the cover, the vision model said
"only two red dots, not the player", and the agent wrote that the
animation "may not have started yet" and reported success.

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


# Text and images drawn on a canvas so faintly or so far off it that no
# one sees them. Recorded as they're drawn (an init script wraps the 2D
# context): the screenshot shows only absence, this says what went
# missing and why. Paths (fill/stroke) are left out — faint grid lines
# and glows are usually meant.
_CANVAS_SPY_JS = r"""
(() => {
  const P = CanvasRenderingContext2D.prototype, seen = new Map();
  window.__pageLookCanvas = seen;
  const alphaOf = s => { if (typeof s !== 'string') return 1;
    const m = s.match(/rgba?\(([^)]+)\)/); if (m) { const p = m[1].split(',');
      return p.length > 3 ? parseFloat(p[3]) : 1; }
    const h = s.match(/^#([0-9a-f]{8})$/i); return h ? parseInt(h[1].slice(6), 16) / 255 : 1; };
  const name = c => 'canvas' + (c.id ? '#' + c.id : '');
  const note = (ctx, what, x, y, w, h, useFill) => {
    try {
      const c = ctx.canvas, a = ctx.globalAlpha * (useFill ? alphaOf(ctx.fillStyle) : 1);
      let why = '';
      if (a < 0.2) why = `drawn at opacity ${+a.toFixed(3)} — nearly invisible` +
                        (useFill ? ` (fillStyle ${ctx.fillStyle}, globalAlpha ${ctx.globalAlpha})` : '');
      else if (x + w < 0 || y + h < 0 || x > c.width || y > c.height)
        why = `drawn outside the ${c.width}×${c.height} canvas, at (${Math.round(x)}, ${Math.round(y)})`;
      if (!why) return;
      const k = `${name(c)}: ${what} ${why}`;
      seen.set(k, (seen.get(k) || 0) + 1);
    } catch (e) {}
  };
  const wrap = (fn, describe) => { const orig = P[fn]; P[fn] = function (...a) {
    describe(this, a); return orig.apply(this, a); }; };
  wrap('fillText', (ctx, a) => note(ctx, `fillText(${JSON.stringify(String(a[0]).slice(0, 12))})`,
                                  a[1], a[2] - 20, 20, 20, true));
  wrap('drawImage', (ctx, a) => note(ctx, 'drawImage', a.length >= 9 ? a[5] : a[1], a.length >= 9 ? a[6] : a[2],
                                   a.length >= 9 ? a[7] : (a[3] || 20), a.length >= 9 ? a[8] : (a[4] || 20), false));
})();
"""


def _vision_prompt(actions: List[str], question: str) -> str:
    after = f", taken after: {'; '.join(actions)}" if actions else ""
    q = question or "Does the page look complete and working?"
    return ("This is a screenshot of a web page" + after + ".\n"
            f"Question: {q}\n"
            "First line: answer the question with YES, NO or UNSURE, then a few words why.\n"
            "Then describe concretely what is visible: the main content, text, images, game pieces "
            "or characters, and anything that looks empty, covered, cut off, too faint or broken. "
            "Describe only what you can see; don't guess what should be there.")


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
                await page.add_init_script(_CANVAS_SPY_JS)
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
                faint = await page.evaluate(
                    "() => [...(window.__pageLookCanvas || new Map())].slice(0, 12)"
                    ".map(([k, n]) => `${k} (×${n})`)")
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

    answer, seen = "", ""
    try:
        from tools.browser_ext_tool import _vision
        reply = await _vision(_vision_prompt(done, question), base64.b64encode(png).decode(), num_predict=400)
        answer, _, seen = reply.strip().partition("\n")
        seen = seen.strip()
    except Exception as exc:  # noqa: BLE001 — the other findings still stand
        seen = f"(no vision model: {exc})"

    # The answer first, on its own: a "NO" that is easy to explain away
    # in a paragraph ("the animation may not have started") isn't here.
    return make_result(True, {
        **({"answer": answer} if answer else {}),
        "covered": covered or "nothing is covered",
        "canvas_drawing": faint or "nothing drawn invisibly",
        "errors": errors[:20] or "none",
        "seen": seen, "actions_done": done, "title": title, "text": text,
        "url": url, "screenshot": shot,
    }, duration_ms=ms())


_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "page_look",
        "description": (
            "Open a web page, HTML report or browser game you made — a workspace file or a "
            "localhost / LAN URL — in a headless browser, do what the person would (click, press "
            "keys, wait), and look at it: a vision model's YES/NO answer to your question and its "
            "description of the screen, which elements are covered by something else, text or "
            "images drawn on a canvas too faintly or off the canvas to see, JS errors and failed "
            "loads, and a screenshot. Use it "
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
