"""
The browser side of the web tasks: a headless Chromium with the real
OpenTeddy extension (browser_ext/), paired to the eval's own person through
the extension's popup, plus the test website (web_site.py).

When OpenTeddy asks before an action, the extension shows its card on the
page; the rig answers it the way the task says (`confirm: allow | deny`)
with a real mouse click, and counts the questions. Only questions on the
test website are answered.

Playwright runs in its own thread and event loop; run_eval.py calls the
public methods from its own thread.
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional

import httpx

import web_site

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXT = os.path.join(ROOT, "browser_ext")

_OVERLAY = """() => {
  const h = [...document.documentElement.children].find(e => e.tagName === 'DIV' && e.style.position === 'fixed');
  if (!h) return null; const r = h.getBoundingClientRect();
  return {right: r.right, bottom: r.bottom, height: r.height};
}"""


class BrowserRig:
    def __init__(self, server: str, key: str) -> None:
        self.server = server.rstrip("/")
        self.key = key
        self.site_url = ""
        self.policy = "deny"
        self.asked: List[Dict[str, str]] = []
        self._srv = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ctx = None
        self._pw = None
        self._ext_id = ""
        self._watch: Optional[asyncio.Task] = None

    # ── Called from run_eval's thread ────────────────────────────────────────

    def start(self) -> None:
        self._srv, self.site_url = web_site.start()
        ready = threading.Event()

        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()

        threading.Thread(target=_run, daemon=True, name="eval-browser").start()
        ready.wait()
        self._call(self._start(), 120)

    def begin_task(self, browser: Dict[str, Any]) -> None:
        web_site.reset()
        self.asked = []
        self.policy = str(browser.get("confirm") or "deny")
        self._call(self._fresh_tabs(browser.get("start")), 60)

    def end_task(self) -> Dict[str, Any]:
        return {"site": web_site.snapshot(), "asked": list(self.asked)}

    def stop(self) -> None:
        if self._loop:
            try:
                self._call(self._stop(), 30)
            except Exception:  # noqa: BLE001
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._srv:
            self._srv.shutdown()

    def _call(self, coro, timeout: float) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    # ── In the browser thread ────────────────────────────────────────────────

    async def _start(self) -> None:
        from playwright.async_api import async_playwright
        self._pw = await async_playwright().start()
        self._ctx = await self._pw.chromium.launch_persistent_context(
            tempfile.mkdtemp(prefix="ot-eval-browser-"), channel="chromium", headless=True, locale="zh-TW",
            viewport={"width": 1280, "height": 900},
            args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}"])
        sw = self._ctx.service_workers[0] if self._ctx.service_workers else \
            await self._ctx.wait_for_event("serviceworker", timeout=30000)
        self._ext_id = sw.url.split("/")[2]
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.post(f"{self.server}/browser/eval-code", json={"key": self.key})
        if r.status_code != 200:
            raise RuntimeError(f"eval pairing code refused ({r.status_code}): is .keys/browser_eval.key from this server?")
        popup = await self._ctx.new_page()
        await popup.goto(f"chrome-extension://{self._ext_id}/popup.html")
        await popup.fill("#server", self.server)
        await popup.fill("#code", r.json()["code"])
        await popup.click("#go")
        for _ in range(100):
            if "已連上" in (await popup.locator("#state").inner_text()):
                break
            await asyncio.sleep(0.2)
        else:
            raise RuntimeError("the extension didn't connect: " + await popup.locator("#err").inner_text())
        await popup.close()
        self._watch = asyncio.create_task(self._answer_questions())

    async def _fresh_tabs(self, start: Optional[str]) -> None:
        keep = await self._ctx.new_page()
        for p in list(self._ctx.pages):
            if p is not keep:
                await p.close()
        if start:
            await keep.goto(self.site_url + start)
        await keep.bring_to_front()

    async def _answer_questions(self) -> None:
        """A card on a test-site page → click 允許 or 不要 like a person would."""
        while True:
            await asyncio.sleep(0.3)
            for page in list(self._ctx.pages):
                if not page.url.startswith(self.site_url):
                    continue
                try:
                    box = await page.evaluate(_OVERLAY)
                except Exception:  # noqa: BLE001 — navigating
                    continue
                if not box or box["height"] < 80:
                    continue
                # Buttons sit right-aligned at the card's bottom: 允許 rightmost, 不要 left of it.
                x = box["right"] - (39 if self.policy == "allow" else 97)
                await page.mouse.click(x, box["bottom"] - 28)
                self.asked.append({"url": page.url, "answer": self.policy})
                await asyncio.sleep(1.0)

    async def _stop(self) -> None:
        if self._watch:
            self._watch.cancel()
        try:
            popup = await self._ctx.new_page()
            await popup.goto(f"chrome-extension://{self._ext_id}/popup.html")
            popup.on("dialog", lambda d: asyncio.ensure_future(d.accept()))
            await popup.click("#unpair")                 # forget the eval's token on the server
            await asyncio.sleep(1.0)
        except Exception:  # noqa: BLE001
            pass
        await self._ctx.close()
        await self._pw.stop()
