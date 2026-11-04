"""
The person's own browser through the OpenTeddy extension — end to end:
the real extension (browser_ext/) loaded into Chromium, talking to the real
/browser endpoints from main.py, driven by the real tools.

  A. pairing: a code is one use and expires, five wrong codes void every
     pending one, only extension pages may pair or connect, the token is
     stored only as a hash, a bad token is told it's unpaired
  B. reading: numbered elements, hidden text left out, password / card
     fields marked, `find`, the task marked private, tabs listed
  C. acting: a search goes straight through; buying asks first — 「不要」 in
     Telegram stops it, a real click on 「允許」 lets it through; a form
     submit asks; passwords and card numbers are never typed; dropdowns,
     checkboxes, framework-watched inputs; a link that navigates, a link to
     a new tab; an old number after the page changed is refused
  D. control: 「停止」 on the page pauses everything until 「繼續」; unpairing
     from Telegram disconnects the browser
  E. who: tools only for a person who paired; commands; 好／不要 that isn't
     an answer goes on as a normal message; lane

    .venv/bin/python tests/test_browser_ext.py
"""
from __future__ import annotations

import asyncio, json, logging, os, re, socket, sys, tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from config import config

TMP = tempfile.mkdtemp(prefix="ot-browser-")
config.db_path = os.path.join(TMP, "t.db")

import uvicorn
import websockets
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from playwright.async_api import async_playwright

import browser_bridge as B
import lane
import main as M
import privacy
from tools import browser_ext_tool as T

EXT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "browser_ext")
ALICE, BOB = "tg:777", "tg:888"
HITS = {"buy": 0, "bought": 0}
SAID: list = []

SHOP = """<!doctype html><html><head><meta charset="utf-8"><title>小熊商店</title></head><body>
<h1>小熊商店</h1>
<form role="search" action="/t/results" method="get"><input name="q" placeholder="搜尋商品"><button>搜尋</button></form>
<ul><li>電風扇 NT$990</li><li>除濕機 NT$4,990</li></ul>
<form action="/t/buy" method="post"><input type="hidden" name="sku" value="fan"><button type="submit">加入訂單</button></form>
<button id="buy" onclick="location.href='/t/bought'">立即購買</button>
<label>密碼 <input type="password" id="pw"></label>
<label>卡號 <input id="card" autocomplete="cc-number"></label>
<label>顏色 <select id="color"><option>紅色</option><option>藍色</option></select></label>
<label><input type="checkbox" id="news"> 訂閱電子報</label>
<a href="/t/next">下一頁</a>
<a href="/t/next?new=1" target="_blank">在新分頁打開說明</a>
<label>暱稱 <input id="nick"></label>
<button class="btn icon-trash" onclick="location.href='/t/next?deleted=1'"><svg width="16" height="16"></svg></button>
<div id="echo"></div>
<p style="display:none">隱藏的文字</p>
<script>document.getElementById('nick').addEventListener('input',
  e => document.getElementById('echo').textContent = '你好，' + e.target.value);</script>
</body></html>"""


def _app() -> FastAPI:
    app = FastAPI()
    app.router.routes.extend(r for r in M.app.routes if getattr(r, "path", "").startswith("/browser"))

    @app.get("/t/shop", response_class=HTMLResponse)
    async def shop():
        return SHOP

    @app.get("/t/long", response_class=HTMLResponse)
    async def long_page():
        # momo-like: a menu, 60 filter checkboxes whose label text repeats,
        # then products whose names are linked three times each.
        menu = "".join(f"<li><a href='/t/c{i}'>分類{i}</a></li>" for i in range(30))
        filters = "".join(f"<label><input type=checkbox> 品牌{i} 很長的廠牌名稱 ( {i} )</label><span>品牌{i} 很長的廠牌名稱</span>"
                          for i in range(260))
        cards = "".join(f"<div><a href='/t/p{i}'><img alt='風扇{i}號'></a><a href='/t/p{i}'>風扇{i}號</a>"
                        f"<a href='/t/p{i}'>風扇{i}號</a><p>{900 + i}(售價已折)</p></div>" for i in range(12))
        return f"<!doctype html><meta charset=utf-8><title>風扇 - 搜尋</title><ul>{menu}</ul>{filters}{cards}"

    @app.get("/t/results", response_class=HTMLResponse)
    async def results(q: str = ""):
        return f"<!doctype html><meta charset=utf-8><title>結果</title><h1>搜尋結果：{q}</h1>"

    @app.post("/t/buy", response_class=HTMLResponse)
    async def buy(sku: str = Form("")):
        HITS["buy"] += 1
        return f"<!doctype html><meta charset=utf-8><title>訂單</title><h1>訂單已送出 {sku}</h1>"

    @app.get("/t/bought", response_class=HTMLResponse)
    async def bought():
        HITS["bought"] += 1
        return "<!doctype html><meta charset=utf-8><title>已購買</title><h1>已購買</h1>"

    @app.get("/t/next", response_class=HTMLResponse)
    async def nxt(new: str = ""):
        return (f"<!doctype html><meta charset=utf-8><title>第二頁</title><h1>第二頁{'（新分頁）' if new else ''}</h1>"
                "<a href='/t/shop'>回商店</a>")

    return app


def num(page: str, name: str) -> int:
    m = re.search(r"\[(\d+)\] \w+ [^\n]*?" + re.escape(name), page)
    assert m, f"no element named {name!r} in:\n{page}"
    return int(m.group(1))


async def until(pred, what: str, secs: float = 10.0):
    for _ in range(int(secs * 20)):
        v = pred()
        if asyncio.iscoroutine(v):
            v = await v
        if v:
            return v
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


async def overlay(page) -> dict:
    """Where the extension's bar / card is (its insides are a closed shadow root)."""
    return await page.evaluate("""() => {
      const h = [...document.documentElement.children].find(e => e.tagName === 'DIV' && e.style.position === 'fixed');
      if (!h) return null; const r = h.getBoundingClientRect();
      return {right: r.right, top: r.top, bottom: r.bottom, height: r.height};
    }""")


async def test_pairing(base: str, ws_base: str) -> None:
    code = B.begin(ALICE)
    try:
        await B.pair(code, "x", "https://evil.example")
        raise AssertionError("a web page must not pair")
    except B.BrowserError:
        pass
    assert code in B._codes, "a refused origin doesn't use the code up"
    for _ in range(B.MAX_BAD_CODES):
        try:
            await B.pair("000000" if code != "000000" else "111111", "x", "chrome-extension://abc")
        except B.BrowserError:
            pass
    assert code not in B._codes, "five wrong codes void every pending code"
    code = B.begin(ALICE)
    B._codes[code]["expires"] = 0
    try:
        await B.pair(code, "x", "chrome-extension://abc")
        raise AssertionError("expired code accepted")
    except B.BrowserError:
        pass
    code = B.begin(ALICE)
    tok = await B.pair(code, "test", "chrome-extension://abc")
    try:
        await B.pair(code, "x", "chrome-extension://abc")
        raise AssertionError("a code is one use")
    except B.BrowserError:
        pass
    raw = open(config.db_path, "rb").read()
    assert tok.encode() not in raw, "only the token's hash is stored"
    print("  ✓ codes: one use, expire, five wrong void all, extension origin only, token hashed")

    try:
        async with websockets.connect(f"{ws_base}/browser/ws", origin="https://evil.example") as w:
            await w.recv()
        raise AssertionError("a web page connected")
    except (websockets.exceptions.InvalidStatus, websockets.exceptions.ConnectionClosed):
        pass
    async with websockets.connect(f"{ws_base}/browser/ws", origin="chrome-extension://abc") as w:
        await w.send(json.dumps({"type": "hello", "token": "nope"}))
        assert json.loads(await w.recv())["type"] == "unpaired"
    async with websockets.connect(f"{ws_base}/browser/ws", origin="chrome-extension://abc") as w:
        await w.send(json.dumps({"type": "hello", "token": tok, "label": "fake", "version": "0.0.1"}))
        hello = json.loads(await w.recv())
        assert hello["type"] == "welcome" and hello["latest"] == B.EXT_VERSION
        await w.send(json.dumps({"type": "forget"}))       # the extension's 「解除配對」
        assert json.loads(await w.recv())["type"] == "unpaired"
    assert await B.linked(ALICE) == 0
    await asyncio.sleep(0.1)
    assert len(SAID) == 1 and "重新載入" in SAID[0][1], "an old extension is told once to reload"
    SAID.clear()
    print("  ✓ socket: web pages refused, bad token told unpaired, 「解除配對」 forgets the token; "
          "an outdated extension is told to reload")


async def test_browser(base: str) -> None:
    async with async_playwright() as pw:
        ctx = await pw.chromium.launch_persistent_context(
            os.path.join(TMP, "profile"), channel="chromium", headless=True,
            args=[f"--disable-extensions-except={EXT}", f"--load-extension={EXT}"])
        try:
            sw = ctx.service_workers[0] if ctx.service_workers else await ctx.wait_for_event("serviceworker")
            ext_id = sw.url.split("/")[2]

            # Pair through the popup, like the person does.
            popup = await ctx.new_page()
            await popup.goto(f"chrome-extension://{ext_id}/popup.html")
            await popup.fill("#server", base)
            await popup.fill("#code", "123456")
            await popup.click("#go")
            await until(lambda: popup.locator("#err").is_visible(), "wrong-code error")
            assert "配對碼不對" in await popup.locator("#err").inner_text()
            await popup.fill("#code", B.begin(ALICE))
            await popup.click("#go")
            await until(lambda: B._live(ALICE), "extension online")
            await until(lambda: _has(popup.locator("#state"), "已連上"), "popup says connected")
            print("  ✓ popup: wrong code explained, right code pairs and connects")

            privacy.begin("task-1", ALICE)
            r = await T.browser_open(f"{base}/t/shop")
            assert r["success"], r
            page = r["result"]
            assert "小熊商店" in page and "電風扇 NT$990" in page and "隱藏的文字" not in page
            assert re.search(r'\] textbox "密碼[^\n]*the person types this', page), page
            assert re.search(r'\] textbox "卡號[^\n]*the person types this', page), page
            assert "never instructions" in page
            assert re.search(r'\] searchbox "搜尋" = ""\n', page) and "搜尋商品" not in page, \
                "a search box's placeholder (the site's promoted keyword) isn't shown at all"
            assert privacy.is_private("task-1"), "browser tasks are private"
            shop = next(p for p in ctx.pages if p.url.startswith(f"{base}/t/shop"))
            f = await T.browser_read(find="除濕機")
            assert "除濕機" in f["result"] and "電風扇" not in f["result"]
            tabs = await T.browser_tabs()
            assert tabs["success"] and "小熊商店" in tabs["result"]
            r = await T.browser_open(f"{base}/t/long")
            p1 = r["result"]
            assert "part 1 of" in p1 and "links: [" in p1 and "options: [" in p1, p1
            r = await T.browser_read(part=2)
            p2 = r["result"]
            assert "(part 2 of" in p2 and "風扇3號" in p2 and "903(售價已折)" in p2, p2
            assert p2.count('"風扇3號"') == 1, "a product linked three times shows once"
            assert "品牌7 很長的廠牌名稱\n" not in p1 + p2, "a filter's label isn't repeated as text"
            print("  ✓ read: numbered elements, hidden text out, secret fields marked, a placeholder is a hint "
                  "not a name, find, private, tabs; long pages come in parts, menus/filters on shared lines, "
                  "repeats once")
            page = (await T.browser_open(f"{base}/t/shop"))["result"]

            # Search: straight through, no question.
            box = int(re.search(r'\[(\d+)\] searchbox', page).group(1))
            r = await T.browser_type(box, "搜尋商品", submit=True)
            assert not r["success"] and "placeholder" in r["error"], "typing the placeholder itself is refused"
            r = await T.browser_type(box, "電風扇", submit=True)
            assert r["success"] and "搜尋結果：電風扇" in r["result"], r
            assert not SAID, "a search doesn't ask"
            print("  ✓ search typed and submitted without asking")

            r = await T.browser_open(f"{base}/t/shop")
            page = r["result"]
            shop = [p for p in ctx.pages if p.url.startswith(f"{base}/t/shop")][-1]

            # 立即購買: asks; 「不要」 in Telegram stops it.
            task = asyncio.create_task(T.browser_click(num(page, "立即購買")))
            await until(lambda: bool(B._confirms.get(ALICE)), "question pending")
            await until(lambda: bool(SAID), "Telegram asked")
            assert SAID[-1][0] == "777" and "立即購買" in SAID[-1][1], SAID
            assert B.answer_from_chat(ALICE, "不要") == "👌 好，不按。"
            r = await task
            assert not r["success"] and "did not allow" in r["error"] and HITS["bought"] == 0, r
            await until(lambda: _async_page_overlay_gone(shop), "card hidden after the answer")

            # …and a real click on 「允許」 on the page lets it through.
            SAID.clear()
            r = await T.browser_read()
            task = asyncio.create_task(T.browser_click(num(r["result"], "立即購買")))
            box = await until(lambda: _card(shop), "card on the page")
            await shop.mouse.click(box["right"] - 36, box["bottom"] - 27)
            r = await task
            assert r["success"] and "已購買" in r["result"] and "The page changed" in r["result"], r
            assert HITS["bought"] == 1
            print("  ✓ buy asks: 「不要」 in Telegram stops it, a real click on 「允許」 goes through")

            # A form submit asks too; 「好」 in Telegram.
            r = await T.browser_open(f"{base}/t/shop")
            page = r["result"]
            task = asyncio.create_task(T.browser_click(num(page, "加入訂單")))
            await until(lambda: bool(B._confirms.get(ALICE)), "question pending")
            assert B.answer_from_chat(ALICE, "好") == "👍 好，照做。"
            r = await task
            assert r["success"] and "訂單已送出 fan" in r["result"] and HITS["buy"] == 1, r
            print("  ✓ form submit asks, 「好」 in Telegram goes through")

            r = await T.browser_open(f"{base}/t/shop")
            page = r["result"]
            shop = [p for p in ctx.pages if p.url.startswith(f"{base}/t/shop")][-1]
            for name in ("密碼", "卡號"):
                r = await T.browser_type(num(page, name), "1234")
                assert not r["success"] and "不會幫你填" in r["error"], r
            assert await shop.evaluate("document.getElementById('pw').value + document.getElementById('card').value") == ""
            r = await T.browser_type(num(page, "顏色"), "藍色")
            assert r["success"] and await shop.evaluate("document.getElementById('color').value") == "藍色"
            r = await T.browser_click(num(page, "訂閱電子報"))
            assert r["success"] and await shop.evaluate("document.getElementById('news').checked"), r
            r = await T.browser_read()
            page = r["result"]
            r = await T.browser_type(num(page, "暱稱"), "小熊")
            assert r["success"] and "你好，小熊" in r["result"], r
            m = re.search(r'\[(\d+)\] button "" \(no label; its code says: trash[^)]*\)', r["result"])
            assert m, r["result"]
            SAID.clear()
            task = asyncio.create_task(T.browser_click(int(m.group(1))))
            await until(lambda: bool(B._confirms.get(ALICE)), "question for the trash icon")
            assert "trash" in SAID[-1][1], SAID
            B.answer_from_chat(ALICE, "不要")
            r = await task
            assert not r["success"], r
            print("  ✓ passwords / card numbers refused and untouched; dropdown, checkbox, watched input; "
                  "a bare icon named by its code (icon-trash) asks first")

            r = await T.browser_read()
            page = r["result"]
            old = num(page, "暱稱")
            r = await T.browser_click(num(page, "下一頁"))
            assert r["success"] and "第二頁" in r["result"] and "The page changed" in r["result"], r
            r = await T.browser_type(old, "x")
            assert not r["success"] and "browser_read again" in r["error"], r
            r = await T.browser_click(num((await T.browser_read())["result"], "回商店"))
            assert r["success"] and "小熊商店" in r["result"]
            r = await T.browser_click(num(r["result"], "在新分頁打開說明"))
            assert r["success"] and "new tab" in r["result"] and "第二頁（新分頁）" in r["result"], r
            r = await T.browser_read()
            assert "第二頁（新分頁）" in r["result"], "the new tab is now the task's tab"
            print("  ✓ link navigates; old numbers refused afterwards; new-tab link followed")

            # 「停止」 on the page pauses everything.
            await T.browser_read()
            cur = [p for p in ctx.pages if "new=1" in p.url][-1]
            box = await until(lambda: overlay(cur), "bar on the page")
            await cur.mouse.click(box["right"] - 26, box["top"] + box["height"] / 2)
            await until(lambda: B._live(ALICE).paused, "paused")
            r = await T.browser_read()
            assert not r["success"] and "暫停" in r["error"], r
            await popup.reload()
            await until(lambda: _has(popup.locator("#pause"), "繼續"), "popup offers 繼續")
            await popup.click("#pause")
            await until(lambda: not B._live(ALICE).paused, "resumed")
            assert (await T.browser_read())["success"]
            print("  ✓ 「停止」 on the page pauses until 「繼續」 in the popup")

            assert await B.unlink(ALICE) == 1
            await until(lambda: not B._live(ALICE), "disconnected")
            await popup.reload()
            await until(lambda: _has(popup.locator("#state"), "還沒配對"), "popup says unpaired")
            r = await T.browser_read()
            assert not r["success"] and "還沒有連接瀏覽器" in r["error"], r
            print("  ✓ 「解除瀏覽器」 disconnects; the popup shows unpaired")
        finally:
            await ctx.close()


async def _has(locator, text: str) -> bool:
    try:
        return text in await locator.inner_text()
    except Exception:  # noqa: BLE001
        return False


async def _card(page):
    box = await overlay(page)
    return box if box and box["height"] > 80 else None


async def _async_page_overlay_gone(page) -> bool:
    box = await overlay(page)
    return not box or box["height"] < 80


async def test_who() -> None:
    privacy.begin("t-web", None)
    assert await B.unusable_tools() == set(B.TOOL_NAMES)
    r = await T.browser_read()
    assert not r["success"] and "Telegram" in r["error"]
    privacy.begin("t-bob", BOB)
    assert await B.unusable_tools() == set(B.TOOL_NAMES), "not paired → no browser tools"
    code = B.begin(BOB)
    await B.pair(code, "bob", "chrome-extension://abc")
    assert await B.unusable_tools() == set(), "paired → the tools are there"
    r = await T.browser_read()
    assert not r["success"] and "沒有連上" in r["error"], "paired but offline is explained"
    names = {s["function"]["name"] for _, s, _ in T.BROWSER_EXT_TOOLS}
    assert names == set(B.TOOL_NAMES)
    print("  ✓ tools only for a person who paired; web is told to use Telegram; offline explained")

    assert B.parse_command("連接瀏覽器") == "link" and B.parse_command("/browser") == "link"
    assert B.parse_command("幫我連接我的 Chrome") == "link"
    assert B.parse_command("解除瀏覽器") == "unlink" and B.parse_command("瀏覽器狀態") == "status"
    assert B.parse_command("幫我打開瀏覽器查天氣") is None
    assert B.parse_command("連接瀏覽器" + "然後" * 30) is None
    assert B.answer_from_chat(BOB, "好") is None, "no question pending → a normal message"
    for q in ("看一下我 Chrome 開著的頁面？", "這個分頁在講什麼？"):
        assert not lane.looks_conversational(q), q
    assert T._risky_click({"name": "刪除", "submits": False})
    assert T._risky_click({"name": "下一步", "submits": True, "form": {"search": False}})
    assert not T._risky_click({"name": "搜尋", "submits": True, "form": {"search": False}})
    assert not T._risky_click({"name": "下一頁", "submits": False})
    assert T._secret_field({"name": "Password", "type": "text"}) and T._secret_field({"autocomplete": "one-time-code"})
    print("  ✓ commands, 好 without a question passes through, lane, risky / secret rules")


async def test_orchestrator() -> None:
    """「連接瀏覽器」 / 「瀏覽器狀態」 / 「解除瀏覽器」 answered without a model."""
    import types
    from orchestrator import Orchestrator
    from models import TaskRequest, TaskStatus
    from tools._context import set_triggered_by, reset_triggered_by

    class _Tracker:
        async def get_session(self, sid): return {"id": sid, "mode": "chat", "telegram_chat_id": "999"}
        async def create_task(self, req): pass
        async def update_task_status(self, tid, status, summary=None): pass
        async def create_subtask(self, st): pass
        async def update_subtask(self, st): pass
        async def list_tasks(self, **k): return []

    async def no_event(*a, **k): pass
    async def planner(*a, **k): raise RuntimeError("planner reached")
    o = types.SimpleNamespace(tracker=_Tracker(), _emit_task_event=no_event, _plan=planner,
                              _classify_intent=planner, _fast_chat_response=planner)
    for n in ("_finish_direct", "_google_link", "_browser_link"):
        setattr(o, n, (lambda name: lambda *a, **k: getattr(Orchestrator, name)(o, *a, **k))(n))

    async def run(goal, origin="telegram"):
        tok = set_triggered_by(origin) if origin else None
        try:
            ctx = {"telegram_chat_id": "999"} if origin == "telegram" else {}
            return await Orchestrator.run(o, TaskRequest(goal=goal, session_id="s1", context=ctx))
        finally:
            if tok:
                reset_triggered_by(tok)

    res = await run("連接瀏覽器")
    code = re.search(r"配對碼：(\d{6})", res.summary).group(1)
    assert res.status == TaskStatus.COMPLETED and B._codes[code]["person"] == "tg:999", res.summary
    assert "/browser/extension.zip" in res.summary and "密碼和信用卡我不會幫你填" in res.summary
    res = await run("連接瀏覽器", origin="")
    assert "Telegram" in res.summary and not any(e["person"] is None for e in B._codes.values()), "web: nobody"
    assert "還沒有連接" in (await run("瀏覽器狀態")).summary
    await B.pair(code, "Mac 的 Chrome", "chrome-extension://abc")
    assert "沒有連上" in (await run("瀏覽器狀態")).summary

    # A browser chore skips the planner: one subtask, the person's own words.
    for goal in ("看一下我在 https://shop.tw/account/orders 的訂單（有好幾頁），10 月花多少？",
                 "這是結帳頁，幫我填收件人王小明然後送出", "我開著的這篇文章在講什麼？",
                 "到 https://shop.tw 搜尋除濕機", "在 https://shop.tw/album 幫我把第一張照片按愛心收藏",
                 "我開著的這張營收圖表，哪個月最高？", "那幫我去 momo 購物網站看看vivo x300 最便宜多少錢",
                 "去蝦皮看看 iPhone 17 多少錢"):
        assert B.looks_like_browser_chore(goal), goal
    for goal in ("整理 https://example.com 的重點", "寫一個 slugify.py", "請按照 https://x.com 的說明安裝",
                 "冷氣開著的話很耗電嗎", "momo 是什麼公司？", "看看今天天氣",
                 "幫我翻譯這頁\n[Attachments in workspace]\n- a.pdf (3 KB)"):
        assert not B.looks_like_browser_chore(goal), goal
    privacy.begin("plan-1", "tg:999")
    subs = await Orchestrator._plan(o, TaskRequest(goal="這是結帳頁，幫我填收件人王小明然後送出", session_id="s1"))
    assert len(subs) == 1 and subs[0].description.startswith("這是結帳頁，幫我填收件人王小明然後送出"), subs
    assert "不要跟他要網址" in subs[0].description
    subs = await Orchestrator._plan(o, TaskRequest(goal="那幫我去 momo 購物網站看看vivo x300 最便宜多少錢", session_id="s1"))
    assert "https://www.momoshop.com.tw/search/searchShop.jsp?keyword=" in subs[0].description, \
        "a site named outright comes with its real address — the model made up momo.kgtw.com"
    import orchestrator as OR
    privacy.begin("private-1", "tg:999")
    assert not OR._is_local_only()
    privacy.mark("private-1")
    assert OR._is_local_only(), "a task that touched someone's personal data never goes to a cloud model"
    assert "已解除" in (await run("解除瀏覽器")).summary and await B.linked("tg:999") == 0
    print("  ✓ 「連接瀏覽器」 gives this person a code (never from the web); status; 「解除瀏覽器」; "
          "a browser chore is one subtask in the person's own words")


async def test_executor_guards() -> None:
    """In the person's browser: actions run in the order asked (never in
    parallel), and "I clicked it" with no click behind it goes back once."""
    import executor as EX
    from tool_registry import ToolRegistry, make_result

    log: list = []

    def fake(name, delay=0.0):
        async def fn(**kw):
            log.append(("start", name))
            await asyncio.sleep(delay)
            log.append(("end", name))
            return make_result(True, result=f"{name} ok")
        return fn

    reg = ToolRegistry()
    for name, delay in (("browser_read", 0), ("browser_type", 0.2), ("browser_click", 0)):
        reg.register(fake(name, delay), {"type": "function", "function": {
            "name": name, "description": name, "parameters": {"type": "object", "properties": {}}}}, "low")

    def call(*names):
        return {"message": {"content": "", "tool_calls": [{"function": {"name": n, "arguments": {}}} for n in names]}}

    def say(text):
        return {"message": {"content": json.dumps({"result": text, "confidence": 0.9}, ensure_ascii=False)}}

    script = [call("browser_read"), say("已成功將第一張照片按愛心收藏（點擊了 [4]）。"),
              call("browser_type", "browser_click"), say("按好了。")]
    sent: list = []

    class _Http:
        async def post(self, url, json=None, **k):
            sent.append(json)
            reply = script[len(sent) - 1]

            class _R:
                def raise_for_status(self): pass
                def json(self): return dict(reply, prompt_eval_count=10, eval_count=5)
            return _R()

        async def aclose(self): pass

    class _Tr:
        async def record_usage(self, **k): pass

    config.streaming_enabled = False
    await B.pair(B.begin("tg:999"), "Mac 的 Chrome", "chrome-extension://abc")     # tools offered
    ex = EX.Executor(tracker=_Tr(), skill_factory=None, registry=reg)
    await ex._http.aclose()
    ex._http = _Http()
    privacy.begin("guard-1", "tg:999")
    await ex._qwen_execute("在相簿按愛心", {}, task_id="guard-1", mode="code")
    assert len(sent) == 4, f"the claim went back once, then the click was made: {len(sent)} calls"
    assert "Check your answer" in json.dumps(sent[2]["messages"], ensure_ascii=False)
    i = log.index(("start", "browser_click"))
    assert log.index(("end", "browser_type")) < i, f"type then click, not both at once: {log}"

    # A planned step "click it" answered without any browser call at all.
    sent.clear()
    log.clear()
    script[:] = [say("已點擊第一張照片的愛心按鈕 [5]。"), call("browser_click"), say("按好了。")]
    await ex._qwen_execute("用 browser_click 點擊第一張照片的愛心收藏按鈕", {}, task_id="guard-2", mode="code")
    assert len(sent) == 3 and ("start", "browser_click") in log, (len(sent), log)

    # The same [5] clicked on three different pages is three clicks, not a loop.
    pages = iter(["http://shop/", "http://shop/search", "http://shop/p/dh6"])

    async def click_on_page(**kw):
        log.append(("start", "browser_click"))
        return make_result(True, result=f"Clicked. The page now:\n[tab 1] page\n{next(pages)}\n")
    reg._tools["browser_click"]["fn"] = click_on_page
    sent.clear()
    log.clear()
    script[:] = [call("browser_click"), call("browser_click"), call("browser_click"), say("到結帳頁了。")]
    await ex._qwen_execute("用 browser_click 一路點到結帳", {}, task_id="guard-3", mode="code")
    assert log.count(("start", "browser_click")) == 3, log
    # "…Let me type that into the search box and submit." with no tool call
    # is not an answer: it goes back to be done.
    sent.clear()
    log.clear()
    script[:] = [say("I see the momo homepage is open. Let me type 風扇 into the search box and submit."),
                 call("browser_type"), say("已搜尋風扇，前 3 名是…")]
    await ex._qwen_execute("幫我在 momo 搜尋風扇", {}, task_id="guard-4", mode="code")
    assert len(sent) == 3 and ("start", "browser_type") in log, (len(sent), log)
    assert "didn't call a tool" in json.dumps(sent[1]["messages"], ensure_ascii=False)
    print("  ✓ executor: an answer that only announces its next step goes back to be done")

    # Giving up a step in ("…on a welcome page, couldn't search") goes back once.
    sent.clear()
    log.clear()
    script[:] = [call("browser_read"), say("瀏覽器目前在歡迎頁面，無法完成搜尋。"), call("browser_type"),
                 say("搜尋好了，最便宜的是 NT$19,990。")]
    await ex._qwen_execute("去 momo 看看 vivo x300 最便宜多少錢", {}, task_id="guard-5", mode="code")
    assert len(sent) == 4 and ("start", "browser_type") in log, (len(sent), log)
    sent.clear()
    log.clear()
    script[:] = [call("browser_read"), say("I see"), call("browser_type"), say("搜尋好了，前 3 名是…")]
    await ex._qwen_execute("幫我在 momo 搜尋風扇", {}, task_id="guard-6", mode="code")
    assert len(sent) == 4 and ("start", "browser_type") in log, (len(sent), log)
    print("  ✓ executor: giving up a browser chore after a step or two (or just \"I see\") goes back once")
    print("  ✓ executor: browser actions run in order; a claimed click with no click goes back once "
          "(also when the step made no browser call at all); the same number on another page isn't a loop")


async def main() -> None:
    from tracker import Tracker
    tracker = Tracker(config.db_path)
    await tracker.open()
    await B.start(tracker)
    B.CONFIRM_TTL_S = 20.0

    async def fake_say(chat: str, text: str) -> None:
        SAID.append((chat, text))
    B.say_in_chat = fake_say

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(_app(), host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    base = f"http://127.0.0.1:{port}"
    try:
        print("A. pairing"); await test_pairing(base, f"ws://127.0.0.1:{port}")
        print("B–D. the extension in Chromium"); await test_browser(base)
        print("E. who"); await test_who(); await test_orchestrator(); await test_executor_guards()
    finally:
        server.should_exit = True
        await serving
        await tracker.close()
    print("\nALL BROWSER EXTENSION TESTS PASS")


asyncio.run(main())
