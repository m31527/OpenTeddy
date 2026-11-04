"""
The web-task harness itself (tests/eval/browser_rig.py, web_site.py and the
site checks), with the model replaced by a script: the rig pairs the real
extension through /browser/eval-code, the scripted tool calls run as the
eval's person, and the rig answers OpenTeddy's questions on the page.

  - the eval key: wrong key refused; a task is the eval's person only when
    the Task API call carried the key, and reusing its id without the key
    doesn't inherit it; the eval person's tasks aren't marked private
  - allow: a support form is submitted and recorded, the question counted
  - deny: 「立即購買」 is refused on the page, nothing is ordered
  - the checks read the site state

    .venv/bin/python tests/test_eval_web.py
"""
from __future__ import annotations

import asyncio, json, logging, os, re, socket, sys, tempfile, types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "eval"))
logging.disable(logging.CRITICAL)

from config import config

TMP = tempfile.mkdtemp(prefix="ot-evalweb-")
config.db_path = os.path.join(TMP, "t.db")

import httpx
import uvicorn
from fastapi import FastAPI

import browser_bridge as B
import main as M
import privacy
import web_site
from browser_rig import BrowserRig
from checks import run_checks
from tools import browser_ext_tool as T


def num(page: str, name: str) -> int:
    m = re.search(r"\[(\d+)\] \w+ \"[^\"]*" + re.escape(name), page)
    assert m, f"no {name!r} in:\n{page}"
    return int(m.group(1))


async def scripted(rig: BrowserRig) -> None:
    loop = asyncio.get_running_loop()

    # allow: the support form goes through after the question.
    await loop.run_in_executor(None, rig.begin_task, {"confirm": "allow"})
    privacy.begin("eval-1", B.EVAL_PERSON)
    page = (await T.browser_open(f"{rig.site_url}/support"))["result"]
    assert "聯絡客服" in page, page
    for name, text in (("姓名", "王小明"), ("電話", "0912345678"), ("問題類型", "冷氣維修"),
                       ("問題描述", "冷氣不冷，請派人來修")):
        r = await T.browser_type(num(page, name), text)
        assert r["success"], r
    r = await T.browser_click(num(page, "送出"))
    assert r["success"] and "已收到你的問題" in r["result"], r
    assert not privacy.is_private("eval-1"), "the eval's own browser isn't private data"
    got = rig.end_task()
    assert len(got["asked"]) == 1 and got["asked"][0]["answer"] == "allow", got
    res = run_checks(dict(got, answer="", tools=[], ws=TMP), [
        {"check": "site_record", "what": "messages", "all": ["王小明", "0912345678", "冷氣維修", "冷氣不冷"]},
        {"check": "asked", "min": 1}])
    assert all(c["ok"] for c in res), res
    print("  ✓ allow: the form is submitted after a real click on 允許; checks read the site")

    # deny: 「立即購買」 is refused on the page.
    await loop.run_in_executor(None, rig.begin_task, {"confirm": "deny", "start": "/shop/p/dh6"})
    privacy.begin("eval-2", B.EVAL_PERSON)
    page = (await T.browser_read())["result"]
    assert "北極熊 6L 除濕機" in page, page
    r = await T.browser_click(num(page, "立即購買"))
    assert not r["success"] and "did not allow" in r["error"], r
    got = rig.end_task()
    assert [a["answer"] for a in got["asked"]] == ["deny"] and "/shop/checkout" not in got["site"]["pages"], got
    res = run_checks(dict(got, answer="", tools=[], ws=TMP), [
        {"check": "site_count", "what": "orders", "equals": 0}, {"check": "asked", "min": 1},
        {"check": "site_never", "any": ["4111 1111 1111 1111"]}])
    assert all(c["ok"] for c in res), res
    print("  ✓ deny: a real click on 不要 stops 立即購買; nothing ordered")

    # By sight: a screenshot goes to the vision model (faked here with the
    # real geometry), the click lands where it says.
    async def where(js: str) -> dict:
        async def _ev():
            page = [p for p in rig._ctx.pages if p.url.startswith(rig.site_url)][-1]
            return await page.evaluate(js)
        return await loop.run_in_executor(None, rig._call, _ev(), 20)

    shots = []

    async def fake_vision(prompt, image, num_predict=200):
        shots.append(image)
        if "Point to" in prompt:
            return json.dumps({"point_2d": [round(target["x"] / target["w"] * 1000), round(target["y"] / target["h"] * 1000)]})
        return "7 月最高，260 萬元。"
    T._vision = fake_vision

    await loop.run_in_executor(None, rig.begin_task, {"confirm": "deny", "start": "/game"})
    privacy.begin("eval-3", B.EVAL_PERSON)
    await asyncio.sleep(0.5)
    target = await where("(() => { const r = document.getElementById('c').getBoundingClientRect();"
                         " return {x: r.left + 160, y: r.top + 145, w: innerWidth, h: innerHeight}; })()")
    r = await T.browser_click_at("the 開始遊戲 button")
    assert r["success"] and "browser_look" in r["result"], r
    assert rig.end_task()["site"]["game"] == ["start"]
    assert shots and shots[-1].startswith("/9j/"), "a JPEG screenshot went to the vision model"
    print("  ✓ by sight: a click on a canvas lands where the vision model points (開始遊戲 started)")

    await loop.run_in_executor(None, rig.begin_task, {"confirm": "deny"})
    privacy.begin("eval-4", B.EVAL_PERSON)
    await T.browser_open(f"{rig.site_url}/album")
    target = await where("(() => { const r = document.querySelectorAll('.card button')[0].getBoundingClientRect();"
                         " return {x: r.left + r.width / 2, y: r.top + r.height / 2, w: innerWidth, h: innerHeight}; })()")
    r = await T.browser_click_at("the trash icon of the first photo")
    got = rig.end_task()
    assert not r["success"] and "did not allow" in r["error"], r
    assert len(got["asked"]) == 1 and got["site"]["deleted"] == [], got
    r = await T.browser_look("哪個月營收最高？")
    assert r["success"] and "260" in r["result"], r
    print("  ✓ by sight: a trash icon asks first (refused, nothing deleted); browser_look answers")

    from config import config
    saved, config.vision_model = config.vision_model, ""
    try:
        assert B.VISION_TOOLS <= await B.unusable_tools(), "no vision model → no vision tools"
    finally:
        config.vision_model = saved
    assert not (B.VISION_TOOLS & await B.unusable_tools())
    print("  ✓ the vision tools are hidden when no vision model is set")


async def main() -> None:
    from tracker import Tracker
    tracker = Tracker(config.db_path)
    await tracker.open()
    await B.start(tracker)
    key = B.eval_key()
    assert oct(os.stat(os.path.join(TMP, ".keys", "browser_eval.key")).st_mode & 0o777) == "0o600"

    app = FastAPI()
    app.router.routes.extend(r for r in M.app.routes if getattr(r, "path", "").startswith("/browser"))
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="off"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    base = f"http://127.0.0.1:{port}"
    rig = None
    try:
        async with httpx.AsyncClient() as c:
            r = await c.post(f"{base}/browser/eval-code", json={"key": "wrong"})
            assert r.status_code == 403
        B.unmark_eval_task("t-x")
        assert not B.is_eval_task("t-x")
        B.mark_eval_task("t-x")
        req = types.SimpleNamespace(id="t-x", context={}, session_id="")
        assert await privacy.resolve(req, None) == B.EVAL_PERSON
        B.unmark_eval_task("t-x")                     # what /tasks does for a call without the key
        assert await privacy.resolve(req, None) is None
        print("  ✓ eval key: wrong key refused; the eval person only with the key, not by reusing a task id")

        rig = BrowserRig(base, key)
        await asyncio.get_running_loop().run_in_executor(None, rig.start)
        assert await B.linked(B.EVAL_PERSON) == 1
        print("  ✓ the rig pairs the real extension as the eval's person")
        await scripted(rig)
    finally:
        if rig:
            await asyncio.get_running_loop().run_in_executor(None, rig.stop)
        await asyncio.sleep(0.5)
        assert await B.linked(B.EVAL_PERSON) == 0, "stopping the rig unpairs it"
        server.should_exit = True
        await serving
        await tracker.close()
    print("  ✓ stopping the rig unpairs it")
    print("\nALL EVAL WEB TESTS PASS")


asyncio.run(main())
