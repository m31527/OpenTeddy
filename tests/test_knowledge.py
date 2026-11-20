"""
The knowledge base (knowledge.py): datasets of FAQs / notes answers come from.

  A. files: a question/answer CSV (also Big5 from Excel, and TSV) becomes
     one entry per row; any other table "column: value"; a text file is cut
     into pieces with its headings kept; wrong types, empty and huge files
     are explained
  B. search: Chinese questions find their FAQ row without a tokenizer;
     small talk and unrelated work find nothing; the same file again
     replaces its rows; deleting a file / an entry / a dataset; meaning
     search (a fake embedder) finds a differently worded entry, and a
     failing embedder leaves keyword search working
  C. scope: each session its own picks; a task's own datasets replace them;
     "always" = every session; off = nothing
  D. prompts: the chat lane, the gate and the executor get the matching
     entries with the rule; nothing when nothing matches; the search tool
     only for a task with datasets
  E. the /knowledge API and POST /tasks {"datasets": [...]}

    .venv/bin/python tests/test_knowledge.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from config import config

TMP = tempfile.mkdtemp(prefix="ot-kb-")
config.db_path = os.path.join(TMP, "t.db")

import knowledge as K

FAQ = """問題,答案,分類
退貨期限是幾天？,收到商品後 7 天內可以申請退貨，商品須保持全新。,售後
運費怎麼算？,單筆滿 NT$990 免運，未滿收 NT$80。,物流
可以開統編嗎？,可以，結帳時在發票欄位填寫統一編號。,發票
營業時間？,客服週一到週五 9:00–18:00。,客服
有哪些付款方式？,信用卡、LINE Pay、ATM 轉帳、超商取貨付款。,付款
多久會出貨？,付款後 1–2 個工作天出貨。,物流
商品有保固嗎？,電器類享原廠一年保固。,售後
退款多久會入帳？,退款核准後 5–7 個工作天退回原付款方式。,售後
"""


def expect_error(fn, *a, contains: str = "") -> None:
    try:
        fn(*a)
    except K.KnowledgeError as exc:
        assert contains in str(exc), exc
        return
    raise AssertionError(f"no error from {fn.__name__}{a!r:.80}")


# ── A ────────────────────────────────────────────────────────────────────────

def test_files() -> None:
    rows = K.parse_file("faq.csv", FAQ.encode("cp950"))
    assert len(rows) == 8 and rows[0] == {"ref": "第 2 列", "title": "退貨期限是幾天？",
                                          "body": "收到商品後 7 天內可以申請退貨，商品須保持全新。\n分類：售後"}, rows[0]
    assert K.parse_file("faq.csv", FAQ.encode("utf-8-sig")) == rows, "UTF-8 with BOM reads the same"
    tsv = "Question\tAnswer\nHow long is shipping?\t1-2 days\n"
    assert K.parse_file("faq.tsv", tsv.encode()) == [{"ref": "第 2 列", "title": "How long is shipping?", "body": "1-2 days"}]
    table = K.parse_file("products.csv", "品名,價格,顏色\n電風扇,990,白\n除濕機,4990,\n".encode())
    assert table[0] == {"ref": "第 2 列", "title": "電風扇", "body": "品名：電風扇\n價格：990\n顏色：白"}, table
    assert table[1]["body"] == "品名：除濕機\n價格：4990", "empty cells are left out"
    print("  ✓ CSV: a question/answer pair per row (UTF-8, BOM, Big5, TSV); other tables as column: value")

    text = ("公司簡介\n\n我們成立於 2015 年，總部在台北。" + "這是一段很長的說明。" * 160
            + "\n\n聯絡方式\n電話 02-1234-5678\n\n" + "Short English note. " * 3)
    pieces = K.parse_file("about.txt", text.encode())
    assert pieces[0]["title"] == "公司簡介" and pieces[0]["body"].startswith("公司簡介\n我們成立於"), pieces[0]
    assert all(len(p["body"]) <= K.CHUNK_CHARS * 1.7 for p in pieces), [len(p["body"]) for p in pieces]
    assert any(p["title"] == "聯絡方式" and "02-1234-5678" in p["body"] for p in pieces), pieces
    assert [p["ref"] for p in pieces] == [f"第 {i} 段" for i in range(1, len(pieces) + 1)]
    print("  ✓ TXT: cut into pieces of about 600 characters, a heading kept with its paragraph")

    expect_error(K.parse_file, "faq.xlsx", b"x", contains=".csv")
    expect_error(K.parse_file, "empty.csv", b"\n\n", contains="空的")
    expect_error(K.parse_file, "head.csv", "問題,答案\n".encode(), contains="只有標題列")
    expect_error(K.parse_file, "blank.txt", b"   \n", contains="沒有文字")
    expect_error(K.parse_file, "big.txt", b"x" * (K.MAX_FILE_BYTES + 1), contains="太大")
    print("  ✓ wrong type, empty, header-only and oversized files are explained")


# ── B ────────────────────────────────────────────────────────────────────────

async def test_search() -> None:
    d = await K.create_dataset("客服 FAQ", "常見問題")
    r = await K.add_file(d["id"], "faq.csv", FAQ.encode("cp950"))
    assert r["added"] == 8 and r["replaced"] == 0 and not r["semantic"]
    ids = [d["id"]]
    cases = {
        "請問你們的退貨期限是幾天呀": "退貨期限是幾天？",
        "運費多少": "運費怎麼算？",
        "我要開發票打統編": "可以開統編嗎？",
        "出貨要等多久": "多久會出貨？",
        "保固多久": "商品有保固嗎？",
        "line pay 可以用嗎": "有哪些付款方式？",
        "客服幾點上班": "營業時間？",
    }
    for q, want in cases.items():
        hits = await K.search(q, ids, k=3)
        assert hits and hits[0].title == want, (q, [h.title for h in hits])
    for q in ("你好", "今天天氣如何", "幫我寫一個 python 程式", "商品推薦有哪些", "可以用街口支付嗎"):
        assert not await K.search(q, ids), (q, [h.title for h in await K.search(q, ids)])
    assert [h.title for h in await K.search("保固", ids, loose=True)][:1] == ["商品有保固嗎？"]
    titles = {h.title for h in await K.search("顧客問：退貨的運費是誰出？請回覆他。", ids)}
    assert {"退貨期限是幾天？", "運費怎麼算？"} <= titles, "each clause is matched on its own too"
    for q in ("幫我寫一封信給客戶說明我們的退貨流程很複雜", "推薦我一台電風扇，預算三千元，要安靜的"):
        assert not await K.search(q, ids), q
    print("  ✓ Chinese questions find their row (also inside a longer request); small talk and unrelated work "
          "find nothing")

    again = "問題,答案\n退貨期限是幾天？,收到商品後 10 天內可以退貨。\n".encode()
    r = await K.add_file(d["id"], "faq.csv", again)
    assert r["added"] == 1 and r["replaced"] == 8
    hits = await K.search("退貨期限", ids)
    assert len(hits) == 1 and "10 天" in hits[0].body, "the same file name replaced its rows (and the index)"
    await K.add_file(d["id"], "faq.csv", FAQ.encode())
    await K.add_file(d["id"], "about.txt", "營業時間\n門市每天 11:00–21:00 營業。".encode())
    ds = (await K.list_datasets())[0]
    assert ds["items"] == 9 and [s["name"] for s in ds["sources"]] == ["faq.csv", "about.txt"], ds
    assert await K.delete_source(d["id"], "about.txt") == 1
    items = await K.list_items(d["id"], q="保固")
    assert items["total"] == 1 and await K.delete_item(items["items"][0]["id"])
    assert not await K.search("保固多久", ids), "a deleted entry is gone from search"
    assert (await K.get_dataset("客服 faq"))["id"] == d["id"], "found by name, any case"
    try:
        await K.resolve([d["id"], "沒有這個"])
        raise AssertionError("unknown dataset accepted")
    except K.KnowledgeError as exc:
        assert "沒有這個" in str(exc)
    try:
        await K.create_dataset("客服 FAQ")
        raise AssertionError("duplicate name accepted")
    except K.KnowledgeError:
        pass
    print("  ✓ same file again replaces its rows; files, entries and names managed; unknown datasets named")

    # Meaning search with a fake embedder: 退錢 shares no words with 退款.
    real_embed, real_model = K.embed, config.knowledge_embed_model
    concept = {"退款": 0, "退錢": 0, "入帳": 0, "運費": 1, "免運": 1, "發票": 2, "統編": 2}

    async def fake_embed(texts):
        out = []
        for t in texts:
            v = [0.05] * 4
            for w, dim in concept.items():
                if w in t:
                    v[dim] += 1.0
            n = sum(x * x for x in v) ** 0.5
            out.append([x / n for x in v])
        return out
    try:
        config.knowledge_embed_model = "fake-embed"
        K.embed = fake_embed
        assert not await K.search("我想退錢", ids), "keyword search alone can't find it"
        r = await K.reindex(d["id"])
        assert r["updated"] == 7 and (await K.list_datasets())[0]["semantic"] == 7
        hits = await K.search("我想退錢", ids)
        assert hits and hits[0].title == "退款多久會入帳？" and hits[0].similarity > 0.9, [h.as_dict() for h in hits]
        assert (await K.search("運費多少", ids))[0].title == "運費怎麼算？", "keyword hits still rank first"

        async def broken(texts):
            raise RuntimeError("model not loaded")
        K.embed = broken
        r = await K.add_file(d["id"], "more.csv", "問題,答案\n可以自取嗎？,可以，台北門市自取。\n".encode())
        assert not r["semantic"] and "關鍵字" in r["note"], r
        assert (await K.search("可以自取嗎", ids))[0].title == "可以自取嗎？", "keyword search carries on"
    finally:
        K.embed, config.knowledge_embed_model = real_embed, real_model
    await K.delete_source(d["id"], "more.csv")
    print("  ✓ meaning search finds 退錢 → 退款 once indexed; a broken embedder leaves keyword search working")


# ── C ────────────────────────────────────────────────────────────────────────

async def test_scope() -> None:
    faq = await K.get_dataset("客服 FAQ")
    internal = await K.create_dataset("內部規定")
    assert not faq["always"] and not internal["always"], "a new dataset is no session's until picked"
    assert await K.scope_for({}, "s-shop") == () and await K.scope_for({}, "s-other") == ()
    assert await K.set_session_datasets("s-shop", ["客服 FAQ"]) == [faq["id"]]
    assert await K.scope_for({}, "s-shop") == (faq["id"],)
    assert await K.scope_for({}, "s-other") == (), "another session isn't affected"
    config.knowledge_enabled = False
    assert await K.scope_for({}, "s-shop") == () and await K.scope_for({"knowledge_datasets": [faq["id"]]}) == ()
    config.knowledge_enabled = True
    assert await K.scope_for({"knowledge_datasets": [internal["id"]]}, "s-shop") == (internal["id"],), \
        "a task's own datasets replace the session's for that task"
    assert await K.scope_for({"knowledge_datasets": []}, "s-shop") == (), "an API call's [] means none"
    await K.update_dataset(internal["id"], always=True)
    assert await K.scope_for({}, "s-other") == (internal["id"],), "marked always: every session"
    assert await K.scope_for({}, "s-shop") == (faq["id"], internal["id"])
    await K.update_dataset(internal["id"], always=False)
    assert await K.set_session_datasets("s-shop", []) == [] and await K.scope_for({}, "s-shop") == ()
    tmp = await K.create_dataset("暫時")
    await K.set_session_datasets("s-tmp", [tmp["id"]])
    await K.delete_dataset(tmp["id"])
    assert await K.session_datasets("s-tmp") == [], "a deleted dataset leaves the sessions that used it"
    try:
        await K.set_session_datasets("s-shop", ["沒有這個"])
        raise AssertionError("unknown accepted")
    except K.KnowledgeError:
        pass
    await K.set_session_datasets("s-shop", ["客服 FAQ"])
    print("  ✓ per session: picked datasets only there; a task's own replace them ([] = none); always = every "
          "session; off = nothing")


# ── D ────────────────────────────────────────────────────────────────────────

class _Stop(Exception):
    pass


async def test_prompts() -> None:
    from orchestrator import Orchestrator
    from models import TaskRequest
    import gate
    faq = await K.get_dataset("客服 FAQ")
    K.begin([faq["id"]])
    seen: dict = {}

    async def recent(req):
        return ""

    async def complete(prompt, system, **k):
        seen["chat"] = prompt
        raise _Stop()
    o = types.SimpleNamespace(memory=None, _recent_turns_block=recent, _orchestrator_complete=complete)
    try:
        await Orchestrator._fast_chat_response(o, TaskRequest(goal="退貨期限是幾天", session_id="s"), "chat", {})
    except _Stop:
        pass
    p = seen["chat"]
    assert "【知識庫：以這裡為準】" in p and "收到商品後 7 天內" in p and "faq.csv 第 2 列" in p, p
    assert K.RULE in p and p.index("【知識庫") < p.index("退貨期限是幾天", p.index(K.RULE)), p

    real_ask = gate.ask

    async def fake_ask(orch, prompt, task_id=""):
        seen["gate"] = prompt
        raise _Stop()
    gate.ask = fake_ask
    try:
        for goal, has in (("運費怎麼算", True), ("你好", False)):
            try:
                await Orchestrator._fast_chat_response(
                    o, TaskRequest(goal=goal, session_id="s", context={"gate": "code"}), "chat", {})
            except _Stop:
                pass
            assert ("【知識庫" in seen["gate"]) == has, (goal, seen["gate"])
    finally:
        gate.ask = real_ask
    print("  ✓ chat lane and gate: matching entries above the question with the rule; none for small talk")

    import executor as EX
    from tool_registry import tool_registry
    tool_registry.auto_register_all()

    class _Http:
        def __init__(self): self.payloads = []
        async def post(self, url, json=None, **k):
            self.payloads.append(json)
            data = {"message": {"content": '{"result": "ok", "confidence": 0.9}'},
                    "prompt_eval_count": 10, "eval_count": 5}
            class _R:
                def raise_for_status(self): pass
                def json(self): return data
            return _R()
        async def aclose(self): pass

    class _Tr:
        async def get_task(self, task_id): return {"goal": "請問退貨期限是幾天？幫我寫成一段客服回覆"}
        async def record_usage(self, **k): pass
        async def append_task_artifact(self, *a, **k): pass

    config.streaming_enabled = False

    async def first(scope):
        K.begin(scope)
        ex = EX.Executor(tracker=_Tr(), skill_factory=None, registry=tool_registry)
        await ex._http.aclose()
        ex._http = _Http()
        await ex._qwen_execute("Write the customer-service reply", {"knowledge_datasets": scope},
                               task_id="t1", subtask_id="s1", mode="code")
        pl = ex._http.payloads[0]
        user = [m["content"] for m in pl["messages"] if m["role"] == "user"][0]
        names = {(t.get("function") or {}).get("name") for t in pl.get("tools") or []}
        return user, names
    user, names = await first([faq["id"]])
    assert "收到商品後 7 天內" in user and "knowledge_search" in names, (user[:400], sorted(names)[:5])
    assert user.endswith(K.REMINDER), "the reminder is the last thing the model reads"
    assert "knowledge_datasets" not in user, "internal context stays out of the prompt"
    user, names = await first([])
    assert "【知識庫" not in user and "knowledge_search" not in names and K.REMINDER not in user
    print("  ✓ executor: the entries for the person's words, and knowledge_search only with datasets")

    from tools.knowledge_tool import knowledge_search
    K.begin([faq["id"]])
    r = await knowledge_search("發票")
    assert r["success"] and "統一編號" in r["result"], r
    r = await knowledge_search("火星移民")
    assert r["success"] and "Nothing" in r["result"]
    r = await knowledge_search("發票", dataset="內部規定")
    assert not r["success"] and "客服 FAQ" in r["error"], r
    K.begin([])
    assert not (await knowledge_search("發票"))["success"]
    print("  ✓ knowledge_search: finds, says when nothing matches, stays inside the task's datasets")


# ── E ────────────────────────────────────────────────────────────────────────

async def test_api() -> None:
    import httpx
    from fastapi import FastAPI, HTTPException
    import main as M
    app = FastAPI()
    app.router.routes.extend(r for r in M.app.routes if getattr(r, "path", "").startswith("/knowledge")
                             or getattr(r, "path", "").endswith("/knowledge"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
        d = (await c.post("/knowledge/datasets", json={"name": "產品"})).json()
        assert d["id"].startswith("kb-") and d["always"] is False and d["items"] == 0, d
        r = await c.post(f"/knowledge/datasets/{d['id']}/files",
                         files={"file": ("products.csv", "品名,價格\n電風扇,990\n".encode("cp950"), "text/csv")})
        assert r.status_code == 200 and r.json()["added"] == 1, r.text
        r = await c.post(f"/knowledge/datasets/{d['id']}/files", files={"file": ("x.pdf", b"%PDF", "application/pdf")})
        assert r.status_code == 400 and ".csv" in r.json()["detail"]
        ov = (await c.get("/knowledge")).json()
        assert ov["enabled"] is True and {x["name"] for x in ov["datasets"]} >= {"產品", "客服 FAQ"}, ov
        s = (await c.post("/knowledge/search", json={"query": "電風扇多少錢", "datasets": ["產品"]})).json()
        assert s["matches"] and "990" in s["matches"][0]["body"], s
        assert (await c.post("/knowledge/search", json={"query": "x", "datasets": ["nope"]})).status_code == 400
        items = (await c.get(f"/knowledge/datasets/{d['id']}/items")).json()
        assert items["total"] == 1 and items["items"][0]["title"] == "電風扇"
        r = await c.patch(f"/knowledge/datasets/{d['id']}", json={"always": True, "description": "價格表"})
        assert r.json()["always"] is True and r.json()["description"] == "價格表"

        # A session picks its datasets.
        sid = "sess-kb-api"
        await M.tracker.create_session(sid, "KB API", mode="code")
        assert (await c.get("/sessions/nope/knowledge")).status_code == 404
        g = (await c.get(f"/sessions/{sid}/knowledge")).json()
        assert g["datasets"] == [] and [x["name"] for x in g["always"]] == ["產品"], g
        r = await c.put(f"/sessions/{sid}/knowledge", json={"datasets": ["客服 FAQ"]})
        assert r.status_code == 200 and [x["name"] for x in r.json()["datasets"]] == ["客服 FAQ"], r.text
        assert await K.scope_for({}, sid) == ((await K.get_dataset("客服 FAQ"))["id"], d["id"])
        assert (await c.put(f"/sessions/{sid}/knowledge", json={"datasets": ["沒有這個"]})).status_code == 400
        assert (await c.put(f"/sessions/{sid}/knowledge", json={"datasets": []})).json()["datasets"] == []
        await c.patch(f"/knowledge/datasets/{d['id']}", json={"always": False})
        assert (await c.delete(f"/knowledge/datasets/{d['id']}/files/products.csv")).json()["deleted"] == 1
        assert (await c.delete(f"/knowledge/datasets/{d['id']}")).json()["deleted"] == d["id"]
        assert (await c.delete(f"/knowledge/datasets/{d['id']}")).status_code == 404
    print("  ✓ /knowledge: create (no session's yet), upload (Big5 CSV), refuse other types, list, search, "
          "browse, edit, delete; /sessions/{id}/knowledge picks a session's datasets")

    from models import TaskCreate
    assert TaskCreate(intent="x", datasets=["客服 FAQ"]).datasets == ["客服 FAQ"] and TaskCreate(intent="x").datasets is None
    M._orchestrator_safety_check = lambda: None
    for on, ds, want in ((False, ["客服 FAQ"], "knowledge base is off"), (True, ["沒有這個"], "沒有這個")):
        config.knowledge_enabled = on
        try:
            await M._start_task(goal="退貨幾天", context={}, session_id=None, datasets=ds)
            raise AssertionError("accepted")
        except HTTPException as exc:
            assert exc.status_code == 400 and want in exc.detail, exc.detail
    config.knowledge_enabled = True
    print("  ✓ POST /tasks datasets: refused while off, unknown names refused before anything starts")


async def main() -> None:
    from tracker import Tracker
    tracker = Tracker(config.db_path)
    await tracker.open()
    await K.start(tracker)
    import main as M
    M.tracker = tracker
    config.knowledge_enabled = True
    config.knowledge_embed_model = ""
    try:
        print("A. files"); test_files()
        print("B. search"); await test_search()
        print("C. scope"); await test_scope()
        print("D. prompts"); await test_prompts()
        print("E. API"); await test_api()
    finally:
        await tracker.close()
    print("\nALL KNOWLEDGE TESTS PASS")


asyncio.run(main())
