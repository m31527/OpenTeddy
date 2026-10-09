"""
Telegram presentation: readable answers and an animated status bubble.

  A. _render_md: Markdown → Telegram HTML (headings, bold, tables, lists,
     links, quotes, code), HTML-escaped, tags always well nested; plain
     mode strips every marker.
  B. _rich_chunks / _send_rich: long replies split with balanced code
     fences, each chunk fits; HTML refused → same chunk as plain text.
  C. Status bubble: nothing for fast answers; after the delay it appears,
     animates, shows "執行中 2/3" from progress events, is deleted before
     the answer; backs off on 429.
  D. _format_result_for_telegram: answer first, quiet footer, no session
     id; failures keep their status on top.

    .venv/bin/python tests/test_telegram_format.py
"""
from __future__ import annotations

import asyncio, logging, os, sys, types
from html.parser import HTMLParser
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import telegram_bridge as TB

_ALLOWED = {"b", "i", "s", "code", "pre", "a", "blockquote"}


class _Nesting(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.errors = [], []
    def handle_starttag(self, tag, attrs):
        if tag not in _ALLOWED:
            self.errors.append(f"tag <{tag}> not supported by Telegram")
        self.stack.append(tag)
    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes {self.stack[-1:]}")
        else:
            self.stack.pop()


def well_formed(html: str) -> None:
    p = _Nesting(); p.feed(html); p.close()
    assert not p.errors and not p.stack, (p.errors, p.stack, html)


# ── A. rendering ─────────────────────────────────────────────────────────────

def test_render() -> None:
    r = TB._render_md
    md = ("必須了解**數據使用規範（Data Use Policy）**和**授權（Licensing）**。\n\n"
          "### 1. 授權（Licensing）的基礎\n\n## **粗體標題**")
    h = r(md)
    assert "**" not in h and "#" not in h, h
    assert "<b>數據使用規範（Data Use Policy）</b>和<b>授權（Licensing）</b>" in h
    assert "<b>1. 授權（Licensing）的基礎</b>" in h and "<b>粗體標題</b>" in h
    well_formed(h)
    print("  ✓ the screenshot's ### and ** become bold lines and bold text")

    h = r("a < b & c > d")
    assert h == "a &lt; b &amp; c &gt; d", h
    h = r("```python\nprint('**x** <y>')\n```\n用 `**raw**` 與 `a<b`")
    assert "<pre>print('**x** &lt;y&gt;')</pre>" in h, h
    assert "<code>**raw**</code>" in h and "<code>a&lt;b</code>" in h
    well_formed(h)
    print("  ✓ HTML escaped; code blocks and inline code kept literal")

    h = r("| 授權 | 可否商用 |\n|---|:---:|\n| CC0 | 可以 |\n| **CC BY-NC** | 不可以 |")
    assert h == "• <b>CC0</b>：可以\n• <b>CC BY-NC</b>：不可以", h
    h = r("| Plan | Price | Seats |\n|---|---|---|\n| Pro | $20 | 5 |")
    assert h == "• <b>Pro</b>\n   Price: $20\n   Seats: 5", h
    print("  ✓ tables become bullets (2 columns inline, more columns as labelled lines)")

    h = r("- 第一\n  - 子項\n* 第二\n- [ ] 待辦\n- [x] 完成\n1. 編號保留\n---\n*斜體* 與 ~~刪除~~ 與 2*3*4")
    assert "• 第一\n   ◦ 子項\n• 第二\n☐ 待辦\n☑ 完成\n1. 編號保留" in h, h
    assert "<i>斜體</i>" in h and "<s>刪除</s>" in h and "2*3*4" in h
    assert "---" not in h
    well_formed(h)
    print("  ✓ bullets, nesting, task lists, italics, strikethrough; 2*3*4 left alone")

    h = r("看 [GBIF 指南](https://gbif.org/x?a=1&b=2) 和 **[粗體連結](https://a.b/c)**\n> 引用一\n> 引用二\n正文")
    assert '<a href="https://gbif.org/x?a=1&amp;b=2">GBIF 指南</a>' in h, h
    assert '<b><a href="https://a.b/c">粗體連結</a></b>' in h
    assert "<blockquote>引用一\n引用二</blockquote>\n正文" in h
    well_formed(h)
    print("  ✓ links (escaped href), bold links, blockquotes")

    h = r("未配對的 **粗體 和 #標籤 與 snake_case_name 及 __init__.py")
    assert "**" not in h and "#標籤" in h and "snake_case_name" in h and "__init__.py" in h, h
    print("  ✓ stray ** dropped; hashtags, snake_case and __dunder__ untouched")

    p = r(md + "\n\n| a | b |\n|---|---|\n| x | y |\n[t](https://u.v)\n`c`", html=False)
    for marker in ("**", "###", "<b>", "|", "`", "]("):
        assert marker not in p, (marker, p)
    assert "t (https://u.v)" in p
    print("  ✓ plain fallback has no markers or tags")


# ── B. chunking and sending ──────────────────────────────────────────────────

class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code, self._p, self.text = status, payload or {"ok": True, "result": {"message_id": 99}}, ""
    def json(self): return self._p


class _Client:
    posts: list = []
    handler = None
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def post(self, url, json=None, **k):
        _Client.posts.append((url.rsplit("/", 1)[-1], json))
        return _Client.handler(url, json) if _Client.handler else _Resp()


async def test_chunks_and_send() -> None:
    para = "這是一段很長的說明文字，**重點**在這裡。" * 12
    md = "\n\n".join([para] * 12) + "\n\n```\n" + "\n".join(f"line {i} <x>" for i in range(400)) + "\n```\n結尾"
    chunks = TB._rich_chunks(md)
    assert len(chunks) > 2
    for c in chunks:
        assert len([l for l in c.splitlines() if l.lstrip().startswith("```")]) % 2 == 0, c[:80]
        assert len(TB._render_md(c, html=False)) <= TB._TELEGRAM_CHUNK_CHARS
        well_formed(TB._render_md(c))
    table = "| 名稱 | 欄位一 | 欄位二 | 欄位三 |\n|---|---|---|---|\n" + \
        "\n".join(f"| 項目{i} | 很長的內容{i} | 更長的內容{i} | 再長一點{i} |" for i in range(160))
    for c in TB._rich_chunks(table):
        assert len(TB._render_md(c, html=False)) <= TB._TELEGRAM_CHUNK_CHARS
    print("  ✓ long replies split, code fences balanced per chunk, tables re-split to fit")

    TB.httpx = types.SimpleNamespace(AsyncClient=_Client)  # type: ignore[assignment]
    _Client.posts, _Client.handler = [], None
    await TB._send_rich("42", "### 標題\n**粗體**")
    (method, body), = _Client.posts
    assert method == "sendMessage" and body["parse_mode"] == "HTML"
    assert body["text"] == "<b>標題</b>\n<b>粗體</b>" and body["link_preview_options"] == {"is_disabled": True}

    _Client.posts = []
    _Client.handler = lambda url, j: _Resp(400, {"ok": False}) if j.get("parse_mode") else _Resp()
    await TB._send_rich("42", "### 標題\n**粗體**")
    assert [b.get("parse_mode") for _, b in _Client.posts] == ["HTML", None]
    assert _Client.posts[1][1]["text"] == "標題\n粗體"
    _Client.handler = None
    print("  ✓ sent as HTML without link previews; refused HTML → same text, plain")


# ── C. status bubble ─────────────────────────────────────────────────────────

async def test_status_bubble() -> None:
    TB.httpx = types.SimpleNamespace(AsyncClient=_Client)  # type: ignore[assignment]
    TB._STATUS_DELAY_S, TB._STATUS_INTERVAL_S = 0.05, 0.03

    _Client.posts = []
    st = TB._ProgressState("42", "t1", zh=True)
    anim = asyncio.create_task(TB._animate_status(st))
    await asyncio.sleep(0.01)
    await TB._stop_status(st, anim, None)
    assert _Client.posts == [], _Client.posts
    print("  ✓ fast answer: no bubble at all")

    _Client.posts = []
    st = TB._ProgressState("42", "t2", zh=True)
    anim = asyncio.create_task(TB._animate_status(st))
    await asyncio.sleep(0.12)
    listener = TB._make_progress_listener(st)
    await listener({"task_id": "other", "type": "subtask.progress", "order": 9, "total": 9})
    await listener({"task_id": "t2", "type": "tool_call"})
    await listener({"task_id": "t2", "type": "subtask.progress", "order": 2, "total": 3})
    await asyncio.sleep(0.08)
    await TB._stop_status(st, anim, None)
    methods = [m for m, _ in _Client.posts]
    texts = [b.get("text", "") for _, b in _Client.posts]
    assert methods[0] == "sendMessage" and texts[0].startswith("🐻 思考中 ●○○"), texts
    assert methods.count("sendMessage") == 1 and "editMessageText" in methods
    assert len({t for t in texts if t}) > 1, "frames change"
    assert any("執行中 2/3" in t for t in texts) and not any("9/9" in t for t in texts)
    assert methods[-1] == "deleteMessage" and _Client.posts[-1][1]["message_id"] == 99
    assert anim.done() and st.message_id is None
    print("  ✓ bubble appears, animates, shows 執行中 2/3, deleted before the answer")

    st = TB._ProgressState("42", "t3", zh=False)
    st.order, st.total = 1, 1
    assert TB._status_text(st, 1) == "🐻 Thinking ○●○"
    st.started -= 75
    assert TB._status_text(st, 0).endswith("1:15")
    print("  ✓ English goal → English bubble; elapsed time shown on long runs")

    _Client.posts = []
    _Client.handler = lambda url, j: _Resp(429, {"ok": False, "parameters": {"retry_after": 0.2}}) \
        if url.endswith("editMessageText") else _Resp()
    st = TB._ProgressState("42", "t4", zh=True)
    anim = asyncio.create_task(TB._animate_status(st))
    await asyncio.sleep(0.15)
    await TB._stop_status(st, anim, None)
    edits = [m for m, _ in _Client.posts if m == "editMessageText"]
    assert len(edits) == 1, f"must wait retry_after, got {len(edits)} edits"
    _Client.handler = None
    print("  ✓ 429 → waits retry_after before the next frame")


# ── D. result footer ─────────────────────────────────────────────────────────

def test_result_format() -> None:
    R = lambda status, summary, n=1: types.SimpleNamespace(
        status=types.SimpleNamespace(value=status), summary=summary, subtasks=[0] * n)
    out = TB._format_result_for_telegram(R("completed", "GBIF 的授權**主要有三種**。"), "sess-uuid", 11.73)
    assert out == "GBIF 的授權**主要有三種**。\n\n*✅ 完成 · 11.7 秒*", out
    assert "sess-uuid" not in out
    assert TB._render_md(out).endswith("<i>✅ 完成 · 11.7 秒</i>")
    out = TB._format_result_for_telegram(R("completed", "Three licences.", 3), "s", 4.0)
    assert out.endswith("*✅ Done · 4.0s · 3 steps*"), out
    out = TB._format_result_for_telegram(R("failed", "找不到檔案"), "s", 2.0)
    assert out.startswith("❌ 沒有完成 · 2.0 秒\n\n找不到檔案"), out
    print("  ✓ answer first + quiet footer, no session id; failures lead with status")


async def main() -> None:
    print("A. rendering"); test_render()
    print("B. chunks + send"); await test_chunks_and_send()
    print("C. status bubble"); await test_status_bubble()
    print("D. result format"); test_result_format()
    print("\nALL TELEGRAM FORMAT TESTS PASS")


asyncio.run(main())
