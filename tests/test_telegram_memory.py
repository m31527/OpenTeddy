"""
Telegram: files are received, and each message sees the previous turns.

  A. Orchestrator._recent_turns_block: same-session turns, oldest first,
     current task and unfinished tasks excluded, attachment paths kept,
     skipped for scheduled runs and session-less tasks, length-capped.
  B. Bridge attachment helpers: document / photo metadata, download into
     <workspace>/uploads with collision suffix, 20 MB cap, getFile error,
     pending manifest (format, TTL, consumed once).
  C. Bridge _dispatch: file without caption → acknowledged and held;
     next text → goal carries the web-UI attachment manifest; file with
     caption → dispatched at once; sticker → nudge.

    .venv/bin/python tests/test_telegram_memory.py
"""
from __future__ import annotations

import asyncio, logging, os, sys, tempfile, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

from config import config
import telegram_bridge as TB


# ── A. recent turns ──────────────────────────────────────────────────────────

async def test_recent_turns() -> None:
    from orchestrator import Orchestrator
    rows = [  # newest first, as tracker.list_tasks returns them
        {"id": "cur", "status": "running", "goal": "現在這句", "summary": ""},
        {"id": "t3", "status": "running", "goal": "還在跑", "summary": ""},
        {"id": "t2", "status": "completed",
         "goal": "[Attachments in workspace — use these paths directly, do not re-upload:\n"
                 "  - uploads/sales.xlsx (11.7 KB, application/vnd.ms-excel)\n]\n\n幫我總結",
         "summary": "營收 120 萬，比上月 +8%"},
        {"id": "t1", "status": "failed", "goal": "今天星期幾", "summary": "星期六"},
    ]
    seen = {}

    async def list_tasks(limit=50, session_id=None):
        seen["limit"], seen["sid"] = limit, session_id
        return rows

    fake = types.SimpleNamespace(tracker=types.SimpleNamespace(list_tasks=list_tasks))
    req = lambda **k: types.SimpleNamespace(**{"id": "cur", "session_id": "s1", "context": {}, **k})
    block = await Orchestrator._recent_turns_block(fake, req())

    assert seen["sid"] == "s1"
    assert "現在這句" not in block and "還在跑" not in block, block
    assert block.index("今天星期幾") < block.index("幫我總結"), "oldest first"
    assert "[附件: uploads/sales.xlsx] 幫我總結" in block, block
    assert "Attachments in workspace" not in block
    assert "助理：營收 120 萬" in block and block.startswith("【最近的對話")
    print("  ✓ same session, oldest first, current/unfinished excluded, attachment path kept")

    assert await Orchestrator._recent_turns_block(fake, req(context={"triggered_by": "schedule"})) == ""
    assert await Orchestrator._recent_turns_block(fake, req(session_id=None)) == ""
    print("  ✓ skipped for scheduled runs and session-less tasks")

    rows[:] = [{"id": f"x{i}", "status": "completed", "goal": "問" * 300, "summary": "答" * 500}
               for i in range(6)]
    block = await Orchestrator._recent_turns_block(fake, req(), max_turns=4, max_chars=1000)
    assert len(block) < 1100 and "…" in block
    rows[:] = []
    assert await Orchestrator._recent_turns_block(fake, req()) == ""

    async def boom(**k): raise RuntimeError("db gone")
    fake.tracker.list_tasks = boom
    assert await Orchestrator._recent_turns_block(fake, req()) == ""
    print("  ✓ capped to max_chars; empty history or tracker error → no block")


# ── B. attachment helpers ────────────────────────────────────────────────────

class _Resp:
    def __init__(self, status=200, payload=None, content=b""):
        self.status_code, self._p, self.content = status, payload, content
    def json(self): return self._p


class _Client:
    getfile = None           # set per test
    download = None
    urls: list = []
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def get(self, url, params=None):
        _Client.urls.append(url)
        return _Client.getfile if url.endswith("/getFile") else _Client.download


def _install_fakes(workspace: str) -> None:
    config.telegram_bot_token = "123:SECRET"
    config.agent_workspace_dir = workspace
    TB.httpx = types.SimpleNamespace(AsyncClient=_Client)  # type: ignore[assignment]

    async def resolve(chat_id, first_goal): return "sess-1"
    async def get_session(sid): return {"id": sid, "workspace_dir": None}
    TB._resolve_or_create_session = resolve  # type: ignore[assignment]
    TB._tracker = types.SimpleNamespace(get_session=get_session)


async def test_attachments(workspace: str) -> None:
    _install_fakes(workspace)

    doc = {"document": {"file_id": "F1", "file_name": "sales.xlsx", "file_size": 2048,
                        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}}
    photo = {"photo": [{"file_id": "small", "file_unique_id": "u1", "file_size": 10},
                       {"file_id": "big", "file_unique_id": "u1", "file_size": 90000}]}
    a = TB._attachment_meta(doc)
    assert a["file_id"] == "F1" and a["name"] == "sales.xlsx" and a["size"] == 2048
    p = TB._attachment_meta(photo)
    assert p["file_id"] == "big" and p["name"] == "photo_u1.jpg" and p["mime"] == "image/jpeg"
    assert TB._attachment_meta({"sticker": {"file_id": "s"}}) is None
    assert TB._attachment_meta({"text": "hi"}) is None
    print("  ✓ document / largest photo recognised; stickers and text are not attachments")

    _Client.getfile = _Resp(200, {"ok": True, "result": {"file_path": "documents/file_7.xlsx"}})
    _Client.download = _Resp(200, content=b"PK\x03\x04xlsx-bytes")
    info = await TB._receive_attachment("42", a)
    dest = os.path.join(workspace, "uploads", "sales.xlsx")
    assert info["rel_path"] == os.path.join("uploads", "sales.xlsx"), info
    assert open(dest, "rb").read() == b"PK\x03\x04xlsx-bytes"
    assert _Client.urls[-1] == "https://api.telegram.org/file/bot123:SECRET/documents/file_7.xlsx"
    print("  ✓ downloaded via getFile into <workspace>/uploads/")

    info2 = await TB._receive_attachment("42", a)
    assert info2["name"] != "sales.xlsx" and info2["name"].startswith("sales_") and info2["name"].endswith(".xlsx")
    info3 = await TB._receive_attachment("42", a)
    assert len({info["name"], info2["name"], info3["name"]}) == 3, (info2, info3)
    assert open(dest, "rb").read() == b"PK\x03\x04xlsx-bytes"
    print("  ✓ same name again (even within a second) → new suffix, earlier files untouched")

    n = len(_Client.urls)
    try:
        await TB._receive_attachment("42", {**a, "size": 25 * 1024 * 1024}); raise AssertionError
    except ValueError as exc:
        assert "20 MB" in str(exc)
    assert len(_Client.urls) == n, "over-size file must not hit the network"
    _Client.getfile = _Resp(400, {"ok": False, "description": "Bad Request: file is too big"})
    try:
        await TB._receive_attachment("42", a); raise AssertionError
    except ValueError as exc:
        assert "too big" in str(exc)
    _Client.getfile = _Resp(200, {"ok": True, "result": {"file_path": "x"}})
    _Client.download = _Resp(404)
    try:
        await TB._receive_attachment("42", a); raise AssertionError
    except ValueError as exc:
        assert "404" in str(exc) and "SECRET" not in str(exc)
    print("  ✓ >20 MB refused locally; getFile / download errors surface without the token")

    TB._pending_attachments.clear()
    assert TB._with_pending_attachments("42", "hi") == "hi"
    TB._pending_attachments["42"] = [
        {"rel_path": "uploads/sales.xlsx", "name": "sales.xlsx", "size_bytes": 12000,
         "content_type": "application/x", "ts": time.monotonic()},
        {"rel_path": "uploads/old.csv", "name": "old.csv", "size_bytes": 10,
         "content_type": "", "ts": time.monotonic() - 3600},          # expired
    ]
    goal = TB._with_pending_attachments("42", "幫我總結")
    assert goal == ("[Attachments in workspace — use these paths directly, do not re-upload:\n"
                    "  - uploads/sales.xlsx (11.7 KB, application/x)\n]\n\n幫我總結"), goal
    assert TB._with_pending_attachments("42", "again") == "again", "consumed once"
    print("  ✓ manifest matches the web UI format; expired items dropped; consumed once")


# ── C. dispatch flow ─────────────────────────────────────────────────────────

async def test_dispatch(workspace: str) -> None:
    _install_fakes(workspace)
    _Client.getfile = _Resp(200, {"ok": True, "result": {"file_path": "documents/f.xlsx"}})
    _Client.download = _Resp(200, content=b"data")
    replies, goals = [], []

    async def send(chat_id, text, **k): replies.append(text)
    async def run_goal(chat_id, goal): goals.append(goal)
    TB._send_reply = send  # type: ignore[assignment]
    TB._run_goal_for_chat = run_goal  # type: ignore[assignment]
    TB._whitelisted_chat_ids = lambda: {"42"}  # type: ignore[assignment]
    async def no_cancel(chat_id, text): return False
    TB._maybe_handle_natural_cancel = no_cancel  # type: ignore[assignment]
    TB._pending_attachments.clear(); TB._running_chats.clear()

    async def dispatch(msg):
        await TB._dispatch({"message": {"chat": {"id": 42}, **msg}})
        t = TB._running_chats.pop("42", None)
        if t: await t

    await dispatch({"document": {"file_id": "F", "file_name": "q3.xlsx", "file_size": 4}})
    assert not goals and "已收到 q3.xlsx" in replies[-1], replies
    await dispatch({"text": "幫我把這幾個資訊快速總結一下"})
    assert goals[-1].startswith("[Attachments in workspace") and "uploads/q3.xlsx" in goals[-1]
    assert goals[-1].endswith("幫我把這幾個資訊快速總結一下")
    print("  ✓ file alone → acknowledged; next message carries it as an attachment")

    await dispatch({"text": "謝謝"})
    assert goals[-1] == "謝謝"
    await dispatch({"document": {"file_id": "F", "file_name": "r.csv", "file_size": 4},
                    "caption": "總結這份"})
    assert "uploads/r.csv" in goals[-1] and goals[-1].endswith("總結這份")
    print("  ✓ attachment used once; file with caption dispatched immediately")

    n = len(goals)
    await dispatch({"sticker": {"file_id": "s"}})
    assert len(goals) == n and "貼圖" in replies[-1]
    await dispatch({"chat": {"id": 7}, "text": "hi"})          # not whitelisted
    assert len(goals) == n
    print("  ✓ stickers get a nudge; non-whitelisted chats still ignored")


async def main() -> None:
    with tempfile.TemporaryDirectory() as ws:
        print("A. recent turns"); await test_recent_turns()
        print("B. attachment helpers"); await test_attachments(ws)
    with tempfile.TemporaryDirectory() as ws:
        print("C. dispatch"); await test_dispatch(ws)
    print("\nALL TELEGRAM MEMORY TESTS PASS")


asyncio.run(main())
