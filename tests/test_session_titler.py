"""Sessions nobody named get a title from the model in quiet time."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)


def test_clean():
    import session_titler as T
    assert T.clean("聖誕節吃東西小遊戲") == "聖誕節吃東西小遊戲"
    assert T.clean("標題：「Pixel 11 價格比較」。") == "Pixel 11 價格比較"
    assert T.clean('Title: "Fix the login bug"\nmore') == "Fix the login bug"
    assert T.clean("**Docker 部署**") == "Docker 部署"
    assert T.clean("-") is None and T.clean("") is None and T.clean("  \n") is None
    assert len(T.clean("長" * 80)) == 40
    assert T.from_request("[Attachments in workspace — use these paths directly\n- a.csv\n]\n分析這份檔案\n第二行") == "分析這份檔案"


def test_idle():
    import idle
    idle.begin()
    assert idle.quiet_for() == 0.0
    idle.end()
    assert 0.0 <= idle.quiet_for() < 1.0
    idle._last_active = time.monotonic() - 100
    assert idle.quiet_for() >= 99


async def _flow():
    import session_titler as T
    from models import TaskRequest, TaskStatus
    from tracker import Tracker

    t = Tracker(db_path=tempfile.mktemp(suffix=".db"))
    await t.open()
    try:
        async def task(sid, goal, summary, status=TaskStatus.COMPLETED):
            req = TaskRequest(goal=goal, session_id=sid)
            await t.create_task(req)
            await t.update_task_status(req.id, status, summary)

        await t.create_session("new", "New session")
        await t.create_session("named", "我的專案")              # named at creation
        await t.create_session("renamed", "New session")
        await t.rename_session("renamed", "手動取的名字")          # the person renamed it
        await t.create_session("empty", "New session")           # no task yet
        await t.create_session("hi", "Default session")
        await task("new", "做一個聖誕節吃東西的小遊戲", "已產出 christmas_eater_game.html")
        await task("named", "整理報表", "完成")
        await task("renamed", "查天氣", "晴天")
        await task("hi", "hi", "你好！")

        ids = {s["id"] for s in await t.sessions_to_title()}
        assert ids == {"new", "hi"}, ids

        seen = []
        replies = {"做一個聖誕節": "聖誕節吃東西小遊戲", "hi": "-"}   # "-": nothing usable

        async def fake_model(prompt, local_only):
            seen.append(prompt)
            return next(v for k, v in replies.items() if k in prompt)

        sent = []

        async def bcast(d):
            sent.append(d)

        T._tracker, T._broadcast = t, bcast
        T._ask_model = fake_model

        assert await T.title_one() and await T.title_one()
        assert not await T.title_one()
        s = await t.get_session("new")
        assert s["title"] == "聖誕節吃東西小遊戲" and s["title_state"] == "auto"
        assert (await t.get_session("hi"))["title"] == "hi"      # its first words, once
        assert {"type": "session.renamed", "session_id": "new", "title": "聖誕節吃東西小遊戲"} in sent
        assert any(p.startswith("Person: 做一個聖誕節吃東西的小遊戲\nOpenTeddy: 已產出") for p in seen)

        # Named once: later turns change nothing.
        await task("new", "幫我改成新年主題", "好了")
        assert not await T.title_one()
        assert (await t.get_session("new"))["title"] == "聖誕節吃東西小遊戲"

        # Named ones are left alone for good; a rename mid-way wins.
        assert await t.sessions_to_title() == []
        await t.create_session("race", "New session")
        await task("race", "寫一首詩", "好的")
        await t.rename_session("race", "詩")
        assert not await t.set_auto_title("race", "寫詩")
        assert (await t.get_session("race"))["title"] == "詩"

        # A stand-in title (cut-off first message) is replaced.
        await t.create_session("agent", "幫我把上個月的銷售資料做成圖表然後")
        await t.rename_session("agent", "幫我把上個月的銷售資料做成圖表然後", by="pending")
        await task("agent", "幫我把上個月的銷售資料做成圖表然後寄給我", "已寄出")
        assert [s["id"] for s in await t.sessions_to_title()] == ["agent"]
    finally:
        await t.close()


def test_title_flow():
    asyncio.run(_flow())


if __name__ == "__main__":
    test_clean()
    test_idle()
    test_title_flow()
    print("ok")
