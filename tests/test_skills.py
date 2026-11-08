"""Skills (agent_skills/) and page_look: the agent can learn how, and can see."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
logging.disable(logging.WARNING)


def test_parse_and_load():
    import agent_skills as A
    s = A.parse("---\nname: x\ndescription: does y\nmetadata:\n  a: 1\n---\n# Guide\nstep", "/f")
    assert s.name == "x" and s.description == "does y" and s.body == "# Guide\nstep"
    assert A.parse("no front matter") is None
    assert A.parse("---\nname: x\n---\nbody") is None            # no description
    skills = A.load()
    assert {"think-before-building", "web-page-debugging"} <= set(skills)
    for s in skills.values():
        assert len(s.description) < 400 and s.body, s.name    # the index stays short
    idx = A.index_text()
    assert idx.count("\n") == len(skills) - 1 and "- web-page-debugging: " in idx
    g = A.guide_text(["web-page-debugging", "nope"])
    assert g.startswith("[Guide: web-page-debugging") and "skill" not in g.split("\n")[0].lower() and "page_look" in g and "nope" not in g


def test_parse_pick():
    import agent_skills as A
    assert A.parse_pick('{"skills": ["web-page-debugging", "made-up", "web-page-debugging"]}') == ["web-page-debugging"]
    assert A.parse_pick('{"skills": []}') == []
    assert A.parse_pick('sure: {"skills": ["think-before-building"]}') == ["think-before-building"]
    assert A.parse_pick("not json") == [] and A.parse_pick('{"skills": "x"}') == []


async def _read_guide():
    from tools.read_guide_tool import READ_GUIDE_TOOLS, read_guide
    desc = READ_GUIDE_TOOLS[0][1]["function"]["description"]
    assert "- think-before-building:" in desc and "- web-page-debugging:" in desc
    r = await read_guide("web-page-debugging")
    assert r["success"] and "page_look" in r["result"]["text"]
    r = await read_guide("nope")
    assert not r["success"] and "think-before-building" in r["error"]
    r = await read_guide("web-page-debugging", file="../think-before-building/SKILL.md")
    assert not r["success"]                                      # stays inside the skill


def test_read_guide():
    asyncio.run(_read_guide())


def test_page_look_targets():
    from tools.page_look_tool import target_url
    for ok in ("http://localhost:8080/", "http://127.0.0.1:3000/x", "http://10.255.147.17:8080/",
               "http://192.168.1.5/", "http://100.80.1.2/", "http://box.local/"):
        assert target_url(ok)[0] == ok, ok
    for no in ("https://www.momo.com.tw/", "http://8.8.8.8/", ""):
        assert target_url(no)[0] is None, no


async def _page_look_sees_cover():
    from config import config
    import tools.page_look_tool as P
    ws = tempfile.mkdtemp()
    shutil.copy(os.path.join(ROOT, "tests", "eval", "fixtures", "xmas_game.html"), ws)
    config.vision_model = ""                                    # no vision model in this test
    from config import set_session_workspace
    set_session_workspace(ws)
    r = await P.page_look("xmas_game.html", ["click 開始遊戲", "wait 300", "press ArrowUp", "jump"], "看得到主角嗎？")
    assert r["success"], r
    out = r["result"]
    assert any(c.startswith("canvas#game is covered by canvas#snowflakes") and "rgb(26, 26, 46)" in c
               for c in out["covered"]), out["covered"]
    assert out["actions_done"] == ["click 開始遊戲", "wait 300", "press ArrowUp"]
    assert any("'jump' failed" in e for e in out["errors"]), out["errors"]
    assert os.path.isfile(out["screenshot"]) and out["screenshot"].startswith(ws)
    assert out["seen"].startswith("(no vision model") and "answer" not in out
    # The emoji are drawn with the grid's 3% fill: seen as they're drawn.
    assert any(d.startswith('canvas#game: fillText("🎅") drawn at opacity 0.03 — nearly invisible')
               for d in out["canvas_drawing"]), out["canvas_drawing"]

    # Fixed the cover → nothing covered any more; the hidden start overlay
    # (opacity 0) doesn't count as covering anything.
    path = os.path.join(ws, "xmas_game.html")
    s = open(path, encoding="utf-8").read()
    s = s.replace("  canvas {\n    border: 3px", "  #game {\n    position: relative; z-index: 1;\n    border: 3px", 1)
    s = s.replace("pointer-events: none; z-index: 0; }", "pointer-events: none; z-index: -1; }", 1)
    open(path, "w", encoding="utf-8").write(s)
    r = await P.page_look("xmas_game.html", ["click #start-btn"])
    assert r["result"]["covered"] == "nothing is covered", r["result"]["covered"]
    assert r["result"]["errors"] == "none"
    assert any("fillText" in d for d in r["result"]["canvas_drawing"])     # still faint

    s = s.replace("  // Foods\n", "  ctx.fillStyle = '#fff';\n  // Foods\n", 1)
    open(path, "w", encoding="utf-8").write(s)
    r = await P.page_look("xmas_game.html", ["click #start-btn"])
    assert r["result"]["canvas_drawing"] == "nothing drawn invisibly", r["result"]["canvas_drawing"]

    r = await P.page_look("https://example.com/")
    assert not r["success"] and "browser_fetch" in r["error"]
    r = await P.page_look("missing.html")
    assert not r["success"] and "not found" in r["error"]


def test_page_look_sees_cover():
    asyncio.run(_page_look_sees_cover())


async def _pick_into_context():
    import types
    import agent_skills as A
    import orchestrator as O
    from models import TaskRequest
    events, calls = [], []

    async def ev(kind, req, **k):
        events.append((kind, k))

    async def fake_pick(orch, goal, task_id=""):
        calls.append(goal)
        return ["web-page-debugging"]

    real, A.pick = A.pick, fake_pick
    try:
        o = types.SimpleNamespace(_emit_task_event=ev)
        req = TaskRequest(goal="遊戲畫面一直飄雪看不到主角", session_id="s1")
        await O.Orchestrator._pick_skills(o, req)
        await O.Orchestrator._pick_skills(o, req)                 # a retry doesn't pick again
        assert calls == ["遊戲畫面一直飄雪看不到主角"]
        assert req.context["skills"] == ["web-page-debugging"] and "page_look" in req.context["skill_guides"]
        assert events == [("task.skills", {"skills": ["web-page-debugging"]})]

        async def broken(orch, goal, task_id=""):
            raise RuntimeError("model down")
        A.pick = broken
        req2 = TaskRequest(goal="x", session_id="s1")
        await O.Orchestrator._pick_skills(o, req2)                # planning goes on without skills
        assert "skill_guides" not in req2.context
    finally:
        A.pick = real


async def _edit_file():
    from config import set_session_workspace
    from tools.file_tool import edit_file
    ws = tempfile.mkdtemp()
    set_session_workspace(ws)
    path = os.path.join(ws, "a.html")
    open(path, "w").write("<style>\n  canvas { z-index: 0; }\n  #x { z-index: 0; }\n</style>\n")
    r = await edit_file("a.html", "  canvas { z-index: 0; }", "  #game { z-index: 1; }")
    assert r["success"] and r["result"]["replaced"] == 1 and "2:   #game { z-index: 1; }" in r["result"]["now_reads"], r
    assert open(path).read() == "<style>\n  #game { z-index: 1; }\n  #x { z-index: 0; }\n</style>\n"
    r = await edit_file("a.html", "z-index: ", "z-index:")              # two matches
    assert not r["success"] and "2 times" in r["error"]
    r = await edit_file("a.html", "z-index: ", "z-index:", replace_all=True)
    assert r["success"] and r["result"]["replaced"] == 2 and "z-index: " not in open(path).read()
    r = await edit_file("a.html", "not there", "x")
    assert not r["success"] and "not found" in r["error"]
    r = await edit_file("nope.html", "a", "b")
    assert not r["success"]


def test_edit_file():
    asyncio.run(_edit_file())


def test_pick_into_context():
    asyncio.run(_pick_into_context())


if __name__ == "__main__":
    test_parse_and_load()
    test_parse_pick()
    test_read_guide()
    test_page_look_targets()
    test_page_look_sees_cover()
    test_pick_into_context()
    test_edit_file()
    print("ALL SKILL TESTS PASS")
