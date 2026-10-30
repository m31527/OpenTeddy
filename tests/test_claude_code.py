"""
OpenTeddy → Claude Code delegation, with a fake `claude` binary.

  A. parse: "用 Claude Code 在 ~/x 修好測試", status questions, not-requests
  B. resolve_directory: path / session folder / default; missing; OpenTeddy's own source refused
  C. run: real argv (allowlist, disallowed push, acceptEdits, json), the
     changed files from git, failure, timeout
  D. background job: notify + a task row in the session; job list
  E. orchestrator: direct path starts the job and replies at once; status

    .venv/bin/python tests/test_claude_code.py
"""
from __future__ import annotations

import asyncio, json, logging, os, stat, subprocess, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from config import config
import claude_code as CC

TMP = tempfile.mkdtemp(prefix="ot-cc-")
REPO = os.path.join(TMP, "shop")
ARGS = os.path.join(TMP, "args.json")
FAKE = os.path.join(TMP, "claude")

open(FAKE, "w").write(f"""#!/usr/bin/env python3
import json, os, sys, time
json.dump(sys.argv[1:], open({ARGS!r}, "w"))
if os.environ.get("FAKE_SLEEP"):
    time.sleep(float(os.environ["FAKE_SLEEP"]))
if os.environ.get("FAKE_FAIL"):
    print(json.dumps({{"type": "result", "is_error": True, "result": "API Error: overloaded"}})); sys.exit(1)
open("app.py", "a").write("# fixed\\n")
open("tests_new.py", "w").write("def test_ok(): pass\\n")
print(json.dumps({{"type": "result", "is_error": False, "result": "已修好 test_api：補上空值檢查，pytest 全部通過。",
                  "num_turns": 7, "total_cost_usd": 0.12}}))
""")
os.chmod(FAKE, os.stat(FAKE).st_mode | stat.S_IEXEC)
os.makedirs(REPO)
subprocess.run("git init -q && git config user.email t@t && git config user.name t && echo x > app.py && "
               "echo y > notes.md && git add -A && git commit -qm init && echo dirty >> notes.md",
               shell=True, cwd=REPO, check=True)
config.claude_code_bin = FAKE
config.claude_code_default_dir = ""
config.claude_code_allowed_tools = ""
config.claude_code_timeout = 1800


def test_parse() -> None:
    assert CC.parse_request("用 Claude Code 在 ~/projects/shop 修好失敗的測試") == ("修好失敗的測試", "~/projects/shop")
    assert CC.parse_request("請 claude code 幫我：把 README 翻成英文") == ("把 README 翻成英文", None)
    assert CC.parse_request("交給 Claude Code 處理 /srv/app 的 lint 錯誤") == ("的 lint 錯誤", "/srv/app")
    assert CC.parse_request("Claude Code 幫我 review 最近的改動") == ("review 最近的改動", None)
    for t in ("Claude Code 做到哪了", "claude code 的進度", "Claude Code 是什麼", "幫我修好測試", ""):
        assert CC.parse_request(t) is None, t
    assert CC.parse_status("Claude Code 做到哪了") and CC.parse_status("claude code 好了沒")
    print("  ✓ requests with / without a folder; status questions; unrelated messages ignored")


def test_resolve() -> None:
    assert CC.resolve_directory(REPO) == os.path.realpath(REPO)
    assert CC.resolve_directory(None, REPO) == os.path.realpath(REPO)
    for bad, msg in ((None, "哪個專案資料夾"), ("/no/such/dir", "找不到資料夾"),
                     (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "OpenTeddy")):
        try:
            CC.resolve_directory(bad); raise AssertionError(bad)
        except CC.ClaudeCodeError as exc:
            assert msg in str(exc), (bad, exc)
    config.claude_code_default_dir = REPO
    assert CC.resolve_directory(None) == os.path.realpath(REPO)
    config.claude_code_default_dir = ""
    print("  ✓ folder from the message, the session, or Settings; missing and OpenTeddy's own source refused")


async def test_run() -> None:
    res = await CC.run("修好失敗的測試", os.path.realpath(REPO))
    argv = json.load(open(ARGS))
    assert argv[:2] == ["-p", "修好失敗的測試"], argv
    i = argv.index("--allowedTools")
    assert "Edit" in argv and "Bash(pytest:*)" in argv and "Bash(git diff:*)" in argv
    j = argv.index("--disallowedTools")
    assert "Bash(git push:*)" in argv[j:] and "Bash(rm -rf:*)" in argv[j:]
    assert argv[argv.index("--permission-mode") + 1] == "acceptEdits" and argv[-2:] == ["--output-format", "json"]
    assert "Don't commit or push" in argv[argv.index("--append-system-prompt") + 1]
    assert res["ok"] and res["result"].startswith("已修好 test_api") and res["turns"] == 7
    assert res["changed"] == [("??", "tests_new.py"), ("M", "app.py")] or set(res["changed"]) == {("??", "tests_new.py"), ("M", "app.py")}, res["changed"]
    assert ("M", "notes.md") not in res["changed"], "already-dirty files aren't attributed to Claude Code"
    text = CC.describe("修好失敗的測試", REPO, res)
    assert text.startswith("🤖 Claude Code 完成「修好失敗的測試」") and "• M app.py" in text and "尚未 commit" in text
    print("  ✓ argv: allowlist, no push / rm -rf, acceptEdits, json; result + only the files it changed")

    os.environ["FAKE_FAIL"] = "1"
    res = await CC.run("x", REPO)
    os.environ.pop("FAKE_FAIL")
    assert not res["ok"] and "overloaded" in res["error"] and CC.describe("x", REPO, res).startswith("❌")
    os.environ["FAKE_SLEEP"] = "3"
    res = await CC.run("x", REPO, timeout_s=0.5)
    os.environ.pop("FAKE_SLEEP")
    assert not res["ok"] and "已停止" in res["error"]
    config.claude_code_bin = "/nope"
    real_which = CC.shutil.which
    CC.shutil.which = lambda *_: None
    saved = CC.os.path.expanduser
    CC.os.path.expanduser = lambda p: "/nope" if p.startswith("~/") else saved(p)
    try:
        assert CC.find_binary() is None
        try:
            await CC.run("x", REPO); raise AssertionError
        except CC.ClaudeCodeError as exc:
            assert "沒有安裝" in str(exc)
    finally:
        CC.shutil.which, CC.os.path.expanduser, config.claude_code_bin = real_which, saved, FAKE
    print("  ✓ failure and timeout reported; no binary explained")


async def test_jobs() -> None:
    sent, done = [], []
    async def notify(t): sent.append(t)
    async def on_done(t, res): done.append(res["ok"])
    jid = CC.start("修好失敗的測試", REPO, notify, on_done)
    assert CC.jobs()[0]["status"] == "running" and "進行中" in CC.describe_jobs()
    await CC._jobs[jid]["task_obj"]
    assert sent and sent[0].startswith("🤖 Claude Code 完成") and done == [True]
    assert CC.jobs()[0]["status"] == "done" and "已完成" in CC.describe_jobs()
    print("  ✓ background job: running → done, message sent, session record written")


async def test_orchestrator() -> None:
    from orchestrator import Orchestrator
    from models import TaskRequest, TaskStatus
    rows = []

    class _Tracker:
        async def get_session(self, sid): return {"id": sid, "mode": "code", "workspace_dir": None}
        async def create_task(self, req): rows.append(("create", req.goal))
        async def update_task_status(self, tid, status, summary=None): rows.append(("status", status, summary))
        async def create_subtask(self, st): pass
        async def update_subtask(self, st): pass
        async def list_tasks(self, **k): return []

    async def no_event(*a, **k): pass
    async def planner(*a, **k): raise RuntimeError("planner reached")
    o = types.SimpleNamespace(tracker=_Tracker(), _emit_task_event=no_event, _plan=planner,
                              _classify_intent=planner, _fast_chat_response=planner)
    for n in ("_finish_direct", "_claude_code", "_asked_again"):
        setattr(o, n, (lambda name: lambda *a, **k: getattr(Orchestrator, name)(o, *a, **k))(n))
    config.telegram_bot_token, config.telegram_default_chat_id = "", ""
    res = await Orchestrator.run(o, TaskRequest(goal=f"用 Claude Code 在 {REPO} 修好失敗的測試", session_id="s1"))
    assert res.status == TaskStatus.COMPLETED and res.summary.startswith("🤖 已交給 Claude Code（工作 `"), res
    jid = res.summary.split("`")[1]
    await CC._jobs[jid]["task_obj"]
    assert ("create", "（Claude Code 完成）修好失敗的測試") in rows, rows
    res = await Orchestrator.run(o, TaskRequest(goal="Claude Code 做到哪了", session_id="s1"))
    assert res.summary.startswith("🤖 Claude Code 工作") and jid in res.summary
    res = await Orchestrator.run(o, TaskRequest(goal="用 Claude Code 修好測試", session_id="s1"))
    assert res.status == TaskStatus.FAILED and "哪個專案資料夾" in res.summary
    print("  ✓ orchestrator: starts the job and answers at once; result recorded in the session; status; asks for a folder")


async def main() -> None:
    print("A. parse"); test_parse()
    print("B. folders"); test_resolve()
    print("C. run"); await test_run()
    print("D. jobs"); await test_jobs()
    print("E. orchestrator"); await test_orchestrator()
    print("\nALL CLAUDE CODE TESTS PASS")


asyncio.run(main())
