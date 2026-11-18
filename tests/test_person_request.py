"""
The executor sees the person's own words, and git stays in the workspace.

In the 11/18 eval the planner turned "run python count_words.py <file
name>" into a step to count the words in every file in the folder, and
the script never read the file it was given. The same day a step ran
`git add -f buggy.py && git commit` in its eval folder, and the commit
landed on OpenTeddy's own main.

  A. the step's prompt carries the person's request when the step's
     wording isn't already it; long requests are cut, and a tracker
     without the task changes nothing
  B. git run by shell_exec or python_exec in a workspace folder that
     isn't a repository finds no repository above the workspace; a
     repository inside the workspace still works

    .venv/bin/python tests/test_person_request.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from config import config
import executor as EX

GOAL = "寫一個 count_words.py：執行 python count_words.py 檔名，印出前 3 個英文單字和次數。"


class _Reg:
    def get_schemas(self): return []
    def get_schemas_by_names(self, n): return []
    def risk_of(self, n): return "medium"
    async def execute(self, name, args, **k): raise AssertionError("no tool expected")


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
    def __init__(self, tasks): self.tasks = tasks
    async def get_task(self, task_id): return self.tasks.get(task_id)
    async def record_usage(self, **k): pass
    async def append_task_artifact(self, *a, **k): pass


async def _first_prompt(tracker, description: str, task_id: str = "t1") -> str:
    ex = EX.Executor(tracker=tracker, skill_factory=None, registry=_Reg())
    await ex._http.aclose()
    ex._http = _Http()
    await ex._qwen_execute(description, {}, task_id=task_id, subtask_id="s1", mode="code")
    return [m["content"] for m in ex._http.payloads[0]["messages"] if m["role"] == "user"][0]


async def test_request_in_prompt() -> None:
    config.streaming_enabled = False
    step = "Write count_words.py that counts the words in every file in the folder"
    prompt = await _first_prompt(_Tr({"t1": {"goal": GOAL}}), step)
    assert "THE PERSON'S REQUEST" in prompt and GOAL in prompt, prompt
    assert prompt.index(GOAL) < prompt.index("Task: " + step), prompt
    print("  ✓ a rewritten step comes with the person's own words above it")

    prompt = await _first_prompt(_Tr({"t1": {"goal": GOAL}}), "Do this: " + GOAL)
    assert "THE PERSON'S REQUEST" not in prompt, prompt
    print("  ✓ a step that already carries the words doesn't repeat them")

    long_goal = "整理這份資料：" + "x" * 3000
    prompt = await _first_prompt(_Tr({"t1": {"goal": long_goal}}), step)
    assert long_goal[:1500] + "…" in prompt and long_goal not in prompt
    print("  ✓ a long request is cut at 1,500 characters")

    for tracker in (_Tr({}), object(), _Tr({"t1": "not a dict"})):
        prompt = await _first_prompt(tracker, step)
        assert "THE PERSON'S REQUEST" not in prompt and "Task: " + step in prompt, prompt
    print("  ✓ no task on record (or no tracker) leaves the prompt as before")


def _git_toplevel(cwd: str, env: dict) -> str:
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd, env=env,
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


async def test_git_confined() -> None:
    from tools.shell_tool import confine_git, execute_shell
    from tools.analytic_tool import python_exec
    root = os.path.realpath(tempfile.mkdtemp())
    saved = config.agent_workspace_dir
    try:
        repo = os.path.join(root, "app")                 # stands in for OpenTeddy's checkout
        ws = os.path.join(repo, "agent-workspace")
        task = os.path.join(ws, "eval", "multi_fix_and_log-r1")
        os.makedirs(task)
        subprocess.run(["git", "init", "-q", repo], check=True)
        config.agent_workspace_dir = ws

        assert _git_toplevel(task, dict(os.environ)) == repo   # the hole: git climbs out
        env = confine_git({"GIT_CEILING_DIRECTORIES": "/somewhere"})
        assert env["GIT_CEILING_DIRECTORIES"].split(os.pathsep)[0] == "/somewhere"
        assert repo in env["GIT_CEILING_DIRECTORIES"].split(os.pathsep)
        assert _git_toplevel(task, confine_git(dict(os.environ))) == ""
        print("  ✓ git in a workspace folder finds no repository above the workspace")

        out = await execute_shell("git rev-parse --show-toplevel || echo NO_REPO", working_dir=task)
        text = str(out.get("result"))
        assert "NO_REPO" in text and repo not in text.replace(task, ""), out
        out = await python_exec(
            "import subprocess\n"
            "r = subprocess.run(['git', 'rev-parse', '--show-toplevel'], capture_output=True, text=True)\n"
            "print('RC', r.returncode)", working_dir=task)
        assert "RC 128" in str(out.get("result")), out
        print("  ✓ shell_exec and python_exec run git with that limit")

        mine = os.path.join(ws, "my-project")
        os.makedirs(os.path.join(mine, "src"))
        subprocess.run(["git", "init", "-q", mine], check=True)
        assert _git_toplevel(os.path.join(mine, "src"), confine_git(dict(os.environ))) == mine
        print("  ✓ a repository inside the workspace still works")
    finally:
        config.agent_workspace_dir = saved
        shutil.rmtree(root, ignore_errors=True)


async def main() -> None:
    print("A. the person's words"); await test_request_in_prompt()
    print("B. git stays in the workspace"); await test_git_confined()
    print("\nALL PERSON REQUEST TESTS PASS")


asyncio.run(main())
