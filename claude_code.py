"""
Hand coding work to Claude Code — OpenTeddy is the butler, Claude Code the
engineer.

"用 Claude Code 在 ~/projects/shop 修好失敗的測試" (Telegram, chat, a
schedule) starts `claude -p` on the OpenTeddy machine in that folder, in the
background, and messages back when it finishes: Claude Code's own summary,
the files it changed, the time it took. Nothing waits on it — a coding task
can take longer than Telegram's 10-minute turn.

Boundaries (Settings → claude_code_*):
  • file edits are accepted (--permission-mode acceptEdits); shell commands
    only from an allowlist: git status/diff/log/show, ls/cat, test runners
  • never git push, reset --hard, rm -rf or sudo; no commits — the changes
    are left uncommitted for you to review
  • only an existing folder on this machine, never OpenTeddy's own source
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from config import config

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED = (
    "Read Edit Write MultiEdit Glob Grep LS TodoWrite "
    "Bash(git status:*) Bash(git diff:*) Bash(git log:*) Bash(git show:*) Bash(ls:*) Bash(cat:*) "
    "Bash(pytest:*) Bash(python -m pytest:*) Bash(python3 -m pytest:*) Bash(.venv/bin/python:*) "
    "Bash(npm test:*) Bash(npm run test:*) Bash(pnpm test:*) Bash(yarn test:*) "
    "Bash(go test:*) Bash(cargo test:*) Bash(make test:*)"
)
DISALLOWED = ["Bash(git push:*)", "Bash(git reset --hard:*)", "Bash(rm -rf:*)", "Bash(sudo:*)"]
_SYSTEM = (
    "You were started by OpenTeddy on behalf of the user, who is not at the computer "
    "(they asked from Telegram). Nobody can answer questions: finish the task on your own "
    "or explain what blocked you. Don't commit or push — leave changes uncommitted for "
    "review. End with a short summary in Traditional Chinese: what you changed and how you "
    "verified it."
)


class ClaudeCodeError(Exception):
    """A delegation that can't start, with a message for the user."""


# ── Parsing ──────────────────────────────────────────────────────────────────

# A delegation needs a delegating word before ("用 / 叫 / 交給 Claude Code …") or
# after ("Claude Code 幫我 … / Claude Code：…"); "Claude Code 是什麼" is a question.
_ASK_BEFORE = re.compile(r"^\s*(?:請|麻煩)?\s*(?:幫我)?\s*(?:用|叫|讓|請|交給|派|ask)\s*claude\s*code\s*"
                         r"(?:幫我|來|去|處理|做|to)?\s*[:：,，]?\s*(.+)$", re.I | re.S)
_ASK_AFTER = re.compile(r"^\s*claude\s*code\s*(?:幫我|來|去|處理|做|to|[:：])\s*[:：,，]?\s*(.+)$", re.I | re.S)
_STATUS = re.compile(r"claude\s*code.{0,12}(進度|做到哪|做完了?嗎|好了沒|狀態|status|done)", re.I)
_PATH = re.compile(r"(?:在|於|到|in)?\s*((?:~|/)[^\s，,。；;「」]+)(?:\s*(?:裡面|裡|目錄|資料夾|專案|repo))?", re.I)


def parse_request(goal: str) -> Optional[Tuple[str, Optional[str]]]:
    """(task, directory or None) for "用 Claude Code …", else None."""
    t = (goal or "").strip()
    if not t or t.startswith("[Attachments in workspace") or _STATUS.search(t):
        return None
    if re.search(r"(嗎|呢|？|\?)\s*$", t):
        return None
    m = _ASK_BEFORE.match(t) or _ASK_AFTER.match(t)
    if not m:
        return None
    task = m.group(1).strip()
    directory = None
    pm = _PATH.search(task)
    if pm:
        directory = pm.group(1)
        task = (task[: pm.start()] + task[pm.end():]).strip(" ，,：:")
    return (task, directory) if task else None


def parse_status(goal: str) -> bool:
    return bool(_STATUS.search(goal or ""))


# ── Running ──────────────────────────────────────────────────────────────────

def find_binary() -> Optional[str]:
    configured = (getattr(config, "claude_code_bin", "") or "").strip()
    for cand in (configured, shutil.which("claude") or "", os.path.expanduser("~/.local/bin/claude"),
                 os.path.expanduser("~/.claude/local/claude")):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def resolve_directory(directory: Optional[str], session_dir: Optional[str] = None) -> str:
    d = directory or session_dir or (getattr(config, "claude_code_default_dir", "") or "").strip()
    if not d:
        raise ClaudeCodeError("要在哪個專案資料夾執行？例如「用 Claude Code 在 ~/projects/shop 修好測試」，"
                              "或在 Settings 設定 claude_code_default_dir。")
    full = os.path.realpath(os.path.expanduser(d))
    if not os.path.isdir(full):
        raise ClaudeCodeError(f"這台機器上找不到資料夾：{d}")
    try:
        from tools.shell_tool import _is_openteddy_source_path
        if _is_openteddy_source_path(full):
            raise ClaudeCodeError("不能讓 Claude Code 在 OpenTeddy 自己的程式碼裡工作。")
    except ImportError:
        pass
    return full


async def _git(cwd: str, *args: str) -> str:
    proc = await asyncio.create_subprocess_exec("git", "-C", cwd, *args, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.DEVNULL)
    out, _ = await proc.communicate()
    return out.decode(errors="replace") if proc.returncode == 0 else ""


def _status_map(porcelain: str) -> Dict[str, str]:
    return {line[3:]: line[:2].strip() for line in porcelain.splitlines() if len(line) > 3}


async def run(task: str, directory: str, timeout_s: Optional[float] = None) -> Dict[str, Any]:
    """Run `claude -p` in `directory` and wait. Returns
    {ok, result, changed:[(status, path)], cost_usd, turns, seconds, error}."""
    binary = find_binary()
    if not binary:
        raise ClaudeCodeError("這台機器沒有安裝 Claude Code（找不到 claude 指令）。安裝並登入後再試。")
    timeout_s = float(timeout_s or getattr(config, "claude_code_timeout", 1800) or 1800)
    allowed = (getattr(config, "claude_code_allowed_tools", "") or DEFAULT_ALLOWED).strip()
    allowed_list = re.findall(r"[A-Za-z]+\([^)]*\)|[A-Za-z]+", allowed)
    before = _status_map(await _git(directory, "status", "--porcelain"))

    cmd = [binary, "-p", task, "--allowedTools", *allowed_list, "--disallowedTools", *DISALLOWED,
           "--permission-mode", "acceptEdits", "--append-system-prompt", _SYSTEM, "--output-format", "json"]
    env = {**os.environ}
    env["PATH"] = os.pathsep.join([os.path.expanduser("~/.local/bin"), env.get("PATH", "")])
    t0 = time.monotonic()
    proc = await asyncio.create_subprocess_exec(*cmd, cwd=directory, env=env, stdin=asyncio.subprocess.DEVNULL,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {"ok": False, "result": "", "changed": [], "seconds": time.monotonic() - t0,
                "error": f"超過 {int(timeout_s // 60)} 分鐘還沒做完，已停止"}
    seconds = time.monotonic() - t0
    text = out.decode(errors="replace").strip()
    data: Dict[str, Any] = {}
    try:
        data = json.loads(text.splitlines()[-1]) if text else {}
    except (ValueError, IndexError):
        data = {}
    after = _status_map(await _git(directory, "status", "--porcelain"))
    changed = sorted((st, path) for path, st in after.items() if before.get(path) != st)
    ok = proc.returncode == 0 and not data.get("is_error")
    error = ""
    if not ok:
        error = (data.get("result") or err.decode(errors="replace") or text or f"exit {proc.returncode}").strip()[:500]
    return {"ok": ok, "result": (data.get("result") or "").strip() if ok else "", "changed": changed,
            "cost_usd": data.get("total_cost_usd"), "turns": data.get("num_turns"),
            "seconds": seconds, "error": error}


def describe(task: str, directory: str, res: Dict[str, Any]) -> str:
    mins, secs = divmod(int(res.get("seconds") or 0), 60)
    took = f"{mins} 分 {secs} 秒" if mins else f"{secs} 秒"
    where = directory.replace(os.path.expanduser("~"), "~")
    if not res["ok"]:
        return f"❌ Claude Code 沒有完成「{task[:40]}」（{took}）\n📁 {where}\n{res['error']}"
    lines = [f"🤖 Claude Code 完成「{task[:40]}」（{took}）", f"📁 {where}", "", res["result"][:2500]]
    if res["changed"]:
        lines += ["", "**變更的檔案**（尚未 commit，請檢查後再提交）："]
        lines += [f"• {st} {path}" for st, path in res["changed"][:15]]
        if len(res["changed"]) > 15:
            lines.append(f"• …還有 {len(res['changed']) - 15} 個")
    else:
        lines += ["", "沒有修改任何檔案。"]
    return "\n".join(lines)


# ── Background jobs ──────────────────────────────────────────────────────────

_jobs: Dict[str, Dict[str, Any]] = {}
Notify = Callable[[str], Awaitable[None]]


def start(task: str, directory: str, notify: Optional[Notify] = None,
          on_done: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None) -> str:
    """Start in the background; `notify(text)` gets the result message."""
    job_id = uuid.uuid4().hex[:8]
    job = {"id": job_id, "task": task, "directory": directory, "status": "running",
           "started": time.time(), "finished": None, "summary": ""}
    _jobs[job_id] = job

    async def _go() -> None:
        try:
            res = await run(task, directory)
        except ClaudeCodeError as exc:
            res = {"ok": False, "result": "", "changed": [], "seconds": time.time() - job["started"], "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            logger.exception("Claude Code job %s failed", job_id)
            res = {"ok": False, "result": "", "changed": [], "seconds": time.time() - job["started"],
                   "error": f"{type(exc).__name__}: {exc}"}
        text = describe(task, directory, res)
        job.update(status="done" if res["ok"] else "failed", finished=time.time(), summary=text,
                   changed=[p for _, p in res["changed"]])
        logger.info("Claude Code job %s %s in %.0fs", job_id, job["status"], res.get("seconds") or 0)
        if notify:
            try:
                await notify(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Claude Code job %s: notify failed: %s", job_id, exc)
        if on_done:
            try:
                await on_done(text, res)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Claude Code job %s: on_done failed: %s", job_id, exc)

    job["task_obj"] = asyncio.create_task(_go(), name=f"claude_code:{job_id}")
    return job_id


def jobs(limit: int = 10) -> List[Dict[str, Any]]:
    rows = sorted(_jobs.values(), key=lambda j: -j["started"])[:limit]
    return [{k: v for k, v in j.items() if k != "task_obj"} for j in rows]


def describe_jobs() -> str:
    rows = jobs(5)
    if not rows:
        return "🤖 目前沒有交給 Claude Code 的工作。"
    out = ["🤖 Claude Code 工作："]
    for j in rows:
        mins = int(((j["finished"] or time.time()) - j["started"]) // 60)
        state = {"running": f"進行中（{mins} 分鐘）", "done": "已完成", "failed": "沒有完成"}[j["status"]]
        out.append(f"• `{j['id']}` {state}：「{j['task'][:30]}」")
    return "\n".join(out)
