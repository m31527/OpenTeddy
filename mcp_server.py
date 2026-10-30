"""
OpenTeddy as an MCP server — Claude Code (or any MCP client) can use the
house, the printer, Telegram and the rest of OpenTeddy.

Claude Code is the better engineer; OpenTeddy is the one that is always
on and connected to the physical world. This lets the engineer ask the
butler: "print this report", "tell me on Telegram when the deploy is
done", "turn off the living-room light", "which devices are offline".

It is a thin client of a running OpenTeddy runtime (local or remote), so
the tools work from any machine that can reach it:

    claude mcp add openteddy -- /path/to/OpenTeddy/openteddy --url http://<runtime>:8000 mcp

Requests run in one dedicated session ("Claude Code"), so OpenTeddy keeps
the thread's context between calls.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import httpx
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

RUNTIME = os.environ.get("OPENTEDDY_URL", "http://127.0.0.1:8000").rstrip("/")
SESSION = os.environ.get("OPENTEDDY_MCP_SESSION", "claude-code-mcp")
TASK_TIMEOUT_S = float(os.environ.get("OPENTEDDY_MCP_TIMEOUT", "600"))

mcp = FastMCP(
    "openteddy",
    log_level="WARNING",          # stderr is Claude Code's MCP log; one line per request is noise
    instructions=(
        "OpenTeddy is the user's always-on home/office AI runtime. Use it for things "
        "outside this repository: the house (lights, air-con, the 小愛 speaker, which "
        "devices are offline), the printer, Telegram messages to the user, weather and "
        "typhoons, schedules, and OpenTeddy's agents (databases, shop and ad reports). "
        "Write requests in the user's language (usually Traditional Chinese)."
    ),
)


async def _call(method: str, path: str, *, json: Any = None, files: Any = None,
                data: Any = None, timeout: float = 30.0) -> Any:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.request(method, RUNTIME + path, json=json, files=files, data=data)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenTeddy {r.status_code}: {r.text[:300]}")
    return r.json()


async def _task(request: str) -> str:
    """Run one request through OpenTeddy and wait for its answer."""
    try:
        res = await _call("POST", "/tasks", timeout=TASK_TIMEOUT_S + 30, json={
            "intent": request, "session_id": SESSION, "wait": True, "auto_lane": True,
            "context": {"triggered_by": "mcp"},
        })
    except httpx.ConnectError:
        return f"❌ OpenTeddy isn't reachable at {RUNTIME} (is the service running?)"
    except Exception as exc:  # noqa: BLE001
        return f"❌ {exc}"
    status = str(res.get("status") or "")
    msg = (res.get("message") or "").strip()
    if status not in ("completed", "TaskStatus.COMPLETED") and not msg:
        msg = f"(status: {status})"
    return msg


@mcp.tool()
async def ask_openteddy(request: str) -> str:
    """Hand any request to OpenTeddy, the user's always-on AI runtime, and get its answer.
    Use for anything outside this repo: questions about the home or office, its agents'
    databases and reports, research, or a recurring job ("每天早上 8 點查營收…" creates a
    schedule). Write the request as the user would say it."""
    return await _task(request)


@mcp.tool()
async def home_control(command: str) -> str:
    """Control or query the house through Home Assistant and the 小愛 speaker, e.g.
    "關客廳燈", "冷氣開 26 度", "叫小愛說：部署完成了", "播放音樂", "家裡有哪些設備離線".
    Locks, alarms and garage doors reply with a code the user must confirm themselves."""
    return await _task(command)


@mcp.tool()
async def print_file(path: str, copies: int = 1, two_sided: Optional[bool] = None,
                     black_and_white: bool = False, pages: str = "") -> str:
    """Print a local file on the user's printer (PDF, images, text, Word / Excel /
    PowerPoint). `pages` like "1-3". The file is uploaded to OpenTeddy, then printed."""
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        return f"❌ File not found: {p}"
    try:
        with open(p, "rb") as fh:
            up = await _call("POST", "/upload", files={"file": (os.path.basename(p), fh)},
                             data={"session_id": SESSION}, timeout=120)
    except Exception as exc:  # noqa: BLE001
        return f"❌ Upload failed: {exc}"
    rel = up.get("rel_path") or ("uploads/" + os.path.basename(p))
    size = up.get("size_bytes") or os.path.getsize(p)
    words = ["列印"]
    if copies and int(copies) > 1:
        words.append(f"{int(copies)} 份")
    if two_sided is True:
        words.append("雙面")
    elif two_sided is False:
        words.append("單面")
    if black_and_white:
        words.append("黑白")
    if pages:
        words.append(f"第 {pages} 頁".replace("-", " 到 "))
    manifest = ("[Attachments in workspace — use these paths directly, do not re-upload:\n"
                f"  - {rel} ({max(size, 1) / 1024:.1f} KB)\n]\n\n")
    return await _task(manifest + " ".join(words))


@mcp.tool()
async def notify_me(message: str) -> str:
    """Send the user a Telegram message (e.g. "部署完成，測試全過"). Use when a long job
    finishes or something needs their attention while they're away from the computer."""
    try:
        res = await _call("POST", "/notify", json={"text": message})
    except Exception as exc:  # noqa: BLE001
        return f"❌ {exc}"
    return "✅ sent to Telegram" if res.get("success") else f"❌ {res.get('error')}"


@mcp.tool()
async def weather(question: str = "今天天氣") -> str:
    """Weather or typhoons, e.g. "桃園明天會下雨嗎", "東京這週天氣", "最近有颱風靠近台灣嗎"."""
    return await _task(question)


@mcp.tool()
async def self_review(days: int = 3) -> str:
    """OpenTeddy's self check-up for the last `days` days: questions it answered badly,
    suspicious skills, failures, schedules that didn't run, compute time, environment."""
    try:
        return (await _call("GET", f"/review?days={int(days)}")).get("text", "")
    except Exception as exc:  # noqa: BLE001
        return f"❌ {exc}"


@mcp.tool()
async def list_schedules() -> str:
    """OpenTeddy's recurring jobs with their next run time."""
    try:
        res = await _call("GET", "/schedules")
    except Exception as exc:  # noqa: BLE001
        return f"❌ {exc}"
    rows = res.get("data") if isinstance(res, dict) else res
    if isinstance(rows, dict):
        rows = rows.get("schedules", [])
    if not rows:
        return "No schedules."
    return "\n".join(f"- {r.get('cron')} → {r.get('goal')} (next: {r.get('next_run_at') or '-'}, "
                     f"{'on' if r.get('enabled') else 'off'}, id {str(r.get('id', ''))[:8]})" for r in rows)


def main(url: Optional[str] = None) -> None:
    global RUNTIME
    if url:
        RUNTIME = url.rstrip("/")
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    mcp.run("stdio")


if __name__ == "__main__":
    main()
