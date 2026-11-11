"""
What a task did on the way to its answer: the model's notes between tool
calls and every action it took, kept with the task (tasks.process) so the
answer can show it folded under "How it got there" — after a reload, on
another device, and above all when a step didn't finish and the answer
alone would make the run look like it found nothing (2026-11-11: the
process had a PChome price table, the answer said "not on sale yet").

Built from the broadcast stream the browser already gets (chat.stream.*,
tool_call / tool_result, process.note), so every runner that reports its
progress — executor, Claude Fix — is recorded without code of its own.
A private task (someone's calendar, mail or browser) is never kept,
same as its summary (privacy.hidden_from_web).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_MAX_ITEMS = 200        # per task
_MAX_TASKS = 40         # in flight; a task that never sends task.done is dropped
_NOTE_CAP = 6000        # chars per note
_ARG_CAP = 300          # chars per string argument
_OUT_CAP = 1500         # chars of a tool's output
_SLIM_OVER = 120_000    # a record bigger than this keeps only short outputs

_live: Dict[str, Dict[str, Any]] = {}


def _entry(task_id: str) -> Dict[str, Any]:
    e = _live.get(task_id)
    if e is None:
        while len(_live) >= _MAX_TASKS:
            _live.pop(next(iter(_live)))
        e = _live[task_id] = {"items": [], "steps": {}, "open": None}
    return e


def _key(ev: Dict[str, Any]) -> str:
    return f"{ev.get('source', '')}|{ev.get('round')}|{ev.get('call_idx') or 0}|{ev.get('tool')}"


def _clip_args(args: Any) -> Any:
    if not isinstance(args, dict):
        return {}
    return {k: (v[:_ARG_CAP] + "…" if isinstance(v, str) and len(v) > _ARG_CAP else v)
            for k, v in args.items()}


def record(ev: Dict[str, Any]) -> None:
    """Fold one broadcast event into its task's record. Cheap and sync."""
    tid = ev.get("task_id")
    if not tid:
        return
    kind = ev.get("type") or ev.get("event") or ""
    if kind == "chat.stream.delta":
        text = ev.get("text") or ""
        if not text:
            return
        e = _entry(tid)
        if e["open"] is None:
            if len(e["items"]) >= _MAX_ITEMS:
                return
            e["open"] = {"k": "note", "text": ""}
            e["items"].append(e["open"])
        if len(e["open"]["text"]) < _NOTE_CAP:
            e["open"]["text"] += text
    elif kind == "chat.stream.end":
        if tid in _live:
            _live[tid]["open"] = None
    elif kind == "process.note":
        e = _entry(tid)
        e["open"] = None
        text = (ev.get("text") or "").strip()
        if text and len(e["items"]) < _MAX_ITEMS:
            e["items"].append({"k": "note", "text": text[:_NOTE_CAP]})
    elif kind == "tool_call":
        e = _entry(tid)
        e["open"] = None
        if len(e["items"]) >= _MAX_ITEMS:
            return
        item = {"k": "step", "tool": ev.get("tool") or "", "args": _clip_args(ev.get("args"))}
        e["items"].append(item)
        e["steps"][_key(ev)] = item
    elif kind == "tool_result":
        e = _live.get(tid)
        item = e["steps"].get(_key(ev)) if e else None
        if item is None:
            return
        item["ok"] = ev.get("success") is not False
        out = (ev.get("output") or "") if item["ok"] else (ev.get("error") or ev.get("output") or "")
        if out:
            item["out"] = str(out)[:_OUT_CAP]


def take(task_id: str) -> List[Dict[str, Any]]:
    """The finished record, cleaned up; forgets the task."""
    e = _live.pop(task_id, None)
    if not e:
        return []
    items: List[Dict[str, Any]] = []
    for it in e["items"]:
        if it["k"] == "note":
            text = it["text"].strip()
            if text:
                items.append({"k": "note", "text": text})
        else:
            items.append(it)
    if len(json.dumps(items, ensure_ascii=False)) > _SLIM_OVER:
        for it in items:
            if "out" in it:
                it["out"] = it["out"][:200]
    return items


async def finish(tracker: Any, task_id: str, *, phase: str = "") -> None:
    """Save the record when the task (or a Claude Fix of it) ends. A later
    phase is appended after a marker instead of replacing the first run."""
    items = take(task_id)
    if not items:
        return
    import privacy
    if privacy.hidden_from_web(task_id):
        return
    if phase:
        items.insert(0, {"k": "phase", "text": phase})
    try:
        await tracker.save_task_process(task_id, items, append=bool(phase))
    except Exception as exc:  # noqa: BLE001
        logger.warning("process record for task %s not saved: %s", task_id, exc)


async def listener(tracker: Any, ev: Dict[str, Any]) -> None:
    """ws_manager listener: record, and save on the task's last event."""
    record(ev)
    kind = ev.get("type") or ev.get("event") or ""
    if kind == "task.done" and ev.get("task_id"):
        await finish(tracker, ev["task_id"])
    elif kind == "claude_fix_done" and ev.get("task_id"):
        await finish(tracker, ev["task_id"], phase="claude-fix")
