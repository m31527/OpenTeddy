"""
An answer's words as the model writes them, for whoever is waiting on that task.

The chat lane (gate.py) answers most turns in one model call. Waiting for
the whole answer before showing or speaking any of it cost the time the
model spends writing it — 2.1 s for a three-sentence answer on spark-a,
all of it before the first word (2026-11-23). Now a caller that wants the
words as they come registers before the task starts:

    q = answer_stream.listen(task_id)       # then start the task with that id
    while (piece := await q.get()) is not answer_stream.END: ...
    answer_stream.close(task_id, q)

and the gate, seeing someone listening, streams its call and pushes each
piece. END comes when the model's text is complete — before the task
itself is done (tracker and memory writes follow). A turn the gate hands
to the agent pushes nothing; the caller then waits for the task's result
as before. Users: /v1/chat/completions with stream (public_api.py) and the
spoken conversation (voice_bridge.py), which starts speaking the first
sentence while the rest is still being written.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List

END = None                         # pushed once the answer's text is complete

_queues: Dict[str, List["asyncio.Queue"]] = {}


def listen(task_id: str) -> "asyncio.Queue":
    q: asyncio.Queue = asyncio.Queue()
    _queues.setdefault(task_id, []).append(q)
    return q


def close(task_id: str, q: "asyncio.Queue") -> None:
    qs = _queues.get(task_id) or []
    if q in qs:
        qs.remove(q)
    if not qs:
        _queues.pop(task_id, None)


def wanted(task_id: str) -> bool:
    return bool(task_id) and bool(_queues.get(task_id))


def push(task_id: str, text: str) -> None:
    if text:
        for q in _queues.get(task_id, []):
            q.put_nowait(text)


def end(task_id: str) -> None:
    for q in _queues.get(task_id, []):
        q.put_nowait(END)
