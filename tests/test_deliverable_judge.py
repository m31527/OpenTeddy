"""
Deliverable judge — the verdict must be driven by the judge's rules, not
by whether the model's prose contains the word "report".

    .venv/bin/python tests/test_deliverable_judge.py
"""
from __future__ import annotations

import asyncio, json, logging, os, sys, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

from config import config
from models import SubTask
from orchestrator import Orchestrator


class _Resp:
    def __init__(self, content): self._c = content
    def raise_for_status(self): pass
    def json(self): return {"message": {"content": self._c}}


def _fake_self(reply: str, captured: dict):
    async def post(url, json=None, timeout=None):  # noqa: A002
        captured["url"] = url; captured["payload"] = json
        return _Resp(reply)
    fs = types.SimpleNamespace()
    fs._http = types.SimpleNamespace(post=post)
    fs._DELIVERABLE_JUDGE_PROMPT = Orchestrator._DELIVERABLE_JUDGE_PROMPT
    fs._file_hash = Orchestrator._file_hash
    return fs


async def main() -> None:
    config.verification_enabled = True
    d = tempfile.mkdtemp()
    html = os.path.join(d, "order_daily.html")
    with open(html, "w") as fh:
        fh.write("<html><body><h1>訂單數量日報</h1><p>2026-09-05: 0 · 2026-09-06: 0</p></body></html>")
    arts = [{"path": html, "size_bytes": os.path.getsize(html)}]
    st = SubTask(parent_task_id="t", description="統計兩日訂單數並產出包含數量對比的 Markdown 報告")

    # 1) request shape: system rides as a role=system message, never top-level
    cap = {}
    out = await Orchestrator._verify_deliverable(_fake_self(json.dumps({"verdict": "PASS", "reason": "real report"}), cap), st, arts)
    p = cap["payload"]
    assert "system" not in p, "top-level system field would be silently dropped by /api/chat"
    assert p["messages"][0]["role"] == "system" and "PASS" in p["messages"][0]["content"], p["messages"][0]
    assert p["messages"][-1]["role"] == "user" and "GOAL:" in p["messages"][-1]["content"]
    assert p.get("format") == "json" and cap["url"].endswith("/api/chat")
    assert out[1] is True and out[2] == html
    print("  ✓ judge prompt travels as role=system; format=json; PASS parsed; path returned")

    # 2) explicit FAIL still fails
    out = await Orchestrator._verify_deliverable(_fake_self(json.dumps({"verdict": "FAIL", "reason": "placeholder"}), {}), st, arts)
    assert out[1] is False
    print("  ✓ explicit FAIL honoured")

    # 3) prose verdict mentioning "report" on a document goal → can't tell → None (was: FAIL)
    out = await Orchestrator._verify_deliverable(_fake_self("This is a report of the daily order counts, both days are zero.", {}), st, arts)
    assert out is None, out
    print("  ✓ prose containing 'report' no longer condemns a report goal")

    # 4) prose with a strong placeholder signal on a CODE goal → FAIL
    st2 = SubTask(parent_task_id="t", description="implement the snake game in JS")
    out = await Orchestrator._verify_deliverable(_fake_self("It is only a skeleton with placeholder functions.", {}), st2, arts)
    assert out is not None and out[1] is False
    print("  ✓ strong placeholder signal still fails a code goal")

    # 5) file hash helper
    h1 = Orchestrator._file_hash(html); assert h1 and h1 == Orchestrator._file_hash(html)
    assert Orchestrator._file_hash(os.path.join(d, "nope")) is None
    print("  ✓ _file_hash stable / None on missing")
    print("\nALL DELIVERABLE JUDGE TESTS PASS")


asyncio.run(main())
