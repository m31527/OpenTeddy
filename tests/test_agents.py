#!/usr/bin/env python3
"""
Tests for user-defined Agents (feature/agents).

    .venv/bin/python tests/test_agents.py

Real SQLite tracker; no LLM involved (agents are pure configuration —
persona injection is exercised at the prompt-assembly level).
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)


async def main():
    from tracker import Tracker
    from models import AgentDefinition, SessionMode

    t = Tracker(db_path=tempfile.mktemp(suffix=".db"))
    await t.open()
    try:
        # 1) CRUD round-trip
        agent = AgentDefinition(
            name="財務 DB 分析師",
            description="分析財務資料庫並產出報告",
            system_prompt="你是財務資料分析師。回答一律附上查詢依據，金額用千分位。",
            mode=SessionMode.ANALYTIC,
            local_only=True,
            db_kind="mysql",
            db_url="mysql://user:secret@10.0.0.5/finance",
            db_label="finance@10.0.0.5",
        )
        await t.upsert_agent(agent)
        row = await t.get_agent(agent.id)
        assert row and row["name"] == "財務 DB 分析師"
        assert row["db_url"].endswith("/finance")
        assert (await t.list_agents())[0]["id"] == agent.id
        print("1) agent CRUD round-trip                       OK")

        # 2) create session from agent → config copied onto session row
        await t.create_session_from_agent("sess-1", row)
        sess = await t.get_session("sess-1")
        assert sess["agent_id"] == agent.id
        assert sess["mode"] == "analytic"
        assert bool(sess["local_only"]) is True
        assert sess["db_kind"] == "mysql" and sess["db_url"].endswith("/finance")
        assert sess["db_label"] == "finance@10.0.0.5"
        print("2) session-from-agent copies mode/db/privacy   OK")

        # 3) editing the agent does NOT need session updates (live lookup):
        #    persona is read from the agents table at task time.
        updated = AgentDefinition(
            **{**agent.model_dump(), "system_prompt": "改版後的人設"},
        )
        await t.upsert_agent(updated)
        assert (await t.get_agent(agent.id))["system_prompt"] == "改版後的人設"
        print("3) agent edit visible to future lookups        OK")

        # 4) persona injection block (executor-side assembly logic)
        persona = f"[Agent: {row['name']}]\n" + row["system_prompt"].strip()
        base = "BASE SYSTEM PROMPT"
        effective = ("[Agent persona — you are acting as this configured "
                     "role for the whole session:]\n" + persona + "\n\n" + base)
        assert effective.index(persona) < effective.index(base), \
            "persona must precede task mechanics"
        print("4) persona block precedes base system prompt   OK")

        # 5) delete agent → sessions detach (not deleted)
        assert await t.delete_agent(agent.id) is True
        assert await t.get_agent(agent.id) is None
        sess = await t.get_session("sess-1")
        assert sess is not None and sess["agent_id"] == "", sess["agent_id"]
        assert sess["db_url"].endswith("/finance"), "session keeps its copied config"
        print("5) delete agent detaches (not deletes) session OK")

        print("\nALL AGENT TESTS PASS")
    finally:
        await t.close()


if __name__ == "__main__":
    asyncio.run(main())
