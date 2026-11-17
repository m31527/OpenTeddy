"""
A skill the system proposes by itself waits for the person.

The 2026-11-14/15 evals left five self-made skills ACTIVE on home. One,
"python_exec_skill" (the python_exec tool, renamed to get past the
tool-name rule), answered every data step afterwards with "no executable
code supplied", counted that as a success 52 times, and took over 6 of the
person's own tasks. Another was called "none".

  A. names: tool names (also with _skill / _tool), "none" and non-names
     are refused for an automatic skill
  B. status: an automatic skill registers as TESTING even when its test
     passes; one the person asks for still goes ACTIVE when tested
  C. the executor's request uses the automatic path, and a refused name
     is no error

    .venv/bin/python tests/test_skill_creation.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.CRITICAL)

from config import config
from models import SkillStatus
from skill_factory import SkillFactory, auto_skill_name_problem
from tool_registry import tool_registry


def test_names() -> None:
    tool_registry.auto_register_all()
    for bad in ("python_exec_skill", "python_exec", "read_file_tool", "none", "None", "n/a", "", "x",
                "做報表", "skill"):
        assert auto_skill_name_problem(bad), bad
    for ok in ("monthly_revenue_report", "taipei_dome_events", "csv_top_products"):
        assert auto_skill_name_problem(ok) == "", ok
    print("  ✓ tool names (+_skill/_tool), 'none' and non-names aren't skill names")


class _Tracker:
    def __init__(self): self.saved = []
    async def get_skill(self, name): return None
    async def upsert_skill(self, s): self.saved.append(s)


def _factory() -> SkillFactory:
    sf = SkillFactory.__new__(SkillFactory)
    sf.tracker = _Tracker()
    sf._loaded, sf._regen_inflight = {}, set()
    async def code(self, *a): return "async def run(input_data):\n    return 'no executable code supplied'\n"
    async def test_input(self, *a): return {}
    sf._generate_code = types.MethodType(code, sf)
    sf._generate_test_input = types.MethodType(test_input, sf)
    sf._write_skill_file = lambda name, code: None
    return sf


async def test_status() -> None:
    config.skill_test_before_register = True
    sf = _factory()
    s = await sf.generate_skill("csv_top_products", "top products from a csv", auto=True)
    assert s.status == SkillStatus.TESTING and s.success_count == 0, s.status
    s = await sf.generate_skill("csv_top_products", "top products from a csv")
    assert s.status == SkillStatus.ACTIVE, "one the person asked for: as before"
    try:
        await sf.generate_skill("python_exec_skill", "The python_exec tool failed…", auto=True)
        raise AssertionError("created")
    except ValueError as exc:
        assert "python_exec tool" in str(exc)
    print("  ✓ an automatic skill waits as TESTING even when its test passes; a requested one goes ACTIVE")


async def test_executor_path() -> None:
    from executor import Executor
    calls = []
    class _SF:
        tracker = _Tracker()
        async def generate_skill(self, name, desc, auto=False):
            calls.append((name, auto))
            raise ValueError(f"Not creating skill {name!r}: it names the python_exec tool")
    ex = Executor.__new__(Executor)
    ex.skill_factory = _SF()
    await ex._request_skill_creation("python_exec_skill", "…")
    assert calls == [("python_exec_skill", True)]
    print("  ✓ the executor asks on the automatic path; a refused name is no error")


async def main() -> None:
    print("A. names"); test_names()
    print("B. status"); await test_status()
    print("C. executor"); await test_executor_path()
    print("\nALL SKILL CREATION TESTS PASS")


asyncio.run(main())
