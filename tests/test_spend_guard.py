"""
Spend guard — money is the other thing a wrong tool call can't take back.

    .venv/bin/python tests/test_spend_guard.py
"""
from __future__ import annotations

import asyncio, logging, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

from config import config
import spend_guard as SG

ACT = "https://graph.facebook.com/v25.0/act_123/"


def reset():
    SG._budget_calls.clear()
    config.ad_max_daily_budget = 0
    config.ad_max_lifetime_budget = 0
    config.ad_max_budget_changes_per_day = 3
    config.ad_allow_activate = False


def main() -> None:
    reset()
    # non-ad hosts are untouched
    assert SG.check_spend("https://api.internal.com/v1/refunds", {"amount": 99999}) is None
    assert SG.check_spend("http://100.72.101.26:8000/v1/videos/sync", form={"daily_budget": "1"}) is None
    print("  ✓ non-ad hosts pass through")

    # pausing always allowed, in body / form / query
    assert SG.check_spend(ACT.replace("act_123/", "") + "120001", {"status": "PAUSED"}) is None
    assert SG.check_spend("https://graph.facebook.com/v25.0/120001?status=PAUSED") is None
    assert SG.check_spend("https://graph.facebook.com/v25.0/120001", form={"status": "paused"}) is None
    print("  ✓ PAUSED allowed via body, query and form")

    # activation refused by default, allowed when configured
    d = SG.check_spend("https://graph.facebook.com/v25.0/120001", {"status": "ACTIVE"})
    assert d and "ACTIVE" in d
    config.ad_allow_activate = True
    assert SG.check_spend("https://graph.facebook.com/v25.0/120001", {"status": "ACTIVE"}) is None
    config.ad_allow_activate = False
    print("  ✓ ACTIVE refused unless OPENTEDDY_AD_ALLOW_ACTIVATE")

    # creating spendable objects must be PAUSED; creatives are free
    d = SG.check_spend(ACT + "campaigns", {"name": "x", "objective": "OUTCOME_SALES"})
    assert d and "PAUSED" in d
    assert SG.check_spend(ACT + "campaigns", {"name": "x", "status": "PAUSED"}) is None
    assert SG.check_spend(ACT + "adsets", form={"name": "x", "status": "PAUSED", "daily_budget": "100"}) is not None  # cap 0
    assert SG.check_spend(ACT + "adcreatives", {"name": "c", "object_story_spec": {}}) is None
    print("  ✓ campaigns/adsets/ads need status=PAUSED; adcreatives allowed")

    # budgets: refused at cap 0, then bounded, then counted per day
    d = SG.check_spend("https://graph.facebook.com/v25.0/2300", {"daily_budget": 5000})
    assert d and "cap is set" in d or "until a cap" in d
    config.ad_max_daily_budget = 5000
    assert SG.check_spend("https://graph.facebook.com/v25.0/2300", {"daily_budget": "5000"}) is None
    d = SG.check_spend("https://graph.facebook.com/v25.0/2300", {"daily_budget": "5001"})
    assert d and "exceeds" in d
    d = SG.check_spend("https://graph.facebook.com/v25.0/2300", {"lifetime_budget": "10"})
    assert d and ("cap" in d)  # lifetime cap still 0
    d = SG.check_spend("https://graph.facebook.com/v25.0/2300", {"daily_budget": "abc"})
    assert d and "parse" in d
    print("  ✓ budget caps: 0 refuses, ≤cap passes, >cap refused, lifetime separate, garbage refused")

    for _ in range(3):
        SG.note_spend_call("https://graph.facebook.com/v25.0/2300", {"daily_budget": "100"})
    assert SG.budget_calls_today() == 3
    d = SG.check_spend("https://graph.facebook.com/v25.0/2300", {"daily_budget": "100"})
    assert d and "already been made today" in d
    SG.note_spend_call("https://graph.facebook.com/v25.0/2300", {"status": "PAUSED"})   # not a budget call
    assert SG.budget_calls_today() == 3
    assert SG.check_spend("https://graph.facebook.com/v25.0/2300", {"status": "PAUSED"}) is None  # pause still fine
    print("  ✓ per-day budget-change limit; pauses don't count and still pass")

    # wired into http_post: refused before any network I/O, regardless of approval
    reset()
    import tools.http_tool as HT
    async def fp():
        return {"credentials": {"meta_token": "t"}, "allowed_domains": ["graph.facebook.com"]}
    HT._agent_http_policy = fp
    async def run():
        r = await HT.http_post("https://graph.facebook.com/v25.0/2300",
                               form={"daily_budget": "100", "access_token": "{{CRED:meta_token}}"})
        assert r["success"] is False and "Spend guard" in r["error"], r
    asyncio.run(run())
    print("  ✓ http_post refuses a budget change at cap 0 without touching the network")
    print("\nALL SPEND GUARD TESTS PASS")


if __name__ == "__main__":
    main()
