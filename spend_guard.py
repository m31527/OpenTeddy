"""
OpenTeddy spend guard — hard limits on API calls that spend money.

The destructive-SQL denylist exists because an LLM with a wrong WHERE
clause deletes data. The same failure mode with an ad-platform token
spends money: a confabulated budget, a prompt-injected "set status to
ACTIVE", a retry loop that re-raises the same budget five times. So the
rules here are enforced server-side on every http_post to an ad-platform
host, and — like the SQL denylist — they hold **regardless of approval
state**. Anything beyond them is done by a human in the platform's own
UI, on purpose.

Rules (all configurable in config / .env):

  * Pausing is always allowed — it is the safe direction.
  * ACTIVATING is refused unless OPENTEDDY_AD_ALLOW_ACTIVATE=true. Agents
    create campaigns/ad sets/ads PAUSED; a human switches them on.
  * Creating a campaign / ad set / ad without status=PAUSED is refused.
  * A budget field (daily_budget, lifetime_budget, bid_amount…) must be
    ≤ the configured cap. The caps default to 0, which refuses every
    budget change until the operator has consciously set them.
  * At most N budget-setting calls per UTC day (default 3), counted only
    on success — a runaway retry loop cannot re-raise a budget all
    afternoon.

Amounts are in the platform's minor units (Meta: cents), exactly as the
API takes them, so a cap of 5000 means $50.00/day.
"""

from __future__ import annotations

import datetime as _dt
import logging
import re
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qsl, urlparse

from config import config

logger = logging.getLogger(__name__)

# Hosts whose POSTs can spend money. Meta today; Google Ads is listed so
# the guard already covers it when that door is opened.
AD_HOSTS: Tuple[str, ...] = ("graph.facebook.com", "googleads.googleapis.com")

_BUDGET_KEYS = ("daily_budget", "lifetime_budget", "bid_amount",
                "campaign_daily_budget", "campaign_lifetime_budget",
                "adset_budget", "budget_amount_micros")
_LIFETIME_KEYS = ("lifetime_budget", "campaign_lifetime_budget")
_CREATE_PATH = re.compile(r"/act_\d+/(campaigns|adsets|ads)$", re.IGNORECASE)

# Per-UTC-day counter of successful budget-setting calls.
_budget_calls: Dict[str, int] = {}


def _today() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%d")


def _is_ad_host(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return False
    return any(host == h or host.endswith("." + h) for h in AD_HOSTS)


def _params(url: str, body: Any, form: Any) -> Dict[str, Any]:
    """Every parameter the platform would see, from query, JSON body and
    multipart form, keys lower-cased. Meta accepts all three."""
    out: Dict[str, Any] = {}
    try:
        for k, v in parse_qsl(urlparse(url).query, keep_blank_values=True):
            out[k.lower()] = v
    except Exception:  # noqa: BLE001
        pass
    for src in (body, form):
        if isinstance(src, dict):
            for k, v in src.items():
                out[str(k).lower()] = v
    return out


def _as_int(v: Any) -> Optional[int]:
    try:
        return int(float(str(v).strip()))
    except Exception:  # noqa: BLE001
        return None


def _budget_fields(p: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in p.items() if k in _BUDGET_KEYS}


def budget_calls_today() -> int:
    return _budget_calls.get(_today(), 0)


def check_spend(url: str, body: Any = None, form: Any = None) -> Optional[str]:
    """Return a refusal message, or None when the call may proceed."""
    if not _is_ad_host(url):
        return None
    p = _params(url, body, form)
    path = urlparse(url).path or ""
    status = str(p.get("status", "")).strip().upper()
    allow_activate = bool(getattr(config, "ad_allow_activate", False))

    # 1) activation
    if status == "ACTIVE" and not allow_activate:
        return (
            "🚫 Spend guard: setting status=ACTIVE is not allowed for agents. "
            "Create and edit PAUSED; a human activates in Ads Manager. "
            "(OPENTEDDY_AD_ALLOW_ACTIVATE=true to change this.)"
        )
    # 2) creating a spendable object must be PAUSED
    if _CREATE_PATH.search(path) and status != "PAUSED":
        return (
            "🚫 Spend guard: campaigns / ad sets / ads must be created with "
            "status=PAUSED. Add status=PAUSED and retry; a human activates "
            "them in Ads Manager."
        )
    # 3) budgets
    budgets = _budget_fields(p)
    if budgets:
        daily_cap = int(getattr(config, "ad_max_daily_budget", 0) or 0)
        life_cap = int(getattr(config, "ad_max_lifetime_budget", 0) or 0)
        for key, raw in budgets.items():
            amount = _as_int(raw)
            if amount is None:
                return f"🚫 Spend guard: could not parse {key}={raw!r} as an amount."
            cap = life_cap if key in _LIFETIME_KEYS else daily_cap
            if cap <= 0:
                return (
                    f"🚫 Spend guard: budget changes are refused until a cap is set "
                    f"({key}={amount}). Set OPENTEDDY_AD_MAX_DAILY_BUDGET / "
                    f"OPENTEDDY_AD_MAX_LIFETIME_BUDGET (minor units, e.g. cents) in .env."
                )
            if amount > cap:
                return (
                    f"🚫 Spend guard: {key}={amount} exceeds the cap of {cap} "
                    f"(minor units). Propose it to a human instead; larger "
                    f"budgets are set in Ads Manager."
                )
        per_day = int(getattr(config, "ad_max_budget_changes_per_day", 3) or 3)
        if budget_calls_today() >= per_day:
            return (
                f"🚫 Spend guard: {per_day} budget changes have already been made "
                f"today (UTC). No more automated budget changes until tomorrow."
            )
    return None


def note_spend_call(url: str, body: Any = None, form: Any = None) -> None:
    """Count a SUCCESSFUL budget-setting call toward today's limit."""
    if not _is_ad_host(url):
        return
    if _budget_fields(_params(url, body, form)):
        day = _today()
        _budget_calls[day] = _budget_calls.get(day, 0) + 1
        # keep the dict from growing forever
        for k in [k for k in _budget_calls if k != day]:
            _budget_calls.pop(k, None)
        logger.info("spend guard: budget change #%d today", _budget_calls[day])
