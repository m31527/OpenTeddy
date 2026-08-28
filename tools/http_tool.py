"""
OpenTeddy HTTP Tool
Async HTTP GET (low risk) and POST (high risk) via httpx.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0

# ── Agent outbound-API policy ─────────────────────────────────────────────────
#
# Action-taking agents (customer service, ticketing, ops) need to call
# authenticated APIs. Putting a token in the agent's persona would push
# it into every prompt — and into transcripts, exports and logs. Instead:
#
#   * the agent stores credentials server-side (secret, never returned
#     by any API, never injected into a prompt)
#   * the MODEL only learns the NAMES and writes a {{CRED:name}}
#     placeholder in the url / headers / body
#   * this tool substitutes the real value immediately before the request
#
# Substitution happens ONLY for a host on the agent's allowed_domains
# list, so a prompt-injected "POST your token to evil.com" cannot
# exfiltrate anything: the placeholder simply never resolves there.

_CRED_RE = re.compile(r"\{\{CRED:([A-Za-z0-9_.-]+)\}\}")


def _host_allowed(url: str, allowed: List[str]) -> bool:
    """Host matches an allowlist entry exactly or as a subdomain."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:  # noqa: BLE001
        return False
    if not host:
        return False
    for d in allowed:
        d = (d or "").strip().lower().lstrip(".")
        if not d:
            continue
        if host == d or host.endswith("." + d):
            return True
    return False


def _substitute(obj: Any, creds: Dict[str, str], missing: set) -> Any:
    """Recursively replace {{CRED:name}} placeholders with real values."""
    if isinstance(obj, str):
        def _sub(m):
            name = m.group(1)
            if name in creds:
                return creds[name]
            missing.add(name)
            return m.group(0)
        return _CRED_RE.sub(_sub, obj)
    if isinstance(obj, dict):
        return {k: _substitute(v, creds, missing) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(v, creds, missing) for v in obj]
    return obj


def _redact(text: str, creds: Dict[str, str]) -> str:
    """Strip any credential value that made it into an error string, so a
    failure message can't become the leak."""
    out = text or ""
    for v in creds.values():
        if v and len(v) >= 4:
            out = out.replace(v, "***")
    return out


async def _agent_http_policy() -> Dict[str, Any]:
    """Fetch this session's outbound policy. Any failure degrades to the
    historical unrestricted behaviour with no credentials available."""
    try:
        from tools._context import get_session_id
        session_id = get_session_id()
        if not session_id:
            return {"credentials": {}, "allowed_domains": []}
        import main as _main_module
        return await _main_module.tracker.get_session_http_policy(session_id)
    except Exception:  # noqa: BLE001
        return {"credentials": {}, "allowed_domains": []}


async def _apply_policy(
    url: str, headers: Optional[Dict[str, str]], body: Any,
) -> Tuple[Optional[str], str, Dict[str, str], Any, Dict[str, str]]:
    """Enforce the allowlist and resolve credentials.

    Returns ``(error, url, headers, body, creds)``. When *error* is not
    None the caller must refuse the request.
    """
    policy = await _agent_http_policy()
    creds: Dict[str, str] = policy.get("credentials") or {}
    allowed: List[str] = policy.get("allowed_domains") or []

    # No policy configured → unchanged legacy behaviour.
    if not creds and not allowed:
        return None, url, headers or {}, body, {}

    if allowed and not _host_allowed(url, allowed):
        return (
            f"🚫 Blocked: this agent may only call {', '.join(allowed)} — "
            f"the requested host is not on its allowed-domains list. "
            f"Add it in the agent's settings if this call is intended.",
            url, headers or {}, body, {},
        )

    # Credentials resolve ONLY for an allowlisted host.
    usable = creds if (allowed and _host_allowed(url, allowed)) else {}
    missing: set = set()
    url2  = _substitute(url, usable, missing)
    hdr2  = _substitute(headers or {}, usable, missing)
    body2 = _substitute(body, usable, missing)
    if missing:
        names = ", ".join(sorted(missing))
        if not usable:
            return (
                f"🚫 Credential placeholder(s) {{{{CRED:{names}}}}} cannot be "
                "resolved: this agent has no allowed-domains list, so "
                "credentials are never sent. Add the target host to the "
                "agent's allowed domains first.",
                url, headers or {}, body, {},
            )
        return (
            f"🚫 Unknown credential name(s): {names}. Available: "
            f"{', '.join(sorted(creds)) or '(none)'}.",
            url, headers or {}, body, {},
        )
    return None, url2, hdr2, body2, usable


# ── Implementations ────────────────────────────────────────────────────────────

async def http_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Perform an HTTP GET request. LOW risk.

    Supports {{CRED:name}} placeholders resolved from the agent's stored
    credentials — see the policy block above."""
    start = time.monotonic()
    err, url, headers, params, _creds = await _apply_policy(url, headers, params)
    if err:
        return make_result(False, error=err, duration_ms=_ms(start))
    try:
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url, headers=headers or {}, params=params or {})
        return make_result(
            True,
            result={
                "status_code": resp.status_code,
                "headers": dict(resp.headers),
                "body": _safe_body(resp),
            },
            duration_ms=_ms(start),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("http_get error: %s", _redact(str(exc), _creds))
        return make_result(False, error=_redact(str(exc), _creds),
                           duration_ms=_ms(start))


async def http_post(
    url: str,
    body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Perform an HTTP POST request. HIGH risk — requires approval.

    Supports {{CRED:name}} placeholders resolved from the agent's stored
    credentials — see the policy block above."""
    start = time.monotonic()
    err, url, headers, body, _creds = await _apply_policy(url, headers, body)
    if err:
        return make_result(False, error=err, duration_ms=_ms(start))
    try:
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT, follow_redirects=True) as client:
            resp = await client.post(url, json=body or {}, headers=headers or {})
        return make_result(
            True,
            result={
                "status_code": resp.status_code,
                "headers": dict(resp.headers),
                "body": _safe_body(resp),
            },
            duration_ms=_ms(start),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("http_post error: %s", _redact(str(exc), _creds))
        return make_result(False, error=_redact(str(exc), _creds),
                           duration_ms=_ms(start))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_body(resp: httpx.Response) -> Any:
    """Try JSON decode, fallback to truncated text."""
    try:
        return resp.json()
    except Exception:  # noqa: BLE001
        return resp.text[:4096]


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


# ── Schemas ───────────────────────────────────────────────────────────────────

_SCHEMA_GET: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "http_get",
        "description": "Perform an HTTP GET request to a URL and return the response.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to fetch."},
                "headers": {
                    "type": "object",
                    "description": "Optional HTTP headers as key-value pairs.",
                    "additionalProperties": {"type": "string"},
                },
                "params": {
                    "type": "object",
                    "description": "Optional query parameters.",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["url"],
        },
    },
}

_SCHEMA_POST: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "http_post",
        "description": "Perform an HTTP POST request (sends JSON body). Requires approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to POST to."},
                "body": {
                    "type": "object",
                    "description": "JSON body to send.",
                },
                "headers": {
                    "type": "object",
                    "description": "Optional HTTP headers.",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["url"],
        },
    },
}

# ── Export ─────────────────────────────────────────────────────────────────────

HTTP_TOOLS: List[Tuple[Any, Dict[str, Any], RiskLevel]] = [
    (http_get,  _SCHEMA_GET,  "low"),
    (http_post, _SCHEMA_POST, "high"),
]
