"""
OpenTeddy HTTP Tool
Async HTTP GET (low risk) and POST (high risk) via httpx.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from tool_registry import RiskLevel, make_result

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0


# ── Implementations ────────────────────────────────────────────────────────────

async def http_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Perform an HTTP GET request. LOW risk."""
    start = time.monotonic()
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
        logger.error("http_get error [%s]: %s", url, exc)
        return make_result(False, error=str(exc), duration_ms=_ms(start))


async def http_post(
    url: str,
    body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Perform an HTTP POST request. HIGH risk — requires approval."""
    start = time.monotonic()
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
        logger.error("http_post error [%s]: %s", url, exc)
        return make_result(False, error=str(exc), duration_ms=_ms(start))


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
