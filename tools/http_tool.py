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
        fail = _http_failure(resp, _creds)
        if fail:
            return make_result(False, error=fail, duration_ms=_ms(start))
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
    form: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    save_to: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Perform an HTTP POST request. HIGH risk — requires approval.

    Encoding: pass `body` for JSON, or `form` for multipart/form-data —
    the equivalent of curl's `-F`, which many APIs (file uploads, media
    generation) require and a JSON body would be rejected by.

    `save_to` writes the raw response to a workspace file instead of
    trying to read it as text — required for binary results (video,
    images, PDFs), which would otherwise come back as truncated garbage.
    The saved file is reported as an artifact so it appears as a
    download.

    `timeout` overrides the 30 s default (capped at 15 min) for slow
    generative endpoints.

    Supports {{CRED:name}} placeholders resolved from the agent's stored
    credentials — see the policy block above."""
    start = time.monotonic()
    err, url, headers, body, _creds = await _apply_policy(url, headers, body)
    if err:
        return make_result(False, error=err, duration_ms=_ms(start))
    if form:
        # Credentials can appear in form fields too.
        _missing: set = set()
        form = _substitute(form, _creds, _missing)
    tmo = _DEFAULT_TIMEOUT if not timeout else max(1.0, min(float(timeout), 900.0))
    try:
        async with httpx.AsyncClient(timeout=tmo, follow_redirects=True) as client:
            if form:
                # (None, value) makes httpx emit a multipart part with no
                # filename — byte-for-byte what `curl -F key=value` sends.
                files = {k: (None, str(v)) for k, v in form.items()}
                resp = await client.post(url, files=files, headers=headers or {})
            else:
                resp = await client.post(url, json=body or {}, headers=headers or {})

        fail = _http_failure(resp, _creds)
        if fail:
            # Never write an error payload to disk. A 24-byte
            # {"error":"Unauthorized"} saved as output.mp4 looks like a
            # produced artifact to every layer above (the empty-artifact
            # guard sees a file, the model reports success) — the user
            # gets told a video was generated when nothing was. Fail
            # loudly with the server's actual message instead.
            return make_result(False, error=fail, duration_ms=_ms(start))

        if save_to:
            saved = _save_response(resp, save_to)
            saved["status_code"] = resp.status_code
            warn = _suspect_not_binary(resp)
            if warn:
                saved["warning"] = warn
            return make_result(True, result=saved, duration_ms=_ms(start))

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
    """Try JSON decode, fallback to truncated text.

    Binary payloads (video/image/pdf) are NOT returned as text — a 4 KB
    slice of an mp4 is noise that also wastes the model's context. The
    caller is told to re-issue the request with save_to instead."""
    ctype = (resp.headers.get("content-type") or "").lower()
    if ctype and not any(
        t in ctype for t in ("json", "text", "xml", "html", "javascript")
    ):
        return (
            f"<binary response: {ctype}, {len(resp.content)} bytes — "
            "not shown. Re-run this call with save_to='filename.ext' to "
            "write it to the workspace.>"
        )
    try:
        return resp.json()
    except Exception:  # noqa: BLE001
        return resp.text[:4096]


def _http_failure(resp: httpx.Response, creds: Dict[str, str]) -> Optional[str]:
    """Return a user-facing error when the response is an HTTP failure.

    Without this a 401 came back as success=True: the model saw a
    "successful" call, save_to wrote {"error":"Unauthorized"} to
    output.mp4, and the user was told their video was generated. Any
    non-2xx is a failure and must say so, quoting the server's own
    message (credential values redacted)."""
    if resp.status_code < 400:
        return None
    try:
        detail = resp.text[:600]
    except Exception:  # noqa: BLE001
        detail = f"<{len(resp.content)} bytes>"
    hint = ""
    if resp.status_code in (401, 403):
        hint = (
            " — the credential was rejected. Check that the agent's "
            "credential value is current and that the header uses "
            "{{CRED:name}} (a literal $VAR or placeholder text is sent "
            "as-is and will 401)."
        )
    return _redact(f"HTTP {resp.status_code}: {detail}{hint}", creds)


def _suspect_not_binary(resp: httpx.Response) -> Optional[str]:
    """Flag a 2xx response that was saved but doesn't look like the
    binary the caller expected — some APIs return 200 with an error
    document. Advisory only; the file is still written."""
    ctype = (resp.headers.get("content-type") or "").lower()
    size = len(resp.content)
    if any(t in ctype for t in ("json", "text/plain", "text/html")) or size < 1024:
        return (
            f"Saved {size} bytes of {ctype or 'unknown type'} — this does "
            "NOT look like a media file. It is probably an error response. "
            "Inspect it before reporting success to the user."
        )
    return None


def _save_response(resp: httpx.Response, save_to: str) -> Dict[str, Any]:
    """Write the raw response body to a workspace-anchored file.

    Same path rules as file_tool: relative paths resolve against the
    session workspace, so a saved video lands where the rest of the
    session's artifacts live (and gets picked up as a download)."""
    import os
    from pathlib import Path
    expanded = os.path.expanduser(save_to)
    if os.path.isabs(expanded):
        p = Path(expanded).resolve()
    else:
        try:
            from config import effective_workspace_dir
            ws = effective_workspace_dir()
        except Exception:  # noqa: BLE001
            ws = os.getcwd()
        p = Path(os.path.join(ws, expanded)).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(resp.content)
    return {
        "path": str(p),
        "bytes_written": len(resp.content),
        "content_type": resp.headers.get("content-type", ""),
    }


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
        "description": (
            "Perform an HTTP POST. Use `body` for a JSON payload, or "
            "`form` for multipart/form-data (the equivalent of curl -F, "
            "required by many upload / media-generation APIs). Use "
            "`save_to` to write a binary response (video, image, pdf) to "
            "a workspace file, and `timeout` for slow endpoints. "
            "Requires approval."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to POST to."},
                "body": {
                    "type": "object",
                    "description": "JSON body to send. Omit when using `form`.",
                },
                "form": {
                    "type": "object",
                    "description": (
                        "multipart/form-data fields (curl -F). Values are "
                        "sent as strings. Use this instead of `body` when "
                        "the API expects form fields rather than JSON."
                    ),
                },
                "headers": {
                    "type": "object",
                    "description": (
                        "Optional HTTP headers, e.g. "
                        '{"Authorization": "Bearer {{CRED:my_token}}"}. '
                        "Credential placeholders are substituted server-side."
                    ),
                    "additionalProperties": {"type": "string"},
                },
                "save_to": {
                    "type": "string",
                    "description": (
                        "Write the raw response to this workspace file "
                        "(e.g. 'output.mp4') instead of returning it as "
                        "text. Required for binary responses."
                    ),
                },
                "timeout": {
                    "type": "number",
                    "description": "Seconds to wait (default 30, max 900).",
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
