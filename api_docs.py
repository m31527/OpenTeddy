"""
Condense API documentation into something an agent can actually use.

An action-taking agent needs to know WHICH endpoints exist, not just
which hosts it may call. Three ways to supply that, all ending in the
same stored text (the same "capture once, inject into the prompt"
pattern as db_schema — fetching docs at task time would cost latency and
can fail mid-task):

    * paste the docs straight in (markdown / plain text) — always works,
      including for internal wikis that aren't machine-readable
    * upload a file (.md / .txt / .json / .yaml)
    * fetch a URL (only useful when the docs are actually reachable)

An OpenAPI / Swagger spec is condensed to a compact endpoint list —
a 100 KB spec is mostly schema boilerplate the model doesn't need, and
pasting it raw would blow the context window. Anything else is stored
as-is (trimmed to the cap).
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

MAX_CHARS = 9000
_MAX_ENDPOINTS = 120
_METHODS = ("get", "post", "put", "patch", "delete")


def _parse_structured(text: str):
    """Return a dict when the text is JSON or YAML, else None."""
    t = (text or "").strip()
    if not t:
        return None
    if t.startswith("{"):
        try:
            v = json.loads(t)
            return v if isinstance(v, dict) else None
        except Exception:  # noqa: BLE001
            return None
    # YAML (OpenAPI specs are commonly YAML). Guard: yaml.safe_load
    # happily returns a plain string for ordinary prose, so require a dict.
    try:
        import yaml
        v = yaml.safe_load(t)
        return v if isinstance(v, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _is_openapi(spec: dict) -> bool:
    return bool(
        isinstance(spec, dict)
        and ("openapi" in spec or "swagger" in spec)
        and isinstance(spec.get("paths"), dict)
    )


def _param_repr(params) -> str:
    out = []
    for p in params or []:
        if not isinstance(p, dict):
            continue
        name = p.get("name")
        if not name:
            continue
        req = "*" if p.get("required") else ""
        out.append(f"{name}{req}")
    return ", ".join(out)


def _body_repr(rb) -> str:
    """Flatten a requestBody's top-level JSON properties to 'a, b*, c'."""
    try:
        schema = (
            (rb or {}).get("content", {})
            .get("application/json", {})
            .get("schema", {})
        )
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        if not props:
            return ""
        return ", ".join(
            f"{k}{'*' if k in required else ''}" for k in list(props)[:20]
        )
    except Exception:  # noqa: BLE001
        return ""


def condense_openapi(spec: dict) -> str:
    """OpenAPI dict → compact endpoint list ('* = required')."""
    info = spec.get("info") or {}
    lines = []
    title = info.get("title") or "API"
    ver = info.get("version") or ""
    lines.append(f"{title} {ver}".strip())
    servers = [
        s.get("url") for s in (spec.get("servers") or [])
        if isinstance(s, dict) and s.get("url")
    ]
    if servers:
        lines.append("base: " + ", ".join(servers[:3]))
    lines.append("endpoints (* = required):")

    count = 0
    for path, ops in (spec.get("paths") or {}).items():
        if not isinstance(ops, dict):
            continue
        for method, op in ops.items():
            if method.lower() not in _METHODS or not isinstance(op, dict):
                continue
            if count >= _MAX_ENDPOINTS:
                lines.append("…(more endpoints omitted)")
                return "\n".join(lines)
            count += 1
            desc = (op.get("summary") or op.get("description") or "").strip()
            desc = desc.splitlines()[0][:110] if desc else ""
            parts = [f"- {method.upper()} {path}"]
            if desc:
                parts.append(f" — {desc}")
            q = _param_repr(op.get("parameters"))
            if q:
                parts.append(f" | params: {q}")
            b = _body_repr(op.get("requestBody"))
            if b:
                parts.append(f" | body: {b}")
            lines.append("".join(parts))
    return "\n".join(lines)


def condense(text: str) -> str:
    """Normalise any supplied API documentation into stored text.

    OpenAPI/Swagger gets condensed to an endpoint list; markdown / plain
    text is kept as written (the author already chose what matters), just
    capped so it can't swamp the prompt.
    """
    text = (text or "").strip()
    if not text:
        return ""
    spec = _parse_structured(text)
    if spec is not None and _is_openapi(spec):
        try:
            out = condense_openapi(spec)
            logger.info("API docs: condensed OpenAPI spec (%d → %d chars)",
                        len(text), len(out))
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAPI condense failed, storing raw: %s", exc)
            out = text
    else:
        out = text
    if len(out) > MAX_CHARS:
        out = out[:MAX_CHARS] + "\n…(API docs truncated — trim to the "
        "endpoints this agent actually needs)"
    return out


async def fetch_and_condense(url: str, headers: dict | None = None,
                             timeout_s: float = 20.0) -> str:
    """Fetch docs from a URL and condense. Raises on fetch failure so the
    caller can tell the user the URL didn't work (rather than silently
    storing nothing)."""
    import httpx
    async with httpx.AsyncClient(timeout=timeout_s, follow_redirects=True) as c:
        resp = await c.get(url, headers=headers or {})
        resp.raise_for_status()
        return condense(resp.text)
