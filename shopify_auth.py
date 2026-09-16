"""
Shopify client-credentials tokens — minted and cached server-side.

A Dev Dashboard app (the only kind Shopify lets a store create since
2026) has no permanent Admin API token. It has a client id + secret, and
`POST https://{shop}/admin/oauth/access_token` exchanges them for a token
that expires after 24 hours. Left to the agent, that means a token in
the prompt every day — the exact thing the credential design forbids.

So the exchange happens here: an agent stores `shopify_store`,
`shopify_client_id` and `shopify_client_secret` as credentials, keeps
writing `{{CRED:shopify_token}}` as before, and http_tool asks this
module for a live token at call time. The model never sees the secret
or the token; the cache keeps the exchange to about once a day.

Requirements Shopify places on this grant: the app must be installed on
the store, and app and store must belong to the same organization.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

TOKEN_KEY = "shopify_token"
_REFRESH_MARGIN_S = 30 * 60          # mint a fresh one when < 30 min remain
_cache: Dict[str, Tuple[str, float]] = {}   # cache key -> (token, expires_at)
last_error: Optional[str] = None


def _store_host(store: str) -> str:
    s = (store or "").strip().replace("https://", "").replace("http://", "").rstrip("/")
    return s


async def get_token(store: str, client_id: str, client_secret: str,
                    timeout_s: float = 20.0) -> str:
    """Return a valid Admin API access token, minting one if needed."""
    global last_error
    host = _store_host(store)
    if not (host and client_id and client_secret):
        raise ValueError("shopify_store, shopify_client_id and shopify_client_secret are all required")
    key = f"{host}|{client_id}"
    tok, exp = _cache.get(key, ("", 0.0))
    if tok and exp - time.monotonic() > _REFRESH_MARGIN_S:
        return tok
    url = f"https://{host}/admin/oauth/access_token"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=5.0)) as c:
            resp = await c.post(url, data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            })
        if resp.status_code >= 400:
            # Shopify's error body is short and specific ("invalid_client",
            # "app not installed on this shop") — worth quoting.
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        tok = str(data.get("access_token") or "")
        if not tok:
            raise RuntimeError(f"no access_token in response: {str(data)[:200]}")
        ttl = float(data.get("expires_in") or 86399)
    except Exception as exc:  # noqa: BLE001
        last_error = f"{type(exc).__name__}: {exc}"
        logger.warning("Shopify token exchange failed for %s: %s", host, last_error)
        raise
    _cache[key] = (tok, time.monotonic() + ttl)
    last_error = None
    logger.info("Shopify token minted for %s (scope=%s, ttl=%ss)", host,
                str(data.get("scope", ""))[:80], int(ttl))
    return tok


async def derive_shopify_token(creds: Dict[str, str]) -> Dict[str, str]:
    """Return a copy of `creds` with `shopify_token` filled in from the
    client-credentials pair, when one is present and no static token is.
    Raises on exchange failure so the caller can say why."""
    if (creds.get(TOKEN_KEY) or "").strip():
        return creds
    cid, sec = creds.get("shopify_client_id", ""), creds.get("shopify_client_secret", "")
    if not (cid and sec):
        return creds
    out = dict(creds)
    out[TOKEN_KEY] = await get_token(creds.get("shopify_store", ""), cid, sec)
    return out


def clear_cache() -> None:
    _cache.clear()
