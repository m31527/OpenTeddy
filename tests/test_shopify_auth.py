"""
Shopify client-credentials tokens — minted server-side, cached, refreshed;
the agent keeps writing {{CRED:shopify_token}} and never sees the secret.

    .venv/bin/python tests/test_shopify_auth.py
"""
from __future__ import annotations

import asyncio, logging, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.disable(logging.WARNING)

import shopify_auth as SA


class _Resp:
    def __init__(self, status, payload, text=""):
        self.status_code, self._p, self.text = status, payload, text or str(payload)
    def json(self): return self._p


class _Client:
    calls: list = []
    status = 200
    payload = {"access_token": "shpat_minted_1", "scope": "read_orders,read_reports", "expires_in": 86399}
    def __init__(self, *a, **k): pass
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    async def post(self, url, data=None, **kw):
        _Client.calls.append({"url": url, "data": data})
        return _Resp(_Client.status, _Client.payload, text=str(_Client.payload))


async def main() -> None:
    import types
    SA.httpx = types.SimpleNamespace(AsyncClient=_Client, Timeout=lambda *a, **k: None)  # type: ignore[assignment]
    SA.clear_cache()

    # 1) mint: form-encoded client_credentials to the store's oauth endpoint
    tok = await SA.get_token("https://moldgogo.myshopify.com/", "cid", "sec")
    assert tok == "shpat_minted_1"
    c = _Client.calls[-1]
    assert c["url"] == "https://moldgogo.myshopify.com/admin/oauth/access_token", c["url"]
    assert c["data"] == {"grant_type": "client_credentials", "client_id": "cid", "client_secret": "sec"}
    print("  ✓ mints via POST /admin/oauth/access_token with grant_type=client_credentials")

    # 2) cached: second call makes no request
    n = len(_Client.calls)
    assert await SA.get_token("moldgogo.myshopify.com", "cid", "sec") == "shpat_minted_1"
    assert len(_Client.calls) == n
    print("  ✓ cached — no second exchange")

    # 3) refresh when close to expiry
    key = "moldgogo.myshopify.com|cid"
    SA._cache[key] = ("shpat_old", time.monotonic() + 60)     # 1 minute left
    _Client.payload = {**_Client.payload, "access_token": "shpat_minted_2"}
    assert await SA.get_token("moldgogo.myshopify.com", "cid", "sec") == "shpat_minted_2"
    print("  ✓ refreshed inside the 30-minute margin")

    # 4) failure surfaces Shopify's message
    _Client.status = 400; _Client.payload = {"error": "invalid_client"}
    SA.clear_cache()
    try:
        await SA.get_token("moldgogo.myshopify.com", "cid", "bad")
        print("  ✗ should have raised")
    except RuntimeError as exc:
        assert "400" in str(exc) and "invalid_client" in str(exc) and SA.last_error
    _Client.status = 200; _Client.payload = {"access_token": "shpat_minted_3", "expires_in": 86399}
    print("  ✓ exchange failure raises with the server's reason")

    # 5) derive: static token wins; pair mints; neither → untouched
    assert (await SA.derive_shopify_token({"shopify_token": "static"}))["shopify_token"] == "static"
    d = await SA.derive_shopify_token({"shopify_store": "moldgogo.myshopify.com",
                                       "shopify_client_id": "cid", "shopify_client_secret": "sec"})
    assert d["shopify_token"] == "shpat_minted_3" and "shopify_client_secret" in d
    assert "shopify_token" not in await SA.derive_shopify_token({"meta_token": "x"})
    print("  ✓ derive_shopify_token: static wins, pair mints, unrelated creds untouched")

    # 6) end-to-end through http_tool's policy: placeholder resolves to the minted token
    import tools.http_tool as HT
    async def fp():
        return {"credentials": {"shopify_store": "moldgogo.myshopify.com",
                                "shopify_client_id": "cid", "shopify_client_secret": "sec"},
                "allowed_domains": ["moldgogo.myshopify.com"]}
    HT._agent_http_policy = fp
    err, url, hdr, body, creds = await HT._apply_policy(
        "https://moldgogo.myshopify.com/admin/api/2026-07/graphql.json",
        {"X-Shopify-Access-Token": "{{CRED:shopify_token}}"}, {"query": "{ shop { name } }"})
    assert err is None and hdr["X-Shopify-Access-Token"] == "shpat_minted_3", (err, hdr)
    # off-allowlist host: refused, nothing minted into it
    err2, *_ = await HT._apply_policy("https://evil.com/x", {"X-Shopify-Access-Token": "{{CRED:shopify_token}}"}, None)
    assert err2 and "Blocked" in err2
    # exchange failure → specific message
    _Client.status = 400; _Client.payload = {"error": "app not installed"}; SA.clear_cache()
    err3, *_ = await HT._apply_policy(
        "https://moldgogo.myshopify.com/admin/api/2026-07/graphql.json",
        {"X-Shopify-Access-Token": "{{CRED:shopify_token}}"}, None)
    assert err3 and "could not be minted" in err3 and "app not installed" in err3, err3
    print("  ✓ http_tool: placeholder → minted token on the allowed host; refused elsewhere; clear error on failure")
    print("\nALL SHOPIFY AUTH TESTS PASS")


asyncio.run(main())
