"""
API keys: how an app reaches OpenTeddy, the OpenAI way.

    Authorization: Bearer ot-…

A key is made in Settings → API 金鑰 and shown once; only its SHA-256 is
kept, with a short hint ("ot-AbC…wxyz") to tell keys apart. Revoking one
stops it at once.

Each key is its own person, "api:<id>" (privacy.py). The sessions it
starts are its own: another key can't read or continue them, and since
memory is kept per session (memory.py), what it learns stays there too.
It never reaches the owner's paired Chrome or accounts. The owner's web UI
still lists those sessions, titled 「🔑 <key name> · …」.

The /v1 routes (public_api.py) are the only part of OpenTeddy a key
opens, and every one of them needs a key.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Tuple

import aiosqlite

from config import config

PREFIX = "ot-"
PERSON_PREFIX = "api:"
SUBPROTOCOL = "openteddy"                 # what /v1/realtime answers with
KEY_SUBPROTOCOL = "openteddy-key."        # a browser can't set headers on a WebSocket: the key rides here

_TABLES = (
    "CREATE TABLE IF NOT EXISTS api_keys ("
    " id TEXT PRIMARY KEY, name TEXT NOT NULL, hash TEXT NOT NULL UNIQUE, hint TEXT NOT NULL,"
    " created_at TEXT NOT NULL, last_used_at TEXT, uses INTEGER NOT NULL DEFAULT 0, revoked_at TEXT)",
    "CREATE TABLE IF NOT EXISTS api_key_sessions ("
    " session_id TEXT PRIMARY KEY, key_id TEXT NOT NULL, created_at TEXT NOT NULL)",
    # What an app set for a whole conversation: its role (`instructions`), and
    # whether it chose to answer without the knowledge base (`"datasets": []`).
    "CREATE TABLE IF NOT EXISTS api_session_settings ("
    " session_id TEXT PRIMARY KEY, instructions TEXT NOT NULL DEFAULT '',"
    " no_datasets INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL)",
    # OpenTeddy's own secrets (the key that signs conversation grants).
    "CREATE TABLE IF NOT EXISTS api_meta (name TEXT PRIMARY KEY, value TEXT NOT NULL)",
)
_ready_for: Optional[str] = None          # the db path the tables were made in


def _db() -> str:
    return config.db_path


async def _ensure() -> None:
    global _ready_for
    if _ready_for == _db():
        return
    async with aiosqlite.connect(_db()) as db:
        for sql in _TABLES:
            await db.execute(sql)
        await db.commit()
    _ready_for = _db()


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.utcnow().isoformat()


def person(key_id: str) -> str:
    return PERSON_PREFIX + key_id


def _public(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: row[k] for k in ("id", "name", "hint", "created_at", "last_used_at", "uses", "revoked_at")}


# ── Keys ─────────────────────────────────────────────────────────────────────

async def create(name: str) -> Tuple[Dict[str, Any], str]:
    """A new key: (its record, the key itself — shown once, never stored)."""
    name = " ".join((name or "").split())[:60]
    if not name:
        raise ValueError("a key needs a name")
    await _ensure()
    token = PREFIX + secrets.token_urlsafe(32)
    row = {"id": uuid.uuid4().hex[:12], "name": name, "hint": f"{token[:6]}…{token[-4:]}",
           "created_at": _now(), "last_used_at": None, "uses": 0, "revoked_at": None}
    async with aiosqlite.connect(_db()) as db:
        await db.execute(
            "INSERT INTO api_keys (id, name, hash, hint, created_at) VALUES (?, ?, ?, ?, ?)",
            (row["id"], name, _hash(token), row["hint"], row["created_at"]))
        await db.commit()
    return row, token


async def list_keys() -> List[Dict[str, Any]]:
    await _ensure()
    async with aiosqlite.connect(_db()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM api_keys ORDER BY created_at DESC") as cur:
            return [_public(r) for r in await cur.fetchall()]


async def revoke(key_id: str) -> bool:
    await _ensure()
    async with aiosqlite.connect(_db()) as db:
        cur = await db.execute("UPDATE api_keys SET revoked_at=? WHERE id=? AND revoked_at IS NULL",
                               (_now(), key_id))
        await db.commit()
        return cur.rowcount > 0


async def verify(token: str) -> Optional[Dict[str, Any]]:
    """The key's record when `token` is a live key (and count the use)."""
    token = (token or "").strip()
    if not token.startswith(PREFIX) or len(token) > 200:
        return None
    await _ensure()
    async with aiosqlite.connect(_db()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM api_keys WHERE hash=? AND revoked_at IS NULL",
                              (_hash(token),)) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        now = _now()
        await db.execute("UPDATE api_keys SET last_used_at=?, uses=uses+1 WHERE id=?", (now, row["id"]))
        await db.commit()
    return {**_public(row), "last_used_at": now}


def token_from(headers: Mapping[str, str]) -> str:
    """The key a request carries: `Authorization: Bearer …` (the OpenAI
    way), else `X-API-Key`, else a WebSocket subprotocol `openteddy-key.…`."""
    auth = (headers.get("authorization") or "").strip()
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip()
    if headers.get("x-api-key"):
        return headers["x-api-key"].strip()
    for proto in (headers.get("sec-websocket-protocol") or "").split(","):
        proto = proto.strip()
        if proto.startswith(KEY_SUBPROTOCOL):
            return proto[len(KEY_SUBPROTOCOL):]
    return ""


# ── Sessions a key owns ──────────────────────────────────────────────────────

async def claim_session(key_id: str, session_id: str) -> None:
    await _ensure()
    async with aiosqlite.connect(_db()) as db:
        await db.execute("INSERT OR IGNORE INTO api_key_sessions (session_id, key_id, created_at) VALUES (?, ?, ?)",
                         (session_id, key_id, _now()))
        await db.commit()


async def owns_session(key_id: str, session_id: str) -> bool:
    if not session_id:
        return False
    await _ensure()
    async with aiosqlite.connect(_db()) as db:
        async with db.execute("SELECT key_id FROM api_key_sessions WHERE session_id=?", (session_id,)) as cur:
            row = await cur.fetchone()
    return bool(row) and row[0] == key_id


# ── What an app set for a conversation ───────────────────────────────────────

_SETTINGS = ("instructions", "no_datasets")


async def _set(session_id: str, column: str, value: Any) -> None:
    assert column in _SETTINGS
    await _ensure()
    async with aiosqlite.connect(_db()) as db:
        await db.execute(f"INSERT INTO api_session_settings (session_id, {column}, updated_at) VALUES (?, ?, ?) "
                         f"ON CONFLICT(session_id) DO UPDATE SET {column}=excluded.{column}, "
                         "updated_at=excluded.updated_at", (session_id, value, _now()))
        await db.commit()


async def _get(session_id: str, column: str) -> Any:
    assert column in _SETTINGS
    if not session_id:
        return None
    await _ensure()
    async with aiosqlite.connect(_db()) as db:
        async with db.execute(f"SELECT {column} FROM api_session_settings WHERE session_id=?",
                              (session_id,)) as cur:
            row = await cur.fetchone()
    return row[0] if row else None


async def set_instructions(session_id: str, text: str) -> None:
    """The conversation's role (`instructions`); "" removes it."""
    await _set(session_id, "instructions", text or "")


async def get_instructions(session_id: str) -> str:
    return (await _get(session_id, "instructions")) or ""


async def set_no_datasets(session_id: str, none: bool) -> None:
    """Whether the conversation answers without the knowledge base — the
    "always" datasets too — because the app named `"datasets": []`."""
    await _set(session_id, "no_datasets", int(bool(none)))


async def no_datasets(session_id: str) -> bool:
    return bool(await _get(session_id, "no_datasets"))


# ── Grants: a key, for one conversation, for a while ─────────────────────────
#
# A LiveKit room's agent speaks for the app that opened the room
# (POST /v1/livekit/token): it needs to act as that app's key, in that one
# conversation, without ever holding the key. OpenTeddy hands it a grant
# instead — "otg-<payload>.<signature>", signed with a secret of its own —
# that /v1 takes in place of the key for listening, speaking and the turns
# of that conversation. It ends when it expires or the key is revoked.

GRANT_PREFIX = "otg-"


async def _grant_secret() -> bytes:
    await _ensure()
    async with aiosqlite.connect(_db()) as db:
        async with db.execute("SELECT value FROM api_meta WHERE name='grant_secret'") as cur:
            row = await cur.fetchone()
        if row is None:
            await db.execute("INSERT OR IGNORE INTO api_meta (name, value) VALUES ('grant_secret', ?)",
                             (secrets.token_hex(32),))
            await db.commit()
            async with db.execute("SELECT value FROM api_meta WHERE name='grant_secret'") as cur:
                row = await cur.fetchone()
    return bytes.fromhex(row[0])


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


async def issue_grant(key_id: str, session_id: str, ttl_s: int) -> str:
    payload = _b64(json.dumps({"k": key_id, "s": session_id, "e": int(time.time()) + int(ttl_s)},
                              separators=(",", ":")).encode())
    sig = _b64(hmac.new(await _grant_secret(), payload.encode(), hashlib.sha256).digest())
    return f"{GRANT_PREFIX}{payload}.{sig}"


async def verify_grant(token: str) -> Optional[Dict[str, Any]]:
    """The key's record, with "grant_session", when `token` is a live grant
    of a live key."""
    token = (token or "").strip()
    if not token.startswith(GRANT_PREFIX) or len(token) > 1000 or "." not in token:
        return None
    payload, _, sig = token[len(GRANT_PREFIX):].partition(".")
    good = _b64(hmac.new(await _grant_secret(), payload.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, good):
        return None
    try:
        claims = json.loads(_unb64(payload))
    except ValueError:
        return None
    if not isinstance(claims, dict) or int(claims.get("e") or 0) < time.time():
        return None
    async with aiosqlite.connect(_db()) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM api_keys WHERE id=? AND revoked_at IS NULL",
                              (str(claims.get("k") or ""),)) as cur:
            row = await cur.fetchone()
    if row is None:
        return None
    return {**_public(row), "grant_session": str(claims.get("s") or "")}
