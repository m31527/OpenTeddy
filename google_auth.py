"""
Each person links their own Google account — calendar and Gmail, read-only.

One OAuth client per OpenTeddy install (the owner creates it once in
Google Cloud and pastes the client ID / secret into Settings); every
person then links their own account from their Telegram private chat:

  1. "連結我的 Google" → a personal sign-in link (one use, 15 minutes,
     PKCE, bound to that chat)
  2. sign in, allow read-only calendar + Gmail
  3. Google sends the browser to http://127.0.0.1:8749/?code=… — a page
     that won't open on a phone; the person copies that address and pastes
     it back into the chat. Google refuses LAN addresses (192.168.x.x) as
     a return address, so this works from anywhere with no network setup.

Refresh tokens are encrypted at rest (key file next to the database, mode
600) and never leave this module; "解除 Google" revokes the token at
Google and deletes it. Scopes: openid email, calendar.readonly,
gmail.readonly.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import secrets
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import httpx

from config import config

logger = logging.getLogger(__name__)

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
REDIRECT_URI = "http://127.0.0.1:8749/"
CAL_SCOPE = "https://www.googleapis.com/auth/calendar.readonly"
MAIL_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
SCOPES = ["openid", "email", CAL_SCOPE, MAIL_SCOPE]
LINK_TTL_S = 900

_tracker: Any = None
_pending: Dict[str, Dict[str, Any]] = {}          # state → {person, verifier, expires}
_access: Dict[str, Tuple[str, float]] = {}        # person → (access token, expires at)

_TABLE = (
    "CREATE TABLE IF NOT EXISTS google_accounts ("
    "  person TEXT PRIMARY KEY, email TEXT NOT NULL DEFAULT '', refresh_token TEXT NOT NULL,"
    "  scopes TEXT NOT NULL DEFAULT '', linked_at TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'ok'"
    ")"
)


class GoogleAuthError(Exception):
    """A linking / token problem, with a message for the person."""


def configured() -> bool:
    return bool((getattr(config, "google_client_id", "") or "").strip()
                and (getattr(config, "google_client_secret", "") or "").strip())


# Said to a person: the feature just isn't on here. Setup steps are for the
# owner (Settings → 📅 Google, README "Personal assistant"), not for them.
NOT_CONFIGURED = "這台 OpenTeddy 沒有開啟 Google 行事曆／Gmail 功能（管理者可以在 Settings → 📅 Google 開啟）。"
NOT_LINKED = ("📅 這要連結你的 Google 帳號才查得到。想用的話說「連結我的 Google」；"
              "不連結也沒關係，其他功能都照常可以用。")


# ── Encryption at rest ───────────────────────────────────────────────────────

def _fernet():
    from cryptography.fernet import Fernet
    path = os.path.join(os.path.dirname(os.path.abspath(config.db_path)) or ".", "google_tokens.key")
    if not os.path.exists(path):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(Fernet.generate_key())
    with open(path, "rb") as fh:
        return Fernet(fh.read().strip())


def _seal(token: str) -> str:
    return _fernet().encrypt(token.encode()).decode()


def _open_sealed(blob: str) -> str:
    return _fernet().decrypt(blob.encode()).decode()


# ── Storage ──────────────────────────────────────────────────────────────────

async def _db():
    if _tracker is None:
        raise RuntimeError("google_auth not started")
    await _tracker.db.execute(_TABLE)
    return _tracker.db


async def start(tracker: Any) -> None:
    global _tracker
    _tracker = tracker
    await _db()
    await tracker.db.commit()


async def account(person: str) -> Optional[Dict[str, Any]]:
    """{email, scopes, linked_at, status} — never the token."""
    db = await _db()
    async with db.execute("SELECT email, scopes, linked_at, status FROM google_accounts WHERE person=?",
                          (person,)) as cur:
        row = await cur.fetchone()
    return dict(row) if row else None


async def granted(person: Optional[str]) -> set:
    """The scopes this person granted and can still use; empty when they
    never linked, unlinked, or the link expired. Linking is optional:
    without it they just don't get calendar and mail."""
    if not person or not configured():
        return set()
    try:
        acct = await account(person)
    except Exception:  # noqa: BLE001
        return set()
    if not acct or acct.get("status") != "ok":
        return set()
    return set((acct.get("scopes") or "").split())


async def count() -> int:
    db = await _db()
    async with db.execute("SELECT COUNT(*) FROM google_accounts WHERE status='ok'") as cur:
        return int((await cur.fetchone())[0])


# ── Linking ──────────────────────────────────────────────────────────────────

_LINK = re.compile(r"^\s*(?:/google|(?:請|幫我)?(?:連結|綁定|連接|登入|link|connect)\s*(?:我的)?\s*(?:google|gmail|google\s*帳號|谷歌))", re.I)
_UNLINK = re.compile(r"^\s*(?:/google_unlink|(?:解除|取消|移除|中斷|斷開)(?:連結|綁定)?\s*(?:我的)?\s*(?:google|gmail|谷歌)|unlink google|disconnect google)", re.I)
_STATUS = re.compile(r"(?:google|gmail).{0,6}(?:連結|綁定)(?:了嗎|狀態|了沒)|我(?:連結|綁定)了哪個\s*google", re.I)


def parse_command(text: str) -> Optional[str]:
    """'link' | 'unlink' | 'status' | None."""
    t = (text or "").strip()
    if not t or len(t) > 40:
        return None
    if _UNLINK.search(t):
        return "unlink"
    if _STATUS.search(t):
        return "status"
    if _LINK.search(t):
        return "link"
    return None


def parse_callback(text: str) -> Optional[Tuple[str, str]]:
    """(code, state) from the address Google sent the browser to."""
    t = (text or "").strip()
    get = lambda k: re.search(rf"(?:^|[?&\s]){k}=([^&\s#]+)", t)
    code, state = get("code"), get("state")
    if not code or not state:
        return None
    return urllib.parse.unquote(code.group(1)), urllib.parse.unquote(state.group(1))


def begin(person: str) -> str:
    """A one-use sign-in link for this person."""
    if not configured():
        raise GoogleAuthError(NOT_CONFIGURED)
    now = time.time()
    for s in [s for s, p in _pending.items() if p["expires"] < now]:
        _pending.pop(s, None)
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(24)
    _pending[state] = {"person": person, "verifier": verifier, "expires": now + LINK_TTL_S}
    return AUTH_URL + "?" + urllib.parse.urlencode({
        "client_id": config.google_client_id.strip(), "redirect_uri": REDIRECT_URI,
        "response_type": "code", "scope": " ".join(SCOPES), "state": state,
        "code_challenge": challenge, "code_challenge_method": "S256",
        "access_type": "offline", "prompt": "consent", "include_granted_scopes": "true",
    })


def _email_from_id_token(id_token: str) -> str:
    try:
        payload = id_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload)).get("email", "")
    except Exception:  # noqa: BLE001
        return ""


async def finish(person: str, code: str, state: str) -> Dict[str, Any]:
    """Exchange the pasted code; returns {email, missing: [scopes not granted]}."""
    p = _pending.get(state)
    if not p or p["expires"] < time.time():
        _pending.pop(state, None)
        raise GoogleAuthError("這個登入連結已過期或用過了，請重新說「連結我的 Google」。")
    if p["person"] != person:
        raise GoogleAuthError("這個登入連結不是給這個聊天的。請在你自己的 Telegram 私訊說「連結我的 Google」。")
    _pending.pop(state, None)
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(TOKEN_URL, data={
            "client_id": config.google_client_id.strip(), "client_secret": config.google_client_secret.strip(),
            "code": code, "code_verifier": p["verifier"], "grant_type": "authorization_code",
            "redirect_uri": REDIRECT_URI,
        })
    if r.status_code >= 400:
        err = (r.json() if r.headers.get("content-type", "").startswith("application/json") else {}).get("error", "")
        logger.info("Google link failed for %s: HTTP %s %s", person, r.status_code, err)
        raise GoogleAuthError("Google 不接受這個登入（可能貼錯網址或已逾時），請重新說「連結我的 Google」。")
    tok = r.json()
    refresh = tok.get("refresh_token")
    if not refresh:
        raise GoogleAuthError("Google 沒有給長期授權。請到 Google 帳戶 → 安全性 → 第三方存取權移除 OpenTeddy，"
                              "再重新說「連結我的 Google」。")
    granted = set((tok.get("scope") or "").split())
    email = _email_from_id_token(tok.get("id_token") or "")
    db = await _db()
    await db.execute(
        "INSERT INTO google_accounts(person, email, refresh_token, scopes, linked_at, status) "
        "VALUES (?, ?, ?, ?, ?, 'ok') ON CONFLICT(person) DO UPDATE SET email=excluded.email, "
        "refresh_token=excluded.refresh_token, scopes=excluded.scopes, linked_at=excluded.linked_at, status='ok'",
        (person, email, _seal(refresh), " ".join(sorted(granted)),
         datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
    )
    await db.commit()
    _access[person] = (tok.get("access_token") or "", time.time() + int(tok.get("expires_in") or 0) - 60)
    logger.info("Google linked for %s", person)
    return {"email": email, "missing": [s for s in (CAL_SCOPE, MAIL_SCOPE) if s not in granted]}


async def unlink(person: str) -> Optional[str]:
    """Revoke at Google and forget it. Returns the email, or None if none was linked."""
    db = await _db()
    async with db.execute("SELECT email, refresh_token FROM google_accounts WHERE person=?", (person,)) as cur:
        row = await cur.fetchone()
    if not row:
        return None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            await client.post(REVOKE_URL, data={"token": _open_sealed(row["refresh_token"])})
    except Exception as exc:  # noqa: BLE001 — forget it locally either way
        logger.info("Google revoke for %s failed: %s", person, exc)
    await db.execute("DELETE FROM google_accounts WHERE person=?", (person,))
    await db.commit()
    _access.pop(person, None)
    return row["email"]


async def access_token(person: str, force: bool = False) -> str:
    """A current access token for this person (refreshed when needed)."""
    hit = _access.get(person)
    if hit and hit[0] and hit[1] > time.time() and not force:
        return hit[0]
    if not configured():
        raise GoogleAuthError(NOT_CONFIGURED)
    db = await _db()
    async with db.execute("SELECT refresh_token, status FROM google_accounts WHERE person=?", (person,)) as cur:
        row = await cur.fetchone()
    if not row:
        raise GoogleAuthError(NOT_LINKED)
    if row["status"] != "ok":
        raise GoogleAuthError("你的 Google 連結已失效，請重新說「連結我的 Google」。")
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(TOKEN_URL, data={
            "client_id": config.google_client_id.strip(), "client_secret": config.google_client_secret.strip(),
            "refresh_token": _open_sealed(row["refresh_token"]), "grant_type": "refresh_token",
        })
    if r.status_code in (400, 401):
        await db.execute("UPDATE google_accounts SET status='expired' WHERE person=?", (person,))
        await db.commit()
        _access.pop(person, None)
        raise GoogleAuthError("你的 Google 連結已失效（被撤銷或過期），請重新說「連結我的 Google」。")
    if r.status_code >= 400:
        raise GoogleAuthError(f"Google 暫時無法回應（HTTP {r.status_code}），請稍後再試。")
    tok = r.json()
    _access[person] = (tok["access_token"], time.time() + int(tok.get("expires_in") or 3600) - 60)
    return tok["access_token"]
