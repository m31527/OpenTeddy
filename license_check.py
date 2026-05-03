"""License verification for OpenTeddy premium skills.

This module lives in the OSS repo on purpose — it's the *contract* that
the @premium decorator (also OSS, in skills/_premium.py) calls into.
But the actual premium skill .py files that USE the decorator live in
the openteddy-pro-content private repo. At desktop-build time, those
private files get copied into skills/ alongside the OSS-generated
skills; the desktop sidecar bundles firebase-admin pre-initialised so
this license check works inside the bundled binary.

For OSS users running `uvicorn main:app` directly:
  - The skills/ dir doesn't contain any @premium-decorated skill files
    in the first place (those are private content), so this code path
    rarely runs.
  - Even if a curious user copied a premium skill file into skills/,
    they likely don't have firebase-admin installed + a service account
    configured, so `is_active_license()` returns False (skill stays
    locked, displays the upgrade message). No accidental free Pro.

Threat model:
  - Casual user: sees the upgrade message, considers buying. ✓
  - Determined hacker: can patch the decorator out by editing the
    bundled .py — possible but requires unpacking the PyInstaller
    bundle, which is friction beyond what 99% of users will attempt.
    DRM-via-decorator is honour-system gate, not crypto. We're fine
    with that — the model is "pay because the convenience is worth
    it", not "we make piracy impossible".
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Avoid hammering Firestore on every invocation. We cache "is this uid
# licensed?" answers for ~5 minutes — long enough to dedupe back-to-back
# skill calls in the same task, short enough that a freshly-purchased
# Lifetime activates within a few minutes even without an app restart.
# (The desktop's onSnapshot listener flips the upgrade-pill UI in ~1 s;
# this cache is just for the backend's skill-gating path.)
import time
_LICENSE_CACHE: dict[str, tuple[bool, float]] = {}
_CACHE_TTL_SEC = 300


async def is_active_license(uid: Optional[str]) -> bool:
    """True iff `users/{uid}.subscription.status == 'active'` in Firestore.

    Returns False (deny access) on any failure path — missing
    firebase-admin SDK, network unavailable, document missing, the
    'subscription' field absent, etc. The default-deny posture means
    OSS installs without Firebase wired up never accidentally unlock
    premium skills.
    """
    if not uid:
        return False

    # Cache hit?
    cached = _LICENSE_CACHE.get(uid)
    if cached is not None:
        ok, expires_at = cached
        if time.monotonic() < expires_at:
            return ok

    ok = await _check_license_uncached(uid)
    _LICENSE_CACHE[uid] = (ok, time.monotonic() + _CACHE_TTL_SEC)
    return ok


def invalidate_license_cache(uid: Optional[str] = None) -> None:
    """Drop the cached license decision for one user (or all users).

    Useful when we know the license state just changed — e.g. the
    Cloud Function webhook just wrote `subscription.status='active'`,
    and we'd rather force a fresh read than wait out the 5 min TTL.
    Called from the eventual webhook→backend notification channel.
    """
    if uid is None:
        _LICENSE_CACHE.clear()
    else:
        _LICENSE_CACHE.pop(uid, None)


async def _check_license_uncached(uid: str) -> bool:
    """The actual Firestore round-trip. Lazy-imports firebase_admin so
    OSS installs that don't have it pip-installed don't pay the import
    cost on every uvicorn boot — they get a clean ImportError → False
    return path."""
    try:
        # Late import. Sidecar bundle has firebase_admin frozen in;
        # plain OSS uvicorn likely doesn't.
        from firebase_admin import firestore, initialize_app
        from firebase_admin._apps import _apps  # type: ignore[attr-defined]
    except ImportError:
        logger.debug("firebase_admin not installed — license stays locked")
        return False

    try:
        # Initialise Firebase Admin lazily. In the sidecar bundle the
        # service-account credentials are picked up from a path we set
        # via GOOGLE_APPLICATION_CREDENTIALS; in Cloud Functions it's
        # automatic; for OSS uvicorn this raises and we deny.
        if not _apps:
            initialize_app()
        db = firestore.client()
        doc = db.collection("users").document(uid).get()
        if not doc.exists:
            return False
        data = doc.to_dict() or {}
        sub = data.get("subscription") or {}
        return sub.get("status") == "active"
    except Exception as exc:  # noqa: BLE001
        # Wide catch — Firestore can throw network errors, auth errors,
        # quota errors, malformed-doc errors. ANY of them should lock
        # the premium skill (better safe than free Pro). Log so an op
        # can spot a recurring issue.
        logger.warning(
            "license check failed for uid=%s: %s — locking premium skill",
            uid, exc,
        )
        return False
