"""@premium decorator for Pro-tier skills.

Premium skills (the .py files that actually carry the IP — Memory Pro,
Data Analyst Pro, Marketing Pro, etc.) live in the
openteddy-pro-content private repo, NOT here. At desktop-build time
those private files get copied into this skills/ directory alongside
the OSS-grown ones, so they share the skill_factory load path.

Each premium skill imports + applies @premium to its run() function:

    from skills._premium import premium

    @premium
    async def run(input_data: dict) -> str:
        # ... actual implementation; runs only when license is active
        ...

Flow at runtime (inside the desktop sidecar):
    skill_factory.invoke_skill('memory_pro', subtask_id, input_data)
        → loads run() from the bundled .py
        → @premium wrapper runs first
        → reads input_data['__user_uid__'] (orchestrator injects this)
        → license_check.is_active_license(uid) → True if Lifetime active
            ✓ falls through to the real implementation
            ✗ returns the canonical upgrade message

Skills must accept the upgrade message as a normal output — it's a
plain string the orchestrator surfaces back to the chat. The user
sees "🔒 This is a Pro skill…" with the existing chat-bubble flow,
no special UI hooks needed.
"""

from functools import wraps
from typing import Any, Awaitable, Callable, Dict

from license_check import is_active_license


# Single canonical upgrade message — keep it consistent across every
# premium skill so the user can recognise the lock state at a glance.
# i18n is intentionally NOT applied here; the message is brand-y on
# purpose ("Lifetime", "$99") and translating obscures the call to
# action. The chat UI may render this as markdown, hence the bullet.
_UPGRADE_MESSAGE = (
    "🔒 **Pro skill — upgrade required**\n\n"
    "This skill is part of OpenTeddy Lifetime ($99 one-time). "
    "Tap **✨ Get Lifetime** in the sidebar to unlock Memory Pro, "
    "Data Analyst Pro, and the rest of the premium pack."
)


def premium(skill_fn: Callable[[Dict[str, Any]], Awaitable[str]]):
    """Decorate a skill's run() to gate execution on Lifetime licence.

    Usage:
        from skills._premium import premium

        @premium
        async def run(input_data: dict) -> str:
            ...

    The wrapper reads the current user's Firebase uid from
    `input_data['__user_uid__']` (the orchestrator injects this when
    handing input_data to invoke_skill) and asks license_check whether
    that uid has an active Lifetime subscription.

    NB: the wrapped function is exposed as `wrapper.is_premium = True`
    so the /skills API endpoint + the Tools/Skills UI can render a
    🔒 badge on premium entries even before they're invoked.
    """
    @wraps(skill_fn)
    async def wrapper(input_data: Dict[str, Any]) -> str:
        uid = (input_data or {}).get("__user_uid__") or ""
        if not await is_active_license(uid):
            return _UPGRADE_MESSAGE
        return await skill_fn(input_data)

    # Surface the premium flag for introspection (skill listing UI,
    # /capabilities endpoint, etc.). Not load-bearing for the gate
    # itself, but useful for the frontend's "🔒 Pro" badge logic.
    wrapper.is_premium = True  # type: ignore[attr-defined]
    return wrapper
