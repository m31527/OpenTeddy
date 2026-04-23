"""
OpenTeddy Settings Store
SQLite-based persistent settings store.

On first run, missing keys are initialised from config (which reads .env).
Subsequent restarts load from DB — .env is only a fallback for brand-new installs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import aiosqlite

from config import config

logger = logging.getLogger("openteddy.settings")


# ── Setting metadata ──────────────────────────────────────────────────────────
# Drives both the DB defaults and the /settings API response.
SETTINGS_META: dict[str, dict[str, Any]] = {
    "orchestrator_model": {
        "label":       "Orchestrator Model (Gemma)",
        "description": "Local Ollama model used as the orchestrator / planner",
        "type":        "model_select",
    },
    "executor_model": {
        "label":       "Executor Model (Qwen)",
        "description": "Local Ollama model used for task execution",
        "type":        "model_select",
    },
    "claude_model": {
        "label":       "Claude Model (Escalation)",
        "description": "Anthropic Claude model for escalation and skill generation",
        "type":        "text",
    },
    "gemma_base_url": {
        "label":       "Orchestrator Ollama URL",
        "description": "Base URL of the Ollama instance serving the orchestrator",
        "type":        "text",
    },
    "qwen_base_url": {
        "label":       "Executor Ollama URL",
        "description": "Base URL of the Ollama instance serving the executor",
        "type":        "text",
    },
    "escalation_threshold": {
        "label":       "Escalation Confidence Threshold",
        "description": "Escalate to Claude when executor confidence is below this value",
        "type":        "float",
        "min":         0.0,
        "max":         1.0,
        "step":        0.05,
    },
    "escalation_failure_limit": {
        "label":       "Escalation Failure Limit",
        "description": "Consecutive failures before escalating to Claude",
        "type":        "int",
        "min":         1,
        "max":         10,
    },
    "gemma_max_tokens": {
        "label":       "Gemma Max Tokens",
        "description": "Maximum output tokens for the Gemma orchestrator",
        "type":        "int",
        "min":         512,
        "max":         8192,
    },
    "qwen_max_tokens": {
        "label":       "Qwen Max Tokens",
        "description": "Maximum output tokens for the Qwen executor",
        "type":        "int",
        "min":         512,
        "max":         8192,
    },
    "skill_match_threshold": {
        "label":       "Skill Match Threshold",
        "description": "Minimum similarity score to reuse an existing skill",
        "type":        "float",
        "min":         0.0,
        "max":         1.0,
        "step":        0.05,
    },
    "skill_promotion_threshold": {
        "label":       "Skill Promotion Threshold",
        "description": "Successful runs required before a skill is promoted to ACTIVE",
        "type":        "int",
        "min":         1,
        "max":         50,
    },
    "subtask_timeout": {
        "label":       "Subtask Timeout (seconds)",
        "description": "Wall-clock hard-stop for a subtask. The real safety net "
                       "is shell_silence_timeout below; this just caps runaway tasks.",
        "type":        "int",
        "min":         60,
        "max":         3600,
    },
    "shell_silence_timeout": {
        "label":       "Shell Silence Timeout (seconds)",
        "description": "Kill a shell command after this many seconds of no output. "
                       "Long builds (docker, pip) stay alive as long as they print progress; "
                       "truly hung commands are caught in this window. Set 0 to disable.",
        "type":        "int",
        "min":         0,
        "max":         600,
    },
    "agent_workspace_dir": {
        "label":       "Agent Workspace Directory",
        "description": "Default working directory for shell commands in Code / Analytic modes. "
                       "Relative paths resolve against the project root.",
        "type":        "text",
    },
}


def _defaults_from_config() -> dict[str, str]:
    """Snapshot current config values for DB initialisation."""
    return {
        "orchestrator_model":       config.gemma_model,
        "executor_model":           config.qwen_model,
        "claude_model":             config.claude_model,
        "gemma_base_url":           config.gemma_base_url,
        "qwen_base_url":            config.qwen_base_url,
        "escalation_threshold":     str(config.escalation_confidence_threshold),
        "escalation_failure_limit": str(config.escalation_failure_limit),
        "gemma_max_tokens":         str(config.gemma_max_tokens),
        "qwen_max_tokens":          str(config.qwen_max_tokens),
        "skill_match_threshold":    str(getattr(config, "skill_match_threshold", "0.4")),
        "skill_promotion_threshold": str(config.skill_promotion_threshold),
        "subtask_timeout":           str(config.subtask_timeout),
        "shell_silence_timeout":     str(config.shell_silence_timeout),
        "agent_workspace_dir":       str(config.agent_workspace_dir),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SettingsStore:
    """Async SQLite-backed key-value settings store."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def init(self) -> None:
        """Create the table and populate any missing keys from current config."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS app_settings (
                    key        TEXT PRIMARY KEY,
                    value      TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            await db.commit()

            defaults = _defaults_from_config()
            for key, value in defaults.items():
                await db.execute(
                    """
                    INSERT INTO app_settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO NOTHING
                    """,
                    (key, value, _now()),
                )
            await db.commit()

            # Auto-migrate stale defaults. Without this, users upgrading
            # from older versions kept the old subtask_timeout=120 (or
            # 180) in their DB, and long docker builds would get killed
            # at 120s while the code thinks the default is 900s. We only
            # migrate values that EXACTLY match a past shipped default
            # so we never override a user's intentional choice.
            await self._migrate_stale_defaults(db)

        logger.info("SettingsStore ready (%d default keys).", len(defaults))

    async def _migrate_stale_defaults(self, db: aiosqlite.Connection) -> None:
        """Replace pre-upgrade default values with the current ones.

        Each entry lists (key, *known_old_defaults) — if the stored value
        matches ANY of them, it's treated as "user never changed it from
        the old default" and bumped to the current code default. If the
        stored value is outside that allowlist, we leave it alone (user
        may have tuned it intentionally).
        """
        stale_matches: list[tuple[str, tuple[str, ...]]] = [
            # subtask_timeout: shipped 120 → 180 → 900
            ("subtask_timeout", ("120", "180")),
            # shell_silence_timeout: added later at 90 — if someone has
            # an exactly-default stored but we lower the default, we'd
            # want to not touch it. Currently 90 is the default, no
            # migration needed here yet; placeholder for future.
        ]
        defaults = _defaults_from_config()
        for key, old_values in stale_matches:
            target = defaults.get(key)
            if not target:
                continue
            async with db.execute(
                "SELECT value FROM app_settings WHERE key=?", (key,)
            ) as cur:
                row = await cur.fetchone()
            if not row:
                continue
            current = row[0]
            if current in old_values and current != target:
                await db.execute(
                    "UPDATE app_settings SET value=?, updated_at=? WHERE key=?",
                    (target, _now(), key),
                )
                logger.info(
                    "Migrated stale default for '%s': %s → %s (was an old "
                    "shipped default; current code default is higher)",
                    key, current, target,
                )
        await db.commit()

    # ── CRUD ───────────────────────────────────────────────────────────────────

    async def get(self, key: str) -> str | None:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT value FROM app_settings WHERE key = ?", (key,)
            ) as cur:
                row = await cur.fetchone()
        return row[0] if row else None

    async def set(self, key: str, value: str) -> None:
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE
                    SET value      = excluded.value,
                        updated_at = excluded.updated_at
                """,
                (key, str(value), _now()),
            )
            await db.commit()

    async def get_all(self) -> dict[str, str]:
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT key, value FROM app_settings"
            ) as cur:
                rows = await cur.fetchall()
        return {r[0]: r[1] for r in rows}

    async def update_many(self, updates: dict[str, str]) -> None:
        ts = _now()
        async with aiosqlite.connect(self._db_path) as db:
            for key, value in updates.items():
                await db.execute(
                    """
                    INSERT INTO app_settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE
                        SET value      = excluded.value,
                            updated_at = excluded.updated_at
                    """,
                    (key, str(value), ts),
                )
            await db.commit()
        logger.info("SettingsStore: updated %d keys.", len(updates))

    # ── Config sync ────────────────────────────────────────────────────────────

    async def apply_to_config(self) -> None:
        """Push DB values into the live config singleton — no server restart needed."""
        settings = await self.get_all()

        def _float(k: str) -> float | None:
            try:
                return float(settings[k])
            except (KeyError, ValueError, TypeError):
                return None

        def _int(k: str) -> int | None:
            try:
                return int(settings[k])
            except (KeyError, ValueError, TypeError):
                return None

        if "orchestrator_model" in settings:
            config.gemma_model = settings["orchestrator_model"]
        if "executor_model" in settings:
            config.qwen_model = settings["executor_model"]
        if "claude_model" in settings:
            config.claude_model = settings["claude_model"]
        if "gemma_base_url" in settings:
            config.gemma_base_url = settings["gemma_base_url"]
        if "qwen_base_url" in settings:
            config.qwen_base_url = settings["qwen_base_url"]

        v = _float("escalation_threshold")
        if v is not None:
            config.escalation_confidence_threshold = v

        v2 = _int("escalation_failure_limit")
        if v2 is not None:
            config.escalation_failure_limit = v2

        v3 = _int("gemma_max_tokens")
        if v3 is not None:
            config.gemma_max_tokens = v3

        v4 = _int("qwen_max_tokens")
        if v4 is not None:
            config.qwen_max_tokens = v4

        v5 = _float("skill_match_threshold")
        if v5 is not None:
            try:
                config.skill_match_threshold = v5  # type: ignore[attr-defined]
            except AttributeError:
                pass

        v6 = _int("skill_promotion_threshold")
        if v6 is not None:
            config.skill_promotion_threshold = v6

        v7 = _int("subtask_timeout")
        if v7 is not None:
            config.subtask_timeout = v7

        v8 = _int("shell_silence_timeout")
        if v8 is not None:
            config.shell_silence_timeout = v8

        if "agent_workspace_dir" in settings and settings["agent_workspace_dir"]:
            # Resolve to absolute once here — see config.py for the rationale.
            import os as _os
            config.agent_workspace_dir = _os.path.abspath(settings["agent_workspace_dir"])

        logger.info("Config updated from SettingsStore.")


# Module-level singleton (db_path shared with tracker)
settings_store = SettingsStore(db_path=config.db_path)
