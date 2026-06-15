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
    "anthropic_api_key": {
        "label":       "Claude API Key",
        "description": "Anthropic API key. Stored in the settings DB; "
                        "overrides the ANTHROPIC_API_KEY env var when set. "
                        "Get one at console.anthropic.com → Settings → API Keys.",
        "type":        "secret",
    },
    # ── Cloud LLM provider switcher + per-vendor key/model ──────────────
    "llm_provider": {
        "label":       "Cloud LLM Provider",
        "description": "Where to route cloud-side calls (escalation, "
                        "skill generation, prompt optimiser). Pick one: "
                        "'anthropic' (uses the Claude key above), "
                        "'openrouter' (one key for 100+ models), "
                        "'openai' (ChatGPT direct), "
                        "'gemini' (Google direct), "
                        "'deepseek' (cheapest, weaker tool use).",
        "type":        "select",
        "options":     ["anthropic", "openrouter", "openai", "gemini", "deepseek"],
    },
    "openrouter_api_key": {
        "label":       "OpenRouter API Key",
        "description": "openrouter.ai/keys. Only used when Cloud LLM "
                        "Provider is 'openrouter'.",
        "type":        "secret",
    },
    "openrouter_model": {
        "label":       "OpenRouter Model",
        "description": "Model id namespaced by upstream provider, e.g. "
                        "'anthropic/claude-sonnet-4' (recommended), "
                        "'openai/gpt-4o', 'google/gemini-2.0-pro'. Full "
                        "catalogue: openrouter.ai/models.",
        "type":        "text",
    },
    "openai_api_key": {
        "label":       "OpenAI (ChatGPT) API Key",
        "description": "platform.openai.com/api-keys. Only used when "
                        "Cloud LLM Provider is 'openai'.",
        "type":        "secret",
    },
    "openai_model": {
        "label":       "OpenAI Model",
        "description": "Model id, e.g. 'gpt-4o' (recommended), 'gpt-4-turbo', "
                        "'gpt-4o-mini' (cheaper). Full list: "
                        "platform.openai.com/docs/models.",
        "type":        "text",
    },
    "gemini_api_key": {
        "label":       "Gemini API Key",
        "description": "aistudio.google.com/app/apikey. Only used when "
                        "Cloud LLM Provider is 'gemini'.",
        "type":        "secret",
    },
    "gemini_model": {
        "label":       "Gemini Model",
        "description": "Model id, e.g. 'gemini-2.0-flash' (recommended — "
                        "fast + cheap), 'gemini-2.0-pro' (strongest), "
                        "'gemini-1.5-flash-8b' (cheapest).",
        "type":        "text",
    },
    "deepseek_api_key": {
        "label":       "Deepseek API Key",
        "description": "platform.deepseek.com/api_keys. Only used when "
                        "Cloud LLM Provider is 'deepseek'.",
        "type":        "secret",
    },
    "deepseek_model": {
        "label":       "Deepseek Model",
        "description": "Model id, e.g. 'deepseek-chat' (recommended), "
                        "'deepseek-reasoner' (slower but more capable).",
        "type":        "text",
    },
    "brave_search_api_key": {
        "label":       "Brave Search API Key",
        "description": "Powers the web_search tool used in Chat mode so "
                        "the local model can ground answers in current "
                        "data instead of hallucinating recent events / "
                        "version numbers. Free tier: 2,000 queries/month. "
                        "Get one at api-dashboard.search.brave.com. "
                        "Leave empty to disable web search.",
        "type":        "secret",
    },
    "llm_mode": {
        "label":       "LLM Mode",
        "description": "How OpenTeddy routes work between local + cloud "
                       "LLMs. 'local' = fully local (Gemma plans, Qwen "
                       "executes, never touches cloud — privacy / "
                       "offline). 'mixed' = local-first with cloud "
                       "safety-net on failure / low confidence (default; "
                       "fast on easy tasks, robust on hard ones). "
                       "'cloud' = cloud-first; the executor is bypassed "
                       "and every subtask is handled by your configured "
                       "Cloud LLM Provider directly. All three modes "
                       "share the same memory / session / task storage, "
                       "so switching modes never loses past work.",
        "type":        "select",
        "options":     ["local", "mixed", "cloud"],
    },
    "escalation_enabled": {
        "label":       "Allow Claude escalation",
        "description": "When OFF, low-confidence / timeout / failure-signal "
                        "triggers do NOT call Claude — the agent marks the "
                        "subtask FAILED and surfaces the local error. Use "
                        "this to enforce a strictly-local run.",
        "type":        "bool",
    },
    "streaming_enabled": {
        "label":       "Stream model output",
        "description": "Stream LLM tokens to the chat as they generate. "
                        "Big perceived-latency win — the answer appears word "
                        "by word instead of all at once after the model "
                        "finishes. Disable for one-shot responses (legacy).",
        "type":        "bool",
    },
    "verification_enabled": {
        "label":       "Per-step deliverable verification",
        "description": "After each successful subtask, run an extra LLM "
                        "judge call to confirm the produced file actually "
                        "matches the goal (catches the 'wrote a report "
                        "instead of a working game' failure mode). On big "
                        "models this adds 5–60s per step — turn OFF on "
                        "DGX Spark / 35B-class setups for a major speedup.",
        "type":        "bool",
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
    "ollama_keep_alive": {
        "label":       "Ollama model VRAM retention",
        "description": "How long Ollama keeps models loaded in VRAM after a "
                       "request. Default 24h means the first task of the day "
                       "doesn't pay the 5-15 s reload cost. Accepts '24h', "
                       "'1h', '30m', '0' (unload immediately), '-1' (forever). "
                       "Sent as the `keep_alive` field on every Ollama request "
                       "— overrides the daemon's own OLLAMA_KEEP_ALIVE env var.",
        "type":        "text",
    },
    "local_engine": {
        "label":       "Local Inference Engine",
        "description": "Backend serving the local executor model: 'ollama' "
                       "(default, cross-platform) or 'vllm' (Linux/CUDA only, "
                       "better under concurrent fleet load). macOS is always "
                       "treated as 'ollama' regardless of this value.",
        "type":        "text",
    },
    "vllm_base_url": {
        "label":       "vLLM Server URL",
        "description": "Base URL of the vLLM OpenAI-compatible server. Only "
                       "used when local_engine='vllm'. Set up via "
                       "scripts/setup-vllm.sh.",
        "type":        "text",
    },
    "unified_model": {
        "label":       "Unified Model (single-model mode)",
        "description": "Leave blank for the default 2-model split (Gemma "
                       "planner + Qwen executor). Set a single model id (e.g. "
                       "'Qwen/Qwen3.6-27B') to run BOTH planning and execution "
                       "on one model — fewer round-trips, one resident model, "
                       "and on vLLM one served instance. Must match the model "
                       "vLLM was launched serving.",
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
    "gemma_temperature": {
        "label":       "Gemma Temperature",
        "description": "Sampling temperature for the orchestrator planner. "
                       "Lower = more deterministic plans.",
        "type":        "float",
        "min":         0.0,
        "max":         1.0,
        "step":        0.05,
    },
    "qwen_temperature": {
        "label":       "Qwen Temperature",
        "description": "Sampling temperature for the executor. Set to 0.0 "
                       "for highly deterministic tool-use (recommended "
                       "for deploy / analytic flows).",
        "type":        "float",
        "min":         0.0,
        "max":         1.0,
        "step":        0.05,
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
    "intent_classifier_enabled": {
        "label":       "Intent classifier (fast-path for chat questions)",
        "description": "Before running the full plan → execute → summary "
                       "loop on a goal, classify it via a fast LLM call. "
                       "If the goal clearly doesn't need tools ('what is "
                       "X', 'explain Y'), bypass the tool loop entirely "
                       "and answer in one turn — typically 25 seconds "
                       "faster for chat-style questions. Turn off if you "
                       "see the classifier mis-routing goals that DO need "
                       "tools, or just want every task to follow the same "
                       "path for predictable timing.",
        "type":        "bool",
    },
    "skill_auto_detect_min_repeats": {
        "label":       "Skill auto-detect — min repeats",
        "description": "After a successful task, scan past task memories "
                       "for semantically-similar goals. When this many "
                       "(or more) similar goals are found, synthesise a "
                       "skill and hand it to SkillFactory. Set to 0 to "
                       "disable auto-detection entirely (skills can "
                       "still be created manually via the API).",
        "type":        "int",
        "min":         0,
        "max":         20,
    },
    "skill_auto_detect_similarity": {
        "label":       "Skill auto-detect — similarity threshold",
        "description": "Cosine-similarity floor (0.0-1.0) for counting "
                       "a past task as 'recurring'. Higher = stricter / "
                       "fewer false-positive skills. 0.85 is a sensible "
                       "default; bump to 0.9 if you see weird skills "
                       "getting generated, drop to 0.75 if expected "
                       "patterns aren't being caught.",
        "type":        "float",
        "min":         0.0,
        "max":         1.0,
        "step":        0.05,
    },
    "approval_auto_approve_after": {
        "label":       "Auto-approve after (seconds)",
        "description": "When > 0, HIGH-risk tool approval prompts "
                       "(shell_exec_write, delete_file, etc.) auto-approve "
                       "if you don't click within this many seconds. 0 = "
                       "disabled (the safer default — wait for explicit "
                       "click, eventually reject if Approval wait timeout "
                       "expires). Use 15–60 s for routine workflows; keep "
                       "at 0 when you want explicit oversight on every "
                       "destructive action.",
        "type":        "int",
        "min":         0,
        "max":         300,
    },
    "approval_wait_timeout": {
        "label":       "Approval wait timeout (seconds)",
        "description": "How long an unanswered HIGH-risk approval sits "
                       "before flipping to REJECTED. Was 300 s (5 min) "
                       "originally — bumped to 1800 s (30 min) default so "
                       "you can grab lunch without losing tasks. Range 60 "
                       "to 7200 s. Doesn't apply when Auto-approve after "
                       "is set (that timeout wins).",
        "type":        "int",
        "min":         60,
        "max":         7200,
    },
    "default_priority": {
        "label":       "Default Task Priority",
        "description": "Reserved for a future priority-aware scheduler. The "
                       "value is written to the tasks.priority column on every "
                       "/run, but NO current code path reads it back to reorder "
                       "execution — tasks fire as a plain asyncio.create_task "
                       "in arrival order. Kept in the API surface so existing "
                       "integrations that already send 'priority' in their "
                       "request body don't break, and so the column is "
                       "populated when the real scheduler ships. Leave at 1 "
                       "unless you're integrating with an external tool that "
                       "reads this column directly.",
        "type":        "int",
        "min":         1,
        "max":         10,
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
    "session_workspace_isolation": {
        "label":       "Per-session workspace isolation",
        "description": "When ON (default), every NEW session gets its "
                       "own subdirectory under "
                       "{agent_workspace_dir}/sessions/<id>/ so files "
                       "produced by one session don't leak into another. "
                       "OFF reverts to the legacy shared-root behaviour "
                       "(useful when you're iterating on one long-running "
                       "project across many sessions). Existing sessions "
                       "are never auto-migrated either way.",
        "type":        "bool",
    },

    # ── Notification credentials (Slice 1: agent-as-service) ──────────────
    "telegram_bot_token": {
        "label":       "Telegram Bot Token",
        "description": "Bot token from @BotFather. Used by the telegram_send tool.",
        "type":        "secret",  # UI masks the value
    },
    "telegram_default_chat_id": {
        "label":       "Telegram Default Chat ID",
        "description": "Optional. When set, telegram_send can omit chat_id "
                       "and send here by default.",
        "type":        "text",
    },
    # Inbound bridge (Telegram → OpenTeddy). When enabled + whitelist
    # populated, telegram_bridge.py polls getUpdates and routes messages
    # from whitelisted chats to a per-chat persistent session.
    "telegram_inbound_enabled": {
        "label":       "Enable Telegram inbound",
        "description": "When ON (and a whitelist is set), OpenTeddy polls "
                       "Telegram for messages from your whitelisted chats and "
                       "runs them as goals. The reply is pushed back to the "
                       "same chat. Needs the server to stay running 24/7.",
        "type":        "bool",
    },
    "telegram_inbound_chat_id_whitelist": {
        "label":       "Inbound chat-ID whitelist",
        "description": "Comma-separated chat_ids that are allowed to drive "
                       "OpenTeddy. Empty = inbound disabled (we refuse to "
                       "listen on an open bot). Find your chat_id by sending "
                       "any message to @userinfobot on Telegram.",
        "type":        "text",
    },
    "smtp_host": {
        "label":       "SMTP Host",
        "description": "e.g. smtp.gmail.com, smtp.sendgrid.net",
        "type":        "text",
    },
    "smtp_port": {
        "label":       "SMTP Port",
        "description": "Typically 587 (STARTTLS) or 465 (SSL)",
        "type":        "int",
        "min":         1,
        "max":         65535,
    },
    "smtp_user": {
        "label":       "SMTP User",
        "description": "Usually your email address or API key user.",
        "type":        "text",
    },
    "smtp_password": {
        "label":       "SMTP Password",
        "description": "SMTP password or app-specific password.",
        "type":        "secret",
    },
    "smtp_from": {
        "label":       "SMTP From Address",
        "description": "Sender address. Defaults to SMTP User when empty.",
        "type":        "text",
    },
    "webhook_secret": {
        "label":       "Webhook Secret",
        "description": "If set, POST /webhooks/{session_id} requires this as "
                       "an X-OpenTeddy-Webhook-Secret header or ?secret= query "
                       "param. Empty = webhook endpoint is OPEN to anyone on "
                       "the network (NOT recommended for public servers).",
        "type":        "secret",
    },
}


def _defaults_from_config() -> dict[str, str]:
    """Snapshot current config values for DB initialisation."""
    return {
        "orchestrator_model":       config.gemma_model,
        "executor_model":           config.qwen_model,
        "claude_model":             config.claude_model,
        "anthropic_api_key":        config.anthropic_api_key,
        "llm_provider":             getattr(config, "llm_provider", "anthropic"),
        "openrouter_api_key":       getattr(config, "openrouter_api_key", ""),
        "openrouter_model":         getattr(config, "openrouter_model", "anthropic/claude-sonnet-4"),
        "openai_api_key":           getattr(config, "openai_api_key", ""),
        "openai_model":             getattr(config, "openai_model", "gpt-4o"),
        "gemini_api_key":           getattr(config, "gemini_api_key", ""),
        "gemini_model":             getattr(config, "gemini_model", "gemini-2.0-flash"),
        "deepseek_api_key":         getattr(config, "deepseek_api_key", ""),
        "deepseek_model":           getattr(config, "deepseek_model", "deepseek-chat"),
        "brave_search_api_key":     getattr(config, "brave_search_api_key", ""),
        # New 3-way LLM mode + legacy bool kept in sync for backward compat
        "llm_mode":                 getattr(config, "llm_mode", "mixed"),
        "escalation_enabled":       "true" if config.escalation_enabled else "false",
        "streaming_enabled":        "true" if config.streaming_enabled else "false",
        "verification_enabled":     "true" if getattr(config, "verification_enabled", True) else "false",
        "gemma_base_url":           config.gemma_base_url,
        "qwen_base_url":            config.qwen_base_url,
        "ollama_keep_alive":        getattr(config, "ollama_keep_alive", "24h"),
        "local_engine":             getattr(config, "local_engine", "ollama"),
        "vllm_base_url":            getattr(config, "vllm_base_url", "http://127.0.0.1:8001"),
        "unified_model":            getattr(config, "unified_model", ""),
        "escalation_threshold":     str(config.escalation_confidence_threshold),
        "escalation_failure_limit": str(config.escalation_failure_limit),
        "gemma_max_tokens":         str(config.gemma_max_tokens),
        "qwen_max_tokens":          str(config.qwen_max_tokens),
        "skill_match_threshold":    str(getattr(config, "skill_match_threshold", "0.4")),
        "skill_promotion_threshold": str(config.skill_promotion_threshold),
        "intent_classifier_enabled":   "true" if getattr(config, "intent_classifier_enabled", True) else "false",
        "skill_auto_detect_min_repeats": str(getattr(config, "skill_auto_detect_min_repeats", 3)),
        "skill_auto_detect_similarity":  str(getattr(config, "skill_auto_detect_similarity", 0.85)),
        "default_priority":          str(getattr(config, "default_priority", 1)),
        "approval_auto_approve_after": str(getattr(config, "approval_auto_approve_after", 0)),
        "approval_wait_timeout":       str(getattr(config, "approval_wait_timeout", 1800)),
        "subtask_timeout":           str(config.subtask_timeout),
        "shell_silence_timeout":     str(config.shell_silence_timeout),
        "agent_workspace_dir":       str(config.agent_workspace_dir),
        "session_workspace_isolation": "true" if getattr(config, "session_workspace_isolation", True) else "false",
        "gemma_temperature":         str(config.gemma_temperature),
        "qwen_temperature":          str(config.qwen_temperature),
        "telegram_bot_token":        str(config.telegram_bot_token),
        "telegram_default_chat_id":  str(config.telegram_default_chat_id),
        "telegram_inbound_enabled":  "true" if getattr(config, "telegram_inbound_enabled", False) else "false",
        "telegram_inbound_chat_id_whitelist": str(getattr(config, "telegram_inbound_chat_id_whitelist", "")),
        "smtp_host":                 str(config.smtp_host),
        "smtp_port":                 str(config.smtp_port),
        "smtp_user":                 str(config.smtp_user),
        "smtp_password":             str(config.smtp_password),
        "smtp_from":                 str(config.smtp_from),
        "webhook_secret":            str(config.webhook_secret),
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

            # One-shot migration: legacy escalation_enabled → llm_mode.
            # When an upgrade lands the new 3-way setting, existing
            # users have `escalation_enabled` stored but no `llm_mode`
            # yet — the INSERT above just wrote the default 'mixed'.
            # If their explicit choice was OFF (= local-only), we don't
            # want to silently flip them to 'mixed'. So: if llm_mode is
            # still at the freshly-inserted default AND escalation was
            # explicitly OFF, promote to 'local'. Same for 'mixed'.
            await self._migrate_escalation_to_llm_mode(db)

        logger.info("SettingsStore ready (%d default keys).", len(defaults))

    async def _migrate_escalation_to_llm_mode(self, db: aiosqlite.Connection) -> None:
        """Promote legacy `escalation_enabled` to the new `llm_mode` field.

        Detection rule: if the user already had an `escalation_enabled`
        row (= they upgraded from a pre-llm_mode build) AND the freshly
        inserted llm_mode is still at the default 'mixed' AND
        escalation_enabled was explicitly OFF → flip llm_mode to 'local'
        so the user's "no cloud calls" preference is preserved.

        We DON'T flip anyone to 'cloud' automatically — that's an
        explicit opt-in via the new UI, since it changes billing.
        """
        # Skip cleanly when the rows aren't there for any reason.
        try:
            async with db.execute(
                "SELECT value FROM app_settings WHERE key='escalation_enabled'"
            ) as cur:
                esc_row = await cur.fetchone()
            async with db.execute(
                "SELECT value FROM app_settings WHERE key='llm_mode'"
            ) as cur:
                mode_row = await cur.fetchone()
        except Exception:  # noqa: BLE001
            return

        if not esc_row or not mode_row:
            return

        esc_off = str(esc_row[0]).strip().lower() in {"0", "false", "no", "off", ""}
        current_mode = str(mode_row[0]).strip().lower()
        if esc_off and current_mode == "mixed":
            await db.execute(
                "UPDATE app_settings SET value=?, updated_at=? WHERE key='llm_mode'",
                ("local", _now()),
            )
            await db.commit()
            logger.info(
                "Migrated legacy escalation_enabled=off → llm_mode='local' "
                "(preserves user's no-cloud preference)."
            )

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
        if settings.get("anthropic_api_key"):
            # Empty string ⇒ keep the env-var fallback. Only overwrite
            # when the user actually saved a key in the UI.
            config.anthropic_api_key = settings["anthropic_api_key"]
        # Cloud LLM provider trio for each of the four OpenAI-compat
        # vendors. Same overwrite-only-on-truthy pattern as Anthropic
        # so empty DB rows don't wipe env-var fallbacks.
        if "llm_provider" in settings and settings["llm_provider"]:
            config.llm_provider = settings["llm_provider"].lower()
        if settings.get("openrouter_api_key"):
            config.openrouter_api_key = settings["openrouter_api_key"]
        if "openrouter_model" in settings and settings["openrouter_model"]:
            config.openrouter_model = settings["openrouter_model"]
        if settings.get("openai_api_key"):
            config.openai_api_key = settings["openai_api_key"]
        if "openai_model" in settings and settings["openai_model"]:
            config.openai_model = settings["openai_model"]
        if settings.get("gemini_api_key"):
            config.gemini_api_key = settings["gemini_api_key"]
        if "gemini_model" in settings and settings["gemini_model"]:
            config.gemini_model = settings["gemini_model"]
        if settings.get("deepseek_api_key"):
            config.deepseek_api_key = settings["deepseek_api_key"]
        if "deepseek_model" in settings and settings["deepseek_model"]:
            config.deepseek_model = settings["deepseek_model"]
        if settings.get("brave_search_api_key"):
            # Same pattern as Anthropic — empty in UI = keep env-var.
            config.brave_search_api_key = settings["brave_search_api_key"]
        # ── 3-way LLM mode (canonical) ────────────────────────────────
        # Priority: explicit llm_mode wins, else derive from legacy
        # escalation_enabled (for users on a pre-llm_mode DB row), else
        # leave at env default.
        raw_mode = (settings.get("llm_mode") or "").strip().lower()
        if raw_mode in {"local", "mixed", "cloud"}:
            config.llm_mode = raw_mode
        elif "escalation_enabled" in settings:
            on = (str(settings["escalation_enabled"]).strip().lower()
                  not in {"0", "false", "no", "off", ""})
            config.llm_mode = "mixed" if on else "local"

        # Legacy bool is now a PURE derivative of llm_mode with the
        # semantics "may escalation.resolve() actually run?":
        #   local → No (privacy guardrail)
        #   mixed → Yes (it's the failure fallback)
        #   cloud → Yes (it's the PRIMARY execution path)
        # The "only mixed = True" derivation I had before was wrong —
        # it caused cloud mode to be blocked by _escalation_blocked()
        # which reads this bool. Keep the two fields in lock-step.
        config.escalation_enabled = (config.llm_mode != "local")
        if "streaming_enabled" in settings:
            config.streaming_enabled = (
                str(settings["streaming_enabled"]).strip().lower()
                not in {"0", "false", "no", "off", ""}
            )
        if "verification_enabled" in settings:
            config.verification_enabled = (
                str(settings["verification_enabled"]).strip().lower()
                not in {"0", "false", "no", "off", ""}
            )
        if "gemma_base_url" in settings:
            config.gemma_base_url = settings["gemma_base_url"]
        if "qwen_base_url" in settings:
            config.qwen_base_url = settings["qwen_base_url"]
        if "local_engine" in settings and settings["local_engine"]:
            config.local_engine = settings["local_engine"].strip().lower()
        if "vllm_base_url" in settings and settings["vllm_base_url"]:
            config.vllm_base_url = settings["vllm_base_url"].strip()
        if "unified_model" in settings:
            # Applied even when blank: clearing the field is how a user
            # returns to the default 2-model split, so we must NOT skip
            # empties the way the other text settings do.
            config.unified_model = (settings["unified_model"] or "").strip()
        if "ollama_keep_alive" in settings and settings["ollama_keep_alive"]:
            # Non-empty only — clearing the field via the UI keeps the
            # 24h default rather than disabling keep_alive entirely
            # (which would force a cold reload on every request).
            config.ollama_keep_alive = settings["ollama_keep_alive"].strip()

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

        if "intent_classifier_enabled" in settings:
            config.intent_classifier_enabled = (
                str(settings["intent_classifier_enabled"]).strip().lower()
                not in {"0", "false", "no", "off", ""}
            )

        vsd = _int("skill_auto_detect_min_repeats")
        if vsd is not None:
            config.skill_auto_detect_min_repeats = max(0, vsd)
        vss = _float("skill_auto_detect_similarity")
        if vss is not None:
            config.skill_auto_detect_similarity = max(0.0, min(1.0, vss))

        vdp = _int("default_priority")
        if vdp is not None:
            # Clamp to the Pydantic range so a hand-edited DB row can't
            # produce a 422 on every /run call.
            config.default_priority = max(1, min(10, vdp))

        vaa = _int("approval_auto_approve_after")
        if vaa is not None:
            config.approval_auto_approve_after = max(0, min(300, vaa))

        vwt = _int("approval_wait_timeout")
        if vwt is not None:
            config.approval_wait_timeout = max(60, min(7200, vwt))

        v7 = _int("subtask_timeout")
        if v7 is not None:
            config.subtask_timeout = v7

        v8 = _int("shell_silence_timeout")
        if v8 is not None:
            config.shell_silence_timeout = v8

        vgt = _float("gemma_temperature")
        if vgt is not None:
            config.gemma_temperature = vgt
        vqt = _float("qwen_temperature")
        if vqt is not None:
            config.qwen_temperature = vqt

        if "agent_workspace_dir" in settings and settings["agent_workspace_dir"]:
            # Resolve to absolute once here — see config.py for the rationale.
            import os as _os
            config.agent_workspace_dir = _os.path.abspath(settings["agent_workspace_dir"])

        if "session_workspace_isolation" in settings:
            config.session_workspace_isolation = (
                str(settings["session_workspace_isolation"]).strip().lower()
                not in {"0", "false", "no", "off", ""}
            )

        # Notification credentials. All strings; blanks are meaningful
        # (= "not configured", tools will return a clear error).
        for k in ("telegram_bot_token", "telegram_default_chat_id",
                  "telegram_inbound_chat_id_whitelist",
                  "smtp_host", "smtp_user", "smtp_password", "smtp_from",
                  "webhook_secret"):
            if k in settings:
                setattr(config, k, settings[k] or "")
        if "telegram_inbound_enabled" in settings:
            config.telegram_inbound_enabled = (
                str(settings["telegram_inbound_enabled"]).strip().lower()
                in {"1", "true", "yes", "on"}
            )
        port_val = _int("smtp_port")
        if port_val is not None:
            config.smtp_port = port_val

        logger.info("Config updated from SettingsStore.")


# Module-level singleton (db_path shared with tracker)
settings_store = SettingsStore(db_path=config.db_path)
