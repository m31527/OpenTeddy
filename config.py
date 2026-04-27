"""
OpenTeddy Configuration
Central configuration management using environment variables.
"""

import contextvars
import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# ── Session-scoped workspace override ─────────────────────────────────────────
# A Code-mode session can point at a specific directory (e.g. the user's
# real project checkout) instead of the shared agent-workspace sandbox.
# The orchestrator sets this ContextVar at the start of each task; every
# shell / deploy tool reads it via `effective_workspace_dir()` and falls
# back to the global default when unset. ContextVars are task-local in
# asyncio, so concurrent sessions don't stomp on each other.
_session_workspace_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "openteddy_session_workspace", default=None,
)

# Privacy guardrail. When the orchestrator starts a task for a
# local-only session, it sets this to True and every code path that
# might dispatch work to Claude checks it via `is_session_local_only()`.
_session_local_only_var: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "openteddy_session_local_only", default=False,
)


def set_session_workspace(path: Optional[str]) -> None:
    """Orchestrator calls this at the start of a task. Pass None to clear."""
    _session_workspace_var.set(path or None)


def set_session_local_only(flag: bool) -> None:
    """Orchestrator calls this at the start of a task so downstream
    code paths (auto-escalation, skill factory Claude calls) can
    cheaply check the guardrail without re-fetching the session row.
    """
    _session_local_only_var.set(bool(flag))


def is_session_local_only() -> bool:
    """Returns True when the current async task is running under a
    session that has opted out of all Anthropic API calls."""
    return bool(_session_local_only_var.get())


def effective_workspace_dir() -> str:
    """Return the workspace the current async task should use.

    Priority:
      1. Session override (set via ``set_session_workspace``)
      2. Global ``config.agent_workspace_dir`` (absolute after init)
    Always returns an absolute path.
    """
    override = _session_workspace_var.get()
    if override:
        return override if os.path.isabs(override) else os.path.abspath(override)
    return config.agent_workspace_dir

# Resolve the project root from THIS file's location, not from the uvicorn
# process cwd. Users launching `uvicorn main:app` from anywhere (home dir,
# /tmp, a venv bin/) must still get the same workspace, otherwise git clones
# scatter across the filesystem depending on where you happened to `cd` to
# before starting the server.
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_default_workspace() -> str:
    """Pick the absolute default for AGENT_WORKSPACE_DIR.

    Priority:
      1. Env var AGENT_WORKSPACE_DIR (absolute wins, relative joined to
         project root — so `./foo` resolves to <OpenTeddy>/foo, never
         to whatever dir you launched uvicorn from).
      2. Project root's ``agent-workspace/`` subdir.
    """
    raw = os.getenv("AGENT_WORKSPACE_DIR")
    if raw:
        return raw if os.path.isabs(raw) else os.path.abspath(os.path.join(_PROJECT_ROOT, raw))
    return os.path.join(_PROJECT_ROOT, "agent-workspace")


@dataclass
class Config:
    # ── Model endpoints ──────────────────────────────────────────────────────
    # Gemma 4 Orchestrator (local Ollama or remote)
    gemma_base_url: str = field(
        default_factory=lambda: os.getenv("GEMMA_BASE_URL", "http://localhost:11434")
    )
    gemma_model: str = field(
        default_factory=lambda: os.getenv("GEMMA_MODEL", "gemma3:4b")
    )

    # Qwen 3 Executor (local Ollama or remote)
    qwen_base_url: str = field(
        default_factory=lambda: os.getenv("QWEN_BASE_URL", "http://localhost:11434")
    )
    qwen_model: str = field(
        default_factory=lambda: os.getenv("QWEN_MODEL", "qwen2.5:3b")
    )

    # Claude upgrade / escalation
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    claude_model: str = field(
        default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-opus-4-6")
    )
    # Master kill-switch for Claude escalation. When False, low-confidence
    # / timeout / failure-signal triggers do NOT call Claude — the agent
    # marks the subtask FAILED and surfaces the local error to the user.
    # Per-session "Local-only" mode still wins when set, but this flips
    # the global default. Default True so existing behaviour is preserved.
    escalation_enabled: bool = field(
        default_factory=lambda: os.getenv("ESCALATION_ENABLED", "true").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    # Stream LLM tokens to the chat as they generate. Massive perceived
    # latency win on small thinking models — the user sees the answer
    # forming instead of staring at a spinner. When OFF, the server
    # waits for the full response and returns it in one shot (legacy).
    streaming_enabled: bool = field(
        default_factory=lambda: os.getenv("STREAMING_ENABLED", "true").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    # Ollama "num_ctx" — how many tokens of input the model is willing
    # to read each turn. Default 16K is a sweet spot for thinking models
    # (qwen3.5 / gemma4): big enough to hold ~10 tool rounds, small
    # enough that Ollama doesn't fall back to CPU layers on a 16 GB Mac.
    # When prompt_eval_count approaches this, the executor compresses
    # older turns (#4 Context watchdog).
    qwen_num_ctx: int = field(
        default_factory=lambda: int(os.getenv("QWEN_NUM_CTX", "16384"))
    )
    gemma_num_ctx: int = field(
        default_factory=lambda: int(os.getenv("GEMMA_NUM_CTX", "16384"))
    )
    # Trigger compression when the most recent prompt_eval_count crosses
    # this fraction of num_ctx. 0.7 leaves 30% headroom for the next
    # tool result + thinking tokens before the model truncates input.
    context_compress_at: float = field(
        default_factory=lambda: float(os.getenv("CONTEXT_COMPRESS_AT", "0.7"))
    )

    # ── Database ─────────────────────────────────────────────────────────────
    db_path: str = field(
        default_factory=lambda: os.getenv("DB_PATH", "openteddy.db")
    )

    # ── Memory (ChromaDB) ────────────────────────────────────────────────────
    memory_db_path: str = field(
        default_factory=lambda: os.getenv("MEMORY_DB_PATH", "./memory_db")
    )

    # ── Skill Factory ────────────────────────────────────────────────────────
    skills_dir: str = field(
        default_factory=lambda: os.getenv("SKILLS_DIR", "skills")
    )

    # ── Notification credentials (Slice 1: agent-as-service) ────────────────
    # Used by tools/notify_tool.py. All blank by default — the tools
    # return a "credential not configured" error pointing at Settings
    # if the agent tries to use them before the user fills these in.
    telegram_bot_token: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", "")
    )
    telegram_default_chat_id: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_DEFAULT_CHAT_ID", "")
    )
    smtp_host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", ""))
    smtp_port: int = field(
        default_factory=lambda: int(os.getenv("SMTP_PORT", "587") or "587")
    )
    smtp_user: str = field(default_factory=lambda: os.getenv("SMTP_USER", ""))
    smtp_password: str = field(default_factory=lambda: os.getenv("SMTP_PASSWORD", ""))
    smtp_from: str = field(default_factory=lambda: os.getenv("SMTP_FROM", ""))

    # Webhook shared-secret. If set, POST /webhooks/{session_id} requires
    # either `X-OpenTeddy-Webhook-Secret: <this>` header or ?secret=<this>
    # query param. Empty = endpoint is OPEN (anyone on the network can
    # trigger); the UI surfaces a clear warning in that case.
    webhook_secret: str = field(
        default_factory=lambda: os.getenv("WEBHOOK_SECRET", "")
    )

    # ── Agent workspace ──────────────────────────────────────────────────────
    # Default working directory for shell_exec_* in Code / Analytic modes.
    # Any command that doesn't specify its own working_dir lands here, so
    # `git clone`, `pip install -t .`, file writes etc. don't scatter across
    # the host.
    #
    # IMPORTANT: resolved via ``_resolve_default_workspace`` so the path
    # is ALWAYS absolute AND anchored to the OpenTeddy project root (the
    # directory containing this config.py). Running `uvicorn main:app`
    # from your home dir, /tmp, or a cron script — they all produce the
    # same workspace. Before this change, `./agent-workspace` resolved
    # against the uvicorn process cwd, which caused `git clone` to land
    # in one place while `docker_project_detect` looked in another.
    agent_workspace_dir: str = field(default_factory=_resolve_default_workspace)

    # ── Token limits ────────────────────────────────────────────────────────
    gemma_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("GEMMA_MAX_TOKENS", "4096"))
    )
    qwen_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("QWEN_MAX_TOKENS", "4096"))
    )

    # ── Sampling temperatures ───────────────────────────────────────────────
    # Lower = more deterministic (less "let me try a different path"
    # randomness). For tool-use-heavy tasks like deploys, 0.0–0.1 is a
    # reasonable floor — reduces the chance the executor decides to
    # re-run an already-done step on a lucky token roll. Raise these
    # for more creative tasks (writing, brainstorming).
    gemma_temperature: float = field(
        default_factory=lambda: float(os.getenv("GEMMA_TEMPERATURE", "0.1"))
    )
    qwen_temperature: float = field(
        default_factory=lambda: float(os.getenv("QWEN_TEMPERATURE", "0.2"))
    )

    # ── Escalation thresholds ────────────────────────────────────────────────
    # If Qwen confidence < this value, escalate to Claude
    escalation_confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("ESCALATION_THRESHOLD", "0.6"))
    )
    # If executor fails this many times in a row, escalate
    escalation_failure_limit: int = field(
        default_factory=lambda: int(os.getenv("ESCALATION_FAILURE_LIMIT", "3"))
    )
    # Max seconds to wait for a single subtask before treating it as hung.
    # This is the **wall-clock** escape hatch — the real timeout mechanism
    # for shell commands is the silence-based one below. 900s (15 min)
    # gives a legitimate `docker compose up --build` enough rope; the
    # silence detector catches actual hangs in ~90s.
    subtask_timeout: int = field(
        default_factory=lambda: int(os.getenv("SUBTASK_TIMEOUT", "900"))
    )

    # Seconds of silence (no stdout/stderr activity) before a shell command
    # is considered hung and killed. This is the PRIMARY guard for
    # long-running docker builds (wall-clock timeout is disabled for
    # them entirely in shell_tool). 180s is roomy enough for npm install
    # / cargo compile pauses during dep resolution or heavy compile
    # phases while still catching real hangs (DNS stuck, interactive
    # prompt, lock contention) in a few minutes. Set to 0 to disable.
    shell_silence_timeout: int = field(
        default_factory=lambda: int(os.getenv("SHELL_SILENCE_TIMEOUT", "180"))
    )

    # ── API server ───────────────────────────────────────────────────────────
    api_host: str = field(
        default_factory=lambda: os.getenv("API_HOST", "0.0.0.0")
    )
    api_port: int = field(
        default_factory=lambda: int(os.getenv("API_PORT", "8000"))
    )

    # ── Self-growth ──────────────────────────────────────────────────────────
    # Minimum number of successful task runs before a skill is promoted
    skill_promotion_threshold: int = field(
        default_factory=lambda: int(os.getenv("SKILL_PROMOTION_THRESHOLD", "5"))
    )
    # Maximum tokens for skill code generation
    max_skill_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_SKILL_TOKENS", "2048"))
    )
    # Minimum similarity score to reuse an existing skill (0.0–1.0)
    skill_match_threshold: float = field(
        default_factory=lambda: float(os.getenv("SKILL_MATCH_THRESHOLD", "0.4"))
    )

    def validate(self) -> None:
        """Raise ValueError for obviously broken configuration."""
        if not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required for Claude escalation. "
                "Set it in your .env file."
            )

    async def reload_from_store(self, store: "SettingsStore") -> None:  # type: ignore[name-defined]  # noqa: F821
        """Pull all settings from the DB store into this config instance.

        This is the hot-reload path — called at startup and after POST /settings.
        No server restart required.
        """
        settings = await store.get_all()

        def _s(k: str) -> str | None:
            return settings.get(k)

        def _f(k: str) -> float | None:
            try:
                return float(settings[k])
            except (KeyError, ValueError, TypeError):
                return None

        def _i(k: str) -> int | None:
            try:
                return int(settings[k])
            except (KeyError, ValueError, TypeError):
                return None

        if _s("orchestrator_model"):
            self.gemma_model = settings["orchestrator_model"]
        if _s("executor_model"):
            self.qwen_model = settings["executor_model"]
        if _s("claude_model"):
            self.claude_model = settings["claude_model"]
        if _s("gemma_base_url"):
            self.gemma_base_url = settings["gemma_base_url"]
        if _s("qwen_base_url"):
            self.qwen_base_url = settings["qwen_base_url"]

        # Claude API key — only overwrite when the user actually saved one
        # in the UI. Empty string keeps the env-var fallback.
        if settings.get("anthropic_api_key"):
            self.anthropic_api_key = settings["anthropic_api_key"]

        # Boolean toggles. We treat anything other than the explicit
        # "off" tokens as truthy so an accidentally-stored "True" / "yes"
        # from a different code path still works.
        _OFF = {"0", "false", "no", "off", ""}
        if "escalation_enabled" in settings:
            self.escalation_enabled = (
                str(settings["escalation_enabled"]).strip().lower() not in _OFF
            )
        if "streaming_enabled" in settings:
            self.streaming_enabled = (
                str(settings["streaming_enabled"]).strip().lower() not in _OFF
            )

        v = _f("escalation_threshold")
        if v is not None:
            self.escalation_confidence_threshold = v

        v2 = _i("escalation_failure_limit")
        if v2 is not None:
            self.escalation_failure_limit = v2

        v3 = _i("gemma_max_tokens")
        if v3 is not None:
            self.gemma_max_tokens = v3

        v4 = _i("qwen_max_tokens")
        if v4 is not None:
            self.qwen_max_tokens = v4

        vgt = _f("gemma_temperature")
        if vgt is not None:
            self.gemma_temperature = vgt
        vqt = _f("qwen_temperature")
        if vqt is not None:
            self.qwen_temperature = vqt

        v5 = _f("skill_match_threshold")
        if v5 is not None:
            self.skill_match_threshold = v5

        v6 = _i("skill_promotion_threshold")
        if v6 is not None:
            self.skill_promotion_threshold = v6

        v7 = _i("subtask_timeout")
        if v7 is not None:
            self.subtask_timeout = v7

        v8 = _i("shell_silence_timeout")
        if v8 is not None:
            self.shell_silence_timeout = v8

        if _s("agent_workspace_dir"):
            # Always store as absolute — keeps every downstream resolution
            # consistent regardless of what cwd uvicorn happens to be in.
            self.agent_workspace_dir = os.path.abspath(settings["agent_workspace_dir"])

        # Notification credentials — strings only (port parsed as int).
        for k in ("telegram_bot_token", "telegram_default_chat_id",
                  "smtp_host", "smtp_user", "smtp_password", "smtp_from",
                  "webhook_secret"):
            if k in settings:
                setattr(self, k, settings[k] or "")
        smtp_port_val = _i("smtp_port")
        if smtp_port_val is not None:
            self.smtp_port = smtp_port_val


# Module-level singleton
config = Config()
