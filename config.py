"""
OpenTeddy Configuration
Central configuration management using environment variables.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


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

    # ── Agent workspace ──────────────────────────────────────────────────────
    # Default working directory for shell_exec_* in Code / Analytic modes.
    # Any command that doesn't specify its own working_dir lands here, so
    # `git clone`, `pip install -t .`, file writes etc. don't scatter across
    # the host. A session can override this in the future; for now one dir
    # for the whole project.
    agent_workspace_dir: str = field(
        default_factory=lambda: os.getenv("AGENT_WORKSPACE_DIR", "./agent-workspace")
    )

    # ── Token limits ────────────────────────────────────────────────────────
    gemma_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("GEMMA_MAX_TOKENS", "4096"))
    )
    qwen_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("QWEN_MAX_TOKENS", "4096"))
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
    # 180s is roomy enough for a small-model (Qwen 2.5 3B) multi-tool-call
    # chain (detect → probe → compose_remap → up → diagnose) without
    # leaving hung daemons unattended for too long. Raise this further if
    # you're on a slow machine; lower it if you're running a big model.
    subtask_timeout: int = field(
        default_factory=lambda: int(os.getenv("SUBTASK_TIMEOUT", "180"))
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

        v5 = _f("skill_match_threshold")
        if v5 is not None:
            self.skill_match_threshold = v5

        v6 = _i("skill_promotion_threshold")
        if v6 is not None:
            self.skill_promotion_threshold = v6

        v7 = _i("subtask_timeout")
        if v7 is not None:
            self.subtask_timeout = v7

        if _s("agent_workspace_dir"):
            self.agent_workspace_dir = settings["agent_workspace_dir"]


# Module-level singleton
config = Config()
