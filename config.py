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

    def validate(self) -> None:
        """Raise ValueError for obviously broken configuration."""
        if not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required for Claude escalation. "
                "Set it in your .env file."
            )


# Module-level singleton
config = Config()
