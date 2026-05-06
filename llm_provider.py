"""
OpenTeddy LLM Provider abstraction
─────────────────────────────────────────────────────────────────────────────
Single seam between OpenTeddy's cloud-side reasoning calls (escalation,
skill authoring, prompt optimisation, summary synthesis) and whatever
LLM SDK actually fulfils them.

Today there's exactly one implementation — :class:`AnthropicProvider`,
which wraps the Anthropic Messages API. The motivation for extracting
the seam *before* a second implementation exists is to make adding
OpenRouter (or a direct OpenAI/Gemini path) zero-friction in a
follow-up — every cloud call already routes through this interface,
and the call sites are provider-agnostic.

Why the interface looks Anthropic-shaped (text + content blocks):

  * Anthropic's content-block model (assistant turns hold a list of
    ``{type: text}`` and ``{type: tool_use}`` blocks; user turns can hold
    ``tool_result`` blocks) is a strict superset of OpenAI's
    ``tool_calls``/``role:tool`` shape — easier to map down than up.
  * The :class:`LLMHistory` object is provider-managed: each provider
    keeps messages in its native format internally, so the caller's
    loop logic stays identical regardless of which provider is wired.

Adding a new provider therefore boils down to:

  1. Subclass :class:`LLMProvider` with the four required methods.
  2. Subclass :class:`LLMHistory` so messages are stored in the
     provider's native shape.
  3. Wire it into :func:`get_default_provider` (today hard-coded to
     Anthropic; later switches on ``config.llm_provider``).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Normalised value objects ─────────────────────────────────────────────────
# These are the only shapes callers see. Provider-specific types
# (anthropic.types.*, openai.*, etc.) never leak past the provider class.

@dataclass
class LLMUsage:
    """Token counts for one API call. Mirrored straight into
    ``Tracker.record_usage`` so the Usage tab + savings pill keep
    working regardless of provider."""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class LLMTextResponse:
    """Result of a one-shot text completion (no tools)."""
    text: str
    usage: LLMUsage


@dataclass
class LLMToolUse:
    """A single tool invocation requested by the model."""
    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class LLMToolTurnResponse:
    """One turn of a tool-use loop. The caller inspects
    :attr:`tool_uses` — empty list means "model is done, treat
    :attr:`text` as the final answer"; non-empty list means the
    caller should execute each tool and feed the results back via
    :meth:`LLMHistory.add_tool_results`.
    """
    text: str
    tool_uses: List[LLMToolUse] = field(default_factory=list)
    usage: LLMUsage = field(default_factory=LLMUsage)


@dataclass
class LLMToolResult:
    """Output of a single tool call, ready to be fed back to the model
    on the next turn."""
    tool_use_id: str
    content: str
    is_error: bool = False


class LLMProviderError(Exception):
    """Raised by any provider for surface-level API errors (auth,
    timeout, rate-limit, malformed response). Callers catch this
    instead of the SDK-specific exception type so the loop logic
    stays provider-agnostic."""


# ── Conversation history ─────────────────────────────────────────────────────

class LLMHistory(ABC):
    """Provider-managed conversation state.

    Each provider subclasses this so it can store messages in whatever
    shape its SDK expects. Callers only see the four mutating methods —
    they never touch the underlying message list.
    """

    @abstractmethod
    def add_user_message(self, text: str) -> None:
        """Append a plain-text user turn."""

    @abstractmethod
    def add_assistant_turn(self, response: LLMToolTurnResponse) -> None:
        """Append the model's previous turn (text + tool uses) so the
        next call has full history. The provider implementation is
        responsible for translating :class:`LLMToolTurnResponse` back
        into its native message shape (e.g. Anthropic content blocks
        vs OpenAI ``tool_calls``)."""

    @abstractmethod
    def add_tool_results(self, results: List[LLMToolResult]) -> None:
        """Append the user-side turn that carries this round's tool
        results back to the model."""


# ── Provider interface ───────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Abstract LLM that supports text completion and tool-use loops.

    Implementations must be **stateless across requests** (apart from
    SDK client caching), because the Settings UI hot-reloads
    ``config.*`` and the next call may use a fresh API key. Use
    ``is_configured()`` to fail fast when no key is set rather than
    letting an SDK auth error bubble up.
    """

    # ── Identity ─────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model id (e.g. ``"claude-opus-4-6"``). Recorded against every
        usage row so the Usage tab can break costs down by model."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Stable string id for the provider — ``"anthropic"`` today,
        ``"openrouter"`` later. Stored in
        ``usage_records.model_provider`` for cost attribution."""

    @abstractmethod
    def is_configured(self) -> bool:
        """``True`` when the provider has a usable API key. The auth
        check is intentionally local (no network round-trip) so the
        UI can pre-flight calls cheaply."""

    @abstractmethod
    def get_unconfigured_message(self) -> str:
        """Friendly user-facing message explaining how to fix a missing
        key. Surfaced in toast / dialog when ``is_configured()`` is
        ``False``."""

    # ── Conversation factory ─────────────────────────────────────────────────

    @abstractmethod
    def new_history(self) -> LLMHistory:
        """Fresh, empty conversation history bound to this provider's
        message format."""

    # ── Text completion ──────────────────────────────────────────────────────

    @abstractmethod
    async def complete_text(
        self,
        user_message: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> LLMTextResponse:
        """One-shot text completion — no tools, no history. Used by:

          * single-subtask escalation
          * final-summary synthesis
          * skill code generation
          * prompt optimiser

        Raises :class:`LLMProviderError` on any API failure.
        """

    # ── Tool-use loop ────────────────────────────────────────────────────────

    @abstractmethod
    async def complete_with_tools(
        self,
        history: LLMHistory,
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> LLMToolTurnResponse:
        """One turn of a tool-use loop.

        Args:
          history: Conversation state from :meth:`new_history`. The
            implementation reads from it but does **not** mutate it on
            this call — the caller appends the assistant response and
            tool results explicitly via the history's mutators.
          tools: Tool schemas in **OpenAI/Ollama format**, i.e.
            ``[{"type": "function", "function": {"name", "description",
            "parameters"}}]``. The provider converts to its own native
            schema internally. (We pick this shape because that's what
            ``ToolRegistry.get_schemas()`` already emits.)
          system: System prompt.
          max_tokens: Cap on response tokens.

        Returns a :class:`LLMToolTurnResponse`. If
        :attr:`tool_uses` is empty the model is done; otherwise the
        caller should execute each tool, append the results via
        :meth:`LLMHistory.add_tool_results`, and call this method again.
        """


# ── Anthropic implementation ─────────────────────────────────────────────────

class _AnthropicHistory(LLMHistory):
    """History stored in Anthropic's native message-content-block shape.

    Anthropic format reminder:
      * user: ``{"role": "user", "content": "text"}`` OR
              ``{"role": "user", "content": [{"type": "tool_result", ...}, ...]}``
      * assistant: ``{"role": "assistant", "content": [<sdk content blocks>]}``
        — we store the raw SDK list so the SDK can re-encode it on the
        next request without us having to round-trip through dicts.
    """

    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []
        # We stash the previous turn's raw SDK content list keyed by
        # position so add_assistant_turn can persist it verbatim.
        # (See add_assistant_turn for why this matters.)
        self._raw_assistant_blocks: Dict[int, Any] = {}

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant_turn(self, response: LLMToolTurnResponse) -> None:
        # Anthropic requires the assistant turn we echo back to match
        # exactly what the SDK emitted (same content-block ordering,
        # same tool_use ids). We cache the raw SDK blocks on the
        # response and re-use them here.
        raw = getattr(response, "_anthropic_raw_content", None)
        if raw is not None:
            self.messages.append({"role": "assistant", "content": raw})
            return

        # Fallback: reconstruct from the normalised fields. Should
        # never trigger for AnthropicProvider, but keeps the contract
        # honest if a different provider's response somehow lands here.
        blocks: List[Dict[str, Any]] = []
        if response.text:
            blocks.append({"type": "text", "text": response.text})
        for tu in response.tool_uses:
            blocks.append({
                "type":  "tool_use",
                "id":    tu.id,
                "name":  tu.name,
                "input": tu.input,
            })
        self.messages.append({"role": "assistant", "content": blocks})

    def add_tool_results(self, results: List[LLMToolResult]) -> None:
        content = [
            {
                "type":        "tool_result",
                "tool_use_id": r.tool_use_id,
                "content":     r.content or "(no output)",
                "is_error":    r.is_error,
            }
            for r in results
        ]
        self.messages.append({"role": "user", "content": content})


class AnthropicProvider(LLMProvider):
    """Default provider — talks to Anthropic's Messages API.

    This is the original behaviour that lived directly in
    ``escalation.py`` and ``skill_factory.py`` before the
    :class:`LLMProvider` extraction. Behaviour is **identical** to
    pre-refactor; only the call sites changed.
    """

    def __init__(self) -> None:
        # Lazy import — keeps the module loadable in environments where
        # ``anthropic`` isn't installed (e.g. landing-page bundles or
        # OSS users on local-only mode). Failure mode is a clear
        # ImportError on first use, not at import time.
        self._sdk_module: Any = None
        self._client: Any = None
        self._cached_key: Optional[str] = None

    # ── Identity ─────────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        from config import config
        return config.claude_model

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def is_configured(self) -> bool:
        from config import config
        return bool((config.anthropic_api_key or "").strip())

    def get_unconfigured_message(self) -> str:
        return (
            "Claude API key is not configured. Open Settings → Model Settings "
            "→ Claude API Key and paste a key from console.anthropic.com, "
            "or enable Local-Only Mode for this session to skip escalation."
        )

    # ── Conversation factory ─────────────────────────────────────────────────

    def new_history(self) -> LLMHistory:
        return _AnthropicHistory()

    # ── Internals ────────────────────────────────────────────────────────────

    def _get_sdk(self) -> Any:
        """Lazy-load the ``anthropic`` SDK module so importing this file
        doesn't require it to be installed."""
        if self._sdk_module is None:
            import anthropic  # noqa: WPS433  (intentional lazy import)
            self._sdk_module = anthropic
        return self._sdk_module

    def _get_client(self) -> Any:
        """(Re)build the SDK client whenever the configured key changes —
        the Settings UI hot-reloads ``config.anthropic_api_key`` via
        ``reload_from_store``, and we want subsequent calls to pick that
        up without restarting the server.

        Pass ``None`` (not ``""``) when no key is configured. The SDK
        treats ``""`` as "set but invalid" and raises *Could not resolve
        authentication method*; ``None`` lets it fall back to the
        ``ANTHROPIC_API_KEY`` env var (or fail with a clearer error).
        """
        from config import config
        sdk = self._get_sdk()
        key = config.anthropic_api_key or None
        if self._client is None or self._cached_key != key:
            self._client = sdk.AsyncAnthropic(api_key=key)
            self._cached_key = key
        return self._client

    @staticmethod
    def _ollama_tools_to_anthropic(
        ollama_tools: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI/Ollama tool schemas to Anthropic ``input_schema``
        shape. Same mapping as the original inline code in
        ``escalation.resolve_whole_task``.

          Ollama:    {type: function, function: {name, description, parameters}}
          Anthropic: {name, description, input_schema}
        """
        out: List[Dict[str, Any]] = []
        for schema in ollama_tools:
            fn = schema.get("function", {})
            out.append({
                "name":         fn.get("name", ""),
                "description":  fn.get("description", ""),
                "input_schema": fn.get(
                    "parameters", {"type": "object", "properties": {}},
                ),
            })
        return out

    # ── Text completion ──────────────────────────────────────────────────────

    async def complete_text(
        self,
        user_message: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> LLMTextResponse:
        sdk = self._get_sdk()
        client = self._get_client()
        kwargs: Dict[str, Any] = {
            "model":      self.model_name,
            "max_tokens": max_tokens,
            "messages":   [{"role": "user", "content": user_message}],
        }
        if system:
            kwargs["system"] = system
        try:
            response = await client.messages.create(**kwargs)
        except sdk.APIError as exc:
            raise LLMProviderError(str(exc)) from exc

        text = response.content[0].text.strip() if response.content else ""
        usage = LLMUsage(
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
        )
        return LLMTextResponse(text=text, usage=usage)

    # ── Tool-use loop ────────────────────────────────────────────────────────

    async def complete_with_tools(
        self,
        history: LLMHistory,
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> LLMToolTurnResponse:
        if not isinstance(history, _AnthropicHistory):
            raise LLMProviderError(
                "AnthropicProvider received non-Anthropic history "
                f"({type(history).__name__}); call provider.new_history() "
                "to get a compatible instance."
            )

        sdk = self._get_sdk()
        client = self._get_client()
        anthropic_tools = self._ollama_tools_to_anthropic(tools)

        kwargs: Dict[str, Any] = {
            "model":      self.model_name,
            "max_tokens": max_tokens,
            "tools":      anthropic_tools,
            "messages":   history.messages,
        }
        if system:
            kwargs["system"] = system

        try:
            response = await client.messages.create(**kwargs)
        except sdk.APIError as exc:
            raise LLMProviderError(str(exc)) from exc

        # Split response into text and tool_use blocks.
        text_parts: List[str] = []
        tool_uses: List[LLMToolUse] = []
        for block in response.content:
            kind = getattr(block, "type", "")
            if kind == "text":
                text_parts.append(getattr(block, "text", ""))
            elif kind == "tool_use":
                tool_uses.append(LLMToolUse(
                    id=block.id,
                    name=block.name,
                    input=dict(block.input) if isinstance(block.input, dict) else {},
                ))
            # Other block types (e.g. thinking) are ignored — the public
            # surface only models text + tool_use today.

        usage = LLMUsage(
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
        )
        out = LLMToolTurnResponse(
            text="\n\n".join(t for t in text_parts if t).strip(),
            tool_uses=tool_uses,
            usage=usage,
        )
        # Stash the raw SDK content list so add_assistant_turn can echo
        # it verbatim — Anthropic requires byte-for-byte match on the
        # assistant turn that's fed back in. Underscore-prefixed so it
        # stays an implementation detail, not part of the public dataclass.
        out._anthropic_raw_content = response.content  # type: ignore[attr-defined]
        return out


# ── Default provider factory ─────────────────────────────────────────────────

_DEFAULT_PROVIDER: Optional[LLMProvider] = None


def get_default_provider() -> LLMProvider:
    """Return the configured default provider (today: always Anthropic).

    This is the single integration point for adding OpenRouter / OpenAI:
    swap on a future ``config.llm_provider`` setting and return a
    different implementation. Every cloud-side call site already routes
    through this function, so no other file needs to change.

    Cached as a module-level singleton — the underlying SDK client
    inside the provider already does its own key-change re-init, so a
    long-lived provider instance is correct.
    """
    global _DEFAULT_PROVIDER
    if _DEFAULT_PROVIDER is None:
        _DEFAULT_PROVIDER = AnthropicProvider()
    return _DEFAULT_PROVIDER
