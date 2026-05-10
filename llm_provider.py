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


# ── OpenRouter implementation ────────────────────────────────────────────────
#
# OpenRouter is an OpenAI-compatible aggregator. Same /chat/completions
# endpoint shape as the official OpenAI API, but with an `Authorization:
# Bearer <key>` header and the model id namespaced by upstream provider
# (e.g. ``"anthropic/claude-sonnet-4"``, ``"openai/gpt-4o"``,
# ``"google/gemini-2.0-pro"``).
#
# Why we don't reuse AnthropicProvider's history: OpenAI's tool-use
# protocol is structurally different from Anthropic's. Where Anthropic
# uses content blocks (assistant turns hold a list of {type:tool_use|
# text} blocks; user turns can hold tool_result blocks), OpenAI uses
# a flat ``tool_calls`` array on the assistant message + role:tool
# messages for results. Two representations, two histories.
#
# We KEEP the public LLMToolTurnResponse / LLMToolUse / LLMToolResult
# interface identical though — callers are agnostic, only the bytes
# on the wire differ.


class _OpenAICompatHistory(LLMHistory):
    """History stored in OpenAI's chat-completions message shape.

    OpenAI format reminder:
      * user:    ``{"role": "user", "content": "text"}``
      * assistant (text only):
          ``{"role": "assistant", "content": "text"}``
      * assistant (with tool calls):
          ``{"role": "assistant", "content": null,
             "tool_calls": [{
                 "id": "call_…",
                 "type": "function",
                 "function": {"name": "…", "arguments": "<JSON>"}
             }]}``
        Note that ``arguments`` is a **JSON-encoded string**, not an
        object — that's an OpenAI quirk we have to honour exactly or
        models 422 us at the API.
      * tool result:
          ``{"role": "tool", "tool_call_id": "call_…", "content": "…"}``
        One message per tool call, NOT a content-block list like
        Anthropic.
    """

    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_assistant_turn(self, response: LLMToolTurnResponse) -> None:
        # Convert our normalised response back into OpenAI's split shape.
        # Two cases:
        #   * No tool calls → simple text-content assistant message.
        #   * Tool calls present → `content: null` + `tool_calls` array.
        #     Including a non-null content with tool_calls in the SAME
        #     message is OpenAI-spec-legal but most models reject it
        #     with 400, so we put any text into a SEPARATE preceding
        #     assistant message when both are present.
        text = response.text or ""
        if not response.tool_uses:
            # Pure text turn.
            self.messages.append({"role": "assistant", "content": text})
            return

        # Tool-using turn. If the model also produced text alongside the
        # tool_uses (Claude-via-OpenRouter does this often, raw GPT-4
        # rarely), split into two assistant messages so neither rejects.
        if text.strip():
            self.messages.append({"role": "assistant", "content": text})

        tool_calls = []
        for tu in response.tool_uses:
            tool_calls.append({
                "id":   tu.id,
                "type": "function",
                "function": {
                    "name":      tu.name,
                    # OpenAI requires this as a JSON-encoded STRING, not
                    # an object. Models 422 the request if we send the
                    # raw dict.
                    "arguments": json_dumps(tu.input),
                },
            })
        self.messages.append({
            "role":       "assistant",
            "content":    None,
            "tool_calls": tool_calls,
        })

    def add_tool_results(self, results: List[LLMToolResult]) -> None:
        # OpenAI: one message per tool result, role:tool, tool_call_id
        # binds it back to the originating call. is_error has no native
        # representation in OpenAI's protocol — we stuff it as a prefix
        # on the content string so the model still sees the signal.
        for r in results:
            content = r.content or "(no output)"
            if r.is_error:
                content = "[ERROR] " + content
            self.messages.append({
                "role":         "tool",
                "tool_call_id": r.tool_use_id,
                "content":      content,
            })


def json_dumps(obj: Any) -> str:
    """Stable, compact JSON encoding for OpenAI's `arguments` field.
    Module-level helper so the import can be lazy at the top of the
    file (json is std-lib but we keep imports explicit per provider)."""
    import json
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class _OpenAICompatProvider(LLMProvider):
    """Abstract base for any LLM endpoint that speaks OpenAI's
    chat-completions protocol.

    Covers four real providers in the current build:
      * OpenRouter — multi-vendor aggregator at openrouter.ai
      * OpenAI    — direct ChatGPT API at api.openai.com
      * Gemini    — Google's OpenAI-compat endpoint at
                    generativelanguage.googleapis.com/v1beta/openai
                    (added Nov 2024; same protocol as OpenAI)
      * Deepseek  — direct at api.deepseek.com (also OpenAI-compat)

    All four share:
      * Bearer-token auth (header: ``Authorization: Bearer <key>``)
      * ``POST /chat/completions`` endpoint
      * The OpenAI message shape we already encode in
        :class:`_OpenAICompatHistory`
      * The OpenAI ``tools`` schema (matches what
        ToolRegistry.get_schemas() already emits)

    Subclasses set five class attributes — base URL, the
    ``provider_name`` we record on usage rows, the ``config.*``
    attribute names that hold the API key + model, and a default
    model id for when ``config.{prefix}_model`` isn't set. That's
    it; everything else is shared.

    Optionally override :meth:`_get_extra_headers` for provider-
    specific bits (OpenRouter uses HTTP-Referer + X-Title for the
    public app leaderboard, e.g.).

    Uses httpx instead of OpenAI's official SDK because (a) the SDK
    is heavyweight relative to the single endpoint we hit, and (b) it
    bakes in OpenAI-only retry / proxy behaviour that doesn't always
    play nice with non-OpenAI compat endpoints.
    """

    # Subclasses MUST override these — they're class-level so each
    # provider gets its own values without re-implementing __init__.
    BASE_URL: str          = ""
    PROVIDER_NAME: str     = ""
    API_KEY_FIELD: str     = ""   # name of config attribute, e.g. "openai_api_key"
    MODEL_FIELD: str       = ""   # name of config attribute, e.g. "openai_model"
    DEFAULT_MODEL: str     = ""   # used when config field is empty
    KEY_HELP_URL: str      = ""   # surfaced in the unconfigured message

    # Output-token-cap parameter name. The legacy OpenAI / vendor-compat
    # name is "max_tokens". OpenAI's newer models (o1, o3, gpt-5 family)
    # rejected it in early 2025 and require "max_completion_tokens"
    # instead — the new param distinguishes "tokens to emit" from
    # "tokens to think with" for reasoning models.
    #   * OpenAI direct      → must be "max_completion_tokens" because
    #                          the user might pick any current model.
    #   * OpenRouter         → accepts the legacy name (it normalises
    #                          internally before forwarding).
    #   * Gemini OpenAI-compat → accepts the legacy name.
    #   * Deepseek           → accepts the legacy name.
    MAX_TOKENS_PARAM: str = "max_tokens"

    def __init__(self) -> None:
        self._client: Any = None  # lazy httpx.AsyncClient
        self._cached_key: Optional[str] = None

    # ── Identity ─────────────────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        from config import config
        return getattr(config, self.MODEL_FIELD, None) or self.DEFAULT_MODEL

    @property
    def provider_name(self) -> str:
        return self.PROVIDER_NAME

    def _get_key(self) -> str:
        from config import config
        return (getattr(config, self.API_KEY_FIELD, "") or "").strip()

    def is_configured(self) -> bool:
        return bool(self._get_key())

    def get_unconfigured_message(self) -> str:
        # Title-case provider name unless it's an acronym we want to
        # preserve. Map the few we know; default to .title() for
        # anything else.
        nice = {
            "openrouter": "OpenRouter",
            "openai":     "OpenAI",
            "gemini":     "Gemini",
            "deepseek":   "Deepseek",
        }.get(self.PROVIDER_NAME, self.PROVIDER_NAME.title())
        return (
            f"{nice} API key is not configured. Get one at "
            f"{self.KEY_HELP_URL}, then paste it in Settings → Model "
            f"Settings → Cloud LLM Provider."
        )

    # ── Conversation factory ─────────────────────────────────────────────────

    def new_history(self) -> LLMHistory:
        return _OpenAICompatHistory()

    # ── Internals ────────────────────────────────────────────────────────────

    def _get_extra_headers(self) -> Dict[str, str]:
        """Override to add provider-specific headers. Defaults to none."""
        return {}

    def _get_client(self) -> Any:
        """Lazy httpx.AsyncClient. Rebuilt whenever the key changes so
        Settings UI hot-reload of credentials takes effect on the next
        request without restarting uvicorn."""
        import httpx
        key = self._get_key()
        if self._client is None or self._cached_key != key:
            if self._client is not None:
                # httpx clients clean up via aclose() but we're in a
                # sync getter; drop the ref + let GC handle sockets.
                self._client = None
            headers = {"Authorization": f"Bearer {key}"}
            headers.update(self._get_extra_headers())
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                # 90s covers a Claude-tier model on a long-context
                # tool-use turn. Cheap providers (Deepseek, GPT-4o)
                # finish much faster — 90s is just the ceiling.
                timeout=httpx.Timeout(90.0, connect=10.0),
                headers=headers,
            )
            self._cached_key = key
        return self._client

    async def _post_chat_completions(
        self, client: Any, body: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Single network call helper that surfaces the upstream's
        actual error body when status >= 400.

        Why: previously the four call sites just did
            resp = await client.post(...); resp.raise_for_status()
            except Exception as exc: raise LLMProviderError(str(exc))
        — and httpx's HTTPStatusError.__str__ only includes the
        URL + status, NOT the response body. A 400 from OpenAI
        bubbled up as just '400 Bad Request' with no clue WHY
        OpenAI rejected the request (invalid model? bad tool
        schema? quota? content policy?). User had to fish the
        real reason out of devtools network tab.

        This helper preserves the body so the LLMProviderError
        message contains everything the upstream told us — the
        UI surfaces it verbatim, log lines have full provenance,
        and "OpenAI says you sent an unknown model 'gpt-5'" is
        actionable instead of "400 Bad Request".
        """
        import json
        try:
            resp = await client.post("/chat/completions", json=body)
        except Exception as exc:  # noqa: BLE001
            # Network-level failure — DNS, TLS, timeout, etc. No body.
            raise LLMProviderError(
                f"{self.PROVIDER_NAME} network error: {exc}"
            ) from exc

        if resp.status_code >= 400:
            # Try to extract the upstream's structured error. OpenAI-
            # compat APIs return JSON like:
            #   {"error": {"message": "...", "type": "...", "code": "..."}}
            # Surface .error.message if present, else the raw text.
            detail = ""
            try:
                err_body = resp.json()
                if isinstance(err_body, dict):
                    err_obj = err_body.get("error")
                    if isinstance(err_obj, dict):
                        detail = err_obj.get("message") or json.dumps(err_obj)
                    elif err_obj:
                        detail = str(err_obj)
                    else:
                        detail = json.dumps(err_body)
            except Exception:  # noqa: BLE001
                # Non-JSON body (HTML 502 from a proxy, plain text, etc.)
                detail = (resp.text or "").strip()[:500]

            logger.error(
                "[%s] %s %s -> %d: %s",
                self.PROVIDER_NAME, "POST", "/chat/completions",
                resp.status_code, detail,
            )
            raise LLMProviderError(
                f"{self.PROVIDER_NAME} API error "
                f"(HTTP {resp.status_code}): {detail or '(empty body)'}"
            )

        try:
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(
                f"{self.PROVIDER_NAME} returned non-JSON response: {exc}"
            ) from exc

    @staticmethod
    def _parse_response(payload: Dict[str, Any]) -> LLMToolTurnResponse:
        """Decode an OpenAI-compat /chat/completions response into our
        normalised LLMToolTurnResponse. Tolerates the dialects we've
        seen across the four upstreams (Claude-via-OR returns
        content as a list of blocks; GPT-4 returns a plain string;
        Gemini occasionally emits null content alongside tool_calls)."""
        choices = payload.get("choices") or []
        if not choices:
            return LLMToolTurnResponse(text="", tool_uses=[], usage=LLMUsage())
        msg = choices[0].get("message") or {}

        # Text content. Most upstreams: string. Some (Claude on OR):
        # list of {type, text} dicts. Coalesce into one string.
        text_val = msg.get("content")
        if isinstance(text_val, list):
            text = "\n\n".join(
                p.get("text", "") for p in text_val
                if isinstance(p, dict) and p.get("type") == "text"
            )
        else:
            text = (text_val or "").strip()

        # Tool calls. OpenAI shape: list of {id, type, function:{name,
        # arguments}} where arguments is a JSON-encoded string.
        tool_uses: List[LLMToolUse] = []
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                import json
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except Exception:  # noqa: BLE001
                # Some models occasionally produce malformed JSON; pass
                # the raw string through as a single 'raw' arg so the
                # tool can still try to recover.
                args = {"_raw": raw_args}
            tool_uses.append(LLMToolUse(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                input=args if isinstance(args, dict) else {},
            ))

        usage_raw = payload.get("usage") or {}
        usage = LLMUsage(
            input_tokens=int(usage_raw.get("prompt_tokens", 0) or 0),
            output_tokens=int(usage_raw.get("completion_tokens", 0) or 0),
        )
        return LLMToolTurnResponse(text=text, tool_uses=tool_uses, usage=usage)

    # ── Text completion ──────────────────────────────────────────────────────

    async def complete_text(
        self,
        user_message: str,
        system: Optional[str] = None,
        max_tokens: int = 2048,
    ) -> LLMTextResponse:
        client = self._get_client()
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_message})

        payload = await self._post_chat_completions(client, {
            "model":               self.model_name,
            "messages":            messages,
            self.MAX_TOKENS_PARAM: max_tokens,
        })
        parsed = self._parse_response(payload)
        return LLMTextResponse(text=parsed.text, usage=parsed.usage)

    # ── Tool-use loop ────────────────────────────────────────────────────────

    async def complete_with_tools(
        self,
        history: LLMHistory,
        tools: List[Dict[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> LLMToolTurnResponse:
        if not isinstance(history, _OpenAICompatHistory):
            raise LLMProviderError(
                f"{type(self).__name__} received non-OpenAI-compat history "
                f"({type(history).__name__}); call provider.new_history() "
                "to get a compatible instance."
            )

        client = self._get_client()
        # ToolRegistry already emits the OpenAI shape — no conversion
        # needed. (That's the whole reason we picked this shape for the
        # interface; see LLMProvider.complete_with_tools docstring.)
        messages = list(history.messages)
        if system:
            # OpenAI puts system as the FIRST message with role:system,
            # not as a top-level kwarg the way Anthropic does.
            messages = [{"role": "system", "content": system}] + messages

        payload = await self._post_chat_completions(client, {
            "model":               self.model_name,
            "messages":            messages,
            "tools":               tools,
            self.MAX_TOKENS_PARAM: max_tokens,
        })
        return self._parse_response(payload)


# ── Concrete OpenAI-compat providers ─────────────────────────────────────────
# Each is a thin subclass that fills in the per-vendor URL + key/model
# field names + default model id + help URL. Add a new provider by
# adding ~10 lines here and one entry in PROVIDER_REGISTRY at the
# bottom of this file (used by the factory + the Settings UI guard).

class OpenRouterProvider(_OpenAICompatProvider):
    BASE_URL      = "https://openrouter.ai/api/v1"
    PROVIDER_NAME = "openrouter"
    API_KEY_FIELD = "openrouter_api_key"
    MODEL_FIELD   = "openrouter_model"
    # Default to Claude Sonnet via OR so behaviour is consistent when
    # users just flip the provider switch without touching the model.
    DEFAULT_MODEL = "anthropic/claude-sonnet-4"
    KEY_HELP_URL  = "openrouter.ai/keys"

    def _get_extra_headers(self) -> Dict[str, str]:
        # OpenRouter ranks "popular apps" by these headers — cheap
        # marketing visibility for OpenTeddy.
        return {
            "HTTP-Referer": "https://openteddy.net",
            "X-Title":      "OpenTeddy",
        }


class OpenAIProvider(_OpenAICompatProvider):
    """Direct ChatGPT API. For users who already pay for OpenAI and
    want to consolidate billing instead of going through OpenRouter.

    Uses ``max_completion_tokens`` instead of the legacy ``max_tokens``.
    OpenAI's o1/o3/gpt-5 family rejects the old param outright — and
    every current chat-completions model accepts the new one — so
    sending the new param universally avoids a model-aware branch."""
    BASE_URL          = "https://api.openai.com/v1"
    PROVIDER_NAME     = "openai"
    API_KEY_FIELD     = "openai_api_key"
    MODEL_FIELD       = "openai_model"
    DEFAULT_MODEL     = "gpt-4o"
    KEY_HELP_URL      = "platform.openai.com/api-keys"
    MAX_TOKENS_PARAM  = "max_completion_tokens"


class GeminiProvider(_OpenAICompatProvider):
    """Google Gemini via the OpenAI-compat endpoint Google added in
    Nov 2024. Same Bearer auth + chat-completions shape, just a
    different base URL. Tool use IS supported on this endpoint."""
    BASE_URL      = "https://generativelanguage.googleapis.com/v1beta/openai"
    PROVIDER_NAME = "gemini"
    API_KEY_FIELD = "gemini_api_key"
    MODEL_FIELD   = "gemini_model"
    DEFAULT_MODEL = "gemini-2.0-flash"
    KEY_HELP_URL  = "aistudio.google.com/app/apikey"


class DeepseekProvider(_OpenAICompatProvider):
    """Deepseek's direct API — pure OpenAI-compat. Cheapest of the
    four (~$0.27 / M input tokens for deepseek-chat). Tool-use
    quality is weaker than Claude / GPT-4 but workable for non-
    long-horizon tasks."""
    BASE_URL      = "https://api.deepseek.com/v1"
    PROVIDER_NAME = "deepseek"
    API_KEY_FIELD = "deepseek_api_key"
    MODEL_FIELD   = "deepseek_model"
    DEFAULT_MODEL = "deepseek-chat"
    KEY_HELP_URL  = "platform.deepseek.com/api_keys"


# ── Provider registry + factory ──────────────────────────────────────────────

# Single source of truth for "which providers exist". The Settings UI
# walks this same registry to render the provider dropdown + the
# conditional key/model rows, so adding a 6th provider is one entry
# here + one row in static/index.html — nothing else needs to change.
PROVIDER_REGISTRY: Dict[str, type] = {
    "anthropic":  AnthropicProvider,
    "openrouter": OpenRouterProvider,
    "openai":     OpenAIProvider,
    "gemini":     GeminiProvider,
    "deepseek":   DeepseekProvider,
}


_DEFAULT_PROVIDER: Optional[LLMProvider] = None
_DEFAULT_PROVIDER_NAME: Optional[str] = None


def get_default_provider() -> LLMProvider:
    """Return the configured default provider, switching on the
    ``config.llm_provider`` setting. See :data:`PROVIDER_REGISTRY` for
    the full set of recognised values; unknown values fall back to
    Anthropic (the validated default).

    Cached as a module-level singleton, BUT we also remember which
    provider name we cached — if the user flips the setting via the
    Settings UI, the next call rebuilds with the new class. Plain
    key-rotation is handled inside the provider's _get_client(); only
    a provider-name flip needs a fresh instance.
    """
    global _DEFAULT_PROVIDER, _DEFAULT_PROVIDER_NAME
    from config import config
    requested = (getattr(config, "llm_provider", None) or "anthropic").lower()
    if _DEFAULT_PROVIDER is not None and _DEFAULT_PROVIDER_NAME == requested:
        return _DEFAULT_PROVIDER
    cls = PROVIDER_REGISTRY.get(requested, AnthropicProvider)
    _DEFAULT_PROVIDER = cls()
    _DEFAULT_PROVIDER_NAME = requested
    return _DEFAULT_PROVIDER
