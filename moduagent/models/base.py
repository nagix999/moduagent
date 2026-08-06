from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Mapping, Protocol, Sequence, runtime_checkable

from moduagent.messages import Message, ToolCall, Usage


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Features supported by a model adapter and its configured endpoint."""

    chat: bool = True
    streaming: bool = True
    tool_calling: bool = True
    parallel_tool_calling: bool = True
    structured_output: bool = True
    embeddings: bool = False
    vision: bool = False
    limits: Mapping[str, Any] = field(default_factory=dict)
    # Appended so existing positional construction retains its 0.3 meaning.
    # ``True`` preserves the permissive behavior of custom/legacy adapters;
    # adapters with a known conflict (notably vLLM) override it to ``False``.
    tool_calling_with_structured_output: bool = True

    @property
    def supports_streaming(self) -> bool:
        return self.streaming

    @property
    def supports_tool_calling(self) -> bool:
        return self.tool_calling

    @property
    def supports_parallel_tool_calling(self) -> bool:
        return self.parallel_tool_calling

    @property
    def supports_structured_output(self) -> bool:
        return self.structured_output

    @property
    def supports_tool_calling_with_structured_output(self) -> bool:
        return self.tool_calling_with_structured_output

    @property
    def supports_embeddings(self) -> bool:
        return self.embeddings

    @property
    def supports_vision(self) -> bool:
        return self.vision


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Provider-neutral input for one chat completion."""

    messages: tuple[Message, ...]
    tools: tuple[Any, ...] = ()
    output_schema: Mapping[str, Any] | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    provider_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(self.tools))
        if self.output_schema is not None:
            object.__setattr__(self, "output_schema", dict(self.output_schema))
        object.__setattr__(self, "options", dict(self.options))
        object.__setattr__(self, "provider_options", dict(self.provider_options))


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Normalized final response returned by every model adapter."""

    message: Message
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage | Mapping[str, Any] = field(default_factory=Usage)
    finish_reason: str | None = None
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        calls = tuple(self.tool_calls or self.message.tool_calls)
        message = self.message
        if message.tool_calls != calls:
            message = replace(message, tool_calls=calls)
        usage = self.usage
        if not isinstance(usage, Usage):
            usage = Usage.from_provider(usage)

        object.__setattr__(self, "message", message)
        object.__setattr__(self, "tool_calls", calls)
        object.__setattr__(self, "usage", usage)
        object.__setattr__(self, "provider_metadata", dict(self.provider_metadata))


@dataclass(frozen=True, slots=True)
class ModelChunk:
    """One text delta, or the terminal normalized response of a stream."""

    delta: str = ""
    response: ModelResponse | None = None
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_final(self) -> bool:
        return self.response is not None


@runtime_checkable
class ModelClient(Protocol):
    @property
    def capabilities(self) -> ModelCapabilities: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...

    def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]: ...

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[tuple[float, ...], ...]: ...


@runtime_checkable
class ModelGateway(Protocol):
    """Run-bound boundary for normalized auxiliary model completion calls."""

    async def complete(
        self,
        model: ModelClient,
        request: ModelRequest,
        *,
        phase: str,
    ) -> ModelResponse: ...


@runtime_checkable
class AuxiliaryModelRequestPreparer(Protocol):
    """Optional gateway capability for planner-style request preparation."""

    async def prepare_auxiliary_model_request(
        self,
        request: ModelRequest,
        *,
        model: ModelClient,
        phase: str,
        skill_phase: str | None,
        protected_from: int,
    ) -> ModelRequest: ...


def validate_request_capabilities(
    request: ModelRequest,
    capabilities: ModelCapabilities,
    *,
    streaming: bool = False,
) -> None:
    if streaming and not capabilities.streaming:
        raise ValueError("the configured model does not support streaming")
    if request.tools and not capabilities.tool_calling:
        raise ValueError("the configured model does not support tool calling")
    if request.output_schema is not None and not capabilities.structured_output:
        raise ValueError("the configured model does not support structured output")
    if (
        request.tools
        and request.output_schema is not None
        and not capabilities.tool_calling_with_structured_output
    ):
        raise ValueError(
            "the configured model requires separate tool-calling and "
            "structured-output requests"
        )
