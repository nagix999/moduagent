from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from moduagent.messages import Message, MessageRole, Usage

from .base import (
    ModelCapabilities,
    ModelChunk,
    ModelRequest,
    ModelResponse,
    validate_request_capabilities,
)
from .errors import ModelProtocolError
from .openai_compatible import (
    _StreamingToolCalls,
    _content_text,
    _provider_metadata,
    _role_value,
    _tool_calls_from_provider,
    _tool_schema_to_dict,
)
from .transport import HttpTransport, HttpxTransport


def _message_to_ollama(message: Message) -> dict[str, Any]:
    role = _role_value(message.role)
    value: dict[str, Any] = {"role": role, "content": message.content or ""}
    if message.tool_calls:
        value["tool_calls"] = [
            {
                "function": {
                    "name": call.name,
                    "arguments": dict(call.arguments),
                }
            }
            for call in message.tool_calls
        ]
    if role == MessageRole.TOOL.value:
        if message.name:
            value["tool_name"] = message.name
        if message.tool_call_id:
            value["tool_call_id"] = message.tool_call_id
    return value


class OllamaClient:
    """Adapter for Ollama's native ``/api/chat`` JSON and JSONL API."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        transport: HttpTransport | None = None,
        timeout: float = 60.0,
        capabilities: ModelCapabilities | None = None,
        default_options: Mapping[str, Any] | None = None,
        provider_options: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url cannot be empty")
        if not model.strip():
            raise ValueError("model cannot be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        normalized_url = base_url.rstrip("/")
        if not normalized_url.endswith("/api/chat"):
            normalized_url += "/api/chat"

        self.base_url = base_url.rstrip("/")
        self.endpoint = normalized_url
        self.model = model
        self.timeout = timeout
        self.transport = transport or HttpxTransport()
        self._capabilities = capabilities or ModelCapabilities(embeddings=True)
        self.default_options = dict(default_options or {})
        self.default_provider_options = dict(provider_options or {})
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **dict(headers or {}),
        }

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def _build_payload(self, request: ModelRequest, *, stream: bool) -> dict[str, Any]:
        validate_request_capabilities(request, self.capabilities, streaming=stream)
        provider_options = {
            **self.default_provider_options,
            **dict(request.provider_options),
        }
        extra_body = provider_options.pop("extra_body", None)
        if isinstance(extra_body, Mapping):
            provider_options.update(extra_body)
        payload: dict[str, Any] = dict(provider_options)
        payload["model"] = self.model
        payload["messages"] = [
            _message_to_ollama(message) for message in request.messages
        ]
        payload["stream"] = stream

        options = {**self.default_options, **dict(request.options)}
        if options:
            payload["options"] = options
        if request.tools:
            payload["tools"] = [_tool_schema_to_dict(tool) for tool in request.tools]
        if request.output_schema is not None:
            payload["format"] = dict(request.output_schema)
        return payload

    def _parse_response(self, value: Mapping[str, Any]) -> ModelResponse:
        if value.get("error"):
            raise ModelProtocolError("model endpoint returned an error response")
        raw_message = value.get("message")
        if not isinstance(raw_message, Mapping):
            raise ModelProtocolError("Ollama response contains no message")
        calls = _tool_calls_from_provider(raw_message.get("tool_calls"))
        message = Message.assistant(_content_text(raw_message.get("content")), calls)
        finish_reason = value.get("done_reason")
        if finish_reason is None and value.get("done"):
            finish_reason = "stop"
        try:
            usage = Usage.from_provider(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelProtocolError("Ollama response contains invalid usage") from exc
        return ModelResponse(
            message=message,
            tool_calls=calls,
            usage=usage,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            provider_metadata=_provider_metadata(value, "ollama"),
        )

    async def complete(self, request: ModelRequest) -> ModelResponse:
        payload = self._build_payload(request, stream=False)
        value = await self.transport.post_json(
            self.endpoint,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        return self._parse_response(value)

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        if not self.capabilities.embeddings:
            raise ValueError("the configured model does not support embeddings")
        payload: dict[str, Any] = {
            **self.default_provider_options,
            "model": self.model,
            "input": inputs if isinstance(inputs, str) else list(inputs),
        }
        embedding_options = {**self.default_options, **dict(options or {})}
        if embedding_options:
            payload["options"] = embedding_options
        endpoint = self.endpoint.rsplit("/api/chat", 1)[0] + "/api/embed"
        value = await self.transport.post_json(
            endpoint,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        rows = value.get("embeddings")
        if not isinstance(rows, Sequence) or isinstance(
            rows,
            (str, bytes, bytearray),
        ):
            raise ModelProtocolError("Ollama embedding response contains no embeddings")
        embeddings: list[tuple[float, ...]] = []
        for vector in rows:
            if not isinstance(vector, Sequence) or isinstance(
                vector, (str, bytes, bytearray)
            ):
                raise ModelProtocolError("Ollama returned an invalid embedding vector")
            try:
                embeddings.append(tuple(float(value) for value in vector))
            except (TypeError, ValueError) as exc:
                raise ModelProtocolError(
                    "Ollama returned a non-numeric embedding vector"
                ) from exc
        return tuple(embeddings)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        payload = self._build_payload(request, stream=True)
        content_parts: list[str] = []
        tool_calls = _StreamingToolCalls()
        usage = Usage()
        finish_reason: str | None = None
        metadata: dict[str, Any] = {"provider": "ollama"}
        saw_terminal_marker = False

        async for raw_line in self.transport.stream_lines(
            self.endpoint,
            headers={**self.headers, "Accept": "application/x-ndjson"},
            json=payload,
            timeout=self.timeout,
        ):
            if isinstance(raw_line, bytes):
                try:
                    raw_line = raw_line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ModelProtocolError(
                        "Ollama returned invalid UTF-8 JSONL"
                    ) from exc
            for raw_part in raw_line.splitlines() or (raw_line,):
                line = raw_part.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ModelProtocolError("Ollama returned invalid JSONL") from exc
                if not isinstance(value, Mapping):
                    raise ModelProtocolError("Ollama returned a non-object JSONL event")
                if value.get("error"):
                    raise ModelProtocolError(
                        "model endpoint returned an error JSONL event"
                    )
                metadata.update(_provider_metadata(value, "ollama"))
                raw_message = value.get("message")
                if raw_message is not None and not isinstance(raw_message, Mapping):
                    raise ModelProtocolError("Ollama returned an invalid JSONL message")
                if isinstance(raw_message, Mapping):
                    text = _content_text(raw_message.get("content"))
                    if text:
                        content_parts.append(text)
                        yield ModelChunk(delta=text, provider_metadata=metadata)
                    tool_calls.add(raw_message.get("tool_calls"))
                if value.get("done"):
                    saw_terminal_marker = True
                    reason = value.get("done_reason")
                    finish_reason = str(reason) if reason is not None else "stop"
                    try:
                        usage = Usage.from_provider(value)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise ModelProtocolError(
                            "Ollama returned invalid JSONL usage"
                        ) from exc

        if not saw_terminal_marker:
            raise ModelProtocolError(
                "Ollama JSONL stream ended without a terminal marker"
            )
        calls = tool_calls.build()
        content = "".join(content_parts)
        message = Message.assistant(content if content or not calls else None, calls)
        response = ModelResponse(
            message=message,
            tool_calls=calls,
            usage=usage,
            finish_reason=finish_reason,
            provider_metadata=metadata,
        )
        yield ModelChunk(response=response, provider_metadata=metadata)
