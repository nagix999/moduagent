from __future__ import annotations

import json
import math
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from moduagent.messages import Message, MessageRole, ToolCall, Usage

from .base import (
    ModelCapabilities,
    ModelChunk,
    ModelRequest,
    ModelResponse,
    validate_request_capabilities,
)
from .errors import ModelProtocolError
from .transport import HttpTransport, HttpxTransport


def _role_value(role: MessageRole | str) -> str:
    return role.value if isinstance(role, MessageRole) else str(role)


def _tool_schema_to_dict(schema: Any) -> dict[str, Any]:
    if hasattr(schema, "to_dict"):
        value = schema.to_dict()
    elif isinstance(schema, Mapping):
        value = dict(schema)
    else:
        raise TypeError("tool schemas must be mappings or provide to_dict()")
    if not isinstance(value, Mapping):
        raise TypeError("tool schema to_dict() must return a mapping")

    result = dict(value)
    if result.get("type") == "function" and isinstance(result.get("function"), Mapping):
        return result
    if "function" in result and isinstance(result["function"], Mapping):
        return {"type": "function", "function": dict(result["function"])}
    if "name" in result:
        return {"type": "function", "function": result}
    raise ValueError("tool schema must contain a function name")


def _tool_call_to_openai(call: ToolCall) -> dict[str, Any]:
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(
                dict(call.arguments), ensure_ascii=False, separators=(",", ":")
            ),
        },
    }


def _message_to_openai(message: Message) -> dict[str, Any]:
    role = _role_value(message.role)
    value: dict[str, Any] = {"role": role, "content": message.content}
    if message.tool_calls:
        value["tool_calls"] = [
            _tool_call_to_openai(call) for call in message.tool_calls
        ]
    if message.tool_call_id:
        value["tool_call_id"] = message.tool_call_id
    if message.name:
        value["name"] = message.name
    return value


def _content_text(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return str(value)


def _arguments_from_provider(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        raise ModelProtocolError("model tool arguments must be a JSON object")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ModelProtocolError("model returned invalid JSON tool arguments") from exc
    if not isinstance(decoded, Mapping):
        raise ModelProtocolError("model returned non-object tool arguments")
    return dict(decoded)


def _tool_calls_from_provider(value: Any) -> tuple[ToolCall, ...]:
    if not value:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ModelProtocolError("model returned invalid tool_calls")

    calls: list[ToolCall] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ModelProtocolError("model returned an invalid tool call")
        function = item.get("function", item)
        if not isinstance(function, Mapping):
            raise ModelProtocolError("model returned an invalid function call")
        name = function.get("name") or item.get("name")
        if not name:
            raise ModelProtocolError("model tool call has no function name")
        call_id = item.get("id") or f"call-{index + 1}"
        arguments = function.get("arguments", item.get("arguments"))
        calls.append(
            ToolCall(
                id=str(call_id),
                name=str(name),
                arguments=_arguments_from_provider(arguments),
            )
        )
    return tuple(calls)


class _StreamingToolCalls:
    def __init__(self) -> None:
        self._calls: dict[int, dict[str, Any]] = {}

    def add(self, chunks: Any) -> None:
        if not chunks:
            return
        if not isinstance(chunks, Sequence) or isinstance(
            chunks, (str, bytes, bytearray)
        ):
            raise ModelProtocolError("model returned invalid streaming tool_calls")
        for position, item in enumerate(chunks):
            if not isinstance(item, Mapping):
                raise ModelProtocolError(
                    "model returned an invalid streaming tool call"
                )
            try:
                index = int(item.get("index", position))
            except (TypeError, ValueError) as exc:
                raise ModelProtocolError(
                    "model returned an invalid streaming tool call index"
                ) from exc
            state = self._calls.setdefault(
                index, {"id": None, "name": None, "argument_parts": [], "arguments": {}}
            )
            if item.get("id"):
                state["id"] = str(item["id"])
            function = item.get("function", item)
            if not isinstance(function, Mapping):
                raise ModelProtocolError(
                    "model returned an invalid streaming function call"
                )
            if function.get("name"):
                state["name"] = str(function["name"])
            arguments = function.get("arguments", item.get("arguments"))
            if isinstance(arguments, Mapping):
                state["arguments"].update(arguments)
            elif arguments is not None:
                state["argument_parts"].append(str(arguments))

    def build(self) -> tuple[ToolCall, ...]:
        calls: list[ToolCall] = []
        for index, state in sorted(self._calls.items()):
            name = state["name"]
            if not name:
                raise ModelProtocolError("streamed tool call has no function name")
            argument_text = "".join(state["argument_parts"])
            arguments = dict(state["arguments"])
            if argument_text:
                arguments.update(_arguments_from_provider(argument_text))
            calls.append(
                ToolCall(
                    id=state["id"] or f"call-{index + 1}",
                    name=name,
                    arguments=arguments,
                )
            )
        return tuple(calls)


def _provider_metadata(value: Mapping[str, Any], provider: str) -> dict[str, Any]:
    metadata = {"provider": provider}
    for key in ("id", "model", "created", "created_at", "system_fingerprint"):
        if key in value:
            metadata[key] = value[key]
    return metadata


class OpenAICompatibleClient:
    """Adapter for OpenAI-compatible ``/chat/completions`` endpoints."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
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
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a finite number")
        if not math.isfinite(float(timeout)):
            raise ValueError("timeout must be finite")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        normalized_url = base_url.rstrip("/")
        if not normalized_url.endswith("/chat/completions"):
            normalized_url += "/chat/completions"

        self.base_url = base_url.rstrip("/")
        self.endpoint = normalized_url
        self.model = model
        self.timeout = timeout
        self.transport = transport or HttpxTransport()
        self._capabilities = capabilities or ModelCapabilities()
        self.default_options = dict(default_options or {})
        self.default_provider_options = dict(provider_options or {})
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            **dict(headers or {}),
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def _provider_name(self) -> str:
        return "openai-compatible"

    def _request_provider_options(self, request: ModelRequest) -> dict[str, Any]:
        result = {**self.default_provider_options, **dict(request.provider_options)}
        extra_body = result.pop("extra_body", None)
        if isinstance(extra_body, Mapping):
            result.update(extra_body)
        return result

    def _build_payload(self, request: ModelRequest, *, stream: bool) -> dict[str, Any]:
        validate_request_capabilities(request, self.capabilities, streaming=stream)
        payload: dict[str, Any] = {}
        payload.update(self.default_options)
        payload.update(request.options)
        payload.update(self._request_provider_options(request))
        payload["model"] = self.model
        payload["messages"] = [
            _message_to_openai(message) for message in request.messages
        ]
        payload["stream"] = stream
        if stream:
            # OpenAI-compatible servers that support it include usage in the final chunk.
            payload.setdefault("stream_options", {"include_usage": True})
        if request.tools:
            payload["tools"] = [_tool_schema_to_dict(tool) for tool in request.tools]
        else:
            # Tool-selection options are invalid on PLAN/FINALIZE requests that do
            # not expose tools. This also neutralizes client-level defaults.
            payload.pop("tool_choice", None)
            payload.pop("parallel_tool_calls", None)
        if request.output_schema is not None:
            name = str(request.output_schema.get("title") or "response")
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": dict(request.output_schema),
                    "strict": True,
                },
            }
        return payload

    def _parse_response(self, value: Mapping[str, Any]) -> ModelResponse:
        choices = value.get("choices")
        if (
            not isinstance(choices, Sequence)
            or isinstance(choices, (str, bytes, bytearray))
            or not choices
        ):
            error = value.get("error")
            if error:
                raise ModelProtocolError("model endpoint returned an error response")
            raise ModelProtocolError("model response contains no choices")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise ModelProtocolError("model response contains an invalid choice")
        raw_message = choice.get("message")
        if not isinstance(raw_message, Mapping):
            raise ModelProtocolError("model response choice contains no message")
        calls = _tool_calls_from_provider(raw_message.get("tool_calls"))
        message = Message.assistant(_content_text(raw_message.get("content")), calls)
        usage_value = value.get("usage")
        try:
            usage = Usage.from_provider(
                usage_value if isinstance(usage_value, Mapping) else None
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ModelProtocolError("model response contains invalid usage") from exc
        return ModelResponse(
            message=message,
            tool_calls=calls,
            usage=usage,
            finish_reason=(
                str(choice["finish_reason"])
                if choice.get("finish_reason") is not None
                else None
            ),
            provider_metadata=_provider_metadata(value, self._provider_name()),
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
            **self.default_options,
            **dict(options or {}),
            "model": self.model,
            "input": inputs if isinstance(inputs, str) else list(inputs),
        }
        endpoint = self.endpoint.rsplit("/chat/completions", 1)[0] + "/embeddings"
        value = await self.transport.post_json(
            endpoint,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        rows = value.get("data")
        if not isinstance(rows, Sequence) or isinstance(
            rows,
            (str, bytes, bytearray),
        ):
            raise ModelProtocolError("embedding response contains no data")
        if any(not isinstance(row, Mapping) for row in rows):
            raise ModelProtocolError("embedding response contains an invalid row")
        try:
            ordered = sorted(
                rows,
                key=lambda row: int(row.get("index", 0)),
            )
        except (TypeError, ValueError) as exc:
            raise ModelProtocolError(
                "embedding response contains an invalid index"
            ) from exc
        embeddings: list[tuple[float, ...]] = []
        for row in ordered:
            vector = row.get("embedding")
            if not isinstance(vector, Sequence) or isinstance(
                vector, (str, bytes, bytearray)
            ):
                raise ModelProtocolError(
                    "embedding response contains an invalid vector"
                )
            try:
                embeddings.append(tuple(float(value) for value in vector))
            except (TypeError, ValueError) as exc:
                raise ModelProtocolError(
                    "embedding response contains a non-numeric vector"
                ) from exc
        return tuple(embeddings)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        payload = self._build_payload(request, stream=True)
        content_parts: list[str] = []
        tool_calls = _StreamingToolCalls()
        usage = Usage()
        finish_reason: str | None = None
        metadata: dict[str, Any] = {"provider": self._provider_name()}
        saw_terminal_marker = False

        async for raw_line in self.transport.stream_lines(
            self.endpoint,
            headers={**self.headers, "Accept": "text/event-stream"},
            json=payload,
            timeout=self.timeout,
        ):
            if isinstance(raw_line, bytes):
                try:
                    raw_line = raw_line.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ModelProtocolError(
                        "model returned invalid UTF-8 SSE data"
                    ) from exc
            for raw_part in raw_line.splitlines() or (raw_line,):
                line = raw_part.strip()
                if not line or line.startswith(":") or line.startswith("event:"):
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    saw_terminal_marker = True
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ModelProtocolError("model returned invalid SSE JSON") from exc
                if not isinstance(value, Mapping):
                    raise ModelProtocolError("model returned a non-object SSE event")
                if value.get("error"):
                    raise ModelProtocolError(
                        "model endpoint returned an error SSE event"
                    )
                metadata.update(_provider_metadata(value, self._provider_name()))
                raw_usage = value.get("usage")
                if isinstance(raw_usage, Mapping):
                    try:
                        usage = Usage.from_provider(raw_usage)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise ModelProtocolError(
                            "model returned invalid SSE usage"
                        ) from exc

                choices = value.get("choices")
                if choices is None and isinstance(raw_usage, Mapping):
                    continue
                if not isinstance(choices, Sequence) or isinstance(
                    choices,
                    (str, bytes, bytearray),
                ):
                    raise ModelProtocolError("model returned invalid SSE choices")
                for choice in choices:
                    if not isinstance(choice, Mapping):
                        raise ModelProtocolError("model returned an invalid SSE choice")
                    if choice.get("finish_reason") is not None:
                        finish_reason = str(choice["finish_reason"])
                        saw_terminal_marker = True
                    delta = choice.get("delta") or choice.get("message") or {}
                    if not isinstance(delta, Mapping):
                        raise ModelProtocolError(
                            "model returned an invalid SSE message delta"
                        )
                    text = _content_text(delta.get("content"))
                    if text:
                        content_parts.append(text)
                        yield ModelChunk(delta=text, provider_metadata=metadata)
                    tool_calls.add(delta.get("tool_calls"))

        if not saw_terminal_marker:
            raise ModelProtocolError("model SSE stream ended without a terminal marker")
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
