from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from moduagent import Agent
from moduagent.config import AgentConfig, RetryConfig
from moduagent.messages import Message
from moduagent.models import (
    ModelCapabilities,
    ModelChunk,
    ModelProtocolError,
    ModelRequest,
    ModelResponse,
    OpenAICompatibleClient,
    classify_model_error,
)
from moduagent.observability import InMemoryDiagnosticSink
from moduagent.runtime import EventType


def _http_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://model.invalid/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(
        f"HTTP {status_code}",
        request=request,
        response=response,
    )


@pytest.mark.parametrize(
    ("error", "retryable", "code"),
    [
        (TimeoutError("late"), True, "model_timeout"),
        (ConnectionError("disconnected"), True, "model_connection_error"),
        (httpx.ProxyError("proxy disconnected"), True, "model_connection_error"),
        (_http_error(408), True, "model_timeout"),
        (_http_error(503), True, "model_http_5xx"),
        (_http_error(400), False, "model_http_4xx"),
        (_http_error(429), False, "model_http_4xx"),
        (ModelProtocolError("bad response"), False, "model_protocol_error"),
        (httpx.RemoteProtocolError("invalid HTTP"), False, "model_protocol_error"),
        (json.JSONDecodeError("bad JSON", "{", 0), False, "model_protocol_error"),
        (ValueError("invalid request"), False, "model_request_invalid"),
        (TypeError("client bug"), False, "model_client_contract_error"),
        (RuntimeError("unknown"), False, "model_invocation_failed"),
    ],
)
def test_model_retry_classification_is_strict(
    error: BaseException,
    retryable: bool,
    code: str,
) -> None:
    classification = classify_model_error(error)

    assert classification.retryable is retryable
    assert classification.code == code


def test_model_retry_classification_preserves_typed_cause() -> None:
    try:
        try:
            raise ConnectionError("socket closed")
        except ConnectionError as cause:
            raise RuntimeError("adapter wrapper") from cause
    except RuntimeError as error:
        classification = classify_model_error(error)

    assert classification.retryable is True
    assert classification.code == "model_connection_error"


class _CompleteSequenceModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self, outcomes: list[BaseException | ModelResponse]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        del request
        raise AssertionError("stream should not be called")
        yield


def _agent(
    model: Any,
    *,
    sink: InMemoryDiagnosticSink | None = None,
) -> Agent:
    return Agent(
        config=AgentConfig(
            "strict-model-retry",
            "Answer safely.",
            retry=RetryConfig(
                max_attempts=3,
                initial_delay=0,
                max_delay=0,
            ),
        ),
        model=model,
        diagnostic_sink=sink,
    )


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (ConnectionError("PRIVATE-CONNECTION"), "model_connection_error"),
        (_http_error(408), "model_timeout"),
        (_http_error(503), "model_http_5xx"),
    ],
)
def test_complete_retries_only_typed_transient_failures(
    error: BaseException,
    expected_code: str,
) -> None:
    async def scenario() -> None:
        model = _CompleteSequenceModel(
            [error, ModelResponse(Message.assistant("recovered"))]
        )
        events = [event async for event in _agent(model).stream_all("run")]

        assert model.calls == 2
        retries = [event for event in events if event.type is EventType.RETRY]
        assert len(retries) == 1
        assert retries[0].data["retryable"] is True
        assert retries[0].data["code"] == expected_code
        assert "PRIVATE-CONNECTION" not in repr(retries[0].to_dict())
        assert events[-1].data["result"].output == "recovered"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (ModelProtocolError("PRIVATE-PROTOCOL"), "model_protocol_error"),
        (
            json.JSONDecodeError("PRIVATE-JSON", "{", 0),
            "model_protocol_error",
        ),
        (_http_error(400), "model_http_4xx"),
        (_http_error(429), "model_http_4xx"),
        (ValueError("PRIVATE-VALIDATION"), "model_request_invalid"),
        (TypeError("PRIVATE-PROGRAMMING"), "model_client_contract_error"),
        (RuntimeError("PRIVATE-UNKNOWN"), "model_invocation_failed"),
    ],
)
def test_complete_does_not_retry_terminal_failures(
    error: BaseException,
    expected_code: str,
) -> None:
    async def scenario() -> None:
        sink = InMemoryDiagnosticSink()
        model = _CompleteSequenceModel(
            [error, ModelResponse(Message.assistant("must not run"))]
        )
        events = [event async for event in _agent(model, sink=sink).stream_all("run")]

        assert model.calls == 1
        assert not any(event.type is EventType.RETRY for event in events)
        result = events[-1].data["result"]
        assert result.metadata["error_summary"]["retryable"] is False
        assert result.metadata["error_summary"]["code"] == expected_code
        assert "PRIVATE" not in repr(events[-1].to_dict())

    asyncio.run(scenario())


class _StreamSequenceModel:
    capabilities = ModelCapabilities(streaming=True)

    def __init__(self, outcomes: list[BaseException | ModelResponse]) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        del request
        raise AssertionError("complete should not be called")

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        del request
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        yield ModelChunk(response=outcome)


def test_stream_uses_the_same_retry_classification() -> None:
    async def scenario() -> None:
        transient = _StreamSequenceModel(
            [
                ConnectionError("disconnected"),
                ModelResponse(Message.assistant("recovered")),
            ]
        )
        transient_events = [
            event async for event in _agent(transient).stream_all("run")
        ]

        assert transient.calls == 2
        assert sum(event.type is EventType.RETRY for event in transient_events) == 1

        terminal = _StreamSequenceModel(
            [
                ModelProtocolError("invalid stream JSON"),
                ModelResponse(Message.assistant("must not run")),
            ]
        )
        terminal_events = [event async for event in _agent(terminal).stream_all("run")]

        assert terminal.calls == 1
        assert not any(event.type is EventType.RETRY for event in terminal_events)

    asyncio.run(scenario())


@pytest.mark.parametrize("diagnostics_enabled", [False, True])
def test_stream_does_not_retry_after_emitting_output(
    diagnostics_enabled: bool,
) -> None:
    class PartialStreamModel:
        capabilities = ModelCapabilities(streaming=True)

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            raise AssertionError("complete should not be called")

        async def stream(
            self,
            request: ModelRequest,
        ) -> AsyncIterator[ModelChunk]:
            del request
            self.calls += 1
            yield ModelChunk(delta="partial")
            raise ConnectionError("stream disconnected")

    async def scenario() -> None:
        sink = InMemoryDiagnosticSink() if diagnostics_enabled else None
        model = PartialStreamModel()
        events = [event async for event in _agent(model, sink=sink).stream_all("run")]

        assert model.calls == 1
        assert not any(event.type is EventType.RETRY for event in events)
        summary = events[-1].data["result"].metadata["error_summary"]
        assert summary["code"] == "model_connection_error"
        assert summary["retryable"] is False

    asyncio.run(scenario())


def test_openai_invalid_tool_argument_json_is_a_protocol_error() -> None:
    class MalformedTransport:
        async def post_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            del args, kwargs
            return {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": "{not-json",
                                    },
                                }
                            ]
                        }
                    }
                ]
            }

        async def stream_lines(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> AsyncIterator[str]:
            del args, kwargs
            raise AssertionError("stream should not be called")
            yield

    async def scenario() -> None:
        client = OpenAICompatibleClient(
            base_url="http://model.invalid/v1",
            model="test",
            transport=MalformedTransport(),
        )

        with pytest.raises(ModelProtocolError) as captured:
            await client.complete(ModelRequest((Message.user("run"),)))

        assert isinstance(captured.value.__cause__, json.JSONDecodeError)

    asyncio.run(scenario())
