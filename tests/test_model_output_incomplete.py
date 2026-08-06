from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import BaseModel

from moduagent import (
    Agent,
    AgentRunError,
    ModelCapabilities,
    ModelOutputIncompleteError,
    ModelRequest,
    ModelResponse,
    RetryConfig,
    tool,
)
from moduagent.messages import FinishReason, Message
from moduagent.models import classify_model_error
from moduagent.observability import InMemoryDiagnosticSink


class _FinalAnswer(BaseModel):
    answer: str


@tool
def read_evidence() -> str:
    """Return deterministic evidence without changing external state."""

    return "verified"


class _SequenceModel:
    capabilities = ModelCapabilities(
        streaming=False,
        tool_calling_with_structured_output=False,
    )

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def _agent_for_final_response(
    final_response: ModelResponse,
    *,
    diagnostic_sink: InMemoryDiagnosticSink | None = None,
) -> tuple[Agent, _SequenceModel]:
    model = _SequenceModel(
        [
            ModelResponse(Message.assistant("Evidence is ready.")),
            final_response,
            ModelResponse(Message.assistant('{"answer":"must not retry"}')),
        ]
    )
    agent = Agent.create(
        model=model,
        instructions="Use verified evidence and return the requested output.",
        tools=[read_evidence],
        output=_FinalAnswer,
        retry=RetryConfig(max_attempts=3, initial_delay=0, max_delay=0),
        diagnostic_sink=diagnostic_sink,
    )
    return agent, model


@pytest.mark.parametrize("provider_finish_reason", ["timeout", "length", "max_tokens"])
@pytest.mark.parametrize("diagnostics_enabled", [False, True])
def test_incomplete_final_output_is_safe_specific_and_never_retried(
    provider_finish_reason: str,
    diagnostics_enabled: bool,
) -> None:
    async def scenario() -> None:
        private_body = "PRIVATE-PARTIAL-MODEL-BODY"
        private_metadata = "PRIVATE-PROVIDER-METADATA"
        sink = InMemoryDiagnosticSink() if diagnostics_enabled else None
        agent, model = _agent_for_final_response(
            ModelResponse(
                Message.assistant(private_body),
                finish_reason=provider_finish_reason,
                provider_metadata={"raw_body": private_metadata},
            ),
            diagnostic_sink=sink,
        )

        result = await agent.run("Return the verified result.")

        assert len(model.requests) == 2
        assert model.requests[0].tools
        assert model.requests[0].output_schema is None
        assert model.requests[1].tools == ()
        assert model.requests[1].output_schema is not None
        assert result.finish_reason is FinishReason.ERROR
        assert result.error == "model output incomplete"
        assert result.output is None
        assert result.error_summary["category"] == "model_protocol"
        assert result.error_summary["code"] == "model_output_incomplete"
        assert result.error_summary["retryable"] is False
        assert result.error_summary["provider_finish_reason"] == provider_finish_reason

        with pytest.raises(AgentRunError) as captured:
            result.raise_for_error()
        public_error = captured.value
        assert public_error.provider_finish_reason == provider_finish_reason
        assert f"provider_finish_reason={provider_finish_reason}" in str(public_error)
        assert private_body not in str(public_error)
        assert private_metadata not in str(public_error)

        serialized = json.dumps(
            {
                "error": result.error,
                "metadata": result.metadata,
                "messages": [message.to_dict() for message in result.messages],
            },
            ensure_ascii=False,
            default=str,
        )
        assert private_body not in serialized
        assert private_metadata not in serialized

        if sink is None:
            assert result.failure_id is None
        else:
            assert result.failure_id is not None
            assert len(sink.records) == 1
            record = sink.records[0]
            assert record.category == "model_protocol"
            assert record.code == "model_output_incomplete"
            assert record.retryable is False
            assert record.safe_details == {
                "provider_finish_reason": provider_finish_reason
            }
            diagnostic_json = json.dumps(record.to_dict(), ensure_ascii=False)
            assert private_body not in diagnostic_json
            assert private_metadata not in diagnostic_json

    asyncio.run(scenario())


def test_successful_final_output_has_no_incomplete_error_metadata() -> None:
    async def scenario() -> None:
        agent, model = _agent_for_final_response(
            ModelResponse(
                Message.assistant('{"answer":"verified"}'),
                finish_reason="stop",
            )
        )

        result = await agent.run("Return the verified result.")

        assert len(model.requests) == 2
        assert result.finish_reason is FinishReason.COMPLETED
        assert result.output == _FinalAnswer(answer="verified")
        assert result.error_summary == {}
        assert "provider_finish_reason" not in result.metadata

    asyncio.run(scenario())


@pytest.mark.parametrize("finish_reason", ["timeout", "length", "max_tokens"])
def test_incomplete_error_accepts_only_safe_provider_finish_reasons(
    finish_reason: str,
) -> None:
    error = ModelOutputIncompleteError(finish_reason)
    classification = classify_model_error(error)

    assert error.finish_reason == finish_reason
    assert str(error) == "model output incomplete"
    assert classification.category == "model_protocol"
    assert classification.code == "model_output_incomplete"
    assert classification.retryable is False
    with pytest.raises(AttributeError):
        error.finish_reason = "PRIVATE-REASON"  # type: ignore[misc]


@pytest.mark.parametrize(
    "finish_reason",
    ["", "stop", "TIMEOUT", "timeout private", "PRIVATE-REASON", None, 1],
)
def test_incomplete_error_rejects_arbitrary_provider_values(
    finish_reason: Any,
) -> None:
    expected = TypeError if not isinstance(finish_reason, str) else ValueError
    with pytest.raises(expected):
        ModelOutputIncompleteError(finish_reason)


def test_agent_run_error_drops_an_untrusted_provider_finish_reason() -> None:
    error = AgentRunError(
        run_id="run-1",
        finish_reason="error",
        error_summary={
            "category": "model_protocol",
            "code": "model_output_incomplete",
            "provider_finish_reason": "PRIVATE-REASON",
        },
    )

    assert error.provider_finish_reason is None
    assert "provider_finish_reason" not in error.error_summary
    assert "PRIVATE-REASON" not in str(error)
