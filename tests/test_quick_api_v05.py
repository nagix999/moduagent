from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from moduagent import (
    Agent,
    AgentConfig,
    ModelRequest,
    ModelResponse,
    PydanticOutputCodec,
    RunLimits,
    StandardExecutionProfile,
)
from moduagent.errors import AgentRunError
from moduagent.messages import FinishReason, Message, Usage
from moduagent.models import VLLMClient
from moduagent.runtime import AgentResult
from moduagent.tools import function_tool, tool


class ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class Answer(BaseModel):
    summary: str


def response(text: str) -> ModelResponse:
    return ModelResponse(Message.assistant(text))


def test_create_and_ask_use_the_existing_standard_runtime() -> None:
    async def scenario() -> None:
        model = ScriptedModel([response("hello")])
        quick = Agent.create(
            model=model,
            name="assistant",
            instructions="Answer briefly.",
        )
        full = Agent(
            config=AgentConfig("assistant", "Answer briefly."),
            model=model,
            execution_profile=StandardExecutionProfile(),
        )

        assert quick.inspect().agent_fingerprint == full.inspect().agent_fingerprint
        assert quick.inspect().execution_profile.kind == "standard"
        assert await quick.ask("Hi", session_id="quick-session") == "hello"

    asyncio.run(scenario())


def test_create_wraps_a_pydantic_output_model() -> None:
    async def scenario() -> None:
        model = ScriptedModel([response('{"summary":"verified"}')])
        agent = Agent.create(
            model=model,
            instructions="Return a structured answer.",
            output=Answer,
        )

        answer = await agent.ask("Check this")

        assert answer == Answer(summary="verified")
        assert agent.inspect().output_contract["structured"] is True
        assert model.requests[0].output_schema is not None

    asyncio.run(scenario())


def test_create_accepts_an_existing_output_codec() -> None:
    codec = PydanticOutputCodec(Answer)
    agent = Agent.create(
        model=ScriptedModel([]),
        instructions="Return a structured answer.",
        output=codec,
    )

    assert agent.runtime.output_codec is codec


def test_create_plan_uses_the_resolved_run_limit_for_plan_generation() -> None:
    limits = RunLimits(max_steps=9)
    agent = Agent.create(
        model=ScriptedModel([]),
        instructions="Plan before acting.",
        execution="plan",
        limits=limits,
    )

    spec = agent.inspect()
    generator = spec.execution_profile.details["plan_generator"]
    assert spec.execution_profile.kind == "plan"
    assert generator["max_steps"] == 9
    assert spec.execution_profile.details["max_step_attempts"] == (
        limits.max_step_attempts
    )
    assert spec.execution_profile.details["max_replans"] == limits.max_replans


def test_create_preserves_an_explicit_execution_profile() -> None:
    profile = StandardExecutionProfile()
    agent = Agent.create(
        model=ScriptedModel([]),
        instructions="Use the supplied profile.",
        execution=profile,
    )

    assert agent.inspect().execution_profile.kind == "standard"


@pytest.mark.parametrize("execution", ["automatic", "", 1])
def test_create_rejects_unknown_execution_modes(execution: Any) -> None:
    expected = ValueError if isinstance(execution, str) else TypeError
    with pytest.raises(expected, match="execution"):
        Agent.create(
            model=ScriptedModel([]),
            instructions="Answer.",
            execution=execution,
        )


def test_create_rejects_an_invalid_output_contract() -> None:
    with pytest.raises(TypeError, match="Pydantic model class or an OutputCodec"):
        Agent.create(
            model=ScriptedModel([]),
            instructions="Answer.",
            output=dict,
        )


def test_agent_result_unwrap_and_explain_success() -> None:
    result = AgentResult(
        run_id="run-1",
        output="done",
        messages=(),
        usage=Usage(input_tokens=3, output_tokens=2, total_tokens=5),
        finish_reason=FinishReason.COMPLETED,
    )

    result.raise_for_error()
    assert result.unwrap() == "done"
    assert result.explain() == (
        "agent run completed "
        "(run_id=run-1, input_tokens=3, output_tokens=2, total_tokens=5)"
    )


def test_agent_result_failure_raises_a_sanitized_error() -> None:
    result = AgentResult(
        run_id="run-2",
        output={"must-not": "escape"},
        messages=(Message.user("private prompt"),),
        usage=Usage(),
        finish_reason=FinishReason.MAX_STEPS,
        error="private provider detail",
        metadata={
            "error_summary": {
                "category": "limit",
                "code": "max_steps_exceeded",
                "retryable": False,
                "resumable": True,
                "failure_id": "failure-1",
                "raw_arguments": {"api_key": "must-not-leak"},
            },
            "private": "must-not-leak",
        },
    )

    with pytest.raises(AgentRunError) as captured:
        result.unwrap()

    error = captured.value
    assert error.run_id == "run-2"
    assert error.finish_reason == "max_steps"
    assert error.failure_id == "failure-1"
    assert dict(error.error_summary) == {
        "category": "limit",
        "code": "max_steps_exceeded",
        "failure_id": "failure-1",
        "retryable": False,
        "resumable": True,
    }
    assert "must-not-leak" not in str(error)
    assert "private provider detail" not in str(error)
    assert result.explain() == str(error)
    with pytest.raises(TypeError):
        error.error_summary["code"] = "changed"  # type: ignore[index]


def test_ask_unwraps_runtime_failures() -> None:
    class FailingModel:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            raise RuntimeError("private provider failure")

    async def scenario() -> None:
        agent = Agent.create(
            model=FailingModel(),
            instructions="Answer.",
        )

        with pytest.raises(AgentRunError) as captured:
            await agent.ask("Hello")

        assert captured.value.finish_reason == "error"
        assert "private provider failure" not in str(captured.value)

    asyncio.run(scenario())


def test_tool_is_the_same_conservative_decorator_as_function_tool() -> None:
    assert tool is function_tool

    @tool(repair_safe=True)
    def lookup(value: str) -> str:
        """Return a value."""

        return value

    assert lookup.name == "lookup"
    assert lookup.repair_safe is True


class StubTransport:
    async def post_json(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("transport should not be called")

    def stream_json(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("transport should not be called")


def test_vllm_from_env_uses_only_the_documented_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MODEL", "company-model")
    monkeypatch.setenv("VLLM_BASE_URL", "http://vllm.internal:8000/v1")
    monkeypatch.setenv("VLLM_API_KEY", "private-token")

    client = VLLMClient.from_env(transport=StubTransport())

    assert client.model == "company-model"
    assert client.base_url == "http://vllm.internal:8000/v1"
    assert client.headers["Authorization"] == "Bearer private-token"


def test_vllm_from_env_requires_a_model_and_defaults_the_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_MODEL", raising=False)
    with pytest.raises(ValueError, match="VLLM_MODEL"):
        VLLMClient.from_env(transport=StubTransport())

    monkeypatch.setenv("VLLM_MODEL", "local-model")
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    client = VLLMClient.from_env(transport=StubTransport())

    assert client.base_url == "http://localhost:8000/v1"
    assert "Authorization" not in client.headers


def test_vllm_from_env_reads_and_validates_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MODEL", "local-model")
    monkeypatch.setenv("VLLM_TIMEOUT", "12.5")

    client = VLLMClient.from_env(transport=StubTransport())
    assert client.timeout == 12.5

    explicit = VLLMClient.from_env(
        transport=StubTransport(),
        timeout=7,
    )
    assert explicit.timeout == 7

    monkeypatch.setenv("VLLM_TIMEOUT", "invalid")
    with pytest.raises(ValueError, match="VLLM_TIMEOUT"):
        VLLMClient.from_env(transport=StubTransport())

    for invalid in ("nan", "inf", "-inf", "0", "-1"):
        monkeypatch.setenv("VLLM_TIMEOUT", invalid)
        with pytest.raises(ValueError, match="finite and positive"):
            VLLMClient.from_env(transport=StubTransport())


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), float("-inf")])
def test_vllm_rejects_non_finite_explicit_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        VLLMClient(
            base_url="http://localhost:8000/v1",
            model="local-model",
            timeout=timeout,
            transport=StubTransport(),
        )
