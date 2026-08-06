from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel

from moduagent import (
    Agent,
    AgentConfig,
    AuthorizationDecision,
    InMemoryConversationStore,
    InMemoryDiagnosticSink,
    ModelRequest,
    ModelResponse,
    NoopEventSink,
    PydanticOutputCodec,
    RunLimits,
    SkillLimits,
    SkillRegistry,
    SkillSelectionResult,
    StandardExecutionProfile,
    ToolCall,
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


class _Falsey:
    def __bool__(self) -> bool:
        return False


class FalseyCheckpointStore(_Falsey):
    async def load(self, run_id: str) -> None:
        del run_id

    async def save(self, run_id: str, context: Any) -> None:
        del run_id, context

    async def delete(self, run_id: str) -> None:
        del run_id


class FalseyAuthorizer(_Falsey):
    async def authorize(self, *args: Any, **kwargs: Any) -> AuthorizationDecision:
        del args, kwargs
        return AuthorizationDecision.allow()


class FalseySkillSelector(_Falsey):
    async def select(self, request: Any) -> SkillSelectionResult:
        del request
        return SkillSelectionResult()


class FalseySkillLimits(SkillLimits):
    def __bool__(self) -> bool:
        return False


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


def test_create_accepts_common_persistence_and_observability_components() -> None:
    store = InMemoryConversationStore()
    event_sink = NoopEventSink()
    diagnostic_sink = InMemoryDiagnosticSink()
    agent = Agent.create(
        model=ScriptedModel([]),
        instructions="Answer.",
        conversation_store=store,
        event_sink=event_sink,
        diagnostic_sink=diagnostic_sink,
        tool_trace_mode="arguments",
    )

    assert agent.runtime.conversation_store is store
    assert agent.runtime.event_sink is event_sink
    assert agent.diagnostic_reporter is not None
    assert agent.diagnostic_reporter.sink is diagnostic_sink
    assert agent.config.tool_trace_mode == "arguments"
    assert agent.inspect().stream_policy["tool_trace_mode"] == "arguments"


def test_create_passes_production_composition_and_config_fields_through() -> None:
    checkpoint_store = FalseyCheckpointStore()
    authorizer = FalseyAuthorizer()
    registry = SkillRegistry()
    selector = FalseySkillSelector()
    skill_limits = FalseySkillLimits(
        max_active_skills=2,
        max_catalog_tokens=17,
    )
    diagnostic_sink = InMemoryDiagnosticSink()
    conversation_store = InMemoryConversationStore()

    assert not checkpoint_store
    assert not authorizer
    assert not registry
    assert not selector
    assert not skill_limits

    agent = Agent.create(
        model=ScriptedModel([]),
        name="production-agent",
        instructions="Answer safely.",
        model_options={"temperature": 0.1, "provider": {"seed": 7}},
        metadata={"environment": "production"},
        finalization_mode="always",
        stream_visibility="all",
        conversation_store=conversation_store,
        checkpoint_store=checkpoint_store,
        diagnostic_sink=diagnostic_sink,
        diagnostic_timeout_seconds=0.75,
        diagnostic_max_pending_deliveries=7,
        tool_authorizer=authorizer,
        skill_registry=registry,
        skill_selector=selector,
        skill_limits=skill_limits,
    )

    assert agent.config.model_options == {
        "temperature": 0.1,
        "provider": {"seed": 7},
    }
    assert agent.config.metadata == {"environment": "production"}
    assert agent.config.finalization_mode == "always"
    assert agent.config.stream_visibility == "all"
    assert agent.runtime.checkpoint_store is checkpoint_store
    assert agent.tool_executor.authorizer is authorizer
    assert agent.skill_runtime is not None
    assert agent.skill_runtime.registry is registry
    assert agent.skill_runtime.selector is selector
    assert agent.skill_runtime.limits is skill_limits
    assert agent.skill_runtime.limits.max_catalog_tokens == 17
    assert agent.diagnostic_reporter is not None
    assert agent.diagnostic_reporter.sink is diagnostic_sink
    assert agent.diagnostic_reporter.timeout_seconds == 0.75
    assert agent.diagnostic_reporter.max_pending_deliveries == 7
    assert agent.inspect().stream_policy["visibility"] == "all"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"model_options": []}, "model_options"),
        ({"metadata": []}, "metadata"),
        ({"finalization_mode": "sometimes"}, "finalization_mode"),
        ({"stream_visibility": "private"}, "stream_visibility"),
    ],
)
def test_create_validates_common_agent_config_fields(
    options: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        Agent.create(
            model=ScriptedModel([]),
            instructions="Answer.",
            **options,
        )


def test_create_validates_production_composition_settings() -> None:
    with pytest.raises(ValueError, match="skill_selector requires skill_registry"):
        Agent.create(
            model=ScriptedModel([]),
            instructions="Answer.",
            skill_selector=FalseySkillSelector(),
        )

    with pytest.raises(ValueError, match="diagnostic timeout_seconds"):
        Agent.create(
            model=ScriptedModel([]),
            instructions="Answer.",
            diagnostic_sink=InMemoryDiagnosticSink(),
            diagnostic_timeout_seconds=0,
        )

    with pytest.raises(ValueError, match="max_pending_deliveries"):
        Agent.create(
            model=ScriptedModel([]),
            instructions="Answer.",
            diagnostic_sink=InMemoryDiagnosticSink(),
            diagnostic_max_pending_deliveries=0,
        )


def test_create_rejects_an_invalid_tool_trace_mode() -> None:
    with pytest.raises(ValueError, match="tool_trace_mode"):
        Agent.create(
            model=ScriptedModel([]),
            instructions="Answer.",
            tool_trace_mode="raw",  # type: ignore[arg-type]
        )


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


def test_agent_result_convenience_properties_are_bounded_and_secret_safe() -> None:
    result = AgentResult(
        run_id="run-safe",
        output=None,
        messages=(),
        usage=Usage(),
        finish_reason=FinishReason.ERROR,
        metadata={
            "error_summary": {
                "category": "tool",
                "code": "failed",
                "retryable": False,
                "raw_provider_error": "must-not-leak",
            },
            "tool_trace": [
                {
                    "call_id": "call-1",
                    "tool_name": "lookup",
                    "success": False,
                    "attempts": 1,
                    "duration_seconds": 0.25,
                    "arguments": {
                        "query": "select 1",
                        "api_key": "must-not-leak",
                        "APIKey": "must-not-leak",
                        "ACCESS_TOKEN": "must-not-leak",
                        "nested": {"accessToken": "must-not-leak"},
                    },
                    "arguments_source": "validated",
                    "private": "must-not-leak",
                }
            ],
            "run_usage": {
                "model_turns": 2,
                "tool_calls": 1,
                "duration_seconds": 0.5,
                "private": "must-not-leak",
            },
        },
    )

    assert dict(result.error_summary) == {
        "category": "tool",
        "code": "failed",
        "retryable": False,
    }
    assert dict(result.run_usage) == {
        "model_turns": 2,
        "tool_calls": 1,
        "duration_seconds": 0.5,
    }
    assert len(result.tool_trace) == 1
    trace = result.tool_trace[0]
    assert "private" not in trace
    assert trace["arguments"]["query"] == "select 1"
    assert trace["arguments"]["api_key"] == "[REDACTED]"
    assert trace["arguments"]["APIKey"] == "[REDACTED]"
    assert trace["arguments"]["ACCESS_TOKEN"] == "[REDACTED]"
    assert trace["arguments"]["nested"]["accessToken"] == "[REDACTED]"
    with pytest.raises(TypeError):
        result.error_summary["code"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.run_usage["tool_calls"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        trace["arguments"]["query"] = "changed"  # type: ignore[index]


def test_coordinator_attaches_run_usage_and_tool_trace_on_success() -> None:
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""

        return a + b

    async def scenario() -> None:
        model = ScriptedModel(
            [
                ModelResponse(
                    Message.assistant(
                        None,
                        (ToolCall("call-1", "add", {"a": 2, "b": 3}),),
                    ),
                    usage=Usage(5, 2, 7),
                ),
                ModelResponse(
                    Message.assistant("5"),
                    usage=Usage(7, 1, 8),
                ),
            ]
        )
        agent = Agent.create(
            model=model,
            instructions="Use add for arithmetic.",
            tools=[add],
        )

        result = await agent.run("2 + 3")

        assert result.finish_reason is FinishReason.COMPLETED
        assert result.run_usage["model_turns"] == 2
        assert result.run_usage["tool_calls"] == 1
        assert result.run_usage["duration_seconds"] >= 0
        assert len(result.tool_trace) == 1
        assert result.tool_trace[0]["tool_name"] == "add"

    asyncio.run(scenario())


def test_coordinator_attaches_run_usage_on_failure() -> None:
    class FailingModel:
        async def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            raise RuntimeError("private provider failure")

    async def scenario() -> None:
        result = await Agent.create(
            model=FailingModel(),
            instructions="Answer.",
        ).run("Hello")

        assert result.finish_reason is FinishReason.ERROR
        assert result.run_usage["model_turns"] == 1
        assert result.run_usage["tool_calls"] == 0
        assert result.run_usage["duration_seconds"] >= 0

    asyncio.run(scenario())


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
