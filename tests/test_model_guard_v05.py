from __future__ import annotations

import asyncio
import json

import pytest

from moduagent import (
    Agent,
    AgentRuntime,
    InMemoryCheckpointStore,
    InMemoryConversationStore,
    InMemorySkillSource,
    ModelSkillSelector,
    NoopEventSink,
    Plan,
    PlanAndExecutePolicy,
    PlanStep,
    RetryConfig,
    RunRequest,
    SkillRegistry,
    StandardDecisionPolicy,
    TextOutputCodec,
    ToolExecutor,
    tool,
)
from moduagent.config import AgentConfig, RunLimits
from moduagent.execution.base import ExecutionBudget
from moduagent.memory import (
    ModelConversationSummarizer,
    SummarizingConversationMemoryPolicy,
    TokenBudget,
)
from moduagent.messages import Message, ToolCall, Usage
from moduagent.models import ModelProtocolError, ModelRequest, ModelResponse
from moduagent.runtime.model_guard import (
    ModelNoProgressError,
    ModelTurnBudgetExceeded,
    NoProgressCircuitBreaker,
)


def _response(
    *,
    content: str | None = None,
    call_id: str = "call-1",
    argument: str = "private-tool-argument",
    input_tokens: int = 1,
    request_id: str = "request-1",
) -> ModelResponse:
    call = ToolCall(
        id=call_id,
        name="lookup",
        arguments={"query": argument},
    )
    return ModelResponse(
        message=Message.assistant(content, (call,)),
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=1,
            total_tokens=input_tokens + 1,
        ),
        finish_reason="tool_calls",
        provider_metadata={"request_id": request_id},
    )


def test_run_limits_append_model_guard_defaults_without_moving_old_fields() -> None:
    limits = RunLimits(3, 7, 45.0, True, 2, 4, 5, 6)

    assert limits.max_steps == 3
    assert limits.max_tool_calls == 7
    assert limits.timeout_seconds == 45.0
    assert limits.parallel_tool_calls is True
    assert limits.max_parallel_tools == 2
    assert limits.max_step_attempts == 4
    assert limits.max_replans == 5
    assert limits.max_tool_repair_attempts == 6
    assert limits.max_model_turns == 32
    assert limits.no_progress_model_turn_threshold == 3


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("max_model_turns", 0, ValueError),
        ("max_model_turns", False, TypeError),
        ("max_model_turns", 1.0, TypeError),
        ("no_progress_model_turn_threshold", 0, ValueError),
        ("no_progress_model_turn_threshold", 1, ValueError),
        ("no_progress_model_turn_threshold", False, TypeError),
        ("no_progress_model_turn_threshold", 1.0, TypeError),
    ],
)
def test_run_limits_validate_model_guard_fields(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        RunLimits(**{field_name: value})  # type: ignore[arg-type]


def test_execution_budget_carries_model_guard_limits_from_config() -> None:
    config = AgentConfig(
        name="guarded",
        instructions="Complete the request.",
        limits=RunLimits(
            max_model_turns=11,
            no_progress_model_turn_threshold=4,
        ),
    )

    budget = ExecutionBudget.from_config(config)

    assert budget.max_model_turns == 11
    assert budget.no_progress_model_turn_threshold == 4


def test_execution_budget_preserves_old_positional_arguments() -> None:
    budget = ExecutionBudget(1, 2, 3, 4, 5, True, 6)

    assert budget.parallel_tool_calls is True
    assert budget.max_parallel_tools == 6
    assert budget.max_model_turns == 32
    assert budget.no_progress_model_turn_threshold == 3


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("max_model_turns", 0, ValueError),
        ("max_model_turns", False, TypeError),
        ("no_progress_model_turn_threshold", 0, ValueError),
        ("no_progress_model_turn_threshold", 1, ValueError),
        ("no_progress_model_turn_threshold", False, TypeError),
    ],
)
def test_execution_budget_validates_model_guard_fields(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    values = {
        "max_steps": 1,
        "max_tool_calls": 1,
        "max_step_attempts": 1,
        "max_replans": 0,
        "max_tool_repair_attempts": 0,
        field_name: value,
    }
    with pytest.raises(error_type):
        ExecutionBudget(**values)  # type: ignore[arg-type]


def test_breaker_trips_on_third_identical_semantic_observation() -> None:
    breaker = NoProgressCircuitBreaker(
        max_model_turns=10,
        no_progress_model_turn_threshold=3,
    )
    secret_prompt = "private-prompt-value"
    secret_argument = "private-tool-argument"

    first = breaker.before_model_attempt(
        {"phase": "act", "secret": secret_prompt, "step": 1}
    )
    assert first.model_turns == 1
    assert (
        breaker.observe_model_response(
            _response(argument=secret_argument)
        ).no_progress_model_turns
        == 1
    )

    breaker.before_model_attempt({"step": 1, "secret": secret_prompt, "phase": "act"})
    assert (
        breaker.observe_model_response(
            _response(
                call_id="different-provider-call-id",
                argument=secret_argument,
                input_tokens=99,
                request_id="different-provider-request-id",
            )
        ).no_progress_model_turns
        == 2
    )

    breaker.before_model_attempt({"secret": secret_prompt, "phase": "act", "step": 1})
    with pytest.raises(ModelNoProgressError) as raised:
        breaker.observe_model_response(
            _response(
                call_id="third-provider-call-id",
                argument=secret_argument,
                input_tokens=100,
                request_id="third-provider-request-id",
            )
        )

    assert raised.value.code == "model_no_progress"
    assert raised.value.snapshot.no_progress_model_turns == 3
    retained = repr(
        [getattr(breaker, slot) for slot in NoProgressCircuitBreaker.__slots__]
    )
    diagnostic = repr((breaker.snapshot, raised.value, raised.value.args))
    assert secret_prompt not in retained
    assert secret_argument not in retained
    assert secret_prompt not in diagnostic
    assert secret_argument not in diagnostic


def test_changed_state_or_response_resets_no_progress_streak() -> None:
    breaker = NoProgressCircuitBreaker(
        max_model_turns=10,
        no_progress_model_turn_threshold=3,
    )

    breaker.before_model_attempt({"phase": "act", "revision": 1})
    breaker.observe_model_response(_response(argument="first"))
    breaker.before_model_attempt({"phase": "act", "revision": 1})
    assert (
        breaker.observe_model_response(
            _response(call_id="new-id", argument="first")
        ).no_progress_model_turns
        == 2
    )

    breaker.before_model_attempt({"phase": "act", "revision": 2})
    assert (
        breaker.observe_model_response(
            _response(call_id="newer-id", argument="first")
        ).no_progress_model_turns
        == 1
    )

    breaker.before_model_attempt({"phase": "act", "revision": 2})
    assert (
        breaker.observe_model_response(
            _response(call_id="latest-id", argument="changed")
        ).no_progress_model_turns
        == 1
    )


def test_mark_progress_clears_streak_and_reopens_no_progress_trip() -> None:
    breaker = NoProgressCircuitBreaker(
        max_model_turns=10,
        no_progress_model_turn_threshold=2,
    )
    state = {"phase": "repair", "step": "step-1"}

    breaker.before_model_attempt(state)
    breaker.observe_model_response(_response())
    breaker.before_model_attempt(state)
    with pytest.raises(ModelNoProgressError):
        breaker.observe_model_response(_response(call_id="new-id"))

    assert breaker.snapshot.tripped is True
    snapshot = breaker.mark_progress()
    assert snapshot.tripped is False
    assert snapshot.no_progress_model_turns == 0
    assert snapshot.model_turns == 2

    breaker.before_model_attempt(state)
    assert (
        breaker.observe_model_response(
            _response(call_id="after-progress")
        ).no_progress_model_turns
        == 1
    )


def test_model_turn_budget_counts_attempts_and_requires_full_reset() -> None:
    breaker = NoProgressCircuitBreaker(
        max_model_turns=1,
        no_progress_model_turn_threshold=3,
    )

    breaker.before_model_attempt({"phase": "plan"})
    breaker.observe_model_response(_response())
    with pytest.raises(ModelTurnBudgetExceeded) as raised:
        breaker.before_model_attempt({"phase": "finalize"})

    assert raised.value.code == "max_model_turns_exceeded"
    assert raised.value.snapshot.model_turns == 1
    assert raised.value.snapshot.remaining_model_turns == 0

    breaker.mark_progress()
    with pytest.raises(ModelTurnBudgetExceeded):
        breaker.before_model_attempt({"phase": "finalize"})

    snapshot = breaker.reset()
    assert snapshot.model_turns == 0
    assert snapshot.tripped is False
    breaker.before_model_attempt({"phase": "finalize"})


def test_failed_attempt_can_be_abandoned_without_resetting_turn_budget() -> None:
    breaker = NoProgressCircuitBreaker(
        max_model_turns=2,
        no_progress_model_turn_threshold=3,
    )

    breaker.before_model_attempt({"phase": "act"})
    snapshot = breaker.abandon_model_attempt()
    assert snapshot.model_turns == 1

    breaker.before_model_attempt({"phase": "act"})
    snapshot = breaker.observe_model_response(_response())
    assert snapshot.model_turns == 2
    assert snapshot.no_progress_model_turns == 1


def test_guard_state_round_trip_keeps_only_digests_and_counters() -> None:
    breaker = NoProgressCircuitBreaker(
        max_model_turns=5,
        no_progress_model_turn_threshold=3,
    )
    breaker.before_model_attempt({"phase": "act", "secret": "private-state"})
    breaker.observe_model_response(_response(argument="private-response"))

    state = breaker.to_state()
    encoded = repr(state)
    assert state["version"] == 2
    assert isinstance(state["salt"], str)
    assert len(state["salt"]) == 64
    assert "private-state" not in encoded
    assert "private-response" not in encoded

    restored = NoProgressCircuitBreaker.from_state(
        state,
        max_model_turns=5,
        no_progress_model_turn_threshold=3,
    )
    assert restored.snapshot == breaker.snapshot
    restored.before_model_attempt({"secret": "private-state", "phase": "act"})
    assert (
        restored.observe_model_response(
            _response(call_id="new-call", argument="private-response")
        ).no_progress_model_turns
        == 2
    )


def test_guard_state_salt_prevents_cross_run_observation_correlation() -> None:
    first = NoProgressCircuitBreaker(
        max_model_turns=5,
        no_progress_model_turn_threshold=3,
    )
    second = NoProgressCircuitBreaker(
        max_model_turns=5,
        no_progress_model_turn_threshold=3,
    )
    semantic_state = {"phase": "act", "step": "step-1"}
    response = _response(argument="same-low-entropy-value")

    for breaker in (first, second):
        breaker.before_model_attempt(semantic_state)
        breaker.observe_model_response(response)

    first_state = first.to_state()
    second_state = second.to_state()
    assert first_state["salt"] != second_state["salt"]
    assert (
        first_state["last_observation_digest"]
        != second_state["last_observation_digest"]
    )


def test_reset_rotates_guard_salt_for_a_new_run() -> None:
    breaker = NoProgressCircuitBreaker()
    first_salt = breaker.to_state()["salt"]

    breaker.reset()

    assert breaker.to_state()["salt"] != first_salt


def test_guard_state_rejects_invalid_or_inconsistent_values() -> None:
    valid = {
        "version": 2,
        "model_turns": 1,
        "no_progress_model_turns": 1,
        "salt": "b" * 64,
        "last_observation_digest": "a" * 64,
        "trip_reason": None,
    }
    with pytest.raises(ValueError, match="version"):
        NoProgressCircuitBreaker.from_state(
            {**valid, "version": 99},
            max_model_turns=5,
            no_progress_model_turn_threshold=3,
        )
    with pytest.raises(ValueError, match="version"):
        # Guard state v1 predates per-run salting and was never released.
        # Rejecting it avoids silently retaining cross-run-correlatable hashes.
        NoProgressCircuitBreaker.from_state(
            {key: value for key, value in valid.items() if key != "salt"}
            | {"version": 1},
            max_model_turns=5,
            no_progress_model_turn_threshold=3,
        )
    with pytest.raises(ValueError, match="salt"):
        NoProgressCircuitBreaker.from_state(
            {**valid, "salt": "not-a-durable-salt"},
            max_model_turns=5,
            no_progress_model_turn_threshold=3,
        )
    with pytest.raises(ValueError, match="digest"):
        NoProgressCircuitBreaker.from_state(
            {**valid, "last_observation_digest": "raw-response"},
            max_model_turns=5,
            no_progress_model_turn_threshold=3,
        )
    with pytest.raises(ValueError, match="exceeds model_turns"):
        NoProgressCircuitBreaker.from_state(
            {
                **valid,
                "model_turns": 1,
                "no_progress_model_turns": 2,
            },
            max_model_turns=5,
            no_progress_model_turn_threshold=3,
        )
    with pytest.raises(ValueError, match="invalid count"):
        NoProgressCircuitBreaker.from_state(
            {
                **valid,
                "model_turns": 2,
                "trip_reason": "model_turn_budget",
            },
            max_model_turns=5,
            no_progress_model_turn_threshold=3,
        )


def test_breaker_rejects_unpaired_calls_and_non_json_semantic_state() -> None:
    breaker = NoProgressCircuitBreaker()

    with pytest.raises(RuntimeError, match="no matching attempt"):
        breaker.observe_model_response(_response())
    with pytest.raises(TypeError, match="must be a mapping"):
        breaker.before_model_attempt(["not", "a", "mapping"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="JSON-like"):
        breaker.before_model_attempt({"unsupported": object()})

    breaker.before_model_attempt({"phase": "act"})
    with pytest.raises(RuntimeError, match="response or abandonment"):
        breaker.before_model_attempt({"phase": "act"})


@pytest.mark.parametrize(
    ("field_name", "value", "error_type"),
    [
        ("max_model_turns", 0, ValueError),
        ("max_model_turns", False, TypeError),
        ("no_progress_model_turn_threshold", 0, ValueError),
        ("no_progress_model_turn_threshold", 1, ValueError),
        ("no_progress_model_turn_threshold", False, TypeError),
    ],
)
def test_breaker_validates_limits(
    field_name: str,
    value: object,
    error_type: type[Exception],
) -> None:
    kwargs = {
        "max_model_turns": 3,
        "no_progress_model_turn_threshold": 2,
        field_name: value,
    }
    with pytest.raises(error_type):
        NoProgressCircuitBreaker(**kwargs)  # type: ignore[arg-type]


def test_agent_stops_before_retry_exceeds_max_model_turns() -> None:
    class TransientModel:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: object) -> ModelResponse:
            del request
            self.calls += 1
            raise ConnectionError("temporary")

    async def scenario() -> None:
        model = TransientModel()
        agent = Agent.create(
            model=model,
            instructions="Answer.",
            limits=RunLimits(max_model_turns=1),
            retry=RetryConfig(
                max_attempts=3,
                initial_delay=0,
                max_delay=0,
            ),
        )

        result = await agent.run("hello")

        assert result.finish_reason.value == "max_model_turns"
        assert result.metadata["error_summary"]["code"] == ("max_model_turns_exceeded")
        assert result.metadata["error_summary"]["model_turns"] == 1
        assert model.calls == 1

    asyncio.run(scenario())


def test_agent_no_progress_ignores_provider_call_ids_and_hides_payload() -> None:
    secret = "private-query"

    @tool
    def lookup(query: str) -> str:
        """Look up a value."""

        del query
        raise ValueError("invalid input")

    class RepeatingModel:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: object) -> ModelResponse:
            del request
            self.calls += 1
            call = ToolCall(
                id=f"provider-call-{self.calls}",
                name="lookup",
                arguments={"query": secret},
            )
            return ModelResponse(
                Message.assistant(None, (call,)),
                finish_reason="tool_calls",
            )

    async def scenario() -> None:
        model = RepeatingModel()
        agent = Agent.create(
            model=model,
            instructions="Use lookup.",
            tools=[lookup],
            limits=RunLimits(
                max_model_turns=10,
                no_progress_model_turn_threshold=2,
            ),
        )

        result = await agent.run("find it")

        assert result.finish_reason.value == "no_progress"
        assert result.metadata["error_summary"]["code"] == "model_no_progress"
        assert result.metadata["error_summary"]["no_progress_model_turns"] == 2
        assert model.calls == 2
        assert secret not in result.explain()
        assert secret not in repr(result.metadata["error_summary"])

    asyncio.run(scenario())


def test_agent_no_progress_stops_identical_successful_tool_loop() -> None:
    secret = "private-success-query"
    tool_calls = 0

    @tool
    def lookup(query: str) -> str:
        """Look up a value."""

        nonlocal tool_calls
        tool_calls += 1
        assert query == secret
        return "unchanged-result"

    class RepeatingModel:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: object) -> ModelResponse:
            del request
            self.calls += 1
            call = ToolCall(
                id=f"provider-call-{self.calls}",
                name="lookup",
                arguments={"query": secret},
            )
            return ModelResponse(
                Message.assistant(None, (call,)),
                finish_reason="tool_calls",
            )

    async def scenario() -> None:
        model = RepeatingModel()
        result = await Agent.create(
            model=model,
            instructions="Use lookup.",
            tools=[lookup],
            limits=RunLimits(
                max_model_turns=10,
                no_progress_model_turn_threshold=2,
            ),
        ).run("find it")

        assert result.finish_reason.value == "no_progress"
        assert result.metadata["error_summary"]["code"] == "model_no_progress"
        assert model.calls == 3
        assert tool_calls == 2
        assert secret not in result.explain()
        assert secret not in repr(result.metadata)

    asyncio.run(scenario())


def test_changed_successful_tool_result_counts_as_progress() -> None:
    tool_calls = 0

    @tool
    def lookup(query: str) -> str:
        """Look up a changing value."""

        nonlocal tool_calls
        del query
        tool_calls += 1
        return f"value-{tool_calls}"

    class ProgressingModel:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: object) -> ModelResponse:
            del request
            self.calls += 1
            if self.calls > 2:
                return ModelResponse(Message.assistant("done"))
            call = ToolCall(
                id=f"provider-call-{self.calls}",
                name="lookup",
                arguments={"query": "same-query"},
            )
            return ModelResponse(
                Message.assistant(None, (call,)),
                finish_reason="tool_calls",
            )

    async def scenario() -> None:
        model = ProgressingModel()
        result = await Agent.create(
            model=model,
            instructions="Use lookup until enough evidence is available.",
            tools=[lookup],
            limits=RunLimits(
                max_model_turns=10,
                no_progress_model_turn_threshold=2,
            ),
        ).run("find it")

        assert result.output == "done"
        assert result.finish_reason.value == "completed"
        assert model.calls == 3
        assert tool_calls == 2

    asyncio.run(scenario())


def test_identical_memory_summary_batches_count_as_progress() -> None:
    class OneRecordPerBatchCounter:
        async def count_request(self, request: ModelRequest) -> int:
            content = request.messages[-1].content or ""
            payload = json.loads(content.split("\n", 1)[1])
            return len(payload["conversation_record_fragments"])

    class AlwaysFitsCounter:
        async def count_request(self, request: ModelRequest) -> int:
            del request
            return 1

    class RepeatingSummaryModel:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            self.calls += 1
            return ModelResponse(Message.assistant("stable summary"))

    class FinalModel:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            self.calls += 1
            return ModelResponse(Message.assistant("final answer"))

    async def scenario() -> None:
        conversations = InMemoryConversationStore()
        await conversations.append(
            "summary-session",
            (
                Message.user("old question"),
                Message.assistant("old answer"),
            ),
        )
        summary_model = RepeatingSummaryModel()
        final_model = FinalModel()
        summarizer = ModelConversationSummarizer(
            model=summary_model,  # type: ignore[arg-type]
            token_counter=OneRecordPerBatchCounter(),
            max_input_tokens=1,
        )
        agent = Agent(
            config=AgentConfig(
                "summary-progress",
                "Answer from memory.",
                limits=RunLimits(
                    max_model_turns=10,
                    no_progress_model_turn_threshold=2,
                ),
            ),
            model=final_model,  # type: ignore[arg-type]
            conversation_store=conversations,
            conversation_memory_policy=SummarizingConversationMemoryPolicy(
                budget=TokenBudget(100_000),
                token_counter=AlwaysFitsCounter(),
                summarizer=summarizer,
                max_history_turns=0,
            ),
        )

        result = await agent.run(
            "new question",
            session_id="summary-session",
        )

        assert result.output == "final answer"
        assert result.finish_reason.value == "completed"
        assert summary_model.calls == 2
        assert final_model.calls == 1

    asyncio.run(scenario())


def test_public_agent_budget_includes_auxiliary_memory_model_calls() -> None:
    class AlwaysFitsCounter:
        async def count_request(self, request: ModelRequest) -> int:
            del request
            return 1

    class CountingModel:
        def __init__(self, answer: str) -> None:
            self.answer = answer
            self.calls = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            self.calls += 1
            return ModelResponse(Message.assistant(self.answer))

    def memory_policy(
        summary_model: CountingModel,
    ) -> SummarizingConversationMemoryPolicy:
        counter = AlwaysFitsCounter()
        return SummarizingConversationMemoryPolicy(
            budget=TokenBudget(100_000),
            token_counter=counter,
            summarizer=ModelConversationSummarizer(
                model=summary_model,  # type: ignore[arg-type]
                token_counter=counter,
            ),
            max_history_turns=0,
        )

    async def conversation() -> InMemoryConversationStore:
        store = InMemoryConversationStore()
        await store.append(
            "summary-budget",
            (
                Message.user("old question"),
                Message.assistant("old answer"),
            ),
        )
        return store

    async def scenario() -> None:
        coordinated_summary = CountingModel("summary")
        coordinated_final = CountingModel("final")
        coordinated = Agent(
            config=AgentConfig(
                "coordinated-budget",
                "Answer from memory.",
                limits=RunLimits(max_model_turns=1),
            ),
            model=coordinated_final,  # type: ignore[arg-type]
            conversation_store=await conversation(),
            conversation_memory_policy=memory_policy(coordinated_summary),
        )

        coordinated_result = await coordinated.run(
            "new question",
            session_id="summary-budget",
        )

        assert coordinated_result.finish_reason.value == "max_model_turns"
        assert coordinated_summary.calls == 1
        assert coordinated_final.calls == 0

        # Direct AgentRuntime construction is a documented legacy boundary:
        # its auxiliary memory client is not routed through the main-loop guard.
        direct_summary = CountingModel("summary")
        direct_final = CountingModel("final")
        direct = AgentRuntime(
            config=AgentConfig(
                "direct-budget-scope",
                "Answer from memory.",
                limits=RunLimits(max_model_turns=1),
            ),
            model=direct_final,  # type: ignore[arg-type]
            decision_policy=StandardDecisionPolicy(),
            tool_executor=ToolExecutor(()),
            conversation_store=await conversation(),
            conversation_memory_policy=memory_policy(direct_summary),
            output_codec=TextOutputCodec(),
            event_sink=NoopEventSink(),
        )

        direct_result = await direct.execute(
            RunRequest("new question", "summary-budget")
        )

        assert direct_result.finish_reason.value == "completed"
        assert direct_summary.calls == 1
        assert direct_final.calls == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "summary_outcome",
    ["protocol_error", "tool_calls", "blank"],
)
def test_public_agent_fails_immediately_on_memory_model_protocol_error(
    summary_outcome: str,
) -> None:
    class AlwaysFitsCounter:
        async def count_request(self, request: ModelRequest) -> int:
            del request
            return 1

    class InvalidSummaryModel:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            self.calls += 1
            if summary_outcome == "protocol_error":
                raise ModelProtocolError("PRIVATE-SUMMARY-PROTOCOL")
            if summary_outcome == "tool_calls":
                call = ToolCall("summary-call", "unexpected", {})
                return ModelResponse(Message.assistant(None, (call,)))
            return ModelResponse(Message.assistant("  "))

    class MainModel:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            self.calls += 1
            return ModelResponse(Message.assistant("must not run"))

    async def scenario() -> None:
        conversations = InMemoryConversationStore()
        await conversations.append(
            "memory-protocol",
            (
                Message.user("old question"),
                Message.assistant("old answer"),
            ),
        )
        counter = AlwaysFitsCounter()
        summary_model = InvalidSummaryModel()
        main_model = MainModel()
        agent = Agent(
            config=AgentConfig(
                "memory-protocol",
                "Answer from memory.",
                limits=RunLimits(max_model_turns=10),
                retry=RetryConfig(
                    max_attempts=3,
                    initial_delay=0,
                    max_delay=0,
                ),
            ),
            model=main_model,  # type: ignore[arg-type]
            conversation_store=conversations,
            conversation_memory_policy=SummarizingConversationMemoryPolicy(
                budget=TokenBudget(100_000),
                token_counter=counter,
                summarizer=ModelConversationSummarizer(
                    model=summary_model,  # type: ignore[arg-type]
                    token_counter=counter,
                ),
                max_history_turns=0,
            ),
        )

        result = await agent.run(
            "new question",
            session_id="memory-protocol",
        )

        assert result.finish_reason.value == "error"
        assert result.metadata["error_summary"]["code"] == "model_protocol_error"
        assert result.metadata["error_summary"]["retryable"] is False
        assert summary_model.calls == 1
        assert main_model.calls == 0
        assert "PRIVATE-SUMMARY-PROTOCOL" not in repr(result)

    asyncio.run(scenario())


def test_skill_selection_preserves_model_turn_guard_terminal() -> None:
    class SelectorModel:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            self.calls += 1
            raise ConnectionError("PRIVATE-SKILL-CONNECTION")

    class MainModel:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            self.calls += 1
            return ModelResponse(Message.assistant("must not run"))

    async def scenario() -> None:
        selector_model = SelectorModel()
        main_model = MainModel()
        registry = SkillRegistry.from_sources(
            InMemorySkillSource(
                {
                    "analysis": (
                        "---\n"
                        "name: analysis\n"
                        "description: Analyze the request.\n"
                        "---\n\n"
                        "Analyze carefully.\n"
                    )
                }
            )
        )
        agent = Agent(
            config=AgentConfig(
                "skill-guard",
                "Answer.",
                limits=RunLimits(max_model_turns=1),
                retry=RetryConfig(
                    max_attempts=3,
                    initial_delay=0,
                    max_delay=0,
                ),
            ),
            model=main_model,  # type: ignore[arg-type]
            skill_registry=registry,
            skill_selector=ModelSkillSelector(
                selector_model,  # type: ignore[arg-type]
            ),
        )

        result = await agent.run("analyze", skill_mode="auto")

        assert result.finish_reason.value == "max_model_turns"
        assert result.metadata["error_summary"]["code"] == ("max_model_turns_exceeded")
        assert result.metadata["error_summary"]["retryable"] is False
        assert result.metadata["error_summary"]["resumable"] is False
        assert result.metadata["error_summary"]["model_turns"] == 1
        assert result.metadata["error_summary"]["max_model_turns"] == 1
        assert selector_model.calls == 1
        assert main_model.calls == 0
        assert "PRIVATE-SKILL-CONNECTION" not in repr(result)

    asyncio.run(scenario())


def test_plan_replan_preserves_model_turn_guard_terminal() -> None:
    class MainModel:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            self.calls += 1
            return ModelResponse(
                Message.assistant(
                    '{"step_id":"S1","status":"blocked",'
                    '"missing_inputs":["new source"],'
                    '"completion_evidence":[]}'
                ),
                finish_reason="stop",
            )

    class ReplanModel:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: ModelRequest) -> ModelResponse:
            del request
            self.calls += 1
            return ModelResponse(Message.assistant("must not run"))

    class GatewayReplanGenerator:
        def __init__(self, replan_model: ReplanModel) -> None:
            self.plan = Plan(
                [
                    PlanStep(
                        step_id="S1",
                        objective="collect evidence",
                        completion_criteria=["evidence collected"],
                    )
                ]
            )
            self.replan_model = replan_model

        async def create(self, context: object) -> Plan:
            del context
            return self.plan

        async def revise(
            self,
            context: object,
            plan: Plan,
            feedback: str,
        ) -> Plan:
            del feedback
            gateway = getattr(context, "model_gateway")
            await gateway.complete(
                self.replan_model,
                ModelRequest((Message.user("revise"),)),
                phase="replan",
            )
            return plan

    async def scenario() -> None:
        main_model = MainModel()
        replan_model = ReplanModel()
        agent = Agent(
            config=AgentConfig(
                "replan-guard",
                "Execute the plan.",
                limits=RunLimits(
                    max_steps=1,
                    max_replans=1,
                    max_model_turns=1,
                ),
            ),
            model=main_model,  # type: ignore[arg-type]
            decision_policy=PlanAndExecutePolicy(GatewayReplanGenerator(replan_model)),
        )

        result = await agent.run("collect evidence")

        assert result.finish_reason.value == "max_model_turns"
        assert result.metadata["error_summary"]["code"] == ("max_model_turns_exceeded")
        assert result.metadata["error_summary"]["retryable"] is False
        assert result.metadata["error_summary"]["resumable"] is False
        assert result.metadata["error_summary"]["model_turns"] == 1
        assert result.metadata["error_summary"]["max_model_turns"] == 1
        assert main_model.calls == 1
        assert replan_model.calls == 0

    asyncio.run(scenario())


def _direct_runtime(
    model: object,
    *,
    config: AgentConfig,
    tools: tuple[object, ...] = (),
    checkpoint_store: InMemoryCheckpointStore | None = None,
) -> AgentRuntime:
    return AgentRuntime(
        config=config,
        model=model,  # type: ignore[arg-type]
        decision_policy=StandardDecisionPolicy(),
        tool_executor=ToolExecutor(tools),  # type: ignore[arg-type]
        conversation_store=InMemoryConversationStore(),
        output_codec=TextOutputCodec(),
        event_sink=NoopEventSink(),
        checkpoint_store=checkpoint_store,
    )


def test_direct_agent_runtime_stops_before_retry_exceeds_model_turns() -> None:
    class TransientModel:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: object) -> ModelResponse:
            del request
            self.calls += 1
            raise ConnectionError("PRIVATE-TRANSIENT-DETAIL")

    async def scenario() -> None:
        model = TransientModel()
        runtime = _direct_runtime(
            model,
            config=AgentConfig(
                "direct-budget",
                "Answer.",
                limits=RunLimits(max_model_turns=1),
                retry=RetryConfig(
                    max_attempts=3,
                    initial_delay=0,
                    max_delay=0,
                ),
            ),
        )

        result = await runtime.execute(RunRequest("hello", "direct-budget"))

        assert result.finish_reason.value == "max_model_turns"
        assert result.error == "model turn budget exhausted"
        assert result.metadata["error_summary"] == {
            "category": "limit",
            "code": "max_model_turns_exceeded",
            "retryable": False,
            "resumable": False,
            "model_turns": 1,
            "max_model_turns": 1,
            "no_progress_model_turns": 0,
            "no_progress_model_turn_threshold": 3,
        }
        assert model.calls == 1
        assert "PRIVATE-TRANSIENT-DETAIL" not in repr(result)

    asyncio.run(scenario())


def test_direct_agent_runtime_guard_terminal_cannot_be_resumed() -> None:
    class TransientModel:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: object) -> ModelResponse:
            del request
            self.calls += 1
            raise ConnectionError("temporary")

    async def scenario() -> None:
        model = TransientModel()
        checkpoints = InMemoryCheckpointStore()
        runtime = _direct_runtime(
            model,
            config=AgentConfig(
                "direct-non-resumable",
                "Answer.",
                limits=RunLimits(max_model_turns=1),
                retry=RetryConfig(
                    max_attempts=2,
                    initial_delay=0,
                    max_delay=0,
                ),
            ),
            checkpoint_store=checkpoints,
        )

        failed = await runtime.execute(RunRequest("hello", "direct-non-resumable"))
        checkpoint = await checkpoints.load(failed.run_id)

        assert failed.finish_reason.value == "max_model_turns"
        assert checkpoint is not None
        assert checkpoint.resume_safety == "not_resumable"
        assert model.calls == 1

        resumed = await runtime.execute(
            RunRequest(
                "",
                "direct-non-resumable",
                resume_run_id=failed.run_id,
            )
        )
        preserved = await checkpoints.load(failed.run_id)

        assert resumed.finish_reason.value == "error"
        assert resumed.error == ("checkpoint is not safely resumable: not_resumable")
        assert model.calls == 1
        assert preserved is not None
        assert preserved.resume_safety == "not_resumable"

    asyncio.run(scenario())


def test_direct_agent_runtime_counts_each_eligible_retry_attempt() -> None:
    class RetryOnceModel:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: object) -> ModelResponse:
            del request
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("temporary")
            return ModelResponse(Message.assistant("recovered"))

    async def scenario() -> None:
        model = RetryOnceModel()
        runtime = _direct_runtime(
            model,
            config=AgentConfig(
                "direct-retry-count",
                "Answer.",
                limits=RunLimits(max_model_turns=2),
                retry=RetryConfig(
                    max_attempts=2,
                    initial_delay=0,
                    max_delay=0,
                ),
            ),
        )

        events = [
            event
            async for event in runtime.stream(
                RunRequest("hello", "direct-retry-count"),
                include_internal=True,
            )
        ]

        started = [event for event in events if event.type.value == "model_started"]
        assert [event.data["model_turn"] for event in started] == [1, 2]
        assert model.calls == 2
        assert events[-1].data["result"].output == "recovered"

    asyncio.run(scenario())


def test_direct_agent_runtime_stops_identical_tool_decision_loop() -> None:
    secret = "PRIVATE-DIRECT-QUERY"
    tool_calls = 0

    @tool
    def lookup(query: str) -> str:
        """Return a stable lookup result."""

        nonlocal tool_calls
        tool_calls += 1
        assert query == secret
        return "stable-result"

    class RepeatingModel:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: object) -> ModelResponse:
            del request
            self.calls += 1
            call = ToolCall(
                id=f"provider-id-{self.calls}",
                name="lookup",
                arguments={"query": secret},
            )
            return ModelResponse(
                Message.assistant(None, (call,)),
                finish_reason="tool_calls",
            )

    async def scenario() -> None:
        model = RepeatingModel()
        runtime = _direct_runtime(
            model,
            config=AgentConfig(
                "direct-no-progress",
                "Use lookup.",
                limits=RunLimits(
                    max_steps=5,
                    max_model_turns=10,
                    no_progress_model_turn_threshold=2,
                ),
            ),
            tools=(lookup,),
        )

        result = await runtime.execute(RunRequest("find it", "direct-no-progress"))

        assert result.finish_reason.value == "no_progress"
        assert result.metadata["error_summary"]["code"] == "model_no_progress"
        assert result.metadata["error_summary"]["model_turns"] == 2
        assert model.calls == 2
        assert tool_calls == 1
        assert secret not in repr(result.metadata["error_summary"])

    asyncio.run(scenario())
