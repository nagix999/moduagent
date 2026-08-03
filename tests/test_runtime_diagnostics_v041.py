from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import fields
from typing import Any

from pydantic import BaseModel

from moduagent import (
    Agent,
    AgentConfig,
    EventType,
    FinishReason,
    InMemoryDiagnosticSink,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    NoopDiagnosticSink,
    Plan,
    PlanAndExecutePolicy,
    PlanStep,
    PydanticOutputCodec,
    RetryConfig,
    StepResult,
    StepValidation,
    ToolCall,
    ToolFailureRecoveryConfig,
    function_tool,
)
from moduagent.runtime.context import RunContext


def test_run_context_appends_diagnostic_fields_after_040_positional_fields() -> None:
    field_names = tuple(item.name for item in fields(RunContext))

    assert field_names[-4:] == (
        "created_at",
        "diagnostic_reporter",
        "primary_failure",
        "tool_failure_ids",
    )


class ScriptedModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self, items: list[ModelResponse | BaseException]) -> None:
        self.items = items
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        item = self.items.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class StaticPlanGenerator:
    async def create(self, context: Any) -> Plan:
        del context
        return Plan(
            [
                PlanStep(
                    step_id="S1",
                    objective="produce a verified result",
                    completion_criteria=["result is verified"],
                )
            ]
        )

    async def revise(self, context: Any, plan: Plan, feedback: str) -> Plan:
        del context, feedback
        return plan


class ExplodingValidator:
    def validate(self, step: PlanStep, result: StepResult) -> StepValidation:
        del step, result
        raise RuntimeError("PRIVATE-VALIDATOR-DETAIL")


class ExplodingPlanGenerator:
    async def create(self, context: Any) -> Plan:
        del context
        raise RuntimeError("PRIVATE-PLAN-GENERATOR-DETAIL")

    async def revise(self, context: Any, plan: Plan, feedback: str) -> Plan:
        del context, plan, feedback
        raise AssertionError("revise must not be called")


def run(awaitable: Any) -> Any:
    return asyncio.run(awaitable)


def test_terminal_model_failure_has_one_correlated_diagnostic() -> None:
    class Response:
        status_code = 503

    class ProviderError(RuntimeError):
        response = Response()

    async def scenario() -> None:
        sink = InMemoryDiagnosticSink()
        agent = Agent(
            config=AgentConfig(
                "diagnostic-model",
                "Answer.",
                retry=RetryConfig(max_attempts=1),
            ),
            model=ScriptedModel([ProviderError("PRIVATE-PROVIDER-BODY bearer SECRET")]),
            diagnostic_sink=sink,
        )

        result = await agent.run("fail")

        assert result.error == "model invocation failed"
        assert result.failure_id is not None
        assert result.metadata["error_summary"]["component"] == "model"
        assert result.metadata["error_summary"]["operation"] == "complete"
        assert result.metadata["error_summary"]["attempt"] == 1
        assert len(sink.records) == 1
        record = sink.get(result.failure_id)
        assert record is not None
        assert record.exception_type == "ProviderError"
        assert record.safe_details["http_status"] == 503
        assert "PRIVATE-PROVIDER-BODY" not in json.dumps(record.to_dict())

    run(scenario())


def test_initial_plan_generation_failure_has_policy_phase_diagnostic() -> None:
    async def scenario() -> None:
        sink = InMemoryDiagnosticSink()
        agent = Agent(
            config=AgentConfig("diagnostic-plan-create", "Plan first."),
            model=ScriptedModel([]),
            decision_policy=PlanAndExecutePolicy(ExplodingPlanGenerator()),
            diagnostic_sink=sink,
        )

        result = await agent.run("fail during planning")

        assert result.failure_id is not None
        assert result.metadata["error_summary"] == {
            "category": "planning",
            "code": "plan_generation_failed",
            "retryable": False,
            "resumable": False,
            "failure_id": result.failure_id,
            "component": "policy",
            "operation": "create_plan",
            "phase": "plan",
            "attempt": 1,
        }
        record = sink.get(result.failure_id)
        assert record is not None
        assert record.component == "policy"
        assert record.operation == "create_plan"
        assert record.phase == "plan"
        assert record.code == "plan_generation_failed"
        assert "PRIVATE-PLAN-GENERATOR-DETAIL" not in json.dumps(record.to_dict())

    run(scenario())


def test_pydantic_output_failure_records_locations_without_input() -> None:
    class Answer(BaseModel):
        total: int

    async def scenario() -> None:
        sink = InMemoryDiagnosticSink()
        agent = Agent(
            config=AgentConfig(
                "diagnostic-output",
                "Return structured output.",
                retry=RetryConfig(max_attempts=1),
            ),
            model=ScriptedModel(
                [ModelResponse(Message.assistant('{"total":"PRIVATE-VALUE"}'))]
            ),
            output_codec=PydanticOutputCodec(Answer),
            diagnostic_sink=sink,
        )

        result = await agent.run("decode")

        assert result.error == "run failed"
        assert result.metadata["error_summary"]["category"] == "output_validation"
        assert result.metadata["error_summary"]["code"] == ("output_validation_failed")
        assert result.failure_id is not None
        record = sink.get(result.failure_id)
        assert record is not None
        assert record.component == "output"
        assert record.operation == "decode"
        assert record.safe_details["validation_errors"][0]["loc"] == ("total",)
        assert "PRIVATE-VALUE" not in json.dumps(record.to_dict())

    run(scenario())


def test_output_codec_failure_has_stable_classification_when_diagnostics_off() -> None:
    class Answer(BaseModel):
        total: int

    async def scenario() -> None:
        default = Agent(
            config=AgentConfig("default-output-error", "Return structured output."),
            model=ScriptedModel(
                [ModelResponse(Message.assistant('{"total":"invalid"}'))]
            ),
            output_codec=PydanticOutputCodec(Answer),
        )
        explicit_noop = Agent(
            config=AgentConfig("noop-output-error", "Return structured output."),
            model=ScriptedModel(
                [ModelResponse(Message.assistant('{"total":"invalid"}'))]
            ),
            output_codec=PydanticOutputCodec(Answer),
            diagnostic_sink=NoopDiagnosticSink(),
        )

        for result in (await default.run("decode"), await explicit_noop.run("decode")):
            assert result.error == "run failed"
            assert result.failure_id is None
            assert result.metadata["error_summary"] == {
                "category": "output_validation",
                "code": "output_validation_failed",
                "retryable": False,
                "resumable": False,
            }

    run(scenario())


def test_recovered_tool_failure_is_correlated_in_event_and_tool_trace() -> None:
    class DatabaseSyntaxError(RuntimeError):
        sqlstate = "42601"

    @function_tool
    def query_db() -> None:
        """Run a read-only database query."""

        raise DatabaseSyntaxError("SELECT PRIVATE_SQL FROM secret_table")

    call = ToolCall("call-db", "query_db", {})

    async def scenario() -> None:
        sink = InMemoryDiagnosticSink()
        agent = Agent(
            config=AgentConfig("diagnostic-tool", "Use query_db."),
            model=ScriptedModel(
                [
                    ModelResponse(Message.assistant(None, (call,)), (call,)),
                    ModelResponse(Message.assistant("The query could not run.")),
                ]
            ),
            tools=[query_db],
            diagnostic_sink=sink,
        )
        assert agent.diagnostic_reporter is not None
        agent.diagnostic_reporter._failure_id_factory = (  # type: ignore[attr-defined]
            lambda: "failure-tool-without-prefix"
        )

        events = [event async for event in agent.stream_all("query")]
        result = events[-1].data["result"]
        tool_event = next(
            event for event in events if event.type is EventType.TOOL_COMPLETED
        )

        assert result.output == "The query could not run."
        assert result.failure_id is None
        failure_id = result.metadata["tool_trace"][0]["failure_id"]
        assert failure_id == "failure-tool-without-prefix"
        assert tool_event.data["failure"]["failure_id"] == failure_id
        record = sink.get(failure_id)
        assert record is not None
        assert record.call_id == "call-db"
        assert record.tool_name == "query_db"
        assert record.safe_details["sqlstate"] == "42601"
        assert "PRIVATE_SQL" not in json.dumps(record.to_dict())

    run(scenario())


def test_legacy_diagnostic_ref_is_never_projected_as_a_failure_id() -> None:
    class diag_LegacyError(RuntimeError):
        pass

    @function_tool
    def broken_tool() -> None:
        """Fail with a legacy exception type that resembles a diagnostic ID."""

        raise diag_LegacyError("PRIVATE-LEGACY-DETAIL")

    call = ToolCall("call-legacy-ref", "broken_tool", {})

    async def scenario() -> None:
        agent = Agent(
            config=AgentConfig("legacy-diagnostic-ref", "Use broken_tool."),
            model=ScriptedModel(
                [
                    ModelResponse(Message.assistant(None, (call,)), (call,)),
                    ModelResponse(Message.assistant("The tool could not run.")),
                ]
            ),
            tools=[broken_tool],
        )

        events = [event async for event in agent.stream_all("run")]
        result = events[-1].data["result"]
        tool_event = next(
            event for event in events if event.type is EventType.TOOL_COMPLETED
        )

        assert result.failure_id is None
        assert "failure_id" not in result.metadata["tool_trace"][0]
        assert "failure_id" not in tool_event.data["failure"]

    run(scenario())


def test_terminal_plan_tool_failure_is_the_result_failure_id() -> None:
    class DatabaseSyntaxError(RuntimeError):
        sqlstate = "42601"

    @function_tool
    def query_db() -> None:
        """Run a read-only database query."""

        raise DatabaseSyntaxError("SELECT PRIVATE_SQL FROM secret_table")

    class QueryPlanGenerator:
        async def create(self, context: Any) -> Plan:
            del context
            return Plan(
                [
                    PlanStep(
                        step_id="S1",
                        objective="query the database",
                        completion_criteria=["a verified result exists"],
                        allowed_tools=["query_db"],
                    )
                ]
            )

        async def revise(
            self,
            context: Any,
            plan: Plan,
            feedback: str,
        ) -> Plan:
            del context, feedback
            return plan

    call = ToolCall("call-db-terminal", "query_db", {})

    async def scenario() -> None:
        sink = InMemoryDiagnosticSink()
        agent = Agent(
            config=AgentConfig("diagnostic-terminal-tool", "Use query_db."),
            model=ScriptedModel(
                [
                    ModelResponse(
                        Message.assistant(None, (call,)),
                        (call,),
                        finish_reason="tool_calls",
                    )
                ]
            ),
            tools=[query_db],
            decision_policy=PlanAndExecutePolicy(
                QueryPlanGenerator(),
                tool_failure_recovery=ToolFailureRecoveryConfig(fallback="fail"),
            ),
            diagnostic_sink=sink,
        )

        result = await agent.run("query")

        assert result.finish_reason is FinishReason.ERROR
        assert result.failure_id is not None
        assert result.metadata["error_summary"]["component"] == "tool"
        assert result.metadata["error_summary"]["operation"] == "invoke"
        assert result.metadata["tool_trace"][0]["failure_id"] == result.failure_id
        record = sink.get(result.failure_id)
        assert record is not None
        assert record.call_id == "call-db-terminal"
        assert record.safe_details["sqlstate"] == "42601"

    run(scenario())


def test_outer_batch_failure_has_only_batch_level_correlation() -> None:
    @function_tool
    def first_tool() -> str:
        return "first"

    @function_tool
    def second_tool() -> str:
        return "second"

    calls = (
        ToolCall("batch-1", "first_tool", {}),
        ToolCall("batch-2", "second_tool", {}),
    )
    executor_error = RuntimeError("PRIVATE-BATCH-EXECUTOR-DETAIL")

    async def scenario() -> None:
        sink = InMemoryDiagnosticSink()
        agent = Agent(
            config=AgentConfig("diagnostic-batch", "Use both tools."),
            model=ScriptedModel(
                [
                    ModelResponse(
                        Message.assistant(None, calls),
                        calls,
                        finish_reason="tool_calls",
                    )
                ]
            ),
            tools=[first_tool, second_tool],
            diagnostic_sink=sink,
        )

        class ExplodingBatchExecutor:
            registry = agent.tool_executor.registry

            async def execute_batch(self, *args: Any, **kwargs: Any) -> None:
                del args, kwargs
                raise executor_error

        agent.runtime.tool_executor = ExplodingBatchExecutor()
        events = [event async for event in agent.stream_all("run both")]
        result = events[-1].data["result"]
        tool_events = [
            event for event in events if event.type is EventType.TOOL_COMPLETED
        ]

        assert result.finish_reason is FinishReason.ERROR
        assert result.failure_id is not None
        assert result.metadata["error_summary"]["component"] == "tool"
        assert result.metadata["error_summary"]["operation"] == "execute_batch"
        assert len(sink.records) == 1
        record = sink.get(result.failure_id)
        assert record is not None
        assert record.operation == "execute_batch"
        assert record.call_id is None
        assert record.tool_name is None
        assert "RuntimeError" in record.cause_types
        assert all("failure_id" not in entry for entry in result.metadata["tool_trace"])
        assert len(tool_events) == 2
        assert all("failure_id" not in event.data["failure"] for event in tool_events)
        assert "PRIVATE-BATCH-EXECUTOR-DETAIL" not in json.dumps(record.to_dict())

    run(scenario())


def test_step_validator_exception_is_terminally_correlated() -> None:
    payload = json.dumps(
        {
            "step_id": "S1",
            "status": "completed",
            "facts": ["done"],
            "completion_evidence": ["verified"],
        }
    )

    async def scenario() -> None:
        sink = InMemoryDiagnosticSink()
        agent = Agent(
            config=AgentConfig("diagnostic-policy", "Execute the plan."),
            model=ScriptedModel([ModelResponse(Message.assistant(payload))]),
            decision_policy=PlanAndExecutePolicy(
                StaticPlanGenerator(),
                step_validator=ExplodingValidator(),
            ),
            diagnostic_sink=sink,
        )

        result = await agent.run("validate")

        assert result.error == "Step validation failed"
        assert result.failure_id is not None
        summary = result.metadata["error_summary"]
        assert summary["category"] == "step_validation"
        assert summary["code"] == "step_validator_failed"
        assert summary["operation"] == "validate_step"
        assert summary["step_id"] == "S1"
        record = sink.get(result.failure_id)
        assert record is not None
        assert record.exception_type == "RuntimeError"
        assert "PRIVATE-VALIDATOR-DETAIL" not in json.dumps(record.to_dict())

    run(scenario())


def test_diagnostics_are_default_off_and_do_not_change_fingerprint() -> None:
    async def scenario() -> None:
        configured = Agent(
            config=AgentConfig(
                "compatibility",
                "Answer.",
                retry=RetryConfig(max_attempts=1),
            ),
            model=ScriptedModel([RuntimeError("private")]),
            diagnostic_sink=InMemoryDiagnosticSink(),
        )
        default = Agent(
            config=AgentConfig(
                "compatibility",
                "Answer.",
                retry=RetryConfig(max_attempts=1),
            ),
            model=ScriptedModel([RuntimeError("private")]),
        )

        configured_result = await configured.run("fail")
        default_result = await default.run("fail")

        assert configured.inspect().agent_fingerprint == (
            default.inspect().agent_fingerprint
        )
        assert configured_result.failure_id is not None
        assert default_result.failure_id is None
        assert default_result.metadata["error_summary"] == {
            "category": "model_invocation",
            "code": "model_invocation_failed",
            "retryable": False,
            "resumable": False,
            "component": "model",
            "operation": "complete",
            "phase": "act",
            "attempt": 1,
        }

    run(scenario())


def test_agent_configures_bounded_diagnostic_delivery() -> None:
    sink = InMemoryDiagnosticSink()
    agent = Agent(
        config=AgentConfig("diagnostic-delivery", "Answer."),
        model=ScriptedModel([ModelResponse(Message.assistant("unused"))]),
        diagnostic_sink=sink,
        diagnostic_timeout_seconds=0.75,
        diagnostic_max_pending_deliveries=7,
    )

    assert agent.diagnostic_reporter is not None
    assert agent.diagnostic_reporter.timeout_seconds == 0.75
    assert agent.diagnostic_reporter.max_pending_deliveries == 7
    assert agent.spec.stream_policy["diagnostic_timeout_seconds"] == 0.75
    assert agent.spec.stream_policy["diagnostic_max_pending_deliveries"] == 7


def test_blocking_sync_event_sink_cannot_stall_an_agent_run() -> None:
    release = threading.Event()

    class BlockingSyncSink:
        def publish(self, event: Any) -> None:
            del event
            release.wait(timeout=2)

    async def scenario() -> None:
        agent = Agent(
            config=AgentConfig("blocking-event-sink", "Answer."),
            model=ScriptedModel([ModelResponse(Message.assistant("completed safely"))]),
            event_sink=BlockingSyncSink(),  # type: ignore[arg-type]
        )
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            result = await asyncio.wait_for(agent.run("answer"), timeout=1)
        finally:
            release.set()

        assert result.finish_reason is FinishReason.COMPLETED
        assert loop.time() - started < 0.75

    run(scenario())


def test_cancellation_resistant_async_event_sink_is_detached_after_timeout() -> None:
    async def scenario() -> None:
        release = asyncio.Event()

        class StubbornAsyncSink:
            async def publish(self, event: Any) -> None:
                del event
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    await release.wait()

        agent = Agent(
            config=AgentConfig("stubborn-event-sink", "Answer."),
            model=ScriptedModel([ModelResponse(Message.assistant("completed safely"))]),
            event_sink=StubbornAsyncSink(),
        )
        loop = asyncio.get_running_loop()
        started = loop.time()
        result = await asyncio.wait_for(agent.run("answer"), timeout=1)
        elapsed = loop.time() - started
        release.set()
        await asyncio.sleep(0)

        assert result.finish_reason is FinishReason.COMPLETED
        assert elapsed < 0.75

    run(scenario())


def test_diagnostic_sink_failure_is_isolated_and_observable() -> None:
    class BrokenSink:
        async def capture(self, record: Any) -> None:
            del record
            raise RuntimeError("diagnostic backend unavailable")

    async def scenario() -> None:
        agent = Agent(
            config=AgentConfig(
                "broken-diagnostic",
                "Answer.",
                retry=RetryConfig(max_attempts=1),
            ),
            model=ScriptedModel([RuntimeError("private provider error")]),
            diagnostic_sink=BrokenSink(),
        )

        result = await agent.run("fail")

        assert result.error == "model invocation failed"
        assert result.failure_id is not None
        assert agent.diagnostic_reporter is not None
        assert agent.diagnostic_reporter.drop_count == 1
        assert isinstance(agent.diagnostic_reporter.last_error, RuntimeError)

    run(scenario())


def test_agent_flushes_bounded_diagnostic_delivery_before_run_returns() -> None:
    class DelayedSink:
        def __init__(self) -> None:
            self.records: list[Any] = []

        async def capture(self, record: Any) -> None:
            await asyncio.sleep(0.01)
            self.records.append(record)

    async def scenario() -> None:
        sink = DelayedSink()
        agent = Agent(
            config=AgentConfig(
                "delayed-diagnostic",
                "Answer.",
                retry=RetryConfig(max_attempts=1),
            ),
            model=ScriptedModel([RuntimeError("private provider error")]),
            diagnostic_sink=sink,
        )

        result = await agent.run("fail")

        assert result.failure_id is not None
        assert [record.failure_id for record in sink.records] == [result.failure_id]

    run(scenario())


def test_terminal_stream_event_observes_committed_diagnostic_record() -> None:
    class DelayedSink:
        def __init__(self) -> None:
            self.records: list[Any] = []

        async def capture(self, record: Any) -> None:
            await asyncio.sleep(0.01)
            self.records.append(record)

    async def scenario() -> None:
        sink = DelayedSink()
        agent = Agent(
            config=AgentConfig(
                "stream-diagnostic-order",
                "Answer.",
                retry=RetryConfig(max_attempts=1),
            ),
            model=ScriptedModel([RuntimeError("private provider error")]),
            diagnostic_sink=sink,
        )

        async for event in agent.stream_all("fail"):
            if event.type is not EventType.RUN_FAILED:
                continue
            result = event.data["result"]
            assert result.failure_id is not None
            assert [record.failure_id for record in sink.records] == [result.failure_id]

    run(scenario())


def test_closing_outer_stream_releases_coordinator_run_state() -> None:
    async def scenario() -> None:
        agent = Agent(
            config=AgentConfig("stream-close", "Answer."),
            model=ScriptedModel([ModelResponse(Message.assistant("unused"))]),
            diagnostic_sink=InMemoryDiagnosticSink(),
        )
        stream = agent.stream_all("close early", session_id="close-session")
        started = await anext(stream)
        assert started.type is EventType.RUN_STARTED

        await stream.aclose()

        assert started.run_id not in agent.runtime._event_publishers
        assert started.run_id not in agent.runtime._coordinator_contexts

    run(scenario())


def test_failure_after_run_started_still_emits_terminal_and_cleans_up() -> None:
    async def scenario() -> None:
        sink = InMemoryDiagnosticSink()
        model = ScriptedModel([ModelResponse(Message.assistant("unused"))])
        agent = Agent(
            config=AgentConfig("late-configuration", "Answer."),
            model=model,
            diagnostic_sink=sink,
        )
        model.capabilities = object()  # type: ignore[assignment]

        events = [event async for event in agent.stream_all("fail safely")]

        assert events[0].type is EventType.RUN_STARTED
        assert events[-1].type is EventType.RUN_FAILED
        result = events[-1].data["result"]
        assert result.failure_id is not None
        assert sink.get(result.failure_id) is not None
        assert result.run_id not in agent.runtime._event_publishers
        assert result.run_id not in agent.runtime._coordinator_contexts

    run(scenario())
