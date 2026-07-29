from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from dataclasses import replace
from typing import Any

from moduagent.config import AgentConfig, RunLimits
from moduagent.decision import (
    Plan,
    PlanAndExecutePolicy,
    PlanStep,
    StandardDecisionPolicy,
    ToolFailureRecoveryConfig,
)
from moduagent.execution import (
    DurableBoundary,
    EngineContext,
    EngineOutcome,
    EngineSnapshot,
    ExecutionBudget,
    ExecutionEngine,
    FinalizationResult,
    PlanEngineState,
    PlanExecutionEngine,
    PlanStateCodec,
    StandardEngineState,
    StandardExecutionEngine,
    StandardExecutionPhase,
    ToolRecoveryController,
    ToolRecoveryControllerConfig,
    ToolRecoveryDecisionKind,
    ToolRecoveryState,
)
from moduagent.messages import (
    FinishReason,
    Message,
    ToolCall,
)
from moduagent.models import (
    ModelCapabilities,
    ModelChunk,
    ModelRequest,
    ModelResponse,
)
from moduagent.runtime.context import RunContext, RunRequest
from moduagent.runtime.events import AgentEvent, EventType
from moduagent.tools import (
    ToolError,
    ToolErrorType,
    ToolExecutionContext,
    ToolFailure,
    ToolRecoveryAction,
    ToolRegistry,
    function_tool,
)

try:
    from moduagent.tools import ToolRuntime
except ImportError:  # pragma: no cover - staggered 0.4 package upgrade
    from moduagent.tools.runtime import ToolRuntime


class StaticPlanGenerator:
    def __init__(self, plan: Plan) -> None:
        self.plan = plan

    async def create(self, context: RunContext) -> Plan:
        del context
        return Plan.from_dict(self.plan.to_dict())

    async def revise(
        self,
        context: RunContext,
        plan: Plan,
        feedback: str,
    ) -> Plan:
        del context, feedback
        return Plan.from_dict(plan.to_dict())


class FakeExecutionServices:
    def __init__(
        self,
        config: AgentConfig,
        responses: list[ModelResponse],
        *,
        tools: tuple[Any, ...] = (),
    ) -> None:
        self._budget = ExecutionBudget.from_config(config)
        self.responses = list(responses)
        self.requests: list[tuple[str, ModelRequest]] = []
        self.snapshots: list[tuple[DurableBoundary, EngineSnapshot]] = []
        self.events: list[AgentEvent] = []
        self.persist_calls = 0
        self.tool_runtime = ToolRuntime(ToolRegistry(tools))

    def budget(self, context: EngineContext) -> ExecutionBudget:
        del context
        return self._budget

    def remaining_seconds(self, context: EngineContext) -> float:
        del context
        return 60.0

    def tool_schemas(
        self,
        context: EngineContext,
        names: frozenset[str] | None = None,
    ):
        del context
        return self.tool_runtime.registry.schemas(names)

    def output_schema(
        self,
        context: EngineContext,
    ) -> Mapping[str, Any] | None:
        del context
        return None

    def decode_output(
        self,
        context: EngineContext,
        response: ModelResponse,
    ) -> Any:
        del context
        return response.message.content or ""

    async def prepare_model_request(
        self,
        context: EngineContext,
        request: ModelRequest,
        *,
        phase: str,
        skill_phase: str | None,
        protected_from: int | None = None,
    ) -> ModelRequest:
        del context, skill_phase, protected_from
        self.requests.append((phase, request))
        return request

    async def request_model(
        self,
        context: EngineContext,
        request: ModelRequest,
        *,
        phase: str,
    ) -> ModelResponse:
        del context, request, phase
        if not self.responses:
            raise AssertionError("unexpected model request")
        return self.responses.pop(0)

    def stream_model(
        self,
        context: EngineContext,
        request: ModelRequest,
        *,
        phase: str,
        delta_event_type=None,
        delta_visibility=None,
        delta_data=None,
    ) -> AsyncIterator[ModelChunk]:
        del (
            context,
            request,
            phase,
            delta_event_type,
            delta_visibility,
            delta_data,
        )

        async def generate() -> AsyncIterator[ModelChunk]:
            if not self.responses:
                raise AssertionError("unexpected model stream")
            response = self.responses.pop(0)
            content = response.message.content or ""
            if content:
                yield ModelChunk(delta=content)
            yield ModelChunk(response=response)

        return generate()

    async def execute_tool_batch(
        self,
        context: EngineContext,
        calls: tuple[ToolCall, ...],
        *,
        allowed_tools: frozenset[str] | None,
        repair_constraint,
    ):
        if allowed_tools is not None:
            disallowed = {call.name for call in calls} - set(allowed_tools)
            if disallowed:
                raise RuntimeError("disallowed Tool")
        return await self.tool_runtime.execute_many(
            calls,
            ToolExecutionContext(
                run_id=context.run.run_id,
                session_id=context.run.request.session_id,
            ),
            repair_constraint=repair_constraint,
        )

    async def record_tool_result(
        self,
        context: EngineContext,
        call: ToolCall,
        result: Any,
    ) -> None:
        del context, call, result

    async def finalize(
        self,
        context: EngineContext,
        request: ModelRequest,
        *,
        phase: str,
    ) -> FinalizationResult:
        await self.publish_event(
            context,
            AgentEvent(
                EventType.FINALIZATION_STARTED,
                context.run.run_id,
                {"phase": phase},
            ),
        )
        response = await self.request_model(
            context,
            request,
            phase=phase,
        )
        content = response.message.content or ""
        finalized = FinalizationResult(
            response=response,
            output=content,
            content=content,
            buffered_deltas=(content,),
        )
        await self.publish_event(
            context,
            AgentEvent(
                EventType.FINALIZATION_COMPLETED,
                context.run.run_id,
                {"phase": phase, "persisted": False},
            ),
        )
        return finalized

    async def persist_finalization(
        self,
        context: EngineContext,
        result: FinalizationResult,
    ) -> FinalizationResult:
        self.persist_calls += 1
        exists = any(
            message.role.value == "assistant"
            and message.metadata.get("moduagent.run_id") == context.run.run_id
            and message.metadata.get("moduagent.public_final") is True
            for message in context.run.messages
        )
        if not exists:
            context.run.add_message(
                Message.assistant(
                    result.content,
                    metadata={
                        "moduagent.run_id": context.run.run_id,
                        "moduagent.public_final": True,
                    },
                )
            )
        return replace(result, persisted=True)

    async def emit_finalization(
        self,
        context: EngineContext,
        result: FinalizationResult,
        *,
        phase: str,
    ) -> None:
        if context.stream_model:
            await self.publish_event(
                context,
                AgentEvent(
                    EventType.FINAL_DELTA,
                    context.run.run_id,
                    {"phase": phase, "delta": result.content},
                ),
            )

    async def checkpoint(
        self,
        context: EngineContext,
        snapshot: EngineSnapshot,
        *,
        boundary: DurableBoundary,
    ) -> None:
        del context
        self.snapshots.append((boundary, snapshot))

    async def publish_event(
        self,
        context: EngineContext,
        event: AgentEvent,
    ) -> AgentEvent:
        del context
        self.events.append(event)
        return event

    async def defer_event(
        self,
        context: EngineContext,
        event: AgentEvent,
    ) -> None:
        await self.publish_event(context, event)


def _context(config: AgentConfig, text: str = "answer") -> EngineContext:
    request = RunRequest(input=text, session_id="session")
    user = Message.user(text)
    run = RunContext(
        run_id="run",
        request=request,
        messages=[Message.system(config.instructions), user],
        new_messages=[user],
        current_run_start=1,
    )
    return EngineContext(run=run, config=config)


def _terminal(emissions) -> EngineOutcome:
    outcomes = [
        emission.outcome for emission in emissions if emission.outcome is not None
    ]
    assert len(outcomes) == 1
    return outcomes[0]


def test_standard_engine_owns_model_policy_and_terminal_loop() -> None:
    async def scenario() -> None:
        config = AgentConfig(
            "standard-v04",
            "Answer briefly.",
            finalization_mode="disabled",
        )
        services = FakeExecutionServices(
            config,
            [ModelResponse(Message.assistant("done"))],
        )
        engine = StandardExecutionEngine(StandardDecisionPolicy())
        context = _context(config)

        assert isinstance(engine, ExecutionEngine)
        state = await engine.initialize(context, services)
        emissions = [
            emission async for emission in engine.execute(context, state, services)
        ]

        outcome = _terminal(emissions)
        assert outcome.finish_reason is FinishReason.COMPLETED
        assert outcome.output == "done"
        assert [phase for phase, _ in services.requests] == ["act"]
        assert [event.type for event in services.events] == [
            EventType.POLICY_DECISION,
        ]
        assert services.snapshots[0][0] is DurableBoundary.INITIALIZED

        encoded = engine.encode_state(state)
        assert engine.decode_state(encoded) == state
        validation = engine.validate_resume(
            EngineSnapshot("standard", 1, encoded),
            {
                "engine_id": "standard",
                "common_state": {
                    "step": state.model_turn,
                    "tool_call_count": state.tool_call_count,
                },
            },
        )
        assert validation.compatible
        corrupted = engine.validate_resume(
            EngineSnapshot("standard", 1, encoded),
            {
                "engine_id": "standard",
                "common_state": {
                    "step": state.model_turn - 1,
                    "tool_call_count": state.tool_call_count,
                },
            },
        )
        assert not corrupted.compatible

    asyncio.run(scenario())


def test_standard_separates_unsupported_combined_tool_and_output_contract() -> None:
    class StructuredServices(FakeExecutionServices):
        def output_schema(
            self,
            context: EngineContext,
        ) -> Mapping[str, Any]:
            del context
            return {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            }

    async def scenario() -> None:
        @function_tool
        def lookup(query: str) -> str:
            return query

        config = AgentConfig(
            "separated-contracts",
            "Answer with the requested schema.",
            finalization_mode="disabled",
        )
        services = StructuredServices(
            config,
            [
                ModelResponse(Message.assistant("draft answer")),
                ModelResponse(Message.assistant('{"answer":"final"}')),
            ],
            tools=(lookup,),
        )
        engine = StandardExecutionEngine(StandardDecisionPolicy())
        base = _context(config)
        context = replace(
            base,
            model_capabilities=ModelCapabilities(
                streaming=False,
                tool_calling_with_structured_output=False,
            ),
        )

        state = await engine.initialize(context, services)
        emissions = [
            emission async for emission in engine.execute(context, state, services)
        ]

        outcome = _terminal(emissions)
        assert outcome.finish_reason is FinishReason.COMPLETED
        assert outcome.output == '{"answer":"final"}'
        assert [phase for phase, _ in services.requests] == ["act", "finalize"]
        act_request = services.requests[0][1]
        assert act_request.tools
        assert act_request.output_schema is None
        final_request = services.requests[1][1]
        assert final_request.tools == ()
        assert final_request.output_schema is not None
        assert services.snapshots[-1][0] is DurableBoundary.FINALIZATION_EMITTED

    asyncio.run(scenario())


def test_standard_preserves_legacy_combined_contract_when_declared_supported() -> None:
    class StructuredServices(FakeExecutionServices):
        def output_schema(
            self,
            context: EngineContext,
        ) -> Mapping[str, Any]:
            del context
            return {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            }

    async def scenario() -> None:
        @function_tool
        def lookup(query: str) -> str:
            return query

        config = AgentConfig(
            "combined-contracts",
            "Answer with the requested schema.",
            finalization_mode="disabled",
        )
        services = StructuredServices(
            config,
            [ModelResponse(Message.assistant('{"answer":"direct"}'))],
            tools=(lookup,),
        )
        engine = StandardExecutionEngine(StandardDecisionPolicy())
        context = _context(config)

        state = await engine.initialize(context, services)
        emissions = [
            emission async for emission in engine.execute(context, state, services)
        ]

        outcome = _terminal(emissions)
        assert outcome.finish_reason is FinishReason.COMPLETED
        assert [phase for phase, _ in services.requests] == ["act"]
        request = services.requests[0][1]
        assert request.tools
        assert request.output_schema is not None

    asyncio.run(scenario())


def test_standard_finalization_response_resumes_without_another_model_call() -> None:
    async def scenario() -> None:
        config = AgentConfig(
            "standard-resume",
            "Answer briefly.",
            finalization_mode="always",
        )
        services = FakeExecutionServices(config, [])
        engine = StandardExecutionEngine(StandardDecisionPolicy())
        context = _context(config)
        state = StandardEngineState(
            phase=StandardExecutionPhase.FINALIZE,
            model_turn=1,
            finalization_response="stable answer",
            finalization_count=1,
        )

        emissions = [
            emission async for emission in engine.execute(context, state, services)
        ]

        outcome = _terminal(emissions)
        assert outcome.finish_reason is FinishReason.COMPLETED
        assert outcome.output == "stable answer"
        assert services.requests == []
        assert state.phase is StandardExecutionPhase.DONE
        assert state.finalization_persisted is True
        assert state.finalization_emitted is True
        assert services.persist_calls == 1
        assert (
            sum(
                message.metadata.get("moduagent.public_final") is True
                for message in context.run.messages
            )
            == 1
        )
        assert services.snapshots[-1][0] is DurableBoundary.FINALIZATION_EMITTED
        encoded = engine.encode_state(state)
        assert encoded["finalization"] == {
            "started": True,
            "response_generated": True,
            "response": "stable answer",
            "invocation_count": 1,
            "persisted": True,
            "emitted": True,
        }

    asyncio.run(scenario())


def test_plan_engine_runs_step_result_commit_and_one_finalization() -> None:
    async def scenario() -> None:
        config = AgentConfig(
            "plan-v04",
            "Use committed evidence.",
            limits=RunLimits(max_steps=2),
            finalization_mode="always",
        )
        step_result = (
            '{"step_id":"answer","status":"completed","facts":["fact"],'
            '"artifacts":{},"uncertainties":[],"missing_inputs":[],'
            '"completion_evidence":["criterion met"]}'
        )
        services = FakeExecutionServices(
            config,
            [
                ModelResponse(Message.assistant(step_result)),
                ModelResponse(Message.assistant("public answer")),
            ],
        )
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(
                Plan(
                    [
                        PlanStep(
                            step_id="answer",
                            objective="produce evidence",
                            completion_criteria=["criterion met"],
                        )
                    ]
                )
            )
        )
        engine = PlanExecutionEngine(policy)
        context = _context(config)

        assert isinstance(engine, ExecutionEngine)
        state = await engine.initialize(context, services)
        assert isinstance(state, PlanEngineState)
        emissions = [
            emission async for emission in engine.execute(context, state, services)
        ]

        outcome = _terminal(emissions)
        assert outcome.finish_reason is FinishReason.COMPLETED
        assert outcome.output == "public answer"
        assert state.finalization.invocation_count == 1
        assert state.finalization.persisted is True
        assert state.finalization.emitted is True
        assert list(state.plan_progress.committed_results) == ["answer"]
        assert [phase for phase, _ in services.requests] == [
            "step_result",
            "finalize",
        ]
        assert EventType.STEP_COMMITTED in {event.type for event in services.events}
        assert EventType.FINALIZATION_COMPLETED in {
            event.type for event in services.events
        }
        final_boundaries = [
            boundary
            for boundary, _ in services.snapshots
            if boundary
            in {
                DurableBoundary.FINALIZATION_RESPONSE,
                DurableBoundary.FINALIZATION_PERSISTED,
                DurableBoundary.FINALIZATION_EMITTED,
            }
        ]
        assert final_boundaries == [
            DurableBoundary.FINALIZATION_RESPONSE,
            DurableBoundary.FINALIZATION_PERSISTED,
            DurableBoundary.FINALIZATION_EMITTED,
        ]

    asyncio.run(scenario())


def test_plan_engine_reapplies_resolved_limits_when_execution_is_resumed() -> None:
    class RecordingPolicy(PlanAndExecutePolicy):
        def __init__(self, generator: StaticPlanGenerator) -> None:
            super().__init__(generator)
            self.configured: list[tuple[int, int, int]] = []

        def configure_limits(
            self,
            *,
            max_step_attempts: int,
            max_replans: int,
            max_tool_repair_attempts: int | None = None,
        ) -> None:
            super().configure_limits(
                max_step_attempts=max_step_attempts,
                max_replans=max_replans,
                max_tool_repair_attempts=max_tool_repair_attempts,
            )

        def configure_tool_repair_limits(
            self,
            *,
            max_tool_repair_attempts: int,
        ) -> None:
            super().configure_tool_repair_limits(
                max_tool_repair_attempts=max_tool_repair_attempts
            )
            self.configured.append(
                (
                    self.max_step_attempts,
                    self.max_replans,
                    self.max_tool_repair_attempts,
                )
            )

    async def scenario() -> None:
        config = AgentConfig(
            "plan-resolved-limits",
            "Use committed evidence.",
            limits=RunLimits(
                max_step_attempts=5,
                max_replans=4,
                max_tool_repair_attempts=3,
            ),
            finalization_mode="always",
        )
        result_payload = (
            '{"step_id":"answer","status":"completed","facts":["done"],'
            '"artifacts":{},"uncertainties":[],"missing_inputs":[],'
            '"completion_evidence":["done"]}'
        )
        services = FakeExecutionServices(
            config,
            [
                ModelResponse(Message.assistant(result_payload)),
                ModelResponse(Message.assistant("done")),
            ],
        )
        policy = RecordingPolicy(
            StaticPlanGenerator(
                Plan(
                    [
                        PlanStep(
                            step_id="answer",
                            objective="answer",
                            completion_criteria=["done"],
                        )
                    ]
                )
            )
        )
        engine = PlanExecutionEngine(policy)
        context = _context(config)

        state = await engine.initialize(context, services)
        # Simulate a fresh policy instance retaining only standalone defaults
        # while Engine state is restored from a checkpoint.
        policy.max_step_attempts = 2
        policy.max_replans = 2
        policy.max_tool_repair_attempts = 1
        _ = [emission async for emission in engine.execute(context, state, services)]

        assert policy.configured == [(5, 4, 3), (5, 4, 3)]

    asyncio.run(scenario())


def test_plan_budget_terminal_reason_survives_state_round_trip() -> None:
    async def scenario() -> None:
        @function_tool
        def lookup(query: str) -> str:
            return query

        call = ToolCall("call-budget", "lookup", {"query": "value"})
        config = AgentConfig(
            "plan-budget-resume",
            "Use committed evidence.",
            limits=RunLimits(max_tool_calls=0),
            finalization_mode="always",
        )
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(
                Plan(
                    [
                        PlanStep(
                            step_id="lookup",
                            objective="lookup",
                            completion_criteria=["value found"],
                            allowed_tools=("lookup",),
                        )
                    ]
                )
            )
        )
        engine = PlanExecutionEngine(policy)
        context = _context(config)
        services = FakeExecutionServices(
            config,
            [ModelResponse(Message.assistant(None, (call,)), (call,))],
            tools=(lookup,),
        )

        state = await engine.initialize(context, services)
        first = _terminal(
            [emission async for emission in engine.execute(context, state, services)]
        )
        restored = engine.decode_state(engine.encode_state(state))
        resumed = _terminal(
            [
                emission
                async for emission in engine.execute(
                    context,
                    restored,
                    FakeExecutionServices(config, [], tools=(lookup,)),
                )
            ]
        )

        assert first.finish_reason is FinishReason.MAX_TOOL_CALLS
        assert resumed.finish_reason is first.finish_reason
        assert resumed.error == first.error == "tool call limit exceeded"
        assert restored.phase.value == "failed"

    asyncio.run(scenario())


def test_plan_finalization_response_resumes_without_duplicate_model_or_message() -> (
    None
):
    class InterruptingPersistServices(FakeExecutionServices):
        async def persist_finalization(
            self,
            context: EngineContext,
            result: FinalizationResult,
        ) -> FinalizationResult:
            del context, result
            self.persist_calls += 1
            raise asyncio.TimeoutError

    async def scenario() -> None:
        config = AgentConfig(
            "plan-finalization-resume",
            "Use committed evidence.",
            finalization_mode="always",
        )
        step_result = (
            '{"step_id":"answer","status":"completed","facts":["fact"],'
            '"artifacts":{},"uncertainties":[],"missing_inputs":[],'
            '"completion_evidence":["criterion met"]}'
        )
        interrupted_services = InterruptingPersistServices(
            config,
            [
                ModelResponse(Message.assistant(step_result)),
                ModelResponse(Message.assistant("stable public answer")),
            ],
        )
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(
                Plan(
                    [
                        PlanStep(
                            step_id="answer",
                            objective="produce evidence",
                            completion_criteria=["criterion met"],
                        )
                    ]
                )
            )
        )
        engine = PlanExecutionEngine(policy)
        context = _context(config)
        state = await engine.initialize(context, interrupted_services)

        try:
            _ = [
                emission
                async for emission in engine.execute(
                    context,
                    state,
                    interrupted_services,
                )
            ]
        except asyncio.TimeoutError:
            pass
        else:
            raise AssertionError("persistence interruption was not raised")

        assert state.phase.value == "finalize"
        assert state.finalization.response == "stable public answer"
        assert state.finalization.persisted is False
        assert state.finalization.emitted is False
        assert len(interrupted_services.requests) == 2
        assert not any(
            message.metadata.get("moduagent.public_final") is True
            for message in context.run.messages
        )

        resumed_services = FakeExecutionServices(config, [])
        emissions = [
            emission
            async for emission in engine.execute(
                context,
                state,
                resumed_services,
            )
        ]

        outcome = _terminal(emissions)
        assert outcome.output == "stable public answer"
        assert resumed_services.requests == []
        assert resumed_services.persist_calls == 1
        assert state.finalization.persisted is True
        assert state.finalization.emitted is True
        assert (
            sum(
                message.metadata.get("moduagent.public_final") is True
                for message in context.run.messages
            )
            == 1
        )

    asyncio.run(scenario())


def test_plan_engine_routes_tool_outcome_through_recovery_controller() -> None:
    async def scenario() -> None:
        invocations: list[str] = []

        @function_tool(repair_safe=True)
        def lookup(query: str) -> str:
            invocations.append(query)
            if query == "bad":
                raise ToolFailure(
                    ToolError(
                        ToolErrorType.EXECUTION_ERROR,
                        "safe correction hint",
                        reason="invalid_query",
                        recovery=ToolRecoveryAction.REPAIR_CALL,
                    )
                )
            return "value"

        bad_call = ToolCall("call-1", "lookup", {"query": "bad"})
        repaired_call = ToolCall("call-2", "lookup", {"query": "good"})
        step_result = (
            '{"step_id":"lookup-step","status":"completed",'
            '"facts":["value"],"artifacts":{},"uncertainties":[],'
            '"missing_inputs":[],"completion_evidence":["value found"]}'
        )
        config = AgentConfig(
            "plan-repair",
            "Use committed evidence.",
            finalization_mode="always",
        )
        services = FakeExecutionServices(
            config,
            [
                ModelResponse(
                    Message.assistant(None, (bad_call,)),
                    finish_reason="tool_calls",
                ),
                ModelResponse(
                    Message.assistant(None, (repaired_call,)),
                    finish_reason="tool_calls",
                ),
                ModelResponse(Message.assistant(step_result)),
                ModelResponse(Message.assistant("public answer")),
            ],
            tools=(lookup,),
        )
        policy = PlanAndExecutePolicy(
            StaticPlanGenerator(
                Plan(
                    [
                        PlanStep(
                            step_id="lookup-step",
                            objective="look up a value",
                            completion_criteria=["value found"],
                            allowed_tools=["lookup"],
                        )
                    ]
                )
            ),
            tool_failure_recovery=ToolFailureRecoveryConfig(fallback="fail"),
        )
        engine = PlanExecutionEngine(policy)
        context = _context(config)
        state = await engine.initialize(context, services)

        emissions = [
            emission async for emission in engine.execute(context, state, services)
        ]

        outcome = _terminal(emissions)
        assert outcome.finish_reason is FinishReason.COMPLETED
        assert invocations == ["bad", "good"]
        assert state.tool_recovery.total_repairs == 1
        assert state.tool_recovery.seen_call_ids == ["call-1", "call-2"]
        assert [phase for phase, _ in services.requests] == [
            "act",
            "act",
            "step_result",
            "finalize",
        ]
        assert EventType.TOOL_REPAIR_SCHEDULED in {
            event.type for event in services.events
        }

    asyncio.run(scenario())


def test_plan_policy_never_receives_raw_tool_failure_or_arguments() -> None:
    class CapturingPlanGenerator(StaticPlanGenerator):
        def __init__(self, plan: Plan) -> None:
            super().__init__(plan)
            self.feedback: list[str] = []

        async def revise(
            self,
            context: RunContext,
            plan: Plan,
            feedback: str,
        ) -> Plan:
            del context
            self.feedback.append(feedback)
            return Plan.from_dict(plan.to_dict())

    async def scenario() -> None:
        secret_sql = "SELECT password FROM private_users"
        secret_password = "db-password-123"

        @function_tool
        def query_database(sql: str, password: str) -> str:
            del sql, password
            raise ToolFailure(
                ToolError(
                    ToolErrorType.EXECUTION_ERROR,
                    f"syntax error in {secret_sql}; credential={secret_password}",
                    reason="syntax_error",
                    recovery=ToolRecoveryAction.REPLAN,
                )
            )

        call = ToolCall(
            "query-1",
            "query_database",
            {"sql": secret_sql, "password": secret_password},
        )
        plan = Plan(
            [
                PlanStep(
                    step_id="query",
                    objective="query the database",
                    completion_criteria=["a safe result is available"],
                    allowed_tools=["query_database"],
                )
            ]
        )
        generator = CapturingPlanGenerator(plan)
        config = AgentConfig(
            "safe-replan",
            "Never disclose database internals.",
            limits=RunLimits(max_replans=1),
        )
        services = FakeExecutionServices(
            config,
            [
                ModelResponse(
                    Message.assistant(None, (call,)),
                    finish_reason="tool_calls",
                )
            ],
            tools=(query_database,),
        )
        engine = PlanExecutionEngine(PlanAndExecutePolicy(generator))
        context = _context(config)
        state = await engine.initialize(context, services)

        iterator = engine.execute(context, state, services).__aiter__()
        while True:
            emission = await iterator.__anext__()
            if (
                emission.event is not None
                and emission.event.type is EventType.PLAN_REVISED
            ):
                break
        await iterator.aclose()

        assert generator.feedback == ["Tool execution failed"]
        projected = str(generator.feedback)
        assert secret_sql not in projected
        assert secret_password not in projected
        snapshot_payload = str(services.snapshots[-1][1].state["tool_recovery"])
        assert secret_sql not in snapshot_payload
        assert secret_password not in snapshot_payload

    asyncio.run(scenario())


def test_plan_state_codec_nests_and_migrates_flat_v3_state() -> None:
    plan = Plan(
        [
            PlanStep(
                step_id="lookup",
                objective="look up",
                completion_criteria=["value found"],
                allowed_tools=["lookup"],
            )
        ]
    )
    legacy_policy = PlanAndExecutePolicy(StaticPlanGenerator(plan))
    config = AgentConfig("codec", "test")
    services = FakeExecutionServices(config, [])
    context = _context(config)
    engine = PlanExecutionEngine(legacy_policy)

    async def initialize() -> PlanEngineState:
        return await engine.initialize(context, services)

    state = asyncio.run(initialize())
    codec = PlanStateCodec()
    encoded = codec.encode(state)

    assert set(encoded) == {
        "phase",
        "plan_progress",
        "step_execution",
        "tool_recovery",
        "finalization",
        "terminal",
    }
    assert codec.encode(codec.decode(encoded)) == encoded

    legacy = state.to_legacy()
    migrated = codec.migrate(3, legacy.to_dict())
    restored = codec.decode(migrated)
    assert restored.plan_progress.plan.steps[0].step_id == "lookup"
    assert restored.tool_recovery.seen_call_ids == []


def test_tool_recovery_controller_builds_constraint_before_new_call_is_seen() -> None:
    async def scenario() -> None:
        invocations: list[str] = []

        @function_tool(repair_safe=True)
        def lookup(query: str) -> str:
            invocations.append(query)
            if query == "bad":
                raise ToolFailure(
                    ToolError(
                        ToolErrorType.EXECUTION_ERROR,
                        "safe correction hint",
                        reason="invalid_query",
                        recovery=ToolRecoveryAction.REPAIR_CALL,
                    )
                )
            return "value"

        runtime = ToolRuntime([lookup])
        controller = ToolRecoveryController(
            ToolRecoveryControllerConfig(fallback="fail")
        )
        state = ToolRecoveryState()
        failed_call = ToolCall("call-1", "lookup", {"query": "bad"})
        controller.record_requested_calls(state, (failed_call,))
        failed = await runtime.execute_many((failed_call,))

        decision = controller.decide(
            failed,
            state,
            step_id="lookup-step",
            max_repair_attempts=1,
        )

        assert decision.kind is ToolRecoveryDecisionKind.REPAIR
        assert decision.repair_constraint is not None
        assert decision.repair_constraint.seen_call_ids == frozenset({"call-1"})
        assert "call-2" not in decision.repair_constraint.seen_call_ids

        repaired_call = ToolCall("call-2", "lookup", {"query": "good"})
        repaired = await runtime.execute_many(
            (repaired_call,),
            repair_constraint=decision.repair_constraint,
        )
        assert repaired.failure_count == 0
        assert invocations == ["bad", "good"]

    asyncio.run(scenario())


def test_tool_recovery_partial_success_fails_closed() -> None:
    async def scenario() -> None:
        @function_tool(repair_safe=True)
        def lookup(query: str) -> str:
            if query == "bad":
                raise ToolFailure(
                    ToolError(
                        ToolErrorType.EXECUTION_ERROR,
                        "invalid",
                        reason="invalid_query",
                        recovery=ToolRecoveryAction.REPAIR_CALL,
                    )
                )
            return query

        runtime = ToolRuntime([lookup])
        calls = (
            ToolCall("ok", "lookup", {"query": "good"}),
            ToolCall("failed", "lookup", {"query": "bad"}),
        )
        controller = ToolRecoveryController()
        state = ToolRecoveryState()
        controller.record_requested_calls(state, calls)
        outcome = await runtime.execute_many(calls)

        decision = controller.decide(
            outcome,
            state,
            step_id="lookup-step",
            max_repair_attempts=2,
        )

        assert outcome.partial_success
        assert decision.kind is ToolRecoveryDecisionKind.FAIL
        assert state.terminal_failure is not None

    asyncio.run(scenario())
