from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import replace
from typing import Any, Protocol, runtime_checkable

from moduagent.decision import DecisionKind, ExecutionDecision
from moduagent.errors import ModuAgentError
from moduagent.decision.planning import (
    ExecutionState,
    PlanAndExecutePolicy,
    PlanStep,
    _STEP_VALIDATION_CODES,
    _STEP_VALIDATION_LOCATIONS,
)
from moduagent.execution.base import (
    CodecBackedEngine,
    DurableBoundary,
    EngineContext,
    EngineEmission,
    EngineOutcome,
    ExecutionServices,
    FinalizationResult,
)
from moduagent.execution.planning.recovery import (
    ToolRecoveryController,
    ToolRecoveryControllerConfig,
    ToolRecoveryDecision,
    ToolRecoveryDecisionKind,
)
from moduagent.execution.planning.state import (
    PlanEngineState,
    PlanExecutionPhase,
    PlanStateCodec,
    ToolRecoveryState,
)
from moduagent.messages import FinishReason, Message
from moduagent.models import ModelRequest, ModelResponse
from moduagent.runtime.context import RunContext
from moduagent.runtime.events import (
    AgentEvent,
    EventType,
    EventVisibility,
)
from moduagent.tools import (
    ToolBatchOutcome,
    ToolError,
    ToolErrorType,
    ToolResult,
)

try:
    from moduagent.tools import ToolRepairConstraint
except ImportError:  # pragma: no cover - staggered 0.4 package upgrade
    from moduagent.tools.runtime import ToolRepairConstraint


_EXECUTOR_PROMPT = (
    "Execute exactly one current plan step. Do not write the public final "
    "answer, perform another step, or claim unsupported facts."
)
_STEP_RESULT_PROMPT = (
    "Return only the current StepResult JSON. Keep step_id unchanged and "
    "provide completion evidence for every completion criterion."
)
_FINALIZER_PROMPT = (
    "Create the one public final response from the original objective and "
    "committed step results. Do not call tools, add new facts, or expose "
    "internal execution logs."
)


@runtime_checkable
class PlanPolicyAdapter(Protocol):
    """Explicit compatibility surface for the 0.3 Plan policy."""

    def configure_limits(
        self,
        *,
        max_step_attempts: int,
        max_replans: int,
    ) -> None: ...

    def configure_tool_repair_limits(
        self,
        *,
        max_tool_repair_attempts: int,
    ) -> None: ...

    def configure_available_tools(
        self,
        available_tools: frozenset[str],
    ) -> None: ...

    async def begin(self, context: RunContext) -> None: ...

    def prepare_step(
        self,
        context: RunContext,
        *,
        has_tools: bool,
    ) -> PlanStep | None: ...

    def needs_step_result_extraction(self, context: RunContext) -> bool: ...

    def step_result_schema(self) -> Mapping[str, Any]: ...

    async def decide(
        self,
        context: RunContext,
        response: ModelResponse,
    ) -> ExecutionDecision: ...

    async def validate_pending(
        self,
        context: RunContext,
    ) -> ExecutionDecision: ...

    async def observe(
        self,
        context: RunContext,
        results: Sequence[ToolResult],
    ) -> ExecutionDecision | None: ...

    def begin_finalization(self, context: RunContext) -> ExecutionState: ...

    def record_final_response(
        self,
        context: RunContext,
        response: str,
        *,
        persisted: bool = False,
        emitted: bool = False,
    ) -> ExecutionState: ...

    def finalization_payload(self, context: RunContext) -> dict[str, Any]: ...


class PlanExecutionEngine(CodecBackedEngine[PlanEngineState]):
    """Strict Plan Engine with explicit services and a nested durable state."""

    engine_id = "plan"
    state_version = 1

    def __init__(
        self,
        policy: PlanPolicyAdapter,
        *,
        recovery_controller: ToolRecoveryController | None = None,
    ) -> None:
        if not isinstance(policy, PlanPolicyAdapter):
            raise TypeError("policy must implement PlanPolicyAdapter")
        if recovery_controller is not None and not isinstance(
            recovery_controller,
            ToolRecoveryController,
        ):
            raise TypeError("recovery_controller must be a ToolRecoveryController")
        self.policy = policy
        self.recovery_controller = recovery_controller or (
            self._legacy_recovery_controller(policy)
        )
        self.state_codec = PlanStateCodec()

    async def initialize(
        self,
        context: EngineContext,
        services: ExecutionServices,
    ) -> PlanEngineState:
        if not isinstance(context, EngineContext):
            raise TypeError("context must be an EngineContext")
        if not isinstance(services, ExecutionServices):
            raise TypeError("services must implement ExecutionServices")
        self._configure_policy(context, services)
        await asyncio.wait_for(
            self.policy.begin(context.run),
            timeout=services.remaining_seconds(context),
        )
        legacy = context.run.execution_state
        if not isinstance(legacy, ExecutionState):
            raise RuntimeError("Plan policy did not initialize an ExecutionState")
        state = PlanEngineState.from_legacy(legacy)
        await services.defer_event(
            context,
            AgentEvent(
                EventType.PLAN_CREATED,
                context.run.run_id,
                {
                    "step_count": len(state.plan_progress.plan.steps),
                    "plan_version": state.plan_progress.plan.version,
                },
                visibility=EventVisibility.INTERNAL,
            ),
        )
        await self._checkpoint(
            context,
            state,
            services,
            DurableBoundary.INITIALIZED,
        )
        return state

    async def execute(
        self,
        context: EngineContext,
        state: PlanEngineState,
        services: ExecutionServices,
    ) -> AsyncIterator[EngineEmission]:
        if not isinstance(context, EngineContext):
            raise TypeError("context must be an EngineContext")
        if not isinstance(state, PlanEngineState):
            raise TypeError("state must be a PlanEngineState")
        if not isinstance(services, ExecutionServices):
            raise TypeError("services must implement ExecutionServices")
        # A resumed run can be executed by a newly constructed Agent/process.
        # Reapply the resolved limits instead of relying on initialize() side
        # effects from the process that created the checkpoint.
        self._configure_policy(context, services)
        budget = services.budget(context)

        while True:
            if services.remaining_seconds(context) <= 0:
                raise asyncio.TimeoutError
            if len(state.plan_progress.plan.steps) > budget.max_steps:
                legacy = state.to_legacy()
                legacy.validation_error = (
                    f"plan exceeds RunLimits.max_steps ({budget.max_steps})"
                )
                legacy.fail_current_step()
                state = self._replace_from_legacy(context, state, legacy)
                yield EngineEmission(
                    outcome=self._outcome(
                        state,
                        FinishReason.MAX_STEPS,
                        error=legacy.validation_error,
                    )
                )
                return

            if state.phase is PlanExecutionPhase.DONE:
                content = state.finalization.response
                if content is None:
                    raise RuntimeError("DONE Plan state has no final response")
                response = ModelResponse(Message.assistant(content))
                yield EngineEmission(
                    outcome=self._outcome(
                        state,
                        FinishReason.COMPLETED,
                        output=services.decode_output(context, response),
                    )
                )
                return
            if state.phase is PlanExecutionPhase.FAILED:
                yield EngineEmission(
                    outcome=self._outcome(
                        state,
                        state.terminal_finish_reason or FinishReason.ERROR,
                        error=state.terminal_error or self._failure_message(state),
                    )
                )
                return

            if state.phase is PlanExecutionPhase.STEP_PREPARE:
                step = state.plan_progress.plan.current
                if step is None:
                    state.phase = PlanExecutionPhase.VERIFY
                    self._sync_legacy(context, state)
                    continue
                step_tools = self._step_tool_schemas(
                    context,
                    services,
                    step,
                )
                self._sync_legacy(context, state)
                prepared = self.policy.prepare_step(
                    context.run,
                    has_tools=bool(step_tools),
                )
                state = self._pull_state(context, state)
                if prepared is None:
                    continue
                context.run.clear_internal_messages()
                event = AgentEvent(
                    EventType.STEP_STARTED,
                    context.run.run_id,
                    {
                        "step_id": prepared.step_id,
                        "objective": prepared.objective,
                        "attempt": prepared.attempt_count,
                        "plan_version": state.plan_progress.plan.version,
                    },
                    visibility=EventVisibility.INTERNAL,
                )
                yield await self._publish(context, services, event)
                await self._checkpoint(
                    context,
                    state,
                    services,
                    DurableBoundary.BEFORE_MODEL,
                )
                continue

            if state.phase is PlanExecutionPhase.TOOL_RECOVERY:
                await self._checkpoint(
                    context,
                    state,
                    services,
                    DurableBoundary.REPAIR_SCHEDULED,
                )
                state.phase = PlanExecutionPhase.ACT_TOOL
                continue

            if state.phase is PlanExecutionPhase.VERIFY:
                self._verify(state)
                self._sync_legacy(context, state)
                self.policy.begin_finalization(context.run)
                state = self._pull_state(context, state)
                # _finalize() owns the FINALIZATION_STARTED barrier for both
                # fresh and resumed FINALIZE states. There is no await or public
                # emission before that barrier, so saving it here as well only
                # writes the same state twice.
                continue

            if state.phase is PlanExecutionPhase.FINALIZE:
                async for emission in self._finalize(
                    context,
                    state,
                    services,
                ):
                    yield emission
                return

            step = state.current_step
            if step is None:
                raise RuntimeError(
                    f"{state.phase.value} phase has no current Plan step"
                )
            if state.phase is PlanExecutionPhase.STEP_VALIDATE:
                self._sync_legacy(context, state)
                decision = await asyncio.wait_for(
                    self.policy.validate_pending(context.run),
                    timeout=services.remaining_seconds(context),
                )
                state = self._pull_state(context, state)
                repair_constraint = None
            elif state.phase in {
                PlanExecutionPhase.ACT_TOOL,
                PlanExecutionPhase.STEP_RESULT,
            }:
                extracting = state.phase is PlanExecutionPhase.STEP_RESULT
                schemas = self._step_tool_schemas(
                    context,
                    services,
                    step,
                )
                repair_constraint = (
                    self.recovery_controller.constraint_for_pending(state.tool_recovery)
                    if state.tool_recovery.pending_failure is not None
                    else None
                )
                request = ModelRequest(
                    messages=self._step_messages(
                        context,
                        state,
                        extracting=extracting,
                    ),
                    tools=() if extracting else schemas,
                    output_schema=(
                        self.policy.step_result_schema() if extracting else None
                    ),
                    options=self._model_options(
                        context,
                        has_tools=not extracting and bool(schemas),
                    ),
                )
                request = await services.prepare_model_request(
                    context,
                    request,
                    phase="step_result" if extracting else "act",
                    skill_phase="act",
                    protected_from=0,
                )
                context.run.step += 1
                await self._checkpoint(
                    context,
                    state,
                    services,
                    DurableBoundary.BEFORE_MODEL,
                )
                response = await self._model_turn(
                    context,
                    services,
                    request,
                    phase="step_result" if extracting else "act",
                )
                calls = tuple(response.tool_calls or response.message.tool_calls)
                context.run.add_internal_message(
                    Message.assistant(
                        response.message.content,
                        calls,
                        metadata=({"moduagent.ephemeral": True} if calls else None),
                    )
                )
                self._sync_legacy(context, state)
                decision = await asyncio.wait_for(
                    self.policy.decide(context.run, response),
                    timeout=services.remaining_seconds(context),
                )
                state = self._pull_state(context, state)
            else:
                raise RuntimeError(
                    f"unsupported Plan execution phase: {state.phase.value}"
                )

            decision_event = AgentEvent(
                EventType.POLICY_DECISION,
                context.run.run_id,
                {
                    "kind": decision.kind.value,
                    "metadata": dict(decision.metadata),
                },
                visibility=EventVisibility.INTERNAL,
            )
            yield await self._publish(context, services, decision_event)

            if decision.kind is DecisionKind.CALL_TOOLS:
                async for emission in self._execute_tools(
                    context,
                    state,
                    services,
                    decision,
                    repair_constraint=repair_constraint,
                ):
                    if emission.outcome is not None:
                        yield emission
                        return
                    yield emission
                state = self._pull_state(context, state)
                continue

            if decision.kind is DecisionKind.RETRY_STEP:
                validation_failure = self._validation_failure(
                    decision.metadata,
                    state,
                    step,
                )
                event = AgentEvent(
                    EventType.STEP_RETRY,
                    context.run.run_id,
                    {
                        "step_id": step.step_id,
                        "attempt": (state.step_execution.step_attempt_count),
                        "reason": decision.metadata.get("reason"),
                        "count_attempt": decision.metadata.get(
                            "count_attempt",
                            True,
                        ),
                        **(
                            {}
                            if validation_failure is None
                            else {
                                "validation_code": validation_failure["code"],
                                "validation_location": validation_failure["location"],
                                **(
                                    {
                                        "validation_cause_code": (
                                            validation_failure["cause_code"]
                                        )
                                    }
                                    if "cause_code" in validation_failure
                                    else {}
                                ),
                            }
                        ),
                    },
                    visibility=EventVisibility.INTERNAL,
                )
                yield await self._publish(context, services, event)
                await self._checkpoint(
                    context,
                    state,
                    services,
                    DurableBoundary.STEP_RESULT_PENDING,
                )
                continue

            if decision.kind is DecisionKind.REPLAN:
                event = AgentEvent(
                    EventType.PLAN_REVISED,
                    context.run.run_id,
                    {
                        "plan_version": state.plan_progress.plan.version,
                        "replan_count": state.plan_progress.replan_count,
                        "reason": decision.metadata.get("reason"),
                    },
                    visibility=EventVisibility.INTERNAL,
                )
                yield await self._publish(context, services, event)
                context.run.clear_internal_messages()
                await self._checkpoint(
                    context,
                    state,
                    services,
                    DurableBoundary.REPLAN_COMPLETED,
                )
                continue

            if decision.kind is DecisionKind.COMMIT_STEP:
                committed = state.plan_progress.committed_results.get(step.step_id)
                if committed is None:
                    raise RuntimeError("committed Plan step has no StepResult")
                for event in (
                    AgentEvent(
                        EventType.STEP_RESULT_CREATED,
                        context.run.run_id,
                        {
                            "step_id": step.step_id,
                            "status": committed.status,
                        },
                        visibility=EventVisibility.INTERNAL,
                    ),
                    AgentEvent(
                        EventType.STEP_VALIDATED,
                        context.run.run_id,
                        {"step_id": step.step_id, "decision": "commit"},
                        visibility=EventVisibility.INTERNAL,
                    ),
                    AgentEvent(
                        EventType.STEP_COMMITTED,
                        context.run.run_id,
                        {
                            "step_id": step.step_id,
                            "result_ref": next(
                                item.result_ref
                                for item in state.plan_progress.plan.steps
                                if item.step_id == step.step_id
                            ),
                            "plan_version": state.plan_progress.plan.version,
                        },
                        visibility=EventVisibility.INTERNAL,
                    ),
                ):
                    yield await self._publish(context, services, event)
                context.run.clear_internal_messages()
                context.run.metadata["_moduagent_resume_safety"] = "resumable"
                await self._checkpoint(
                    context,
                    state,
                    services,
                    DurableBoundary.STEP_COMMITTED,
                )
                continue

            if decision.kind is DecisionKind.FINALIZE:
                self._sync_legacy(context, state)
                self.policy.begin_finalization(context.run)
                state = self._pull_state(context, state)
                continue

            if decision.kind is DecisionKind.FAIL:
                validation_failure = self._validation_failure(
                    decision.metadata,
                    state,
                    step,
                )
                yield await self._publish(
                    context,
                    services,
                    AgentEvent(
                        EventType.STEP_FAILED,
                        context.run.run_id,
                        {
                            "step_id": step.step_id,
                            "attempt": state.step_execution.step_attempt_count,
                            "reason": decision.metadata.get("reason"),
                            **(
                                {}
                                if validation_failure is None
                                else {
                                    "validation_code": validation_failure["code"],
                                    "validation_location": (
                                        validation_failure["location"]
                                    ),
                                    **(
                                        {
                                            "validation_cause_code": (
                                                validation_failure["cause_code"]
                                            )
                                        }
                                        if "cause_code" in validation_failure
                                        else {}
                                    ),
                                }
                            ),
                        },
                        visibility=EventVisibility.INTERNAL,
                    ),
                )
                yield EngineEmission(
                    outcome=self._outcome(
                        state,
                        FinishReason.ERROR,
                        error=(decision.error_message or self._failure_message(state)),
                        validation_failure=validation_failure,
                    )
                )
                return

            raise RuntimeError(f"unsupported strict decision: {decision.kind.value}")

    def _configure_policy(
        self,
        context: EngineContext,
        services: ExecutionServices,
    ) -> None:
        budget = services.budget(context)
        available = services.tool_schemas(context)
        self.policy.configure_available_tools(
            frozenset(schema.name for schema in available)
        )
        self.policy.configure_limits(
            max_step_attempts=budget.max_step_attempts,
            max_replans=budget.max_replans,
        )
        self.policy.configure_tool_repair_limits(
            max_tool_repair_attempts=budget.max_tool_repair_attempts,
        )

    async def _execute_tools(
        self,
        context: EngineContext,
        state: PlanEngineState,
        services: ExecutionServices,
        decision: ExecutionDecision,
        *,
        repair_constraint: ToolRepairConstraint | None,
    ) -> AsyncIterator[EngineEmission]:
        step = state.current_step
        if step is None:
            raise RuntimeError("Tool execution has no current Plan step")
        calls = tuple(decision.tool_calls)
        budget = services.budget(context)
        schemas = self._step_tool_schemas(context, services, step)
        exempt_tools = {
            schema.name for schema in schemas if not schema.counts_toward_tool_limit
        }
        counted_calls = sum(call.name not in exempt_tools for call in calls)
        if context.run.tool_call_count + counted_calls > budget.max_tool_calls:
            legacy = state.to_legacy()
            legacy.validation_error = "tool call limit exceeded"
            for call in calls:
                rejected = ToolResult.failed(
                    call_id=call.id,
                    tool_name=call.name,
                    error=ToolError(
                        ToolErrorType.EXECUTION_ERROR,
                        legacy.validation_error,
                    ),
                )
                await services.record_tool_result(context, call, rejected)
                context.run.add_internal_message(
                    Message.tool(
                        rejected.model_content(),
                        call_id=call.id,
                        name=call.name,
                        metadata={"moduagent.ephemeral": True},
                    )
                )
            legacy.fail_current_step()
            state = self._replace_from_legacy(context, state, legacy)
            yield EngineEmission(
                outcome=self._outcome(
                    state,
                    FinishReason.MAX_TOOL_CALLS,
                    error="tool call limit exceeded",
                )
            )
            return

        context.run.tool_call_count += counted_calls
        try:
            outcome = await services.execute_tool_batch(
                context,
                calls,
                allowed_tools=frozenset(step.allowed_tools),
                repair_constraint=repair_constraint,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error_message = (
                "tool execution timed out"
                if isinstance(exc, asyncio.TimeoutError)
                else str(exc)
                if isinstance(exc, ModuAgentError)
                else "tool execution failed"
            )
            error_type = (
                ToolErrorType.TIMEOUT
                if isinstance(exc, asyncio.TimeoutError)
                else ToolErrorType.EXECUTION_ERROR
            )
            for call in calls:
                rejected = ToolResult.failed(
                    call_id=call.id,
                    tool_name=call.name,
                    error=ToolError(error_type, error_message),
                )
                context.run.add_internal_message(
                    Message.tool(
                        rejected.model_content(),
                        call_id=call.id,
                        name=call.name,
                        metadata={"moduagent.ephemeral": True},
                    )
                )
            legacy = state.to_legacy()
            legacy.validation_error = error_message
            legacy.failure = {
                "terminal_reason": error_message,
                "reason": error_message,
            }
            legacy.fail_current_step()
            state = self._replace_from_legacy(context, state, legacy)
            await self._checkpoint(
                context,
                state,
                services,
                DurableBoundary.AFTER_TOOL_OUTCOME,
            )
            if isinstance(exc, asyncio.TimeoutError):
                raise
            yield EngineEmission(
                outcome=self._outcome(
                    state,
                    FinishReason.ERROR,
                    error=error_message,
                )
            )
            return
        recovery_state = self._copy_recovery_state(state.tool_recovery)
        recovery_decision = self.recovery_controller.decide(
            outcome,
            recovery_state,
            step_id=step.step_id,
            max_repair_attempts=budget.max_tool_repair_attempts,
        )
        safe_views = {view.call_id: view for view in outcome.sanitized_failure_views}
        for call, result in zip(outcome.calls, outcome.results):
            if result.success:
                content = result.model_content()
            else:
                view = safe_views.get(result.call_id)
                error = (
                    {"type": "execution_error", "retryable": False}
                    if view is None
                    else {
                        key: value
                        for key, value in view.to_dict().items()
                        if key
                        in {
                            "type",
                            "reason",
                            "recovery",
                            "retryable",
                            "message",
                        }
                    }
                )
                content = json.dumps(
                    {"success": False, "error": error},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            context.run.add_internal_message(
                Message.tool(
                    content,
                    call_id=call.id,
                    name=call.name,
                    metadata={"moduagent.ephemeral": True},
                )
            )
        context.run.metadata["_moduagent_resume_safety"] = (
            "manual_required" if outcome.success_count else "resumable"
        )
        previous_recovery_state = state.tool_recovery
        state.tool_recovery = recovery_state
        try:
            await self._checkpoint(
                context,
                state,
                services,
                DurableBoundary.AFTER_TOOL_OUTCOME,
            )
        finally:
            state.tool_recovery = previous_recovery_state
        self._sync_legacy(context, state)
        observed = await asyncio.wait_for(
            self.policy.observe(
                context.run,
                self._policy_result_projection(outcome),
            ),
            timeout=services.remaining_seconds(context),
        )
        state = self._pull_state(context, state)
        observed = self._reconcile_recovery(
            context,
            state,
            recovery_state,
            recovery_decision,
            observed,
        )
        if observed is not None:
            event = AgentEvent(
                EventType.POLICY_DECISION,
                context.run.run_id,
                {
                    "kind": observed.kind.value,
                    "metadata": dict(observed.metadata),
                    "source": "tool_results",
                },
                visibility=EventVisibility.INTERNAL,
            )
            yield await self._publish(context, services, event)

        if observed is not None and observed.kind is DecisionKind.RETRY_TOOL:
            failure = observed.metadata.get("tool_failure", {})
            event = AgentEvent(
                EventType.TOOL_REPAIR_SCHEDULED,
                context.run.run_id,
                {
                    "step_id": step.step_id,
                    "tool_name": failure.get("tool_name"),
                    "failed_call_id": failure.get("call_id"),
                    "error_type": failure.get("error_type"),
                    "reason": failure.get("reason"),
                    "repair_attempt": observed.metadata.get("repair_attempt"),
                    "max_attempts": services.budget(context).max_tool_repair_attempts,
                },
                visibility=EventVisibility.INTERNAL,
            )
            yield await self._publish(context, services, event)
            await self._checkpoint(
                context,
                state,
                services,
                DurableBoundary.REPAIR_SCHEDULED,
            )
            return

        if observed is not None and observed.kind is DecisionKind.REPLAN:
            event = AgentEvent(
                EventType.PLAN_REVISED,
                context.run.run_id,
                {
                    "plan_version": state.plan_progress.plan.version,
                    "replan_count": state.plan_progress.replan_count,
                    "reason": observed.metadata.get("reason"),
                },
                visibility=EventVisibility.INTERNAL,
            )
            yield await self._publish(context, services, event)
            context.run.clear_internal_messages()
            await self._checkpoint(
                context,
                state,
                services,
                DurableBoundary.REPLAN_COMPLETED,
            )
            return

        if (
            observed is not None and observed.kind is DecisionKind.FAIL
        ) or state.phase is PlanExecutionPhase.FAILED:
            validation_failure = (
                None
                if observed is None
                else self._validation_failure(observed.metadata, state, step)
            )
            event = AgentEvent(
                EventType.STEP_FAILED,
                context.run.run_id,
                {
                    "step_id": step.step_id,
                    "reason": (
                        None if observed is None else observed.metadata.get("reason")
                    ),
                    **(
                        {}
                        if validation_failure is None
                        else {
                            "validation_code": validation_failure["code"],
                            "validation_location": validation_failure["location"],
                            **(
                                {
                                    "validation_cause_code": (
                                        validation_failure["cause_code"]
                                    )
                                }
                                if "cause_code" in validation_failure
                                else {}
                            ),
                        }
                    ),
                },
                visibility=EventVisibility.INTERNAL,
            )
            yield await self._publish(context, services, event)
            yield EngineEmission(
                outcome=self._outcome(
                    state,
                    FinishReason.ERROR,
                    error=(
                        observed.error_message
                        if observed is not None
                        else self._failure_message(state)
                    ),
                    validation_failure=validation_failure,
                )
            )
            return

        await self._checkpoint(
            context,
            state,
            services,
            DurableBoundary.AFTER_TOOL_OUTCOME,
        )

    async def _model_turn(
        self,
        context: EngineContext,
        services: ExecutionServices,
        request: ModelRequest,
        *,
        phase: str,
    ) -> ModelResponse:
        response: ModelResponse | None = None
        if context.stream_model:
            async for chunk in services.stream_model(
                context,
                request,
                phase=phase,
                delta_event_type=EventType.STEP_MODEL_DELTA,
                delta_visibility=EventVisibility.INTERNAL,
                delta_data={"step": context.run.step},
            ):
                if chunk.response is not None:
                    response = chunk.response
        else:
            response = await services.request_model(
                context,
                request,
                phase=phase,
            )
        if response is None:
            raise RuntimeError("Plan model invocation ended without a response")
        return response

    async def _finalize(
        self,
        context: EngineContext,
        state: PlanEngineState,
        services: ExecutionServices,
    ) -> AsyncIterator[EngineEmission]:
        buffered_deltas: tuple[str, ...] = ()
        finalized: FinalizationResult
        if state.finalization.response is None:
            await self._checkpoint(
                context,
                state,
                services,
                DurableBoundary.FINALIZATION_STARTED,
            )
            self._sync_legacy(context, state)
            payload = self.policy.finalization_payload(context.run)
            request = ModelRequest(
                messages=(
                    Message.system(context.config.instructions),
                    Message.system(_FINALIZER_PROMPT),
                    Message.user(json.dumps(payload, ensure_ascii=False, default=str)),
                ),
                tools=(),
                output_schema=services.output_schema(context),
                options=self._model_options(context, has_tools=False),
            )
            request = await services.prepare_model_request(
                context,
                request,
                phase="finalize",
                skill_phase="finalize",
                protected_from=0,
            )
            finalized = await services.finalize(
                context,
                request,
                phase="finalize",
            )
            buffered_deltas = finalized.buffered_deltas
            self._sync_legacy(context, state)
            self.policy.record_final_response(
                context.run,
                finalized.content,
                persisted=False,
            )
            state = self._pull_state(context, state)
            await self._checkpoint(
                context,
                state,
                services,
                DurableBoundary.FINALIZATION_RESPONSE,
            )
        else:
            response = ModelResponse(Message.assistant(state.finalization.response))
            finalized = FinalizationResult(
                response=response,
                output=services.decode_output(context, response),
                content=state.finalization.response,
                persisted=state.finalization.persisted,
            )

        content = state.finalization.response
        if content is None:
            raise RuntimeError("finalization did not record a stable response")
        if not state.finalization.persisted:
            finalized = await services.persist_finalization(
                context,
                finalized,
            )
            if finalized.content != content or not finalized.persisted:
                raise RuntimeError(
                    "finalization persistence did not confirm the response"
                )
            self._sync_legacy(context, state)
            self.policy.record_final_response(
                context.run,
                content,
                persisted=True,
            )
            state = self._pull_state(context, state)
            await self._checkpoint(
                context,
                state,
                services,
                DurableBoundary.FINALIZATION_PERSISTED,
            )

        emit = not state.finalization.emitted
        if emit:
            self._sync_legacy(context, state)
            self.policy.record_final_response(
                context.run,
                content,
                persisted=True,
                emitted=True,
            )
            state = self._pull_state(context, state)
            await self._checkpoint(
                context,
                state,
                services,
                DurableBoundary.FINALIZATION_EMITTED,
            )
        if emit:
            await services.emit_finalization(
                context,
                replace(
                    finalized,
                    buffered_deltas=buffered_deltas,
                    persisted=True,
                ),
                phase="finalize",
            )
        yield EngineEmission(
            outcome=self._outcome(
                state,
                FinishReason.COMPLETED,
                output=finalized.output,
            )
        )

    def _step_messages(
        self,
        context: EngineContext,
        state: PlanEngineState,
        *,
        extracting: bool,
    ) -> tuple[Message, ...]:
        step = state.current_step
        if step is None:
            raise RuntimeError("Plan step context is missing")
        dependencies = {
            dependency: state.plan_progress.committed_results[dependency].model_dump(
                mode="json"
            )
            for dependency in step.dependencies
            if dependency in state.plan_progress.committed_results
        }
        messages = [
            Message.system(context.config.instructions),
            Message.system(_EXECUTOR_PROMPT),
            Message.user(
                json.dumps(
                    {
                        "current_step": step.to_dict(),
                        "dependency_results": dependencies,
                    },
                    ensure_ascii=False,
                    default=str,
                )
            ),
            *context.run.internal_messages,
        ]
        if extracting:
            feedback = state.step_execution.validation_feedback
            suffix = "" if not feedback else f"\nValidation feedback: {feedback}"
            messages.append(Message.user(f"{_STEP_RESULT_PROMPT}{suffix}"))
        elif state.tool_recovery.pending_failure is not None:
            failure = state.tool_recovery.pending_failure
            repair_context = {
                key: failure[key]
                for key in (
                    "tool_name",
                    "type",
                    "error_type",
                    "reason",
                    "feedback",
                    "message",
                )
                if key in failure
            }
            messages.append(
                Message.user(
                    "Call exactly the same Tool once with corrected arguments "
                    "and a new call ID. Do not repeat identical arguments, "
                    "select another Tool, or claim completion.\n"
                    + json.dumps(repair_context, ensure_ascii=False)
                )
            )
        else:
            messages.append(
                Message.user(
                    "Use an allowed Tool if needed. Stop after collecting "
                    "enough evidence; the runtime requests StepResult separately."
                )
            )
        return tuple(messages)

    @staticmethod
    def _legacy_recovery_controller(
        policy: PlanPolicyAdapter,
    ) -> ToolRecoveryController:
        """Translate the 0.3 policy option into the public recovery contract."""

        if not isinstance(policy, PlanAndExecutePolicy):
            return ToolRecoveryController()
        recovery = policy.tool_failure_recovery
        if recovery is None:
            return ToolRecoveryController(
                ToolRecoveryControllerConfig(
                    fallback=("replan" if policy.revise_on_tool_failure else "fail"),
                    allow_same_step_repair=False,
                )
            )
        return ToolRecoveryController(
            ToolRecoveryControllerConfig(
                fallback=recovery.fallback,
                allow_same_step_repair=True,
                require_repair_safe=recovery.require_repair_safe,
                feedback_mode=recovery.feedback_mode,
            )
        )

    @staticmethod
    def _copy_recovery_state(state: ToolRecoveryState) -> ToolRecoveryState:
        return ToolRecoveryState(
            active_calls={
                call_id: dict(call) for call_id, call in state.active_calls.items()
            },
            seen_call_ids=list(state.seen_call_ids),
            pending_failure=(
                None if state.pending_failure is None else dict(state.pending_failure)
            ),
            repair_count_by_step=dict(state.repair_count_by_step),
            total_repairs=state.total_repairs,
            terminal_failure=(
                None if state.terminal_failure is None else dict(state.terminal_failure)
            ),
        )

    @staticmethod
    def _policy_result_projection(
        outcome: ToolBatchOutcome,
    ) -> tuple[ToolResult, ...]:
        """Remove raw values, failures and arguments at the policy boundary."""

        safe_failures = {view.call_id: view for view in outcome.sanitized_failure_views}
        internal_failures = {failure.call_id: failure for failure in outcome.failures}
        projected: list[ToolResult] = []
        for result in outcome.results:
            if result.success:
                projected.append(
                    ToolResult.succeeded(
                        call_id=result.call_id,
                        tool_name=result.tool_name,
                        value=None,
                        attempts=result.attempts,
                        duration_seconds=result.duration_seconds,
                        repair_safe=result.repair_safe,
                    )
                )
                continue
            view = safe_failures.get(result.call_id)
            failure = internal_failures.get(result.call_id)
            if view is None:
                error = ToolError(
                    ToolErrorType.EXECUTION_ERROR,
                    "Tool execution failed",
                    reason="execution_error",
                )
                repair_safe = False
            else:
                error = ToolError(
                    view.error_type,
                    view.message or "Tool execution failed",
                    retryable=view.retryable,
                    reason=view.reason,
                    recovery=view.recovery,
                )
                repair_safe = bool(
                    failure is not None
                    and (failure.safety_profile.changed_argument_repair_safe)
                )
            projected.append(
                ToolResult.failed(
                    call_id=result.call_id,
                    tool_name=result.tool_name,
                    error=error,
                    attempts=result.attempts,
                    duration_seconds=result.duration_seconds,
                    repair_safe=repair_safe,
                )
            )
        return tuple(projected)

    @staticmethod
    def _reconcile_recovery(
        context: EngineContext,
        state: PlanEngineState,
        recovery_state: ToolRecoveryState,
        decision: ToolRecoveryDecision,
        observed: ExecutionDecision | None,
    ) -> ExecutionDecision | None:
        """Validate the legacy facade transition against the Engine decision.

        ``PlanAndExecutePolicy.observe`` remains a 0.3 state-transition
        compatibility adapter. The public controller is authoritative for the
        recovery kind, safe failure view and repair accounting.
        """

        if decision.kind is ToolRecoveryDecisionKind.CONTINUE:
            compatible = (
                observed is None and state.phase is PlanExecutionPhase.STEP_RESULT
            )
            normalized = None
        elif decision.kind is ToolRecoveryDecisionKind.REPAIR:
            compatible = (
                observed is not None
                and observed.kind is DecisionKind.RETRY_TOOL
                and state.phase is PlanExecutionPhase.TOOL_RECOVERY
            )
            normalized = ExecutionDecision(
                DecisionKind.RETRY_TOOL,
                metadata={
                    "reason": decision.reason,
                    "repair_attempt": decision.repair_attempt,
                    "count_attempt": False,
                    "tool_failure": (
                        {} if decision.failure is None else decision.failure.to_dict()
                    ),
                },
            )
        elif decision.kind is ToolRecoveryDecisionKind.REPLAN:
            compatible = state.phase is PlanExecutionPhase.STEP_PREPARE and (
                observed is None or observed.kind is DecisionKind.REPLAN
            )
            normalized = ExecutionDecision(
                DecisionKind.REPLAN,
                metadata={"reason": decision.reason},
            )
        else:
            compatible = state.phase is PlanExecutionPhase.FAILED and (
                observed is None or observed.kind is DecisionKind.FAIL
            )
            normalized = ExecutionDecision(
                DecisionKind.FAIL,
                error_message="Tool execution failed",
                metadata={
                    "reason": decision.reason,
                    "tool_failure": (
                        {} if decision.failure is None else decision.failure.to_dict()
                    ),
                },
            )

        if compatible:
            if decision.kind in {
                ToolRecoveryDecisionKind.CONTINUE,
                ToolRecoveryDecisionKind.REPAIR,
                ToolRecoveryDecisionKind.FAIL,
            }:
                state.tool_recovery = recovery_state
                if decision.kind is not ToolRecoveryDecisionKind.CONTINUE:
                    state.step_execution.validation_feedback = decision.reason
                PlanExecutionEngine._sync_legacy(context, state)
            return normalized

        legacy = state.to_legacy()
        legacy.validation_error = (
            "Tool recovery transition did not match the Engine decision"
        )
        legacy.failure = {
            "terminal_reason": legacy.validation_error,
            "reason": decision.reason,
        }
        if decision.failure is not None:
            legacy.failure.update(decision.failure.to_dict())
        legacy.fail_current_step()
        PlanExecutionEngine._replace_from_legacy(context, state, legacy)
        return ExecutionDecision(
            DecisionKind.FAIL,
            error_message="Tool recovery invariant failed",
            metadata={
                "reason": legacy.validation_error,
                "tool_failure": dict(legacy.failure),
            },
        )

    @staticmethod
    def _model_options(
        context: EngineContext,
        *,
        has_tools: bool,
    ) -> dict[str, Any]:
        options = dict(context.config.model_options)
        if not has_tools:
            options.pop("tool_choice", None)
            options.pop("parallel_tool_calls", None)
        return options

    @staticmethod
    def _step_tool_schemas(
        context: EngineContext,
        services: ExecutionServices,
        step: PlanStep,
    ) -> tuple[Any, ...]:
        if not step.allowed_tools:
            return ()
        return services.tool_schemas(
            context,
            frozenset(step.allowed_tools),
        )

    @staticmethod
    def _verify(state: PlanEngineState) -> None:
        plan = state.plan_progress.plan
        if not plan.complete:
            raise RuntimeError("Plan verification found incomplete steps")
        missing = {
            step.step_id
            for step in plan.steps
            if step.status.value == "completed"
            and step.step_id not in state.plan_progress.committed_results
        }
        if missing:
            raise RuntimeError(
                "Plan verification found completed steps without results: "
                + ", ".join(sorted(missing))
            )

    @staticmethod
    def _sync_legacy(
        context: EngineContext,
        state: PlanEngineState,
    ) -> ExecutionState:
        legacy = state.to_legacy()
        context.run.execution_state = legacy
        serialized = legacy.to_dict()
        context.run.policy_state["execution_state"] = serialized
        context.run.policy_state["plan"] = serialized["plan"]
        return legacy

    @staticmethod
    def _pull_state(
        context: EngineContext,
        current: PlanEngineState,
    ) -> PlanEngineState:
        legacy = context.run.execution_state
        if not isinstance(legacy, ExecutionState):
            raise RuntimeError("Plan policy lost its ExecutionState")
        updated = PlanEngineState.from_legacy(legacy)
        current.phase = updated.phase
        current.plan_progress = updated.plan_progress
        current.step_execution = updated.step_execution
        current.tool_recovery = updated.tool_recovery
        current.finalization = updated.finalization
        return current

    @staticmethod
    def _replace_from_legacy(
        context: EngineContext,
        current: PlanEngineState,
        legacy: ExecutionState,
    ) -> PlanEngineState:
        context.run.execution_state = legacy
        context.run.policy_state["execution_state"] = legacy.to_dict()
        context.run.policy_state["plan"] = legacy.plan.to_dict()
        return PlanExecutionEngine._pull_state(context, current)

    @staticmethod
    def _failure_message(state: PlanEngineState) -> str:
        failure = state.tool_recovery.terminal_failure
        if failure is not None:
            for key in ("feedback", "terminal_reason", "message", "reason"):
                value = failure.get(key)
                if value:
                    return str(value)
        return (
            state.step_execution.validation_feedback
            or "strict Plan-and-Execute execution failed"
        )

    @staticmethod
    def _validation_failure(
        metadata: Mapping[str, Any],
        state: PlanEngineState,
        step: PlanStep,
    ) -> dict[str, Any] | None:
        """Project only framework-owned validation codes across the Engine boundary."""

        code = metadata.get("validation_code")
        location = metadata.get("validation_location")
        if (
            not isinstance(code, str)
            or code not in _STEP_VALIDATION_CODES
            or not isinstance(location, str)
            or location not in _STEP_VALIDATION_LOCATIONS
        ):
            return None
        projected: dict[str, Any] = {
            "code": code,
            "location": location,
            "phase": state.phase.value,
            "step_id": step.step_id,
            "attempt": max(1, state.step_execution.step_attempt_count),
        }
        cause_code = metadata.get("validation_cause_code")
        if (
            isinstance(cause_code, str)
            and cause_code in _STEP_VALIDATION_CODES
            and cause_code != code
        ):
            projected["cause_code"] = cause_code
        return projected

    @staticmethod
    def _outcome(
        state: PlanEngineState,
        reason: FinishReason,
        *,
        output: Any = None,
        error: str | None = None,
        validation_failure: Mapping[str, Any] | None = None,
    ) -> EngineOutcome:
        if reason is not FinishReason.COMPLETED:
            state.terminal_finish_reason = reason
            state.terminal_error = error
        metadata: dict[str, Any] = {
            "plan": state.plan_progress.plan.to_dict(),
            "plan_usage": {
                "phase": state.phase.value,
                "committed_steps": len(state.plan_progress.committed_results),
                "replans": state.plan_progress.replan_count,
                "finalization_calls": (state.finalization.invocation_count),
            },
        }
        if state.tool_recovery.total_repairs:
            metadata["plan_usage"]["tool_repairs"] = state.tool_recovery.total_repairs
        if state.tool_recovery.terminal_failure is not None:
            metadata["failure"] = dict(state.tool_recovery.terminal_failure)
        if validation_failure is not None:
            metadata["validation_failure"] = dict(validation_failure)
        return EngineOutcome(
            reason,
            output=output,
            error=error,
            metadata=metadata,
        )


__all__ = ["PlanExecutionEngine", "PlanPolicyAdapter"]
