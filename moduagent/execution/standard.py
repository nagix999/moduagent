from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from moduagent.decision import DecisionKind, ExecutionDecision
from moduagent.errors import ModuAgentError
from moduagent.execution.base import (
    CodecBackedEngine,
    DurableBoundary,
    EngineContext,
    EngineEmission,
    EngineOutcome,
    EngineSnapshot,
    EngineStateCodec,
    ExecutionServices,
    FinalizationResult,
    ResumeValidation,
)
from moduagent.messages import FinishReason, Message
from moduagent.models import ModelRequest, ModelResponse
from moduagent.runtime.context import RunContext
from moduagent.runtime.events import AgentEvent, EventType, EventVisibility
from moduagent.tools import ToolError, ToolErrorType, ToolResult


_FINALIZATION_PROMPT = (
    "Using the preceding execution and Tool results, return only the final "
    "answer that matches the requested output contract. Do not call tools."
)


class StandardExecutionPhase(str, Enum):
    ACT = "act"
    FINALIZE = "finalize"
    DONE = "done"
    FAILED = "failed"


@runtime_checkable
class StandardPolicyAdapter(Protocol):
    """The explicit Standard policy surface consumed by the Engine."""

    async def begin(self, context: RunContext) -> None: ...

    async def decide(
        self,
        context: RunContext,
        response: ModelResponse,
    ) -> ExecutionDecision: ...

    async def observe(
        self,
        context: RunContext,
        results: Sequence[ToolResult],
    ) -> ExecutionDecision | None: ...

    def should_stop(self, context: RunContext) -> bool: ...


@dataclass(slots=True)
class StandardEngineState:
    phase: StandardExecutionPhase = StandardExecutionPhase.ACT
    model_turn: int = 0
    tool_call_count: int = 0
    finalization_response: str | None = None
    finalization_count: int = 0
    finalization_persisted: bool = False
    finalization_emitted: bool = False
    terminal_finish_reason: FinishReason | None = None
    terminal_error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.phase, StandardExecutionPhase):
            self.phase = StandardExecutionPhase(str(self.phase))
        for field_name in ("model_turn", "tool_call_count", "finalization_count"):
            if type(getattr(self, field_name)) is not int:
                raise TypeError(f"{field_name} must be an integer")
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if self.finalization_response is not None and not isinstance(
            self.finalization_response,
            str,
        ):
            raise TypeError("finalization_response must be a string")
        for field_name in (
            "finalization_persisted",
            "finalization_emitted",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a bool")
        if self.terminal_finish_reason is not None and not isinstance(
            self.terminal_finish_reason,
            FinishReason,
        ):
            self.terminal_finish_reason = FinishReason(str(self.terminal_finish_reason))
        if self.terminal_error is not None and not isinstance(self.terminal_error, str):
            raise TypeError("terminal_error must be a string")
        if (
            self.phase is StandardExecutionPhase.FAILED
            and self.terminal_finish_reason is None
        ):
            self.terminal_finish_reason = FinishReason.ERROR
        if (
            self.finalization_persisted or self.finalization_emitted
        ) and self.finalization_response is None:
            raise ValueError("finalization markers require a response")


class StandardStateCodec(EngineStateCodec[StandardEngineState]):
    engine_id = "standard"
    state_version = 1

    def encode(self, state: StandardEngineState) -> Mapping[str, Any]:
        if not isinstance(state, StandardEngineState):
            raise TypeError("state must be a StandardEngineState")
        return {
            "phase": state.phase.value,
            "model_turn": state.model_turn,
            "tool_call_count": state.tool_call_count,
            "finalization": {
                "started": (
                    state.finalization_count > 0
                    or state.finalization_response is not None
                ),
                "response_generated": (state.finalization_response is not None),
                "response": state.finalization_response,
                "invocation_count": state.finalization_count,
                "persisted": state.finalization_persisted,
                "emitted": state.finalization_emitted,
            },
            "terminal": {
                "finish_reason": (
                    None
                    if state.terminal_finish_reason is None
                    else state.terminal_finish_reason.value
                ),
                "error": state.terminal_error,
            },
        }

    def decode(self, payload: Mapping[str, Any]) -> StandardEngineState:
        if not isinstance(payload, Mapping):
            raise TypeError("Standard state payload must be a mapping")
        raw_finalization = payload.get("finalization")
        if raw_finalization is None:
            finalization: Mapping[str, Any] = {
                "response": payload.get("finalization_response"),
                "invocation_count": payload.get("finalization_count", 0),
                "persisted": payload.get("finalization_persisted", False),
                "emitted": payload.get("finalization_emitted", False),
            }
        elif isinstance(raw_finalization, Mapping):
            finalization = raw_finalization
        else:
            raise ValueError("Standard finalization state must be an object")
        raw_terminal = payload.get("terminal", {})
        if not isinstance(raw_terminal, Mapping):
            raise ValueError("Standard terminal state must be an object")
        return StandardEngineState(
            phase=StandardExecutionPhase(
                str(payload.get("phase", StandardExecutionPhase.ACT.value))
            ),
            model_turn=int(payload.get("model_turn", payload.get("step", 0))),
            tool_call_count=int(payload.get("tool_call_count", 0)),
            finalization_response=(
                None
                if finalization.get("response") is None
                else str(finalization["response"])
            ),
            finalization_count=int(finalization.get("invocation_count", 0)),
            finalization_persisted=bool(finalization.get("persisted", False)),
            finalization_emitted=bool(finalization.get("emitted", False)),
            terminal_finish_reason=(
                None
                if raw_terminal.get("finish_reason") is None
                else FinishReason(str(raw_terminal["finish_reason"]))
            ),
            terminal_error=(
                None
                if raw_terminal.get("error") is None
                else str(raw_terminal["error"])
            ),
        )

    def migrate(
        self,
        from_version: int,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if type(from_version) is not int:
            raise TypeError("from_version must be an integer")
        if not isinstance(payload, Mapping):
            raise TypeError("Standard state payload must be a mapping")
        if from_version in {0, 1, 3}:
            return self.encode(self.decode(payload))
        raise ValueError(f"unsupported Standard state version: {from_version}")


class StandardExecutionEngine(CodecBackedEngine[StandardEngineState]):
    """0.3 DecisionPolicy compatibility adapter with an Engine-owned loop."""

    engine_id = "standard"
    state_version = 1

    def __init__(self, policy: StandardPolicyAdapter) -> None:
        if not isinstance(policy, StandardPolicyAdapter):
            raise TypeError("policy must implement StandardPolicyAdapter")
        self.policy = policy
        self.state_codec = StandardStateCodec()

    async def initialize(
        self,
        context: EngineContext,
        services: ExecutionServices,
    ) -> StandardEngineState:
        if not isinstance(context, EngineContext):
            raise TypeError("context must be an EngineContext")
        if not isinstance(services, ExecutionServices):
            raise TypeError("services must implement ExecutionServices")
        await self.policy.begin(context.run)
        state = StandardEngineState(
            model_turn=context.run.step,
            tool_call_count=context.run.tool_call_count,
        )
        await self._checkpoint(
            context,
            state,
            services,
            DurableBoundary.INITIALIZED,
        )
        return state

    def validate_resume(
        self,
        snapshot: EngineSnapshot,
        resolved_spec: Mapping[str, Any],
    ) -> ResumeValidation:
        validation = super().validate_resume(snapshot, resolved_spec)
        if not validation.compatible:
            return validation
        raw_common = resolved_spec.get("common_state")
        if raw_common is None:
            return validation
        if not isinstance(raw_common, Mapping):
            return ResumeValidation.rejected("resolved common_state must be a mapping")
        try:
            common_step = raw_common["step"]
            common_tool_calls = raw_common["tool_call_count"]
            if (
                type(common_step) is not int
                or common_step < 0
                or type(common_tool_calls) is not int
                or common_tool_calls < 0
            ):
                raise ValueError
            payload = (
                snapshot.state
                if snapshot.state_version == self.state_version
                else self.migrate_state(
                    snapshot.state_version,
                    snapshot.state,
                )
            )
            state = self.decode_state(payload)
        except (KeyError, TypeError, ValueError, OverflowError):
            return ResumeValidation.rejected("Standard resume counters are invalid")
        if common_step != state.model_turn:
            return ResumeValidation.rejected(
                "Standard model turn does not match common step"
            )
        if common_tool_calls != state.tool_call_count:
            return ResumeValidation.rejected(
                "Standard Tool count does not match common Tool count"
            )
        return validation

    async def execute(
        self,
        context: EngineContext,
        state: StandardEngineState,
        services: ExecutionServices,
    ) -> AsyncIterator[EngineEmission]:
        if not isinstance(context, EngineContext):
            raise TypeError("context must be an EngineContext")
        if not isinstance(state, StandardEngineState):
            raise TypeError("state must be a StandardEngineState")
        if not isinstance(services, ExecutionServices):
            raise TypeError("services must implement ExecutionServices")
        budget = services.budget(context)
        tools = services.tool_schemas(context)
        output_schema = services.output_schema(context)
        staged_finalization = context.config.finalization_mode == "always" or (
            bool(tools and output_schema is not None)
            and (
                context.config.finalization_mode == "structured_only"
                or not (context.model_capabilities.tool_calling_with_structured_output)
            )
        )
        last_response: ModelResponse | None = None
        last_assistant_message: Message | None = None

        if state.phase is StandardExecutionPhase.DONE:
            if state.finalization_response is None:
                raise RuntimeError("DONE Standard state has no final response")
            response = ModelResponse(Message.assistant(state.finalization_response))
            yield EngineEmission(
                outcome=EngineOutcome(
                    FinishReason.COMPLETED,
                    output=services.decode_output(context, response),
                )
            )
            return
        if state.phase is StandardExecutionPhase.FAILED:
            yield EngineEmission(
                outcome=EngineOutcome(
                    state.terminal_finish_reason or FinishReason.ERROR,
                    error=state.terminal_error or "Standard execution failed",
                )
            )
            return
        if state.phase is StandardExecutionPhase.FINALIZE:
            async for emission in self._finalize(
                context,
                state,
                services,
                output_schema,
            ):
                yield emission
            return

        while state.model_turn < budget.max_steps:
            if services.remaining_seconds(context) <= 0:
                raise asyncio.TimeoutError
            if self.policy.should_stop(context.run) and last_response is not None:
                if staged_finalization:
                    async for emission in self._finalize(
                        context,
                        state,
                        services,
                        output_schema,
                    ):
                        yield emission
                    return
                if last_assistant_message is not None and not (
                    last_response.tool_calls or last_response.message.tool_calls
                ):
                    self._promote_public_response(
                        context.run,
                        last_assistant_message,
                    )
                self._record_final_response(
                    context,
                    state,
                    last_response.message.content,
                    replay_safe=True,
                )
                state.phase = StandardExecutionPhase.DONE
                yield EngineEmission(
                    outcome=EngineOutcome(
                        FinishReason.COMPLETED,
                        output=services.decode_output(context, last_response),
                    )
                )
                return

            state.phase = StandardExecutionPhase.ACT
            state.model_turn += 1
            context.run.step = state.model_turn
            request = ModelRequest(
                messages=tuple(context.run.messages),
                tools=tools,
                output_schema=(None if staged_finalization else output_schema),
                options=dict(context.config.model_options),
            )
            request = await services.prepare_model_request(
                context,
                request,
                phase="act",
                skill_phase="act",
            )
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
                phase="act",
                public_deltas=not tools and not staged_finalization,
            )
            last_response = response
            calls = tuple(response.tool_calls or response.message.tool_calls)
            assistant_message = Message.assistant(
                response.message.content,
                calls,
                metadata={"moduagent.ephemeral": True},
            )
            last_assistant_message = assistant_message
            context.run.add_message(
                assistant_message,
                persist=False,
            )

            decision = await self.policy.decide(context.run, response)
            if (
                decision.kind is DecisionKind.FINISH
                and not staged_finalization
                and not calls
                and decision.final_output is None
            ):
                self._promote_public_response(context.run, assistant_message)
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

            if decision.kind is DecisionKind.FINISH:
                if staged_finalization and decision.final_output is None:
                    async for emission in self._finalize(
                        context,
                        state,
                        services,
                        output_schema,
                    ):
                        yield emission
                    return
                output = (
                    decision.final_output
                    if decision.final_output is not None
                    else services.decode_output(context, response)
                )
                self._record_final_response(
                    context,
                    state,
                    response.message.content,
                    replay_safe=decision.final_output is None,
                )
                state.phase = StandardExecutionPhase.DONE
                yield EngineEmission(
                    outcome=EngineOutcome(FinishReason.COMPLETED, output=output)
                )
                return

            if decision.kind is DecisionKind.FAIL:
                state.phase = StandardExecutionPhase.FAILED
                state.terminal_finish_reason = FinishReason.ERROR
                state.terminal_error = (
                    decision.error_message or "decision policy failed the run"
                )
                yield EngineEmission(
                    outcome=EngineOutcome(
                        state.terminal_finish_reason,
                        error=state.terminal_error,
                    )
                )
                return

            if decision.kind is DecisionKind.CONTINUE:
                instruction = decision.metadata.get("instruction")
                if instruction:
                    context.run.add_message(
                        Message.system(str(instruction)),
                        persist=False,
                    )
                context.run.metadata["_moduagent_resume_safety"] = "resumable"
                await self._checkpoint(
                    context,
                    state,
                    services,
                    DurableBoundary.AFTER_TOOL_OUTCOME,
                )
                continue

            if decision.kind is not DecisionKind.CALL_TOOLS:
                raise RuntimeError(
                    "Standard Engine received a Plan-only decision: "
                    f"{decision.kind.value}"
                )

            calls = tuple(decision.tool_calls)
            exempt_tools = {
                schema.name for schema in tools if not schema.counts_toward_tool_limit
            }
            counted_calls = sum(call.name not in exempt_tools for call in calls)
            if state.tool_call_count + counted_calls > budget.max_tool_calls:
                for call in calls:
                    rejected = ToolResult.failed(
                        call_id=call.id,
                        tool_name=call.name,
                        error=ToolError(
                            ToolErrorType.EXECUTION_ERROR,
                            "tool call limit exceeded",
                            reason="tool_call_limit_exceeded",
                        ),
                    )
                    await services.record_tool_result(context, call, rejected)
                    context.run.add_message(
                        Message.tool(
                            self._safe_failure_content(
                                error_type=ToolErrorType.EXECUTION_ERROR,
                                reason="tool_call_limit_exceeded",
                            ),
                            call_id=call.id,
                            name=call.name,
                            metadata={"moduagent.ephemeral": True},
                        ),
                        persist=False,
                    )
                state.phase = StandardExecutionPhase.FAILED
                state.terminal_finish_reason = FinishReason.MAX_TOOL_CALLS
                state.terminal_error = "tool call limit exceeded"
                yield EngineEmission(
                    outcome=EngineOutcome(
                        state.terminal_finish_reason,
                        error=state.terminal_error,
                    )
                )
                return

            state.tool_call_count += counted_calls
            context.run.tool_call_count = state.tool_call_count
            try:
                outcome = await services.execute_tool_batch(
                    context,
                    calls,
                    allowed_tools=None,
                    repair_constraint=None,
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
                    context.run.add_message(
                        Message.tool(
                            self._safe_failure_content(
                                error_type=error_type,
                                reason=error_type.value,
                                message=error_message,
                            ),
                            call_id=call.id,
                            name=call.name,
                            metadata={"moduagent.ephemeral": True},
                        ),
                        persist=False,
                    )
                await self._checkpoint(
                    context,
                    state,
                    services,
                    DurableBoundary.AFTER_TOOL_OUTCOME,
                )
                raise
            safe_views = {
                view.call_id: view for view in outcome.sanitized_failure_views
            }
            for call, result in zip(outcome.calls, outcome.results):
                view = safe_views.get(result.call_id)
                context.run.add_message(
                    Message.tool(
                        (
                            result.model_content()
                            if result.success
                            else self._safe_failure_content(view=view)
                        ),
                        call_id=call.id,
                        name=call.name,
                        metadata={"moduagent.ephemeral": True},
                    ),
                    persist=False,
                )

            await self._checkpoint(
                context,
                state,
                services,
                DurableBoundary.AFTER_TOOL_OUTCOME,
            )
            await self.policy.observe(
                context.run,
                self._policy_result_projection(outcome),
            )

        state.phase = StandardExecutionPhase.FAILED
        state.terminal_finish_reason = FinishReason.MAX_STEPS
        state.terminal_error = "model step limit exceeded"
        yield EngineEmission(
            outcome=EngineOutcome(
                state.terminal_finish_reason,
                error=state.terminal_error,
            )
        )

    @staticmethod
    def _promote_public_response(
        context: RunContext,
        message: Message,
    ) -> None:
        metadata = dict(message.metadata)
        metadata.pop("moduagent.ephemeral", None)
        public_message = replace(message, metadata=metadata)
        for index in range(len(context.messages) - 1, -1, -1):
            if context.messages[index] is message:
                context.messages[index] = public_message
                context.new_messages.append(public_message)
                return
        raise RuntimeError("Standard response disappeared before publication")

    @staticmethod
    def _record_final_response(
        context: EngineContext,
        state: StandardEngineState,
        content: str | None,
        *,
        replay_safe: bool,
    ) -> None:
        """Make a stable final response the next replay-safe boundary."""

        state.finalization_response = content if replay_safe else None
        context.run.metadata["_moduagent_resume_safety"] = (
            "resumable" if content is not None and replay_safe else "manual_required"
        )

    @staticmethod
    def _safe_failure_content(
        *,
        view: Any | None = None,
        error_type: ToolErrorType = ToolErrorType.EXECUTION_ERROR,
        reason: str = "execution_error",
        message: str | None = None,
    ) -> str:
        error: dict[str, Any]
        if view is None:
            error = {
                "type": error_type.value,
                "reason": reason,
                "retryable": False,
            }
            if message is not None:
                error["message"] = message
        else:
            error = {
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
        return json.dumps(
            {"success": False, "error": error},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _policy_result_projection(
        outcome: Any,
    ) -> tuple[ToolResult, ...]:
        safe_views = {view.call_id: view for view in outcome.sanitized_failure_views}
        projected: list[ToolResult] = []
        for result in outcome.results:
            if result.success:
                projected.append(result)
                continue
            view = safe_views.get(result.call_id)
            if view is None:
                error = ToolError(
                    ToolErrorType.EXECUTION_ERROR,
                    "tool execution failed",
                    reason="execution_error",
                )
            else:
                error = ToolError(
                    view.error_type,
                    view.message or "tool execution failed",
                    retryable=view.retryable,
                    reason=view.reason,
                    recovery=view.recovery,
                )
            projected.append(
                ToolResult.failed(
                    call_id=result.call_id,
                    tool_name=result.tool_name,
                    error=error,
                    attempts=result.attempts,
                    duration_seconds=result.duration_seconds,
                    repair_safe=result.repair_safe,
                )
            )
        return tuple(projected)

    async def _model_turn(
        self,
        context: EngineContext,
        services: ExecutionServices,
        request: ModelRequest,
        *,
        phase: str,
        public_deltas: bool,
    ) -> ModelResponse:
        response: ModelResponse | None = None
        if context.stream_model:
            async for chunk in services.stream_model(
                context,
                request,
                phase=phase,
                delta_event_type=EventType.MODEL_DELTA,
                delta_visibility=(
                    EventVisibility.PUBLIC
                    if public_deltas
                    else EventVisibility.INTERNAL
                ),
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
            raise RuntimeError("model invocation ended without a response")
        return response

    async def _finalize(
        self,
        context: EngineContext,
        state: StandardEngineState,
        services: ExecutionServices,
        output_schema: Mapping[str, Any] | None,
    ) -> AsyncIterator[EngineEmission]:
        state.phase = StandardExecutionPhase.FINALIZE
        buffered_deltas: tuple[str, ...] = ()
        finalized: FinalizationResult
        if state.finalization_response is None:
            state.finalization_count += 1
            await self._checkpoint(
                context,
                state,
                services,
                DurableBoundary.FINALIZATION_STARTED,
            )
            request = ModelRequest(
                messages=(
                    *context.run.messages,
                    Message.user(_FINALIZATION_PROMPT),
                ),
                tools=(),
                output_schema=output_schema,
                options={
                    key: value
                    for key, value in context.config.model_options.items()
                    if key not in {"tool_choice", "parallel_tool_calls"}
                },
            )
            request = await services.prepare_model_request(
                context,
                request,
                phase="finalize",
                skill_phase="finalize",
            )
            finalized = await services.finalize(
                context,
                request,
                phase="finalize",
            )
            self._record_final_response(
                context,
                state,
                finalized.content,
                replay_safe=True,
            )
            buffered_deltas = finalized.buffered_deltas
            await self._checkpoint(
                context,
                state,
                services,
                DurableBoundary.FINALIZATION_RESPONSE,
            )
        else:
            response = ModelResponse(Message.assistant(state.finalization_response))
            finalized = FinalizationResult(
                response=response,
                output=services.decode_output(context, response),
                content=state.finalization_response,
                persisted=state.finalization_persisted,
            )

        content = state.finalization_response
        if content is None:
            raise RuntimeError("finalization did not record a stable response")
        if not state.finalization_persisted:
            finalized = await services.persist_finalization(
                context,
                finalized,
            )
            if finalized.content != content or not finalized.persisted:
                raise RuntimeError(
                    "finalization persistence did not confirm the response"
                )
            state.finalization_persisted = True
            await self._checkpoint(
                context,
                state,
                services,
                DurableBoundary.FINALIZATION_PERSISTED,
            )

        emit = not state.finalization_emitted
        if emit:
            state.finalization_emitted = True
            state.phase = StandardExecutionPhase.DONE
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
            outcome=EngineOutcome(
                FinishReason.COMPLETED,
                output=finalized.output,
            )
        )


__all__ = [
    "StandardEngineState",
    "StandardExecutionEngine",
    "StandardExecutionPhase",
    "StandardPolicyAdapter",
    "StandardStateCodec",
]
