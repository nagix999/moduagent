from __future__ import annotations

import asyncio
import inspect
import json
import math
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import replace
from typing import Any, TypeVar

from moduagent.config import AgentConfig
from moduagent.decision import DecisionKind, DecisionPolicy
from moduagent.decision.planning import ExecutionState, RunPhase
from moduagent.memory import (
    ConversationMemoryPolicy,
    FullConversationMemoryPolicy,
    MemoryPhase,
    MemoryRequest,
)
from moduagent.messages import FinishReason, Message, MessageRole
from moduagent.models import ModelClient, ModelRequest, ModelResponse
from moduagent.observability import EventSink, mask_sensitive
from moduagent.output import OutputCodec
from moduagent.persistence import (
    CheckpointStore,
    ConversationStore,
)
from moduagent.runtime.context import (
    AgentResult,
    RunContext,
    RunRequest,
    RunStatus,
)
from moduagent.runtime.events import AgentEvent, EventType, EventVisibility
from moduagent.skills.prompting import compose_skill_prompt, is_ephemeral_message
from moduagent.skills.runtime import SkillRuntime
from moduagent.skills.tools import (
    SKILL_READ_TOOL_NAME,
    SKILL_RESOURCE_TOOL_NAMES,
    SKILL_SEARCH_TOOL_NAME,
)
from moduagent.tools import (
    ToolError,
    ToolErrorType,
    ToolExecutionContext,
    ToolExecutor,
    ToolRecoveryAction,
    ToolResult,
)
from moduagent.tools.base import _TOOL_REPAIR_METADATA_KEY


T = TypeVar("T")

_FINALIZATION_STATE_KEY = "_moduagent_structured_finalization"
_FINALIZATION_OUTPUT_KEY = "_moduagent_structured_output"
_FINALIZATION_PENDING = "pending"
_FINALIZATION_COMPLETED = "completed"
_FINALIZATION_PROMPT = (
    "Using the preceding execution and tool results, return only the final answer "
    "that matches the required response schema. Do not call tools."
)
_STRICT_EXECUTOR_PROMPT = (
    "Execute exactly one current plan step. Do not write the public final answer, "
    "perform another step, or claim unsupported facts. Use only the supplied tools. "
    "When the runtime requests a StepResult, return only that strict JSON object."
)
_STRICT_STEP_RESULT_PROMPT = (
    "Return the current step result now. Use only the StepResult schema, keep the "
    "step_id unchanged, and provide one completion_evidence item for every "
    "completion criterion. Do not include final_answer, verdict, recommendation, "
    "or any field outside the schema."
)
_STRICT_FINALIZER_PROMPT = (
    "Create the one public final response from the original objective and committed "
    "step results. Do not call tools, add new facts, expose internal execution logs, "
    "or perform more work. Return only the requested public response."
)
_RUN_ID_METADATA_KEY = "moduagent.run_id"
_PUBLIC_FINAL_METADATA_KEY = "moduagent.public_final"
_TOOL_TRACE_METADATA_KEY = "_moduagent_tool_trace"
_PUBLIC_TOOL_TRACE_KEY = "tool_trace"
_TOOL_TRACE_ARGUMENT_BYTES = 4096
_TOOL_TRACE_TEXT_CHARS = 256


class _StrictToolCallLimitError(RuntimeError):
    pass


class _StrictToolResultProtocolError(RuntimeError):
    pass


class AgentRuntime:
    """Provider-neutral model/tool loop and execution lifecycle."""

    def __init__(
        self,
        *,
        config: AgentConfig,
        model: ModelClient,
        decision_policy: DecisionPolicy,
        tool_executor: ToolExecutor,
        conversation_store: ConversationStore,
        output_codec: OutputCodec,
        event_sink: EventSink,
        checkpoint_store: CheckpointStore | None = None,
        conversation_memory_policy: ConversationMemoryPolicy | None = None,
        skill_runtime: SkillRuntime | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.decision_policy = decision_policy
        self.tool_executor = tool_executor
        self.conversation_store = conversation_store
        self.output_codec = output_codec
        self.event_sink = event_sink
        self.checkpoint_store = checkpoint_store
        self.skill_runtime = skill_runtime
        self.conversation_memory_policy = (
            conversation_memory_policy
            if conversation_memory_policy is not None
            else FullConversationMemoryPolicy()
        )
        if (
            bool(
                getattr(decision_policy, "strict_plan_execution", False)
                or getattr(decision_policy, "strict_execution", False)
            )
            and config.finalization_mode == "disabled"
        ):
            raise ValueError(
                "strict Plan-and-Execute requires finalization_mode to be enabled"
            )
        self._session_locks: dict[str, asyncio.Lock] = {}

    async def execute(self, request: RunRequest) -> AgentResult:
        result: AgentResult | None = None
        async for event in self._events(request, stream_model=False):
            if event.type in (EventType.RUN_COMPLETED, EventType.RUN_FAILED):
                candidate = event.data.get("result")
                if isinstance(candidate, AgentResult):
                    result = candidate
        if result is None:
            raise RuntimeError("agent run ended without a terminal result")
        return result

    async def stream(
        self,
        request: RunRequest,
        *,
        include_internal: bool | None = None,
    ) -> AsyncIterator[AgentEvent]:
        include_internal = (
            self.config.stream_visibility == "all"
            if include_internal is None
            else include_internal
        )
        async for event in self._events(request, stream_model=True):
            if include_internal or event.visibility is EventVisibility.PUBLIC:
                yield event

    async def _events(
        self, request: RunRequest, *, stream_model: bool
    ) -> AsyncIterator[AgentEvent]:
        lock = self._session_locks.setdefault(request.session_id, asyncio.Lock())
        async with lock:
            async for event in self._run(request, stream_model=stream_model):
                yield event

    async def _run(
        self, request: RunRequest, *, stream_model: bool
    ) -> AsyncIterator[AgentEvent]:
        run_id = request.resume_run_id or uuid.uuid4().hex
        deadline = (
            asyncio.get_running_loop().time() + self.config.limits.timeout_seconds
        )
        strict_execution = bool(
            getattr(self.decision_policy, "strict_plan_execution", False)
            or getattr(self.decision_policy, "strict_execution", False)
        )
        user_message = Message.user(
            request.input,
            metadata=(
                {
                    _RUN_ID_METADATA_KEY: run_id,
                    "moduagent.public_input": True,
                }
                if strict_execution
                else None
            ),
        )
        initial_metadata = {"agent": self.config.name, **dict(self.config.metadata)}
        # This key is runtime-owned. AgentConfig.metadata must not be able to
        # forge persisted Tool audit entries.
        initial_metadata.pop(_TOOL_TRACE_METADATA_KEY, None)
        initial_metadata.pop(_PUBLIC_TOOL_TRACE_KEY, None)
        context = RunContext(
            run_id=run_id,
            request=request,
            messages=[Message.system(self.config.instructions), user_message],
            new_messages=[user_message],
            metadata=initial_metadata,
            current_run_start=1,
        )
        last_response: ModelResponse | None = None

        event = AgentEvent(
            EventType.RUN_STARTED,
            run_id,
            {
                "agent": self.config.name,
                "session_id": request.session_id,
                "user_context": dict(request.user_context),
            },
        )
        await self._publish(event)
        yield event

        try:
            if request.resume_run_id:
                if self.checkpoint_store is None:
                    raise RuntimeError("checkpoint_store is required to resume a run")
                checkpoint = await self._within(
                    deadline,
                    lambda: self.checkpoint_store.load(request.resume_run_id),
                )
                if checkpoint is None:
                    raise LookupError(f"checkpoint not found: {request.resume_run_id}")
                if checkpoint.session_id != request.session_id:
                    raise ValueError("checkpoint session_id does not match the request")
                context = checkpoint.to_context()
                self._normalize_context_tool_trace(context)
                event = AgentEvent(
                    EventType.CHECKPOINT_LOADED,
                    context.run_id,
                    {"step": context.step, "status": context.status.value},
                )
                await self._publish(event)
                yield event
            else:
                history = await self._within(
                    deadline,
                    lambda: self.conversation_store.load(request.session_id),
                )
                context.messages = [
                    Message.system(self.config.instructions),
                    *history,
                    user_message,
                ]
                context.current_run_start = 1 + len(history)

            async for skill_event in self._skill_events(
                context,
                deadline,
                resumed=bool(request.resume_run_id),
            ):
                yield skill_event
            await self._save_checkpoint(context, deadline)

            context.status = RunStatus.RUNNING
            tool_schemas = self._tool_schemas(context)
            context.metadata["_moduagent_available_tools"] = [
                schema.name for schema in tool_schemas
            ]
            configure_limits = getattr(self.decision_policy, "configure_limits", None)
            if strict_execution and callable(configure_limits):
                configure_limits(
                    max_step_attempts=self.config.limits.max_step_attempts,
                    max_replans=self.config.limits.max_replans,
                )
            configure_repair_limits = getattr(
                self.decision_policy,
                "configure_tool_repair_limits",
                None,
            )
            if strict_execution and callable(configure_repair_limits):
                configure_repair_limits(
                    max_tool_repair_attempts=getattr(
                        self.config.limits,
                        "max_tool_repair_attempts",
                        1,
                    )
                )
            creating_plan = strict_execution and context.execution_state is None
            await self._within(deadline, lambda: self.decision_policy.begin(context))
            await self._save_checkpoint(context, deadline)

            if strict_execution:
                state = context.execution_state
                if not isinstance(state, ExecutionState):
                    raise RuntimeError(
                        "strict Plan-and-Execute did not initialize ExecutionState"
                    )
                if creating_plan:
                    plan_event = AgentEvent(
                        EventType.PLAN_CREATED,
                        run_id,
                        {
                            "step_count": len(state.plan.steps),
                            "plan_version": state.plan.version,
                        },
                        visibility=EventVisibility.INTERNAL,
                    )
                    await self._publish(plan_event)
                    yield plan_event
                async for strict_event in self._strict_plan_events(
                    context,
                    tool_schemas,
                    deadline,
                    stream_model=stream_model,
                ):
                    yield strict_event
                return

            output_schema = self.output_codec.schema()
            staged_finalization = self.config.finalization_mode == "always" or (
                self.config.finalization_mode == "structured_only"
                and bool(tool_schemas and output_schema is not None)
            )

            if staged_finalization and context.policy_state.get(
                _FINALIZATION_STATE_KEY
            ) in (_FINALIZATION_PENDING, _FINALIZATION_COMPLETED):
                async for event in self._finalization_events(
                    context,
                    output_schema,
                    deadline,
                    stream_model=stream_model,
                ):
                    yield event
                return

            while context.step < self.config.limits.max_steps:
                if self.decision_policy.should_stop(context) and last_response:
                    if staged_finalization:
                        context.policy_state[_FINALIZATION_STATE_KEY] = (
                            _FINALIZATION_PENDING
                        )
                        await self._save_checkpoint(context, deadline)
                        async for event in self._finalization_events(
                            context,
                            output_schema,
                            deadline,
                            stream_model=stream_model,
                        ):
                            yield event
                        return
                    result = await self._finalize(
                        context,
                        last_response,
                        FinishReason.COMPLETED,
                        deadline,
                    )
                    event = AgentEvent(
                        EventType.RUN_COMPLETED, run_id, {"result": result}
                    )
                    await self._publish(event)
                    yield event
                    return

                context.step += 1
                context.status = RunStatus.WAITING_FOR_MODEL
                await self._save_checkpoint(context, deadline)
                request_model = ModelRequest(
                    messages=tuple(context.messages),
                    tools=tool_schemas,
                    output_schema=None if staged_finalization else output_schema,
                    options=dict(self.config.model_options),
                )
                request_model, memory_event = await self._prepare_model_request(
                    context,
                    request_model,
                    phase=MemoryPhase.ACT,
                    deadline=deadline,
                    skill_phase="act",
                )
                if memory_event is not None:
                    yield memory_event

                response: ModelResponse | None = None
                async for event in self._model_events(
                    context,
                    request_model,
                    deadline,
                    stream_model=stream_model,
                    phase="act",
                ):
                    if event.type is EventType.MODEL_COMPLETED:
                        candidate = event.data.get("response")
                        if isinstance(candidate, ModelResponse):
                            response = candidate
                    yield event
                if response is None:
                    raise RuntimeError("model invocation ended without a response")
                last_response = response

                decision = await self._within(
                    deadline,
                    lambda: self.decision_policy.decide(context, response),
                )
                context.metadata.update(dict(decision.metadata))
                decision_event = AgentEvent(
                    EventType.POLICY_DECISION,
                    run_id,
                    {"kind": decision.kind.value, "metadata": dict(decision.metadata)},
                )
                await self._publish(decision_event)
                yield decision_event

                if decision.kind is DecisionKind.FINISH:
                    if staged_finalization and decision.final_output is None:
                        context.policy_state[_FINALIZATION_STATE_KEY] = (
                            _FINALIZATION_PENDING
                        )
                        await self._save_checkpoint(context, deadline)
                        async for event in self._finalization_events(
                            context,
                            output_schema,
                            deadline,
                            stream_model=stream_model,
                        ):
                            yield event
                        return
                    output = (
                        decision.final_output
                        if decision.final_output is not None
                        else self.output_codec.decode(response)
                    )
                    result = await self._finalize(
                        context,
                        response,
                        FinishReason.COMPLETED,
                        deadline,
                        output=output,
                    )
                    event = AgentEvent(
                        EventType.RUN_COMPLETED, run_id, {"result": result}
                    )
                    await self._publish(event)
                    yield event
                    return

                if decision.kind is DecisionKind.FAIL:
                    raise RuntimeError(
                        decision.error_message or "decision policy failed the run"
                    )

                if decision.kind is DecisionKind.CONTINUE:
                    instruction = decision.metadata.get("instruction")
                    if instruction:
                        context.add_message(
                            Message.system(str(instruction)), persist=False
                        )
                    await self._save_checkpoint(context, deadline)
                    continue

                calls = tuple(decision.tool_calls)
                skill_resource_calls = tuple(
                    call for call in calls if call.name in SKILL_RESOURCE_TOOL_NAMES
                )
                business_calls = tuple(
                    call for call in calls if call.name not in SKILL_RESOURCE_TOOL_NAMES
                )
                protocol_error: str | None = None
                if skill_resource_calls and business_calls:
                    protocol_error = (
                        "a model response cannot mix Skill resource and business tools"
                    )
                elif skill_resource_calls:
                    if self.skill_runtime is None:
                        protocol_error = "Skill resource tools are not configured"
                    else:
                        next_resource_reads = context.skill_state.resource_reads + len(
                            skill_resource_calls
                        )
                        if (
                            next_resource_reads
                            > self.skill_runtime.limits.max_resource_reads
                        ):
                            protocol_error = "Skill resource read limit exceeded"
                        else:
                            context.skill_state = replace(
                                context.skill_state,
                                resource_reads=next_resource_reads,
                            )
                if protocol_error is not None:
                    rejected_results = self._record_rejected_tool_calls(
                        context,
                        calls,
                        error_message=protocol_error,
                    )
                    for call, rejected in zip(calls, rejected_results):
                        is_skill_resource = call.name in SKILL_RESOURCE_TOOL_NAMES
                        event_data: dict[str, Any] = {
                            "tool_name": call.name,
                            "success": False,
                            "error": protocol_error,
                        }
                        if is_skill_resource:
                            event_data.update(
                                {
                                    "skill_name": call.arguments.get("skill_name"),
                                    "path": call.arguments.get("path"),
                                }
                            )
                        else:
                            event_data.update({"tool_call": call, "result": rejected})
                        tool_event = AgentEvent(
                            EventType.TOOL_COMPLETED,
                            run_id,
                            event_data,
                        )
                        await self._publish(tool_event)
                        yield tool_event
                    # This is a terminal protocol rejection. The outer error path
                    # persists the now-complete Tool Call block and checkpoint.
                    raise RuntimeError(protocol_error)
                if (
                    context.tool_call_count + len(business_calls)
                    > self.config.limits.max_tool_calls
                ):
                    for call in business_calls:
                        rejected = ToolResult.failed(
                            call_id=call.id,
                            tool_name=call.name,
                            error=ToolError(
                                ToolErrorType.EXECUTION_ERROR,
                                "tool call limit exceeded",
                            ),
                        )
                        self._record_tool_trace(context, call, rejected)
                        context.add_message(
                            Message.tool(
                                self._json(
                                    {
                                        "success": False,
                                        "error": "tool call limit exceeded",
                                    }
                                ),
                                call_id=call.id,
                                name=call.name,
                            )
                        )
                    result = await self._finalize(
                        context,
                        response,
                        FinishReason.MAX_TOOL_CALLS,
                        deadline,
                        output=None,
                    )
                    event = AgentEvent(
                        EventType.RUN_COMPLETED, run_id, {"result": result}
                    )
                    await self._publish(event)
                    yield event
                    return

                context.status = RunStatus.WAITING_FOR_TOOLS
                context.tool_call_count += len(business_calls)
                for call in calls:
                    is_skill_resource = call.name in SKILL_RESOURCE_TOOL_NAMES
                    event_data: dict[str, Any] = {"tool_name": call.name}
                    if is_skill_resource:
                        event_data.update(
                            {
                                "skill_name": call.arguments.get("skill_name"),
                                "path": call.arguments.get("path"),
                                "resource_operation": (
                                    "read" if call.name.endswith("_read") else "search"
                                ),
                            }
                        )
                    else:
                        event_data["tool_call"] = call
                    tool_event = AgentEvent(
                        EventType.TOOL_STARTED,
                        run_id,
                        event_data,
                    )
                    await self._publish(tool_event)
                    yield tool_event

                tool_context = ToolExecutionContext(
                    run_id=run_id,
                    session_id=context.request.session_id,
                    user_context=dict(context.request.user_context),
                    metadata={
                        "agent": self.config.name,
                        "active_skills": [
                            activation.to_dict()
                            for activation in context.skill_state.active_skills
                        ],
                    },
                )
                try:
                    remaining = self._remaining(deadline)
                    capabilities = getattr(self.model, "capabilities", None)
                    parallel_tools = bool(
                        self.config.limits.parallel_tool_calls
                        and getattr(capabilities, "parallel_tool_calling", True)
                    )
                    results = await asyncio.wait_for(
                        self.tool_executor.execute_many(
                            calls,
                            tool_context,
                            parallel=parallel_tools,
                            max_parallel=self.config.limits.max_parallel_tools,
                        ),
                        timeout=remaining,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    error_message = (
                        "tool execution timed out"
                        if isinstance(exc, asyncio.TimeoutError)
                        else (str(exc) or exc.__class__.__name__)
                    )
                    for call in calls:
                        is_skill_resource = call.name in SKILL_RESOURCE_TOOL_NAMES
                        rejected = ToolResult.failed(
                            call_id=call.id,
                            tool_name=call.name,
                            error=ToolError(
                                (
                                    ToolErrorType.TIMEOUT
                                    if isinstance(exc, asyncio.TimeoutError)
                                    else ToolErrorType.EXECUTION_ERROR
                                ),
                                error_message,
                            ),
                        )
                        self._record_tool_trace(context, call, rejected)
                        context.add_message(
                            Message.tool(
                                self._json({"success": False, "error": error_message}),
                                call_id=call.id,
                                name=call.name,
                                metadata=(
                                    {"moduagent.ephemeral": True}
                                    if is_skill_resource
                                    else None
                                ),
                            ),
                            persist=not is_skill_resource,
                        )
                        event_data = {
                            "tool_name": call.name,
                            "success": False,
                            "error": error_message,
                        }
                        if is_skill_resource:
                            event_data.update(
                                {
                                    "skill_name": call.arguments.get("skill_name"),
                                    "path": call.arguments.get("path"),
                                }
                            )
                        else:
                            event_data["tool_call"] = call
                        tool_event = AgentEvent(
                            EventType.TOOL_COMPLETED,
                            run_id,
                            event_data,
                        )
                        await self._publish(tool_event)
                        yield tool_event
                    raise

                effective_results: list[ToolResult] = []
                for call, result in zip(calls, results):
                    is_skill_resource = call.name in SKILL_RESOURCE_TOOL_NAMES
                    if (
                        is_skill_resource
                        and result.success
                        and self.skill_runtime is not None
                    ):
                        added_tokens = self._resource_tokens(result.value)
                        next_resource_tokens = (
                            context.skill_state.resource_tokens + added_tokens
                        )
                        total_skill_tokens = (
                            context.skill_state.instruction_tokens
                            + next_resource_tokens
                        )
                        if (
                            next_resource_tokens
                            > self.skill_runtime.limits.max_resource_tokens
                            or total_skill_tokens
                            > self.skill_runtime.limits.max_total_skill_tokens
                        ):
                            result = ToolResult.failed(
                                call_id=call.id,
                                tool_name=call.name,
                                error=ToolError(
                                    ToolErrorType.RESULT_TOO_LARGE,
                                    "Skill resource token budget exceeded",
                                ),
                                attempts=result.attempts,
                                duration_seconds=result.duration_seconds,
                            )
                        else:
                            context.skill_state = replace(
                                context.skill_state,
                                resource_tokens=next_resource_tokens,
                            )
                    effective_results.append(result)
                    self._record_tool_trace(context, call, result)
                    context.add_message(
                        Message.tool(
                            self._tool_result_content(result),
                            call_id=call.id,
                            name=call.name,
                            metadata=(
                                {"moduagent.ephemeral": True}
                                if is_skill_resource
                                else None
                            ),
                        ),
                        persist=not is_skill_resource,
                    )
                    event_data = {
                        "tool_name": call.name,
                        "success": result.success,
                    }
                    if is_skill_resource:
                        event_data.update(
                            {
                                "skill_name": call.arguments.get("skill_name"),
                                "path": call.arguments.get("path"),
                            }
                        )
                    else:
                        event_data.update({"tool_call": call, "result": result})
                    tool_event = AgentEvent(
                        EventType.TOOL_COMPLETED,
                        run_id,
                        event_data,
                    )
                    await self._publish(tool_event)
                    yield tool_event
                    if is_skill_resource:
                        value = result.value if isinstance(result.value, dict) else {}
                        resource_event = AgentEvent(
                            EventType.SKILL_RESOURCE_READ,
                            run_id,
                            {
                                "skill_name": call.arguments.get("skill_name"),
                                "path": call.arguments.get("path"),
                                "operation": (
                                    "read" if call.name.endswith("_read") else "search"
                                ),
                                "success": result.success,
                                "digest": value.get("digest"),
                                "truncated": value.get("truncated"),
                                "returned_bytes": value.get("returned_bytes"),
                                "scanned_bytes": value.get("scanned_bytes"),
                            },
                        )
                        await self._publish(resource_event)
                        yield resource_event
                await self._within(
                    deadline,
                    lambda: self.decision_policy.observe(
                        context, tuple(effective_results)
                    ),
                )
                context.status = RunStatus.RUNNING
                await self._save_checkpoint(context, deadline)

            result = await self._finalize(
                context,
                last_response,
                FinishReason.MAX_STEPS,
                deadline,
                output=None,
            )
            event = AgentEvent(EventType.RUN_COMPLETED, run_id, {"result": result})
            await self._publish(event)
            yield event
        except asyncio.CancelledError:
            context.status = RunStatus.CANCELLED
            await self._save_checkpoint_safely(context)
            raise
        except asyncio.TimeoutError:
            context.status = RunStatus.FAILED
            await self._persist_safely(context)
            await self._save_checkpoint_safely(context)
            result = self._result(
                context,
                FinishReason.TIMEOUT,
                error="run timed out",
            )
            event = AgentEvent(EventType.RUN_FAILED, run_id, {"result": result})
            await self._publish(event)
            yield event
        except Exception as exc:
            context.status = RunStatus.FAILED
            await self._persist_safely(context)
            await self._save_checkpoint_safely(context)
            error_message = str(exc) or exc.__class__.__name__
            if (
                strict_execution
                and isinstance(context.execution_state, ExecutionState)
                and context.execution_state.phase is RunPhase.FAILED
            ):
                error_message = self._strict_failure_message(
                    context.execution_state,
                    error_message,
                )
            result = self._result(
                context,
                FinishReason.ERROR,
                error=error_message,
            )
            event = AgentEvent(EventType.RUN_FAILED, run_id, {"result": result})
            await self._publish(event)
            yield event

    async def _strict_plan_events(
        self,
        context: RunContext,
        available_tool_schemas: tuple[Any, ...],
        deadline: float,
        *,
        stream_model: bool,
    ) -> AsyncIterator[AgentEvent]:
        policy = self.decision_policy
        state = context.execution_state
        if not isinstance(state, ExecutionState):
            raise RuntimeError("strict Plan-and-Execute state is missing")

        if state.phase is RunPhase.DONE:
            if state.final_response is None or not state.final_emitted:
                raise RuntimeError(
                    "completed execution state has no emitted final output"
                )
            try:
                output = self.output_codec.decode(state.final_response)
            except Exception as exc:
                raise RuntimeError(
                    "stored finalization response validation failed"
                ) from exc
            context.status = RunStatus.COMPLETED
            result = self._result(
                context,
                FinishReason.COMPLETED,
                output=output,
            )
            completed = AgentEvent(
                EventType.RUN_COMPLETED,
                context.run_id,
                {"result": result, "resumed_terminal": True},
            )
            await self._publish(completed)
            yield completed
            return

        while True:
            state = context.execution_state
            if not isinstance(state, ExecutionState):
                raise RuntimeError("strict Plan-and-Execute state was lost")
            if len(state.plan.steps) > self.config.limits.max_steps:
                state.fail_current_step()
                state.validation_error = (
                    f"plan exceeds RunLimits.max_steps ({self.config.limits.max_steps})"
                )
                self._sync_execution_state(context, state)
                await self._persist_pending_messages(context, deadline)
                await self._save_checkpoint(context, deadline)
                result = self._result(
                    context,
                    FinishReason.MAX_STEPS,
                    error=state.validation_error,
                )
                completed = AgentEvent(
                    EventType.RUN_COMPLETED,
                    context.run_id,
                    {"result": result},
                )
                await self._publish(completed)
                yield completed
                return
            if state.phase is RunPhase.FAILED:
                raise RuntimeError(self._strict_failure_message(state))
            if state.phase is RunPhase.DONE:
                raise RuntimeError("DONE cannot transition to another execution phase")

            if state.phase is RunPhase.STEP_PREPARE:
                step = state.plan.current
                if step is None:
                    state.phase = RunPhase.VERIFY
                    self._sync_execution_state(context, state)
                    await self._save_checkpoint(context, deadline)
                    continue
                step_tools = self._strict_step_tool_schemas(
                    state,
                    available_tool_schemas,
                )
                prepare_step = getattr(policy, "prepare_step", None)
                if not callable(prepare_step):
                    raise RuntimeError(
                        "strict Plan-and-Execute policy must provide prepare_step()"
                    )
                prepared = prepare_step(context, has_tools=bool(step_tools))
                if prepared is None:
                    continue
                context.clear_internal_messages()
                started = AgentEvent(
                    EventType.STEP_STARTED,
                    context.run_id,
                    {
                        "step_id": prepared.step_id,
                        "objective": prepared.objective,
                        "attempt": prepared.attempt_count,
                        "plan_version": state.plan.version,
                    },
                    visibility=EventVisibility.INTERNAL,
                )
                await self._publish(started)
                yield started
                await self._save_checkpoint(context, deadline)
                continue

            if state.phase is RunPhase.VERIFY:
                if not state.plan.complete:
                    raise RuntimeError("plan verification found incomplete steps")
                missing_results = {
                    step.step_id
                    for step in state.plan.steps
                    if step.status.value == "completed"
                    and step.step_id not in state.committed_results
                }
                if missing_results:
                    raise RuntimeError(
                        "plan verification found completed steps without results: "
                        + ", ".join(sorted(missing_results))
                    )
                begin_finalization = getattr(policy, "begin_finalization", None)
                if not callable(begin_finalization):
                    raise RuntimeError(
                        "strict Plan-and-Execute policy must provide "
                        "begin_finalization()"
                    )
                begin_finalization(context)
                await self._save_checkpoint(context, deadline)
                continue

            if state.phase is RunPhase.FINALIZE:
                async for final_event in self._strict_finalization_events(
                    context,
                    deadline,
                    stream_model=stream_model,
                ):
                    yield final_event
                return

            step_tools: tuple[Any, ...] = ()
            if state.phase is RunPhase.STEP_VALIDATE:
                step = state.current_step
                if step is None:
                    raise RuntimeError("STEP_VALIDATE phase has no current plan step")
                validate_pending = getattr(policy, "validate_pending", None)
                if not callable(validate_pending):
                    raise RuntimeError(
                        "strict Plan-and-Execute policy must provide validate_pending()"
                    )
                decision = await self._within(
                    deadline,
                    lambda: validate_pending(context),
                )
            else:
                if state.phase is not RunPhase.ACT:
                    raise RuntimeError(
                        f"unsupported strict execution phase: {state.phase.value}"
                    )

                step = state.current_step
                if step is None:
                    raise RuntimeError("ACT phase has no current plan step")
                extracting = bool(
                    getattr(policy, "needs_step_result_extraction")(context)
                )
                step_tools = self._strict_step_tool_schemas(
                    state,
                    available_tool_schemas,
                )
                request_tools = () if extracting else step_tools
                output_schema = (
                    getattr(policy, "step_result_schema")() if extracting else None
                )
                options = dict(self.config.model_options)
                if not request_tools:
                    options.pop("tool_choice", None)
                    options.pop("parallel_tool_calls", None)
                request_model = ModelRequest(
                    messages=self._strict_step_messages(
                        context,
                        state,
                        extracting=extracting,
                    ),
                    tools=request_tools,
                    output_schema=output_schema,
                    options=options,
                )
                request_model, memory_event = await self._prepare_model_request(
                    context,
                    request_model,
                    phase=(MemoryPhase.STEP_RESULT if extracting else MemoryPhase.ACT),
                    deadline=deadline,
                    skill_phase="act",
                    protected_from=0,
                )
                if memory_event is not None:
                    yield memory_event

                context.step += 1
                context.status = RunStatus.WAITING_FOR_MODEL
                await self._save_checkpoint(context, deadline)
                response: ModelResponse | None = None
                async for model_event in self._model_events(
                    context,
                    request_model,
                    deadline,
                    stream_model=stream_model,
                    phase="step_result" if extracting else "act",
                    record_response=False,
                    record_internal=True,
                    visibility=EventVisibility.INTERNAL,
                    delta_event_type=EventType.STEP_MODEL_DELTA,
                ):
                    if model_event.type is EventType.MODEL_COMPLETED:
                        candidate = model_event.data.get("response")
                        if isinstance(candidate, ModelResponse):
                            response = candidate
                    yield model_event
                if response is None:
                    raise RuntimeError(
                        "strict model invocation ended without a response"
                    )

                decision = await self._within(
                    deadline,
                    lambda: policy.decide(context, response),
                )
            decision_event = AgentEvent(
                EventType.POLICY_DECISION,
                context.run_id,
                {
                    "kind": decision.kind.value,
                    "metadata": dict(decision.metadata),
                },
                visibility=EventVisibility.INTERNAL,
            )
            await self._publish(decision_event)
            yield decision_event

            if decision.kind is DecisionKind.CALL_TOOLS:
                result_box: list[ToolResult] = []
                try:
                    async for tool_event in self._strict_tool_events(
                        context,
                        tuple(decision.tool_calls),
                        allowed_schemas=step_tools,
                        deadline=deadline,
                        result_box=result_box,
                    ):
                        yield tool_event
                except _StrictToolCallLimitError:
                    state.fail_current_step()
                    state.validation_error = "tool call limit exceeded"
                    self._sync_execution_state(context, state)
                    await self._persist_pending_messages(context, deadline)
                    await self._save_checkpoint(context, deadline)
                    result = self._result(
                        context,
                        FinishReason.MAX_TOOL_CALLS,
                        error="tool call limit exceeded",
                    )
                    completed = AgentEvent(
                        EventType.RUN_COMPLETED,
                        context.run_id,
                        {"result": result},
                    )
                    await self._publish(completed)
                    yield completed
                    return
                before_replans = state.replan_count
                was_repairing = isinstance(
                    getattr(state, "pending_tool_failure", None),
                    Mapping,
                )
                tool_decision = await self._within(
                    deadline,
                    lambda: policy.observe(context, tuple(result_box)),
                )
                state = context.execution_state
                if not isinstance(state, ExecutionState):
                    raise RuntimeError("strict execution state was lost after tools")
                if tool_decision is not None:
                    observed = AgentEvent(
                        EventType.POLICY_DECISION,
                        context.run_id,
                        {
                            "kind": tool_decision.kind.value,
                            "metadata": dict(tool_decision.metadata),
                            "source": "tool_results",
                        },
                        visibility=EventVisibility.INTERNAL,
                    )
                    await self._publish(observed)
                    yield observed

                if (
                    tool_decision is not None
                    and tool_decision.kind is DecisionKind.RETRY_TOOL
                ):
                    failure = tool_decision.metadata.get("tool_failure", {})
                    scheduled = AgentEvent(
                        EventType.TOOL_REPAIR_SCHEDULED,
                        context.run_id,
                        {
                            "step_id": step.step_id,
                            "tool_name": failure.get("tool_name"),
                            "failed_call_id": failure.get("call_id"),
                            "error_type": failure.get("error_type"),
                            "reason": failure.get("reason"),
                            "repair_attempt": tool_decision.metadata.get(
                                "repair_attempt"
                            ),
                            "max_attempts": getattr(
                                policy,
                                "max_tool_repair_attempts",
                                None,
                            ),
                        },
                        visibility=EventVisibility.INTERNAL,
                    )
                    await self._publish(scheduled)
                    yield scheduled
                    await self._save_checkpoint(context, deadline)
                    continue

                repair_requested = any(
                    str(
                        getattr(
                            getattr(result.error, "recovery", None),
                            "value",
                            getattr(result.error, "recovery", ""),
                        )
                    )
                    == "repair_call"
                    for result in result_box
                    if result.error is not None
                )
                if (
                    tool_decision is not None
                    and tool_decision.kind
                    in {
                        DecisionKind.REPLAN,
                        DecisionKind.FAIL,
                    }
                    and (was_repairing or repair_requested)
                ):
                    exhausted = AgentEvent(
                        EventType.TOOL_REPAIR_EXHAUSTED,
                        context.run_id,
                        {
                            "step_id": step.step_id,
                            "reason": tool_decision.metadata.get("reason"),
                            "fallback": tool_decision.kind.value,
                            "repair_attempts": getattr(
                                state,
                                "tool_repair_counts",
                                {},
                            ).get(step.step_id, 0),
                        },
                        visibility=EventVisibility.INTERNAL,
                    )
                    await self._publish(exhausted)
                    yield exhausted

                if (
                    tool_decision is not None
                    and tool_decision.kind is DecisionKind.FAIL
                ):
                    failed = AgentEvent(
                        EventType.STEP_FAILED,
                        context.run_id,
                        {
                            "step_id": step.step_id,
                            "reason": tool_decision.metadata.get("reason"),
                        },
                        visibility=EventVisibility.INTERNAL,
                    )
                    await self._publish(failed)
                    yield failed
                    await self._save_checkpoint(context, deadline)
                    raise RuntimeError(
                        self._strict_failure_message(
                            state,
                            tool_decision.error_message,
                        )
                    )

                if (
                    tool_decision is not None
                    and tool_decision.kind is DecisionKind.REPLAN
                ) or state.replan_count > before_replans:
                    revised = AgentEvent(
                        EventType.PLAN_REVISED,
                        context.run_id,
                        {
                            "plan_version": state.plan.version,
                            "replan_count": state.replan_count,
                            "reason": state.validation_error,
                        },
                        visibility=EventVisibility.INTERNAL,
                    )
                    await self._publish(revised)
                    yield revised
                    context.clear_internal_messages()
                if state.phase is RunPhase.FAILED:
                    failed = AgentEvent(
                        EventType.STEP_FAILED,
                        context.run_id,
                        {
                            "step_id": step.step_id,
                            "reason": state.validation_error,
                        },
                        visibility=EventVisibility.INTERNAL,
                    )
                    await self._publish(failed)
                    yield failed
                    await self._save_checkpoint(context, deadline)
                    raise RuntimeError(
                        self._strict_failure_message(
                            state,
                            "Tool execution failed",
                        )
                    )
                await self._save_checkpoint(context, deadline)
                continue

            if decision.kind is DecisionKind.RETRY_STEP:
                retry_event = AgentEvent(
                    EventType.STEP_RETRY,
                    context.run_id,
                    {
                        "step_id": step.step_id,
                        "attempt": step.attempt_count,
                        "reason": decision.metadata.get("reason"),
                        "count_attempt": decision.metadata.get("count_attempt", True),
                    },
                    visibility=EventVisibility.INTERNAL,
                )
                await self._publish(retry_event)
                yield retry_event
                await self._save_checkpoint(context, deadline)
                continue

            if decision.kind is DecisionKind.REPLAN:
                revised = AgentEvent(
                    EventType.PLAN_REVISED,
                    context.run_id,
                    {
                        "plan_version": state.plan.version,
                        "replan_count": state.replan_count,
                        "reason": decision.metadata.get("reason"),
                    },
                    visibility=EventVisibility.INTERNAL,
                )
                await self._publish(revised)
                yield revised
                context.clear_internal_messages()
                await self._save_checkpoint(context, deadline)
                continue

            if decision.kind is DecisionKind.COMMIT_STEP:
                state = context.execution_state
                if not isinstance(state, ExecutionState):
                    raise RuntimeError("strict execution state was lost at commit")
                result = state.committed_results.get(step.step_id)
                if result is None:
                    raise RuntimeError("committed step has no StepResult")
                result_created = AgentEvent(
                    EventType.STEP_RESULT_CREATED,
                    context.run_id,
                    {
                        "step_id": step.step_id,
                        "status": result.status,
                    },
                    visibility=EventVisibility.INTERNAL,
                )
                validated = AgentEvent(
                    EventType.STEP_VALIDATED,
                    context.run_id,
                    {
                        "step_id": step.step_id,
                        "decision": "commit",
                    },
                    visibility=EventVisibility.INTERNAL,
                )
                committed = AgentEvent(
                    EventType.STEP_COMMITTED,
                    context.run_id,
                    {
                        "step_id": step.step_id,
                        "result_ref": step.result_ref,
                        "plan_version": state.plan.version,
                    },
                    visibility=EventVisibility.INTERNAL,
                )
                for internal_event in (result_created, validated, committed):
                    await self._publish(internal_event)
                    yield internal_event
                context.clear_internal_messages()
                await self._save_checkpoint(context, deadline)
                continue

            if decision.kind is DecisionKind.FINALIZE:
                getattr(policy, "begin_finalization")(context)
                await self._save_checkpoint(context, deadline)
                continue

            if decision.kind is DecisionKind.FAIL:
                failure_message = self._strict_failure_message(
                    state,
                    decision.error_message,
                )
                failure = self._normalized_execution_failure(
                    getattr(state, "failure", None)
                )
                if (
                    failure is not None
                    and failure.get("recovery") == "repair_call"
                    and getattr(state, "total_tool_repairs", 0) > 0
                ):
                    exhausted = AgentEvent(
                        EventType.TOOL_REPAIR_EXHAUSTED,
                        context.run_id,
                        {
                            "step_id": step.step_id,
                            "reason": failure.get("terminal_reason"),
                            "fallback": "fail",
                            "repair_attempts": getattr(
                                state,
                                "tool_repair_counts",
                                {},
                            ).get(step.step_id, 0),
                        },
                        visibility=EventVisibility.INTERNAL,
                    )
                    await self._publish(exhausted)
                    yield exhausted
                failed = AgentEvent(
                    EventType.STEP_FAILED,
                    context.run_id,
                    {
                        "step_id": step.step_id,
                        "reason": failure_message,
                    },
                    visibility=EventVisibility.INTERNAL,
                )
                await self._publish(failed)
                yield failed
                await self._save_checkpoint(context, deadline)
                raise RuntimeError(failure_message)
            raise RuntimeError(f"unsupported strict decision: {decision.kind.value}")

    @classmethod
    def _strict_failure_message(
        cls,
        state: ExecutionState,
        fallback: str | None = None,
    ) -> str:
        failure = cls._normalized_execution_failure(getattr(state, "failure", None))
        if failure is not None:
            feedback = failure.get("feedback") or failure.get("message")
            terminal_reason = failure.get("terminal_reason")
            if feedback and terminal_reason and terminal_reason not in str(feedback):
                return f"{feedback}; {terminal_reason}"
            if feedback:
                return str(feedback)
            if terminal_reason:
                return str(terminal_reason)
        # validation_error may contain provider, validator, or backend details.
        # Only an explicit public fallback or the sanitized failure contract may
        # cross into AgentResult.error.
        return str(fallback or "strict Plan-and-Execute execution failed")

    @classmethod
    def _normalized_execution_failure(
        cls,
        raw: Any,
    ) -> dict[str, Any] | None:
        if not isinstance(raw, Mapping):
            return None
        normalized: dict[str, Any] = {}
        short_text_fields = {
            "step_id",
            "call_id",
            "tool_name",
            "error_type",
            "reason",
            "recovery",
        }
        message_fields = {
            "feedback",
            "fallback_reason",
            "terminal_reason",
            "message",
        }
        for key in short_text_fields:
            value = raw.get(key)
            if value is not None:
                normalized[key] = cls._tool_trace_text(value)
        for key in message_fields:
            value = raw.get(key)
            if value is not None:
                text = str(value)
                normalized[key] = text if len(text) <= 512 else f"{text[:509]}..."
        for key in ("retryable", "repair_safe"):
            if key in raw:
                normalized[key] = bool(raw[key])
        for key in (
            "failure_count",
            "success_count",
            "result_count",
            "repair_attempts",
        ):
            if key in raw:
                try:
                    count = int(raw[key])
                except (TypeError, ValueError, OverflowError):
                    count = 0
                normalized[key] = min(max(count, 0), 1_000_000)
        return normalized or None

    def _strict_step_messages(
        self,
        context: RunContext,
        state: ExecutionState,
        *,
        extracting: bool,
    ) -> tuple[Message, ...]:
        step = state.current_step
        if step is None:
            raise RuntimeError("strict step context has no current step")
        dependencies = {
            dependency: state.committed_results[dependency].model_dump(mode="json")
            for dependency in step.dependencies
            if dependency in state.committed_results
        }
        payload = {
            "current_step": step.to_dict(),
            "dependency_results": dependencies,
        }
        messages: list[Message] = [
            Message.system(self.config.instructions),
            Message.system(_STRICT_EXECUTOR_PROMPT),
            Message.user(self._json(payload)),
            *self._strict_internal_messages(context, state),
        ]
        if extracting:
            feedback = (
                ""
                if not state.validation_error
                else f"\nValidation feedback: {state.validation_error}"
            )
            messages.append(Message.user(f"{_STRICT_STEP_RESULT_PROMPT}{feedback}"))
        elif isinstance(
            getattr(state, "pending_tool_failure", None),
            Mapping,
        ):
            failure = (
                self._normalized_execution_failure(state.pending_tool_failure) or {}
            )
            repair_context = {
                key: failure.get(key)
                for key in (
                    "tool_name",
                    "error_type",
                    "reason",
                    "feedback",
                )
                if failure.get(key) is not None
            }
            messages.append(
                Message.user(
                    "The previous allowed Tool call failed. Call exactly the same "
                    "Tool once with corrected arguments and a new call ID. Do not "
                    "repeat identical arguments, select another Tool, add another "
                    "Tool call, return a StepResult, or claim completion until the "
                    "Tool succeeds.\n" + self._json(repair_context)
                )
            )
        else:
            messages.append(
                Message.user(
                    "Use an allowed tool when it is needed for this step. "
                    "Stop after collecting enough evidence; the runtime will "
                    "request the strict StepResult separately."
                )
            )
        return tuple(messages)

    def _strict_internal_messages(
        self,
        context: RunContext,
        state: ExecutionState,
    ) -> tuple[Message, ...]:
        """Project legacy checkpoints without replaying raw failure details."""

        pending = self._normalized_execution_failure(
            getattr(state, "pending_tool_failure", None)
        )
        if pending is None or not pending.get("call_id"):
            return tuple(context.internal_messages)
        failed_call_id = str(pending["call_id"])
        projected: list[Message] = []
        for message in context.internal_messages:
            if (
                message.role is MessageRole.TOOL
                and message.tool_call_id == failed_call_id
            ):
                projected.append(
                    Message.tool(
                        self._strict_failure_protocol_content(pending),
                        call_id=failed_call_id,
                        name=message.name or str(pending.get("tool_name", "")),
                        metadata=message.metadata,
                    )
                )
            else:
                projected.append(message)
        return tuple(projected)

    @classmethod
    def _strict_failure_protocol_content(
        cls,
        failure: Mapping[str, Any],
    ) -> str:
        error: dict[str, Any] = {
            "type": cls._tool_trace_text(
                failure.get("error_type", ToolErrorType.EXECUTION_ERROR.value)
            ),
            "retryable": bool(failure.get("retryable", False)),
        }
        if failure.get("reason"):
            error["reason"] = cls._tool_trace_text(failure["reason"])
        if failure.get("recovery"):
            error["recovery"] = cls._tool_trace_text(failure["recovery"])
        return cls._json({"success": False, "error": error})

    @classmethod
    def _strict_tool_result_content(cls, result: ToolResult) -> str:
        if result.success:
            for method_name in ("model_content", "to_message_content"):
                method = getattr(result, method_name, None)
                if callable(method):
                    return str(method())
            return cls._json({"success": True, "value": result.value})
        error = result.error
        failure: dict[str, Any] = {
            "error_type": (
                ToolErrorType.EXECUTION_ERROR.value
                if error is None
                else error.type.value
            ),
            "retryable": bool(error.retryable) if error is not None else False,
        }
        if error is not None and error.reason:
            failure["reason"] = error.reason
        recovery = None if error is None else error.recovery
        if recovery is not None:
            failure["recovery"] = getattr(recovery, "value", recovery)
        return cls._strict_failure_protocol_content(failure)

    @staticmethod
    def _strict_step_tool_schemas(
        state: ExecutionState,
        available: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        step = state.current_step or state.plan.current
        if step is None or not step.allowed_tools:
            return ()
        by_name = {schema.name: schema for schema in available}
        unknown = set(step.allowed_tools) - set(by_name)
        if unknown:
            raise RuntimeError(
                "plan step requests unavailable tools: " + ", ".join(sorted(unknown))
            )
        allowed = set(step.allowed_tools)
        return tuple(schema for schema in available if schema.name in allowed)

    async def _strict_tool_events(
        self,
        context: RunContext,
        calls: tuple[Any, ...],
        *,
        allowed_schemas: tuple[Any, ...],
        deadline: float,
        result_box: list[ToolResult],
    ) -> AsyncIterator[AgentEvent]:
        allowed_names = {schema.name for schema in allowed_schemas}
        disallowed = sorted({call.name for call in calls} - allowed_names)
        if disallowed:
            message = "step attempted unavailable tools: " + ", ".join(disallowed)
            self._record_internal_rejected_calls(context, calls, message)
            raise RuntimeError("step attempted a disallowed tool")

        skill_resource_calls = tuple(
            call for call in calls if call.name in SKILL_RESOURCE_TOOL_NAMES
        )
        business_calls = tuple(
            call for call in calls if call.name not in SKILL_RESOURCE_TOOL_NAMES
        )
        if skill_resource_calls and business_calls:
            message = "a model response cannot mix Skill resource and business tools"
            self._record_internal_rejected_calls(context, calls, message)
            raise RuntimeError(message)
        if skill_resource_calls:
            if self.skill_runtime is None:
                message = "Skill resource tools are not configured"
                self._record_internal_rejected_calls(context, calls, message)
                raise RuntimeError(message)
            next_reads = context.skill_state.resource_reads + len(skill_resource_calls)
            if next_reads > self.skill_runtime.limits.max_resource_reads:
                message = "Skill resource read limit exceeded"
                self._record_internal_rejected_calls(context, calls, message)
                raise RuntimeError(message)
            context.skill_state = replace(
                context.skill_state,
                resource_reads=next_reads,
            )

        if (
            context.tool_call_count + len(business_calls)
            > self.config.limits.max_tool_calls
        ):
            self._record_internal_rejected_calls(
                context,
                calls,
                "tool call limit exceeded",
            )
            raise _StrictToolCallLimitError("tool call limit exceeded")

        context.status = RunStatus.WAITING_FOR_TOOLS
        context.tool_call_count += len(business_calls)
        for call in calls:
            started = AgentEvent(
                EventType.TOOL_STARTED,
                context.run_id,
                {
                    "tool_name": call.name,
                    "tool_call": call,
                    "step_id": (
                        context.execution_state.current_step_id
                        if isinstance(context.execution_state, ExecutionState)
                        else None
                    ),
                },
                visibility=EventVisibility.INTERNAL,
            )
            await self._publish(started)
            yield started

        tool_metadata: dict[str, Any] = {
            "agent": self.config.name,
            "active_skills": [
                activation.to_dict() for activation in context.skill_state.active_skills
            ],
        }
        state = context.execution_state
        pending_failure = (
            getattr(state, "pending_tool_failure", None)
            if isinstance(state, ExecutionState)
            else None
        )
        if (
            isinstance(pending_failure, Mapping)
            and pending_failure.get("tool_name")
            and pending_failure.get("invocation_fingerprint")
        ):
            tool_metadata[_TOOL_REPAIR_METADATA_KEY] = {
                "tool_name": str(pending_failure["tool_name"]),
                "invocation_fingerprint": str(
                    pending_failure["invocation_fingerprint"]
                ),
            }
        tool_context = ToolExecutionContext(
            run_id=context.run_id,
            session_id=context.request.session_id,
            user_context=dict(context.request.user_context),
            metadata=tool_metadata,
        )
        try:
            capabilities = getattr(self.model, "capabilities", None)
            parallel = bool(
                self.config.limits.parallel_tool_calls
                and getattr(capabilities, "parallel_tool_calling", True)
            )
            returned_results = await asyncio.wait_for(
                self.tool_executor.execute_many(
                    calls,
                    tool_context,
                    parallel=parallel,
                    max_parallel=self.config.limits.max_parallel_tools,
                ),
                timeout=self._remaining(deadline),
            )
            # Strict execution accepts only the bounded collection contract used
            # by ToolExecutor. Materializing an arbitrary iterable here could
            # block the event loop forever after the run timeout has expired.
            if type(returned_results) not in (list, tuple):
                raise _StrictToolResultProtocolError
            if len(returned_results) != len(calls):
                raise _StrictToolResultProtocolError
            raw_results = tuple(returned_results)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            self._fail_strict_tool_calls(
                context,
                calls,
                "tool execution timed out",
            )
            raise
        except _StrictToolResultProtocolError as exc:
            error_message = "tool executor returned mismatched results"
            self._fail_strict_tool_calls(
                context,
                calls,
                error_message,
            )
            raise RuntimeError(error_message) from exc
        except Exception as exc:
            self._fail_strict_tool_calls(
                context,
                calls,
                "tool execution failed",
            )
            raise RuntimeError("tool execution failed") from exc

        if any(
            not isinstance(result, ToolResult)
            or type(result.call_id) is not str
            or type(result.tool_name) is not str
            or result.call_id != call.id
            or result.tool_name != call.name
            or type(result.success) is not bool
            or (result.success and result.error is not None)
            or (not result.success and not isinstance(result.error, ToolError))
            or type(result.attempts) is not int
            or result.attempts < 0
            or type(result.duration_seconds) not in (int, float)
            or not math.isfinite(result.duration_seconds)
            or result.duration_seconds < 0
            or type(result.repair_safe) is not bool
            or (
                result.invocation_arguments is not None
                and type(result.invocation_arguments) is not dict
            )
            or (
                result.error is not None
                and (
                    type(result.error.type) is not ToolErrorType
                    or type(result.error.message) is not str
                    or type(result.error.retryable) is not bool
                    or type(result.error.details) is not dict
                    or (
                        result.error.reason is not None
                        and type(result.error.reason) is not str
                    )
                    or (
                        result.error.recovery is not None
                        and type(result.error.recovery) is not ToolRecoveryAction
                    )
                )
            )
            for call, result in zip(calls, raw_results)
        ):
            error_message = "tool executor returned mismatched results"
            self._fail_strict_tool_calls(
                context,
                calls,
                error_message,
            )
            raise RuntimeError(error_message)

        effective: list[ToolResult] = []
        for call, raw_result in zip(calls, raw_results):
            try:
                result = raw_result
                is_skill_resource = call.name in SKILL_RESOURCE_TOOL_NAMES
                if (
                    is_skill_resource
                    and result.success
                    and self.skill_runtime is not None
                ):
                    added_tokens = self._resource_tokens(result.value)
                    next_resource_tokens = (
                        context.skill_state.resource_tokens + added_tokens
                    )
                    total_skill_tokens = (
                        context.skill_state.instruction_tokens + next_resource_tokens
                    )
                    if (
                        next_resource_tokens
                        > self.skill_runtime.limits.max_resource_tokens
                        or total_skill_tokens
                        > self.skill_runtime.limits.max_total_skill_tokens
                    ):
                        result = ToolResult.failed(
                            call_id=call.id,
                            tool_name=call.name,
                            error=ToolError(
                                ToolErrorType.RESULT_TOO_LARGE,
                                "Skill resource token budget exceeded",
                            ),
                            attempts=result.attempts,
                            duration_seconds=result.duration_seconds,
                        )
                    else:
                        context.skill_state = replace(
                            context.skill_state,
                            resource_tokens=next_resource_tokens,
                        )
                content = self._strict_tool_result_content(result)
                self._record_tool_trace(context, call, result)
                context.add_internal_message(
                    Message.tool(
                        content,
                        call_id=call.id,
                        name=call.name,
                        metadata={"moduagent.ephemeral": True},
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error_message = "tool executor returned mismatched results"
                self._fail_strict_tool_calls(
                    context,
                    calls,
                    error_message,
                )
                raise RuntimeError(error_message) from exc
            effective.append(result)
            completed = AgentEvent(
                EventType.TOOL_COMPLETED,
                context.run_id,
                {
                    "tool_name": call.name,
                    "success": result.success,
                    "tool_call": call,
                    "result": result,
                },
                visibility=EventVisibility.INTERNAL,
            )
            await self._publish(completed)
            yield completed
            if is_skill_resource:
                value = result.value if isinstance(result.value, dict) else {}
                resource_event = AgentEvent(
                    EventType.SKILL_RESOURCE_READ,
                    context.run_id,
                    {
                        "skill_name": call.arguments.get("skill_name"),
                        "path": call.arguments.get("path"),
                        "operation": (
                            "read" if call.name.endswith("_read") else "search"
                        ),
                        "success": result.success,
                        "digest": value.get("digest"),
                        "truncated": value.get("truncated"),
                        "returned_bytes": value.get("returned_bytes"),
                        "scanned_bytes": value.get("scanned_bytes"),
                    },
                    visibility=EventVisibility.INTERNAL,
                )
                await self._publish(resource_event)
                yield resource_event
        result_box.extend(effective)
        context.status = RunStatus.RUNNING

    async def _strict_finalization_events(
        self,
        context: RunContext,
        deadline: float,
        *,
        stream_model: bool,
    ) -> AsyncIterator[AgentEvent]:
        policy = self.decision_policy
        state = context.execution_state
        if not isinstance(state, ExecutionState):
            raise RuntimeError("strict finalization state is missing")
        buffered_deltas: tuple[str, ...] = ()

        if state.final_response is None:
            started = AgentEvent(
                EventType.FINALIZATION_STARTED,
                context.run_id,
                {"count": state.finalization_count},
                visibility=EventVisibility.INTERNAL,
            )
            await self._publish(started)
            yield started
            context.status = RunStatus.WAITING_FOR_MODEL
            await self._save_checkpoint(context, deadline)
            options = dict(self.config.model_options)
            options.pop("tool_choice", None)
            options.pop("parallel_tool_calls", None)
            payload = getattr(policy, "finalization_payload")(context)
            request = ModelRequest(
                messages=(
                    Message.system(self.config.instructions),
                    Message.system(_STRICT_FINALIZER_PROMPT),
                    Message.user(self._json(payload)),
                ),
                tools=(),
                output_schema=self.output_codec.schema(),
                options=options,
            )
            request, memory_event = await self._prepare_model_request(
                context,
                request,
                phase=MemoryPhase.FINALIZE,
                deadline=deadline,
                skill_phase="finalize",
                protected_from=0,
            )
            if memory_event is not None:
                yield memory_event
            response: ModelResponse | None = None
            async for model_event in self._model_events(
                context,
                request,
                deadline,
                stream_model=stream_model,
                phase="finalize",
                record_response=False,
                visibility=EventVisibility.INTERNAL,
                buffer_deltas=True,
            ):
                if model_event.type is EventType.MODEL_COMPLETED:
                    candidate = model_event.data.get("response")
                    if isinstance(candidate, ModelResponse):
                        response = candidate
                    buffered_deltas = tuple(
                        str(item)
                        for item in model_event.data.get("buffered_deltas", ())
                    )
                yield model_event
            if response is None:
                raise RuntimeError("finalization returned no response")
            if response.tool_calls or response.message.tool_calls:
                raise RuntimeError("finalization returned tool calls")
            finish_reason = (response.finish_reason or "").lower()
            if finish_reason in {"timeout", "length", "max_tokens"}:
                raise RuntimeError(
                    f"incomplete finalization response ({finish_reason})"
                )
            raw_response = response.message.content
            if raw_response is None or not str(raw_response).strip():
                raise RuntimeError("finalization response is empty")
            # Decode before any public delta or persistence. An invalid public
            # schema therefore cannot leak a partial final response.
            try:
                output = self.output_codec.decode(response)
            except Exception as exc:
                raise RuntimeError("finalization response validation failed") from exc
            getattr(policy, "record_final_response")(
                context,
                str(raw_response),
            )
            state = context.execution_state
            await self._save_checkpoint(context, deadline)
        else:
            try:
                output = self.output_codec.decode(state.final_response)
            except Exception as exc:
                raise RuntimeError(
                    "stored finalization response validation failed"
                ) from exc

        if not isinstance(state, ExecutionState) or state.final_response is None:
            raise RuntimeError("finalization did not record a stable response")
        final_message_exists = any(
            message.role.value == "assistant"
            and message.metadata.get(_RUN_ID_METADATA_KEY) == context.run_id
            and message.metadata.get(_PUBLIC_FINAL_METADATA_KEY) is True
            for message in context.messages
        )
        if not final_message_exists:
            context.add_message(
                Message.assistant(
                    state.final_response,
                    metadata={
                        _RUN_ID_METADATA_KEY: context.run_id,
                        _PUBLIC_FINAL_METADATA_KEY: True,
                    },
                )
            )

        if not state.final_persisted:
            await self._persist_pending_messages(context, deadline)
            getattr(policy, "record_final_response")(
                context,
                state.final_response,
                persisted=True,
            )
            state = context.execution_state
            await self._save_checkpoint(context, deadline)

        if not isinstance(state, ExecutionState):
            raise RuntimeError("strict finalization state was lost")
        if not state.final_emitted:
            # Persist the at-most-once emission marker before exposing buffered
            # tokens. Durable exactly-once delivery still requires an external
            # outbox, but resume will never re-run the model or re-emit here.
            getattr(policy, "record_final_response")(
                context,
                state.final_response,
                persisted=True,
                emitted=True,
            )
            state = context.execution_state
            context.status = RunStatus.COMPLETED
            await self._save_checkpoint(context, deadline)
            if stream_model:
                public_deltas = (
                    buffered_deltas
                    if buffered_deltas
                    and "".join(buffered_deltas) == state.final_response
                    else (state.final_response,)
                )
                for delta in public_deltas:
                    delta_event = AgentEvent(
                        EventType.FINAL_DELTA,
                        context.run_id,
                        {
                            "phase": "finalize",
                            "delta": delta,
                        },
                        visibility=EventVisibility.PUBLIC,
                    )
                    await self._publish(delta_event)
                    yield delta_event
            finalized = AgentEvent(
                EventType.FINALIZATION_COMPLETED,
                context.run_id,
                {
                    "count": state.finalization_count,
                    "persisted": state.final_persisted,
                },
            )
            await self._publish(finalized)
            yield finalized

        result = self._result(
            context,
            FinishReason.COMPLETED,
            output=output,
        )
        completed = AgentEvent(
            EventType.RUN_COMPLETED,
            context.run_id,
            {"result": result},
        )
        await self._publish(completed)
        yield completed

    def _fail_strict_tool_calls(
        self,
        context: RunContext,
        calls: tuple[Any, ...],
        error_message: str,
    ) -> None:
        state = context.execution_state
        if isinstance(state, ExecutionState):
            state.validation_error = error_message
            state.fail_current_step()
            self._sync_execution_state(context, state)
        self._record_internal_rejected_calls(
            context,
            calls,
            error_message,
        )

    def _record_internal_rejected_calls(
        self,
        context: RunContext,
        calls: tuple[Any, ...],
        error_message: str,
    ) -> None:
        for call in calls:
            result = ToolResult.failed(
                call_id=call.id,
                tool_name=call.name,
                error=ToolError(
                    ToolErrorType.EXECUTION_ERROR,
                    error_message,
                ),
            )
            self._record_tool_trace(context, call, result)
            context.add_internal_message(
                Message.tool(
                    self._tool_result_content(result),
                    call_id=call.id,
                    name=call.name,
                    metadata={"moduagent.ephemeral": True},
                )
            )

    def _record_tool_trace(
        self,
        context: RunContext,
        call: Any,
        result: ToolResult,
    ) -> None:
        """Persist a bounded, sanitized business-Tool audit summary."""

        if (
            self.config.tool_trace_mode == "off"
            or call.name in SKILL_RESOURCE_TOOL_NAMES
        ):
            return

        trace = self._normalized_tool_trace(
            context.metadata.get(_TOOL_TRACE_METADATA_KEY)
        )
        # Successful business calls are already bounded by max_tool_calls. Keep
        # the same bound for terminal protocol rejections as well.
        if len(trace) >= self._tool_trace_limit:
            context.metadata[_TOOL_TRACE_METADATA_KEY] = trace
            return

        state = context.execution_state
        step_id = state.current_step_id if isinstance(state, ExecutionState) else None
        error = result.error
        recovery = getattr(error, "recovery", None) if error is not None else None
        entry: dict[str, Any] = {
            "step_id": step_id,
            "call_id": str(call.id),
            "tool_name": str(call.name),
            "success": bool(result.success),
            "attempts": int(result.attempts),
            "duration_seconds": float(result.duration_seconds),
            "error": (
                None
                if error is None
                else {
                    "type": error.type.value,
                    "retryable": bool(error.retryable),
                    **(
                        {"reason": str(error.reason)}
                        if getattr(error, "reason", None)
                        else {}
                    ),
                    **(
                        {"recovery": str(getattr(recovery, "value", recovery))}
                        if recovery is not None
                        else {}
                    ),
                }
            ),
        }
        pending_failure = getattr(state, "pending_tool_failure", None)
        if isinstance(pending_failure, Mapping):
            recovery_of_call_id = pending_failure.get("call_id")
            if recovery_of_call_id:
                entry["recovery_of_call_id"] = str(recovery_of_call_id)
        if self.config.tool_trace_mode == "arguments":
            invocation_arguments = getattr(result, "invocation_arguments", None)
            entry["arguments"] = (
                invocation_arguments
                if isinstance(invocation_arguments, Mapping)
                else call.arguments
            )
            entry["arguments_source"] = (
                "validated"
                if isinstance(invocation_arguments, Mapping)
                else "requested"
            )
        sanitized = self._sanitize_tool_trace_entry(entry)
        if sanitized is not None:
            trace.append(sanitized)
        context.metadata[_TOOL_TRACE_METADATA_KEY] = trace

    @property
    def _tool_trace_limit(self) -> int:
        return max(1, self.config.limits.max_tool_calls)

    @staticmethod
    def _tool_trace_text(value: Any) -> str:
        try:
            text = str(value)
        except Exception:
            text = type(value).__name__
        if len(text) <= _TOOL_TRACE_TEXT_CHARS:
            return text
        return f"{text[: _TOOL_TRACE_TEXT_CHARS - 3]}..."

    def _sanitize_tool_trace_arguments(
        self,
        arguments: Mapping[str, Any],
    ) -> Any:
        try:
            masked = mask_sensitive(dict(arguments))
            serialized = json.dumps(
                masked,
                ensure_ascii=False,
                default=str,
                separators=(",", ":"),
            )
        except Exception:
            return {"_unavailable": True}
        size_bytes = len(serialized.encode("utf-8"))
        if size_bytes > _TOOL_TRACE_ARGUMENT_BYTES:
            return {
                "_truncated": True,
                "size_bytes": size_bytes,
            }
        return json.loads(serialized)

    def _sanitize_tool_trace_entry(
        self,
        raw_entry: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        if self.config.tool_trace_mode == "off":
            return None
        try:
            attempts = int(raw_entry.get("attempts", 0))
        except (TypeError, ValueError, OverflowError):
            attempts = 0
        attempts = min(max(attempts, 0), 1_000_000)
        try:
            duration = float(raw_entry.get("duration_seconds", 0.0))
        except (TypeError, ValueError, OverflowError):
            duration = 0.0
        if not math.isfinite(duration) or duration < 0:
            duration = 0.0

        raw_error = raw_entry.get("error")
        error = None
        if isinstance(raw_error, Mapping):
            error = {
                "type": self._tool_trace_text(
                    raw_error.get("type", ToolErrorType.EXECUTION_ERROR.value)
                ),
                "retryable": bool(raw_error.get("retryable", False)),
            }
            if raw_error.get("reason"):
                error["reason"] = self._tool_trace_text(raw_error["reason"])
            if raw_error.get("recovery"):
                error["recovery"] = self._tool_trace_text(raw_error["recovery"])
        raw_step_id = raw_entry.get("step_id")
        entry: dict[str, Any] = {
            "step_id": (
                None if raw_step_id is None else self._tool_trace_text(raw_step_id)
            ),
            "call_id": self._tool_trace_text(raw_entry.get("call_id", "")),
            "tool_name": self._tool_trace_text(raw_entry.get("tool_name", "")),
            "success": bool(raw_entry.get("success", False)),
            "attempts": attempts,
            "duration_seconds": duration,
            "error": error,
        }
        if raw_entry.get("recovery_of_call_id"):
            entry["recovery_of_call_id"] = self._tool_trace_text(
                raw_entry["recovery_of_call_id"]
            )
        if self.config.tool_trace_mode == "arguments":
            raw_arguments = raw_entry.get("arguments")
            if isinstance(raw_arguments, Mapping):
                entry["arguments"] = self._sanitize_tool_trace_arguments(raw_arguments)
                entry["arguments_source"] = (
                    "validated"
                    if raw_entry.get("arguments_source") == "validated"
                    else "requested"
                )
        return entry

    def _normalized_tool_trace(self, raw_trace: Any) -> list[dict[str, Any]]:
        if self.config.tool_trace_mode == "off" or not isinstance(raw_trace, list):
            return []
        normalized: list[dict[str, Any]] = []
        for raw_entry in raw_trace:
            if len(normalized) >= self._tool_trace_limit:
                break
            if not isinstance(raw_entry, Mapping):
                continue
            entry = self._sanitize_tool_trace_entry(raw_entry)
            if entry is not None:
                normalized.append(entry)
        return normalized

    def _normalize_context_tool_trace(self, context: RunContext) -> None:
        context.metadata.pop(_PUBLIC_TOOL_TRACE_KEY, None)
        trace = self._normalized_tool_trace(
            context.metadata.get(_TOOL_TRACE_METADATA_KEY)
        )
        if trace:
            context.metadata[_TOOL_TRACE_METADATA_KEY] = trace
        else:
            context.metadata.pop(_TOOL_TRACE_METADATA_KEY, None)

    @staticmethod
    def _sync_execution_state(
        context: RunContext,
        state: ExecutionState,
    ) -> None:
        context.execution_state = state
        context.policy_state["execution_state"] = state.to_dict()
        context.policy_state["plan"] = state.plan.to_dict()

    async def _skill_events(
        self,
        context: RunContext,
        deadline: float,
        *,
        resumed: bool,
    ) -> AsyncIterator[AgentEvent]:
        skills_requested = context.request.skill_mode != "disabled" or bool(
            context.skill_state.active_skills
        )
        if not skills_requested:
            return
        if self.skill_runtime is None:
            raise RuntimeError(
                "this run requests Skills but the Agent has no skill_registry"
            )

        discovered = AgentEvent(
            EventType.SKILLS_DISCOVERED,
            context.run_id,
            {
                "count": len(self.skill_runtime.registry),
                "catalog_digest": self.skill_runtime.registry.catalog_digest,
            },
        )
        await self._publish(discovered)
        yield discovered

        try:
            restored = resumed and bool(context.skill_state.catalog_digest)
            if restored:
                report = await self._within(
                    deadline,
                    lambda: self.skill_runtime.arestore(context),
                )
            else:
                started = AgentEvent(
                    EventType.SKILL_SELECTION_STARTED,
                    context.run_id,
                    {
                        "mode": context.request.skill_mode,
                        "requested": list(context.request.requested_skills),
                        "resumed": resumed,
                    },
                )
                await self._publish(started)
                yield started
                business_tools = tuple(
                    tool.name
                    for tool in self.tool_executor.registry
                    if tool.name not in SKILL_RESOURCE_TOOL_NAMES
                )
                report = await self._within(
                    deadline,
                    lambda: self.skill_runtime.activate(
                        context,
                        available_tools=business_tools,
                    ),
                )
                completed = AgentEvent(
                    EventType.SKILL_SELECTION_COMPLETED,
                    context.run_id,
                    {
                        "mode": context.request.skill_mode,
                        "selected": list(report.selected),
                        "catalog_tokens": report.catalog_tokens,
                        "instruction_tokens": report.instruction_tokens,
                        "usage": report.usage,
                    },
                )
                await self._publish(completed)
                yield completed
        except Exception as exc:
            failed = AgentEvent(
                EventType.SKILL_ERROR,
                context.run_id,
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            await self._publish(failed)
            yield failed
            raise

        selected_by = {
            activation.name: activation.selected_by
            for activation in context.skill_state.active_skills
        }
        if not restored:
            for name in report.selected:
                selected = AgentEvent(
                    EventType.SKILL_SELECTED,
                    context.run_id,
                    {
                        "name": name,
                        "selected_by": selected_by[name],
                    },
                )
                await self._publish(selected)
                yield selected
        for activation in context.skill_state.active_skills:
            activated = AgentEvent(
                EventType.SKILL_ACTIVATED,
                context.run_id,
                {
                    "name": activation.name,
                    "version": activation.version,
                    "digest": activation.digest,
                    "source_id": activation.source_id,
                    "selected_by": activation.selected_by,
                    "resumed": restored,
                },
            )
            await self._publish(activated)
            yield activated

    def _tool_schemas(self, context: RunContext) -> tuple[Any, ...]:
        if self.skill_runtime is None:
            return tuple(self.tool_executor.registry.schemas())
        business_names = {
            tool.name
            for tool in self.tool_executor.registry
            if tool.name not in SKILL_RESOURCE_TOOL_NAMES
        }
        allowed = self.skill_runtime.allowed_tool_names(context)
        if allowed is None:
            selected = set(business_names)
        else:
            selected = set(allowed)
            if self.skill_runtime.has_resources(context):
                selected.add(SKILL_READ_TOOL_NAME)
            if self.skill_runtime.supports_resource_search(context):
                selected.add(SKILL_SEARCH_TOOL_NAME)
        return tuple(self.tool_executor.registry.schemas(selected))

    async def _model_events(
        self,
        context: RunContext,
        request: ModelRequest,
        deadline: float,
        *,
        stream_model: bool,
        phase: str,
        record_response: bool = True,
        record_internal: bool = False,
        visibility: EventVisibility = EventVisibility.PUBLIC,
        delta_event_type: EventType = EventType.MODEL_DELTA,
        buffer_deltas: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        response: ModelResponse | None = None
        emitted_delta = False
        buffered_deltas: list[str] = []
        for attempt in range(1, self.config.retry.max_attempts + 1):
            event = AgentEvent(
                EventType.MODEL_STARTED,
                context.run_id,
                {"step": context.step, "attempt": attempt, "phase": phase},
                visibility=visibility,
            )
            await self._publish(event)
            yield event
            try:
                capabilities = getattr(self.model, "capabilities", None)
                supports_streaming = bool(getattr(capabilities, "streaming", False))
                if stream_model and supports_streaming:
                    iterator = self.model.stream(request).__aiter__()
                    while True:
                        try:
                            chunk = await asyncio.wait_for(
                                iterator.__anext__(),
                                timeout=self._remaining(deadline),
                            )
                        except StopAsyncIteration:
                            break
                        if chunk.delta:
                            emitted_delta = True
                            buffered_deltas.append(chunk.delta)
                            if not buffer_deltas:
                                delta_event = AgentEvent(
                                    delta_event_type,
                                    context.run_id,
                                    {
                                        "step": context.step,
                                        "phase": phase,
                                        "delta": chunk.delta,
                                        "metadata": dict(chunk.provider_metadata),
                                    },
                                    visibility=visibility,
                                )
                                await self._publish(delta_event)
                                yield delta_event
                        if chunk.response is not None:
                            response = chunk.response
                    if response is None:
                        raise RuntimeError(
                            "model stream ended without a final response"
                        )
                else:
                    response = await asyncio.wait_for(
                        self.model.complete(request),
                        timeout=self._remaining(deadline),
                    )
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if emitted_delta or attempt >= self.config.retry.max_attempts:
                    raise
                retry_event = AgentEvent(
                    EventType.RETRY,
                    context.run_id,
                    {
                        "operation": "model",
                        "attempt": attempt,
                        "phase": phase,
                        "error": str(exc),
                    },
                    visibility=visibility,
                )
                await self._publish(retry_event)
                yield retry_event
                await asyncio.sleep(
                    min(
                        self.config.retry.delay_for(attempt),
                        self._remaining(deadline),
                    )
                )

        if response is None:
            raise RuntimeError("model returned no response")
        context.usage = context.usage + response.usage
        if record_response:
            response_calls = response.tool_calls or response.message.tool_calls
            ephemeral = any(
                call.name in SKILL_RESOURCE_TOOL_NAMES for call in response_calls
            )
            context.add_message(
                Message.assistant(
                    response.message.content,
                    response_calls,
                    metadata=({"moduagent.ephemeral": True} if ephemeral else None),
                ),
                persist=not ephemeral,
            )
        elif record_internal:
            response_calls = response.tool_calls or response.message.tool_calls
            context.add_internal_message(
                Message.assistant(response.message.content, response_calls)
            )
        event = AgentEvent(
            EventType.MODEL_COMPLETED,
            context.run_id,
            {
                "step": context.step,
                "phase": phase,
                "response": response,
                "usage": response.usage,
                "buffered_deltas": tuple(buffered_deltas),
            },
            visibility=visibility,
        )
        await self._publish(event)
        yield event

    async def _prepare_model_request(
        self,
        context: RunContext,
        request: ModelRequest,
        *,
        phase: MemoryPhase,
        deadline: float,
        skill_phase: str | None = None,
        protected_from: int | None = None,
    ) -> tuple[ModelRequest, AgentEvent | None]:
        request = replace(
            request,
            messages=compose_skill_prompt(
                request.messages,
                context.skill_messages,
                phase=skill_phase,
            ),
        )
        protected_boundary = (
            context.current_run_start + len(context.skill_messages)
            if protected_from is None
            else protected_from
        )
        memory = await self._within(
            deadline,
            lambda: self.conversation_memory_policy.prepare(
                MemoryRequest(
                    run_id=context.run_id,
                    session_id=context.request.session_id,
                    phase=phase,
                    model_request=request,
                    protected_from=protected_boundary,
                    user_context=context.request.user_context,
                )
            ),
        )
        context.usage = context.usage + memory.usage
        prepared = replace(request, messages=tuple(memory.messages))
        compacted = (
            prepared.messages != request.messages
            or memory.summarized_messages > 0
            or memory.dropped_messages > 0
        )
        if not compacted:
            return prepared, None

        data = dict(memory.metadata)
        data.update(
            {
                "phase": phase.value,
                "original_tokens": memory.original_tokens,
                "selected_tokens": memory.selected_tokens,
                "summarized_messages": memory.summarized_messages,
                "dropped_messages": memory.dropped_messages,
            }
        )
        event = AgentEvent(EventType.MEMORY_COMPACTED, context.run_id, data)
        await self._publish(event)
        return prepared, event

    async def _finalization_events(
        self,
        context: RunContext,
        output_schema: Any,
        deadline: float,
        *,
        stream_model: bool,
    ) -> AsyncIterator[AgentEvent]:
        state = context.policy_state.get(_FINALIZATION_STATE_KEY)
        response: ModelResponse | None = None

        if state != _FINALIZATION_COMPLETED:
            context.policy_state[_FINALIZATION_STATE_KEY] = _FINALIZATION_PENDING
            context.status = RunStatus.WAITING_FOR_MODEL
            await self._save_checkpoint(context, deadline)
            options = dict(self.config.model_options)
            options.pop("tool_choice", None)
            options.pop("parallel_tool_calls", None)
            request = ModelRequest(
                messages=(*context.messages, Message.user(_FINALIZATION_PROMPT)),
                tools=(),
                output_schema=output_schema,
                options=options,
            )
            request, memory_event = await self._prepare_model_request(
                context,
                request,
                phase=MemoryPhase.FINALIZE,
                deadline=deadline,
                skill_phase="finalize",
            )
            if memory_event is not None:
                yield memory_event
            async for event in self._model_events(
                context,
                request,
                deadline,
                stream_model=stream_model,
                phase="finalize",
                record_response=False,
            ):
                if event.type is EventType.MODEL_COMPLETED:
                    candidate = event.data.get("response")
                    if isinstance(candidate, ModelResponse):
                        response = candidate
                yield event
            if response is None:
                raise RuntimeError("finalization returned no response")
            if response.tool_calls or response.message.tool_calls:
                raise RuntimeError("finalization returned tool calls")
            finish_reason = (response.finish_reason or "").lower()
            if finish_reason in {"timeout", "length", "max_tokens"}:
                raise RuntimeError(
                    f"incomplete finalization response ({finish_reason})"
                )
            try:
                output = self.output_codec.decode(response)
            except Exception as exc:
                raise RuntimeError("finalization response validation failed") from exc
            raw_response = response.message.content
            if raw_response is None:
                raise RuntimeError("finalization response is empty")
            context.add_message(
                Message.assistant(
                    raw_response,
                    metadata={
                        _RUN_ID_METADATA_KEY: context.run_id,
                        _PUBLIC_FINAL_METADATA_KEY: True,
                    },
                )
            )
            context.policy_state[_FINALIZATION_OUTPUT_KEY] = raw_response
            context.policy_state[_FINALIZATION_STATE_KEY] = _FINALIZATION_COMPLETED
            context.status = RunStatus.RUNNING
            await self._save_checkpoint(context, deadline)
        else:
            if _FINALIZATION_OUTPUT_KEY not in context.policy_state:
                raise RuntimeError("finalization response is missing")
            try:
                output = self.output_codec.decode(
                    context.policy_state[_FINALIZATION_OUTPUT_KEY]
                )
            except Exception as exc:
                raise RuntimeError(
                    "stored finalization response validation failed"
                ) from exc

        result = await self._finalize(
            context,
            response,
            FinishReason.COMPLETED,
            deadline,
            output=output,
        )
        event = AgentEvent(EventType.RUN_COMPLETED, context.run_id, {"result": result})
        await self._publish(event)
        yield event

    async def _finalize(
        self,
        context: RunContext,
        response: ModelResponse | None,
        reason: FinishReason,
        deadline: float,
        *,
        output: Any = None,
    ) -> AgentResult:
        if output is None and reason is FinishReason.COMPLETED and response is not None:
            output = self.output_codec.decode(response)
        await self._persist_pending_messages(context, deadline)
        if self.checkpoint_store is not None:
            await self._within(
                deadline,
                lambda: self.checkpoint_store.delete(context.run_id),
            )
        context.status = RunStatus.COMPLETED
        return self._result(context, reason, output=output)

    def _result(
        self,
        context: RunContext,
        reason: FinishReason,
        *,
        output: Any = None,
        error: str | None = None,
    ) -> AgentResult:
        metadata = {
            key: value
            for key, value in context.metadata.items()
            if not key.startswith("_moduagent_")
            and key
            not in {
                "execution_state",
                "validation",
                "requires_step_result",
                _PUBLIC_TOOL_TRACE_KEY,
            }
        }
        if isinstance(context.execution_state, ExecutionState):
            state = context.execution_state
            metadata["plan"] = state.plan.to_dict()
            metadata["plan_usage"] = {
                "phase": state.phase.value,
                "committed_steps": len(state.committed_results),
                "replans": state.replan_count,
                "finalization_calls": state.finalization_count,
            }
            total_tool_repairs = int(getattr(state, "total_tool_repairs", 0) or 0)
            if total_tool_repairs:
                metadata["plan_usage"]["tool_repairs"] = total_tool_repairs
            failure = self._normalized_execution_failure(
                getattr(state, "failure", None)
            )
            if failure is not None:
                metadata["failure"] = mask_sensitive(failure)
        tool_trace = self._normalized_tool_trace(
            context.metadata.get(_TOOL_TRACE_METADATA_KEY)
        )
        if tool_trace:
            metadata[_PUBLIC_TOOL_TRACE_KEY] = tool_trace
        if context.skill_state.catalog_digest:
            metadata["skill_usage"] = {
                "catalog_digest": context.skill_state.catalog_digest,
                "active_skills": len(context.skill_state.active_skills),
                "resource_reads": context.skill_state.resource_reads,
                "instruction_tokens": context.skill_state.instruction_tokens,
                "resource_tokens": context.skill_state.resource_tokens,
            }
        return AgentResult(
            run_id=context.run_id,
            output=output,
            messages=tuple(
                message
                for message in context.messages
                if not is_ephemeral_message(message)
            ),
            usage=context.usage,
            finish_reason=reason,
            error=error,
            metadata=metadata,
        )

    async def _save_checkpoint(self, context: RunContext, deadline: float) -> None:
        if self.checkpoint_store is None:
            return
        await self._within(
            deadline,
            lambda: self.checkpoint_store.save(context.run_id, context),
        )

    async def _save_checkpoint_safely(self, context: RunContext) -> None:
        if self.checkpoint_store is None:
            return
        try:
            await asyncio.wait_for(
                self.checkpoint_store.save(context.run_id, context), timeout=1.0
            )
        except Exception:
            pass

    async def _persist_pending_messages(
        self,
        context: RunContext,
        deadline: float,
    ) -> None:
        if not context.new_messages:
            return
        existing = await self._within(
            deadline,
            lambda: self.conversation_store.load(context.request.session_id),
        )
        existing_keys = {
            key
            for message in existing
            if (key := self._message_idempotency_key(message)) is not None
        }
        additions: list[Message] = []
        for message in context.new_messages:
            key = self._message_idempotency_key(message)
            if key is None or key not in existing_keys:
                additions.append(message)
                if key is not None:
                    existing_keys.add(key)
        if additions:
            await self._within(
                deadline,
                lambda: self.conversation_store.append(
                    context.request.session_id,
                    tuple(additions),
                ),
            )
        context.new_messages.clear()

    async def _persist_safely(self, context: RunContext) -> None:
        if not context.new_messages:
            return
        try:
            deadline = asyncio.get_running_loop().time() + 1.0
            await self._persist_pending_messages(
                context,
                deadline,
            )
        except Exception:
            pass

    @staticmethod
    def _message_idempotency_key(
        message: Message,
    ) -> tuple[str, str, bool, bool] | None:
        run_id = message.metadata.get(_RUN_ID_METADATA_KEY)
        if not isinstance(run_id, str) or not run_id:
            return None
        return (
            run_id,
            message.role.value,
            message.metadata.get(_PUBLIC_FINAL_METADATA_KEY) is True,
            message.metadata.get("moduagent.public_input") is True,
        )

    async def _publish(self, event: AgentEvent) -> None:
        try:
            result = self.event_sink.publish(event)
            if inspect.isawaitable(result):
                await result
        except Exception:
            pass

    async def _within(self, deadline: float, factory: Callable[[], Awaitable[T]]) -> T:
        return await asyncio.wait_for(factory(), timeout=self._remaining(deadline))

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError
        return remaining

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @classmethod
    def _resource_tokens(cls, value: Any) -> int:
        payload = cls._json(value).encode("utf-8")
        return max(1, (len(payload) + 2) // 3)

    def _tool_result_content(self, result: ToolResult) -> str:
        for method_name in ("model_content", "to_message_content"):
            method = getattr(result, method_name, None)
            if callable(method):
                return str(method())
        error = getattr(result, "error", None)
        return self._json(
            {
                "success": bool(result.success),
                "value": getattr(result, "value", None),
                "error": (getattr(error, "message", str(error)) if error else None),
            }
        )

    def _record_rejected_tool_calls(
        self,
        context: RunContext,
        calls: tuple[Any, ...],
        *,
        error_message: str,
    ) -> tuple[ToolResult, ...]:
        # A response containing any Skill resource call is itself ephemeral.
        # Keep every matching Tool result ephemeral as one protocol block so
        # ConversationStore never receives an orphan from a mixed response.
        ephemeral_block = any(call.name in SKILL_RESOURCE_TOOL_NAMES for call in calls)
        results: list[ToolResult] = []
        for call in calls:
            result = ToolResult.failed(
                call_id=call.id,
                tool_name=call.name,
                error=ToolError(
                    ToolErrorType.EXECUTION_ERROR,
                    error_message,
                ),
            )
            results.append(result)
            self._record_tool_trace(context, call, result)
            context.add_message(
                Message.tool(
                    self._tool_result_content(result),
                    call_id=call.id,
                    name=call.name,
                    metadata=(
                        {"moduagent.ephemeral": True} if ephemeral_block else None
                    ),
                ),
                persist=not ephemeral_block,
            )
        return tuple(results)
