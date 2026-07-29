from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from moduagent.errors import (
    ExecutionInvariantError,
    ModelInvocationError,
    ModuAgentError,
    PersistenceError,
    SkillError,
    ToolInvocationError,
)
from moduagent.execution.base import (
    DurableBoundary,
    EngineContext,
    EngineSnapshot,
    ExecutionBudget,
    FinalizationResult,
)
from moduagent.memory import MemoryPhase
from moduagent.messages import Message, MessageRole
from moduagent.models import (
    ModelChunk,
    ModelClient,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    validate_request_capabilities,
)
from moduagent.runtime.events import AgentEvent
from moduagent.runtime.events import EventType
from moduagent.runtime.events import EventVisibility
from moduagent.runtime.context import RunStatus
from moduagent.skills.tools import SKILL_RESOURCE_TOOL_NAMES
from moduagent.tools import (
    FailureProjector,
    InternalToolFailure,
    ToolBatchOutcome,
    ToolError,
    ToolErrorType,
    ToolExecutionContext,
    ToolFailureClassification,
    ToolRepairConstraint,
    ToolResult,
    ToolSafetyProfile,
    ToolSchema,
    fingerprint_tool_arguments,
)


_RUN_ID_METADATA_KEY = "moduagent.run_id"
_PUBLIC_FINAL_METADATA_KEY = "moduagent.public_final"
_ENGINE_OWNED_POLICY_KEYS = frozenset(
    {
        "_moduagent_engine_snapshot",
        "execution_state",
        "plan",
    }
)


def _contains_unsupported_projection(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "unsupported_type" in value:
            return True
        return any(_contains_unsupported_projection(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_unsupported_projection(item) for item in value)
    return False


@dataclass(slots=True)
class RuntimeServices:
    """Concrete Engine services backed by the existing production runtime.

    This adapter is the compatibility seam while the public ``AgentRuntime``
    remains available in 0.4. Engines see only the explicit service contract.
    """

    runtime: Any
    deadline: float
    _pending_events: list[AgentEvent] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _pending_after_events: list[AgentEvent] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
    _bound_context: EngineContext | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _last_engine_snapshot: EngineSnapshot | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _pending_event_signal: asyncio.Event = field(
        default_factory=asyncio.Event,
        init=False,
        repr=False,
    )

    def bind(self, context: EngineContext) -> None:
        """Bind this per-run service bundle as the auxiliary ModelGateway."""

        if self._bound_context is not None and self._bound_context is not context:
            raise ExecutionInvariantError("RuntimeServices cannot be rebound")
        self._bound_context = context
        context.run.model_gateway = self

    async def complete(
        self,
        model: ModelClient,
        request: ModelRequest,
        *,
        phase: str,
    ) -> ModelResponse:
        """Run an auxiliary model call through the common provider boundary."""

        context = self._bound_context
        if context is None:
            raise ExecutionInvariantError("ModelGateway is not bound to a run")
        return await self._complete_model(context, model, request, phase=phase)

    def drain_events(self) -> tuple[AgentEvent, ...]:
        """Return service-owned events waiting to enter the public stream."""

        pending_sequences = {event.sequence for event in self._pending_events}
        related_after = [
            event
            for event in self._pending_after_events
            if event.sequence - 1 in pending_sequences
        ]
        events = tuple(
            sorted(
                (*self._pending_events, *related_after),
                key=lambda event: event.sequence,
            )
        )
        self._pending_events.clear()
        if related_after:
            related_ids = {event.event_id for event in related_after}
            self._pending_after_events[:] = [
                event
                for event in self._pending_after_events
                if event.event_id not in related_ids
            ]
        if not self._pending_events:
            self._pending_event_signal.clear()
        return events

    async def wait_for_events(self) -> None:
        """Wait until a service-owned event is ready for live streaming."""

        while not self._pending_events:
            self._pending_event_signal.clear()
            if self._pending_events:
                return
            await self._pending_event_signal.wait()

    def _enqueue_event(self, event: AgentEvent) -> None:
        self._pending_events.append(event)
        self._pending_event_signal.set()

    def drain_after_events(self) -> tuple[AgentEvent, ...]:
        """Return service events ordered after the current Engine event."""

        events = tuple(self._pending_after_events)
        self._pending_after_events.clear()
        return events

    def budget(self, context: EngineContext) -> ExecutionBudget:
        return ExecutionBudget.from_config(context.config)

    def remaining_seconds(self, context: EngineContext) -> float:
        return self.runtime._remaining(self.deadline)

    def tool_schemas(
        self,
        context: EngineContext,
        names: frozenset[str] | None = None,
    ) -> tuple[ToolSchema, ...]:
        available = tuple(self.runtime._tool_schemas(context.run))
        if names is None:
            return available
        by_name = {schema.name: schema for schema in available}
        unknown = names.difference(by_name)
        if unknown:
            raise ValueError(
                "unknown or unavailable tools: " + ", ".join(sorted(unknown))
            )
        return tuple(schema for schema in available if schema.name in names)

    def output_schema(
        self,
        context: EngineContext,
    ) -> Mapping[str, Any] | None:
        schema = self.runtime.output_codec.schema()
        return None if schema is None else dict(schema)

    def decode_output(
        self,
        context: EngineContext,
        response: ModelResponse,
    ) -> Any:
        return self.runtime.output_codec.decode(response)

    async def prepare_model_request(
        self,
        context: EngineContext,
        request: ModelRequest,
        *,
        phase: str,
        skill_phase: str | None,
        protected_from: int | None = None,
    ) -> ModelRequest:
        self.runtime._normalize_skill_resource_messages(context.run)
        prepared, memory_event = await self.runtime._prepare_model_request(
            context.run,
            request,
            phase=_memory_phase(phase),
            deadline=self.deadline,
            skill_phase=skill_phase,
            protected_from=protected_from,
        )
        if memory_event is not None:
            self._enqueue_event(self.runtime._published_event(memory_event))
        return prepared

    async def request_model(
        self,
        context: EngineContext,
        request: ModelRequest,
        *,
        phase: str,
    ) -> ModelResponse:
        return await self._complete_model(
            context,
            self.runtime.model,
            request,
            phase=phase,
        )

    async def _complete_model(
        self,
        context: EngineContext,
        model: ModelClient,
        request: ModelRequest,
        *,
        phase: str,
    ) -> ModelResponse:
        capabilities = getattr(model, "capabilities", None)
        if capabilities is None:
            capabilities = ModelCapabilities()
        if not isinstance(capabilities, ModelCapabilities):
            raise ModelInvocationError("model capabilities are invalid")
        if not capabilities.chat:
            raise ModelInvocationError("model does not support chat")
        try:
            validate_request_capabilities(request, capabilities)
        except Exception as exc:
            raise ModelInvocationError(
                "model request is incompatible with configured capabilities"
            ) from exc
        context.run.status = RunStatus.WAITING_FOR_MODEL
        for attempt in range(1, context.config.retry.max_attempts + 1):
            await self._queue_model_started(
                context,
                attempt=attempt,
                phase=phase,
            )
            try:
                response = await asyncio.wait_for(
                    model.complete(request),
                    timeout=self.remaining_seconds(context),
                )
                if not isinstance(response, ModelResponse):
                    raise TypeError("model must return ModelResponse")
                context.run.usage = context.run.usage + response.usage
                context.run.status = RunStatus.RUNNING
                await self._queue_model_completed(
                    context,
                    response=response,
                    attempt=attempt,
                    phase=phase,
                )
                return response
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if attempt >= context.config.retry.max_attempts:
                    if isinstance(exc, asyncio.TimeoutError):
                        raise
                    raise ModelInvocationError("model invocation failed") from exc
                await self._queue_retry_event(
                    context,
                    attempt=attempt,
                    phase=phase,
                    error=exc,
                )
                delay = min(
                    context.config.retry.delay_for(attempt),
                    self.remaining_seconds(context),
                )
                if delay:
                    await asyncio.sleep(delay)
        raise AssertionError("model retry loop ended unexpectedly")

    async def stream_model(
        self,
        context: EngineContext,
        request: ModelRequest,
        *,
        phase: str,
        delta_event_type: EventType | None = EventType.MODEL_DELTA,
        delta_visibility: EventVisibility = EventVisibility.PUBLIC,
        delta_data: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[ModelChunk]:
        if delta_event_type is not None and not isinstance(
            delta_event_type,
            EventType,
        ):
            raise TypeError("delta_event_type must be an EventType or None")
        if not isinstance(delta_visibility, EventVisibility):
            raise TypeError("delta_visibility must be an EventVisibility")
        if delta_data is not None and not isinstance(delta_data, Mapping):
            raise TypeError("delta_data must be a mapping or None")
        model = self.runtime.model
        capabilities = getattr(model, "capabilities", None)
        if capabilities is None:
            capabilities = replace(
                ModelCapabilities(),
                streaming=callable(getattr(model, "stream", None)),
            )
        if not isinstance(capabilities, ModelCapabilities):
            raise ModelInvocationError("model capabilities are invalid")
        if not capabilities.chat:
            raise ModelInvocationError("model does not support chat")
        can_stream = bool(
            capabilities.streaming and callable(getattr(model, "stream", None))
        )
        try:
            validate_request_capabilities(
                request,
                capabilities,
                streaming=can_stream,
            )
        except Exception as exc:
            raise ModelInvocationError(
                "model request is incompatible with configured capabilities"
            ) from exc
        if not can_stream:
            yield ModelChunk(
                response=await self.request_model(
                    context,
                    request,
                    phase=phase,
                )
            )
            return

        emitted_output = False
        context.run.status = RunStatus.WAITING_FOR_MODEL
        for attempt in range(1, context.config.retry.max_attempts + 1):
            await self._queue_model_started(
                context,
                attempt=attempt,
                phase=phase,
            )
            try:
                iterator = model.stream(request).__aiter__()
                terminal = False
                response: ModelResponse | None = None
                while True:
                    try:
                        chunk = await asyncio.wait_for(
                            iterator.__anext__(),
                            timeout=self.remaining_seconds(context),
                        )
                    except StopAsyncIteration:
                        break
                    if not isinstance(chunk, ModelChunk):
                        raise TypeError("model stream must yield ModelChunk")
                    emitted_output = emitted_output or bool(chunk.delta)
                    terminal = terminal or chunk.response is not None
                    if chunk.response is not None:
                        response = chunk.response
                    if chunk.delta and delta_event_type is not None:
                        await self.defer_event(
                            context,
                            AgentEvent(
                                delta_event_type,
                                context.run.run_id,
                                {
                                    **dict(delta_data or {}),
                                    "phase": phase,
                                    "delta": chunk.delta,
                                },
                                visibility=delta_visibility,
                            ),
                        )
                    yield chunk
                if not terminal:
                    raise RuntimeError("model stream ended without a final response")
                if response is None:
                    raise RuntimeError("model stream returned no terminal response")
                context.run.usage = context.run.usage + response.usage
                context.run.status = RunStatus.RUNNING
                await self._queue_model_completed(
                    context,
                    response=response,
                    attempt=attempt,
                    phase=phase,
                )
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if emitted_output or attempt >= context.config.retry.max_attempts:
                    if isinstance(exc, asyncio.TimeoutError):
                        raise
                    raise ModelInvocationError("model invocation failed") from exc
                await self._queue_retry_event(
                    context,
                    attempt=attempt,
                    phase=phase,
                    error=exc,
                )
                delay = min(
                    context.config.retry.delay_for(attempt),
                    self.remaining_seconds(context),
                )
                if delay:
                    await asyncio.sleep(delay)
        raise AssertionError("model stream retry loop ended unexpectedly")

    async def execute_tool_batch(
        self,
        context: EngineContext,
        calls: tuple[Any, ...],
        *,
        allowed_tools: frozenset[str] | None,
        repair_constraint: ToolRepairConstraint | None,
    ) -> ToolBatchOutcome:
        if allowed_tools is not None:
            disallowed = sorted({call.name for call in calls}.difference(allowed_tools))
            if disallowed:
                raise ExecutionInvariantError(
                    "Engine attempted tools outside its scope: " + ", ".join(disallowed)
                )
        executor = self.runtime.tool_executor
        execute_batch = getattr(executor, "execute_batch", None)
        execute_many = getattr(executor, "execute_many", None)
        if not callable(execute_batch) and not callable(execute_many):
            raise ToolInvocationError("Tool executor must provide a batch operation")
        tool_context = ToolExecutionContext(
            run_id=context.run.run_id,
            session_id=context.run.request.session_id,
            user_context=dict(context.run.request.user_context),
            metadata={
                "agent": context.config.name,
                "active_skills": [
                    activation.to_dict()
                    for activation in context.run.skill_state.active_skills
                ],
            },
        )
        capabilities = getattr(self.runtime.model, "capabilities", None)
        parallel = bool(
            context.config.limits.parallel_tool_calls
            and getattr(capabilities, "parallel_tool_calling", True)
        )
        self._validate_skill_resource_batch(context, calls)
        if self.runtime.checkpoint_store is not None:
            # Until a post-outcome Engine checkpoint proves otherwise, an
            # interrupted invocation may have completed outside this process.
            context.run.metadata["_moduagent_resume_safety"] = "manual_required"
            if self._last_engine_snapshot is None:
                raise ExecutionInvariantError(
                    "Tool execution requires a durable Engine snapshot"
                )
            # Persist the fail-closed invocation intent before any Tool event
            # or side effect. A failed save aborts the invocation.
            await self.checkpoint(
                context,
                self._last_engine_snapshot,
                boundary=DurableBoundary.TOOL_INVOCATION_PENDING,
            )
        for call in calls:
            await self.defer_event(
                context,
                AgentEvent(
                    EventType.TOOL_STARTED,
                    context.run.run_id,
                    {
                        "tool_call": call,
                        "step": context.run.step,
                    },
                    visibility=EventVisibility.INTERNAL,
                ),
            )
        context.run.status = RunStatus.WAITING_FOR_TOOLS
        try:
            if callable(execute_batch):
                try:
                    outcome = await asyncio.wait_for(
                        execute_batch(
                            calls,
                            tool_context,
                            parallel=parallel,
                            max_parallel=(context.config.limits.max_parallel_tools),
                            repair_constraint=repair_constraint,
                        ),
                        timeout=self.remaining_seconds(context),
                    )
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    raise
                except ModuAgentError:
                    raise
                except Exception:
                    raise ToolInvocationError("tool execution failed") from None
            else:
                try:
                    raw_results = await asyncio.wait_for(
                        execute_many(
                            calls,
                            tool_context,
                            parallel=parallel,
                            max_parallel=(context.config.limits.max_parallel_tools),
                            repair_constraint=repair_constraint,
                        ),
                        timeout=self.remaining_seconds(context),
                    )
                except asyncio.TimeoutError:
                    raise
                except asyncio.CancelledError:
                    raise
                except ModuAgentError:
                    raise
                except Exception:
                    raise ToolInvocationError("tool execution failed") from None
                outcome = self._legacy_tool_outcome(calls, raw_results)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._queue_tool_execution_failure(
                context,
                calls,
                timed_out=isinstance(exc, asyncio.TimeoutError),
            )
            raise
        finally:
            context.run.status = RunStatus.RUNNING
        if not isinstance(outcome, ToolBatchOutcome):
            await self._queue_tool_execution_failure(context, calls)
            raise ToolInvocationError("tool executor returned mismatched results")
        outcome = self._apply_skill_resource_limits(context, outcome)
        safe_views = {view.call_id: view for view in outcome.sanitized_failure_views}
        for call, result in zip(outcome.calls, outcome.results):
            self.runtime._record_tool_trace(context.run, call, result)
            await self._queue_tool_completed(
                context,
                call,
                result,
                failure=(
                    None
                    if result.success or result.call_id not in safe_views
                    else safe_views[result.call_id].to_dict()
                ),
            )
        return outcome

    async def record_tool_result(
        self,
        context: EngineContext,
        call: Any,
        result: ToolResult,
    ) -> None:
        """Record an Engine-created rejection in the common audit trace."""

        if not isinstance(result, ToolResult):
            raise TypeError("result must be a ToolResult")
        self.runtime._record_tool_trace(context.run, call, result)
        failure = None
        if not result.success:
            error = result.error
            recovery = getattr(getattr(error, "recovery", None), "value", None)
            failure = {
                "type": getattr(
                    getattr(error, "type", None),
                    "value",
                    ToolErrorType.EXECUTION_ERROR.value,
                ),
                "reason": (
                    getattr(error, "reason", None)
                    or ToolErrorType.EXECUTION_ERROR.value
                ),
                "retryable": bool(getattr(error, "retryable", False)),
                **({} if recovery is None else {"recovery": recovery}),
            }
        await self._queue_tool_completed(
            context,
            call,
            result,
            failure=failure,
        )

    async def finalize(
        self,
        context: EngineContext,
        request: ModelRequest,
        *,
        phase: str,
    ) -> FinalizationResult:
        await self.defer_event(
            context,
            AgentEvent(
                EventType.FINALIZATION_STARTED,
                context.run.run_id,
                {"phase": phase},
                visibility=EventVisibility.INTERNAL,
            ),
        )
        response: ModelResponse | None = None
        deltas: list[str] = []
        if context.stream_model:
            async for chunk in self.stream_model(
                context,
                request,
                phase=phase,
                delta_event_type=None,
            ):
                if chunk.delta:
                    deltas.append(chunk.delta)
                if chunk.response is not None:
                    response = chunk.response
        else:
            response = await self.request_model(context, request, phase=phase)
        if response is None:
            raise ModelInvocationError("finalization returned no response")
        if response.tool_calls or response.message.tool_calls:
            raise ModelInvocationError("finalization returned tool calls")
        finish_reason = (response.finish_reason or "").lower()
        if finish_reason in {"timeout", "length", "max_tokens"}:
            raise ModelInvocationError(
                f"incomplete finalization response ({finish_reason})"
            )
        output = self.runtime.output_codec.decode(response)
        content = response.message.content
        if content is None or not str(content).strip():
            raise ModelInvocationError("finalization response is empty")
        finalized = FinalizationResult(
            response=response,
            output=output,
            content=str(content),
            buffered_deltas=tuple(deltas),
        )
        await self.defer_event(
            context,
            AgentEvent(
                EventType.FINALIZATION_COMPLETED,
                context.run.run_id,
                {
                    "phase": phase,
                    "persisted": False,
                },
                visibility=EventVisibility.INTERNAL,
            ),
        )
        return finalized

    async def persist_finalization(
        self,
        context: EngineContext,
        finalized: FinalizationResult,
    ) -> FinalizationResult:
        """Durably store one public final response before Engine emission.

        The Engine checkpoints the generated response first, calls this
        operation, and then checkpoints its persisted marker. Message identity
        makes resume and repeated persistence attempts idempotent.
        """

        if not isinstance(finalized, FinalizationResult):
            raise TypeError("finalized must be a FinalizationResult")
        if finalized.persisted:
            return finalized
        exists = any(
            message.role is MessageRole.ASSISTANT
            and message.metadata.get(_RUN_ID_METADATA_KEY) == context.run.run_id
            and message.metadata.get(_PUBLIC_FINAL_METADATA_KEY) is True
            for message in context.run.messages
        )
        if not exists:
            context.run.add_message(
                Message.assistant(
                    finalized.content,
                    metadata={
                        _RUN_ID_METADATA_KEY: context.run.run_id,
                        _PUBLIC_FINAL_METADATA_KEY: True,
                    },
                )
            )
        await self.runtime._persist_pending_messages(
            context.run,
            self.deadline,
        )
        return replace(finalized, persisted=True)

    async def emit_finalization(
        self,
        context: EngineContext,
        finalized: FinalizationResult,
        *,
        phase: str,
    ) -> None:
        if not isinstance(finalized, FinalizationResult):
            raise TypeError("finalized must be a FinalizationResult")
        if not finalized.persisted:
            raise ExecutionInvariantError(
                "finalization must be persisted before public emission"
            )
        if not context.stream_model:
            return
        deltas = (
            finalized.buffered_deltas
            if finalized.buffered_deltas
            and "".join(finalized.buffered_deltas) == finalized.content
            else (finalized.content,)
        )
        for value in deltas:
            await self.defer_event(
                context,
                AgentEvent(
                    EventType.FINAL_DELTA,
                    context.run.run_id,
                    {"phase": phase, "delta": value},
                    visibility=EventVisibility.PUBLIC,
                ),
            )

    async def checkpoint(
        self,
        context: EngineContext,
        snapshot: EngineSnapshot,
        *,
        boundary: DurableBoundary,
    ) -> None:
        self._last_engine_snapshot = snapshot
        context.run.metadata["_moduagent_engine_id"] = snapshot.engine_id
        context.run.metadata["_moduagent_engine_state_version"] = snapshot.state_version
        context.run.metadata["_moduagent_engine"] = {
            "engine_id": snapshot.engine_id,
            "state_version": snapshot.state_version,
            "state": dict(snapshot.state),
            "durable_boundary": boundary.value,
        }
        # Preserve the opaque Engine state even when a legacy CheckpointStore
        # adapter is used. SnapshotStore implementations receive the v4 state
        # directly; the compatibility key is a lossless fallback for 0.3
        # stores whose save() method accepts only RunContext.
        context.run.policy_state["_moduagent_engine_snapshot"] = {
            "engine_id": snapshot.engine_id,
            "state_version": snapshot.state_version,
            "state": dict(snapshot.state),
        }
        if self.runtime.checkpoint_store is not None:
            saved_event = self.runtime._reserve_event(
                AgentEvent(
                    EventType.CHECKPOINT_SAVED,
                    context.run.run_id,
                    {
                        "engine_id": snapshot.engine_id,
                        "state_version": snapshot.state_version,
                        "boundary": boundary.value,
                    },
                )
            )
            try:
                await self._save_engine_snapshot(
                    context,
                    snapshot,
                    timeout=self.remaining_seconds(context),
                )
            except BaseException:
                self.runtime._reserved_events.pop(
                    (saved_event.run_id, saved_event.event_id),
                    None,
                )
                raise
            self._enqueue_event(
                await self.runtime._dispatch_reserved_event(saved_event)
            )

    async def checkpoint_safely(
        self,
        context: EngineContext,
        snapshot: EngineSnapshot,
        *,
        boundary: DurableBoundary = DurableBoundary.INTERRUPTED,
    ) -> None:
        """Best-effort checkpoint using an independent failure-path budget."""

        try:
            context.run.metadata["_moduagent_engine_id"] = snapshot.engine_id
            context.run.metadata["_moduagent_engine_state_version"] = (
                snapshot.state_version
            )
            context.run.metadata["_moduagent_engine"] = {
                "engine_id": snapshot.engine_id,
                "state_version": snapshot.state_version,
                "state": dict(snapshot.state),
                "durable_boundary": boundary.value,
            }
            context.run.policy_state["_moduagent_engine_snapshot"] = {
                "engine_id": snapshot.engine_id,
                "state_version": snapshot.state_version,
                "state": dict(snapshot.state),
            }
            await self._save_engine_snapshot(context, snapshot, timeout=1.0)
        except Exception:
            pass

    async def publish_event(
        self,
        context: EngineContext,
        event: AgentEvent,
    ) -> AgentEvent:
        if event.run_id != context.run.run_id:
            raise ExecutionInvariantError(
                "Execution service event does not match the active run"
            )
        resource_call = self._skill_resource_call(event)
        resource_result: ToolResult | None = None
        if resource_call is not None:
            data: dict[str, Any] = {
                "tool_name": resource_call.name,
                "skill_name": resource_call.arguments.get("skill_name"),
                "path": resource_call.arguments.get("path"),
            }
            if event.type is EventType.TOOL_STARTED:
                data["resource_operation"] = (
                    "read" if resource_call.name.endswith("_read") else "search"
                )
            elif event.type is EventType.TOOL_COMPLETED:
                data["success"] = bool(event.data.get("success", False))
                candidate = event.data.get("result")
                if isinstance(candidate, ToolResult):
                    resource_result = candidate
            event = replace(event, data=data)
        else:
            event = self._project_tool_event(event)
        published = await self.runtime._publish(event)
        published = published if isinstance(published, AgentEvent) else event
        if resource_call is not None and event.type is EventType.TOOL_COMPLETED:
            value = (
                resource_result.value
                if resource_result is not None
                and isinstance(resource_result.value, dict)
                else {}
            )
            resource_event = AgentEvent(
                EventType.SKILL_RESOURCE_READ,
                context.run.run_id,
                {
                    "skill_name": resource_call.arguments.get("skill_name"),
                    "path": resource_call.arguments.get("path"),
                    "operation": (
                        "read" if resource_call.name.endswith("_read") else "search"
                    ),
                    "success": bool(event.data.get("success", False)),
                    "digest": value.get("digest"),
                    "truncated": value.get("truncated"),
                    "returned_bytes": value.get("returned_bytes"),
                    "scanned_bytes": value.get("scanned_bytes"),
                },
            )
            resource_published = await self.runtime._publish(resource_event)
            self._pending_after_events.append(
                resource_published
                if isinstance(resource_published, AgentEvent)
                else resource_event
            )
        return published

    async def defer_event(
        self,
        context: EngineContext,
        event: AgentEvent,
    ) -> None:
        self._enqueue_event(await self.publish_event(context, event))

    async def _queue_retry_event(
        self,
        context: EngineContext,
        *,
        attempt: int,
        phase: str,
        error: Exception,
    ) -> None:
        event = AgentEvent(
            EventType.RETRY,
            context.run.run_id,
            {
                "operation": "model",
                "attempt": attempt,
                "phase": phase,
                "error": "model request failed",
                "error_type": type(error).__name__,
            },
        )
        self._enqueue_event(await self.publish_event(context, event))

    async def _queue_model_started(
        self,
        context: EngineContext,
        *,
        attempt: int,
        phase: str,
    ) -> None:
        await self.defer_event(
            context,
            AgentEvent(
                EventType.MODEL_STARTED,
                context.run.run_id,
                {
                    "step": context.run.step,
                    "attempt": attempt,
                    "phase": phase,
                },
                visibility=EventVisibility.INTERNAL,
            ),
        )

    async def _queue_model_completed(
        self,
        context: EngineContext,
        *,
        response: ModelResponse,
        attempt: int,
        phase: str,
    ) -> None:
        await self.defer_event(
            context,
            AgentEvent(
                EventType.MODEL_COMPLETED,
                context.run.run_id,
                {
                    "step": context.run.step,
                    "attempt": attempt,
                    "phase": phase,
                    "usage": response.usage,
                    "finish_reason": response.finish_reason,
                },
                visibility=EventVisibility.INTERNAL,
            ),
        )

    async def _queue_tool_completed(
        self,
        context: EngineContext,
        call: Any,
        result: ToolResult,
        *,
        failure: Mapping[str, Any] | None,
    ) -> None:
        await self.defer_event(
            context,
            AgentEvent(
                EventType.TOOL_COMPLETED,
                context.run.run_id,
                {
                    "tool_call": call,
                    "result": result,
                    "success": result.success,
                    "step": context.run.step,
                    **({} if failure is None else {"failure": dict(failure)}),
                },
                visibility=EventVisibility.INTERNAL,
            ),
        )

    async def _queue_tool_execution_failure(
        self,
        context: EngineContext,
        calls: tuple[Any, ...],
        *,
        timed_out: bool = False,
    ) -> None:
        error_type = (
            ToolErrorType.TIMEOUT if timed_out else ToolErrorType.EXECUTION_ERROR
        )
        for call in calls:
            result = ToolResult.failed(
                call_id=call.id,
                tool_name=call.name,
                error=ToolError(
                    error_type,
                    (
                        "tool execution timed out"
                        if timed_out
                        else "tool execution failed"
                    ),
                    reason=error_type.value,
                ),
            )
            self.runtime._record_tool_trace(context.run, call, result)
            await self._queue_tool_completed(
                context,
                call,
                result,
                failure={
                    "type": error_type.value,
                    "reason": error_type.value,
                    "retryable": False,
                },
            )

    @staticmethod
    def _project_tool_event(event: AgentEvent) -> AgentEvent:
        """Remove Tool payloads and arguments before crossing the event boundary."""

        if event.type not in {
            EventType.TOOL_STARTED,
            EventType.TOOL_COMPLETED,
        }:
            return event
        call = event.data.get("tool_call")
        result = event.data.get("result")
        tool_name = event.data.get("tool_name")
        call_id = event.data.get("call_id")
        if call is not None:
            tool_name = getattr(call, "name", tool_name)
            call_id = getattr(call, "id", call_id)

        projected: dict[str, Any] = {"tool_name": str(tool_name or "unknown")}
        if isinstance(call_id, str) and call_id:
            projected["call_id"] = call_id
        step_id = event.data.get("step_id")
        if isinstance(step_id, str) and step_id:
            projected["step_id"] = step_id
        step = event.data.get("step")
        if type(step) is int and step >= 0:
            projected["step"] = step

        if event.type is EventType.TOOL_STARTED:
            arguments = getattr(call, "arguments", None)
            if isinstance(arguments, Mapping):
                try:
                    projected["arguments_fingerprint"] = fingerprint_tool_arguments(
                        arguments
                    )
                except (TypeError, ValueError):
                    pass
            return replace(event, data=projected)

        success = bool(
            event.data.get(
                "success",
                getattr(result, "success", False),
            )
        )
        projected["success"] = success
        if not success:
            safe_failure = event.data.get("failure")
            if isinstance(safe_failure, Mapping):
                projected["failure"] = {
                    key: value
                    for key, value in safe_failure.items()
                    if key
                    in {
                        "type",
                        "reason",
                        "recovery",
                        "retryable",
                        "call_id",
                        "tool_name",
                        "arguments_fingerprint",
                        "invocation_fingerprint",
                        "message",
                    }
                }
            else:
                error = getattr(result, "error", None)
                error_type = getattr(getattr(error, "type", None), "value", None)
                recovery = getattr(getattr(error, "recovery", None), "value", None)
                failure: dict[str, Any] = {
                    "type": str(error_type or ToolErrorType.EXECUTION_ERROR.value),
                    "reason": str(
                        getattr(error, "reason", None)
                        or error_type
                        or ToolErrorType.EXECUTION_ERROR.value
                    ),
                    "retryable": bool(getattr(error, "retryable", False)),
                }
                if recovery is not None:
                    failure["recovery"] = recovery
                projected["failure"] = failure
        return replace(event, data=projected)

    async def _save_engine_snapshot(
        self,
        context: EngineContext,
        snapshot: EngineSnapshot,
        *,
        timeout: float,
    ) -> None:
        store = self.runtime.checkpoint_store
        if store is None:
            return
        save_snapshot = getattr(store, "save_snapshot", None)
        if callable(save_snapshot):
            # Import lazily to keep the Engine contracts below persistence and
            # to avoid a package initialization cycle.
            from moduagent.persistence import RunCheckpoint

            compatibility_policy_state = {
                key: value
                for key, value in context.run.policy_state.items()
                if key not in _ENGINE_OWNED_POLICY_KEYS
            }
            # Build the compatibility part without re-migrating the mutable
            # legacy Plan facade. The explicit EngineSnapshot is authoritative
            # and may legitimately be ahead of that facade during repair.
            checkpoint = replace(
                RunCheckpoint.from_context(context.run),
                execution_state=None,
                policy_state=compatibility_policy_state,
                engine_id=snapshot.engine_id,
                engine_state_version=snapshot.state_version,
                engine_state=snapshot.state,
            )
            envelope = checkpoint.to_snapshot()
            common_state = replace(
                envelope.common_state,
                compatibility_policy_state=compatibility_policy_state,
            )
            agent_spec = getattr(self.runtime, "agent_spec", None)
            fingerprint = getattr(agent_spec, "agent_fingerprint", None)
            if isinstance(fingerprint, str) and fingerprint:
                envelope = replace(
                    envelope,
                    agent_fingerprint=fingerprint,
                    common_state=common_state,
                )
            else:
                envelope = replace(
                    envelope,
                    common_state=common_state,
                )
            envelope = replace(envelope, engine=snapshot)
            try:
                await asyncio.wait_for(
                    save_snapshot(envelope),
                    timeout=timeout,
                )
            except (asyncio.CancelledError, asyncio.TimeoutError):
                raise
            except PersistenceError:
                raise
            except Exception as exc:
                raise PersistenceError("checkpoint persistence failed") from exc
            return
        try:
            await asyncio.wait_for(
                store.save(context.run.run_id, context.run),
                timeout=timeout,
            )
        except (asyncio.CancelledError, asyncio.TimeoutError):
            raise
        except PersistenceError:
            raise
        except Exception as exc:
            raise PersistenceError("checkpoint persistence failed") from exc

    @staticmethod
    def _legacy_tool_outcome(
        calls: tuple[Any, ...],
        raw_results: Any,
    ) -> ToolBatchOutcome:
        error = "tool executor returned mismatched results"
        if not isinstance(raw_results, (tuple, list)):
            raise ToolInvocationError(error)
        results = tuple(raw_results)
        if len(results) != len(calls):
            raise ToolInvocationError(error)
        for call, result in zip(calls, results):
            try:
                if not isinstance(result, ToolResult):
                    raise TypeError
                # Reconstruct to re-run dataclass invariants in case a custom
                # executor mutated a frozen ToolResult with object.__setattr__.
                checked = ToolResult(
                    call_id=result.call_id,
                    tool_name=result.tool_name,
                    success=result.success,
                    value=result.value,
                    error=result.error,
                    attempts=result.attempts,
                    duration_seconds=result.duration_seconds,
                    invocation_arguments=result.invocation_arguments,
                    repair_safe=result.repair_safe,
                )
                if checked.call_id != call.id or checked.tool_name != call.name:
                    raise ValueError
                model_payload = json.loads(checked.model_content())
                if _contains_unsupported_projection(model_payload):
                    raise ValueError
                if not checked.success:
                    # The rich failure classification cannot be reconstructed
                    # safely from an arbitrary legacy executor.
                    raise ValueError
            except Exception:
                raise ToolInvocationError(error) from None
        try:
            return ToolBatchOutcome(calls=tuple(calls), results=results)
        except Exception:
            raise ToolInvocationError(error) from None

    def _validate_skill_resource_batch(
        self,
        context: EngineContext,
        calls: tuple[Any, ...],
    ) -> None:
        resource_calls = tuple(
            call for call in calls if call.name in SKILL_RESOURCE_TOOL_NAMES
        )
        if not resource_calls:
            return
        business_calls = tuple(
            call for call in calls if call.name not in SKILL_RESOURCE_TOOL_NAMES
        )
        if business_calls:
            raise SkillError(
                "a model response cannot mix Skill resource and business tools"
            )
        if self.runtime.skill_runtime is None:
            raise SkillError("Skill resource tools are not configured")
        next_reads = context.run.skill_state.resource_reads + len(resource_calls)
        if next_reads > self.runtime.skill_runtime.limits.max_resource_reads:
            raise SkillError("Skill resource read limit exceeded")
        context.run.skill_state = replace(
            context.run.skill_state,
            resource_reads=next_reads,
        )

    def _apply_skill_resource_limits(
        self,
        context: EngineContext,
        outcome: ToolBatchOutcome,
    ) -> ToolBatchOutcome:
        if self.runtime.skill_runtime is None or not any(
            call.name in SKILL_RESOURCE_TOOL_NAMES for call in outcome.calls
        ):
            return outcome
        results = list(outcome.results)
        failure_by_call = {failure.call_id: failure for failure in outcome.failures}
        view_by_call = {view.call_id: view for view in outcome.sanitized_failure_views}
        for index, (call, result) in enumerate(zip(outcome.calls, outcome.results)):
            if call.name not in SKILL_RESOURCE_TOOL_NAMES or not result.success:
                continue
            added_tokens = self.runtime._resource_tokens(result.value)
            next_tokens = context.run.skill_state.resource_tokens + added_tokens
            total_tokens = context.run.skill_state.instruction_tokens + next_tokens
            limits = self.runtime.skill_runtime.limits
            if (
                next_tokens > limits.max_resource_tokens
                or total_tokens > limits.max_total_skill_tokens
            ):
                error = ToolError(
                    ToolErrorType.RESULT_TOO_LARGE,
                    "Skill resource token budget exceeded",
                )
                rejected = ToolResult.failed(
                    call_id=call.id,
                    tool_name=call.name,
                    error=error,
                    attempts=result.attempts,
                    duration_seconds=result.duration_seconds,
                )
                results[index] = rejected
                classification = ToolFailureClassification(
                    error_type=ToolErrorType.RESULT_TOO_LARGE,
                    stable_reason=ToolErrorType.RESULT_TOO_LARGE.value,
                    safe_message=error.message,
                )
                failure = InternalToolFailure(
                    call_id=call.id,
                    tool_name=call.name,
                    classification=classification,
                    safety_profile=ToolSafetyProfile(),
                    attempts=rejected.attempts,
                )
                failure_by_call[call.id] = failure
                view_by_call[call.id] = FailureProjector().project(
                    failure,
                    include_safe_message=True,
                )
            else:
                context.run.skill_state = replace(
                    context.run.skill_state,
                    resource_tokens=next_tokens,
                )
        failures = tuple(
            failure_by_call[result.call_id] for result in results if not result.success
        )
        views = tuple(
            view_by_call[result.call_id] for result in results if not result.success
        )
        return ToolBatchOutcome(
            calls=outcome.calls,
            results=tuple(results),
            failures=failures,
            sanitized_failure_views=views,
        )

    @staticmethod
    def _skill_resource_call(event: AgentEvent) -> Any | None:
        if event.type not in {
            EventType.TOOL_STARTED,
            EventType.TOOL_COMPLETED,
        }:
            return None
        call = event.data.get("tool_call")
        if call is None or getattr(call, "name", None) not in SKILL_RESOURCE_TOOL_NAMES:
            return None
        return call


def _memory_phase(value: str) -> MemoryPhase:
    normalized = value.lower()
    aliases = {
        "act_tool": MemoryPhase.ACT,
        "tool_recovery": MemoryPhase.ACT,
        "tool_repair": MemoryPhase.ACT,
    }
    if normalized in aliases:
        return aliases[normalized]
    return MemoryPhase(normalized)


__all__ = ["RuntimeServices"]
