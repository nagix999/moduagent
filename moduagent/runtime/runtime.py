from __future__ import annotations

import asyncio
import inspect
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from typing import Any, TypeVar

from moduagent.config import AgentConfig
from moduagent.decision import DecisionKind, DecisionPolicy
from moduagent.memory import (
    ConversationMemoryPolicy,
    FullConversationMemoryPolicy,
    MemoryPhase,
    MemoryRequest,
)
from moduagent.messages import FinishReason, Message
from moduagent.models import ModelClient, ModelRequest, ModelResponse
from moduagent.observability import EventSink
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
from moduagent.runtime.events import AgentEvent, EventType
from moduagent.tools import ToolExecutionContext, ToolExecutor, ToolResult


T = TypeVar("T")

_FINALIZATION_STATE_KEY = "_moduagent_structured_finalization"
_FINALIZATION_OUTPUT_KEY = "_moduagent_structured_output"
_FINALIZATION_PENDING = "pending"
_FINALIZATION_COMPLETED = "completed"
_FINALIZATION_PROMPT = (
    "Using the preceding execution and tool results, return only the final answer "
    "that matches the required response schema. Do not call tools."
)


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
    ) -> None:
        self.config = config
        self.model = model
        self.decision_policy = decision_policy
        self.tool_executor = tool_executor
        self.conversation_store = conversation_store
        self.output_codec = output_codec
        self.event_sink = event_sink
        self.checkpoint_store = checkpoint_store
        self.conversation_memory_policy = (
            conversation_memory_policy
            if conversation_memory_policy is not None
            else FullConversationMemoryPolicy()
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

    async def stream(self, request: RunRequest) -> AsyncIterator[AgentEvent]:
        async for event in self._events(request, stream_model=True):
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
        user_message = Message.user(request.input)
        context = RunContext(
            run_id=run_id,
            request=request,
            messages=[Message.system(self.config.instructions), user_message],
            new_messages=[user_message],
            metadata={"agent": self.config.name, **dict(self.config.metadata)},
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

            context.status = RunStatus.RUNNING
            await self._within(deadline, lambda: self.decision_policy.begin(context))
            await self._save_checkpoint(context, deadline)

            tool_schemas = tuple(self.tool_executor.registry.schemas())
            output_schema = self.output_codec.schema()
            staged_finalization = bool(tool_schemas and output_schema is not None)

            if staged_finalization and context.policy_state.get(
                _FINALIZATION_STATE_KEY
            ) in (_FINALIZATION_PENDING, _FINALIZATION_COMPLETED):
                async for event in self._structured_finalization_events(
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
                        async for event in self._structured_finalization_events(
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
                        async for event in self._structured_finalization_events(
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
                if (
                    context.tool_call_count + len(calls)
                    > self.config.limits.max_tool_calls
                ):
                    for call in calls:
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
                context.tool_call_count += len(calls)
                for call in calls:
                    tool_event = AgentEvent(
                        EventType.TOOL_STARTED,
                        run_id,
                        {"tool_call": call, "tool_name": call.name},
                    )
                    await self._publish(tool_event)
                    yield tool_event

                tool_context = ToolExecutionContext(
                    run_id=run_id,
                    session_id=context.request.session_id,
                    user_context=dict(context.request.user_context),
                    metadata={"agent": self.config.name},
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
                        context.add_message(
                            Message.tool(
                                self._json({"success": False, "error": error_message}),
                                call_id=call.id,
                                name=call.name,
                            )
                        )
                        tool_event = AgentEvent(
                            EventType.TOOL_COMPLETED,
                            run_id,
                            {
                                "tool_call": call,
                                "tool_name": call.name,
                                "success": False,
                                "error": error_message,
                            },
                        )
                        await self._publish(tool_event)
                        yield tool_event
                    raise

                for call, result in zip(calls, results):
                    context.add_message(
                        Message.tool(
                            self._tool_result_content(result),
                            call_id=call.id,
                            name=call.name,
                        )
                    )
                    tool_event = AgentEvent(
                        EventType.TOOL_COMPLETED,
                        run_id,
                        {
                            "tool_call": call,
                            "tool_name": call.name,
                            "result": result,
                            "success": result.success,
                        },
                    )
                    await self._publish(tool_event)
                    yield tool_event
                await self._within(
                    deadline,
                    lambda: self.decision_policy.observe(context, results),
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
            result = self._result(
                context,
                FinishReason.ERROR,
                error=str(exc) or exc.__class__.__name__,
            )
            event = AgentEvent(EventType.RUN_FAILED, run_id, {"result": result})
            await self._publish(event)
            yield event

    async def _model_events(
        self,
        context: RunContext,
        request: ModelRequest,
        deadline: float,
        *,
        stream_model: bool,
        phase: str,
        record_response: bool = True,
    ) -> AsyncIterator[AgentEvent]:
        response: ModelResponse | None = None
        emitted_delta = False
        for attempt in range(1, self.config.retry.max_attempts + 1):
            event = AgentEvent(
                EventType.MODEL_STARTED,
                context.run_id,
                {"step": context.step, "attempt": attempt, "phase": phase},
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
                            delta_event = AgentEvent(
                                EventType.MODEL_DELTA,
                                context.run_id,
                                {
                                    "step": context.step,
                                    "phase": phase,
                                    "delta": chunk.delta,
                                    "metadata": dict(chunk.provider_metadata),
                                },
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
            context.add_message(
                Message.assistant(
                    response.message.content,
                    response.tool_calls or response.message.tool_calls,
                )
            )
        event = AgentEvent(
            EventType.MODEL_COMPLETED,
            context.run_id,
            {
                "step": context.step,
                "phase": phase,
                "response": response,
                "usage": response.usage,
            },
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
    ) -> tuple[ModelRequest, AgentEvent | None]:
        memory = await self._within(
            deadline,
            lambda: self.conversation_memory_policy.prepare(
                MemoryRequest(
                    run_id=context.run_id,
                    session_id=context.request.session_id,
                    phase=phase,
                    model_request=request,
                    protected_from=context.current_run_start,
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

    async def _structured_finalization_events(
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
                raise RuntimeError("structured finalization returned no response")
            if response.tool_calls or response.message.tool_calls:
                raise RuntimeError("structured finalization returned tool calls")
            output = self.output_codec.decode(response)
            context.policy_state[_FINALIZATION_OUTPUT_KEY] = response.message.content
            context.policy_state[_FINALIZATION_STATE_KEY] = _FINALIZATION_COMPLETED
            context.status = RunStatus.RUNNING
            await self._save_checkpoint(context, deadline)
        else:
            if _FINALIZATION_OUTPUT_KEY not in context.policy_state:
                raise RuntimeError("structured finalization response is missing")
            output = self.output_codec.decode(
                context.policy_state[_FINALIZATION_OUTPUT_KEY]
            )

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
        await self._within(
            deadline,
            lambda: self.conversation_store.append(
                context.request.session_id, tuple(context.new_messages)
            ),
        )
        context.new_messages.clear()
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
        return AgentResult(
            run_id=context.run_id,
            output=output,
            messages=tuple(context.messages),
            usage=context.usage,
            finish_reason=reason,
            error=error,
            metadata=dict(context.metadata),
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

    async def _persist_safely(self, context: RunContext) -> None:
        if not context.new_messages:
            return
        try:
            await asyncio.wait_for(
                self.conversation_store.append(
                    context.request.session_id, tuple(context.new_messages)
                ),
                timeout=1.0,
            )
            context.new_messages.clear()
        except Exception:
            pass

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
