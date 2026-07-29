from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import re
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from moduagent.errors import (
    CancellationError,
    ConfigurationError,
    CheckpointNotFoundError,
    ExecutionInvariantError,
    MemoryError as FrameworkMemoryError,
    ModelInvocationError,
    ModuAgentError,
    OutputValidationError,
    PersistenceError,
    SkillError as FrameworkSkillError,
    StateMigrationError,
    ToolAuthorizationError,
    ToolInvocationError,
    ToolRecoveryError,
    ToolValidationError,
)
from moduagent.execution.base import (
    DurableBoundary,
    EngineContext,
    EngineEmission,
    EngineOutcome,
    EngineSnapshot,
    ExecutionEngine,
)
from moduagent.execution.standard import StandardExecutionEngine
from moduagent.messages import FinishReason, Message, MessageRole
from moduagent.memory import (
    ConversationMemoryOverflowError,
    MemoryIntegrityError,
)
from moduagent.models import ModelCapabilities
from moduagent.observability._background import run_in_daemon_thread
from moduagent.observability.sinks import (
    _event_sink_is_noop,
    _event_sink_requires_coordinator_copy,
)
from moduagent.persistence import RunCheckpoint, RunSnapshot
from moduagent.runtime.context import (
    AgentResult,
    RunContext,
    RunRequest,
    RunStatus,
)
from moduagent.runtime.events import (
    AgentEvent,
    EventPublisher,
    EventType,
    EventVisibility,
)
from moduagent.runtime.metadata import is_runtime_owned_metadata_key
from moduagent.runtime.runtime import AgentRuntime
from moduagent.runtime.services import RuntimeServices
from moduagent.skills.tools import SKILL_RESOURCE_TOOL_NAMES


_RUN_ID_METADATA_KEY = "moduagent.run_id"
_ENGINE_SNAPSHOT_POLICY_KEY = "_moduagent_engine_snapshot"
_ENGINE_INITIALIZED_POLICY_KEY = "_moduagent_engine_initialized"
_TERMINAL_EVENT_TYPES = frozenset({EventType.RUN_COMPLETED, EventType.RUN_FAILED})
_ENGINE_OUTCOME_RUNTIME_METADATA_KEYS = frozenset(
    {"failure", "plan", "plan_usage", "validation_failure"}
)
_STEP_VALIDATION_CODES = frozenset(
    {
        "step_result_incomplete",
        "step_result_tool_call_forbidden",
        "step_result_tool_call_invalid",
        "step_result_required",
        "step_result_schema_invalid",
        "step_result_id_mismatch",
        "step_result_max_attempts_exceeded",
        "step_validation_state_incomplete",
        "step_validator_failed",
        "step_validation_rejected",
        "step_validation_max_attempts_exceeded",
    }
)
_STEP_VALIDATION_LOCATIONS = frozenset({"act", "step_result", "step_validator"})
_PUBLIC_STREAM_EVENT_TYPES = frozenset(
    {
        EventType.MODEL_DELTA,
        EventType.FINAL_DELTA,
        *_TERMINAL_EVENT_TYPES,
    }
)
_EVENT_SINK_TIMEOUT_SECONDS = 0.25
_EVENT_SINK_MIN_DRAIN_SECONDS = 1.0
_EVENT_SINK_MAX_DRAIN_SECONDS = 15.0
_EVENT_SINK_QUEUE_MAX_SIZE = 1_024
_DESCRIPTOR_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "authorization",
        "bearer",
        "bearer_token",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "set_cookie",
        "token",
    }
)
_DESCRIPTOR_SENSITIVE_SUFFIXES = (
    "_api_key",
    "_authorization",
    "_credential",
    "_credentials",
    "_password",
    "_private_key",
    "_refresh_token",
    "_secret",
    "_token",
)
_DESCRIPTOR_HEADER_KEYS = frozenset({"header", "headers", "http_headers"})


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    """Retrieve a detached observability task result without surfacing it."""

    if task.cancelled():
        return
    try:
        task.exception()
    except BaseException:
        pass


@dataclass(frozen=True, slots=True)
class _PublishedEventStamp:
    """Small compatibility record that never retains an event payload."""

    event_schema_version: int
    visibility: EventVisibility
    session_id: str | None
    engine_id: str | None
    sequence: int

    @classmethod
    def from_event(cls, event: AgentEvent) -> "_PublishedEventStamp":
        return cls(
            event_schema_version=event.event_schema_version,
            visibility=event.visibility,
            session_id=event.session_id,
            engine_id=event.engine_id,
            sequence=event.sequence,
        )

    def apply(self, event: AgentEvent) -> AgentEvent:
        if (
            event.event_schema_version == self.event_schema_version
            and event.visibility is self.visibility
            and event.session_id == self.session_id
            and event.engine_id == self.engine_id
            and event.sequence == self.sequence
        ):
            return event
        return replace(
            event,
            event_schema_version=self.event_schema_version,
            visibility=self.visibility,
            session_id=self.session_id,
            engine_id=self.engine_id,
            sequence=self.sequence,
        )


def _normalize_descriptor_key(value: str) -> str:
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", str(value).strip())
    return text.lower().replace("-", "_").replace(" ", "_")


def _descriptor_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _descriptor_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_descriptor_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def _redact_descriptor(value: Any, *, key: str = "") -> Any:
    normalized = _normalize_descriptor_key(key)
    sensitive = normalized in _DESCRIPTOR_SENSITIVE_KEYS or normalized.endswith(
        _DESCRIPTOR_SENSITIVE_SUFFIXES
    )
    if sensitive:
        return "[REDACTED]"
    if isinstance(value, Mapping):
        if normalized in _DESCRIPTOR_HEADER_KEYS:
            return {str(item_key): "[REDACTED]" for item_key in value}
        return {
            str(item_key): _redact_descriptor(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_descriptor(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        try:
            parsed = urlsplit(value)
        except ValueError:
            return value
        if not parsed.scheme or not parsed.netloc:
            return value
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
        netloc = f"{hostname}{port}"
        query = urlencode(
            [
                (
                    query_key,
                    (
                        "[REDACTED]"
                        if (
                            _normalize_descriptor_key(query_key)
                            in _DESCRIPTOR_SENSITIVE_KEYS
                            or _normalize_descriptor_key(query_key).endswith(
                                _DESCRIPTOR_SENSITIVE_SUFFIXES
                            )
                        )
                        else query_value
                    ),
                )
                for query_key, query_value in parse_qsl(
                    parsed.query,
                    keep_blank_values=True,
                )
            ]
        )
        return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))
    return value


class RunCoordinator(AgentRuntime):
    """Engine-neutral owner of one Agent run's common lifecycle.

    ``AgentRuntime`` remains the 0.3 compatibility base and supplies the
    established Memory, Skill, persistence, trace and result adapters.
    Execution phases and transition rules live exclusively in the injected
    :class:`ExecutionEngine`.
    """

    def __init__(
        self,
        *,
        engine: ExecutionEngine[Any] | None = None,
        resolved_spec: Mapping[str, Any] | None = None,
        **runtime_options: Any,
    ) -> None:
        super().__init__(**runtime_options)
        resolved_engine: ExecutionEngine[Any] = (
            StandardExecutionEngine(self.decision_policy) if engine is None else engine
        )
        if not isinstance(resolved_engine, ExecutionEngine):
            raise TypeError("engine must implement ExecutionEngine")
        self.engine = resolved_engine
        spec = dict(resolved_spec or {})
        configured_engine = spec.get("engine_id")
        if configured_engine not in (None, resolved_engine.engine_id):
            raise ValueError("resolved_spec engine_id does not match engine")
        spec["engine_id"] = resolved_engine.engine_id
        spec.setdefault("state_version", resolved_engine.state_version)
        self.resolved_spec = spec

        # Run IDs are globally unique. Keeping these maps run-scoped lets one
        # Agent serve different sessions concurrently without sharing sequence
        # counters or event identity.
        self._event_publishers: dict[str, EventPublisher] = {}
        self._coordinator_contexts: dict[str, RunContext] = {}
        self._published_events: dict[tuple[str, str], _PublishedEventStamp] = {}
        self._reserved_events: dict[tuple[str, str], AgentEvent] = {}
        self._sink_queues: dict[str, asyncio.Queue[AgentEvent]] = {}
        self._sink_workers: dict[str, asyncio.Task[None]] = {}

    async def _run(
        self,
        request: RunRequest,
        *,
        stream_model: bool,
    ) -> AsyncIterator[AgentEvent]:
        if not isinstance(request, RunRequest):
            raise TypeError("request must be a RunRequest")
        self._validate_engine_descriptor()
        run_id = request.resume_run_id or uuid.uuid4().hex
        deadline = (
            asyncio.get_running_loop().time() + self.config.limits.timeout_seconds
        )
        context = self._new_context(request, run_id, history=())
        resumed_snapshot: RunSnapshot | None = None
        setup_error: BaseException | None = None

        # Resume is loaded before the EventPublisher is created so the first
        # new event continues the durable monotonic sequence.
        try:
            if request.resume_run_id:
                resumed_snapshot, checkpoint = await self._load_resume(
                    request,
                    deadline,
                )
                context = checkpoint.to_context()
                self._normalize_context_tool_trace(context)
            else:
                history = await self._persistence_within(
                    deadline,
                    lambda: self.conversation_store.load(request.session_id),
                    operation="conversation",
                )
                context = self._new_context(
                    request,
                    run_id,
                    history=tuple(history),
                )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, PersistenceError):
                # A bootstrap assembled without durable history must never be
                # resumed as if the conversation had loaded successfully.
                context.metadata["_moduagent_resume_safety"] = "manual_required"
            setup_error = exc
        setup_succeeded = setup_error is None
        cleanup_writes_allowed = setup_succeeded and resumed_snapshot is None

        initial_sequence = (
            resumed_snapshot.common_state.event_sequence
            if resumed_snapshot is not None
            else int(context.metadata.get("_moduagent_event_sequence", 0) or 0)
        )
        publisher = EventPublisher(
            run_id=run_id,
            session_id=request.session_id,
            engine_id=self.engine.engine_id,
            initial_sequence=initial_sequence,
        )
        self._event_publishers[run_id] = publisher
        self._coordinator_contexts[run_id] = context
        context.diagnostic_reporter = self.diagnostic_reporter
        context.metadata["_moduagent_engine_id"] = self.engine.engine_id
        context.metadata["_moduagent_engine_state_version"] = self.engine.state_version
        context.metadata["_moduagent_event_sequence"] = initial_sequence
        self._attach_agent_fingerprint(context)

        services = RuntimeServices(self, deadline)
        engine_context: EngineContext | None = None
        state: Any = None

        try:
            started = await self._publish(
                AgentEvent(
                    EventType.RUN_STARTED,
                    run_id,
                    {
                        "agent": self.config.name,
                        "session_id": request.session_id,
                        "user_context": dict(request.user_context),
                        "queue_wait_seconds": self._session_queue_wait_seconds(),
                    },
                )
            )
            yield started

            engine_context = EngineContext(
                run=context,
                config=self.config,
                stream_model=stream_model,
                resolved_spec=self._engine_spec(),
                model_capabilities=self._model_capabilities(),
            )
            services.bind(engine_context)
            if setup_error is not None:
                raise setup_error

            # A Skill failure may happen before the Engine can create its first
            # snapshot. Persist an explicit bootstrap marker so a later resume
            # retries initialization instead of decoding the compatibility
            # checkpoint as an initialized Engine state.
            if resumed_snapshot is None:
                context.policy_state.setdefault(
                    _ENGINE_INITIALIZED_POLICY_KEY,
                    False,
                )

            if resumed_snapshot is not None:
                loaded = await self._publish(
                    AgentEvent(
                        EventType.CHECKPOINT_LOADED,
                        run_id,
                        {
                            "step": context.step,
                            "status": context.status.value,
                            "state_version": (resumed_snapshot.engine.state_version),
                        },
                    )
                )
                yield loaded

            async for skill_event in self._skill_events(
                context,
                deadline,
                resumed=resumed_snapshot is not None,
            ):
                for pending in services.drain_events():
                    yield pending
                yield self._published_event(skill_event)
            for pending in services.drain_events():
                yield pending

            context.status = RunStatus.RUNNING
            needs_initialization = (
                resumed_snapshot is None
                or context.policy_state.get(_ENGINE_INITIALIZED_POLICY_KEY) is False
            )
            if needs_initialization:
                context.policy_state[_ENGINE_INITIALIZED_POLICY_KEY] = True
                try:
                    state = await self.engine.initialize(
                        engine_context,
                        services,
                    )
                except BaseException:
                    context.policy_state[_ENGINE_INITIALIZED_POLICY_KEY] = False
                    raise
            else:
                state = self._resume_engine_state(resumed_snapshot)
            cleanup_writes_allowed = True

            for pending in services.drain_events():
                yield pending

            outcome: EngineOutcome | None = None
            iterator = self.engine.execute(
                engine_context,
                state,
                services,
            ).__aiter__()
            next_emission: asyncio.Task[EngineEmission] | None = None
            event_ready: asyncio.Task[None] | None = None
            try:
                next_emission = asyncio.create_task(iterator.__anext__())
                while True:
                    for pending in services.drain_events():
                        yield pending

                    event_ready = asyncio.create_task(services.wait_for_events())
                    done, _ = await asyncio.wait(
                        (next_emission, event_ready),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if event_ready not in done:
                        event_ready.cancel()
                        with suppress(asyncio.CancelledError):
                            await event_ready
                    else:
                        await event_ready
                    event_ready = None

                    for pending in services.drain_events():
                        yield pending
                    if not next_emission.done():
                        continue

                    try:
                        emission = next_emission.result()
                    except StopAsyncIteration:
                        break
                    except BaseException:
                        for pending in services.drain_after_events():
                            yield pending
                        raise
                    next_emission = None

                    if not isinstance(emission, EngineEmission):
                        raise TypeError("ExecutionEngine must yield EngineEmission")
                    if emission.event is not None:
                        if outcome is not None:
                            raise ExecutionInvariantError(
                                "ExecutionEngine emitted an event after its outcome"
                            )
                        if emission.event.type in _TERMINAL_EVENT_TYPES:
                            raise ExecutionInvariantError(
                                "terminal events are owned by RunCoordinator"
                            )
                        if emission.event.run_id != run_id:
                            raise ExecutionInvariantError(
                                "ExecutionEngine emitted an event for another run"
                            )
                        published_key = (run_id, emission.event.event_id)
                        if published_key not in self._published_events:
                            published = await self._publish(emission.event)
                        else:
                            published = self._published_event(emission.event)
                        yield published
                        for pending in services.drain_after_events():
                            yield pending
                    else:
                        if outcome is not None:
                            raise ExecutionInvariantError(
                                "ExecutionEngine emitted more than one outcome"
                            )
                        outcome = emission.outcome
                    next_emission = asyncio.create_task(iterator.__anext__())
            finally:
                if event_ready is not None and not event_ready.done():
                    event_ready.cancel()
                    with suppress(asyncio.CancelledError):
                        await event_ready
                if next_emission is not None and not next_emission.done():
                    next_emission.cancel()
                    with suppress(asyncio.CancelledError):
                        await next_emission
                close_iterator = getattr(iterator, "aclose", None)
                if callable(close_iterator):
                    with suppress(Exception):
                        await close_iterator()

            for pending in services.drain_events():
                yield pending
            for pending in services.drain_after_events():
                yield pending
            if outcome is None:
                raise ExecutionInvariantError(
                    "ExecutionEngine ended without a terminal outcome"
                )

            result, event_type = await self._finish_outcome(
                engine_context,
                services,
                state,
                outcome,
                deadline,
            )
            await self._flush_diagnostics_safely(run_id)
            terminal = await self._publish_terminal(
                AgentEvent(event_type, run_id, {"result": result})
            )
            if event_type is EventType.RUN_FAILED or self._retain_terminal_checkpoint():
                await self._checkpoint_state_safely(
                    engine_context,
                    services,
                    state,
                )
            yield terminal
        except GeneratorExit:
            context.status = RunStatus.CANCELLED
            context.metadata["_moduagent_terminal_reason"] = (
                FinishReason.CANCELLED.value
            )
            self._normalize_skill_resource_messages(context)
            if cleanup_writes_allowed:
                await asyncio.shield(self._persist_safely(context))
                if engine_context is not None:
                    await asyncio.shield(
                        self._checkpoint_state_safely(
                            engine_context,
                            services,
                            state,
                        )
                    )
            raise
        except asyncio.CancelledError:
            context.status = RunStatus.CANCELLED
            context.metadata["_moduagent_terminal_reason"] = (
                FinishReason.CANCELLED.value
            )
            self._normalize_skill_resource_messages(context)
            if cleanup_writes_allowed:
                await self._persist_safely(context)
                if engine_context is not None:
                    await self._checkpoint_state_safely(
                        engine_context,
                        services,
                        state,
                    )
            raise
        except asyncio.TimeoutError as exc:
            context.status = RunStatus.FAILED
            context.metadata["_moduagent_terminal_reason"] = FinishReason.TIMEOUT.value
            diagnostic = await self._capture_terminal_failure(context, exc)
            self._normalize_skill_resource_messages(context)
            if cleanup_writes_allowed:
                await self._persist_safely(context)
                if engine_context is not None:
                    await self._checkpoint_state_safely(
                        engine_context,
                        services,
                        state,
                    )
            result = self._result(
                context,
                FinishReason.TIMEOUT,
                error="run timed out",
            )
            summary = {
                "category": "timeout",
                "code": "run_timeout",
                "retryable": True,
                "resumable": self._is_safely_resumable(context),
            }
            summary.update(diagnostic)
            result = self._with_error_summary(result, summary)
            for pending in services.drain_events():
                yield pending
            for pending in services.drain_after_events():
                yield pending
            await self._flush_diagnostics_safely(run_id)
            terminal = await self._publish_terminal(
                AgentEvent(EventType.RUN_FAILED, run_id, {"result": result})
            )
            if cleanup_writes_allowed and engine_context is not None:
                await self._checkpoint_state_safely(
                    engine_context,
                    services,
                    state,
                )
            yield terminal
        except Exception as exc:
            context.status = RunStatus.FAILED
            context.metadata["_moduagent_terminal_reason"] = FinishReason.ERROR.value
            diagnostic = await self._capture_terminal_failure(context, exc)
            self._normalize_skill_resource_messages(context)
            if cleanup_writes_allowed:
                await self._persist_safely(context)
                if engine_context is not None:
                    await self._checkpoint_state_safely(
                        engine_context,
                        services,
                        state,
                    )
            result = self._result(
                context,
                FinishReason.ERROR,
                error=self._public_error(exc),
            )
            summary = dict(self._error_summary(exc, context=context))
            summary.update(diagnostic)
            result = self._with_error_summary(result, summary)
            for pending in services.drain_events():
                yield pending
            for pending in services.drain_after_events():
                yield pending
            await self._flush_diagnostics_safely(run_id)
            terminal = await self._publish_terminal(
                AgentEvent(EventType.RUN_FAILED, run_id, {"result": result})
            )
            if cleanup_writes_allowed and engine_context is not None:
                await self._checkpoint_state_safely(
                    engine_context,
                    services,
                    state,
                )
            yield terminal
        finally:
            await self._close_sink_worker(run_id)
            await self._flush_diagnostics_safely(run_id)
            clear_diagnostics = getattr(
                self.diagnostic_reporter,
                "clear_run",
                None,
            )
            if callable(clear_diagnostics):
                try:
                    cleared = clear_diagnostics(run_id)
                    if inspect.isawaitable(cleared):
                        await cleared
                except Exception:
                    pass
            self._event_publishers.pop(run_id, None)
            self._coordinator_contexts.pop(run_id, None)
            stale = [key for key in self._published_events if key[0] == run_id]
            for key in stale:
                self._published_events.pop(key, None)
            stale_reserved = [key for key in self._reserved_events if key[0] == run_id]
            for key in stale_reserved:
                self._reserved_events.pop(key, None)

    async def _flush_diagnostics_safely(self, run_id: str) -> None:
        flush_diagnostics = getattr(
            self.diagnostic_reporter,
            "flush_run",
            None,
        )
        if not callable(flush_diagnostics):
            return
        try:
            flushed = flush_diagnostics(run_id)
            if inspect.isawaitable(flushed):
                await flushed
        except Exception:
            pass

    async def _load_resume(
        self,
        request: RunRequest,
        deadline: float,
    ) -> tuple[RunSnapshot, RunCheckpoint]:
        store = self.checkpoint_store
        if store is None:
            raise RuntimeError("checkpoint_store is required to resume a run")
        run_id = request.resume_run_id
        if run_id is None:
            raise ValueError("resume_run_id is required")

        load_snapshot = getattr(store, "load_snapshot", None)
        if callable(load_snapshot):
            snapshot = await self._persistence_within(
                deadline,
                lambda: load_snapshot(run_id),
                operation="checkpoint",
            )
            if snapshot is None:
                raise CheckpointNotFoundError("checkpoint not found")
            if not isinstance(snapshot, RunSnapshot):
                raise TypeError("load_snapshot() must return RunSnapshot or None")
            checkpoint = RunCheckpoint.from_snapshot(snapshot)
        else:
            checkpoint = await self._persistence_within(
                deadline,
                lambda: store.load(run_id),
                operation="checkpoint",
            )
            if checkpoint is None:
                raise CheckpointNotFoundError("checkpoint not found")
            if not isinstance(checkpoint, RunCheckpoint):
                raise TypeError("checkpoint load() must return RunCheckpoint or None")
            snapshot = checkpoint.to_snapshot()

        if checkpoint.session_id != request.session_id:
            raise StateMigrationError(
                "checkpoint session_id does not match the request"
            )
        if snapshot.engine.engine_id != self.engine.engine_id:
            raise StateMigrationError(
                "checkpoint engine does not match the configured engine"
            )
        if snapshot.common_state.resume_safety not in {
            "resumable",
            "terminal",
        }:
            raise StateMigrationError(
                "checkpoint is not safely resumable: "
                f"{snapshot.common_state.resume_safety}"
            )
        current_fingerprint = self._agent_fingerprint()
        if (
            snapshot.agent_fingerprint != "legacy-unbound"
            and current_fingerprint is not None
            and snapshot.agent_fingerprint != current_fingerprint
        ):
            raise StateMigrationError(
                "checkpoint Agent fingerprint does not match configuration"
            )
        return snapshot, checkpoint

    def _resume_engine_state(self, snapshot: RunSnapshot) -> Any:
        resolved_spec = dict(self._engine_spec())
        resolved_spec["common_state"] = {
            "step": snapshot.common_state.step,
            "tool_call_count": snapshot.common_state.tool_call_count,
        }
        validation = self.engine.validate_resume(
            snapshot.engine,
            resolved_spec,
        )
        if not validation.compatible:
            raise StateMigrationError(
                "checkpoint cannot be resumed: " + validation.reason
            )
        payload: Mapping[str, Any] = snapshot.engine.state
        try:
            if snapshot.engine.state_version != self.engine.state_version:
                payload = self.engine.migrate_state(
                    snapshot.engine.state_version,
                    payload,
                )
            return self.engine.decode_state(payload)
        except StateMigrationError:
            raise
        except Exception as exc:
            raise StateMigrationError(
                "checkpoint Engine state cannot be decoded"
            ) from exc

    async def _finish_outcome(
        self,
        context: EngineContext,
        services: RuntimeServices,
        state: Any,
        outcome: EngineOutcome,
        deadline: float,
    ) -> tuple[AgentResult, EventType]:
        outcome = replace(
            outcome,
            metadata=self._project_outcome_metadata(outcome.metadata),
        )
        failed = outcome.finish_reason in {
            FinishReason.ERROR,
            FinishReason.TIMEOUT,
            FinishReason.CANCELLED,
            FinishReason.MAX_STEPS,
            FinishReason.MAX_TOOL_CALLS,
        }
        context.run.status = (
            RunStatus.CANCELLED
            if outcome.finish_reason is FinishReason.CANCELLED
            else RunStatus.FAILED
            if failed
            else RunStatus.COMPLETED
        )
        context.run.metadata["_moduagent_terminal_reason"] = outcome.finish_reason.value
        if outcome.metadata:
            context.run.metadata.update(dict(outcome.metadata))

        self._normalize_skill_resource_messages(context.run)
        await self._persist_pending_messages(context.run, deadline)
        if failed or self._retain_terminal_checkpoint():
            await self._checkpoint_state_safely(
                context,
                services,
                state,
            )
        elif self.checkpoint_store is not None:
            await self._persistence_within(
                deadline,
                lambda: self.checkpoint_store.delete(context.run.run_id),
                operation="checkpoint",
            )

        base = self._result(
            context.run,
            outcome.finish_reason,
            output=outcome.output,
            error=outcome.error,
        )
        metadata = dict(base.metadata)
        metadata.update(dict(outcome.metadata))
        if failed:
            summary = dict(self._outcome_error_summary(outcome, context=context.run))
            primary_failure = context.run.primary_failure
            if isinstance(primary_failure, Mapping):
                summary.update(dict(primary_failure))
            else:
                tool_failure = outcome.metadata.get("failure")
                if isinstance(tool_failure, Mapping):
                    call_id = tool_failure.get("call_id")
                    failure_id = (
                        context.run.tool_failure_ids.get(call_id)
                        if isinstance(call_id, str)
                        else None
                    )
                    if failure_id:
                        phase, step_id, attempt = self._diagnostic_location(context.run)
                        summary.update(
                            {
                                "failure_id": failure_id,
                                "component": "tool",
                                "operation": "invoke",
                                **({} if phase is None else {"phase": phase}),
                                **({} if step_id is None else {"step_id": step_id}),
                                **({} if attempt is None else {"attempt": attempt}),
                            }
                        )
            metadata["error_summary"] = summary
        result = (
            base
            if metadata == dict(base.metadata)
            else replace(base, metadata=metadata)
        )
        return (
            result,
            EventType.RUN_FAILED if failed else EventType.RUN_COMPLETED,
        )

    @staticmethod
    def _project_outcome_metadata(
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        projected: dict[str, Any] = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                continue
            if (
                is_runtime_owned_metadata_key(key)
                and key not in _ENGINE_OUTCOME_RUNTIME_METADATA_KEYS
            ):
                continue
            if key == "validation_failure":
                safe_validation = RunCoordinator._project_validation_failure(value)
                if safe_validation is not None:
                    projected[key] = safe_validation
                continue
            projected[key] = value
        return projected

    @staticmethod
    def _project_validation_failure(value: Any) -> Mapping[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        code = value.get("code")
        location = value.get("location")
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
        }
        cause_code = value.get("cause_code")
        if (
            isinstance(cause_code, str)
            and cause_code in _STEP_VALIDATION_CODES
            and cause_code != code
        ):
            projected["cause_code"] = cause_code
        phase = value.get("phase")
        if isinstance(phase, str) and phase in {
            "act_tool",
            "failed",
            "step_result",
            "step_validate",
        }:
            projected["phase"] = phase
        step_id = value.get("step_id")
        if (
            isinstance(step_id, str)
            and 0 < len(step_id) <= 256
            and all(character.isprintable() for character in step_id)
        ):
            projected["step_id"] = step_id
        attempt = value.get("attempt")
        if type(attempt) is int and 0 <= attempt <= 1_000_000:
            projected["attempt"] = attempt
        return projected

    async def _checkpoint_state_safely(
        self,
        context: EngineContext,
        services: RuntimeServices,
        state: Any,
    ) -> None:
        if self.checkpoint_store is None:
            return
        if state is None:
            # Skill selection can fail before an Engine has initialized. The
            # empty payload is never decoded: the common bootstrap marker makes
            # resume run initialize() again. Keeping the configured Engine ID
            # prevents a Plan run from being misclassified as Standard.
            await services.checkpoint_safely(
                context,
                EngineSnapshot(
                    engine_id=self.engine.engine_id,
                    state_version=self.engine.state_version,
                    state={},
                ),
                boundary=DurableBoundary.INTERRUPTED,
            )
            return
        try:
            snapshot = EngineSnapshot(
                engine_id=self.engine.engine_id,
                state_version=self.engine.state_version,
                state=self.engine.encode_state(state),
            )
        except Exception:
            return
        await services.checkpoint_safely(
            context,
            snapshot,
            boundary=DurableBoundary.INTERRUPTED,
        )

    def _new_context(
        self,
        request: RunRequest,
        run_id: str,
        *,
        history: tuple[Message, ...],
    ) -> RunContext:
        user_message = Message.user(
            request.input,
            metadata=(
                {
                    _RUN_ID_METADATA_KEY: run_id,
                    "moduagent.public_input": True,
                }
                if self._retain_terminal_checkpoint()
                else None
            ),
        )
        metadata = {
            "agent": self.config.name,
            "_moduagent_session_queue_wait_seconds": (
                self._session_queue_wait_seconds()
            ),
            **{
                key: value
                for key, value in self.config.metadata.items()
                if not is_runtime_owned_metadata_key(key)
            },
        }
        context = RunContext(
            run_id=run_id,
            request=request,
            messages=[
                Message.system(self.config.instructions),
                *history,
                user_message,
            ],
            new_messages=[user_message],
            metadata=metadata,
            current_run_start=1 + len(history),
        )
        return context

    def _engine_spec(self) -> Mapping[str, Any]:
        spec = dict(self.resolved_spec)
        spec["engine_id"] = self.engine.engine_id
        spec["state_version"] = self.engine.state_version
        fingerprint = self._agent_fingerprint()
        if fingerprint is not None:
            spec["agent_fingerprint"] = fingerprint
        return spec

    def _model_capabilities(self) -> ModelCapabilities:
        capabilities = getattr(self.model, "capabilities", None)
        if capabilities is None:
            return ModelCapabilities()
        if not isinstance(capabilities, ModelCapabilities):
            raise ConfigurationError("model capabilities are invalid")
        return capabilities

    def _validate_engine_descriptor(self) -> None:
        if self.engine.engine_id != self.resolved_spec.get("engine_id"):
            raise ConfigurationError(
                "execution Engine ID changed after Agent composition"
            )
        if self.engine.state_version != self.resolved_spec.get("state_version"):
            raise ConfigurationError(
                "execution Engine state version changed after Agent composition"
            )
        if self.engine.state_codec.engine_id != self.engine.engine_id:
            raise ConfigurationError(
                "execution Engine codec ID no longer matches the Engine"
            )
        if self.engine.state_codec.state_version != self.engine.state_version:
            raise ConfigurationError(
                "execution Engine codec version no longer matches the Engine"
            )
        details = self.resolved_spec.get("details", {})
        if not isinstance(details, Mapping):
            return
        expected_fingerprint = details.get("configuration_fingerprint")
        if isinstance(expected_fingerprint, str):
            encoded = json.dumps(
                _redact_descriptor(_descriptor_plain(self.engine.configuration)),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
            current_fingerprint = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
            if current_fingerprint != expected_fingerprint:
                raise ConfigurationError(
                    "execution Engine configuration changed after Agent composition"
                )
        expected_requirements = details.get("required_capabilities")
        if isinstance(expected_requirements, Mapping) and dict(
            self.engine.required_capabilities
        ) != dict(expected_requirements):
            raise ConfigurationError(
                "execution Engine capability requirements changed after composition"
            )

    def _retain_terminal_checkpoint(self) -> bool:
        return bool(self.resolved_spec.get("retain_terminal_checkpoint", False))

    @staticmethod
    def _public_error(error: Exception) -> str:
        """Project an exception to the terminal result without raw provider data."""

        if isinstance(error, StateMigrationError):
            return "checkpoint state migration failed"
        if isinstance(error, CheckpointNotFoundError):
            return "checkpoint not found"
        if isinstance(error, PersistenceError):
            return "persistence operation failed"
        if isinstance(error, OutputValidationError):
            # Keep the 0.4.0 public string stable; structured diagnostics carry
            # the more precise output-validation classification.
            return "run failed"
        if isinstance(error, FrameworkMemoryError) and not isinstance(
            error,
            (ConversationMemoryOverflowError, MemoryIntegrityError),
        ):
            return "conversation memory preparation failed"
        if isinstance(error, ModuAgentError):
            message = " ".join(
                "".join(
                    character if character.isprintable() else " "
                    for character in str(error)
                ).split()
            )
            if message:
                return message[:512]
        return "run failed"

    def _error_summary(
        self,
        error: Exception,
        *,
        context: RunContext | None = None,
    ) -> Mapping[str, Any]:
        category = "execution"
        code = "run_failed"
        retryable = False
        if isinstance(error, CheckpointNotFoundError):
            category, code = "persistence", "checkpoint_not_found"
        elif isinstance(error, StateMigrationError):
            category, code = "state_migration", "checkpoint_migration_failed"
        elif isinstance(error, PersistenceError):
            category, code = "persistence", "persistence_failed"
            retryable = True
        elif isinstance(error, ModelInvocationError):
            category, code = "model_invocation", "model_invocation_failed"
            retryable = True
        elif isinstance(error, OutputValidationError):
            category, code = "output_validation", "output_validation_failed"
        elif isinstance(error, ToolAuthorizationError):
            category, code = "tool_authorization", "tool_authorization_failed"
        elif isinstance(error, ToolValidationError):
            category, code = "tool_validation", "tool_validation_failed"
        elif isinstance(error, ToolRecoveryError):
            category, code = "tool_recovery", "tool_recovery_failed"
        elif isinstance(error, ToolInvocationError):
            category, code = "tool_invocation", "tool_invocation_failed"
            retryable = True
        elif isinstance(error, FrameworkMemoryError):
            category, code = "memory", "memory_preparation_failed"
        elif isinstance(error, FrameworkSkillError):
            category, code = "skill", "skill_activation_failed"
        elif isinstance(error, ConfigurationError):
            category, code = "configuration", "invalid_configuration"
        elif isinstance(error, ExecutionInvariantError):
            category, code = "execution_invariant", "execution_invariant_failed"
        elif isinstance(error, CancellationError):
            category, code = "cancellation", "run_cancelled"
        resumable = (
            False
            if isinstance(error, (CheckpointNotFoundError, StateMigrationError))
            else self._is_safely_resumable(context)
        )
        return {
            "category": category,
            "code": code,
            "retryable": retryable,
            "resumable": resumable,
        }

    async def _capture_terminal_failure(
        self,
        context: RunContext,
        error: BaseException,
    ) -> Mapping[str, Any]:
        existing = context.primary_failure
        if (
            isinstance(existing, Mapping)
            and isinstance(existing.get("failure_id"), str)
            and existing["failure_id"]
        ):
            return dict(existing)

        reporter = self.diagnostic_reporter
        capture = getattr(reporter, "capture_exception", None)
        if not callable(capture):
            return {}

        component, operation = self._diagnostic_operation(error)
        phase, step_id, attempt = self._diagnostic_location(context)
        summary = dict(self._error_summary_for_diagnostic(error, context))
        capture_options: dict[str, Any] = {}
        if isinstance(error, OutputValidationError):
            validation_fields = self._output_validation_fields()
            if validation_fields:
                capture_options["validation_fields"] = validation_fields
        try:
            failure_id = await capture(
                exception=error,
                run_id=context.run_id,
                component=component,
                operation=operation,
                phase=phase,
                step_id=step_id,
                attempt=attempt,
                category=str(summary["category"]),
                code=str(summary["code"]),
                retryable=bool(summary["retryable"]),
                terminal=True,
                **capture_options,
            )
        except Exception:
            return {}
        if not isinstance(failure_id, str) or not failure_id:
            return {}

        return {
            "failure_id": failure_id,
            "component": component,
            "operation": operation,
            **({} if phase is None else {"phase": phase}),
            **({} if step_id is None else {"step_id": step_id}),
            **({} if attempt is None else {"attempt": attempt}),
        }

    def _output_validation_fields(self) -> frozenset[str]:
        try:
            schema = self.output_codec.schema()
        except Exception:
            return frozenset()
        if not isinstance(schema, Mapping):
            return frozenset()

        fields: set[str] = set()
        pending: list[Any] = [schema]
        seen: set[int] = set()
        while pending and len(seen) < 1024 and len(fields) < 1024:
            value = pending.pop()
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            if isinstance(value, Mapping):
                properties = value.get("properties")
                if isinstance(properties, Mapping):
                    fields.update(
                        str(name)[:256] for name in properties if isinstance(name, str)
                    )
                pending.extend(value.values())
            elif isinstance(value, (list, tuple)):
                pending.extend(value)
        return frozenset(fields)

    def _error_summary_for_diagnostic(
        self,
        error: BaseException,
        context: RunContext,
    ) -> Mapping[str, Any]:
        if isinstance(error, asyncio.TimeoutError):
            return {
                "category": "timeout",
                "code": "run_timeout",
                "retryable": True,
            }
        if isinstance(error, Exception):
            return self._error_summary(error, context=context)
        return {
            "category": "execution",
            "code": "run_failed",
            "retryable": False,
        }

    @staticmethod
    def _diagnostic_operation(error: BaseException) -> tuple[str, str]:
        if isinstance(error, ModelInvocationError):
            return "model", "invoke"
        if isinstance(error, OutputValidationError):
            return "output", "decode"
        if isinstance(
            error,
            (ToolInvocationError, ToolValidationError, ToolAuthorizationError),
        ):
            return "tool", "invoke"
        if isinstance(error, PersistenceError):
            return "persistence", "operation"
        if isinstance(error, FrameworkMemoryError):
            return "memory", "prepare"
        if isinstance(error, FrameworkSkillError):
            return "skill", "activate"
        if isinstance(error, ConfigurationError):
            return "configuration", "validate"
        if isinstance(error, asyncio.TimeoutError):
            return "runtime", "deadline"
        return "runtime", "execute"

    @staticmethod
    def _diagnostic_location(
        context: RunContext,
    ) -> tuple[str | None, str | None, int | None]:
        state = context.execution_state
        phase_value = getattr(state, "phase", None)
        phase = getattr(phase_value, "value", phase_value)
        if not isinstance(phase, str) or not phase:
            phase = None

        step_id = getattr(state, "current_step_id", None)
        attempt = None
        step_execution = getattr(state, "step_execution", None)
        if step_execution is not None:
            step_id = getattr(step_execution, "current_step_id", step_id)
            attempt = getattr(step_execution, "step_attempt_count", None)
        if not isinstance(step_id, str) or not step_id:
            step_id = None
        if type(attempt) is not int or attempt < 1:
            attempt = None
        return phase, step_id, attempt

    def _outcome_error_summary(
        self,
        outcome: EngineOutcome,
        *,
        context: RunContext | None = None,
    ) -> Mapping[str, Any]:
        codes = {
            FinishReason.TIMEOUT: ("timeout", "run_timeout", True),
            FinishReason.CANCELLED: ("cancellation", "run_cancelled", False),
            FinishReason.MAX_STEPS: ("limit", "max_steps_exceeded", False),
            FinishReason.MAX_TOOL_CALLS: (
                "limit",
                "max_tool_calls_exceeded",
                False,
            ),
            FinishReason.ERROR: ("execution", "execution_failed", False),
        }
        category, code, retryable = codes.get(
            outcome.finish_reason,
            ("execution", "execution_failed", False),
        )
        validation_failure = outcome.metadata.get("validation_failure")
        if isinstance(validation_failure, Mapping):
            validation_code = validation_failure.get("code")
            validation_location = validation_failure.get("location")
            if (
                isinstance(validation_code, str)
                and validation_code in _STEP_VALIDATION_CODES
                and isinstance(validation_location, str)
                and validation_location in _STEP_VALIDATION_LOCATIONS
            ):
                summary: dict[str, Any] = {
                    "category": "step_validation",
                    "code": validation_code,
                    "component": "policy",
                    "operation": validation_location,
                    "retryable": False,
                    "resumable": self._is_safely_resumable(context),
                }
                phase = validation_failure.get("phase")
                if isinstance(phase, str) and phase in {
                    "act_tool",
                    "failed",
                    "step_result",
                    "step_validate",
                }:
                    summary["phase"] = phase
                step_id = validation_failure.get("step_id")
                if isinstance(step_id, str) and 0 < len(step_id) <= 256:
                    summary["step_id"] = step_id
                attempt = validation_failure.get("attempt")
                if type(attempt) is int and 0 <= attempt <= 1_000_000:
                    summary["attempt"] = attempt
                return summary
        failure = outcome.metadata.get("failure")
        if isinstance(failure, Mapping):
            category = "tool_recovery"
            reason = failure.get("reason")
            if isinstance(reason, str) and reason:
                code = reason[:128]
            retryable = bool(failure.get("retryable", False))
        return {
            "category": category,
            "code": code,
            "retryable": retryable,
            "resumable": self._is_safely_resumable(context),
        }

    def _is_safely_resumable(self, context: RunContext | None) -> bool:
        if self.checkpoint_store is None:
            return False
        if context is None:
            return True
        safety = str(context.metadata.get("_moduagent_resume_safety", "resumable"))
        return safety in {"resumable", "terminal"}

    @staticmethod
    def _with_error_summary(
        result: AgentResult,
        summary: Mapping[str, Any],
    ) -> AgentResult:
        return replace(
            result,
            metadata={
                **dict(result.metadata),
                "error_summary": dict(summary),
            },
        )

    def _agent_fingerprint(self) -> str | None:
        value = getattr(
            getattr(self, "agent_spec", None),
            "agent_fingerprint",
            None,
        )
        return value if isinstance(value, str) and value else None

    def _attach_agent_fingerprint(self, context: RunContext) -> None:
        fingerprint = self._agent_fingerprint()
        if fingerprint is not None:
            context.metadata["_moduagent_agent_fingerprint"] = fingerprint

    @staticmethod
    def _normalize_skill_resource_messages(context: RunContext) -> None:
        replacements: dict[int, Message] = {}
        normalized: list[Message] = []
        for message in context.messages:
            ephemeral = (
                message.role is MessageRole.TOOL
                and message.name in SKILL_RESOURCE_TOOL_NAMES
            ) or (
                message.role is MessageRole.ASSISTANT
                and any(
                    call.name in SKILL_RESOURCE_TOOL_NAMES
                    for call in message.tool_calls
                )
            )
            if not ephemeral:
                normalized.append(message)
                continue
            replacement = replace(
                message,
                metadata={
                    **dict(message.metadata),
                    "moduagent.ephemeral": True,
                },
            )
            replacements[id(message)] = replacement
            normalized.append(replacement)
        if not replacements:
            return
        context.messages[:] = normalized
        context.new_messages[:] = [
            message
            for message in context.new_messages
            if id(message) not in replacements
        ]

    async def _publish(self, event: AgentEvent) -> AgentEvent:
        """Publish a non-terminal event through the run-scoped envelope."""

        if not isinstance(event, AgentEvent):
            raise TypeError("event must be an AgentEvent")
        if event.type in _TERMINAL_EVENT_TYPES:
            raise ExecutionInvariantError("terminal events are owned by RunCoordinator")
        return await self._publish_event(event)

    async def _publish_terminal(self, event: AgentEvent) -> AgentEvent:
        """Publish the single terminal event from the Coordinator-owned path."""

        if not isinstance(event, AgentEvent):
            raise TypeError("event must be an AgentEvent")
        if event.type not in _TERMINAL_EVENT_TYPES:
            raise ValueError("_publish_terminal requires a terminal event")
        return await self._publish_event(event)

    async def _publish_event(self, event: AgentEvent) -> AgentEvent:
        """Stamp once, isolate sink failures, and return the published object."""

        published = self._reserve_event(event)
        return await self._dispatch_reserved_event(published)

    def _reserve_event(self, event: AgentEvent) -> AgentEvent:
        """Allocate an event sequence before a related durable write."""

        if not isinstance(event, AgentEvent):
            raise TypeError("event must be an AgentEvent")
        if event.type not in _PUBLIC_STREAM_EVENT_TYPES:
            event = replace(event, visibility=EventVisibility.INTERNAL)
        publisher = self._event_publishers.get(event.run_id)
        if publisher is None:
            raise ExecutionInvariantError(
                "event does not belong to an active Coordinator run"
            )
        published = publisher.stamp(event)
        context = self._coordinator_contexts.get(event.run_id)
        if context is not None:
            context.metadata["_moduagent_event_sequence"] = published.sequence
        key = (published.run_id, published.event_id)
        if key in self._reserved_events or key in self._published_events:
            raise ExecutionInvariantError("event identity was already published")
        self._reserved_events[key] = published
        return published

    async def _dispatch_reserved_event(self, event: AgentEvent) -> AgentEvent:
        """Dispatch one previously stamped event without advancing its sequence."""

        key = (event.run_id, event.event_id)
        reserved = self._reserved_events.pop(key, None)
        if reserved is None or reserved != event:
            raise ExecutionInvariantError("event was not reserved by this Coordinator")
        queue = self._sink_queues.get(event.run_id)
        self._published_events[key] = _PublishedEventStamp.from_event(event)
        if _event_sink_is_noop(self.event_sink):
            return event
        if queue is None:
            queue = asyncio.Queue(maxsize=_EVENT_SINK_QUEUE_MAX_SIZE)
            self._sink_queues[event.run_id] = queue
            self._sink_workers[event.run_id] = asyncio.create_task(
                self._sink_worker(queue)
            )
        # A bounded queue prevents a slow external sink from retaining an
        # unlimited number of large terminal/delta payloads. Backpressure is
        # charged only after a full 1,024-event burst; Noop sinks never enter
        # this path.
        await queue.put(event)
        # Start the ordered worker without charging sink latency to the run.
        await asyncio.sleep(0)
        if event.type in _TERMINAL_EVENT_TYPES:
            pending_count = queue.qsize() + 1
            drain_timeout = min(
                _EVENT_SINK_MAX_DRAIN_SECONDS,
                max(
                    _EVENT_SINK_MIN_DRAIN_SECONDS,
                    pending_count * _EVENT_SINK_TIMEOUT_SECONDS + 0.5,
                ),
            )
            try:
                await asyncio.wait_for(
                    queue.join(),
                    timeout=drain_timeout,
                )
            except asyncio.TimeoutError:
                pass
        return event

    async def _sink_worker(
        self,
        queue: asyncio.Queue[AgentEvent],
    ) -> None:
        timed_out = False
        while True:
            event = await queue.get()
            invocation: asyncio.Task[None] | None = None
            try:
                if not timed_out:
                    invocation = asyncio.create_task(self._invoke_event_sink(event))
                    completed, _ = await asyncio.wait(
                        (invocation,),
                        timeout=_EVENT_SINK_TIMEOUT_SECONDS,
                    )
                    if not completed:
                        # ``wait_for`` waits indefinitely when an adapter
                        # suppresses cancellation. Detach after opening the
                        # run-scoped circuit instead.
                        timed_out = True
                        invocation.cancel()
                        invocation.add_done_callback(_consume_task_result)
                    else:
                        await invocation
            except asyncio.CancelledError:
                if invocation is not None and not invocation.done():
                    invocation.cancel()
                    invocation.add_done_callback(_consume_task_result)
                raise
            except Exception:
                # Observability cannot alter execution, including cancellation
                # and timeout behavior.
                pass
            finally:
                queue.task_done()

    async def _close_sink_worker(self, run_id: str) -> None:
        worker = self._sink_workers.pop(run_id, None)
        self._sink_queues.pop(run_id, None)
        if worker is None:
            return
        worker.cancel()
        try:
            await worker
        except BaseException:
            pass

    async def _invoke_event_sink(self, event: AgentEvent) -> None:
        # Sinks are untrusted observability adapters. Give them an isolated
        # object graph so mutation cannot alter the stream or terminal result.
        sink_event = (
            copy.deepcopy(event)
            if _event_sink_requires_coordinator_copy(self.event_sink)
            else event
        )
        publisher = self.event_sink.publish
        if inspect.iscoroutinefunction(publisher):
            await publisher(sink_event)
            return
        result = await run_in_daemon_thread(publisher, sink_event)
        if inspect.isawaitable(result):
            await result

    def _published_event(self, event: AgentEvent) -> AgentEvent:
        """Resolve events emitted by inherited compatibility helpers."""

        stamp = self._published_events.get((event.run_id, event.event_id))
        return event if stamp is None else stamp.apply(event)


__all__ = ["RunCoordinator"]
