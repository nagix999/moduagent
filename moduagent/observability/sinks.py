from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import math
import re
import sys
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, TextIO, runtime_checkable

from moduagent.messages import Usage
from moduagent.observability._background import run_in_daemon_thread
from moduagent.runtime.events import AgentEvent, EventType, EventVisibility


DEFAULT_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "authorization",
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
_STABLE_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_OMIT = object()
_STABLE_CODE_FIELDS = frozenset(
    {
        "boundary",
        "category",
        "code",
        "component",
        "decision",
        "error_type",
        "fallback",
        "finish_reason",
        "kind",
        "operation",
        "phase",
        "reason",
        "recovery",
        "selected_by",
        "source",
        "status",
        "type",
        "validation_cause_code",
        "validation_code",
        "validation_location",
    }
)
_HASHED_CORRELATION_FIELDS = frozenset(
    {"call_id", "failed_call_id", "result_ref", "step_id"}
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
_OBSERVABILITY_EVENT_FIELDS: Mapping[EventType, tuple[str, ...]] = {
    EventType.RUN_STARTED: ("agent", "session_id", "queue_wait_seconds"),
    EventType.CHECKPOINT_LOADED: ("step", "status", "state_version"),
    EventType.CHECKPOINT_SAVED: (
        "engine_id",
        "state_version",
        "boundary",
        "duration_seconds",
    ),
    EventType.SKILLS_DISCOVERED: ("count", "catalog_digest"),
    EventType.SKILL_SELECTION_STARTED: ("mode", "resumed"),
    EventType.SKILL_SELECTION_COMPLETED: (
        "mode",
        "catalog_tokens",
        "instruction_tokens",
    ),
    EventType.SKILL_SELECTED: ("name", "selected_by"),
    EventType.SKILL_ACTIVATED: (
        "name",
        "version",
        "digest",
        "source_id",
        "selected_by",
        "resumed",
    ),
    EventType.SKILL_RESOURCE_READ: (
        "skill_name",
        "operation",
        "success",
        "digest",
        "truncated",
        "returned_bytes",
        "scanned_bytes",
    ),
    EventType.SKILL_SKIPPED: ("name", "skill_name"),
    EventType.SKILL_DENIED: ("name", "skill_name", "error_type"),
    EventType.SKILL_ERROR: ("error_type",),
    EventType.MODEL_STARTED: (
        "step",
        "attempt",
        "model_turn",
        "phase",
        "message_count",
        "tool_count",
        "has_output_schema",
        "streaming",
    ),
    EventType.MODEL_DELTA: (),
    EventType.MODEL_COMPLETED: (),
    EventType.MODEL_FAILED: (
        "step",
        "attempt",
        "model_turn",
        "phase",
        "duration_seconds",
        "error_type",
        "code",
        "retryable",
        "terminal",
    ),
    EventType.MEMORY_COMPACTED: (
        "phase",
        "original_tokens",
        "selected_tokens",
        "summarized_messages",
        "dropped_messages",
        "duration_seconds",
    ),
    EventType.DELEGATION_REQUESTED: (
        "parent_tool_call_id",
        "caller_agent_id",
        "callee_agent_id",
        "status",
        "code",
    ),
    EventType.DELEGATION_AUTHORIZED: (
        "parent_tool_call_id",
        "caller_agent_id",
        "callee_agent_id",
        "status",
        "code",
    ),
    EventType.DELEGATION_REJECTED: (
        "parent_tool_call_id",
        "caller_agent_id",
        "callee_agent_id",
        "status",
        "code",
    ),
    EventType.DELEGATION_STARTED: (
        "parent_tool_call_id",
        "caller_agent_id",
        "callee_agent_id",
        "status",
        "code",
    ),
    EventType.DELEGATION_RESUMED: (
        "parent_tool_call_id",
        "caller_agent_id",
        "callee_agent_id",
        "status",
        "code",
    ),
    EventType.DELEGATION_COMPLETED: (
        "parent_tool_call_id",
        "caller_agent_id",
        "callee_agent_id",
        "status",
        "code",
    ),
    EventType.DELEGATION_FAILED: (
        "parent_tool_call_id",
        "caller_agent_id",
        "callee_agent_id",
        "status",
        "code",
    ),
    EventType.DELEGATION_RECONCILIATION_REQUIRED: (
        "parent_tool_call_id",
        "caller_agent_id",
        "callee_agent_id",
        "status",
        "code",
    ),
    EventType.TOOL_STARTED: (),
    EventType.TOOL_COMPLETED: (),
    EventType.TOOL_REPAIR_SCHEDULED: (
        "step_id",
        "tool_name",
        "failed_call_id",
        "error_type",
        "reason",
        "repair_attempt",
        "max_attempts",
    ),
    EventType.TOOL_REPAIR_EXHAUSTED: (
        "step_id",
        "fallback",
        "repair_attempts",
    ),
    EventType.POLICY_DECISION: ("kind", "source"),
    EventType.PLAN_CREATED: ("step_count", "plan_version"),
    EventType.STEP_STARTED: (
        "step_id",
        "attempt",
        "plan_version",
    ),
    EventType.STEP_MODEL_DELTA: (),
    EventType.STEP_RESULT_CREATED: ("step_id", "status"),
    EventType.STEP_VALIDATED: ("step_id", "decision"),
    EventType.STEP_COMMITTED: ("step_id", "result_ref", "plan_version"),
    EventType.STEP_RETRY: (
        "step_id",
        "attempt",
        "count_attempt",
        "validation_code",
        "validation_cause_code",
        "validation_location",
    ),
    EventType.STEP_FAILED: (
        "step_id",
        "attempt",
        "failure_id",
        "validation_code",
        "validation_cause_code",
        "validation_location",
    ),
    EventType.PLAN_REVISED: ("plan_version", "replan_count"),
    EventType.FINALIZATION_STARTED: ("phase", "count"),
    EventType.FINAL_DELTA: (),
    EventType.FINALIZATION_COMPLETED: ("phase", "count", "persisted"),
    EventType.RETRY: (
        "operation",
        "attempt",
        "model_turn",
        "phase",
        "duration_seconds",
        "error_type",
        "code",
        "retryable",
    ),
    EventType.RUN_COMPLETED: (),
    EventType.RUN_FAILED: (),
}


@runtime_checkable
class EventSink(Protocol):
    async def publish(self, event: AgentEvent) -> None: ...


class NoopEventSink:
    """Event sink marker whose dispatch path can be removed completely."""

    async def publish(self, event: AgentEvent) -> None:
        return None

    async def emit(self, event: AgentEvent) -> None:
        await self.publish(event)


class CompositeEventSink:
    """Fan out events without allowing a failed sink to alter agent execution."""

    # RunCoordinator delegates child mutation isolation to this exact built-in
    # type, avoiding the former Coordinator copy plus one copy per child.
    handles_event_isolation = True

    @property
    def content_safe(self) -> bool:
        return all(getattr(sink, "content_safe", False) is True for sink in self.sinks)

    def __init__(
        self,
        sinks: Iterable[EventSink] | EventSink = (),
        *additional_sinks: EventSink,
        on_error: Callable[[EventSink, BaseException], Any] | None = None,
    ) -> None:
        if callable(getattr(sinks, "publish", None)) or callable(
            getattr(sinks, "emit", None)
        ):
            self.sinks = (sinks, *additional_sinks)
        else:
            self.sinks = (*tuple(sinks), *additional_sinks)
        self.on_error = on_error
        self.last_errors: tuple[BaseException, ...] = ()

    async def publish(self, event: AgentEvent) -> None:
        active_sinks = tuple(
            sink for sink in self.sinks if not _event_sink_is_noop(sink)
        )
        if not active_sinks:
            self.last_errors = ()
            return
        results = await asyncio.gather(
            *(
                _publish(sink, _isolated_event_for_sink(sink, event))
                for sink in active_sinks
            ),
            return_exceptions=True,
        )
        failures: list[BaseException] = []
        for sink, result in zip(active_sinks, results):
            if isinstance(result, BaseException):
                failures.append(result)
                if self.on_error is not None:
                    try:
                        await _call_observability_adapter(
                            self.on_error,
                            sink,
                            result,
                        )
                    except Exception:
                        # Error reporting is observability too; it must stay isolated.
                        pass
        self.last_errors = tuple(failures)

    async def emit(self, event: AgentEvent) -> None:
        await self.publish(event)


class LoggingEventSink:
    content_safe = True

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        level: int = logging.INFO,
        sensitive_keys: Iterable[str] = DEFAULT_SENSITIVE_KEYS,
        replacement: str = "[REDACTED]",
        include_deltas: bool = False,
    ) -> None:
        if type(include_deltas) is not bool:
            raise TypeError("include_deltas must be a bool")
        self.logger = logger or logging.getLogger("moduagent.events")
        self.level = level
        self.sensitive_keys = frozenset(_normalize_key(key) for key in sensitive_keys)
        self.replacement = replacement
        self.include_deltas = include_deltas
        self.last_error: BaseException | None = None

    async def publish(self, event: AgentEvent) -> None:
        if (
            event.type
            in {
                EventType.MODEL_DELTA,
                EventType.STEP_MODEL_DELTA,
                EventType.FINAL_DELTA,
            }
            and not self.include_deltas
        ):
            return
        try:
            payload = mask_sensitive(
                _observability_event_to_dict(event),
                sensitive_keys=self.sensitive_keys,
                replacement=self.replacement,
            )
            await run_in_daemon_thread(
                self.logger.log,
                _event_log_level(event, self.level),
                "agent_event %s",
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            )
            self.last_error = None
        except Exception as exc:
            self.last_error = exc

    async def emit(self, event: AgentEvent) -> None:
        await self.publish(event)


class ConsoleEventSink:
    """Render content-free Agent progress for terminals and notebooks.

    The pretty view uses only the same sealed projection as
    :class:`LoggingEventSink`. Prompts, model deltas, tool arguments/results,
    and private model reasoning are never rendered. Use ``output_format="json"``
    when the same stream is consumed by an operational log collector.
    """

    content_safe = True
    _DELTA_EVENTS = frozenset(
        {
            EventType.MODEL_DELTA,
            EventType.STEP_MODEL_DELTA,
            EventType.FINAL_DELTA,
        }
    )
    _SUMMARY_EVENTS = frozenset(
        {
            EventType.RUN_STARTED,
            EventType.MODEL_STARTED,
            EventType.MODEL_COMPLETED,
            EventType.MODEL_FAILED,
            EventType.TOOL_STARTED,
            EventType.TOOL_COMPLETED,
            EventType.TOOL_REPAIR_SCHEDULED,
            EventType.TOOL_REPAIR_EXHAUSTED,
            EventType.DELEGATION_STARTED,
            EventType.DELEGATION_COMPLETED,
            EventType.DELEGATION_FAILED,
            EventType.FINALIZATION_STARTED,
            EventType.FINALIZATION_COMPLETED,
            EventType.RETRY,
            EventType.STEP_RETRY,
            EventType.RUN_COMPLETED,
            EventType.RUN_FAILED,
        }
    )

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        output_format: str = "pretty",
        detail: str = "summary",
        language: str = "en",
        color: bool | None = None,
        include_timestamp: bool = False,
    ) -> None:
        if stream is not None and not callable(getattr(stream, "write", None)):
            raise TypeError("stream must provide write() or be None")
        if output_format not in {"pretty", "json"}:
            raise ValueError("output_format must be pretty or json")
        if detail not in {"summary", "detailed"}:
            raise ValueError("detail must be summary or detailed")
        if language not in {"en", "ko"}:
            raise ValueError("language must be en or ko")
        if color is not None and type(color) is not bool:
            raise TypeError("color must be a bool or None")
        if type(include_timestamp) is not bool:
            raise TypeError("include_timestamp must be a bool")
        self.stream = stream
        self.output_format = output_format
        self.detail = detail
        self.language = language
        self.color = color
        self.include_timestamp = include_timestamp
        self.last_error: BaseException | None = None
        self._write_lock = threading.Lock()

    async def publish(self, event: AgentEvent) -> None:
        if event.type in self._DELTA_EVENTS:
            return
        if self.detail == "summary" and event.type not in self._SUMMARY_EVENTS:
            return
        try:
            record = _observability_event_to_dict(event)
            line = (
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if self.output_format == "json"
                else _pretty_console_line(
                    event,
                    record,
                    language=self.language,
                    color=self._color_enabled(),
                    include_timestamp=self.include_timestamp,
                )
            )
            await run_in_daemon_thread(self._write_line, line)
            self.last_error = None
        except Exception as exc:
            self.last_error = exc

    async def emit(self, event: AgentEvent) -> None:
        await self.publish(event)

    def _color_enabled(self) -> bool:
        if self.color is not None:
            return self.color
        stream = self.stream if self.stream is not None else sys.stderr
        isatty = getattr(stream, "isatty", None)
        try:
            return bool(isatty()) if callable(isatty) else False
        except Exception:
            return False

    def _write_line(self, line: str) -> None:
        stream = self.stream if self.stream is not None else sys.stderr
        with self._write_lock:
            stream.write(f"{line}\n")
            flush = getattr(stream, "flush", None)
            if callable(flush):
                flush()


@runtime_checkable
class MetricRecorder(Protocol):
    def increment(
        self,
        name: str,
        value: int | float = 1,
        labels: Mapping[str, str] | None = None,
    ) -> Any: ...

    def observe(
        self,
        name: str,
        value: int | float,
        labels: Mapping[str, str] | None = None,
    ) -> Any: ...


class InMemoryMetricRecorder:
    """Small default recorder; production telemetry is supplied by injection."""

    def __init__(self) -> None:
        self.counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = (
            defaultdict(float)
        )
        self.observations: dict[
            tuple[str, tuple[tuple[str, str], ...]], list[float]
        ] = defaultdict(list)

    def increment(
        self,
        name: str,
        value: int | float = 1,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        self.counters[_metric_key(name, labels)] += float(value)

    def observe(
        self,
        name: str,
        value: int | float,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        self.observations[_metric_key(name, labels)].append(float(value))


class MetricsEventSink:
    content_safe = True

    """Record run, latency, tool, failure and token metrics from AgentEvents."""

    def __init__(
        self,
        recorder: MetricRecorder | None = None,
        *,
        prefix: str = "moduagent",
    ) -> None:
        self.recorder = recorder if recorder is not None else InMemoryMetricRecorder()
        self._inline_recorder = type(self.recorder) is InMemoryMetricRecorder
        self.prefix = prefix.rstrip(".")
        self._run_started_at: dict[str, datetime] = {}
        self.last_error: BaseException | None = None

    async def publish(self, event: AgentEvent) -> None:
        try:
            event_name = _event_name(event)
            await self._increment("events.total", labels={"event_type": event_name})

            if event.type is EventType.RUN_STARTED:
                self._run_started_at[event.run_id] = event.occurred_at
                queue_wait = event.data.get("queue_wait_seconds")
                if (
                    isinstance(queue_wait, (int, float))
                    and not isinstance(queue_wait, bool)
                    and math.isfinite(float(queue_wait))
                    and queue_wait >= 0
                ):
                    await self._observe(
                        "run.queue_wait_seconds",
                        float(queue_wait),
                    )

            if event.type is EventType.TOOL_STARTED:
                await self._increment(
                    "tool_calls.total",
                    labels={"tool": _tool_name(event.data)},
                )

            if event.type is EventType.MODEL_STARTED:
                await self._increment(
                    "model.calls",
                    labels={"phase": _metric_phase(event.data)},
                )
            if event.type is EventType.MODEL_COMPLETED:
                duration = _metric_duration(event.data)
                if duration is not None:
                    await self._observe(
                        "model.duration_seconds",
                        duration,
                        labels={"phase": _metric_phase(event.data)},
                    )
            if event.type is EventType.MODEL_FAILED:
                labels = {
                    "phase": _metric_phase(event.data),
                    "code": _metric_code(event.data),
                }
                await self._increment("model.calls.failed", labels=labels)
                duration = _metric_duration(event.data)
                if duration is not None:
                    await self._observe(
                        "model.failed_duration_seconds",
                        duration,
                        labels=labels,
                    )

            if event.type is EventType.TOOL_COMPLETED and _event_failed(event.data):
                await self._increment(
                    "tool_calls.failed",
                    labels={"tool": _tool_name(event.data)},
                )
            if event.type is EventType.TOOL_COMPLETED:
                duration = _metric_duration(event.data)
                if duration is not None:
                    await self._observe(
                        "tool.duration_seconds",
                        duration,
                        labels={"tool": _tool_name(event.data)},
                    )

            if event.type is EventType.MEMORY_COMPACTED:
                duration = _metric_duration(event.data)
                if duration is not None:
                    await self._observe(
                        "memory.prepare_seconds",
                        duration,
                        labels={"phase": _metric_phase(event.data)},
                    )

            if event.type is EventType.CHECKPOINT_SAVED:
                await self._increment("checkpoint.saves")
                duration = _metric_duration(event.data)
                if duration is not None:
                    await self._observe(
                        "checkpoint.duration_seconds",
                        duration,
                    )

            if event.type is EventType.PLAN_CREATED:
                await self._increment(
                    "plan.steps.created",
                    int(event.data.get("step_count", 0)),
                )
            if event.type is EventType.STEP_COMMITTED:
                await self._increment("plan.steps.committed")
            if event.type is EventType.STEP_RETRY:
                await self._increment("plan.steps.retried")
            if event.type is EventType.TOOL_REPAIR_SCHEDULED:
                await self._increment("tool_repairs.scheduled")
            if event.type is EventType.TOOL_REPAIR_EXHAUSTED:
                await self._increment("tool_repairs.exhausted")
            if event.type is EventType.STEP_FAILED:
                await self._increment("plan.steps.failed")
            if event.type is EventType.PLAN_REVISED:
                await self._increment("plan.replans")
            if event.type is EventType.FINALIZATION_STARTED:
                await self._increment("finalization.calls")
            if event.type in (EventType.STEP_MODEL_DELTA, EventType.FINAL_DELTA):
                delta = str(event.data.get("delta", ""))
                token_estimate = max(0, (len(delta.encode("utf-8")) + 2) // 3)
                visibility = (
                    "public"
                    if event.visibility is EventVisibility.PUBLIC
                    else "internal"
                )
                await self._increment(
                    f"stream.{visibility}_tokens",
                    token_estimate,
                )

            if event.type in (EventType.RUN_COMPLETED, EventType.RUN_FAILED):
                status = (
                    "completed" if event.type is EventType.RUN_COMPLETED else "failed"
                )
                await self._increment("runs.total", labels={"status": status})
                started_at = self._run_started_at.pop(event.run_id, None)
                if started_at is not None:
                    duration = max(
                        0.0,
                        (event.occurred_at - started_at).total_seconds(),
                    )
                    await self._observe(
                        "run.duration_seconds",
                        duration,
                        labels={"status": status},
                    )
                usage = _event_usage(event.data)
                if usage is not None:
                    for token_type, value in (
                        ("input", usage.input_tokens),
                        ("output", usage.output_tokens),
                        ("total", usage.total_tokens),
                    ):
                        await self._increment(
                            "tokens.total",
                            value,
                            labels={"type": token_type},
                        )
            self.last_error = None
        except Exception as exc:
            self.last_error = exc

    async def emit(self, event: AgentEvent) -> None:
        await self.publish(event)

    async def _increment(
        self,
        suffix: str,
        value: int | float = 1,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        if self._inline_recorder:
            self.recorder.increment(
                f"{self.prefix}.{suffix}",
                value,
                labels=labels,
            )
            return
        await _call_observability_adapter(
            self.recorder.increment,
            f"{self.prefix}.{suffix}",
            value,
            labels=labels,
        )

    async def _observe(
        self,
        suffix: str,
        value: int | float,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        if self._inline_recorder:
            self.recorder.observe(
                f"{self.prefix}.{suffix}",
                value,
                labels=labels,
            )
            return
        await _call_observability_adapter(
            self.recorder.observe,
            f"{self.prefix}.{suffix}",
            value,
            labels=labels,
        )


class AuditEventSink:
    """Write security-relevant, recursively redacted event records."""

    content_safe = True

    DEFAULT_EVENT_TYPES = frozenset(
        {
            EventType.RUN_STARTED,
            EventType.TOOL_STARTED,
            EventType.TOOL_COMPLETED,
            EventType.TOOL_REPAIR_SCHEDULED,
            EventType.TOOL_REPAIR_EXHAUSTED,
            EventType.STEP_FAILED,
            EventType.RUN_COMPLETED,
            EventType.RUN_FAILED,
        }
    )

    def __init__(
        self,
        writer: Any | None = None,
        *,
        event_types: Iterable[EventType] | None = None,
        sensitive_keys: Iterable[str] = DEFAULT_SENSITIVE_KEYS,
        replacement: str = "[REDACTED]",
    ) -> None:
        self.records: list[Mapping[str, Any]] = []
        self._inline_writer = writer is None
        self.writer = self.records.append if self._inline_writer else writer
        self.event_types = frozenset(
            self.DEFAULT_EVENT_TYPES if event_types is None else event_types
        )
        self.sensitive_keys = frozenset(_normalize_key(key) for key in sensitive_keys)
        self.replacement = replacement
        self.last_error: BaseException | None = None

    async def publish(self, event: AgentEvent) -> None:
        if event.type not in self.event_types:
            return
        try:
            record = mask_sensitive(
                _observability_event_to_dict(event),
                sensitive_keys=self.sensitive_keys,
                replacement=self.replacement,
            )
            if self._inline_writer:
                self.records.append(record)
            else:
                await _write_audit(self.writer, record)
            self.last_error = None
        except Exception as exc:
            self.last_error = exc

    async def emit(self, event: AgentEvent) -> None:
        await self.publish(event)


def event_to_dict(event: AgentEvent) -> dict[str, Any]:
    return event.to_dict()


def _observability_event_to_dict(event: AgentEvent) -> dict[str, Any]:
    """Project sensitive event objects before built-in logging/audit sinks."""

    record = _observability_event_envelope(event)
    if event.type in {EventType.RUN_COMPLETED, EventType.RUN_FAILED}:
        result = event.data.get("result")
        usage = _event_usage(event.data)
        finish_reason = getattr(result, "finish_reason", None)
        if isinstance(finish_reason, Enum):
            finish_reason = finish_reason.value
        messages = getattr(result, "messages", ())
        error = _safe_public_error(getattr(result, "error", None))
        metadata = getattr(result, "metadata", {})
        error_summary = _safe_error_summary(metadata)
        safe_finish_reason = _safe_observability_field(
            "finish_reason",
            finish_reason
            or ("error" if event.type is EventType.RUN_FAILED else "completed"),
        )
        record["data"] = {
            "finish_reason": (
                safe_finish_reason
                if safe_finish_reason is not _OMIT
                else ("error" if event.type is EventType.RUN_FAILED else "completed")
            ),
            "has_output": getattr(result, "output", None) is not None,
            "has_error": error is not None,
            "error": error,
            "error_summary": error_summary,
            "message_count": (
                len(messages) if isinstance(messages, (list, tuple)) else 0
            ),
            "usage": (
                {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                }
                if usage is not None
                else None
            ),
        }
        return record
    if event.type in {EventType.TOOL_STARTED, EventType.TOOL_COMPLETED}:
        data = _project_observability_fields(
            event.data,
            (
                "tool_name",
                "call_id",
                "step",
                "step_id",
                "attempt",
                "duration_seconds",
                "success",
                "arguments_fingerprint",
            ),
        )
        failure = event.data.get("failure")
        if isinstance(failure, Mapping):
            projected_failure = _project_observability_fields(
                failure,
                (
                    "type",
                    "reason",
                    "recovery",
                    "retryable",
                    "arguments_fingerprint",
                    "invocation_fingerprint",
                    "failure_id",
                ),
            )
            if projected_failure:
                data["failure"] = projected_failure
        record["data"] = data
        return record
    if event.type is EventType.MODEL_COMPLETED:
        usage = _event_usage(event.data)
        finish_reason = event.data.get("finish_reason")
        if isinstance(finish_reason, Enum):
            finish_reason = finish_reason.value
        data = _project_observability_fields(
            {
                "step": event.data.get("step"),
                "attempt": event.data.get("attempt"),
                "model_turn": event.data.get("model_turn"),
                "phase": event.data.get("phase"),
                "duration_seconds": event.data.get("duration_seconds"),
                "finish_reason": finish_reason,
                "tool_call_count": event.data.get("tool_call_count"),
            },
            (
                "step",
                "attempt",
                "model_turn",
                "phase",
                "duration_seconds",
                "finish_reason",
                "tool_call_count",
            ),
        )
        data["usage"] = (
            {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
            }
            if usage is not None
            else None
        )
        record["data"] = data
        return record
    if event.type in {
        EventType.MODEL_DELTA,
        EventType.STEP_MODEL_DELTA,
        EventType.FINAL_DELTA,
    }:
        raw_delta = event.data.get("delta", "")
        delta = raw_delta if isinstance(raw_delta, str) else ""
        record["data"] = {
            **_project_observability_fields(
                event.data,
                ("step", "phase"),
            ),
            "delta_chars": len(delta),
            "delta_bytes": len(delta.encode("utf-8")),
        }
        return record
    data = _project_observability_fields(
        event.data,
        _OBSERVABILITY_EVENT_FIELDS.get(event.type, ()),
    )
    if event.type is EventType.SKILL_SELECTION_STARTED:
        requested = event.data.get("requested")
        if isinstance(requested, (list, tuple, set, frozenset)):
            data["requested_count"] = len(requested)
    elif event.type is EventType.SKILL_SELECTION_COMPLETED:
        selected = event.data.get("selected")
        if isinstance(selected, (list, tuple, set, frozenset)):
            data["selected_count"] = len(selected)
    record["data"] = data
    return record


def _observability_event_envelope(event: AgentEvent) -> dict[str, Any]:
    event_type = _event_name(event)
    timestamp = _utc_iso(event.occurred_at)
    return {
        "type": event_type,
        "occurred_at": timestamp,
        "event_id": event.event_id,
        "event_type": event_type,
        "event_schema_version": event.event_schema_version,
        "visibility": event.visibility.value,
        "run_id": event.run_id,
        "session_id": event.session_id,
        "engine_id": event.engine_id,
        "sequence": event.sequence,
        "timestamp": timestamp,
        "data": {},
    }


def _project_observability_fields(
    data: Mapping[str, Any],
    field_names: Iterable[str],
) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for key in field_names:
        if key not in data:
            continue
        value = _safe_observability_field(key, data[key])
        if value is not _OMIT:
            projected[key] = value
    return projected


def _safe_observability_field(key: str, value: Any) -> Any:
    if isinstance(value, Enum):
        value = value.value
    if value is None:
        return _OMIT
    if type(value) is bool or type(value) is int:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _OMIT
    if not isinstance(value, str):
        return _OMIT
    if key in {"validation_code", "validation_cause_code"}:
        if value not in _STEP_VALIDATION_CODES:
            return _OMIT
    elif key == "validation_location" and value not in _STEP_VALIDATION_LOCATIONS:
        return _OMIT
    if key in _HASHED_CORRELATION_FIELDS:
        return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
    text = _safe_public_error(value)
    if text is None:
        return _OMIT
    limit = 128 if key in _STABLE_CODE_FIELDS else 256
    text = text[:limit]
    if key in _STABLE_CODE_FIELDS and _STABLE_CODE_PATTERN.fullmatch(text) is None:
        return _OMIT
    return text


def _event_log_level(event: AgentEvent, configured_level: int) -> int:
    if event.type in {EventType.RETRY, EventType.STEP_RETRY}:
        return logging.WARNING
    if event.type is EventType.MODEL_FAILED:
        return logging.ERROR if event.data.get("terminal") is True else logging.WARNING
    if event.type in {EventType.RUN_FAILED, EventType.STEP_FAILED}:
        return logging.ERROR
    if event.type is EventType.TOOL_COMPLETED and _event_failed(event.data):
        return logging.ERROR
    return configured_level


def _safe_public_error(value: Any) -> str | None:
    """Return the already-public error as bounded printable text."""

    if not isinstance(value, str):
        return None
    text = " ".join(
        "".join(
            character if character.isprintable() else " " for character in value
        ).split()
    )
    return text[:512] or None


def _safe_error_summary(metadata: Any) -> dict[str, Any] | None:
    """Project only the stable, secret-safe terminal diagnostic fields."""

    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get("error_summary")
    if not isinstance(raw, Mapping):
        return None

    summary: dict[str, Any] = {}
    for key in (
        "category",
        "code",
        "component",
        "failure_id",
        "operation",
        "phase",
        "step_id",
    ):
        safe_value = _safe_observability_field(key, raw.get(key))
        if safe_value is not _OMIT and safe_value is not None:
            summary[key] = safe_value
    for key in ("retryable", "resumable"):
        value = raw.get(key)
        if type(value) is bool:
            summary[key] = value
    attempt = raw.get("attempt")
    if type(attempt) is int and 0 <= attempt <= 1_000_000:
        summary["attempt"] = attempt
    provider_finish_reason = raw.get("provider_finish_reason")
    if provider_finish_reason in {"timeout", "length", "max_tokens"}:
        summary["provider_finish_reason"] = provider_finish_reason
    return summary or None


_CONSOLE_LABELS: Mapping[str, Mapping[EventType, str]] = {
    "en": {
        EventType.RUN_STARTED: "Agent run started",
        EventType.CHECKPOINT_LOADED: "Checkpoint restored",
        EventType.CHECKPOINT_SAVED: "Checkpoint saved",
        EventType.SKILLS_DISCOVERED: "Skills discovered",
        EventType.SKILL_SELECTION_STARTED: "Selecting skills",
        EventType.SKILL_SELECTION_COMPLETED: "Skill selection completed",
        EventType.SKILL_SELECTED: "Skill selected",
        EventType.SKILL_ACTIVATED: "Skill activated",
        EventType.SKILL_RESOURCE_READ: "Skill resource read",
        EventType.SKILL_SKIPPED: "Skill skipped",
        EventType.SKILL_DENIED: "Skill denied",
        EventType.SKILL_ERROR: "Skill failed",
        EventType.MODEL_STARTED: "Generating model response",
        EventType.MODEL_COMPLETED: "Model response received",
        EventType.MODEL_FAILED: "Model request failed",
        EventType.MEMORY_COMPACTED: "Context memory prepared",
        EventType.DELEGATION_REQUESTED: "Delegation requested",
        EventType.DELEGATION_AUTHORIZED: "Delegation authorized",
        EventType.DELEGATION_REJECTED: "Delegation rejected",
        EventType.DELEGATION_STARTED: "Delegated Agent running",
        EventType.DELEGATION_RESUMED: "Delegated Agent resumed",
        EventType.DELEGATION_COMPLETED: "Delegated Agent completed",
        EventType.DELEGATION_FAILED: "Delegated Agent failed",
        EventType.DELEGATION_RECONCILIATION_REQUIRED: "Delegation needs review",
        EventType.TOOL_STARTED: "Running tool",
        EventType.TOOL_COMPLETED: "Tool completed",
        EventType.TOOL_REPAIR_SCHEDULED: "Repairing tool call",
        EventType.TOOL_REPAIR_EXHAUSTED: "Tool repair exhausted",
        EventType.POLICY_DECISION: "Policy decision",
        EventType.PLAN_CREATED: "Plan created",
        EventType.STEP_STARTED: "Plan step running",
        EventType.STEP_RESULT_CREATED: "Step result created",
        EventType.STEP_VALIDATED: "Step validated",
        EventType.STEP_COMMITTED: "Step committed",
        EventType.STEP_RETRY: "Retrying plan step",
        EventType.STEP_FAILED: "Plan step failed",
        EventType.PLAN_REVISED: "Plan revised",
        EventType.FINALIZATION_STARTED: "Composing final answer",
        EventType.FINALIZATION_COMPLETED: "Final answer composed",
        EventType.RETRY: "Retrying operation",
        EventType.RUN_COMPLETED: "Agent run completed",
        EventType.RUN_FAILED: "Agent run failed",
    },
    "ko": {
        EventType.RUN_STARTED: "Agent 실행 시작",
        EventType.CHECKPOINT_LOADED: "체크포인트 복원 완료",
        EventType.CHECKPOINT_SAVED: "체크포인트 저장 완료",
        EventType.SKILLS_DISCOVERED: "Skill 탐색 완료",
        EventType.SKILL_SELECTION_STARTED: "Skill 선택 중",
        EventType.SKILL_SELECTION_COMPLETED: "Skill 선택 완료",
        EventType.SKILL_SELECTED: "Skill 선택됨",
        EventType.SKILL_ACTIVATED: "Skill 활성화",
        EventType.SKILL_RESOURCE_READ: "Skill 리소스 읽기 완료",
        EventType.SKILL_SKIPPED: "Skill 건너뜀",
        EventType.SKILL_DENIED: "Skill 사용 거부됨",
        EventType.SKILL_ERROR: "Skill 처리 실패",
        EventType.MODEL_STARTED: "모델 응답 생성 중",
        EventType.MODEL_COMPLETED: "모델 응답 수신 완료",
        EventType.MODEL_FAILED: "모델 호출 실패",
        EventType.MEMORY_COMPACTED: "대화 컨텍스트 준비 완료",
        EventType.DELEGATION_REQUESTED: "Agent 위임 요청",
        EventType.DELEGATION_AUTHORIZED: "Agent 위임 승인",
        EventType.DELEGATION_REJECTED: "Agent 위임 거부",
        EventType.DELEGATION_STARTED: "위임 Agent 실행 중",
        EventType.DELEGATION_RESUMED: "위임 Agent 실행 재개",
        EventType.DELEGATION_COMPLETED: "위임 Agent 실행 완료",
        EventType.DELEGATION_FAILED: "위임 Agent 실행 실패",
        EventType.DELEGATION_RECONCILIATION_REQUIRED: "위임 상태 확인 필요",
        EventType.TOOL_STARTED: "Tool 실행 중",
        EventType.TOOL_COMPLETED: "Tool 실행 완료",
        EventType.TOOL_REPAIR_SCHEDULED: "Tool 호출 복구 중",
        EventType.TOOL_REPAIR_EXHAUSTED: "Tool 호출 복구 실패",
        EventType.POLICY_DECISION: "실행 정책 판단",
        EventType.PLAN_CREATED: "실행 계획 생성 완료",
        EventType.STEP_STARTED: "계획 단계 실행 중",
        EventType.STEP_RESULT_CREATED: "단계 결과 생성 완료",
        EventType.STEP_VALIDATED: "단계 결과 검증 완료",
        EventType.STEP_COMMITTED: "계획 단계 반영 완료",
        EventType.STEP_RETRY: "계획 단계 재시도 중",
        EventType.STEP_FAILED: "계획 단계 실패",
        EventType.PLAN_REVISED: "실행 계획 수정 완료",
        EventType.FINALIZATION_STARTED: "최종 답변 구성 중",
        EventType.FINALIZATION_COMPLETED: "최종 답변 구성 완료",
        EventType.RETRY: "작업 재시도 중",
        EventType.RUN_COMPLETED: "Agent 실행 완료",
        EventType.RUN_FAILED: "Agent 실행 실패",
    },
}

_CONSOLE_STARTED_EVENTS = frozenset(
    {
        EventType.RUN_STARTED,
        EventType.SKILL_SELECTION_STARTED,
        EventType.MODEL_STARTED,
        EventType.DELEGATION_REQUESTED,
        EventType.DELEGATION_STARTED,
        EventType.TOOL_STARTED,
        EventType.STEP_STARTED,
        EventType.FINALIZATION_STARTED,
    }
)
_CONSOLE_FAILED_EVENTS = frozenset(
    {
        EventType.SKILL_DENIED,
        EventType.SKILL_ERROR,
        EventType.MODEL_FAILED,
        EventType.DELEGATION_REJECTED,
        EventType.DELEGATION_FAILED,
        EventType.DELEGATION_RECONCILIATION_REQUIRED,
        EventType.TOOL_REPAIR_EXHAUSTED,
        EventType.STEP_FAILED,
        EventType.RUN_FAILED,
    }
)
_CONSOLE_RETRY_EVENTS = frozenset(
    {
        EventType.RETRY,
        EventType.STEP_RETRY,
        EventType.TOOL_REPAIR_SCHEDULED,
        EventType.PLAN_REVISED,
    }
)


def _pretty_console_line(
    event: AgentEvent,
    record: Mapping[str, Any],
    *,
    language: str,
    color: bool,
    include_timestamp: bool,
) -> str:
    data = record.get("data")
    safe_data = data if isinstance(data, Mapping) else {}
    failed_tool = event.type is EventType.TOOL_COMPLETED and (
        safe_data.get("success") is False or "failure" in safe_data
    )
    if event.type in _CONSOLE_FAILED_EVENTS or failed_tool:
        icon, ansi = "✗", "31"
    elif event.type in _CONSOLE_RETRY_EVENTS:
        icon, ansi = "↻", "33"
    elif event.type in _CONSOLE_STARTED_EVENTS:
        icon, ansi = "●", "36"
    else:
        icon, ansi = "✓", "32"
    label = _CONSOLE_LABELS[language].get(
        event.type,
        event.type.value.replace("_", " "),
    )
    if color:
        icon = f"\x1b[{ansi}m{icon}\x1b[0m"
        label = f"\x1b[1m{label}\x1b[0m"
    indent = (
        ""
        if event.type
        in {EventType.RUN_STARTED, EventType.RUN_COMPLETED, EventType.RUN_FAILED}
        else "  "
    )
    if event.type in {
        EventType.TOOL_STARTED,
        EventType.TOOL_COMPLETED,
        EventType.TOOL_REPAIR_SCHEDULED,
        EventType.TOOL_REPAIR_EXHAUSTED,
        EventType.STEP_STARTED,
        EventType.STEP_RESULT_CREATED,
        EventType.STEP_VALIDATED,
        EventType.STEP_COMMITTED,
        EventType.STEP_RETRY,
        EventType.STEP_FAILED,
    }:
        indent = "    "
    prefix = (
        f"[{event.occurred_at.astimezone().strftime('%H:%M:%S')}] "
        if include_timestamp
        else ""
    )
    details = _pretty_console_details(event.type, safe_data, language=language)
    suffix = f" · {' · '.join(details)}" if details else ""
    return f"{prefix}{indent}{icon} {label}{suffix}"


def _pretty_console_details(
    event_type: EventType,
    data: Mapping[str, Any],
    *,
    language: str,
) -> list[str]:
    details: list[str] = []
    if event_type in {EventType.TOOL_STARTED, EventType.TOOL_COMPLETED}:
        _append_console_detail(details, data.get("tool_name"))
    elif event_type in {
        EventType.SKILL_SELECTED,
        EventType.SKILL_ACTIVATED,
        EventType.SKILL_SKIPPED,
        EventType.SKILL_DENIED,
    }:
        _append_console_detail(details, data.get("name") or data.get("skill_name"))
    elif event_type in {
        EventType.DELEGATION_REQUESTED,
        EventType.DELEGATION_AUTHORIZED,
        EventType.DELEGATION_REJECTED,
        EventType.DELEGATION_STARTED,
        EventType.DELEGATION_RESUMED,
        EventType.DELEGATION_COMPLETED,
        EventType.DELEGATION_FAILED,
    }:
        caller = data.get("caller_agent_id")
        callee = data.get("callee_agent_id")
        if isinstance(caller, str) and isinstance(callee, str):
            details.append(f"{caller} → {callee}")

    phase = data.get("phase")
    if isinstance(phase, str):
        details.append(f"phase={phase}")
    turn = data.get("model_turn")
    if type(turn) is int:
        details.append(f"turn={turn}")
    attempt = data.get("attempt")
    if type(attempt) is int and attempt > 1:
        details.append(f"attempt={attempt}")
    duration = data.get("duration_seconds")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        details.append(_format_console_duration(float(duration)))
    usage = data.get("usage")
    if isinstance(usage, Mapping) and type(usage.get("total_tokens")) is int:
        unit = "토큰" if language == "ko" else "tokens"
        details.append(f"{usage['total_tokens']:,} {unit}")
    tool_calls = data.get("tool_call_count")
    if type(tool_calls) is int and tool_calls > 0:
        unit = "개 Tool 호출" if language == "ko" else "tool calls"
        details.append(f"{tool_calls} {unit}")
    if event_type in {EventType.RUN_COMPLETED, EventType.RUN_FAILED}:
        summary = data.get("error_summary")
        if isinstance(summary, Mapping):
            _append_console_detail(details, summary.get("operation"))
            _append_console_detail(details, summary.get("code"))
    failure = data.get("failure")
    if isinstance(failure, Mapping):
        _append_console_detail(details, failure.get("type"))
        _append_console_detail(details, failure.get("reason"))
    elif event_type in _CONSOLE_FAILED_EVENTS:
        _append_console_detail(details, data.get("code") or data.get("error_type"))
    return details


def _append_console_detail(details: list[str], value: Any) -> None:
    if isinstance(value, str) and value and value not in details:
        details.append(value)


def _format_console_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{max(0.0, seconds) * 1_000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{seconds / 60:.1f}m"


def mask_sensitive(
    value: Any,
    *,
    sensitive_keys: Iterable[str] = DEFAULT_SENSITIVE_KEYS,
    replacement: str = "[REDACTED]",
) -> Any:
    """Recursively redact secret-bearing keys while preserving record shape."""

    keys = frozenset(_normalize_key(key) for key in sensitive_keys)

    def visit(item: Any) -> Any:
        if isinstance(item, Mapping):
            masked: dict[str, Any] = {}
            for key, nested in item.items():
                text_key = str(key)
                masked[text_key] = (
                    replacement if _is_sensitive_key(text_key, keys) else visit(nested)
                )
            return masked
        if isinstance(item, tuple):
            return tuple(visit(nested) for nested in item)
        if isinstance(item, list):
            return [visit(nested) for nested in item]
        return item

    return visit(value)


def _event_sink_is_noop(sink: Any) -> bool:
    """Return whether an exact built-in sink has no observable side effects."""

    if type(sink) is NoopEventSink:
        return True
    if type(sink) is CompositeEventSink:
        return all(_event_sink_is_noop(child) for child in sink.sinks)
    return False


def _event_sink_requires_coordinator_copy(sink: Any) -> bool:
    """Keep legacy custom sinks isolated while sharing with trusted built-ins."""

    return type(sink) not in {
        NoopEventSink,
        CompositeEventSink,
        ConsoleEventSink,
        LoggingEventSink,
        MetricsEventSink,
        AuditEventSink,
    }


def _isolated_event_for_sink(sink: Any, event: AgentEvent) -> AgentEvent:
    if _event_sink_requires_coordinator_copy(sink):
        return copy.deepcopy(event)
    return event


async def _publish(sink: Any, event: AgentEvent) -> None:
    publisher = getattr(sink, "publish", None)
    if not callable(publisher):
        publisher = getattr(sink, "emit", None)
    if not callable(publisher):
        raise TypeError("event sink must provide publish() or emit()")
    await _call_observability_adapter(publisher, event)


async def _write_audit(writer: Any, record: Mapping[str, Any]) -> None:
    if callable(writer):
        method = writer
    else:
        method = next(
            (
                getattr(writer, name)
                for name in ("write", "record", "append")
                if callable(getattr(writer, name, None))
            ),
            None,
        )
        if method is None:
            raise TypeError("audit writer must be callable or provide write()")
    await _call_observability_adapter(method, record)


async def _call_observability_adapter(
    function: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    if inspect.iscoroutinefunction(function):
        return await function(*args, **kwargs)
    if kwargs:
        result = await run_in_daemon_thread(
            lambda: function(*args, **kwargs),
        )
    else:
        result = await run_in_daemon_thread(function, *args)
    if inspect.isawaitable(result):
        return await result
    return result


def _event_name(event: AgentEvent) -> str:
    return event.type.value if isinstance(event.type, Enum) else str(event.type)


def _event_failed(data: Mapping[str, Any]) -> bool:
    if (
        data.get("success") is False
        or data.get("error") is not None
        or data.get("failure") is not None
    ):
        return True
    result = data.get("result")
    return getattr(result, "success", True) is False


def _tool_name(data: Mapping[str, Any]) -> str:
    name = data.get("tool_name", data.get("name"))
    if name is None:
        call = data.get("tool_call", data.get("call"))
        name = getattr(call, "name", None)
        if isinstance(call, Mapping):
            name = call.get("name", name)
    return str(name or "unknown")


def _metric_phase(data: Mapping[str, Any]) -> str:
    phase = data.get("phase")
    if isinstance(phase, Enum):
        phase = phase.value
    if (
        isinstance(phase, str)
        and len(phase) <= 64
        and _STABLE_CODE_PATTERN.fullmatch(phase) is not None
    ):
        return phase
    return "unknown"


def _metric_code(data: Mapping[str, Any]) -> str:
    code = data.get("code")
    if (
        isinstance(code, str)
        and len(code) <= 128
        and _STABLE_CODE_PATTERN.fullmatch(code) is not None
    ):
        return code
    return "unknown"


def _metric_duration(data: Mapping[str, Any]) -> float | None:
    value = data.get("duration_seconds")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        return None
    return float(value)


def _event_usage(data: Mapping[str, Any]) -> Usage | None:
    usage: Any = data.get("usage")
    result = data.get("result")
    if usage is None and result is not None:
        usage = getattr(result, "usage", None)
        if isinstance(result, Mapping):
            usage = result.get("usage", usage)
    if isinstance(usage, Usage):
        return usage
    if isinstance(usage, Mapping):
        return Usage.from_provider(usage)
    return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return _utc_iso(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_safe(getattr(value, field.name))
            for field in fields(value)
        }
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump(mode="json"))
    return repr(value)


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).strip().lower())


def _is_sensitive_key(key: str, sensitive_keys: frozenset[str]) -> bool:
    normalized = _normalize_key(key)
    return (
        normalized in sensitive_keys
        or normalized.endswith("password")
        or normalized.endswith("secret")
        or normalized.endswith("token")
        or normalized.endswith("apikey")
        or normalized.endswith("privatekey")
        or normalized.endswith("credential")
        or normalized.endswith("credentials")
        or normalized.endswith("authorization")
    )


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _metric_key(
    name: str, labels: Mapping[str, str] | None
) -> tuple[str, tuple[tuple[str, str], ...]]:
    return name, tuple(sorted((labels or {}).items()))


__all__ = [
    "AuditEventSink",
    "CompositeEventSink",
    "DEFAULT_SENSITIVE_KEYS",
    "EventSink",
    "InMemoryMetricRecorder",
    "LoggingEventSink",
    "MetricRecorder",
    "MetricsEventSink",
    "NoopEventSink",
    "event_to_dict",
    "mask_sensitive",
]
