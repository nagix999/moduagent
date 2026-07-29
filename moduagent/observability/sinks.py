from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

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
    EventType.MODEL_STARTED: ("step", "attempt", "phase"),
    EventType.MODEL_DELTA: (),
    EventType.MODEL_COMPLETED: (),
    EventType.MEMORY_COMPACTED: (
        "phase",
        "original_tokens",
        "selected_tokens",
        "summarized_messages",
        "dropped_messages",
        "duration_seconds",
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
    EventType.RETRY: ("operation", "attempt", "phase", "error_type"),
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
    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        level: int = logging.INFO,
        sensitive_keys: Iterable[str] = DEFAULT_SENSITIVE_KEYS,
        replacement: str = "[REDACTED]",
    ) -> None:
        self.logger = logger or logging.getLogger("moduagent.events")
        self.level = level
        self.sensitive_keys = frozenset(_normalize_key(key) for key in sensitive_keys)
        self.replacement = replacement
        self.last_error: BaseException | None = None

    async def publish(self, event: AgentEvent) -> None:
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
                "phase": event.data.get("phase"),
                "duration_seconds": event.data.get("duration_seconds"),
                "finish_reason": finish_reason,
                "tool_call_count": event.data.get("tool_call_count"),
            },
            (
                "step",
                "attempt",
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
    return summary or None


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
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def _is_sensitive_key(key: str, sensitive_keys: frozenset[str]) -> bool:
    normalized = _normalize_key(key)
    return (
        normalized in sensitive_keys
        or normalized.endswith("_password")
        or normalized.endswith("_secret")
        or normalized.endswith("_token")
        or normalized.endswith("_api_key")
        or normalized.endswith("_private_key")
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
