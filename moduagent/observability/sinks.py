from __future__ import annotations

import asyncio
import inspect
import json
import logging
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from moduagent.messages import Usage
from moduagent.runtime.events import AgentEvent, EventType


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


@runtime_checkable
class EventSink(Protocol):
    async def publish(self, event: AgentEvent) -> None: ...


class NoopEventSink:
    async def publish(self, event: AgentEvent) -> None:
        return None

    async def emit(self, event: AgentEvent) -> None:
        await self.publish(event)


class CompositeEventSink:
    """Fan out events without allowing a failed sink to alter agent execution."""

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
        results = await asyncio.gather(
            *(_publish(sink, event) for sink in self.sinks),
            return_exceptions=True,
        )
        failures: list[BaseException] = []
        for sink, result in zip(self.sinks, results):
            if isinstance(result, BaseException):
                failures.append(result)
                if self.on_error is not None:
                    try:
                        callback_result = self.on_error(sink, result)
                        if inspect.isawaitable(callback_result):
                            await callback_result
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
                event_to_dict(event),
                sensitive_keys=self.sensitive_keys,
                replacement=self.replacement,
            )
            self.logger.log(
                self.level,
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
        self.prefix = prefix.rstrip(".")
        self._run_started_at: dict[str, datetime] = {}
        self.last_error: BaseException | None = None

    async def publish(self, event: AgentEvent) -> None:
        try:
            event_name = _event_name(event)
            await self._increment("events.total", labels={"event_type": event_name})

            if event.type is EventType.RUN_STARTED:
                self._run_started_at[event.run_id] = event.occurred_at

            if event.type is EventType.TOOL_STARTED:
                await self._increment(
                    "tool_calls.total",
                    labels={"tool": _tool_name(event.data)},
                )

            if event.type is EventType.TOOL_COMPLETED and _event_failed(event.data):
                await self._increment(
                    "tool_calls.failed",
                    labels={"tool": _tool_name(event.data)},
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
        result = self.recorder.increment(
            f"{self.prefix}.{suffix}", value, labels=labels
        )
        if inspect.isawaitable(result):
            await result

    async def _observe(
        self,
        suffix: str,
        value: int | float,
        labels: Mapping[str, str] | None = None,
    ) -> None:
        result = self.recorder.observe(f"{self.prefix}.{suffix}", value, labels=labels)
        if inspect.isawaitable(result):
            await result


class AuditEventSink:
    """Write security-relevant, recursively redacted event records."""

    DEFAULT_EVENT_TYPES = frozenset(
        {
            EventType.RUN_STARTED,
            EventType.TOOL_STARTED,
            EventType.TOOL_COMPLETED,
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
        self.writer = self.records.append if writer is None else writer
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
                event_to_dict(event),
                sensitive_keys=self.sensitive_keys,
                replacement=self.replacement,
            )
            await _write_audit(self.writer, record)
            self.last_error = None
        except Exception as exc:
            self.last_error = exc

    async def emit(self, event: AgentEvent) -> None:
        await self.publish(event)


def event_to_dict(event: AgentEvent) -> dict[str, Any]:
    return {
        "type": _event_name(event),
        "run_id": event.run_id,
        "occurred_at": _utc_iso(event.occurred_at),
        "data": _json_safe(event.data),
    }


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


async def _publish(sink: Any, event: AgentEvent) -> None:
    publisher = getattr(sink, "publish", None)
    if not callable(publisher):
        publisher = getattr(sink, "emit", None)
    if not callable(publisher):
        raise TypeError("event sink must provide publish() or emit()")
    result = publisher(event)
    if inspect.isawaitable(result):
        await result


async def _write_audit(writer: Any, record: Mapping[str, Any]) -> None:
    if callable(writer):
        result = writer(record)
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
        result = method(record)
    if inspect.isawaitable(result):
        await result


def _event_name(event: AgentEvent) -> str:
    return event.type.value if isinstance(event.type, Enum) else str(event.type)


def _event_failed(data: Mapping[str, Any]) -> bool:
    if data.get("success") is False or data.get("error") is not None:
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
