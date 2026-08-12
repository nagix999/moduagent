"""Content-free progress and failure diagnostics for the RAG pipeline.

The records in this module deliberately exclude exception messages, document
paths, document content, service URLs, request/response bodies, and arbitrary
metadata.  A failure is described with a stable stage code, bounded exception
type names, and a small allowlist of numeric operating-system/HTTP facts.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
import threading
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, TextIO


_STABLE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_COUNT_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_EVENT_STATUSES = frozenset({"started", "completed", "failed"})
_MAX_CAUSE_TYPES = 5
_MAX_COUNTS = 32
_CURRENT_CORRELATION_ID: ContextVar[str | None] = ContextVar(
    "rag_pipeline_correlation_id",
    default=None,
)


@dataclass(frozen=True, slots=True)
class PipelineLogEvent:
    """One immutable, content-free pipeline lifecycle record."""

    sequence: int
    operation: str
    stage: str
    status: str
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str | None = None
    source_id: str | None = None
    generation_id: str | None = None
    item_index: int | None = None
    item_count: int | None = None
    counts: Mapping[str, int] = field(default_factory=dict)
    error_code: str | None = None
    exception_type: str | None = None
    cause_types: tuple[str, ...] = ()
    errno: int | None = None
    http_status: int | None = None

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("sequence must be a positive integer")
        for name in ("operation", "stage"):
            value = getattr(self, name)
            if not _is_stable_label(value):
                raise ValueError(f"{name} must be a stable bounded label")
        if self.status not in _EVENT_STATUSES:
            raise ValueError("status must be started, completed, or failed")
        if not isinstance(self.occurred_at, datetime):
            raise TypeError("occurred_at must be a datetime")
        occurred_at = self.occurred_at
        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=timezone.utc)
        else:
            occurred_at = occurred_at.astimezone(timezone.utc)
        object.__setattr__(self, "occurred_at", occurred_at)

        for name in ("correlation_id", "source_id", "generation_id"):
            value = getattr(self, name)
            if value is not None and not _is_stable_label(value):
                raise ValueError(f"{name} must be a stable bounded label or None")
        for name in ("item_index", "item_count"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if (
            self.item_index is not None
            and self.item_count is not None
            and self.item_index > self.item_count
        ):
            raise ValueError("item_index cannot exceed item_count")

        counts = _validated_counts(self.counts)
        object.__setattr__(self, "counts", MappingProxyType(counts))

        for name in ("error_code", "exception_type"):
            value = getattr(self, name)
            if value is not None and not _is_stable_label(value):
                raise ValueError(f"{name} must be a stable bounded label or None")
        causes = tuple(self.cause_types)
        if len(causes) > _MAX_CAUSE_TYPES or any(
            not _is_stable_label(value) for value in causes
        ):
            raise ValueError("cause_types must contain bounded stable type names")
        object.__setattr__(self, "cause_types", causes)

        if self.errno is not None and (
            type(self.errno) is not int or not -(2**31) <= self.errno < 2**31
        ):
            raise ValueError("errno must be a bounded integer or None")
        if self.http_status is not None and (
            type(self.http_status) is not int or not 100 <= self.http_status <= 599
        ):
            raise ValueError("http_status must be between 100 and 599 or None")
        if self.status != "failed" and any(
            value is not None
            for value in (
                self.error_code,
                self.exception_type,
                self.errno,
                self.http_status,
            )
        ):
            raise ValueError("failure fields are only valid on failed events")
        if self.status != "failed" and self.cause_types:
            raise ValueError("cause_types are only valid on failed events")

    def to_dict(self, *, include_timestamp: bool = True) -> dict[str, Any]:
        """Return the sealed public projection used by the console renderer."""

        if type(include_timestamp) is not bool:
            raise TypeError("include_timestamp must be a bool")
        value: dict[str, Any] = {
            "sequence": self.sequence,
            "operation": self.operation,
            "stage": self.stage,
            "status": self.status,
        }
        if include_timestamp:
            value["occurred_at"] = self.occurred_at.isoformat()
        for name in (
            "correlation_id",
            "source_id",
            "generation_id",
            "item_index",
            "item_count",
            "error_code",
            "exception_type",
            "errno",
            "http_status",
        ):
            item = getattr(self, name)
            if item is not None:
                value[name] = item
        if self.counts:
            value["counts"] = dict(self.counts)
        if self.cause_types:
            value["cause_types"] = list(self.cause_types)
        return value


class PipelineExecutionLog:
    """Bounded collector with an optional immediate, Jupyter-friendly console.

    Console delivery is best-effort: a broken or closed stream never changes
    the pipeline outcome.  Callers can inspect :attr:`events` and
    :attr:`latest_failure` after an Agent failure.
    """

    def __init__(
        self,
        *,
        max_events: int = 1_000,
        console: bool = False,
        stream: TextIO | None = None,
        sink: Callable[[PipelineLogEvent], None] | None = None,
        include_timestamp: bool = False,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if type(max_events) is not int:
            raise TypeError("max_events must be an integer")
        if not 1 <= max_events <= 100_000:
            raise ValueError("max_events must be between one and 100000")
        if type(console) is not bool:
            raise TypeError("console must be a bool")
        if type(include_timestamp) is not bool:
            raise TypeError("include_timestamp must be a bool")
        if stream is not None and not callable(getattr(stream, "write", None)):
            raise TypeError("stream must provide write() or be None")
        if sink is not None and not callable(sink):
            raise TypeError("sink must be callable or None")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable or None")
        self.max_events = max_events
        self._console = console
        self._stream = stream
        self._sink = sink
        self.include_timestamp = include_timestamp
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._events: deque[PipelineLogEvent] = deque(maxlen=max_events)
        self._sequence = 0
        self._correlation_sequence = 0
        self._latest_failure: PipelineLogEvent | None = None
        self._failure_by_run: OrderedDict[str, PipelineLogEvent | None] = OrderedDict()
        self._lock = threading.Lock()
        self.dropped_events = 0
        self.console_error_count = 0
        self.last_console_error_type: str | None = None
        self.sink_error_count = 0
        self.last_sink_error_type: str | None = None

    @classmethod
    def console(
        cls,
        *,
        max_events: int = 1_000,
        stream: TextIO | None = None,
        sink: Callable[[PipelineLogEvent], None] | None = None,
        include_timestamp: bool = False,
        clock: Callable[[], datetime] | None = None,
    ) -> PipelineExecutionLog:
        """Create a collector that also renders each event immediately."""

        return cls(
            max_events=max_events,
            console=True,
            stream=stream,
            sink=sink,
            include_timestamp=include_timestamp,
            clock=clock,
        )

    @property
    def events(self) -> tuple[PipelineLogEvent, ...]:
        with self._lock:
            return tuple(self._events)

    @property
    def latest(self) -> PipelineLogEvent | None:
        with self._lock:
            return self._events[-1] if self._events else None

    @property
    def latest_failure(self) -> PipelineLogEvent | None:
        with self._lock:
            return self._latest_failure

    def emit(
        self,
        operation: str,
        stage: str,
        status: str,
        *,
        source_id: str | None = None,
        generation_id: str | None = None,
        item_index: int | None = None,
        item_count: int | None = None,
        counts: Mapping[str, int] | None = None,
        error: BaseException | None = None,
        error_code: str | None = None,
    ) -> PipelineLogEvent:
        """Append and optionally print one sealed event.

        ``error`` is inspected only for type names, ``errno``, and HTTP status;
        it is never retained and ``str(error)`` is never evaluated.
        """

        if error is not None and not isinstance(error, BaseException):
            raise TypeError("error must be a BaseException or None")
        if status == "failed":
            resolved_error_code = error_code or _default_error_code(stage)
            failure = _failure_projection(error)
        else:
            if error is not None or error_code is not None:
                raise ValueError("error fields require failed status")
            resolved_error_code = None
            failure = _FailureProjection()

        with self._lock:
            self._sequence += 1
            event = PipelineLogEvent(
                sequence=self._sequence,
                operation=operation,
                stage=stage,
                status=status,
                occurred_at=self._clock(),
                correlation_id=_CURRENT_CORRELATION_ID.get(),
                source_id=source_id,
                generation_id=generation_id,
                item_index=item_index,
                item_count=item_count,
                counts={} if counts is None else counts,
                error_code=resolved_error_code,
                exception_type=failure.exception_type,
                cause_types=failure.cause_types,
                errno=failure.errno,
                http_status=failure.http_status,
            )
            if len(self._events) == self.max_events:
                self.dropped_events += 1
            self._events.append(event)
            if status == "started" and stage == "run":
                # A new serialized manager operation must not inherit a stale
                # failure from an earlier notebook request.
                self._latest_failure = None
            if status == "failed":
                self._latest_failure = event
        if self._sink is not None:
            self._deliver_sink(event)
        if self._console:
            self._write_console(event)
        return event

    def stage(
        self,
        operation: str,
        stage: str,
        *,
        source_id: str | None = None,
        generation_id: str | None = None,
        item_index: int | None = None,
        item_count: int | None = None,
        counts: Mapping[str, int] | None = None,
        error_code: str | None = None,
    ) -> _PipelineStage:
        """Create a sync/async scope that emits start and terminal events."""

        return _PipelineStage(
            self,
            operation=operation,
            stage=stage,
            source_id=source_id,
            generation_id=generation_id,
            item_index=item_index,
            item_count=item_count,
            counts=counts,
            error_code=error_code,
        )

    def bind(self, correlation_id: str) -> _PipelineCorrelation:
        """Bind events emitted by the current async task to one request."""

        if not _is_stable_label(correlation_id):
            raise ValueError("correlation_id must be a stable bounded label")
        return _PipelineCorrelation(correlation_id)

    def associate_run(
        self,
        run_id: str,
        correlation_id: str,
    ) -> PipelineLogEvent | None:
        """Seal the exact pipeline failure, if any, for one Agent run ID."""

        if not _is_stable_label(run_id):
            raise ValueError("run_id must be a stable bounded label")
        if not _is_stable_label(correlation_id):
            raise ValueError("correlation_id must be a stable bounded label")
        with self._lock:
            failure = self._failure_after_unlocked(
                correlation_id,
                after_sequence=0,
            )
            self._failure_by_run[run_id] = failure
            self._failure_by_run.move_to_end(run_id)
            while len(self._failure_by_run) > self.max_events:
                self._failure_by_run.popitem(last=False)
            return failure

    def failure_for_run(self, run_id: str) -> PipelineLogEvent | None:
        if not _is_stable_label(run_id):
            return None
        with self._lock:
            return self._failure_by_run.get(run_id)

    def _failure_after(
        self,
        correlation_id: str | None,
        *,
        after_sequence: int,
    ) -> PipelineLogEvent | None:
        with self._lock:
            return self._failure_after_unlocked(
                correlation_id,
                after_sequence=after_sequence,
            )

    def _failure_after_unlocked(
        self,
        correlation_id: str | None,
        *,
        after_sequence: int,
    ) -> PipelineLogEvent | None:
        return next(
            (
                event
                for event in reversed(self._events)
                if event.status == "failed"
                and event.sequence > after_sequence
                and event.correlation_id == correlation_id
            ),
            None,
        )

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._latest_failure = None
            self._failure_by_run.clear()
            self.dropped_events = 0

    def _write_console(self, event: PipelineLogEvent) -> None:
        try:
            stream = self._stream if self._stream is not None else sys.stdout
            payload = json.dumps(
                event.to_dict(include_timestamp=self.include_timestamp),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            stream.write(f"rag_index_progress {payload}\n")
            flush = getattr(stream, "flush", None)
            if callable(flush):
                flush()
            self.last_console_error_type = None
        except BaseException as exc:
            # Console output is observability and cannot alter index execution.
            self.console_error_count += 1
            self.last_console_error_type = _safe_type_name(exc)

    def _deliver_sink(self, event: PipelineLogEvent) -> None:
        try:
            assert self._sink is not None
            self._sink(event)
            self.last_sink_error_type = None
        except BaseException as exc:
            # User display/capture callbacks are observability only.
            self.sink_error_count += 1
            self.last_sink_error_type = _safe_type_name(exc)

    def _new_pipeline_correlation_id(self) -> str:
        with self._lock:
            self._correlation_sequence += 1
            return f"pipe_{self._correlation_sequence:032x}"


class _PipelineStage(
    AbstractContextManager["_PipelineStage"],
    AbstractAsyncContextManager["_PipelineStage"],
):
    def __init__(
        self,
        execution_log: PipelineExecutionLog,
        *,
        operation: str,
        stage: str,
        source_id: str | None,
        generation_id: str | None,
        item_index: int | None,
        item_count: int | None,
        counts: Mapping[str, int] | None,
        error_code: str | None,
    ) -> None:
        self._execution_log = execution_log
        self._fields = {
            "source_id": source_id,
            "generation_id": generation_id,
            "item_index": item_index,
            "item_count": item_count,
            "counts": counts,
        }
        self.operation = operation
        self.stage_name = stage
        self.error_code = error_code
        self._entered = False
        self._finished = False
        self._started_sequence: int | None = None
        self._correlation_id: str | None = None
        self._correlation_token: Token[str | None] | None = None

    def __enter__(self) -> _PipelineStage:
        if self._entered:
            raise RuntimeError("a pipeline stage scope cannot be re-entered")
        self._entered = True
        if self.stage_name == "run" and _CURRENT_CORRELATION_ID.get() is None:
            self._correlation_token = _CURRENT_CORRELATION_ID.set(
                self._execution_log._new_pipeline_correlation_id()
            )
        try:
            started = self._execution_log.emit(
                self.operation,
                self.stage_name,
                "started",
                **self._fields,
            )
        except BaseException:
            if self._correlation_token is not None:
                _CURRENT_CORRELATION_ID.reset(self._correlation_token)
                self._correlation_token = None
            self._entered = False
            raise
        self._started_sequence = started.sequence
        self._correlation_id = started.correlation_id
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        del exc_type, traceback
        try:
            self._finish(exc)
        finally:
            if self._correlation_token is not None:
                _CURRENT_CORRELATION_ID.reset(self._correlation_token)
                self._correlation_token = None
        return False

    async def __aenter__(self) -> _PipelineStage:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        return self.__exit__(exc_type, exc, traceback)

    def _finish(self, error: BaseException | None) -> None:
        if not self._entered:
            raise RuntimeError("a pipeline stage scope was not entered")
        if self._finished:
            return
        self._finished = True
        if error is not None and self._started_sequence is not None:
            inner_failure = self._execution_log._failure_after(
                self._correlation_id,
                after_sequence=self._started_sequence,
            )
            if inner_failure is not None and inner_failure.operation == self.operation:
                # An inner scope already captured the same propagating failure.
                # Keep that more precise stage as the authoritative diagnosis.
                return
        self._execution_log.emit(
            self.operation,
            self.stage_name,
            "failed" if error is not None else "completed",
            error=error,
            error_code=self.error_code if error is not None else None,
            **self._fields,
        )


class _PipelineCorrelation(AbstractContextManager["_PipelineCorrelation"]):
    def __init__(self, correlation_id: str) -> None:
        self.correlation_id = correlation_id
        self._token: Token[str | None] | None = None

    def __enter__(self) -> _PipelineCorrelation:
        if self._token is not None:
            raise RuntimeError("a pipeline correlation scope cannot be re-entered")
        self._token = _CURRENT_CORRELATION_ID.set(self.correlation_id)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        del exc_type, exc, traceback
        if self._token is None:
            raise RuntimeError("a pipeline correlation scope was not entered")
        _CURRENT_CORRELATION_ID.reset(self._token)
        self._token = None
        return False


@dataclass(frozen=True, slots=True)
class _FailureProjection:
    exception_type: str | None = None
    cause_types: tuple[str, ...] = ()
    errno: int | None = None
    http_status: int | None = None


def _failure_projection(error: BaseException | None) -> _FailureProjection:
    if error is None:
        return _FailureProjection()
    chain = _exception_chain(error)
    return _FailureProjection(
        exception_type=_safe_type_name(chain[0]),
        cause_types=tuple(_safe_type_name(item) for item in chain[1:]),
        errno=_first_safe_errno(chain),
        http_status=_first_http_status(chain),
    )


def _exception_chain(error: BaseException) -> tuple[BaseException, ...]:
    values: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and len(values) <= _MAX_CAUSE_TYPES:
        identity = id(current)
        if identity in seen:
            break
        seen.add(identity)
        values.append(current)
        cause = _base_exception_attribute(current, "__cause__")
        suppressed = _base_exception_attribute(current, "__suppress_context__")
        if cause is None and suppressed is not True:
            cause = _base_exception_attribute(current, "__context__")
        current = cause if isinstance(cause, BaseException) else None
    return tuple(values)


def _first_safe_errno(chain: tuple[BaseException, ...]) -> int | None:
    for error in chain:
        value = _static_attribute(error, "errno")
        if type(value) is int and -(2**31) <= value < 2**31:
            return value
    return None


def _first_http_status(chain: tuple[BaseException, ...]) -> int | None:
    for error in chain:
        status = _static_attribute(error, "status_code")
        if status is None:
            response = _static_attribute(error, "response")
            status = _static_attribute(response, "status_code")
        if type(status) is int and 100 <= status <= 599:
            return status
    return None


def _base_exception_attribute(error: BaseException, name: str) -> Any:
    try:
        descriptor = inspect.getattr_static(BaseException, name)
        return descriptor.__get__(error, BaseException)
    except BaseException:
        return None


def _static_attribute(value: Any, name: str) -> Any:
    if value is None:
        return None
    try:
        attribute = inspect.getattr_static(value, name, None)
    except BaseException:
        return None
    if name == "errno" and isinstance(value, OSError):
        # Bind only the trusted built-in descriptor. An arbitrary exception
        # property could run user code or reveal data while being inspected.
        descriptor = inspect.getattr_static(OSError, "errno", None)
        if descriptor is not None:
            try:
                return descriptor.__get__(value, OSError)
            except BaseException:
                return None
    if hasattr(attribute, "__get__"):
        return None
    return attribute


def _safe_type_name(value: BaseException) -> str:
    try:
        name = type.__getattribute__(type(value), "__name__")
    except BaseException:
        return "unknown"
    return name if _is_stable_label(name) else "unknown"


def _default_error_code(stage: str) -> str:
    value = f"{stage}_failed"
    return value if _is_stable_label(value) else "pipeline_stage_failed"


def _validated_counts(value: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError("counts must be a mapping")
    if len(value) > _MAX_COUNTS:
        raise ValueError("counts cannot contain more than 32 entries")
    result: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or _COUNT_LABEL.fullmatch(key) is None:
            raise ValueError("count keys must be bounded stable labels")
        if type(item) is not int or not 0 <= item <= 2**63 - 1:
            raise ValueError("count values must be bounded non-negative integers")
        result[key] = item
    return result


def _is_stable_label(value: Any) -> bool:
    return isinstance(value, str) and _STABLE_LABEL.fullmatch(value) is not None


__all__ = ["PipelineExecutionLog", "PipelineLogEvent"]
