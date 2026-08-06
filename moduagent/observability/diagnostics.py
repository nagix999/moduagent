from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import os
import re
import uuid
import weakref
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError

from moduagent.models.errors import ModelOutputIncompleteError
from moduagent.observability._background import run_in_daemon_thread


_MAX_IDENTIFIER_CHARS = 256
_MAX_DETAIL_DEPTH = 4
_MAX_DETAIL_ITEMS = 32
_MAX_DETAIL_TEXT_CHARS = 256
_MAX_DETAIL_BYTES = 4096
_MAX_VALIDATION_ERRORS = 20
_MAX_VALIDATION_LOCATION_ITEMS = 16
_MAX_RECENT_FAILURE_IDS = 4096
_SQLSTATE_PATTERN = re.compile(r"^[A-Za-z0-9]{5}$")
_SENSITIVE_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "database_url",
        "dsn",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "set_cookie",
        "sql",
        "token",
    }
)
_SENSITIVE_SUFFIXES = (
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


class _FrozenDict(dict[str, Any]):
    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("diagnostic mappings are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenDict:
        del memo
        return self


@dataclass(frozen=True, slots=True)
class DiagnosticFrame:
    filename: str
    function: str
    lineno: int

    def __post_init__(self) -> None:
        filename = _bounded_identifier(
            self.filename.replace("\\", "/").rsplit("/", 1)[-1],
            fallback="unknown",
        )
        function = _bounded_identifier(self.function, fallback="unknown")
        if type(self.lineno) is not int or self.lineno < 0:
            raise ValueError("diagnostic frame lineno must be a non-negative integer")
        object.__setattr__(self, "filename", filename)
        object.__setattr__(self, "function", function)

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "function": self.function,
            "lineno": self.lineno,
        }


@dataclass(frozen=True, slots=True)
class FailureDiagnostic:
    failure_id: str
    run_id: str
    component: str
    operation: str
    category: str
    code: str
    exception_type: str
    cause_types: tuple[str, ...] = ()
    safe_details: Mapping[str, Any] = field(default_factory=dict)
    frames: tuple[DiagnosticFrame, ...] = ()
    phase: str | None = None
    step_id: str | None = None
    call_id: str | None = None
    tool_name: str | None = None
    attempt: int | None = None
    terminal: bool = True
    retryable: bool = False
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: int = 1

    def __post_init__(self) -> None:
        for field_name in (
            "failure_id",
            "run_id",
            "component",
            "operation",
            "category",
            "code",
            "exception_type",
        ):
            object.__setattr__(
                self,
                field_name,
                _bounded_identifier(
                    getattr(self, field_name),
                    fallback="unknown",
                ),
            )
        causes = tuple(
            _bounded_identifier(value, fallback="unknown") for value in self.cause_types
        )
        if not all(isinstance(value, str) for value in self.cause_types):
            raise TypeError("diagnostic cause_types must contain strings")
        object.__setattr__(self, "cause_types", causes)
        if not isinstance(self.safe_details, Mapping):
            raise TypeError("diagnostic safe_details must be a mapping")
        object.__setattr__(
            self,
            "safe_details",
            _freeze_safe_details(self.safe_details),
        )
        frames = tuple(self.frames)
        if not all(isinstance(value, DiagnosticFrame) for value in frames):
            raise TypeError("diagnostic frames must contain DiagnosticFrame values")
        object.__setattr__(self, "frames", frames)
        for field_name in ("phase", "step_id", "call_id", "tool_name"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _bounded_identifier(value, fallback="unknown"),
                )
        if self.attempt is not None and (
            type(self.attempt) is not int or self.attempt < 1
        ):
            raise ValueError("diagnostic attempt must be a positive integer or None")
        if type(self.terminal) is not bool:
            raise TypeError("diagnostic terminal must be a bool")
        if type(self.retryable) is not bool:
            raise TypeError("diagnostic retryable must be a bool")
        if not isinstance(self.occurred_at, datetime):
            raise TypeError("diagnostic occurred_at must be a datetime")
        object.__setattr__(self, "occurred_at", _as_utc(self.occurred_at))
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported failure diagnostic schema version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "failure_id": self.failure_id,
            "run_id": self.run_id,
            "component": self.component,
            "operation": self.operation,
            "category": self.category,
            "code": self.code,
            "exception_type": self.exception_type,
            "cause_types": list(self.cause_types),
            "safe_details": _thaw(self.safe_details),
            "frames": [frame.to_dict() for frame in self.frames],
            "phase": self.phase,
            "step_id": self.step_id,
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "attempt": self.attempt,
            "terminal": self.terminal,
            "retryable": self.retryable,
            "occurred_at": self.occurred_at.isoformat(),
        }


@runtime_checkable
class DiagnosticSink(Protocol):
    async def capture(self, record: FailureDiagnostic) -> None: ...


class NoopDiagnosticSink:
    async def capture(self, record: FailureDiagnostic) -> None:
        _validate_record(record)


class InMemoryDiagnosticSink:
    def __init__(self, *, max_records: int = 1000) -> None:
        if type(max_records) is not int:
            raise TypeError("max_records must be an integer")
        if max_records < 1:
            raise ValueError("max_records must be at least 1")
        self.max_records = max_records
        self._records: list[FailureDiagnostic] = []
        self._by_id: dict[str, FailureDiagnostic] = {}

    @property
    def records(self) -> tuple[FailureDiagnostic, ...]:
        return tuple(self._records)

    async def capture(self, record: FailureDiagnostic) -> None:
        _validate_record(record)
        existing = self._by_id.get(record.failure_id)
        if existing is not None:
            if existing != record:
                raise ValueError("diagnostic failure_id already has another record")
            return
        if len(self._records) >= self.max_records:
            evicted = self._records.pop(0)
            self._by_id.pop(evicted.failure_id, None)
        self._records.append(record)
        self._by_id[record.failure_id] = record

    def get(self, failure_id: str) -> FailureDiagnostic | None:
        return self._by_id.get(failure_id)

    def for_run(self, run_id: str) -> tuple[FailureDiagnostic, ...]:
        return tuple(record for record in self._records if record.run_id == run_id)

    def clear(self) -> None:
        self._records.clear()
        self._by_id.clear()


class LoggingDiagnosticSink:
    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        level: int = logging.ERROR,
    ) -> None:
        self.logger = logger or logging.getLogger("moduagent.diagnostics")
        self.level = level

    async def capture(self, record: FailureDiagnostic) -> None:
        _validate_record(record)
        message = json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        await run_in_daemon_thread(
            self.logger.log,
            self.level,
            "agent_failure %s",
            message,
        )


class CompositeDiagnosticSink:
    def __init__(
        self,
        sinks: Iterable[DiagnosticSink] | DiagnosticSink = (),
        *additional_sinks: DiagnosticSink,
    ) -> None:
        if callable(getattr(sinks, "capture", None)):
            self.sinks = (sinks, *additional_sinks)
        else:
            self.sinks = (*tuple(sinks), *additional_sinks)
        self.last_errors: tuple[BaseException, ...] = ()

    async def capture(self, record: FailureDiagnostic) -> None:
        _validate_record(record)
        results = await asyncio.gather(
            *(_capture_sink(sink, record) for sink in self.sinks),
            return_exceptions=True,
        )
        self.last_errors = tuple(
            result for result in results if isinstance(result, BaseException)
        )
        if not results:
            raise RuntimeError("composite diagnostic sink has no children")
        if len(self.last_errors) == len(results):
            raise RuntimeError("all diagnostic sinks rejected the record")


class DiagnosticReporter:
    def __init__(
        self,
        sink: DiagnosticSink | None = None,
        *,
        timeout_seconds: float = 0.25,
        max_cause_depth: int = 5,
        max_frames: int = 32,
        max_pending_deliveries: int = 1024,
        failure_id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ValueError("diagnostic timeout_seconds must be a positive number")
        for value, field_name in (
            (max_cause_depth, "max_cause_depth"),
            (max_frames, "max_frames"),
            (max_pending_deliveries, "max_pending_deliveries"),
        ):
            if type(value) is not int:
                raise TypeError(f"{field_name} must be an integer")
            minimum = 1 if field_name == "max_pending_deliveries" else 0
            if value < minimum:
                raise ValueError(f"{field_name} must be at least {minimum}")
        self.sink = sink if sink is not None else NoopDiagnosticSink()
        self.timeout_seconds = float(timeout_seconds)
        self.max_cause_depth = max_cause_depth
        self.max_frames = max_frames
        self.max_pending_deliveries = max_pending_deliveries
        self._failure_id_factory = failure_id_factory or (
            lambda: f"diag_{uuid.uuid4().hex}"
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._recent_failure_ids: deque[str] = deque()
        self._recent_failure_id_set: set[str] = set()
        self._pending: dict[str, set[asyncio.Task[None]]] = {}
        self._timeout_handles: dict[
            asyncio.Task[None],
            asyncio.TimerHandle,
        ] = {}
        self._accounted_cancellations: weakref.WeakSet[asyncio.Task[None]] = (
            weakref.WeakSet()
        )
        self.drop_count = 0
        self.last_error: BaseException | None = None

    async def capture_exception(
        self,
        exception: BaseException,
        *,
        run_id: str,
        component: str,
        operation: str,
        category: str,
        code: str,
        phase: str | None = None,
        step_id: str | None = None,
        call_id: str | None = None,
        tool_name: str | None = None,
        attempt: int | None = None,
        terminal: bool = True,
        retryable: bool = False,
        safe_details: Mapping[str, Any] | None = None,
        validation_fields: Iterable[str] | None = None,
    ) -> str | None:
        if not isinstance(exception, BaseException):
            raise TypeError("exception must be a BaseException")
        if safe_details is not None and not isinstance(safe_details, Mapping):
            raise TypeError("safe_details must be a mapping or None")
        known_validation_fields = _validation_field_names(validation_fields)
        if isinstance(self.sink, NoopDiagnosticSink):
            return None
        run_key = _bounded_identifier(run_id, fallback="unknown")

        chain = _exception_chain(exception, self.max_cause_depth)
        details = dict(safe_details or {})
        details.update(
            {
                key: value
                for key, value in _extract_safe_details(
                    chain,
                    validation_fields=known_validation_fields,
                ).items()
                if key not in details
            }
        )
        record = FailureDiagnostic(
            failure_id=self._new_failure_id(),
            run_id=run_key,
            component=component,
            operation=operation,
            category=category,
            code=code,
            exception_type=_exception_type(exception),
            cause_types=tuple(_exception_type(item) for item in chain[1:]),
            safe_details=details,
            frames=_extract_frames(chain, self.max_frames),
            phase=phase,
            step_id=step_id,
            call_id=call_id,
            tool_name=tool_name,
            attempt=attempt,
            terminal=terminal,
            retryable=retryable,
            occurred_at=self._clock(),
        )

        self._schedule_delivery(run_key, record)
        # Give sinks that complete without blocking one deterministic turn.
        # Slow sinks continue in the background and cannot consume run budget.
        await asyncio.sleep(0)
        return record.failure_id

    async def flush_run(self, run_id: str) -> None:
        run_key = _bounded_identifier(run_id, fallback="unknown")
        pending = tuple(self._pending.get(run_key, ()))
        if not pending:
            return
        _, unfinished = await asyncio.wait(
            pending,
            timeout=self.timeout_seconds,
        )
        for task in unfinished:
            self._timeout_delivery(task)
        # Run normal callbacks for cooperative cancellations. A sink that
        # suppresses cancellation remains detached but cannot block the run.
        await asyncio.sleep(0)

    def clear_run(self, run_id: str) -> None:
        run_key = _bounded_identifier(run_id, fallback="unknown")
        pending = tuple(self._pending.pop(run_key, ()))
        for task in pending:
            if task.done():
                continue
            if task not in self._accounted_cancellations:
                self._accounted_cancellations.add(task)
                self.drop_count += 1
                self.last_error = asyncio.CancelledError()
            handle = self._timeout_handles.pop(task, None)
            if handle is not None:
                handle.cancel()
            task.cancel()

    def _new_failure_id(self) -> str:
        value = self._failure_id_factory()
        candidate = _bounded_identifier(
            value,
            fallback=f"diag_{uuid.uuid4().hex}",
        )
        while candidate in self._recent_failure_id_set:
            candidate = f"diag_{uuid.uuid4().hex}"
        if len(self._recent_failure_ids) >= _MAX_RECENT_FAILURE_IDS:
            evicted = self._recent_failure_ids.popleft()
            self._recent_failure_id_set.discard(evicted)
        self._recent_failure_ids.append(candidate)
        self._recent_failure_id_set.add(candidate)
        return candidate

    def _schedule_delivery(
        self,
        run_id: str,
        record: FailureDiagnostic,
    ) -> None:
        if len(self._timeout_handles) >= self.max_pending_deliveries:
            self.drop_count += 1
            self.last_error = RuntimeError("diagnostic pending delivery limit reached")
            return
        loop = asyncio.get_running_loop()
        task = loop.create_task(self._deliver(record))
        self._pending.setdefault(run_id, set()).add(task)
        self._timeout_handles[task] = loop.call_later(
            self.timeout_seconds,
            self._timeout_delivery,
            task,
        )
        task.add_done_callback(
            lambda completed, run_id=run_id: self._delivery_done(
                run_id,
                completed,
            )
        )

    async def _deliver(self, record: FailureDiagnostic) -> None:
        try:
            await _capture_sink(self.sink, record)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self.drop_count += 1
            self.last_error = exc
        else:
            current = asyncio.current_task()
            if current not in self._accounted_cancellations:
                self.last_error = None

    def _timeout_delivery(self, task: asyncio.Task[None]) -> None:
        if task.done():
            return
        if task not in self._accounted_cancellations:
            self._accounted_cancellations.add(task)
            self.drop_count += 1
            self.last_error = asyncio.TimeoutError()
        task.cancel()

    def _delivery_done(
        self,
        run_id: str,
        task: asyncio.Task[None],
    ) -> None:
        handle = self._timeout_handles.pop(task, None)
        if handle is not None:
            handle.cancel()
        pending = self._pending.get(run_id)
        if pending is not None:
            pending.discard(task)
            if not pending:
                self._pending.pop(run_id, None)
        if task.cancelled() and task not in self._accounted_cancellations:
            self.drop_count += 1
            self.last_error = asyncio.CancelledError()
        self._accounted_cancellations.discard(task)


async def _capture_sink(sink: Any, record: FailureDiagnostic) -> None:
    capture = getattr(sink, "capture", None)
    if not callable(capture):
        raise TypeError("diagnostic sink must provide capture()")
    if inspect.iscoroutinefunction(capture):
        await capture(record)
        return
    result = await run_in_daemon_thread(capture, record)
    if inspect.isawaitable(result):
        await result


def _validate_record(record: FailureDiagnostic) -> None:
    if not isinstance(record, FailureDiagnostic):
        raise TypeError("diagnostic sink requires a FailureDiagnostic")


def _bounded_identifier(value: Any, *, fallback: str) -> str:
    if not isinstance(value, str):
        raise TypeError("diagnostic identifiers must be strings")
    text = " ".join(
        "".join(character if character.isprintable() else " " for character in value)
        .strip()
        .split()
    )
    text = text or fallback
    if len(text) <= _MAX_IDENTIFIER_CHARS:
        return text
    return text[:_MAX_IDENTIFIER_CHARS]


def _normalize_key(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return text


def _sensitive_key(value: Any) -> bool:
    key = _normalize_key(value)
    return key in _SENSITIVE_KEYS or key.endswith(_SENSITIVE_SUFFIXES)


def _freeze_safe_details(value: Mapping[str, Any]) -> Mapping[str, Any]:
    bounded = _safe_value(value, depth=0)
    if not isinstance(bounded, Mapping):
        bounded = {}
    serialized = json.dumps(
        _thaw(bounded),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(serialized) > _MAX_DETAIL_BYTES:
        return _FrozenDict(
            {
                "truncated": True,
                "original_bytes": len(serialized),
            }
        )
    return bounded


def _safe_value(value: Any, *, depth: int) -> Any:
    if depth > _MAX_DETAIL_DEPTH:
        return "[TRUNCATED]"
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return _safe_value(value.value, depth=depth)
    if isinstance(value, str):
        text = "".join(
            character if character.isprintable() else " " for character in value
        )
        return text[:_MAX_DETAIL_TEXT_CHARS]
    if isinstance(value, Mapping):
        items: dict[str, Any] = {}
        for index, (key, nested) in enumerate(value.items()):
            if index >= _MAX_DETAIL_ITEMS:
                items["truncated"] = True
                break
            safe_key = _bounded_identifier(str(key), fallback="unknown")
            items[safe_key] = (
                "[REDACTED]"
                if _sensitive_key(safe_key)
                else _safe_value(nested, depth=depth + 1)
            )
        return _FrozenDict(items)
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        result = tuple(
            _safe_value(item, depth=depth + 1) for item in items[:_MAX_DETAIL_ITEMS]
        )
        if len(items) > _MAX_DETAIL_ITEMS:
            return (*result, "[TRUNCATED]")
        return result
    return _FrozenDict({"unsupported_type": _exception_type(value)})


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _exception_type(value: Any) -> str:
    value_type = type(value)
    try:
        name = type.__getattribute__(value_type, "__name__")
    except BaseException:
        name = "unknown"
    return _bounded_identifier(name, fallback="unknown")


def _exception_chain(
    error: BaseException,
    max_cause_depth: int,
) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and len(chain) <= max_cause_depth:
        identity = id(current)
        if identity in seen:
            break
        seen.add(identity)
        chain.append(current)
        cause = _base_exception_attribute(current, "__cause__")
        suppress_context = _base_exception_attribute(
            current,
            "__suppress_context__",
        )
        if cause is None and suppress_context is not True:
            cause = _base_exception_attribute(current, "__context__")
        current = cause if isinstance(cause, BaseException) else None
    return tuple(chain)


def _extract_frames(
    chain: tuple[BaseException, ...],
    max_frames: int,
) -> tuple[DiagnosticFrame, ...]:
    if max_frames == 0:
        return ()
    # Keep the most recent frames so a deep stack retains the innermost
    # failure site instead of only its outer callers.
    frames: deque[DiagnosticFrame] = deque(maxlen=max_frames)
    for error in chain:
        traceback = _base_exception_attribute(error, "__traceback__")
        while traceback is not None:
            code = traceback.tb_frame.f_code
            frames.append(
                DiagnosticFrame(
                    filename=os.path.basename(code.co_filename),
                    function=code.co_name,
                    lineno=traceback.tb_lineno,
                )
            )
            traceback = traceback.tb_next
    return tuple(frames)


def _base_exception_attribute(error: BaseException, name: str) -> Any:
    """Read trusted BaseException state without subclass attribute hooks."""

    try:
        descriptor = inspect.getattr_static(BaseException, name)
        return descriptor.__get__(error, BaseException)
    except BaseException:
        return None


def _extract_safe_details(
    chain: tuple[BaseException, ...],
    *,
    validation_fields: frozenset[str],
) -> Mapping[str, Any]:
    details: dict[str, Any] = {}
    validation_errors: list[Mapping[str, Any]] = []
    for error in chain:
        if type(error) is ModelOutputIncompleteError and error.finish_reason in {
            "timeout",
            "length",
            "max_tokens",
        }:
            details.setdefault(
                "provider_finish_reason",
                error.finish_reason,
            )
        sqlstate = _safe_attribute(error, "sqlstate")
        if sqlstate is None:
            sqlstate = _safe_attribute(error, "pgcode")
        if (
            "sqlstate" not in details
            and isinstance(sqlstate, str)
            and _SQLSTATE_PATTERN.fullmatch(sqlstate) is not None
        ):
            details["sqlstate"] = sqlstate.upper()

        errno = _safe_attribute(error, "errno")
        if "errno" not in details and type(errno) is int:
            details["errno"] = errno

        status = _safe_attribute(error, "status_code")
        if status is None:
            response = _safe_attribute(error, "response")
            status = _safe_attribute(response, "status_code")
        if (
            "http_status" not in details
            and type(status) is int
            and 100 <= status <= 599
        ):
            details["http_status"] = status

        if isinstance(error, ValidationError):
            validation_errors.extend(
                _pydantic_error_details(
                    error,
                    validation_fields=validation_fields,
                )
            )
    if validation_errors:
        details["validation_errors"] = validation_errors[:_MAX_VALIDATION_ERRORS]
    return details


def _safe_attribute(value: Any, name: str) -> Any:
    if value is None:
        return None
    try:
        # Static lookup avoids invoking exception properties, descriptors, or
        # user-defined __getattribute__ while diagnostics are on the run path.
        attribute = inspect.getattr_static(value, name, None)
    except BaseException:
        return None
    if name == "errno" and isinstance(value, OSError):
        # Bind the trusted built-in descriptor directly. This preserves useful
        # errno data without invoking a hostile subclass property.
        descriptor = inspect.getattr_static(OSError, "errno", None)
        if descriptor is not None:
            try:
                return descriptor.__get__(value, OSError)
            except BaseException:
                return None
    if hasattr(attribute, "__get__"):
        # Unknown descriptors are deliberately not evaluated on the run path.
        return None
    return attribute


def _pydantic_error_details(
    error: ValidationError,
    *,
    validation_fields: frozenset[str],
) -> list[Mapping[str, Any]]:
    try:
        raw_errors = error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
    except TypeError:
        try:
            raw_errors = error.errors()
        except Exception:
            return []
    except Exception:
        return []
    details: list[Mapping[str, Any]] = []
    for item in raw_errors[:_MAX_VALIDATION_ERRORS]:
        if not isinstance(item, Mapping):
            continue
        error_type = item.get("type")
        location = item.get("loc", ())
        if not isinstance(error_type, str):
            continue
        if not isinstance(location, (list, tuple)):
            location = ()
        safe_location: list[str | int] = []
        for part in location[:_MAX_VALIDATION_LOCATION_ITEMS]:
            if isinstance(part, str):
                safe_location.append(
                    part[:_MAX_DETAIL_TEXT_CHARS]
                    if part in validation_fields
                    else "[DYNAMIC_KEY]"
                )
            elif type(part) is int:
                safe_location.append("[INDEX_OR_KEY]")
            else:
                safe_location.append(_exception_type(part))
        details.append(
            {
                "type": error_type[:_MAX_DETAIL_TEXT_CHARS],
                "loc": safe_location,
            }
        )
    return details


def _validation_field_names(
    values: Iterable[str] | None,
) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, (str, bytes)):
        raise TypeError("validation_fields must be an iterable of strings")
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("validation_fields must contain strings")
        result.add(value[:_MAX_DETAIL_TEXT_CHARS])
        if len(result) >= _MAX_DETAIL_ITEMS * _MAX_DETAIL_DEPTH:
            break
    return frozenset(result)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = [
    "CompositeDiagnosticSink",
    "DiagnosticFrame",
    "DiagnosticReporter",
    "DiagnosticSink",
    "FailureDiagnostic",
    "InMemoryDiagnosticSink",
    "LoggingDiagnosticSink",
    "NoopDiagnosticSink",
]
