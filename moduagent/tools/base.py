from __future__ import annotations

import asyncio
import base64
import json
import math
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from threading import Event, Thread
from typing import Any, Protocol, TypeVar, runtime_checkable
from uuid import UUID

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class ToolSchema(Mapping[str, Any]):
    """Canonical model-facing JSON schema for a tool.

    ``ToolSchema`` is also a mapping so existing integrations that expect an
    OpenAI-style dictionary can use it without an explicit conversion.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("tool name cannot be empty")
        if not isinstance(self.parameters, Mapping):
            raise TypeError("tool parameters must be a mapping")

    def to_function_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"type": "function", "function": self.to_function_dict()}

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(("type", "function"))

    def __len__(self) -> int:
        return 2


@dataclass(frozen=True, slots=True)
class ToolExecutionContext(Mapping[str, Any]):
    """Execution metadata supplied to authorizers and tools.

    The mapping view delegates to ``user_context`` for compatibility with
    simple authorizers that previously received only that dictionary.
    """

    run_id: str = ""
    session_id: str | None = None
    user_context: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    tool_call_id: str | None = None
    attempt: int = 1

    def __post_init__(self) -> None:
        if self.attempt < 1:
            raise ValueError("attempt must be at least 1")

    def for_call(self, call_id: str, *, attempt: int = 1) -> "ToolExecutionContext":
        return replace(self, tool_call_id=call_id, attempt=attempt)

    def __getitem__(self, key: str) -> Any:
        return self.user_context[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.user_context)

    def __len__(self) -> int:
        return len(self.user_context)


class ToolErrorType(str, Enum):
    NOT_FOUND = "not_found"
    UNKNOWN_TOOL = "not_found"
    INVALID_ARGUMENTS = "invalid_arguments"
    VALIDATION_ERROR = "invalid_arguments"
    UNAUTHORIZED = "unauthorized"
    AUTHORIZATION_DENIED = "unauthorized"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"
    EXECUTION_FAILED = "execution_error"
    RESULT_TOO_LARGE = "result_too_large"
    CANCELLED = "cancelled"


# Both names are exported because integrations commonly call this either a
# type or a code. They intentionally refer to the same enum.
ToolErrorCode = ToolErrorType
ToolErrorKind = ToolErrorType


@dataclass(frozen=True, slots=True)
class ToolError:
    type: ToolErrorType
    message: str
    retryable: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.type, ToolErrorType):
            object.__setattr__(self, "type", ToolErrorType(str(self.type)))

    @property
    def code(self) -> ToolErrorType:
        return self.type

    @property
    def kind(self) -> ToolErrorType:
        return self.type

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "type": self.type.value,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.details:
            value["details"] = _json_safe(dict(self.details))
        return value


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    tool_name: str
    success: bool
    value: Any = None
    error: ToolError | None = None
    attempts: int = 0
    duration_seconds: float = 0.0
    # The validated/default-expanded arguments that reached invoke_validated().
    # Deliberately omitted from to_dict()/model_content() to avoid widening the
    # public Tool result and model-visible payload with potentially sensitive data.
    invocation_arguments: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.success and self.error is not None:
            raise ValueError("a successful tool result cannot contain an error")
        if not self.success and self.error is None:
            raise ValueError("a failed tool result must contain an error")
        if self.attempts < 0:
            raise ValueError("attempts cannot be negative")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative")
        if self.invocation_arguments is not None:
            if not isinstance(self.invocation_arguments, Mapping):
                raise TypeError("invocation_arguments must be a mapping")
            object.__setattr__(
                self,
                "invocation_arguments",
                dict(self.invocation_arguments),
            )

    @classmethod
    def succeeded(
        cls,
        *,
        call_id: str,
        tool_name: str,
        value: Any,
        attempts: int = 1,
        duration_seconds: float = 0.0,
        invocation_arguments: Mapping[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            call_id=call_id,
            tool_name=tool_name,
            success=True,
            value=value,
            attempts=attempts,
            duration_seconds=duration_seconds,
            invocation_arguments=invocation_arguments,
        )

    @classmethod
    def failed(
        cls,
        *,
        call_id: str,
        tool_name: str,
        error: ToolError,
        attempts: int = 0,
        duration_seconds: float = 0.0,
        invocation_arguments: Mapping[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            call_id=call_id,
            tool_name=tool_name,
            success=False,
            error=error,
            attempts=attempts,
            duration_seconds=duration_seconds,
            invocation_arguments=invocation_arguments,
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "attempts": self.attempts,
            "duration_seconds": self.duration_seconds,
        }
        if self.success:
            value["value"] = _json_safe(self.value)
        else:
            value["error"] = self.error.to_dict() if self.error else None
        return value

    def model_content(self) -> str:
        """Serialize a compact, provider-independent tool message payload."""

        payload: dict[str, Any] = {"success": self.success}
        if self.success:
            payload["value"] = _json_safe(self.value)
        else:
            payload["error"] = self.error.to_dict() if self.error else None
        return json.dumps(payload, ensure_ascii=False, default=_json_default)

    def to_message_content(self) -> str:
        return self.model_content()


@runtime_checkable
class Tool(Protocol):
    name: str
    description: str
    idempotent: bool
    timeout_seconds: float | None
    max_result_bytes: int | None

    @property
    def schema(self) -> ToolSchema: ...

    def validate_arguments(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]: ...

    async def invoke(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> Any: ...


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if _is_pandas_missing(value):
        return None
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (Decimal, UUID)):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if _is_dataframe_like(value):
        return value.to_dict(orient="records")
    if _is_numpy_value(value):
        converter = value.tolist if hasattr(value, "tolist") else value.item
        return converter()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _is_dataframe_like(value: Any) -> bool:
    """Recognize pandas-compatible tabular frames without importing pandas."""

    try:
        return (
            getattr(value, "ndim", None) == 2
            and hasattr(value, "columns")
            and hasattr(value, "index")
            and callable(getattr(value, "to_dict", None))
        )
    except Exception:
        return False


def _is_numpy_value(value: Any) -> bool:
    module = type(value).__module__
    return module == "numpy" or module.startswith("numpy.")


def _is_pandas_missing(value: Any) -> bool:
    module = type(value).__module__
    return module.startswith("pandas.") and type(value).__name__ in {
        "NAType",
        "NaTType",
    }


def _json_key(value: Any, seen: set[int]) -> str:
    normalized = _normalize_json(value, seen)
    if isinstance(normalized, str):
        return normalized
    if normalized is None:
        return "null"
    if normalized is True:
        return "true"
    if normalized is False:
        return "false"
    if isinstance(normalized, (int, float)):
        return str(normalized)
    return repr(normalized)


def _normalize_json(value: Any, seen: set[int]) -> Any:
    """Recursively normalize common tool values to JSON-native structures."""

    if isinstance(value, Enum):
        return _normalize_json(value.value, seen)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if _is_pandas_missing(value):
        return None
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (Decimal, UUID)):
        return str(value)
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}

    value_id = id(value)
    if value_id in seen:
        return "<recursive reference>"
    seen.add(value_id)
    try:
        if isinstance(value, BaseModel):
            return _normalize_json(value.model_dump(mode="json"), seen)
        if is_dataclass(value) and not isinstance(value, type):
            return _normalize_json(asdict(value), seen)
        if isinstance(value, Mapping):
            return {
                _json_key(key, seen): _normalize_json(item, seen)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple, set, frozenset)):
            return [_normalize_json(item, seen) for item in value]

        if _is_dataframe_like(value):
            try:
                records = value.to_dict(orient="records")
            except Exception:
                return repr(value)
            return _normalize_json(records, seen)

        if _is_numpy_value(value):
            try:
                if hasattr(value, "tolist") and callable(value.tolist):
                    converted = value.tolist()
                elif hasattr(value, "item") and callable(value.item):
                    converted = value.item()
                else:
                    return repr(value)
            except Exception:
                return repr(value)
            return _normalize_json(converted, seen)

        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                converted = to_dict()
            except Exception:
                return repr(value)
            return _normalize_json(converted, seen)
        return repr(value)
    finally:
        seen.remove(value_id)


def _json_safe(value: Any) -> Any:
    """Return recursively normalized JSON-native tool data."""

    try:
        normalized = _normalize_json(value, set())
        return json.loads(json.dumps(normalized, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, OverflowError):
        return repr(value)


def _serialized_size(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            default=_json_default,
            separators=(",", ":"),
        ).encode("utf-8")
    )


T = TypeVar("T")


async def _run_sync_in_daemon(function: Callable[[], T]) -> T:
    """Run blocking code without using a non-daemon executor worker.

    A timed-out function may continue in the background, but its daemon thread
    neither blocks the event loop nor prevents interpreter shutdown.
    """

    completed = Event()
    outcome: dict[str, Any] = {}

    def run() -> None:
        try:
            outcome["value"] = function()
        except BaseException as exc:  # re-raised on the event-loop thread
            outcome["error"] = exc
        finally:
            completed.set()

    Thread(target=run, daemon=True).start()
    while not completed.is_set():
        await asyncio.sleep(0.001)
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


async def _await_if_needed(value: T | Awaitable[T]) -> T:
    if isinstance(value, Awaitable):
        return await value
    return value
