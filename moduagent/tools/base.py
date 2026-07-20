from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from enum import Enum
from threading import Event, Thread
from typing import Any, Protocol, TypeVar, runtime_checkable

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

    def __post_init__(self) -> None:
        if self.success and self.error is not None:
            raise ValueError("a successful tool result cannot contain an error")
        if not self.success and self.error is None:
            raise ValueError("a failed tool result must contain an error")
        if self.attempts < 0:
            raise ValueError("attempts cannot be negative")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative")

    @classmethod
    def succeeded(
        cls,
        *,
        call_id: str,
        tool_name: str,
        value: Any,
        attempts: int = 1,
        duration_seconds: float = 0.0,
    ) -> "ToolResult":
        return cls(
            call_id=call_id,
            tool_name=tool_name,
            success=True,
            value=value,
            attempts=attempts,
            duration_seconds=duration_seconds,
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
    ) -> "ToolResult":
        return cls(
            call_id=call_id,
            tool_name=tool_name,
            success=False,
            error=error,
            attempts=attempts,
            duration_seconds=duration_seconds,
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
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _json_safe(value: Any) -> Any:
    """Return JSON-native data, falling back to a bounded textual value."""

    try:
        return json.loads(json.dumps(value, ensure_ascii=False, default=_json_default))
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
