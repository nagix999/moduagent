from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from moduagent.messages import Message, Usage
from moduagent.runtime.context import RunContext, RunRequest, RunStatus


_CHECKPOINT_VERSION = 1


@dataclass(frozen=True, slots=True)
class RunCheckpoint:
    """Serializable snapshot of the state required to resume a run."""

    run_id: str
    session_id: str
    messages: tuple[Message, ...]
    input: str = ""
    user_context: Mapping[str, Any] = field(default_factory=dict)
    new_messages: tuple[Message, ...] = ()
    step: int = 0
    tool_call_count: int = 0
    status: RunStatus = RunStatus.CREATED
    policy_state: Mapping[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    current_run_start: int = 0

    def __post_init__(self) -> None:
        _validate_identifier(self.run_id, "run_id")
        _validate_identifier(self.session_id, "session_id")
        if self.step < 0 or self.tool_call_count < 0:
            raise ValueError("checkpoint counters cannot be negative")
        if not 0 <= self.current_run_start <= len(self.messages):
            raise ValueError("current_run_start must reference the message sequence")
        if not isinstance(self.status, RunStatus):
            object.__setattr__(self, "status", RunStatus(str(self.status)))
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "new_messages", tuple(self.new_messages))
        if not all(isinstance(message, Message) for message in self.messages):
            raise TypeError("messages must contain Message instances")
        if not all(isinstance(message, Message) for message in self.new_messages):
            raise TypeError("new_messages must contain Message instances")

    @classmethod
    def from_context(
        cls,
        context: RunContext,
        *,
        created_at: datetime | None = None,
    ) -> "RunCheckpoint":
        now = datetime.now(timezone.utc)
        return cls(
            run_id=context.run_id,
            session_id=context.request.session_id,
            input=context.request.input,
            user_context=dict(context.request.user_context),
            messages=tuple(context.messages),
            new_messages=tuple(context.new_messages),
            step=context.step,
            tool_call_count=context.tool_call_count,
            status=context.status,
            policy_state=dict(context.policy_state),
            usage=context.usage,
            metadata=dict(context.metadata),
            created_at=created_at or now,
            updated_at=now,
            current_run_start=context.current_run_start,
        )

    def to_context(self) -> RunContext:
        request = RunRequest(
            input=self.input,
            session_id=self.session_id,
            user_context=dict(self.user_context),
            resume_run_id=self.run_id,
        )
        return RunContext(
            run_id=self.run_id,
            request=request,
            messages=list(self.messages),
            new_messages=list(self.new_messages),
            step=self.step,
            tool_call_count=self.tool_call_count,
            status=self.status,
            policy_state=dict(self.policy_state),
            usage=self.usage,
            metadata=dict(self.metadata),
            current_run_start=self.current_run_start,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": _CHECKPOINT_VERSION,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "input": self.input,
            "user_context": dict(self.user_context),
            "messages": [message.to_dict() for message in self.messages],
            "new_messages": [message.to_dict() for message in self.new_messages],
            "step": self.step,
            "tool_call_count": self.tool_call_count,
            "status": self.status.value,
            "policy_state": dict(self.policy_state),
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "total_tokens": self.usage.total_tokens,
                "provider": dict(self.usage.provider),
            },
            "metadata": dict(self.metadata),
            "current_run_start": self.current_run_start,
            "created_at": _as_utc(self.created_at).isoformat(),
            "updated_at": _as_utc(self.updated_at).isoformat(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunCheckpoint":
        version = int(value.get("version", _CHECKPOINT_VERSION))
        if version != _CHECKPOINT_VERSION:
            raise ValueError(f"unsupported checkpoint version: {version}")
        raw_usage = value.get("usage", {})
        if not isinstance(raw_usage, Mapping):
            raise ValueError("checkpoint usage must be an object")
        usage = Usage(
            input_tokens=int(raw_usage.get("input_tokens", 0)),
            output_tokens=int(raw_usage.get("output_tokens", 0)),
            total_tokens=int(raw_usage.get("total_tokens", 0)),
            provider=dict(raw_usage.get("provider", {})),
        )
        return cls(
            run_id=str(value["run_id"]),
            session_id=str(value["session_id"]),
            input=str(value.get("input", "")),
            user_context=dict(value.get("user_context", {})),
            messages=tuple(
                Message.from_dict(message) for message in value.get("messages", ())
            ),
            new_messages=tuple(
                Message.from_dict(message) for message in value.get("new_messages", ())
            ),
            step=int(value.get("step", 0)),
            tool_call_count=int(value.get("tool_call_count", 0)),
            status=RunStatus(str(value.get("status", RunStatus.CREATED.value))),
            policy_state=dict(value.get("policy_state", {})),
            usage=usage,
            metadata=dict(value.get("metadata", {})),
            created_at=_parse_datetime(value.get("created_at")),
            updated_at=_parse_datetime(value.get("updated_at")),
            # Version 1 checkpoints created before conversation memory support do
            # not contain this boundary. Treat every non-system message as part of
            # the active run because its real start cannot be inferred safely.
            current_run_start=int(
                value.get(
                    "current_run_start",
                    min(1, len(value.get("messages", ()))),
                )
            ),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> "RunCheckpoint":
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8")
        value = json.loads(payload)
        if not isinstance(value, Mapping):
            raise ValueError("checkpoint payload must be a JSON object")
        return cls.from_dict(value)


@runtime_checkable
class CheckpointStore(Protocol):
    async def load(self, run_id: str) -> RunCheckpoint | None: ...

    async def save(
        self,
        run_id: str,
        context: RunContext | RunCheckpoint,
    ) -> None: ...

    async def delete(self, run_id: str) -> None: ...


class InMemoryCheckpointStore:
    def __init__(
        self,
        *,
        ttl_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _validate_ttl(ttl_seconds)
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._checkpoints: dict[str, RunCheckpoint] = {}
        self._expires_at: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def load(self, run_id: str) -> RunCheckpoint | None:
        _validate_identifier(run_id, "run_id")
        async with self._lock:
            expires_at = self._expires_at.get(run_id)
            if expires_at is not None and expires_at <= self._clock():
                self._checkpoints.pop(run_id, None)
                self._expires_at.pop(run_id, None)
                return None
            checkpoint = self._checkpoints.get(run_id)
            return (
                None
                if checkpoint is None
                else RunCheckpoint.from_json(checkpoint.to_json())
            )

    async def save(
        self,
        run_id: str | RunCheckpoint,
        context: RunContext | RunCheckpoint | None = None,
    ) -> None:
        checkpoint = _coerce_checkpoint(run_id, context)
        # Serialize now so invalid policy/metadata state fails at the save boundary.
        stored = RunCheckpoint.from_json(checkpoint.to_json())
        async with self._lock:
            self._checkpoints[checkpoint.run_id] = stored
            if self._ttl_seconds is not None:
                self._expires_at[checkpoint.run_id] = self._clock() + self._ttl_seconds

    async def delete(self, run_id: str) -> None:
        _validate_identifier(run_id, "run_id")
        async with self._lock:
            self._checkpoints.pop(run_id, None)
            self._expires_at.pop(run_id, None)


class RedisCheckpointStore:
    """Checkpoint storage over an injected synchronous or asynchronous client."""

    def __init__(
        self,
        client: Any,
        *,
        key_prefix: str = "moduagent:checkpoint:",
        ttl_seconds: int | None = None,
    ) -> None:
        _validate_ttl(ttl_seconds)
        for method in ("get", "set", "delete"):
            if not callable(getattr(client, method, None)):
                raise TypeError(f"Redis client must provide {method}()")
        self._client = client
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds

    async def load(self, run_id: str) -> RunCheckpoint | None:
        key = self._key(run_id)
        payload = await _call(self._client.get, key)
        return None if payload is None else RunCheckpoint.from_json(payload)

    async def save(
        self,
        run_id: str | RunCheckpoint,
        context: RunContext | RunCheckpoint | None = None,
    ) -> None:
        checkpoint = _coerce_checkpoint(run_id, context)
        await _redis_set(
            self._client,
            self._key(checkpoint.run_id),
            checkpoint.to_json(),
            self._ttl_seconds,
        )

    async def delete(self, run_id: str) -> None:
        await _call(self._client.delete, self._key(run_id))

    def _key(self, run_id: str) -> str:
        _validate_identifier(run_id, "run_id")
        return f"{self._key_prefix}{run_id}"


async def _redis_set(
    client: Any, key: str, payload: str, ttl_seconds: int | None
) -> None:
    if ttl_seconds is None:
        await _call(client.set, key, payload)
        return
    try:
        await _call(client.set, key, payload, ex=ttl_seconds)
    except TypeError:
        await _call(client.set, key, payload)
        expire = getattr(client, "expire", None)
        if not callable(expire):
            raise TypeError(
                "Redis client must provide expire when set(ex=) is unsupported"
            )
        await _call(expire, key, ttl_seconds)


async def _call(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    result = function(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")


def _validate_ttl(value: float | None) -> None:
    if value is not None and value <= 0:
        raise ValueError("ttl_seconds must be positive")


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_datetime(value: Any) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return _as_utc(parsed)


def _coerce_checkpoint(
    run_id: str | RunCheckpoint,
    context: RunContext | RunCheckpoint | None,
) -> RunCheckpoint:
    """Implement the PDF ``save(run_id, context)`` contract.

    Passing a RunCheckpoint alone remains supported as a compact adapter API.
    """

    if isinstance(run_id, RunCheckpoint) and context is None:
        return run_id
    if not isinstance(run_id, str):
        raise TypeError("run_id must be a string")
    _validate_identifier(run_id, "run_id")
    if isinstance(context, RunContext):
        checkpoint = RunCheckpoint.from_context(context)
    elif isinstance(context, RunCheckpoint):
        checkpoint = context
    else:
        raise TypeError("context must be a RunContext or RunCheckpoint")
    if checkpoint.run_id != run_id:
        raise ValueError("run_id does not match context.run_id")
    return checkpoint


__all__ = [
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "RedisCheckpointStore",
    "RunCheckpoint",
]
