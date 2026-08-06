from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from moduagent.errors import PersistenceError
from moduagent.messages import Message
from moduagent.persistence._sync import call_maybe_async


_SERIALIZATION_VERSION = 1


class ConversationStoreCapacityError(PersistenceError):
    """An in-memory conversation cannot fit within its configured capacity."""


def serialize_messages(messages: Sequence[Message]) -> str:
    """Serialize a conversation without relying on provider-specific models."""

    return json.dumps(
        {
            "version": _SERIALIZATION_VERSION,
            "messages": [message.to_dict() for message in messages],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def deserialize_messages(payload: str | bytes | bytearray | None) -> list[Message]:
    if payload is None:
        return []
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    value = json.loads(payload)
    # Accept the original list-only form as well as the versioned envelope.
    rows = value if isinstance(value, list) else value.get("messages", [])
    if not isinstance(rows, list):
        raise ValueError("conversation payload must contain a message list")
    return [Message.from_dict(row) for row in rows]


@runtime_checkable
class ConversationStore(Protocol):
    async def load(self, session_id: str) -> list[Message]: ...

    async def append(self, session_id: str, messages: Sequence[Message]) -> None: ...

    async def clear(self, session_id: str) -> None: ...


@runtime_checkable
class IdempotentConversationStore(Protocol):
    """Optional durable capability used for crash-safe run persistence."""

    supports_idempotent_append: bool

    async def append_once(
        self,
        session_id: str,
        idempotency_key: str,
        messages: Sequence[Message],
    ) -> bool:
        """Atomically append once; return ``False`` for an identical replay."""

        ...


class InMemoryConversationStore:
    """Bounded conversation storage for tests and single-process development.

    ``max_total_bytes`` measures the UTF-8 JSON rows retained for messages. It
    is a deterministic storage bound rather than an estimate of Python object
    overhead. Capacity eviction is session-granular and least-recently-used;
    the session currently being appended is never silently evicted.

    TTL cleanup is lazy so constructing this store never creates a background
    task. Expired sessions are swept at the configured interval during normal
    operations, or explicitly through :meth:`sweep_expired`.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float | None = None,
        max_sessions: int | None = None,
        max_total_bytes: int | None = None,
        ttl_sweep_interval_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _validate_ttl(ttl_seconds)
        _validate_optional_positive_int(max_sessions, "max_sessions")
        _validate_optional_positive_int(max_total_bytes, "max_total_bytes")
        if (
            isinstance(ttl_sweep_interval_seconds, bool)
            or not isinstance(ttl_sweep_interval_seconds, (int, float))
            or not math.isfinite(float(ttl_sweep_interval_seconds))
            or ttl_sweep_interval_seconds <= 0
        ):
            raise ValueError("ttl_sweep_interval_seconds must be positive")
        self._ttl_seconds = ttl_seconds
        self._max_sessions = max_sessions
        self._max_total_bytes = max_total_bytes
        self._ttl_sweep_interval_seconds = float(ttl_sweep_interval_seconds)
        self._clock = clock
        # Retain immutable JSON rows rather than caller-owned Message objects.
        # Message is a frozen dataclass, but its metadata and Tool arguments can
        # contain mutable nested values that would otherwise invalidate byte
        # accounting after an append or load.
        self._messages: dict[str, list[str]] = {}
        self._idempotency: dict[str, dict[str, str]] = {}
        self._expires_at: dict[str, float] = {}
        self._session_bytes: dict[str, int] = {}
        self._total_bytes = 0
        self._last_access: dict[str, int] = {}
        self._access_sequence = 0
        self._next_sweep_at = self._clock()
        self._lock = asyncio.Lock()
        self.supports_idempotent_append = True

    async def load(self, session_id: str) -> list[Message]:
        _validate_identifier(session_id, "session_id")
        async with self._lock:
            now = self._clock()
            self._sweep_expired_if_due(now)
            self._evict_if_expired(session_id, now=now)
            rows = self._messages.get(session_id)
            if rows is None:
                return []
            self._touch(session_id)
            snapshot = tuple(rows)
        return [_decode_message_row(row) for row in snapshot]

    async def append(self, session_id: str, messages: Sequence[Message]) -> None:
        _validate_identifier(session_id, "session_id")
        additions = _validated_messages(messages)
        if not additions:
            return
        rows = [_encode_message_row(message) for message in additions]
        added_bytes = _message_rows_size(rows)
        async with self._lock:
            now = self._clock()
            self._sweep_expired_if_due(now)
            self._evict_if_expired(session_id, now=now)
            self._ensure_session_fits(session_id, added_bytes)
            self._messages.setdefault(session_id, []).extend(rows)
            self._record_added_bytes(session_id, added_bytes)
            self._touch(session_id)
            self._refresh_expiry(session_id, now=now)
            self._enforce_capacity(protected_session_id=session_id, now=now)

    async def append_once(
        self,
        session_id: str,
        idempotency_key: str,
        messages: Sequence[Message],
    ) -> bool:
        _validate_identifier(session_id, "session_id")
        _validate_identifier(idempotency_key, "idempotency_key")
        additions = _validated_messages(messages)
        if not additions:
            return False
        rows = [_encode_message_row(message) for message in additions]
        digest = _message_rows_digest(rows)
        added_bytes = _message_rows_size(rows)
        async with self._lock:
            now = self._clock()
            self._sweep_expired_if_due(now)
            self._evict_if_expired(session_id, now=now)
            recorded = self._idempotency.get(session_id, {})
            existing = recorded.get(idempotency_key)
            if existing is not None:
                if existing != digest:
                    raise ValueError(
                        "idempotency key was reused with different messages"
                    )
                if session_id in self._messages:
                    self._touch(session_id)
                return False
            self._ensure_session_fits(session_id, added_bytes)
            self._messages.setdefault(session_id, []).extend(rows)
            recorded = self._idempotency.setdefault(session_id, {})
            recorded[idempotency_key] = digest
            self._record_added_bytes(session_id, added_bytes)
            self._touch(session_id)
            self._refresh_expiry(session_id, now=now)
            self._enforce_capacity(protected_session_id=session_id, now=now)
            return True

    async def clear(self, session_id: str) -> None:
        _validate_identifier(session_id, "session_id")
        async with self._lock:
            self._delete_session(session_id)

    async def sweep_expired(self) -> int:
        """Remove every expired session and return the number removed."""

        async with self._lock:
            now = self._clock()
            removed = self._sweep_expired(now)
            self._next_sweep_at = now + self._ttl_sweep_interval_seconds
            return removed

    async def stats(self) -> Mapping[str, int]:
        """Return a content-free snapshot of current in-memory usage."""

        async with self._lock:
            now = self._clock()
            self._sweep_expired_if_due(now)
            return {
                "sessions": len(self._messages),
                "total_bytes": self._total_bytes,
            }

    def _evict_if_expired(self, session_id: str, *, now: float) -> None:
        expires_at = self._expires_at.get(session_id)
        if expires_at is not None and expires_at <= now:
            self._delete_session(session_id)

    def _refresh_expiry(self, session_id: str, *, now: float) -> None:
        if self._ttl_seconds is not None:
            self._expires_at[session_id] = now + self._ttl_seconds

    def _sweep_expired_if_due(self, now: float) -> None:
        if self._ttl_seconds is None or now < self._next_sweep_at:
            return
        self._sweep_expired(now)
        self._next_sweep_at = now + self._ttl_sweep_interval_seconds

    def _sweep_expired(self, now: float) -> int:
        expired = [
            session_id
            for session_id, expires_at in self._expires_at.items()
            if expires_at <= now
        ]
        for session_id in expired:
            self._delete_session(session_id)
        return len(expired)

    def _touch(self, session_id: str) -> None:
        self._access_sequence += 1
        self._last_access[session_id] = self._access_sequence

    def _record_added_bytes(self, session_id: str, added_bytes: int) -> None:
        self._session_bytes[session_id] = (
            self._session_bytes.get(session_id, 0) + added_bytes
        )
        self._total_bytes += added_bytes

    def _ensure_session_fits(self, session_id: str, added_bytes: int) -> None:
        if self._max_total_bytes is None:
            return
        session_bytes = self._session_bytes.get(session_id, 0) + added_bytes
        if session_bytes > self._max_total_bytes:
            raise ConversationStoreCapacityError(
                "conversation session exceeds max_total_bytes"
            )

    def _enforce_capacity(self, *, protected_session_id: str, now: float) -> None:
        if self._capacity_exceeded():
            # Interval gating controls routine cleanup cost, not eviction
            # correctness. Never discard a live LRU session while an expired
            # session can satisfy the same capacity pressure.
            self._sweep_expired(now)
        while self._capacity_exceeded():
            candidates = (
                session_id
                for session_id in self._messages
                if session_id != protected_session_id
            )
            victim = min(
                candidates,
                key=lambda session_id: self._last_access.get(session_id, 0),
                default=None,
            )
            if victim is None:
                raise ConversationStoreCapacityError(
                    "conversation store cannot satisfy configured capacity"
                )
            self._delete_session(victim)

    def _capacity_exceeded(self) -> bool:
        return bool(
            (
                self._max_sessions is not None
                and len(self._messages) > self._max_sessions
            )
            or (
                self._max_total_bytes is not None
                and self._total_bytes > self._max_total_bytes
            )
        )

    def _delete_session(self, session_id: str) -> None:
        self._messages.pop(session_id, None)
        self._idempotency.pop(session_id, None)
        self._expires_at.pop(session_id, None)
        self._last_access.pop(session_id, None)
        removed_bytes = self._session_bytes.pop(session_id, 0)
        self._total_bytes = max(0, self._total_bytes - removed_bytes)


class RedisConversationStore:
    """Redis-backed store using only an injected, Redis-like client.

    Redis list commands are preferred because each append is atomic. A client that
    only exposes ``get``/``set`` is also supported for small adapters and fakes.
    """

    def __init__(
        self,
        client: Any,
        *,
        key_prefix: str = "moduagent:conversation:",
        ttl_seconds: int | None = None,
        use_lists: bool | None = None,
    ) -> None:
        _validate_ttl(ttl_seconds)
        self._client = client
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds
        if not callable(getattr(client, "delete", None)):
            raise TypeError("Redis client must provide delete()")
        list_capable = callable(getattr(client, "lrange", None)) and callable(
            getattr(client, "rpush", None)
        )
        self._use_lists = list_capable if use_lists is None else use_lists
        if self._use_lists and not list_capable:
            raise TypeError("Redis list mode requires lrange and rpush")
        if not self._use_lists and not (
            callable(getattr(client, "get", None))
            and callable(getattr(client, "set", None))
        ):
            raise TypeError("Redis client must provide get/set or lrange/rpush")
        self._fallback_locks: dict[str, asyncio.Lock] = {}
        self._fallback_lock_users: dict[str, int] = {}
        self.supports_idempotent_append = bool(
            self._use_lists and callable(getattr(client, "eval", None))
        )

    async def load(self, session_id: str) -> list[Message]:
        key = self._key(session_id)
        if self._use_lists:
            rows = await _call(self._client.lrange, key, 0, -1)
            return [_decode_message_row(row) for row in (rows or ())]
        return deserialize_messages(await _call(self._client.get, key))

    async def append(self, session_id: str, messages: Sequence[Message]) -> None:
        key = self._key(session_id)
        additions = list(messages)
        if not all(isinstance(message, Message) for message in additions):
            raise TypeError("messages must contain Message instances")
        if not additions:
            return

        if self._use_lists:
            rows = [_encode_message_row(message) for message in additions]
            await _call(self._client.rpush, key, *rows)
            await self._expire(key)
            return

        # The fallback is serialized per store instance. Production Redis clients
        # should expose list operations to preserve cross-process append atomicity.
        lock = self._fallback_locks.setdefault(key, asyncio.Lock())
        self._fallback_lock_users[key] = self._fallback_lock_users.get(key, 0) + 1
        try:
            async with lock:
                current = deserialize_messages(await _call(self._client.get, key))
                current.extend(additions)
                await _redis_set(
                    self._client,
                    key,
                    serialize_messages(current),
                    self._ttl_seconds,
                )
        finally:
            users = self._fallback_lock_users.get(key, 1) - 1
            if users <= 0:
                self._fallback_lock_users.pop(key, None)
                if self._fallback_locks.get(key) is lock:
                    self._fallback_locks.pop(key, None)
            else:
                self._fallback_lock_users[key] = users

    async def append_once(
        self,
        session_id: str,
        idempotency_key: str,
        messages: Sequence[Message],
    ) -> bool:
        key = self._key(session_id)
        _validate_identifier(idempotency_key, "idempotency_key")
        additions = _validated_messages(messages)
        if not additions:
            return False
        if not self.supports_idempotent_append:
            raise RuntimeError(
                "Redis append_once requires list mode and an eval-capable client"
            )
        rows = [_encode_message_row(message) for message in additions]
        digest = _message_batch_digest(additions)
        ttl = 0 if self._ttl_seconds is None else self._ttl_seconds
        result = await _call(
            self._client.eval,
            _REDIS_APPEND_ONCE_SCRIPT,
            2,
            key,
            f"{key}:idempotency",
            idempotency_key,
            digest,
            ttl,
            *rows,
        )
        numeric = int(result)
        if numeric < 0:
            raise ValueError("idempotency key was reused with different messages")
        return numeric == 1

    async def clear(self, session_id: str) -> None:
        key = self._key(session_id)
        await _call(self._client.delete, key)
        await _call(self._client.delete, f"{key}:idempotency")
        self._fallback_locks.pop(key, None)
        self._fallback_lock_users.pop(key, None)

    def _key(self, session_id: str) -> str:
        _validate_identifier(session_id, "session_id")
        return f"{self._key_prefix}{session_id}"

    async def _expire(self, key: str) -> None:
        if self._ttl_seconds is not None:
            expire = getattr(self._client, "expire", None)
            if not callable(expire):
                raise TypeError("Redis client must provide expire when TTL is set")
            await _call(expire, key, self._ttl_seconds)


@runtime_checkable
class ConversationRepository(Protocol):
    """Minimal DB adapter expected by :class:`DatabaseConversationStore`."""

    async def load_messages(
        self, session_id: str
    ) -> Sequence[str | Mapping[str, Any]]: ...

    async def append_messages(
        self, session_id: str, messages: Sequence[str]
    ) -> None: ...

    async def clear_messages(self, session_id: str) -> None: ...


class DatabaseConversationStore:
    """Long-term conversation store backed by an injected repository.

    The repository receives one JSON document per message, keeping this package
    independent of a database driver and table layout.
    """

    def __init__(self, repository: ConversationRepository) -> None:
        self._repository = repository
        for method in ("load_messages", "append_messages", "clear_messages"):
            if not callable(getattr(repository, method, None)):
                raise TypeError(f"repository must provide {method}()")
        self.supports_idempotent_append = callable(
            getattr(repository, "append_messages_once", None)
        )

    async def load(self, session_id: str) -> list[Message]:
        _validate_identifier(session_id, "session_id")
        rows = await _call(self._repository.load_messages, session_id)
        return [_decode_message_row(row) for row in (rows or ())]

    async def append(self, session_id: str, messages: Sequence[Message]) -> None:
        _validate_identifier(session_id, "session_id")
        additions = list(messages)
        if not all(isinstance(message, Message) for message in additions):
            raise TypeError("messages must contain Message instances")
        if additions:
            await _call(
                self._repository.append_messages,
                session_id,
                [_encode_message_row(message) for message in additions],
            )

    async def append_once(
        self,
        session_id: str,
        idempotency_key: str,
        messages: Sequence[Message],
    ) -> bool:
        _validate_identifier(session_id, "session_id")
        _validate_identifier(idempotency_key, "idempotency_key")
        additions = _validated_messages(messages)
        method = getattr(self._repository, "append_messages_once", None)
        if not callable(method):
            raise RuntimeError("repository must provide atomic append_messages_once()")
        result = await _call(
            method,
            session_id,
            idempotency_key,
            [_encode_message_row(message) for message in additions],
            _message_batch_digest(additions),
        )
        if not isinstance(result, bool):
            raise TypeError("append_messages_once() must return a bool")
        return result

    async def clear(self, session_id: str) -> None:
        _validate_identifier(session_id, "session_id")
        await _call(self._repository.clear_messages, session_id)


def _encode_message_row(message: Message) -> str:
    return json.dumps(message.to_dict(), ensure_ascii=False, separators=(",", ":"))


def _decode_message_row(row: str | bytes | bytearray | Mapping[str, Any]) -> Message:
    if isinstance(row, Mapping):
        value = row
    else:
        if isinstance(row, (bytes, bytearray)):
            row = row.decode("utf-8")
        value = json.loads(row)
    if not isinstance(value, Mapping):
        raise ValueError("stored message must be a JSON object")
    return Message.from_dict(value)


def _validated_messages(messages: Sequence[Message]) -> list[Message]:
    additions = list(messages)
    if not all(isinstance(message, Message) for message in additions):
        raise TypeError("messages must contain Message instances")
    return additions


def _message_batch_digest(messages: Sequence[Message]) -> str:
    payload = serialize_messages(messages).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _message_rows_size(rows: Sequence[str]) -> int:
    return sum(len(row.encode("utf-8")) for row in rows)


def _message_rows_digest(rows: Sequence[str]) -> str:
    payload = (
        '{"version":'
        + str(_SERIALIZATION_VERSION)
        + ',"messages":['
        + ",".join(rows)
        + "]}"
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


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
    return await call_maybe_async(function, *args, **kwargs)


def _validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")


def _validate_ttl(value: float | None) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError("ttl_seconds must be positive")


def _validate_optional_positive_int(value: int | None, name: str) -> None:
    if value is not None and (type(value) is not int or value < 1):
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "ConversationStoreCapacityError",
    "ConversationRepository",
    "ConversationStore",
    "DatabaseConversationStore",
    "InMemoryConversationStore",
    "IdempotentConversationStore",
    "RedisConversationStore",
    "deserialize_messages",
    "serialize_messages",
]


_REDIS_APPEND_ONCE_SCRIPT = """
local existing = redis.call('HGET', KEYS[2], ARGV[1])
if existing then
  if existing == ARGV[2] then
    return 0
  end
  return -1
end
for index = 4, #ARGV do
  redis.call('RPUSH', KEYS[1], ARGV[index])
end
redis.call('HSET', KEYS[2], ARGV[1], ARGV[2])
local ttl = tonumber(ARGV[3])
if ttl and ttl > 0 then
  redis.call('EXPIRE', KEYS[1], ttl)
  redis.call('EXPIRE', KEYS[2], ttl)
end
return 1
"""
