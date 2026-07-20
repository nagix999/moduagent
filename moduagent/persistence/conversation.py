from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from moduagent.messages import Message


_SERIALIZATION_VERSION = 1


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


class InMemoryConversationStore:
    """Conversation storage for tests and single-process development."""

    def __init__(
        self,
        *,
        ttl_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _validate_ttl(ttl_seconds)
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._messages: dict[str, list[Message]] = {}
        self._expires_at: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def load(self, session_id: str) -> list[Message]:
        _validate_identifier(session_id, "session_id")
        async with self._lock:
            self._evict_if_expired(session_id)
            return list(self._messages.get(session_id, ()))

    async def append(self, session_id: str, messages: Sequence[Message]) -> None:
        _validate_identifier(session_id, "session_id")
        additions = list(messages)
        if not all(isinstance(message, Message) for message in additions):
            raise TypeError("messages must contain Message instances")
        if not additions:
            return
        async with self._lock:
            self._evict_if_expired(session_id)
            self._messages.setdefault(session_id, []).extend(additions)
            self._refresh_expiry(session_id)

    async def clear(self, session_id: str) -> None:
        _validate_identifier(session_id, "session_id")
        async with self._lock:
            self._messages.pop(session_id, None)
            self._expires_at.pop(session_id, None)

    def _evict_if_expired(self, session_id: str) -> None:
        expires_at = self._expires_at.get(session_id)
        if expires_at is not None and expires_at <= self._clock():
            self._messages.pop(session_id, None)
            self._expires_at.pop(session_id, None)

    def _refresh_expiry(self, session_id: str) -> None:
        if self._ttl_seconds is not None:
            self._expires_at[session_id] = self._clock() + self._ttl_seconds


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
        async with lock:
            current = deserialize_messages(await _call(self._client.get, key))
            current.extend(additions)
            await _redis_set(
                self._client,
                key,
                serialize_messages(current),
                self._ttl_seconds,
            )

    async def clear(self, session_id: str) -> None:
        key = self._key(session_id)
        await _call(self._client.delete, key)
        self._fallback_locks.pop(key, None)

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


__all__ = [
    "ConversationRepository",
    "ConversationStore",
    "DatabaseConversationStore",
    "InMemoryConversationStore",
    "RedisConversationStore",
    "deserialize_messages",
    "serialize_messages",
]
