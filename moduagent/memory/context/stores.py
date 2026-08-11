from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from moduagent.persistence._sync import call_maybe_async

from .errors import (
    ContextMemoryCursorRegressionError,
    ContextMemoryIntegrityError,
    ContextMemorySerializationError,
)
from .models import (
    ConversationSummarySnapshot,
    MemoryStateKey,
    decode_summary_snapshot,
    encode_summary_snapshot,
)


@runtime_checkable
class ContextMemoryStateStore(Protocol):
    """CAS store for tenant-bound v2 conversation summary snapshots."""

    durable: bool

    async def load(
        self,
        key: MemoryStateKey,
    ) -> ConversationSummarySnapshot | None: ...

    async def save_if_version(
        self,
        expected_version: int,
        next_snapshot: ConversationSummarySnapshot,
    ) -> bool:
        """Create at version 1 when expected is 0, otherwise compare-and-swap."""

        ...

    async def clear(self, key: MemoryStateKey) -> None: ...


class InMemoryContextMemoryStateStore:
    """Reference CAS implementation for conformance tests and local development."""

    durable = False

    def __init__(self) -> None:
        self._payloads: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def load(
        self,
        key: MemoryStateKey,
    ) -> ConversationSummarySnapshot | None:
        storage_key = _storage_key(key)
        async with self._lock:
            payload = self._payloads.get(storage_key)
        return None if payload is None else decode_summary_snapshot(payload)

    async def save_if_version(
        self,
        expected_version: int,
        next_snapshot: ConversationSummarySnapshot,
    ) -> bool:
        _validate_transition_shape(expected_version, next_snapshot)
        storage_key = next_snapshot.key.to_storage_key()
        payload = encode_summary_snapshot(next_snapshot)
        async with self._lock:
            current_payload = self._payloads.get(storage_key)
            current = (
                None
                if current_payload is None
                else decode_summary_snapshot(current_payload)
            )
            if not _expected_matches(current, expected_version):
                return False
            _validate_monotonic_cursor(current, next_snapshot)
            self._payloads[storage_key] = payload
            return True

    async def clear(self, key: MemoryStateKey) -> None:
        storage_key = _storage_key(key)
        async with self._lock:
            self._payloads.pop(storage_key, None)


class RedisMemoryStateStore:
    """Redis-backed v2 summary store using one Lua CAS operation per write.

    The injected client may be synchronous or asynchronous but must provide
    ``get``, ``delete`` and ``eval``. Redis is appropriate for durable-enough
    session summary state; it is not a canonical long-term Memory store.
    """

    durable = True

    def __init__(
        self,
        client: Any,
        *,
        key_prefix: str = "moduagent:context-memory:",
        ttl_seconds: int | None = None,
    ) -> None:
        if not isinstance(key_prefix, str) or not key_prefix:
            raise ValueError("key_prefix cannot be empty")
        if ttl_seconds is not None and (
            type(ttl_seconds) is not int or ttl_seconds < 1
        ):
            raise ValueError("ttl_seconds must be a positive integer")
        for method in ("get", "delete", "eval"):
            if not callable(getattr(client, method, None)):
                raise TypeError(f"Redis client must provide {method}()")
        self._client = client
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds

    async def load(
        self,
        key: MemoryStateKey,
    ) -> ConversationSummarySnapshot | None:
        storage_key = self._key(key)
        payload = await _call(self._client.get, storage_key)
        if payload is None:
            return None
        snapshot = decode_summary_snapshot(payload)
        _require_loaded_key(snapshot, key)
        return snapshot

    async def save_if_version(
        self,
        expected_version: int,
        next_snapshot: ConversationSummarySnapshot,
    ) -> bool:
        _validate_transition_shape(expected_version, next_snapshot)
        result = await _call(
            self._client.eval,
            _REDIS_MEMORY_CAS_SCRIPT,
            1,
            self._key(next_snapshot.key),
            expected_version,
            next_snapshot.version,
            next_snapshot.covered_through_sequence,
            encode_summary_snapshot(next_snapshot),
            0 if self._ttl_seconds is None else self._ttl_seconds,
        )
        try:
            code = int(result)
        except (TypeError, ValueError) as exc:
            raise ContextMemoryIntegrityError(
                "Redis Context Memory CAS returned an invalid result"
            ) from exc
        if code == 1:
            return True
        if code == 0:
            return False
        if code == -1:
            raise ContextMemoryCursorRegressionError(
                "covered_through_sequence cannot move backwards"
            )
        if code == -2:
            raise ContextMemorySerializationError(
                "stored Redis Context Memory state is invalid"
            )
        raise ContextMemoryIntegrityError(
            f"Redis Context Memory CAS returned unknown code {code}"
        )

    async def clear(self, key: MemoryStateKey) -> None:
        await _call(self._client.delete, self._key(key))

    def _key(self, key: MemoryStateKey) -> str:
        return f"{self._key_prefix}{_storage_key(key)}"


@runtime_checkable
class MemoryStateRepository(Protocol):
    """Transactional repository required by ``DatabaseMemoryStateStore``."""

    async def load_memory_state(
        self,
        storage_key: str,
    ) -> str | bytes | bytearray | Mapping[str, Any] | None: ...

    async def compare_and_swap_memory_state(
        self,
        storage_key: str,
        expected_version: int,
        next_version: int,
        covered_through_sequence: int,
        payload: str,
    ) -> bool:
        """Atomically write only when the stored version equals expected."""

        ...

    async def clear_memory_state(self, storage_key: str) -> None: ...


class DatabaseMemoryStateStore:
    """Database-backed v2 state over an injected transactional repository."""

    durable = True

    def __init__(
        self,
        repository: MemoryStateRepository,
        *,
        key_prefix: str = "moduagent:context-memory:",
    ) -> None:
        if not isinstance(key_prefix, str) or not key_prefix:
            raise ValueError("key_prefix cannot be empty")
        for method in (
            "load_memory_state",
            "compare_and_swap_memory_state",
            "clear_memory_state",
        ):
            if not callable(getattr(repository, method, None)):
                raise TypeError(f"repository must provide {method}()")
        self._repository = repository
        self._key_prefix = key_prefix

    async def load(
        self,
        key: MemoryStateKey,
    ) -> ConversationSummarySnapshot | None:
        payload = await _call(
            self._repository.load_memory_state,
            self._key(key),
        )
        if payload is None:
            return None
        snapshot = decode_summary_snapshot(payload)
        _require_loaded_key(snapshot, key)
        return snapshot

    async def save_if_version(
        self,
        expected_version: int,
        next_snapshot: ConversationSummarySnapshot,
    ) -> bool:
        _validate_transition_shape(expected_version, next_snapshot)
        current = await self.load(next_snapshot.key)
        if not _expected_matches(current, expected_version):
            return False
        _validate_monotonic_cursor(current, next_snapshot)
        result = await _call(
            self._repository.compare_and_swap_memory_state,
            self._key(next_snapshot.key),
            expected_version,
            next_snapshot.version,
            next_snapshot.covered_through_sequence,
            encode_summary_snapshot(next_snapshot),
        )
        if not isinstance(result, bool):
            raise TypeError("compare_and_swap_memory_state() must return a bool")
        return result

    async def clear(self, key: MemoryStateKey) -> None:
        await _call(self._repository.clear_memory_state, self._key(key))

    def _key(self, key: MemoryStateKey) -> str:
        return f"{self._key_prefix}{_storage_key(key)}"


def _validate_transition_shape(
    expected_version: int,
    next_snapshot: ConversationSummarySnapshot,
) -> None:
    if type(expected_version) is not int or expected_version < 0:
        raise ValueError("expected_version must be a non-negative integer")
    if not isinstance(next_snapshot, ConversationSummarySnapshot):
        raise TypeError("next_snapshot must be a ConversationSummarySnapshot")
    if next_snapshot.version != expected_version + 1:
        raise ValueError("next snapshot version must equal expected_version + 1")


def _expected_matches(
    current: ConversationSummarySnapshot | None,
    expected_version: int,
) -> bool:
    if current is None:
        return expected_version == 0
    return current.version == expected_version


def _validate_monotonic_cursor(
    current: ConversationSummarySnapshot | None,
    next_snapshot: ConversationSummarySnapshot,
) -> None:
    if (
        current is not None
        and next_snapshot.covered_through_sequence < current.covered_through_sequence
    ):
        raise ContextMemoryCursorRegressionError(
            "covered_through_sequence cannot move backwards"
        )


def _require_loaded_key(
    snapshot: ConversationSummarySnapshot,
    expected: MemoryStateKey,
) -> None:
    if snapshot.key != expected:
        raise ContextMemoryIntegrityError(
            "stored Context Memory snapshot does not match its composite key"
        )


def _storage_key(key: MemoryStateKey) -> str:
    if not isinstance(key, MemoryStateKey):
        raise TypeError("key must be a MemoryStateKey")
    return key.to_storage_key()


async def _call(function: Any, *args: Any) -> Any:
    return await call_maybe_async(function, *args)


__all__ = [
    "ContextMemoryStateStore",
    "DatabaseMemoryStateStore",
    "InMemoryContextMemoryStateStore",
    "MemoryStateRepository",
    "RedisMemoryStateStore",
]


_REDIS_MEMORY_CAS_SCRIPT = """
local current = redis.call('GET', KEYS[1])
local expected_version = tonumber(ARGV[1])
local next_version = tonumber(ARGV[2])
local next_cursor = tonumber(ARGV[3])

if not current then
  if expected_version ~= 0 then
    return 0
  end
else
  local ok, decoded = pcall(cjson.decode, current)
  if not ok or type(decoded) ~= 'table' then
    return -2
  end
  local current_version = tonumber(decoded['version'])
  local current_cursor = tonumber(decoded['covered_through_sequence'])
  if not current_version or not current_cursor then
    return -2
  end
  if current_version ~= expected_version then
    return 0
  end
  if next_cursor < current_cursor then
    return -1
  end
end

if next_version ~= expected_version + 1 then
  return -2
end
redis.call('SET', KEYS[1], ARGV[4])
local ttl = tonumber(ARGV[5])
if ttl and ttl > 0 then
  redis.call('EXPIRE', KEYS[1], ttl)
end
return 1
"""
