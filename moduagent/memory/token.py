from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import secrets
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from moduagent.models import ModelRequest

if TYPE_CHECKING:
    from moduagent.models import VLLMClient


@dataclass(frozen=True, slots=True)
class TokenBudget:
    context_window_tokens: int
    reserved_output_tokens: int = 0
    safety_margin_tokens: int = 0

    def __post_init__(self) -> None:
        if self.context_window_tokens < 1:
            raise ValueError("context_window_tokens must be at least 1")
        if self.reserved_output_tokens < 0:
            raise ValueError("reserved_output_tokens cannot be negative")
        if self.safety_margin_tokens < 0:
            raise ValueError("safety_margin_tokens cannot be negative")
        if self.input_tokens < 1:
            raise ValueError(
                "reserved_output_tokens and safety_margin_tokens must leave "
                "at least one input token"
            )

    @property
    def input_tokens(self) -> int:
        return (
            self.context_window_tokens
            - self.reserved_output_tokens
            - self.safety_margin_tokens
        )


@runtime_checkable
class TokenCounter(Protocol):
    async def count_request(self, request: ModelRequest) -> int: ...


class ApproximateTokenCounter:
    """Conservative local estimate for development and fallback behavior.

    The estimate includes the serialized messages, tools, output schema, model
    options, and provider options. Production deployments should prefer the
    target model's tokenizer and retain an appropriate safety margin.
    """

    def __init__(
        self,
        bytes_per_token: float = 2.0,
        per_message_tokens: int = 4,
        base_tokens: int = 2,
    ) -> None:
        if bytes_per_token <= 0:
            raise ValueError("bytes_per_token must be positive")
        if per_message_tokens < 0:
            raise ValueError("per_message_tokens cannot be negative")
        if base_tokens < 0:
            raise ValueError("base_tokens cannot be negative")
        self.bytes_per_token = float(bytes_per_token)
        self.per_message_tokens = per_message_tokens
        self.base_tokens = base_tokens

    async def count_request(self, request: ModelRequest) -> int:
        payload = {
            "messages": [message.to_dict() for message in request.messages],
            "tools": [_json_safe(tool) for tool in request.tools],
            "output_schema": _json_safe(request.output_schema),
            "options": _json_safe(request.options),
            "provider_options": _json_safe(request.provider_options),
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        content_tokens = math.ceil(
            len(serialized.encode("utf-8")) / self.bytes_per_token
        )
        return (
            self.base_tokens
            + len(request.messages) * self.per_message_tokens
            + content_tokens
        )


class VLLMTokenCounter:
    """Use the configured vLLM server's chat template for exact token counts."""

    def __init__(self, client: VLLMClient) -> None:
        if not callable(getattr(client, "count_tokens", None)):
            raise TypeError("client must provide count_tokens()")
        self.client = client

    async def count_request(self, request: ModelRequest) -> int:
        return await self.client.count_tokens(request)


@dataclass(frozen=True, slots=True)
class _TokenCountCacheEntry:
    count: int
    expires_at: float | None


class CachingTokenCounter:
    """Bounded in-process cache for an exact or approximate token counter.

    Concurrent requests with the same content share one delegate call. Only a
    successful count is cached. Cache keys are process-local keyed digests, so
    prompts, Tool arguments, and schemas are not retained by the cache.
    """

    def __init__(
        self,
        delegate: TokenCounter,
        *,
        max_entries: int = 1_024,
        ttl_seconds: float | None = 300.0,
    ) -> None:
        if not callable(getattr(delegate, "count_request", None)):
            raise TypeError("delegate must provide count_request()")
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive when provided")
        self.delegate = delegate
        self.max_entries = max_entries
        self.ttl_seconds = float(ttl_seconds) if ttl_seconds is not None else None
        self._digest_key = secrets.token_bytes(32)
        self._cache: OrderedDict[bytes, _TokenCountCacheEntry] = OrderedDict()
        self._inflight: dict[bytes, asyncio.Task[int]] = {}
        self._lock = asyncio.Lock()

    async def count_request(self, request: ModelRequest) -> int:
        key = _request_digest(request, digest_key=self._digest_key)
        async with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                if entry.expires_at is None or entry.expires_at > time.monotonic():
                    self._cache.move_to_end(key)
                    return entry.count
                self._cache.pop(key, None)

            task = self._inflight.get(key)
            if task is None:
                task = asyncio.create_task(self._count_and_cache(key, request))
                self._inflight[key] = task
                task.add_done_callback(_consume_task_exception)

        # A cancelled waiter must not cancel the shared delegate operation for
        # other callers.
        return await asyncio.shield(task)

    async def _count_and_cache(self, key: bytes, request: ModelRequest) -> int:
        task = asyncio.current_task()
        try:
            count = await self.delegate.count_request(request)
            expires_at = (
                time.monotonic() + self.ttl_seconds
                if self.ttl_seconds is not None
                else None
            )
            async with self._lock:
                self._cache[key] = _TokenCountCacheEntry(count, expires_at)
                self._cache.move_to_end(key)
                while len(self._cache) > self.max_entries:
                    self._cache.popitem(last=False)
            return count
        finally:
            async with self._lock:
                if self._inflight.get(key) is task:
                    self._inflight.pop(key, None)


def _consume_task_exception(task: asyncio.Task[int]) -> None:
    if task.cancelled():
        return
    task.exception()


def _request_digest(
    request: ModelRequest,
    *,
    digest_key: bytes,
) -> bytes:
    payload = {
        "messages": [_json_safe(message.to_dict()) for message in request.messages],
        "tools": [_json_safe(tool) for tool in request.tools],
        "output_schema": _json_safe(request.output_schema),
        "options": _json_safe(request.options),
        "provider_options": _json_safe(request.provider_options),
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hmac.new(digest_key, serialized, hashlib.sha256).digest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _json_safe(model_dump(mode="json"))
    if is_dataclass(value):
        return _json_safe(asdict(value))
    return repr(value)


__all__ = [
    "ApproximateTokenCounter",
    "CachingTokenCounter",
    "TokenBudget",
    "TokenCounter",
    "VLLMTokenCounter",
]
