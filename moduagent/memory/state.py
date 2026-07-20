from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    summary: str
    covered_message_count: int
    covered_prefix_digest: str
    policy_fingerprint: str

    def __post_init__(self) -> None:
        if not self.summary.strip():
            raise ValueError("summary cannot be empty")
        if self.covered_message_count < 1:
            raise ValueError("covered_message_count must be at least 1")
        if not self.covered_prefix_digest:
            raise ValueError("covered_prefix_digest cannot be empty")
        if not self.policy_fingerprint:
            raise ValueError("policy_fingerprint cannot be empty")


@runtime_checkable
class MemoryStateStore(Protocol):
    async def load(self, session_id: str) -> MemorySnapshot | None: ...

    async def save(self, session_id: str, snapshot: MemorySnapshot) -> None: ...

    async def clear(self, session_id: str) -> None: ...


class InMemoryMemoryStateStore:
    """Process-local summary cache for tests and single-process deployments."""

    def __init__(self) -> None:
        self._snapshots: dict[str, MemorySnapshot] = {}
        self._lock = asyncio.Lock()

    async def load(self, session_id: str) -> MemorySnapshot | None:
        async with self._lock:
            return self._snapshots.get(session_id)

    async def save(self, session_id: str, snapshot: MemorySnapshot) -> None:
        async with self._lock:
            self._snapshots[session_id] = snapshot

    async def clear(self, session_id: str) -> None:
        async with self._lock:
            self._snapshots.pop(session_id, None)


__all__ = [
    "InMemoryMemoryStateStore",
    "MemorySnapshot",
    "MemoryStateStore",
]
