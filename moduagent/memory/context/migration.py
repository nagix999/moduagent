from __future__ import annotations

from collections.abc import Sequence

from moduagent.memory.state import MemorySnapshot, MemoryStateStore

from .models import (
    ConversationSummary,
    ConversationSummarySnapshot,
    MemoryStateKey,
)


class ScopedLegacyMemoryStateStore:
    """Explicit tenant/Agent binding for a session-only 0.5 state store.

    ``MemorySnapshot`` has no tenant or Agent identity. Automatic migration may
    therefore read it only through this adapter, which attests that the legacy
    namespace is dedicated to exactly one tenant/Agent pair. Applications must
    not wrap one shared unpartitioned legacy namespace with multiple scopes.
    """

    __slots__ = ("_agent_id", "_durable", "_store", "_tenant_id")

    def __init__(
        self,
        store: MemoryStateStore,
        *,
        tenant_id: str,
        agent_id: str,
    ) -> None:
        if isinstance(store, ScopedLegacyMemoryStateStore):
            raise ValueError("a scoped legacy state store cannot be wrapped again")
        if not isinstance(store, MemoryStateStore):
            raise TypeError("store must implement MemoryStateStore")
        MemoryStateKey(
            tenant_id=tenant_id,
            agent_id=agent_id,
            session_id="scope-validation",
            policy_fingerprint="scope-validation",
        )
        self._store = store
        self._tenant_id = tenant_id
        self._agent_id = agent_id
        self._durable = bool(getattr(store, "durable", False))

    @property
    def store(self) -> MemoryStateStore:
        return self._store

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def durable(self) -> bool:
        return self._durable

    async def load(self, session_id: str) -> MemorySnapshot | None:
        return await self.store.load(session_id)

    async def save(self, session_id: str, snapshot: MemorySnapshot) -> None:
        await self.store.save(session_id, snapshot)

    async def clear(self, session_id: str) -> None:
        await self.store.clear(session_id)


def migrate_memory_snapshot(
    legacy: MemorySnapshot,
    *,
    key: MemoryStateKey,
    covered_through_sequence: int | None = None,
    source_message_ids: Sequence[str] | None = None,
    version: int = 1,
) -> ConversationSummarySnapshot:
    """Copy-migrate a 0.5 ``MemorySnapshot`` into the v2 durable schema.

    Legacy snapshots did not persist message IDs or a store-issued sequence.
    Callers with an authoritative mapping should supply both. Otherwise the
    covered message count is used as the cursor and one explicit legacy-prefix
    provenance marker is retained; no synthetic per-message identity is invented.
    """

    if not isinstance(legacy, MemorySnapshot):
        raise TypeError("legacy must be a MemorySnapshot")
    if not isinstance(key, MemoryStateKey):
        raise TypeError("key must be a MemoryStateKey")
    if legacy.policy_fingerprint != key.policy_fingerprint:
        raise ValueError("legacy policy fingerprint does not match the v2 key")
    cursor = (
        legacy.covered_message_count
        if covered_through_sequence is None
        else covered_through_sequence
    )
    identifiers = (
        (f"legacy-prefix:{legacy.covered_prefix_digest}",)
        if source_message_ids is None
        else tuple(source_message_ids)
    )
    return ConversationSummarySnapshot(
        tenant_id=key.tenant_id,
        agent_id=key.agent_id,
        session_id=key.session_id,
        policy_fingerprint=key.policy_fingerprint,
        covered_through_sequence=cursor,
        covered_prefix_digest=legacy.covered_prefix_digest,
        structured_summary=ConversationSummary(summary=legacy.summary),
        source_message_ids=identifiers,
        version=version,
    )


__all__ = ["ScopedLegacyMemoryStateStore", "migrate_memory_snapshot"]
