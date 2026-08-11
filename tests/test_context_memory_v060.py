from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

import pytest

import moduagent
import moduagent.memory as memory_api
from moduagent.memory.context import (
    CONTEXT_SUMMARY_SCHEMA_VERSION,
    MAX_CONTEXT_MEMORY_SNAPSHOT_BYTES,
    MAX_CONVERSATION_SUMMARY_FIELD_ITEMS,
    MAX_CONVERSATION_SUMMARY_ITEM_BYTES,
    MAX_CONVERSATION_SUMMARY_SERIALIZED_BYTES,
    MAX_CONVERSATION_SUMMARY_TEXT_BYTES,
    ContextAssembler,
    ContextBudgetExceededError,
    ContextItem,
    ContextMemoryCursorRegressionError,
    ContextMemorySerializationError,
    ConversationSummary,
    ConversationSummarySnapshot,
    DatabaseMemoryStateStore,
    InMemoryContextMemoryStateStore,
    MemoryStateKey,
    RedisMemoryStateStore,
    decode_summary_snapshot,
    encode_summary_snapshot,
    migrate_memory_snapshot,
)
from moduagent.memory.state import MemorySnapshot


class _DatabaseRepository:
    def __init__(self) -> None:
        self.rows: dict[str, str] = {}
        self.lock = asyncio.Lock()
        self.cas_calls = 0

    async def load_memory_state(self, storage_key: str) -> str | None:
        return self.rows.get(storage_key)

    async def compare_and_swap_memory_state(
        self,
        storage_key: str,
        expected_version: int,
        next_version: int,
        covered_through_sequence: int,
        payload: str,
    ) -> bool:
        async with self.lock:
            self.cas_calls += 1
            await asyncio.sleep(0)
            current = self.rows.get(storage_key)
            current_version = 0 if current is None else json.loads(current)["version"]
            if current_version != expected_version:
                return False
            if next_version != expected_version + 1:
                raise AssertionError("store passed an invalid next version")
            if current is not None:
                current_cursor = json.loads(current)["covered_through_sequence"]
                if covered_through_sequence < current_cursor:
                    raise AssertionError("store passed a regressing cursor")
            self.rows[storage_key] = payload
            return True

    async def clear_memory_state(self, storage_key: str) -> None:
        self.rows.pop(storage_key, None)


class _RedisClient:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.lock = asyncio.Lock()
        self.expirations: list[tuple[str, int]] = []

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def delete(self, key: str) -> int:
        return int(self.values.pop(key, None) is not None)

    async def eval(
        self,
        script: str,
        numkeys: int,
        key: str,
        *arguments: Any,
    ) -> int:
        del script
        assert numkeys == 1
        expected, next_version, next_cursor, payload, ttl = arguments
        async with self.lock:
            await asyncio.sleep(0)
            current = self.values.get(key)
            if current is None:
                if int(expected) != 0:
                    return 0
            else:
                decoded = json.loads(current)
                if decoded["version"] != int(expected):
                    return 0
                if int(next_cursor) < decoded["covered_through_sequence"]:
                    return -1
            if int(next_version) != int(expected) + 1:
                return -2
            self.values[key] = str(payload)
            if int(ttl) > 0:
                self.expirations.append((key, int(ttl)))
            return 1


def _key(*, tenant_id: str = "tenant-a") -> MemoryStateKey:
    return MemoryStateKey(
        tenant_id=tenant_id,
        agent_id="support-agent:v2",
        session_id="session/한글.42",
        policy_fingerprint="sha256:policy.alpha",
    )


def _snapshot(
    key: MemoryStateKey,
    *,
    version: int,
    cursor: int,
    label: str = "base",
) -> ConversationSummarySnapshot:
    return ConversationSummarySnapshot(
        tenant_id=key.tenant_id,
        agent_id=key.agent_id,
        session_id=key.session_id,
        policy_fingerprint=key.policy_fingerprint,
        covered_through_sequence=cursor,
        covered_prefix_digest=f"sha256:{label}:{cursor}",
        structured_summary=ConversationSummary(
            summary=f"Summary {label}",
            facts=(f"fact:{label}",),
            decisions=(f"decision:{label}",),
            open_items=("follow-up",),
        ),
        source_message_ids=(f"message:{label}:first", f"message:{label}:last"),
        version=version,
    )


def _store(kind: str):
    if kind == "memory":
        return InMemoryContextMemoryStateStore(), None
    if kind == "database":
        repository = _DatabaseRepository()
        return DatabaseMemoryStateStore(repository), repository
    if kind == "redis":
        client = _RedisClient()
        return RedisMemoryStateStore(client, ttl_seconds=90), client
    raise AssertionError(f"unknown test store {kind}")


def test_composite_key_is_reversible_and_tenant_bound() -> None:
    first = _key(tenant_id="tenant:a/한글")
    second = _key(tenant_id="tenant:a.한글")

    assert first.to_storage_key() != second.to_storage_key()
    assert MemoryStateKey.from_storage_key(first.to_storage_key()) == first
    assert MemoryStateKey.from_storage_key(second.to_storage_key()) == second
    with pytest.raises(ValueError, match="storage key"):
        MemoryStateKey.from_storage_key("summary-v2.ambiguous")


def test_context_memory_size_limits_are_exported_from_public_namespaces() -> None:
    expected = {
        "MAX_CONTEXT_MEMORY_SNAPSHOT_BYTES": MAX_CONTEXT_MEMORY_SNAPSHOT_BYTES,
        "MAX_CONVERSATION_SUMMARY_FIELD_ITEMS": (MAX_CONVERSATION_SUMMARY_FIELD_ITEMS),
        "MAX_CONVERSATION_SUMMARY_ITEM_BYTES": (MAX_CONVERSATION_SUMMARY_ITEM_BYTES),
        "MAX_CONVERSATION_SUMMARY_SERIALIZED_BYTES": (
            MAX_CONVERSATION_SUMMARY_SERIALIZED_BYTES
        ),
        "MAX_CONVERSATION_SUMMARY_TEXT_BYTES": (MAX_CONVERSATION_SUMMARY_TEXT_BYTES),
    }

    for name, value in expected.items():
        assert getattr(memory_api, name) == value
        assert getattr(moduagent, name) == value


def test_summary_snapshot_v2_round_trip_is_strict_and_structured() -> None:
    snapshot = _snapshot(_key(), version=1, cursor=12)

    encoded = encode_summary_snapshot(snapshot)
    decoded = decode_summary_snapshot(encoded)

    assert decoded == snapshot
    assert decoded.summary_schema_version == CONTEXT_SUMMARY_SCHEMA_VERSION == 2
    assert decoded.structured_summary.facts == ("fact:base",)
    corrupted = json.loads(encoded)
    corrupted["summary_schema_version"] = 3
    with pytest.raises(ContextMemorySerializationError):
        decode_summary_snapshot(corrupted)

    with pytest.raises(ValueError, match="source_message_ids cannot exceed"):
        replace(
            snapshot,
            source_message_ids=tuple(f"message:{index}" for index in range(257)),
        )


def test_structured_summary_enforces_public_size_limits() -> None:
    with pytest.raises(ValueError, match="summary exceeds"):
        ConversationSummary(summary="x" * (MAX_CONVERSATION_SUMMARY_TEXT_BYTES + 1))

    with pytest.raises(ValueError, match="facts cannot exceed"):
        ConversationSummary(
            summary="bounded",
            facts=tuple(
                f"fact-{index}"
                for index in range(MAX_CONVERSATION_SUMMARY_FIELD_ITEMS + 1)
            ),
        )

    # Korean text proves the item cap is measured in UTF-8 bytes rather than
    # Python characters, while remaining below the 512-character ID guard.
    with pytest.raises(ValueError, match="facts item exceeds"):
        ConversationSummary(
            summary="bounded",
            facts=("가" * (MAX_CONVERSATION_SUMMARY_ITEM_BYTES // 3 + 1),),
        )

    large_field = tuple(
        f"{index:03d}:" + "x" * 508
        for index in range(MAX_CONVERSATION_SUMMARY_FIELD_ITEMS)
    )
    with pytest.raises(ValueError, match="structured conversation summary exceeds"):
        ConversationSummary(
            summary="bounded",
            facts=large_field,
            decisions=large_field,
            preferences=large_field,
            open_items=large_field,
            tool_observations=large_field,
        )

    assert MAX_CONVERSATION_SUMMARY_SERIALIZED_BYTES < len(
        json.dumps(
            {
                "facts": large_field,
                "decisions": large_field,
                "preferences": large_field,
                "open_items": large_field,
                "tool_observations": large_field,
            }
        ).encode("utf-8")
    )


def test_snapshot_decoder_rejects_payload_over_public_byte_limit() -> None:
    with pytest.raises(ContextMemorySerializationError, match="exceeds"):
        decode_summary_snapshot(" " * (MAX_CONTEXT_MEMORY_SNAPSHOT_BYTES + 1))


def test_legacy_memory_snapshot_copy_migrates_without_inventing_message_ids() -> None:
    key = _key()
    legacy = MemorySnapshot(
        summary="Legacy summary",
        covered_message_count=37,
        covered_prefix_digest="legacy-prefix-digest",
        policy_fingerprint=key.policy_fingerprint,
    )

    migrated = migrate_memory_snapshot(legacy, key=key)

    assert migrated.key == key
    assert migrated.summary_schema_version == 2
    assert migrated.covered_through_sequence == 37
    assert migrated.structured_summary.summary == "Legacy summary"
    assert migrated.source_message_ids == ("legacy-prefix:legacy-prefix-digest",)
    assert migrated.version == 1


@pytest.mark.parametrize("kind", ["memory", "database", "redis"])
def test_context_memory_store_cas_conformance(kind: str) -> None:
    async def scenario() -> None:
        store, backend = _store(kind)
        key = _key()
        initial = _snapshot(key, version=1, cursor=10)
        updated = _snapshot(key, version=2, cursor=24, label="updated")

        assert store.durable is (kind != "memory")
        assert await store.load(key) is None
        assert await store.save_if_version(0, initial) is True
        assert await store.save_if_version(0, initial) is False
        assert await store.load(key) == initial
        assert await store.save_if_version(1, updated) is True
        assert await store.save_if_version(1, updated) is False
        assert await store.load(key) == updated

        regressing = _snapshot(key, version=3, cursor=23, label="stale")
        with pytest.raises(ContextMemoryCursorRegressionError):
            await store.save_if_version(2, regressing)
        assert await store.load(key) == updated

        other_key = _key(tenant_id="tenant-b")
        other = _snapshot(other_key, version=1, cursor=1, label="other")
        assert await store.save_if_version(0, other) is True
        assert await store.load(other_key) == other
        assert await store.load(key) == updated

        await store.clear(key)
        assert await store.load(key) is None
        assert await store.load(other_key) == other
        if kind == "redis":
            assert backend.expirations

    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["memory", "database", "redis"])
def test_context_memory_store_allows_only_one_concurrent_cas_winner(
    kind: str,
) -> None:
    async def scenario() -> None:
        store, _ = _store(kind)
        key = _key()
        assert await store.save_if_version(
            0,
            _snapshot(key, version=1, cursor=10),
        )
        writer_a = _snapshot(key, version=2, cursor=20, label="writer-a")
        writer_b = _snapshot(key, version=2, cursor=21, label="writer-b")

        results = await asyncio.gather(
            store.save_if_version(1, writer_a),
            store.save_if_version(1, writer_b),
        )

        assert sorted(results) == [False, True]
        loaded = await store.load(key)
        assert loaded in {writer_a, writer_b}
        assert loaded is not None
        assert loaded.version == 2
        assert loaded.covered_through_sequence in {20, 21}

    asyncio.run(scenario())


def _item(
    item_id: str,
    *,
    priority: int,
    required: bool = False,
    atomic_group: str | None = None,
    compressible: bool = False,
    min_tokens: int = 1,
    max_tokens: int = 1,
) -> ContextItem:
    return ContextItem(
        item_id=item_id,
        source="test",
        payload={"id": item_id},
        priority=priority,
        required=required,
        atomic_group=atomic_group,
        compressible=compressible,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        authority="runtime",
        provenance_refs=(f"source:{item_id}",),
    )


def test_context_assembler_preserves_required_and_atomic_groups() -> None:
    items = (
        _item("system", priority=100, required=True, min_tokens=4, max_tokens=4),
        _item("call", priority=90, atomic_group="tool-1", min_tokens=3, max_tokens=3),
        _item("result", priority=90, atomic_group="tool-1", min_tokens=3, max_tokens=3),
        _item("recent", priority=10, min_tokens=5, max_tokens=5),
    )

    result = ContextAssembler().assemble(items, token_budget=10)

    assert [entry.item.item_id for entry in result.items] == [
        "system",
        "call",
        "result",
    ]
    assert result.dropped_item_ids == ("recent",)
    assert result.used_tokens == 10
    assert result.remaining_tokens == 0


def test_context_assembler_allocates_by_priority_after_required_minimum() -> None:
    required = _item(
        "required",
        priority=50,
        required=True,
        compressible=True,
        min_tokens=3,
        max_tokens=8,
    )
    higher_priority_optional = _item(
        "optional",
        priority=100,
        compressible=True,
        min_tokens=2,
        max_tokens=4,
    )

    result = ContextAssembler().assemble(
        (required, higher_priority_optional),
        token_budget=9,
    )

    allocations = {entry.item.item_id: entry.allocated_tokens for entry in result.items}
    assert allocations == {"required": 5, "optional": 4}
    assert result.used_tokens == 9


def test_context_assembler_fails_when_a_required_atomic_group_cannot_fit() -> None:
    items = (
        _item(
            "commitment",
            priority=100,
            required=True,
            atomic_group="task-state",
            min_tokens=4,
            max_tokens=4,
        ),
        _item(
            "task",
            priority=100,
            atomic_group="task-state",
            min_tokens=3,
            max_tokens=3,
        ),
    )

    with pytest.raises(ContextBudgetExceededError) as captured:
        ContextAssembler().assemble(items, token_budget=6)

    assert captured.value.required_tokens == 7
    assert captured.value.available_tokens == 6
