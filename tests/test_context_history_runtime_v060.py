from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from moduagent import EventType, function_tool
from moduagent.agent import Agent
from moduagent.config import AgentConfig
from moduagent.memory import MemoryPhase, MemoryRequest, SummaryResult, TokenBudget
from moduagent.memory.context import (
    MAX_CONVERSATION_SUMMARY_TEXT_BYTES,
    ContextHistoryPaginationRequiredError,
    ContextHistoryCursorInvalidatedError,
    ContextHistoryTailOverflowError,
    ContextMemoryIntegrityError,
    ContextMemorySerializationError,
    ContextMemoryWriteConflictError,
    ConversationSummary,
    ConversationSummarySnapshot,
    DurableContextHistoryLoader,
    DurableSummarizingConversationMemoryPolicy,
    InMemoryContextMemoryStateStore,
    MemoryStateKey,
    ScopedLegacyMemoryStateStore,
    migrate_memory_snapshot,
)
from moduagent.memory.policies import _messages_digest
from moduagent.memory.state import (
    InMemoryMemoryStateStore,
    MemorySnapshot,
)
from moduagent.messages import Message, MessageRole, ToolCall, Usage
from moduagent.models import ModelRequest, ModelResponse
from moduagent.observability import AuditEventSink
from moduagent.persistence.conversation import InMemoryConversationStore


class _NoFullLoadConversationStore(InMemoryConversationStore):
    supports_tenant_agent_scope = True

    def __init__(
        self,
        *,
        tenant_id: str = "tenant-a",
        agent_id: str = "agent-a",
    ) -> None:
        super().__init__()
        self.tenant_id = tenant_id
        self.agent_id = agent_id
        self.full_load_calls = 0
        self.tail_calls: list[tuple[str, int, int]] = []

    async def load(self, session_id: str) -> list[Message]:
        del session_id
        self.full_load_calls += 1
        raise AssertionError("durable Context Memory must not load full history")

    async def load_tail(
        self,
        session_id: str,
        after_sequence: int = 0,
        limit: int = 100,
    ):
        self.tail_calls.append((session_id, after_sequence, limit))
        return await super().load_tail(session_id, after_sequence, limit)


class _MessageCountTokenCounter:
    async def count_request(self, request: ModelRequest) -> int:
        return len(request.messages)


class _ToggleFailStateStore(InMemoryContextMemoryStateStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_load = False

    async def load(self, key: MemoryStateKey):
        if self.fail_load:
            raise ContextMemorySerializationError("corrupt state")
        return await super().load(key)


class _CountingLegacyStateStore(InMemoryMemoryStateStore):
    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0
        self.clear_calls = 0

    async def load(self, session_id: str):
        self.load_calls += 1
        return await super().load(session_id)

    async def clear(self, session_id: str) -> None:
        self.clear_calls += 1
        await super().clear(session_id)


class _BarrierCreateStateStore(InMemoryContextMemoryStateStore):
    def __init__(self) -> None:
        super().__init__()
        self._arrivals = 0
        self._both_arrived = asyncio.Event()

    async def save_if_version(self, expected_version, next_snapshot):
        if expected_version == 0:
            self._arrivals += 1
            if self._arrivals == 2:
                self._both_arrived.set()
            await self._both_arrived.wait()
        return await super().save_if_version(expected_version, next_snapshot)


class _RecordingSummarizer:
    cache_fingerprint = "recording-summary-v1"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Message, ...], str | None]] = []

    async def summarize(
        self,
        messages: tuple[Message, ...],
        *,
        previous_summary: str | None = None,
    ) -> SummaryResult:
        self.calls.append((messages, previous_summary))
        contents = ",".join(message.content or "" for message in messages)
        prefix = "" if previous_summary is None else f"{previous_summary}|"
        return SummaryResult(f"{prefix}{contents}", Usage(2, 1, 3))


class _OversizedSummarizer:
    cache_fingerprint = "oversized-summary-v1"

    async def summarize(
        self,
        messages: tuple[Message, ...],
        *,
        previous_summary: str | None = None,
    ) -> SummaryResult:
        del messages, previous_summary
        return SummaryResult(
            "x" * (MAX_CONVERSATION_SUMMARY_TEXT_BYTES + 1),
            Usage(2, 1, 3),
        )


class _OneResponseModel:
    def __init__(self, response: str) -> None:
        self.response = response
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(Message.assistant(self.response))


def _snapshot(
    key: MemoryStateKey,
    *,
    cursor: int = 9_990,
    anchor_message_id: str = "message:1",
):
    return ConversationSummarySnapshot(
        tenant_id=key.tenant_id,
        agent_id=key.agent_id,
        session_id=key.session_id,
        policy_fingerprint=key.policy_fingerprint,
        covered_through_sequence=cursor,
        covered_prefix_digest=f"digest:{cursor}",
        structured_summary=ConversationSummary(
            summary="Customer chose the annual plan.",
            decisions=("annual-plan",),
        ),
        source_message_ids=(anchor_message_id,),
        version=1,
    )


def _request(
    history: tuple[Message, ...],
    *,
    session_id: str,
    current: str = "current",
) -> MemoryRequest:
    messages = (Message.system("system"), *history, Message.user(current))
    return MemoryRequest(
        run_id=f"run:{current}",
        session_id=session_id,
        phase=MemoryPhase.ACT,
        model_request=ModelRequest(messages=messages),
        protected_from=len(messages) - 1,
    )


def test_loader_reads_only_after_summary_cursor_and_projects_safe_boundary() -> None:
    async def scenario() -> None:
        session_id = "long-session"
        state_store = InMemoryContextMemoryStateStore()
        key = MemoryStateKey("tenant-a", "agent-a", session_id, "policy-a")
        conversation_store = _NoFullLoadConversationStore()
        await conversation_store.append(
            session_id,
            [Message.user(f"message-{index}") for index in range(1, 10_001)],
        )
        anchor = await conversation_store.load_tail(session_id, 9_989, 1)
        assert await state_store.save_if_version(
            0,
            _snapshot(key, anchor_message_id=anchor.items[0].message_id),
        )
        conversation_store.tail_calls.clear()
        loader = DurableContextHistoryLoader(
            state_store=state_store,
            tenant_id=key.tenant_id,
            agent_id=key.agent_id,
            policy_fingerprint=key.policy_fingerprint,
            page_size=64,
            max_tail_messages=64,
        )

        view = await loader.load_history(conversation_store, session_id)

        assert view.started_after_sequence == 9_990
        assert view.loaded_through_sequence == 10_000
        assert len(view.tail) == 10
        assert conversation_store.tail_calls == [
            (session_id, 9_989, 1),
            (session_id, 9_990, 64),
            (session_id, 9_989, 1),
            (session_id, 9_990, 10),
            (session_id, 9_989, 1),
        ]
        assert conversation_store.full_load_calls == 0
        projected = view.messages
        assert projected[0].role is MessageRole.USER
        assert "untrusted JSON" in (projected[0].content or "")
        assert "annual-plan" in (projected[0].content or "")
        assert projected[1].metadata["_moduagent_context_memory_sequence"] == 9_991
        assert "structured_summary" not in projected[0].metadata

    asyncio.run(scenario())


def test_loader_fails_bounded_when_uncompacted_tail_is_too_large() -> None:
    async def scenario() -> None:
        store = _NoFullLoadConversationStore()
        await store.append("uncompacted", [Message.user(str(i)) for i in range(100)])
        loader = DurableContextHistoryLoader(
            state_store=InMemoryContextMemoryStateStore(),
            tenant_id="tenant-a",
            agent_id="agent-a",
            policy_fingerprint="policy-a",
            page_size=3,
            max_tail_messages=5,
        )

        with pytest.raises(ContextHistoryTailOverflowError) as captured:
            await loader.load_history(store, "uncompacted")

        assert captured.value.max_tail_messages == 5
        assert store.tail_calls == [
            ("uncompacted", 0, 3),
            ("uncompacted", 3, 2),
        ]
        assert store.full_load_calls == 0

    asyncio.run(scenario())


def test_loader_rejects_load_tail_that_is_only_a_full_blob_fallback() -> None:
    class BlobCompatibilityStore:
        supports_bounded_load_tail = False

        async def load(self, session_id: str) -> list[Message]:
            raise AssertionError(session_id)

        async def load_tail(self, *args: Any, **kwargs: Any):
            raise AssertionError((args, kwargs))

        async def append(self, session_id: str, messages: list[Message]) -> None:
            del session_id, messages

        async def clear(self, session_id: str) -> None:
            del session_id

    loader = DurableContextHistoryLoader(
        state_store=InMemoryContextMemoryStateStore(),
        tenant_id="tenant-a",
        agent_id="agent-a",
        policy_fingerprint="policy-a",
    )

    with pytest.raises(ContextHistoryPaginationRequiredError):
        asyncio.run(loader.load_history(BlobCompatibilityStore(), "session"))


def test_loader_requires_authoritative_ids_for_lazy_legacy_migration() -> None:
    async def scenario() -> None:
        session_id = "legacy-unverified"
        conversation_store = _NoFullLoadConversationStore()
        await conversation_store.append(
            session_id,
            [Message.user("u1"), Message.assistant("a1")],
        )
        state_store = InMemoryContextMemoryStateStore()
        key = MemoryStateKey("tenant-a", "agent-a", session_id, "policy-a")
        migrated = migrate_memory_snapshot(
            MemorySnapshot(
                summary="legacy summary",
                covered_message_count=2,
                covered_prefix_digest="legacy-digest",
                policy_fingerprint=key.policy_fingerprint,
            ),
            key=key,
        )
        assert await state_store.save_if_version(0, migrated)
        loader = DurableContextHistoryLoader(
            state_store=state_store,
            tenant_id=key.tenant_id,
            agent_id=key.agent_id,
            policy_fingerprint=key.policy_fingerprint,
        )

        with pytest.raises(ContextHistoryCursorInvalidatedError, match="legacy"):
            await loader.load_history(conversation_store, session_id)

        # Marker-only state is now eligible for safe lazy backfill. Its bogus
        # digest is rejected after one bounded canonical-prefix page; the loader
        # still never falls back to full history.
        assert conversation_store.tail_calls == [(session_id, 0, 2)]
        assert conversation_store.full_load_calls == 0

    asyncio.run(scenario())


def test_loader_automatically_migrates_verified_legacy_snapshot_with_bounded_io() -> (
    None
):
    async def scenario() -> None:
        session_id = "legacy-auto-migration"
        conversation_store = _NoFullLoadConversationStore()
        messages = [
            (
                Message.user(f"u-{index}")
                if index % 2
                else Message.assistant(f"a-{index}")
            )
            for index in range(1, 311)
        ]
        await conversation_store.append(session_id, messages)
        v2_store = InMemoryContextMemoryStateStore()
        legacy_store = _CountingLegacyStateStore()
        scoped_legacy_store = ScopedLegacyMemoryStateStore(
            legacy_store,
            tenant_id="tenant-a",
            agent_id="agent-a",
        )
        policy = DurableSummarizingConversationMemoryPolicy(
            budget=TokenBudget(2_000),
            summarizer=_RecordingSummarizer(),
            state_store=v2_store,
            tenant_id="tenant-a",
            agent_id="agent-a",
            token_counter=_MessageCountTokenCounter(),
            history_page_size=41,
            legacy_state_store=scoped_legacy_store,
        )
        covered = tuple(messages[:300])
        await legacy_store.save(
            session_id,
            MemorySnapshot(
                summary="verified legacy summary",
                covered_message_count=len(covered),
                covered_prefix_digest=_messages_digest(covered),
                policy_fingerprint=policy.policy_fingerprint,
            ),
        )

        view = await policy.load_history(conversation_store, session_id)
        key = MemoryStateKey(
            "tenant-a",
            "agent-a",
            session_id,
            policy.policy_fingerprint,
        )
        migrated = await v2_store.load(key)

        assert migrated is not None
        assert migrated.version == 1
        assert migrated.covered_through_sequence == 300
        assert migrated.covered_prefix_digest != _messages_digest(covered)
        assert len(migrated.source_message_ids) == 256
        assert not any(
            value.startswith("legacy-prefix:") for value in migrated.source_message_ids
        )
        assert view.snapshot == migrated
        assert [item.message.content for item in view.tail] == [
            message.content for message in messages[300:]
        ]
        assert legacy_store.load_calls == 1
        assert conversation_store.full_load_calls == 0
        # The one-time authoritative scan is paginated. Source-anchor checks
        # may use a larger bounded window after the state has been backfilled.
        migration_calls = conversation_store.tail_calls[:16]
        assert migration_calls
        assert all(limit <= 41 for _, _, limit in migration_calls)

        conversation_store.tail_calls.clear()
        second = await policy.load_history(conversation_store, session_id)
        assert second.snapshot == migrated
        assert legacy_store.load_calls == 1
        assert conversation_store.full_load_calls == 0

    asyncio.run(scenario())


def test_loader_legacy_migration_digest_mismatch_fails_closed_without_v2_write() -> (
    None
):
    async def scenario() -> None:
        session_id = "legacy-digest-mismatch"
        conversation_store = _NoFullLoadConversationStore()
        await conversation_store.append(
            session_id,
            [Message.user("u1"), Message.assistant("a1")],
        )
        v2_store = InMemoryContextMemoryStateStore()
        legacy_store = _CountingLegacyStateStore()
        scoped_legacy_store = ScopedLegacyMemoryStateStore(
            legacy_store,
            tenant_id="tenant-a",
            agent_id="agent-a",
        )
        policy = DurableSummarizingConversationMemoryPolicy(
            budget=TokenBudget(100),
            summarizer=_RecordingSummarizer(),
            state_store=v2_store,
            tenant_id="tenant-a",
            agent_id="agent-a",
            token_counter=_MessageCountTokenCounter(),
            legacy_state_store=scoped_legacy_store,
        )
        await legacy_store.save(
            session_id,
            MemorySnapshot(
                summary="must not be trusted",
                covered_message_count=2,
                covered_prefix_digest="incorrect-digest",
                policy_fingerprint=policy.policy_fingerprint,
            ),
        )

        with pytest.raises(ContextHistoryCursorInvalidatedError, match="digest/count"):
            await policy.load_history(conversation_store, session_id)

        key = MemoryStateKey(
            "tenant-a",
            "agent-a",
            session_id,
            policy.policy_fingerprint,
        )
        assert await v2_store.load(key) is None
        assert conversation_store.full_load_calls == 0
        assert conversation_store.tail_calls == [(session_id, 0, 2)]

    asyncio.run(scenario())


def test_loader_legacy_migration_cas_race_has_one_verified_winner() -> None:
    async def scenario() -> None:
        session_id = "legacy-cas-race"
        conversation_store = _NoFullLoadConversationStore()
        messages = tuple(Message.user(f"m-{index}") for index in range(1, 33))
        await conversation_store.append(session_id, messages)
        v2_store = _BarrierCreateStateStore()
        legacy_store = _CountingLegacyStateStore()
        scoped_legacy_store = ScopedLegacyMemoryStateStore(
            legacy_store,
            tenant_id="tenant-a",
            agent_id="agent-a",
        )
        policies = tuple(
            DurableSummarizingConversationMemoryPolicy(
                budget=TokenBudget(100),
                summarizer=_RecordingSummarizer(),
                state_store=v2_store,
                tenant_id="tenant-a",
                agent_id="agent-a",
                token_counter=_MessageCountTokenCounter(),
                history_page_size=7,
                legacy_state_store=scoped_legacy_store,
            )
            for _ in range(2)
        )
        assert policies[0].policy_fingerprint == policies[1].policy_fingerprint
        await legacy_store.save(
            session_id,
            MemorySnapshot(
                summary="one legacy summary",
                covered_message_count=len(messages),
                covered_prefix_digest=_messages_digest(messages),
                policy_fingerprint=policies[0].policy_fingerprint,
            ),
        )

        views = await asyncio.gather(
            *(
                policy.load_history(conversation_store, session_id)
                for policy in policies
            )
        )
        key = MemoryStateKey(
            "tenant-a",
            "agent-a",
            session_id,
            policies[0].policy_fingerprint,
        )
        winner = await v2_store.load(key)

        assert winner is not None
        assert winner.version == 1
        assert winner.covered_through_sequence == len(messages)
        assert all(view.snapshot == winner for view in views)
        assert all(view.tail == () for view in views)
        assert legacy_store.load_calls == 2
        assert conversation_store.full_load_calls == 0

    asyncio.run(scenario())


def test_loader_accepts_legacy_migration_with_authoritative_source_ids() -> None:
    async def scenario() -> None:
        session_id = "legacy-verified"
        conversation_store = _NoFullLoadConversationStore()
        await conversation_store.append(
            session_id,
            [
                Message.user("u1"),
                Message.assistant("a1"),
                Message.user("u2"),
            ],
        )
        prefix = await conversation_store.load_tail(session_id, 0, 2)
        state_store = InMemoryContextMemoryStateStore()
        key = MemoryStateKey("tenant-a", "agent-a", session_id, "policy-a")
        migrated = migrate_memory_snapshot(
            MemorySnapshot(
                summary="verified legacy summary",
                covered_message_count=2,
                covered_prefix_digest="legacy-digest",
                policy_fingerprint=key.policy_fingerprint,
            ),
            key=key,
            source_message_ids=tuple(item.message_id for item in prefix.items),
        )
        assert await state_store.save_if_version(0, migrated)
        conversation_store.tail_calls.clear()
        loader = DurableContextHistoryLoader(
            state_store=state_store,
            tenant_id=key.tenant_id,
            agent_id=key.agent_id,
            policy_fingerprint=key.policy_fingerprint,
        )

        view = await loader.load_history(conversation_store, session_id)

        assert view.started_after_sequence == 2
        assert [item.message.content for item in view.tail] == ["u2"]
        assert "verified legacy summary" in (view.messages[0].content or "")
        assert conversation_store.full_load_calls == 0

    asyncio.run(scenario())


def test_durable_policy_compacts_incrementally_and_advances_absolute_cursor() -> None:
    async def scenario() -> None:
        session_id = "incremental"
        conversation_store = _NoFullLoadConversationStore()
        await conversation_store.append(
            session_id,
            [
                Message.user("u1"),
                Message.assistant("a1"),
                Message.user("u2"),
                Message.assistant("a2"),
                Message.user("u3"),
                Message.assistant("a3"),
            ],
        )
        state_store = InMemoryContextMemoryStateStore()
        summarizer = _RecordingSummarizer()
        policy = DurableSummarizingConversationMemoryPolicy(
            budget=TokenBudget(1_000),
            summarizer=summarizer,
            state_store=state_store,
            tenant_id="tenant-a",
            agent_id="agent-a",
            token_counter=_MessageCountTokenCounter(),
            max_history_turns=1,
            history_page_size=16,
        )

        first_view = await policy.load_history(conversation_store, session_id)
        first_request = _request(first_view.messages, session_id=session_id)
        first_result = await policy.prepare(first_request)
        key = MemoryStateKey(
            "tenant-a",
            "agent-a",
            session_id,
            policy.policy_fingerprint,
        )
        first_state = await state_store.load(key)

        assert first_state is not None
        assert first_state.covered_through_sequence == 4
        assert first_state.version == 1
        assert len(first_state.source_message_ids) == 4
        assert first_result.summarized_messages == 4
        assert summarizer.calls[0][1] is None
        replay = await policy.prepare(first_request)
        assert replay.metadata["cache_hit"] is True
        assert len(summarizer.calls) == 1

        await conversation_store.append(
            session_id,
            [Message.user("u4"), Message.assistant("a4")],
        )
        conversation_store.tail_calls.clear()
        second_view = await policy.load_history(conversation_store, session_id)
        assert conversation_store.tail_calls == [
            (session_id, 0, 4),
            (session_id, 4, 16),
            (session_id, 0, 4),
            (session_id, 4, 4),
            (session_id, 0, 4),
        ]
        assert second_view.started_after_sequence == 4
        assert len(second_view.tail) == 4
        assert first_result.messages[1] == second_view.messages[0]
        assert first_result.messages[1].role is MessageRole.USER

        second_result = await policy.prepare(
            _request(second_view.messages, session_id=session_id, current="next")
        )
        second_state = await state_store.load(key)

        assert second_state is not None
        assert second_state.covered_through_sequence == 6
        assert second_state.version == 2
        assert len(second_state.source_message_ids) == 6
        assert second_result.summarized_messages == 3
        assert summarizer.calls[1][1] == first_state.structured_summary.summary
        assert [message.content for message in summarizer.calls[1][0]] == ["u3", "a3"]
        assert conversation_store.full_load_calls == 0

    asyncio.run(scenario())


def test_oversized_custom_summary_fails_before_cas_persistence() -> None:
    async def scenario() -> None:
        session_id = "oversized-summary"
        conversation_store = _NoFullLoadConversationStore()
        await conversation_store.append(
            session_id,
            [Message.user("u1"), Message.assistant("a1")],
        )
        state_store = InMemoryContextMemoryStateStore()
        policy = DurableSummarizingConversationMemoryPolicy(
            budget=TokenBudget(1_000),
            summarizer=_OversizedSummarizer(),
            state_store=state_store,
            tenant_id="tenant-a",
            agent_id="agent-a",
            token_counter=_MessageCountTokenCounter(),
            max_history_turns=0,
        )
        view = await policy.load_history(conversation_store, session_id)

        with pytest.raises(ContextMemoryIntegrityError, match="bounded v2 schema"):
            await policy.prepare(_request(view.messages, session_id=session_id))

        key = MemoryStateKey(
            "tenant-a",
            "agent-a",
            session_id,
            policy.policy_fingerprint,
        )
        assert await state_store.load(key) is None

    asyncio.run(scenario())


def test_durable_policy_concurrent_compaction_has_one_cas_state() -> None:
    async def scenario() -> None:
        session_id = "race"
        conversation_store = _NoFullLoadConversationStore()
        await conversation_store.append(
            session_id,
            [
                Message.user("u1"),
                Message.assistant("a1"),
                Message.user("u2"),
                Message.assistant("a2"),
            ],
        )
        state_store = InMemoryContextMemoryStateStore()
        summarizer = _RecordingSummarizer()
        policies = tuple(
            DurableSummarizingConversationMemoryPolicy(
                budget=TokenBudget(1_000),
                summarizer=summarizer,
                state_store=state_store,
                tenant_id="tenant-a",
                agent_id="agent-a",
                token_counter=_MessageCountTokenCounter(),
                max_history_turns=0,
            )
            for _ in range(2)
        )
        assert policies[0].policy_fingerprint == policies[1].policy_fingerprint
        views = await asyncio.gather(
            *(
                policy.load_history(conversation_store, session_id)
                for policy in policies
            )
        )

        results = await asyncio.gather(
            *(
                policy.prepare(_request(view.messages, session_id=session_id))
                for policy, view in zip(policies, views, strict=True)
            )
        )

        key = MemoryStateKey(
            "tenant-a",
            "agent-a",
            session_id,
            policies[0].policy_fingerprint,
        )
        state = await state_store.load(key)
        assert state is not None
        assert state.version == 1
        assert state.covered_through_sequence == 4
        assert len(summarizer.calls) in {1, 2}
        assert all(
            result.messages[1].metadata["moduagent.memory"] == "summary-v2"
            for result in results
        )

    asyncio.run(scenario())


def test_durable_policy_never_converts_state_integrity_failure_to_fallback() -> None:
    async def scenario() -> None:
        session_id = "integrity-failure"
        conversation_store = _NoFullLoadConversationStore()
        await conversation_store.append(
            session_id,
            [Message.user("u1"), Message.assistant("a1")],
        )
        state_store = _ToggleFailStateStore()
        policy = DurableSummarizingConversationMemoryPolicy(
            budget=TokenBudget(1_000),
            summarizer=_RecordingSummarizer(),
            state_store=state_store,
            tenant_id="tenant-a",
            agent_id="agent-a",
            token_counter=_MessageCountTokenCounter(),
            max_history_turns=0,
        )
        view = await policy.load_history(conversation_store, session_id)
        state_store.fail_load = True

        with pytest.raises(ContextMemoryIntegrityError):
            await policy.prepare(_request(view.messages, session_id=session_id))

    asyncio.run(scenario())


def test_late_defined_integrity_subclass_propagates_as_original_type() -> None:
    class AdapterIntegrityFailure(ContextMemoryIntegrityError):
        pass

    class AdapterStateStore(InMemoryContextMemoryStateStore):
        def __init__(self) -> None:
            super().__init__()
            self.fail_load = False

        async def load(self, key: MemoryStateKey):
            if self.fail_load:
                raise AdapterIntegrityFailure("adapter state is inconsistent")
            return await super().load(key)

    async def scenario() -> None:
        session_id = "late-integrity-subclass"
        conversation_store = _NoFullLoadConversationStore()
        await conversation_store.append(
            session_id,
            [Message.user("u1"), Message.assistant("a1")],
        )
        state_store = AdapterStateStore()
        policy = DurableSummarizingConversationMemoryPolicy(
            budget=TokenBudget(1_000),
            summarizer=_RecordingSummarizer(),
            state_store=state_store,
            tenant_id="tenant-a",
            agent_id="agent-a",
            token_counter=_MessageCountTokenCounter(),
            max_history_turns=0,
        )
        view = await policy.load_history(conversation_store, session_id)
        state_store.fail_load = True

        with pytest.raises(
            AdapterIntegrityFailure,
            match="adapter state is inconsistent",
        ):
            await policy.prepare(_request(view.messages, session_id=session_id))

    asyncio.run(scenario())


def test_stale_view_rejects_summary_that_advanced_beyond_its_target() -> None:
    async def scenario() -> None:
        session_id = "future-summary-race"
        conversation_store = _NoFullLoadConversationStore()
        await conversation_store.append(
            session_id,
            [
                Message.user(f"u{index}")
                if index % 2
                else Message.assistant(f"a{index}")
                for index in range(1, 11)
            ],
        )
        state_store = InMemoryContextMemoryStateStore()
        summarizer = _RecordingSummarizer()

        def policy() -> DurableSummarizingConversationMemoryPolicy:
            return DurableSummarizingConversationMemoryPolicy(
                budget=TokenBudget(1_000),
                summarizer=summarizer,
                state_store=state_store,
                tenant_id="tenant-a",
                agent_id="agent-a",
                token_counter=_MessageCountTokenCounter(),
                max_history_turns=0,
            )

        stale_policy = policy()
        winner_policy = policy()
        stale_view = await stale_policy.load_history(conversation_store, session_id)
        await conversation_store.append(
            session_id,
            [Message.user("u11"), Message.assistant("a12")],
        )
        winner_view = await winner_policy.load_history(
            conversation_store,
            session_id,
        )
        await winner_policy.prepare(
            _request(winner_view.messages, session_id=session_id)
        )

        with pytest.raises(ContextMemoryWriteConflictError):
            await stale_policy.prepare(
                _request(stale_view.messages, session_id=session_id)
            )

        state = await state_store.load(
            MemoryStateKey(
                "tenant-a",
                "agent-a",
                session_id,
                stale_policy.policy_fingerprint,
            )
        )
        assert state is not None
        assert state.covered_through_sequence == 12
        assert len(summarizer.calls) == 1

    asyncio.run(scenario())


def test_same_cursor_with_divergent_prefix_digest_is_not_reused() -> None:
    async def scenario() -> None:
        session_id = "divergent-summary-race"
        conversation_store = _NoFullLoadConversationStore()
        await conversation_store.append(
            session_id,
            [
                Message.user("u1"),
                Message.assistant("a1"),
                Message.user("u2"),
                Message.assistant("a2"),
            ],
        )
        state_store = InMemoryContextMemoryStateStore()
        summarizer = _RecordingSummarizer()
        policy = DurableSummarizingConversationMemoryPolicy(
            budget=TokenBudget(1_000),
            summarizer=summarizer,
            state_store=state_store,
            tenant_id="tenant-a",
            agent_id="agent-a",
            token_counter=_MessageCountTokenCounter(),
            max_history_turns=0,
        )
        stale_view = await policy.load_history(conversation_store, session_id)
        key = MemoryStateKey(
            "tenant-a",
            "agent-a",
            session_id,
            policy.policy_fingerprint,
        )
        assert await state_store.save_if_version(
            0,
            ConversationSummarySnapshot(
                tenant_id=key.tenant_id,
                agent_id=key.agent_id,
                session_id=key.session_id,
                policy_fingerprint=key.policy_fingerprint,
                covered_through_sequence=4,
                covered_prefix_digest="divergent-prefix",
                structured_summary=ConversationSummary(summary="wrong branch"),
                source_message_ids=tuple(item.message_id for item in stale_view.tail),
                version=1,
            ),
        )

        with pytest.raises(ContextMemoryWriteConflictError):
            await policy.prepare(_request(stale_view.messages, session_id=session_id))

        assert summarizer.calls == []

    asyncio.run(scenario())


def test_durable_summary_source_ids_stay_bounded_for_long_sessions() -> None:
    async def scenario() -> None:
        session_id = "bounded-provenance"
        conversation_store = _NoFullLoadConversationStore()
        messages = [
            (Message.user(f"u{index}") if index % 2 else Message.assistant(f"a{index}"))
            for index in range(1, 301)
        ]
        await conversation_store.append(session_id, messages)
        state_store = InMemoryContextMemoryStateStore()
        policy = DurableSummarizingConversationMemoryPolicy(
            budget=TokenBudget(10_000),
            summarizer=_RecordingSummarizer(),
            state_store=state_store,
            tenant_id="tenant-a",
            agent_id="agent-a",
            token_counter=_MessageCountTokenCounter(),
            max_history_turns=0,
        )

        view = await policy.load_history(conversation_store, session_id)
        await policy.prepare(_request(view.messages, session_id=session_id))
        state = await state_store.load(
            MemoryStateKey(
                "tenant-a",
                "agent-a",
                session_id,
                policy.policy_fingerprint,
            )
        )

        assert state is not None
        assert state.covered_through_sequence == 300
        assert len(state.source_message_ids) == 256
        assert state.source_message_ids[0] == view.tail[0].message_id
        assert state.source_message_ids[-1] == view.tail[-1].message_id
        assert len(set(state.source_message_ids)) == 256

    asyncio.run(scenario())


def test_coordinator_uses_policy_loader_without_content_metadata_leakage() -> None:
    async def scenario() -> None:
        session_id = "runtime-long-session"
        conversation_store = _NoFullLoadConversationStore(agent_id="context-agent")
        await conversation_store.append(
            session_id,
            [
                Message.user("old-user"),
                Message.assistant("old-assistant"),
                Message.user("recent-user"),
                Message.assistant("recent-assistant"),
            ],
        )
        state_store = InMemoryContextMemoryStateStore()
        policy = DurableSummarizingConversationMemoryPolicy(
            budget=TokenBudget(1_000),
            summarizer=_RecordingSummarizer(),
            state_store=state_store,
            tenant_id="tenant-a",
            agent_id="context-agent",
            token_counter=_MessageCountTokenCounter(),
            max_history_turns=8,
        )
        key = MemoryStateKey(
            "tenant-a",
            "context-agent",
            session_id,
            policy.policy_fingerprint,
        )
        sensitive_summary = "SENSITIVE_CONTEXT_MARKER customer chose annual plan"
        anchor = await conversation_store.load_tail(session_id, 1, 1)
        conversation_store.tail_calls.clear()
        assert await state_store.save_if_version(
            0,
            ConversationSummarySnapshot(
                tenant_id=key.tenant_id,
                agent_id=key.agent_id,
                session_id=key.session_id,
                policy_fingerprint=key.policy_fingerprint,
                covered_through_sequence=2,
                covered_prefix_digest="digest:2",
                structured_summary=ConversationSummary(summary=sensitive_summary),
                source_message_ids=(anchor.items[0].message_id,),
                version=1,
            ),
        )
        model = _OneResponseModel("done")
        audit = AuditEventSink()
        agent = Agent(
            config=AgentConfig("context-agent", "Answer."),
            model=model,
            conversation_store=conversation_store,
            context_memory_policy=policy,
            event_sink=audit,
        )

        result = await agent.run("current-user", session_id=session_id)

        assert result.output == "done"
        assert conversation_store.full_load_calls == 0
        assert conversation_store.tail_calls == [
            (session_id, 1, 1),
            (session_id, 2, 256),
            (session_id, 1, 1),
            (session_id, 2, 2),
            (session_id, 1, 1),
        ]
        assert sensitive_summary in repr(model.requests[0].messages)
        assert sensitive_summary not in repr(result.metadata)
        assert sensitive_summary not in repr(audit.records)

    asyncio.run(scenario())


def test_runtime_context_assembler_budgets_actual_task_and_tool_protocol_request() -> (
    None
):
    @function_tool
    def echo(value: str) -> str:
        """Return one value for the Context assembly integration test."""

        return value

    class ToolSequenceModel:
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            if len(self.requests) == 1:
                return ModelResponse(
                    Message.assistant(
                        None,
                        (
                            ToolCall(
                                "call-context-1",
                                "echo",
                                {"value": "verified"},
                            ),
                        ),
                    )
                )
            return ModelResponse(Message.assistant("done"))

    async def scenario() -> None:
        session_id = "runtime-context-assembly"
        sensitive = "PRIVATE-OLDER-CONTEXT"
        conversation_store = _NoFullLoadConversationStore(agent_id="context-agent")
        await conversation_store.append(
            session_id,
            [
                Message.user(sensitive),
                Message.assistant("old answer"),
                Message.user("recent question"),
                Message.assistant("recent answer"),
            ],
        )
        policy = DurableSummarizingConversationMemoryPolicy(
            budget=TokenBudget(20),
            summarizer=_RecordingSummarizer(),
            state_store=InMemoryContextMemoryStateStore(),
            tenant_id="tenant-a",
            agent_id="context-agent",
            token_counter=_MessageCountTokenCounter(),
            max_history_turns=1,
        )
        model = ToolSequenceModel()
        agent = Agent(
            config=AgentConfig("context-agent", "Answer with tools."),
            model=model,
            tools=[echo],
            conversation_store=conversation_store,
            context_memory_policy=policy,
        )

        events = [
            event
            async for event in agent.stream_all(
                "current task",
                session_id=session_id,
                user_context={"tenant_id": "tenant-a"},
            )
        ]

        assert events[-1].data["result"].output == "done"
        assert len(model.requests) == 2
        second = model.requests[1]
        call_index = next(
            index
            for index, message in enumerate(second.messages)
            if message.role is MessageRole.ASSISTANT and message.tool_calls
        )
        assert second.messages[call_index].tool_calls[0].id == "call-context-1"
        assert second.messages[call_index + 1].role is MessageRole.TOOL
        assert second.messages[call_index + 1].tool_call_id == "call-context-1"
        assert second.messages[call_index - 1].content == "current task"

        compacted = [
            event for event in events if event.type is EventType.MEMORY_COMPACTED
        ]
        assert compacted
        final_memory = compacted[-1].data
        assert final_memory["context_assembly_algorithm"] == "context-assembler-v1"
        assert final_memory["context_assembly_dropped_items"] == 0
        assert final_memory["context_assembly_budget_tokens"] == 20
        assert final_memory["context_assembly_used_tokens"] <= 20
        assert final_memory["context_assembly_source_counts"] == {
            "conversation_summary": 1,
            "current_run": 0,
            "recent_turn": 2,
            "request_contract": 1,
            "system_policy": 1,
            "task_projection": 1,
            "tool_protocol": 2,
        } or final_memory["context_assembly_source_counts"] == {
            "conversation_summary": 1,
            "recent_turn": 2,
            "request_contract": 1,
            "system_policy": 1,
            "task_projection": 1,
            "tool_protocol": 2,
        }
        assert sensitive not in repr(final_memory)
        assert "verified" not in repr(final_memory)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("policy_tenant", "policy_agent", "user_context"),
    [
        ("tenant-a", "context-agent", {"tenant_id": "tenant-b"}),
        ("tenant-a", "another-agent", {}),
    ],
)
def test_coordinator_rejects_context_memory_scope_mismatch_before_model(
    policy_tenant: str,
    policy_agent: str,
    user_context: dict[str, str],
) -> None:
    async def scenario() -> None:
        conversation_store = _NoFullLoadConversationStore()
        model = _OneResponseModel("must-not-run")
        policy = DurableSummarizingConversationMemoryPolicy(
            budget=TokenBudget(1_000),
            summarizer=_RecordingSummarizer(),
            state_store=InMemoryContextMemoryStateStore(),
            tenant_id=policy_tenant,
            agent_id=policy_agent,
            token_counter=_MessageCountTokenCounter(),
        )
        agent = Agent(
            config=AgentConfig("context-agent", "Answer."),
            model=model,
            conversation_store=conversation_store,
            context_memory_policy=policy,
        )

        result = await agent.run(
            "current-user",
            session_id="scope-session",
            user_context=user_context,
        )

        assert result.finish_reason.value == "error"
        assert result.metadata["error_summary"] == {
            "category": "configuration",
            "code": "invalid_configuration",
            "retryable": False,
            "resumable": False,
        }
        assert model.requests == []
        assert conversation_store.tail_calls == []
        assert conversation_store.full_load_calls == 0

    asyncio.run(scenario())


def test_loader_rejects_stale_cursor_after_session_clear_and_reuse() -> None:
    async def scenario() -> None:
        session_id = "reused-session"
        conversation_store = _NoFullLoadConversationStore()
        await conversation_store.append(
            session_id,
            [Message.user("old-u"), Message.assistant("old-a")],
        )
        old_prefix = await conversation_store.load_tail(session_id, 0, 2)
        state_store = InMemoryContextMemoryStateStore()
        key = MemoryStateKey("tenant-a", "agent-a", session_id, "policy-a")
        assert await state_store.save_if_version(
            0,
            replace(
                _snapshot(
                    key,
                    cursor=2,
                    anchor_message_id=old_prefix.items[-1].message_id,
                ),
                source_message_ids=tuple(item.message_id for item in old_prefix.items),
            ),
        )
        await conversation_store.clear(session_id)
        await conversation_store.append(
            session_id,
            [
                Message.user("new-u1"),
                # Keep the old final anchor identical. Validation must still
                # notice that another covered message changed.
                Message.assistant("old-a"),
                Message.user("new-u2"),
                Message.assistant("new-a2"),
            ],
        )
        conversation_store.tail_calls.clear()
        loader = DurableContextHistoryLoader(
            state_store=state_store,
            tenant_id=key.tenant_id,
            agent_id=key.agent_id,
            policy_fingerprint=key.policy_fingerprint,
        )

        with pytest.raises(ContextHistoryCursorInvalidatedError):
            await loader.load_history(conversation_store, session_id)

        assert conversation_store.tail_calls == [(session_id, 0, 2)]
        assert conversation_store.full_load_calls == 0

    asyncio.run(scenario())


def test_loader_revalidation_rejects_clear_reuse_during_tail_read() -> None:
    class ClearReuseAfterReadStore(_NoFullLoadConversationStore):
        def __init__(self) -> None:
            super().__init__()
            self.armed = False
            self.mutated = False

        async def load_tail(
            self,
            session_id: str,
            after_sequence: int = 0,
            limit: int = 100,
        ):
            page = await super().load_tail(session_id, after_sequence, limit)
            if self.armed and not self.mutated:
                self.mutated = True
                await super().clear(session_id)
                await super().append(
                    session_id,
                    [Message.user("new-u"), Message.assistant("new-a")],
                )
            return page

    async def scenario() -> None:
        session_id = "reuse-during-read"
        conversation_store = ClearReuseAfterReadStore()
        await conversation_store.append(
            session_id,
            [Message.user("old-u"), Message.assistant("old-a")],
        )
        conversation_store.armed = True
        loader = DurableContextHistoryLoader(
            state_store=InMemoryContextMemoryStateStore(),
            tenant_id="tenant-a",
            agent_id="agent-a",
            policy_fingerprint="policy-a",
            page_size=16,
        )

        with pytest.raises(ContextHistoryCursorInvalidatedError, match="tail changed"):
            await loader.load_history(conversation_store, session_id)

        assert conversation_store.tail_calls == [
            (session_id, 0, 16),
            (session_id, 0, 2),
        ]
        assert conversation_store.full_load_calls == 0

    asyncio.run(scenario())


def test_policy_coordinated_clear_removes_summary_before_conversation() -> None:
    async def scenario() -> None:
        session_id = "clear-session"
        conversation_store = _NoFullLoadConversationStore()
        await conversation_store.append(
            session_id,
            [Message.user("old-u"), Message.assistant("old-a")],
        )
        state_store = InMemoryContextMemoryStateStore()
        policy = DurableSummarizingConversationMemoryPolicy(
            budget=TokenBudget(1_000),
            summarizer=_RecordingSummarizer(),
            state_store=state_store,
            tenant_id="tenant-a",
            agent_id="agent-a",
            token_counter=_MessageCountTokenCounter(),
            max_history_turns=0,
        )
        view = await policy.load_history(conversation_store, session_id)
        await policy.prepare(_request(view.messages, session_id=session_id))
        key = MemoryStateKey(
            "tenant-a",
            "agent-a",
            session_id,
            policy.policy_fingerprint,
        )
        assert await state_store.load(key) is not None

        await policy.clear_history(conversation_store, session_id)

        assert await state_store.load(key) is None
        empty = await conversation_store.load_tail(session_id, 0, 1)
        assert empty.items == ()

    asyncio.run(scenario())
