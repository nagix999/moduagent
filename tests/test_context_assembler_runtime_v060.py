from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from moduagent import (
    Agent,
    AgentConfig,
    ContextAssembler,
    ContextBudgetExceededError,
    ContextHistoryPaginationRequiredError,
    ContextMemoryIntegrityError,
    ConversationSummary,
    ConversationSummarySnapshot,
    DurableContextHistoryLoader,
    DurableSummarizingConversationMemoryPolicy,
    InMemoryContextMemoryStateStore,
    InMemoryConversationStore,
    MemoryPhase,
    MemoryRequest,
    MemoryStateKey,
    ModelRequest,
    ModelResponse,
    ScopedConversationStore,
    ScopedLegacyMemoryStateStore,
    SummaryResult,
    TokenBudget,
)
from moduagent.errors import ModelInvocationError
from moduagent.memory.context.runtime_assembly import select_runtime_context
from moduagent.memory.context.history import _source_validation_windows
from moduagent.memory.state import InMemoryMemoryStateStore
from moduagent.messages import Message, ToolCall, Usage


class _MessageCounter:
    async def count_request(self, request: ModelRequest) -> int:
        return len(request.messages)


class _SchemaAwareCounter:
    async def count_request(self, request: ModelRequest) -> int:
        return (
            len(request.messages)
            + 4 * len(request.tools)
            + (7 if request.output_schema is not None else 0)
        )


class _NonAdditiveCounter:
    async def count_request(self, request: ModelRequest) -> int:
        turns = [
            message.content
            for message in request.messages
            if (message.content or "").startswith("turn-")
        ]
        # The full three-turn candidate is additive, while exactly two turns
        # trigger provider chat-template overhead. This forces exact correction
        # after the assembler's calibrated allocation.
        return len(request.messages) + (2 if len(turns) == 2 else 0)


class _BoundaryAwareCounter:
    async def count_request(self, request: ModelRequest) -> int:
        total = 0
        for message in request.messages:
            if message.metadata.get("moduagent.memory") == "summary-v2":
                total += 50 if "HUGE-SUMMARY" in (message.content or "") else 2
            else:
                total += 1
        return total


class _FixedSummarizer:
    cache_fingerprint = "fixed-summary-v1"

    def __init__(self, summary: str) -> None:
        self.summary = summary

    async def summarize(
        self,
        messages: tuple[Message, ...],
        *,
        previous_summary: str | None = None,
    ) -> SummaryResult:
        del messages, previous_summary
        return SummaryResult(self.summary, Usage(1, 1, 2))


class _FailingSummarizer:
    cache_fingerprint = "failing-summary-v1"

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def summarize(
        self,
        messages: tuple[Message, ...],
        *,
        previous_summary: str | None = None,
    ) -> SummaryResult:
        del messages, previous_summary
        raise self.error


class _RecordingModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(Message.assistant("done"))


class _CountingConversationStore(InMemoryConversationStore):
    def __init__(self) -> None:
        super().__init__()
        self.clear_calls = 0
        self.tail_calls = 0

    async def clear(self, session_id: str) -> None:
        self.clear_calls += 1
        await super().clear(session_id)

    async def load_tail(self, session_id: str, after_sequence=0, limit=100):
        self.tail_calls += 1
        return await super().load_tail(session_id, after_sequence, limit)


class _CountingStateStore(InMemoryContextMemoryStateStore):
    def __init__(self) -> None:
        super().__init__()
        self.load_calls = 0
        self.clear_calls = 0

    async def load(self, key: MemoryStateKey):
        self.load_calls += 1
        return await super().load(key)

    async def clear(self, key: MemoryStateKey) -> None:
        self.clear_calls += 1
        await super().clear(key)


class _WinnerStateStore(InMemoryContextMemoryStateStore):
    def __init__(self) -> None:
        super().__init__()
        self.injected = False

    async def save_if_version(self, expected_version, next_snapshot):
        if expected_version == 0 and not self.injected:
            self.injected = True
            winner = replace(
                next_snapshot,
                structured_summary=ConversationSummary(summary="HUGE-SUMMARY" * 20),
            )
            assert await super().save_if_version(0, winner)
            return False
        return await super().save_if_version(expected_version, next_snapshot)


def _scoped_store(
    *,
    tenant_id: str = "tenant-a",
    agent_id: str = "agent-a",
    raw: InMemoryConversationStore | None = None,
) -> ScopedConversationStore:
    return ScopedConversationStore(
        raw or InMemoryConversationStore(),
        tenant_id=tenant_id,
        agent_id=agent_id,
    )


def test_selector_counts_tools_and_output_schema_in_required_unified_budget() -> None:
    async def scenario() -> None:
        system = (Message.system("system"),)
        protected = (Message.user("task"),)
        old_turn = (Message.user("old"), Message.assistant("answer"))
        request = ModelRequest(
            messages=(*system, *old_turn, *protected),
            tools=({"name": "lookup"},),
            output_schema={"type": "object"},
        )

        selection = await select_runtime_context(
            assembler=ContextAssembler(),
            model_request=request,
            system=system,
            summary=None,
            turns=(old_turn,),
            protected=protected,
            token_counter=_SchemaAwareCounter(),
            token_budget=13,
        )

        assert selection.selected_tokens == 13
        assert selection.selected_turns == ()
        assert selection.messages == (*system, *protected)

    asyncio.run(scenario())


def test_selector_exact_correction_keeps_only_newest_contiguous_suffix() -> None:
    async def scenario() -> None:
        system = (Message.system("system"),)
        protected = (Message.user("task"),)
        turns = tuple((Message.user(f"turn-{index}"),) for index in range(3))
        request = ModelRequest(
            messages=(*system, *(item for turn in turns for item in turn), *protected)
        )

        selection = await select_runtime_context(
            assembler=ContextAssembler(),
            model_request=request,
            system=system,
            summary=None,
            turns=turns,
            protected=protected,
            token_counter=_NonAdditiveCounter(),
            token_budget=4,
        )

        assert selection.selected_turns == (turns[-1],)
        assert [message.content for message in selection.messages] == [
            "system",
            "turn-2",
            "task",
        ]
        assert selection.selected_tokens == 3

    asyncio.run(scenario())


def test_selector_preserves_required_tool_call_and_all_results_atomically() -> None:
    async def scenario() -> None:
        calls = (
            ToolCall("call-a", "a", {}),
            ToolCall("call-b", "b", {}),
        )
        system = (Message.system("system"),)
        protected = (
            Message.user("task"),
            Message.assistant(None, calls),
            Message.tool("a-result", call_id="call-a", name="a"),
            Message.tool("b-result", call_id="call-b", name="b"),
        )
        request = ModelRequest(messages=(*system, *protected))
        selection = await select_runtime_context(
            assembler=ContextAssembler(),
            model_request=request,
            system=system,
            summary=None,
            turns=(),
            protected=protected,
            token_counter=_MessageCounter(),
            token_budget=5,
        )
        assert selection.messages == request.messages

        with pytest.raises(ContextBudgetExceededError):
            await select_runtime_context(
                assembler=ContextAssembler(),
                model_request=request,
                system=system,
                summary=None,
                turns=(),
                protected=protected,
                token_counter=_MessageCounter(),
                token_budget=4,
            )

    asyncio.run(scenario())


def test_unselected_new_summary_never_advances_v2_cursor() -> None:
    async def scenario() -> None:
        session_id = "summary-preflight"
        conversation_store = _scoped_store()
        await conversation_store.append(
            session_id,
            [Message.user("old-u"), Message.assistant("old-a")],
        )
        state_store = InMemoryContextMemoryStateStore()
        policy = DurableSummarizingConversationMemoryPolicy(
            budget=TokenBudget(5),
            summarizer=_FixedSummarizer("HUGE-SUMMARY" * 20),
            state_store=state_store,
            tenant_id="tenant-a",
            agent_id="agent-a",
            token_counter=_BoundaryAwareCounter(),
            max_history_turns=0,
        )
        view = await policy.load_history(conversation_store, session_id)
        request_messages = (
            Message.system("system"),
            *view.messages,
            Message.user("current"),
        )
        result = await policy.prepare(
            MemoryRequest(
                run_id="run-summary-preflight",
                session_id=session_id,
                phase=MemoryPhase.ACT,
                model_request=ModelRequest(messages=request_messages),
                protected_from=len(request_messages) - 1,
            )
        )
        key = MemoryStateKey(
            "tenant-a", "agent-a", session_id, policy.policy_fingerprint
        )

        assert await state_store.load(key) is None
        assert result.messages == (request_messages[0], request_messages[-1])
        reloaded = await policy.load_history(conversation_store, session_id)
        assert reloaded.snapshot is None
        assert [item.message.content for item in reloaded.tail] == ["old-u", "old-a"]

    asyncio.run(scenario())


def test_persisted_summary_that_cannot_fit_is_omitted_before_model() -> None:
    async def scenario() -> None:
        session_id = "persisted-summary-unfit"
        conversation_store = _scoped_store(agent_id="context-agent")
        await conversation_store.append(
            session_id,
            [Message.user("old-u"), Message.assistant("old-a")],
        )
        prefix = await conversation_store.load_tail(session_id, 0, 2)
        state_store = InMemoryContextMemoryStateStore()
        policy = DurableSummarizingConversationMemoryPolicy(
            budget=TokenBudget(5),
            summarizer=_FixedSummarizer("unused"),
            state_store=state_store,
            tenant_id="tenant-a",
            agent_id="context-agent",
            token_counter=_BoundaryAwareCounter(),
            max_history_turns=0,
        )
        key = MemoryStateKey(
            "tenant-a", "context-agent", session_id, policy.policy_fingerprint
        )
        assert await state_store.save_if_version(
            0,
            ConversationSummarySnapshot(
                tenant_id=key.tenant_id,
                agent_id=key.agent_id,
                session_id=key.session_id,
                policy_fingerprint=key.policy_fingerprint,
                covered_through_sequence=2,
                covered_prefix_digest="verified-prefix",
                structured_summary=ConversationSummary(summary="HUGE-SUMMARY" * 20),
                source_message_ids=tuple(item.message_id for item in prefix.items),
                version=1,
            ),
        )
        model = _RecordingModel()
        agent = Agent(
            config=AgentConfig("context-agent", "Answer."),
            model=model,
            conversation_store=conversation_store,
            context_memory_policy=policy,
        )

        result = await agent.run(
            "current",
            session_id=session_id,
            user_context={"tenant_id": "tenant-a"},
        )

        assert result.finish_reason.value == "completed"
        assert len(model.requests) == 1
        request = model.requests[0]
        assert [message.content for message in request.messages] == [
            "Answer.",
            "current",
        ]
        assert "HUGE-SUMMARY" not in repr(request)
        persisted = await state_store.load(key)
        assert persisted is not None
        assert persisted.version == 1

    asyncio.run(scenario())


def test_unfit_cas_winner_is_persisted_but_omitted_from_request() -> None:
    async def scenario() -> None:
        session_id = "winner-summary-revalidation"
        conversation_store = _scoped_store()
        await conversation_store.append(
            session_id,
            [Message.user("old-u"), Message.assistant("old-a")],
        )
        state_store = _WinnerStateStore()
        policy = DurableSummarizingConversationMemoryPolicy(
            budget=TokenBudget(5),
            summarizer=_FixedSummarizer("short"),
            state_store=state_store,
            tenant_id="tenant-a",
            agent_id="agent-a",
            token_counter=_BoundaryAwareCounter(),
            max_history_turns=0,
        )
        view = await policy.load_history(conversation_store, session_id)
        messages = (Message.system("system"), *view.messages, Message.user("current"))

        result = await policy.prepare(
            MemoryRequest(
                run_id="run-winner",
                session_id=session_id,
                phase=MemoryPhase.ACT,
                model_request=ModelRequest(messages=messages),
                protected_from=len(messages) - 1,
            )
        )

        assert result.messages == (messages[0], messages[-1])
        assert result.summarized_messages == 0
        assert result.dropped_messages == 2
        winner = await state_store.load(
            MemoryStateKey("tenant-a", "agent-a", session_id, policy.policy_fingerprint)
        )
        assert winner is not None
        assert "HUGE-SUMMARY" in winner.structured_summary.summary

    asyncio.run(scenario())


def test_summary_transport_failure_with_persisted_boundary_falls_back() -> None:
    async def scenario() -> None:
        session_id = "boundary-transport-fallback"
        conversation_store = _scoped_store()
        await conversation_store.append(
            session_id,
            [
                Message.user("old-u"),
                Message.assistant("old-a"),
                Message.user("new-u"),
                Message.assistant("new-a"),
            ],
        )
        prefix = await conversation_store.load_tail(session_id, 0, 2)
        state_store = InMemoryContextMemoryStateStore()
        policy = DurableSummarizingConversationMemoryPolicy(
            budget=TokenBudget(5),
            summarizer=_FailingSummarizer(
                ModelInvocationError("private transport detail")
            ),
            state_store=state_store,
            tenant_id="tenant-a",
            agent_id="agent-a",
            token_counter=_BoundaryAwareCounter(),
            max_history_turns=0,
        )
        key = MemoryStateKey(
            "tenant-a", "agent-a", session_id, policy.policy_fingerprint
        )
        persisted = ConversationSummarySnapshot(
            tenant_id=key.tenant_id,
            agent_id=key.agent_id,
            session_id=key.session_id,
            policy_fingerprint=key.policy_fingerprint,
            covered_through_sequence=2,
            covered_prefix_digest="verified-prefix",
            structured_summary=ConversationSummary(summary="persisted summary"),
            source_message_ids=tuple(item.message_id for item in prefix.items),
            version=1,
        )
        assert await state_store.save_if_version(0, persisted)
        view = await policy.load_history(conversation_store, session_id)
        messages = (Message.system("system"), *view.messages, Message.user("current"))

        result = await policy.prepare(
            MemoryRequest(
                run_id="run-boundary-fallback",
                session_id=session_id,
                phase=MemoryPhase.ACT,
                model_request=ModelRequest(messages=messages),
                protected_from=len(messages) - 1,
            )
        )

        assert result.messages == (messages[0], messages[-1])
        assert result.metadata["summary_error"] == "ModelInvocationError"
        assert result.summarized_messages == 0
        assert await state_store.load(key) == persisted

    asyncio.run(scenario())


def test_scoped_conversation_store_isolates_same_public_session_across_tenants() -> (
    None
):
    async def scenario() -> None:
        raw = InMemoryConversationStore()
        tenant_a = _scoped_store(tenant_id="tenant-a", raw=raw)
        tenant_b = _scoped_store(tenant_id="tenant-b", raw=raw)
        assert tenant_a.scoped_session_id("same") != tenant_b.scoped_session_id("same")

        await tenant_a.append("same", [Message.user("tenant-a-secret")])
        await tenant_b.append("same", [Message.user("tenant-b-secret")])

        assert [message.content for message in await tenant_a.load("same")] == [
            "tenant-a-secret"
        ]
        assert [message.content for message in await tenant_b.load("same")] == [
            "tenant-b-secret"
        ]

    asyncio.run(scenario())


def test_loader_and_clear_reject_raw_or_mismatched_scope_before_any_store_io() -> None:
    async def scenario() -> None:
        raw = _CountingConversationStore()
        state = _CountingStateStore()
        loader = DurableContextHistoryLoader(
            state_store=state,
            tenant_id="tenant-a",
            agent_id="agent-a",
            policy_fingerprint="policy-a",
        )
        with pytest.raises(ContextHistoryPaginationRequiredError):
            await loader.load_history(raw, "same")
        with pytest.raises(ContextHistoryPaginationRequiredError):
            await loader.clear_history(raw, "same")
        assert state.load_calls == 0
        assert state.clear_calls == 0
        assert raw.tail_calls == 0
        assert raw.clear_calls == 0

        tenant_b = _scoped_store(tenant_id="tenant-b", raw=raw)
        with pytest.raises(ContextMemoryIntegrityError, match="scope"):
            await loader.load_history(tenant_b, "same")
        with pytest.raises(ContextMemoryIntegrityError, match="scope"):
            await loader.clear_history(tenant_b, "same")
        assert state.load_calls == 0
        assert state.clear_calls == 0
        assert raw.tail_calls == 0
        assert raw.clear_calls == 0

    asyncio.run(scenario())


def test_scope_wrappers_reject_nested_or_cross_scope_migration_bindings() -> None:
    raw_conversation = InMemoryConversationStore()
    scoped = _scoped_store(raw=raw_conversation)
    with pytest.raises(ValueError, match="cannot be wrapped again"):
        ScopedConversationStore(
            scoped,
            tenant_id="tenant-b",
            agent_id="agent-a",
        )

    raw_legacy = InMemoryMemoryStateStore()
    scoped_legacy = ScopedLegacyMemoryStateStore(
        raw_legacy,
        tenant_id="tenant-b",
        agent_id="agent-a",
    )
    with pytest.raises(ContextMemoryIntegrityError, match="scope"):
        DurableSummarizingConversationMemoryPolicy(
            budget=TokenBudget(100),
            summarizer=_FixedSummarizer("summary"),
            state_store=InMemoryContextMemoryStateStore(),
            tenant_id="tenant-a",
            agent_id="agent-a",
            legacy_state_store=scoped_legacy,
        )
    with pytest.raises(ValueError, match="cannot be wrapped again"):
        ScopedLegacyMemoryStateStore(
            scoped_legacy,
            tenant_id="tenant-b",
            agent_id="agent-a",
        )


def test_v2_source_ids_with_legacy_text_prefix_are_not_marker_state() -> None:
    snapshot = ConversationSummarySnapshot(
        tenant_id="tenant-a",
        agent_id="agent-a",
        session_id="custom-source-ids",
        policy_fingerprint="policy-a",
        covered_through_sequence=2,
        covered_prefix_digest="v2-chained-digest",
        structured_summary=ConversationSummary(summary="valid v2 summary"),
        source_message_ids=("legacy-prefix:custom-1", "legacy-prefix:custom-2"),
        version=1,
    )

    assert _source_validation_windows(snapshot) == ((0, snapshot.source_message_ids),)


def test_single_legacy_text_source_id_is_not_marker_when_digest_differs() -> None:
    snapshot = ConversationSummarySnapshot(
        tenant_id="tenant-a",
        agent_id="agent-a",
        session_id="custom-single-source-id",
        policy_fingerprint="policy-a",
        covered_through_sequence=1,
        covered_prefix_digest="v2-chained-digest",
        structured_summary=ConversationSummary(summary="valid v2 summary"),
        source_message_ids=("legacy-prefix:custom-store-id",),
        version=1,
    )

    assert _source_validation_windows(snapshot) == ((0, snapshot.source_message_ids),)
