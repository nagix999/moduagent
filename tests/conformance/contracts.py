from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from moduagent.definitions import AgentEndpoint, AgentRef
from moduagent.delegation import (
    BudgetLease,
    BudgetLedger,
    BudgetStateStore,
    DelegationContext,
    DelegationEndpointError,
    DelegationOutcome,
    DelegationOutcomeStatus,
    DelegationReceipt,
    DelegationReceiptStatus,
    DelegationReceiptStore,
    ExecutionGroupBudgetState,
    ExecutionGroupLimits,
    InMemoryBudgetLedger,
    RunLineage,
)
from moduagent.memory.context import (
    ContextMemoryCursorRegressionError,
    ContextMemoryStateStore,
    ConversationSummary,
    ConversationSummarySnapshot,
    MemoryStateKey,
)
from moduagent.messages import Message, MessageRole
from moduagent.models import (
    ModelCapabilities,
    ModelClient,
    ModelRequest,
    ModelResponse,
    classify_model_error,
)
from moduagent.persistence import CheckpointStore, ConversationStore, RunCheckpoint
from moduagent.persistence.conversation import IdempotentConversationStore


async def assert_model_provider_contract(
    client: ModelClient,
    *,
    expected_content: str = "contract-ok",
) -> ModelRequest:
    """Check the provider-neutral request and normalized result boundary."""

    assert isinstance(client, ModelClient)
    assert isinstance(client.capabilities, ModelCapabilities)
    request = ModelRequest(
        messages=(
            Message.system("Follow the contract."),
            Message.user("Return the conformance marker."),
        ),
        options={"temperature": 0},
        provider_options={"seed": 17},
    )
    original = copy.deepcopy(request)

    response = await client.complete(request)

    assert request == original
    assert isinstance(response, ModelResponse)
    assert response.message.role is MessageRole.ASSISTANT
    assert response.message.content == expected_content
    assert response.tool_calls == response.message.tool_calls
    assert response.finish_reason == "stop"
    assert response.usage.input_tokens == 2
    assert response.usage.output_tokens == 3
    assert response.usage.total_tokens == 5
    assert isinstance(response.provider_metadata, dict)
    return request


async def assert_model_provider_error_contract(
    client: ModelClient,
    *,
    category: str,
    code: str,
    retryable: bool,
) -> None:
    """Check that provider errors retain the framework's safe classification."""

    request = ModelRequest(messages=(Message.user("fail deterministically"),))
    with pytest.raises(Exception) as captured:
        await client.complete(request)

    classification = classify_model_error(captured.value)
    assert classification.category == category
    assert classification.code == code
    assert classification.retryable is retryable


async def assert_conversation_store_contract(store: ConversationStore) -> None:
    """Check identity isolation, copy-safe round trips, and append-once support."""

    assert isinstance(store, ConversationStore)
    session_a = "contract-session-a"
    session_b = "contract-session-b"
    source_metadata = {"nested": {"value": "original"}}
    first = Message.user("안녕하세요", metadata=source_metadata)
    expected_first = Message.from_dict(copy.deepcopy(first.to_dict()))
    second = Message.assistant("hello")

    assert await store.load(session_a) == []
    await store.append(session_a, (first,))
    source_metadata["nested"]["value"] = "caller-mutated"
    await store.append(session_a, (second,))
    await store.append(session_b, (Message.user("isolated"),))

    loaded = await store.load(session_a)
    assert loaded == [expected_first, second]
    assert await store.load(session_b) == [Message.user("isolated")]
    nested = loaded[0].metadata["nested"]
    assert isinstance(nested, dict)
    nested["value"] = "loaded-copy-mutated"
    assert await store.load(session_a) == [expected_first, second]

    if getattr(store, "supports_idempotent_append", False) is True:
        assert isinstance(store, IdempotentConversationStore)
        batch = (Message.user("append once"), Message.assistant("recorded"))
        assert await store.append_once(session_a, "run:contract", batch) is True
        assert await store.append_once(session_a, "run:contract", batch) is False
        with pytest.raises(ValueError):
            await store.append_once(
                session_a,
                "run:contract",
                (Message.user("different payload"),),
            )
        assert (await store.load(session_a))[-2:] == list(batch)

    await store.clear(session_a)
    assert await store.load(session_a) == []
    assert await store.load(session_b) == [Message.user("isolated")]


async def assert_checkpoint_store_contract(store: CheckpointStore) -> None:
    """Check run identity, exact checkpoint round trip, copy isolation, and delete."""

    assert isinstance(store, CheckpointStore)
    timestamp = datetime(2026, 8, 1, 9, 30, tzinfo=timezone.utc)
    checkpoint = RunCheckpoint(
        run_id="contract-run-a",
        session_id="contract-session-a",
        messages=(Message.user("resume", metadata={"nested": {"value": 1}}),),
        metadata={"trace": {"value": "original"}},
        created_at=timestamp,
        updated_at=timestamp,
    )
    expected = RunCheckpoint.from_json(checkpoint.to_json())

    assert await store.load(checkpoint.run_id) is None
    await store.save(checkpoint.run_id, checkpoint)
    loaded = await store.load(checkpoint.run_id)
    assert loaded == expected
    assert await store.load("contract-run-b") is None
    assert loaded is not None
    trace = loaded.metadata["trace"]
    assert isinstance(trace, dict)
    trace["value"] = "loaded-copy-mutated"
    assert await store.load(checkpoint.run_id) == expected
    with pytest.raises(ValueError):
        await store.save("contract-run-b", checkpoint)

    await store.delete(checkpoint.run_id)
    assert await store.load(checkpoint.run_id) is None


async def assert_receipt_store_contract(store: DelegationReceiptStore) -> None:
    """Check atomic claim/CAS and immutable delegation identity."""

    assert isinstance(store, DelegationReceiptStore)
    initial = DelegationReceipt(
        delegation_id="dlg:contract-a",
        execution_group_id="group:contract-a",
        root_run_id="run:root-a",
        parent_run_id="run:root-a",
        parent_tool_call_id="call:delegate-a",
        caller_agent_ref=AgentRef("parent", "1.0.0"),
        callee_agent_ref=AgentRef("child", "1.0.0"),
        child_run_id="run:child-a",
        request_digest="sha256:" + ("a" * 64),
        context_digest="ctx:contract-a",
        owner_token="owner:initial",
        lease_id="lease:contract-a",
    )

    first, second = await asyncio.gather(store.claim(initial), store.claim(initial))
    assert sorted((first.created, second.created)) == [False, True]
    assert await store.get(initial.delegation_id) == initial

    writer_a = replace(
        initial,
        status=DelegationReceiptStatus.RUNNING,
        owner_token="owner:writer-a",
        revision=2,
    )
    writer_b = replace(
        initial,
        status=DelegationReceiptStatus.RUNNING,
        owner_token="owner:writer-b",
        revision=2,
    )
    results = await asyncio.gather(
        store.compare_and_swap(initial.delegation_id, 1, writer_a),
        store.compare_and_swap(initial.delegation_id, 1, writer_b),
    )
    assert sorted(results) == [False, True]
    winner = await store.get(initial.delegation_id)
    assert winner == writer_a or winner == writer_b
    assert winner is not None
    assert await store.compare_and_swap(initial.delegation_id, 1, writer_a) is False

    wrong_identity = replace(
        winner,
        delegation_id="dlg:contract-b",
        revision=winner.revision + 1,
    )
    with pytest.raises(ValueError):
        await store.compare_and_swap(
            initial.delegation_id,
            winner.revision,
            wrong_identity,
        )
    assert await store.get("dlg:contract-b") is None


async def assert_budget_state_store_contract(store: BudgetStateStore) -> None:
    """Check atomic create/CAS and execution-group identity fencing."""

    assert isinstance(store, BudgetStateStore)
    initial = ExecutionGroupBudgetState(
        execution_group_id="group:budget-a",
        limits=ExecutionGroupLimits(),
        absolute_deadline=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    created = await asyncio.gather(store.create(initial), store.create(initial))
    assert sorted(created) == [False, True]
    assert await store.load(initial.execution_group_id) == initial

    writer_a = replace(initial, revision=2, model_turns=1)
    writer_b = replace(initial, revision=2, tool_calls=1)
    results = await asyncio.gather(
        store.compare_and_swap(initial.execution_group_id, 1, writer_a),
        store.compare_and_swap(initial.execution_group_id, 1, writer_b),
    )
    assert sorted(results) == [False, True]
    winner = await store.load(initial.execution_group_id)
    assert winner == writer_a or winner == writer_b
    assert winner is not None
    assert (
        await store.compare_and_swap(initial.execution_group_id, 1, writer_a) is False
    )

    wrong_identity = replace(
        winner,
        execution_group_id="group:budget-b",
        revision=winner.revision + 1,
    )
    with pytest.raises(ValueError):
        await store.compare_and_swap(
            initial.execution_group_id,
            winner.revision,
            wrong_identity,
        )
    assert await store.load("group:budget-b") is None


async def assert_context_memory_store_contract(
    store: ContextMemoryStateStore,
) -> None:
    """Check composite identity, atomic CAS, and monotonic summary cursor."""

    assert isinstance(store, ContextMemoryStateStore)
    key = _memory_key("tenant-a")
    initial = _summary_snapshot(key, version=1, cursor=10, label="initial")
    assert await store.load(key) is None
    assert await store.save_if_version(0, initial) is True
    assert await store.save_if_version(0, initial) is False
    assert await store.load(key) == initial

    writer_a = _summary_snapshot(key, version=2, cursor=20, label="writer-a")
    writer_b = _summary_snapshot(key, version=2, cursor=21, label="writer-b")
    results = await asyncio.gather(
        store.save_if_version(1, writer_a),
        store.save_if_version(1, writer_b),
    )
    assert sorted(results) == [False, True]
    winner = await store.load(key)
    assert winner == writer_a or winner == writer_b
    assert winner is not None

    regressing = _summary_snapshot(
        key,
        version=3,
        cursor=winner.covered_through_sequence - 1,
        label="regressing",
    )
    with pytest.raises(ContextMemoryCursorRegressionError):
        await store.save_if_version(2, regressing)
    assert await store.load(key) == winner

    other_key = _memory_key("tenant-b")
    other = _summary_snapshot(other_key, version=1, cursor=1, label="other")
    assert await store.save_if_version(0, other) is True
    assert await store.load(other_key) == other
    await store.clear(key)
    assert await store.load(key) is None
    assert await store.load(other_key) == other


async def assert_agent_invoker_contract(
    invoker: Any,
    endpoint_factory: Callable[[object], AgentEndpoint],
) -> None:
    """Check typed outcomes, child identity, and cancellation propagation."""

    context, lease, ledger, child_run_id = await _delegated_invocation_scope()
    request = ConformanceDelegationRequest(task="verify")

    successful = _SuccessfulEndpointHandler()
    outcome = await invoker.invoke(
        endpoint_factory(successful),
        request,
        context,
        lease,
        ledger,
        child_run_id=child_run_id,
    )
    assert isinstance(outcome, DelegationOutcome)
    assert outcome.status is DelegationOutcomeStatus.COMPLETED
    assert outcome.child_run_id == child_run_id
    assert outcome.output == ConformanceDelegationOutput(result="ok")
    assert successful.received_identity == (
        context.execution_group_id,
        lease.lease_id,
        child_run_id,
    )

    with pytest.raises(DelegationEndpointError) as invalid_output:
        await invoker.invoke(
            endpoint_factory(_InvalidEndpointHandler()),
            request,
            context,
            lease,
            ledger,
            child_run_id=child_run_id,
        )
    assert invalid_output.value.code == "delegation_endpoint_protocol_error"

    with pytest.raises(DelegationEndpointError) as wrong_child:
        await invoker.invoke(
            endpoint_factory(_WrongChildEndpointHandler()),
            request,
            context,
            lease,
            ledger,
            child_run_id=child_run_id,
        )
    assert wrong_child.value.code == "delegation_child_run_id_mismatch"

    blocking = _BlockingEndpointHandler()
    task = asyncio.create_task(
        invoker.invoke(
            endpoint_factory(blocking),
            request,
            context,
            lease,
            ledger,
            child_run_id=child_run_id,
        )
    )
    await asyncio.wait_for(blocking.started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert blocking.cancelled.is_set()


class ConformanceDelegationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str


class ConformanceDelegationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: str


class _SuccessfulEndpointHandler:
    def __init__(self) -> None:
        self.received_identity: tuple[str, str, str] | None = None

    async def _run_delegated(
        self,
        request: BaseModel,
        *,
        context: DelegationContext,
        budget: BudgetLease,
        budget_ledger: BudgetLedger,
        child_run_id: str,
    ) -> DelegationOutcome:
        del request, budget_ledger
        self.received_identity = (
            context.execution_group_id,
            budget.lease_id,
            child_run_id,
        )
        return DelegationOutcome(
            DelegationOutcomeStatus.COMPLETED,
            child_run_id,
            finish_reason="completed",
            output=ConformanceDelegationOutput(result="ok"),
        )


class _InvalidEndpointHandler:
    async def _run_delegated(self, *_: Any, **__: Any) -> object:
        return {"result": "not-a-DelegationOutcome"}


class _WrongChildEndpointHandler:
    async def _run_delegated(self, *_: Any, **__: Any) -> DelegationOutcome:
        return DelegationOutcome(
            DelegationOutcomeStatus.COMPLETED,
            "run:wrong-child",
            finish_reason="completed",
            output=ConformanceDelegationOutput(result="wrong"),
        )


class _BlockingEndpointHandler:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def _run_delegated(self, *_: Any, **__: Any) -> DelegationOutcome:
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


async def _delegated_invocation_scope() -> tuple[
    DelegationContext,
    BudgetLease,
    InMemoryBudgetLedger,
    str,
]:
    caller = AgentRef("contract-parent", "1.0.0")
    callee = AgentRef("contract-child", "1.0.0")
    root_run_id = "run:contract-root"
    child_run_id = "run:contract-child"
    group_id = "group:contract"
    deadline = datetime.now(timezone.utc) + timedelta(seconds=30)
    limits = ExecutionGroupLimits()
    lineage = RunLineage.root(run_id=root_run_id, agent=caller).child(
        child=callee,
        parent_run_id=root_run_id,
        delegation_id="dlg:contract-invoker",
        parent_tool_call_id="call:contract-invoker",
    )
    context = DelegationContext(
        lineage=lineage,
        execution_group_id=group_id,
        principal="principal:contract",
        tenant="tenant:contract",
        absolute_deadline=deadline,
        child_session_id="session:contract-child",
    )
    ledger = InMemoryBudgetLedger()
    await ledger.ensure_group(group_id, limits, absolute_deadline=deadline)
    lease = await ledger.reserve_delegation(
        group_id,
        callee,
        lease_id="lease:contract-invoker",
    )
    return context, lease, ledger, child_run_id


def _memory_key(tenant_id: str) -> MemoryStateKey:
    return MemoryStateKey(
        tenant_id=tenant_id,
        agent_id="contract-agent",
        session_id="contract-session",
        policy_fingerprint="sha256:contract-policy",
    )


def _summary_snapshot(
    key: MemoryStateKey,
    *,
    version: int,
    cursor: int,
    label: str,
) -> ConversationSummarySnapshot:
    return ConversationSummarySnapshot(
        tenant_id=key.tenant_id,
        agent_id=key.agent_id,
        session_id=key.session_id,
        policy_fingerprint=key.policy_fingerprint,
        covered_through_sequence=cursor,
        covered_prefix_digest=f"sha256:{label}:{cursor}",
        structured_summary=ConversationSummary(
            summary=f"summary:{label}",
            facts=(f"fact:{label}",),
        ),
        source_message_ids=(f"message:{label}",),
        version=version,
    )


__all__ = [
    "assert_agent_invoker_contract",
    "assert_budget_state_store_contract",
    "assert_checkpoint_store_contract",
    "assert_context_memory_store_contract",
    "assert_conversation_store_contract",
    "assert_model_provider_contract",
    "assert_model_provider_error_contract",
    "assert_receipt_store_contract",
]
