from __future__ import annotations

import copy
import asyncio
from types import SimpleNamespace

import pytest

from moduagent import (
    Agent,
    FinishReason,
    FullConversationMemoryPolicy,
    Message,
    ModelResponse,
    RecentTurnsConversationMemoryPolicy,
    RuntimeBindings,
    RunRequest,
    ScopedConversationStore,
    ToolCall,
    function_tool,
)
from moduagent.delegation import BudgetExceeded
from moduagent.persistence import (
    InMemoryCheckpointStore,
    InMemoryConversationStore,
    RunCheckpoint,
    RunSnapshot,
    StateMigrationError,
)
from moduagent.runtime import AgentEvent, EventPublisher, EventType


class _NoCallModel:
    async def complete(self, request):  # pragma: no cover - construction only
        raise AssertionError(f"unexpected model request: {request!r}")


class _OneResponseModel:
    async def complete(self, request):
        del request
        return ModelResponse(Message.assistant("done"))


def test_runtime_rejects_dynamic_tenant_store_scope_mismatch_before_model() -> None:
    async def scenario() -> None:
        raw_store = InMemoryConversationStore()
        scoped_store = ScopedConversationStore(
            raw_store,
            tenant_id="tenant-a",
            agent_id="scoped-agent",
        )
        await scoped_store.append(
            "shared-session",
            (Message.user("TENANT-A-SECRET"),),
        )
        agent = Agent.create(
            name="scoped-agent",
            model=_NoCallModel(),
            instructions="test",
            context_memory=RecentTurnsConversationMemoryPolicy(4),
            runtime_bindings=RuntimeBindings(
                conversation_store=scoped_store,
                tenant_context_provider=lambda: {"tenant_id": "tenant-b"},
            ),
        )

        result = await agent.run("tenant b request", session_id="shared-session")

        assert result.finish_reason is FinishReason.ERROR
        assert result.error_summary["category"] == "configuration"
        assert result.error_summary["code"] == "invalid_configuration"
        assert await scoped_store.load("shared-session") == [
            Message.user("TENANT-A-SECRET")
        ]

    asyncio.run(scenario())


class _RecordingLedger:
    def __init__(self) -> None:
        self.model: list[tuple[str, int]] = []
        self.tools: list[tuple[str, int]] = []

    async def reserve_model_turn(self, group_id: str, *, count: int = 1) -> None:
        self.model.append((group_id, count))

    async def reserve_tool_call(self, group_id: str, *, count: int = 1) -> None:
        self.tools.append((group_id, count))


class _RejectingLedger(_RecordingLedger):
    def __init__(self, rejected_operation: str) -> None:
        super().__init__()
        self.rejected_operation = rejected_operation

    async def reserve_model_turn(self, group_id: str, *, count: int = 1) -> None:
        if self.rejected_operation == "model":
            raise BudgetExceeded("execution_group_model_turns_exceeded")
        await super().reserve_model_turn(group_id, count=count)

    async def reserve_tool_call(self, group_id: str, *, count: int = 1) -> None:
        if self.rejected_operation == "tool":
            raise BudgetExceeded("execution_group_tool_calls_exceeded")
        await super().reserve_tool_call(group_id, count=count)


def test_context_memory_is_canonical_and_memory_remains_an_alias() -> None:
    policy = FullConversationMemoryPolicy()

    canonical = Agent.create(
        model=_NoCallModel(),
        instructions="test",
        context_memory=policy,
    )
    compatibility = Agent.create(
        model=_NoCallModel(),
        instructions="test",
        memory=policy,
    )

    assert canonical.context_memory_policy is policy
    assert canonical.conversation_memory_policy is policy
    assert compatibility.context_memory_policy is policy
    with pytest.raises(ValueError, match="either context_memory"):
        Agent.create(
            model=_NoCallModel(),
            instructions="test",
            memory=policy,
            context_memory=policy,
        )


def test_event_schema_v2_carries_content_free_delegation_identity() -> None:
    publisher = EventPublisher(
        run_id="child-run",
        session_id="child-session",
        engine_id="standard",
        execution_group_id="group-1",
        root_run_id="root-run",
        parent_run_id="parent-run",
        delegation_id="delegation-1",
        agent_id="researcher",
        agent_version="2.1.0",
        depth=1,
    )

    event = publisher.create(EventType.DELEGATION_STARTED)
    payload = event.to_dict()

    assert event.event_schema_version == 2
    assert payload["execution_group_id"] == "group-1"
    assert payload["root_run_id"] == "root-run"
    assert payload["parent_run_id"] == "parent-run"
    assert payload["child_run_id"] == "child-run"
    assert payload["delegation_id"] == "delegation-1"
    assert payload["agent_id"] == "researcher"
    assert payload["agent_version"] == "2.1.0"
    assert payload["depth"] == 1


def test_event_publisher_rejects_forged_delegation_correlation() -> None:
    root = EventPublisher(
        run_id="root-run",
        session_id="root-session",
        engine_id="standard",
        execution_group_id="root-run",
        root_run_id="root-run",
        agent_id="supervisor",
        agent_version="1.0.0",
    )
    child = EventPublisher(
        run_id="child-run",
        session_id="child-session",
        engine_id="standard",
        execution_group_id="root-run",
        root_run_id="root-run",
        parent_run_id="root-run",
        delegation_id="delegation-1",
        agent_id="worker",
        agent_version="1.0.0",
        depth=1,
    )

    with pytest.raises(ValueError, match="delegation correlation"):
        root.stamp(
            AgentEvent(
                EventType.MODEL_STARTED,
                "root-run",
                execution_group_id="root-run",
                root_run_id="root-run",
                child_run_id="forged-child",
                delegation_id="forged-delegation",
                agent_id="supervisor",
                agent_version="1.0.0",
            )
        )
    with pytest.raises(ValueError, match="delegation correlation"):
        child.stamp(
            AgentEvent(
                EventType.MODEL_STARTED,
                "child-run",
                execution_group_id="root-run",
                root_run_id="root-run",
                parent_run_id="root-run",
                child_run_id="other-child",
                delegation_id="delegation-1",
                agent_id="worker",
                agent_version="1.0.0",
                depth=1,
            )
        )


def test_event_publisher_has_an_explicit_related_delegation_path() -> None:
    publisher = EventPublisher(
        run_id="parent-run",
        session_id="parent-session",
        engine_id="standard",
        execution_group_id="root-run",
        root_run_id="root-run",
        agent_id="supervisor",
        agent_version="1.0.0",
    )
    event = AgentEvent(
        EventType.DELEGATION_STARTED,
        "parent-run",
        execution_group_id="root-run",
        root_run_id="root-run",
        child_run_id="child-run",
        delegation_id="delegation-1",
        agent_id="supervisor",
        agent_version="1.0.0",
    )

    with pytest.raises(ValueError, match="delegation correlation"):
        publisher.stamp(event)
    stamped = publisher.stamp(event, allow_related_delegation=True)

    assert stamped.child_run_id == "child-run"
    assert stamped.delegation_id == "delegation-1"


def test_event_schema_v1_migrates_to_a_v2_root_event() -> None:
    event = AgentEvent.from_dict(
        {
            "type": "run_started",
            "run_id": "legacy-run",
            "event_id": "legacy-event",
            "event_schema_version": 1,
            "occurred_at": "2026-08-10T00:00:00Z",
            "visibility": "public",
            "data": {},
        }
    )

    assert event.event_schema_version == 2
    assert event.root_run_id == "legacy-run"
    assert event.execution_group_id == "legacy-run"
    assert event.parent_run_id is None
    assert event.depth == 0
    assert event.sequence == 1
    assert AgentEvent.from_dict(event.to_dict()) == event


@pytest.mark.parametrize(
    "missing_field",
    ("event_id", "sequence", "execution_group_id", "root_run_id"),
)
def test_native_event_v2_rejects_missing_envelope_identity(
    missing_field: str,
) -> None:
    publisher = EventPublisher(
        run_id="run-v2",
        session_id="session-v2",
        engine_id="standard",
    )
    payload = publisher.create(EventType.RUN_STARTED).to_dict()
    payload.pop(missing_field)

    with pytest.raises(ValueError, match="missing envelope fields"):
        AgentEvent.from_dict(payload)


def test_native_event_v2_round_trips_without_identity_repair() -> None:
    publisher = EventPublisher(
        run_id="run-v2",
        session_id="session-v2",
        engine_id="standard",
    )
    original = publisher.create(EventType.RUN_STARTED)

    restored = AgentEvent.from_dict(original.to_dict())

    assert restored == original


def test_checkpoint_v4_is_copy_migrated_to_v5_as_a_root_run() -> None:
    payload = RunCheckpoint(
        run_id="legacy-run",
        session_id="legacy-session",
        messages=(Message.user("hello"),),
    ).to_dict()
    for key in (
        "run_lineage",
        "execution_group_id",
        "agent_ref",
        "agent_definition_fingerprint",
        "delegation_id",
        "parent_tool_call_id",
        "budget_lease_id",
    ):
        payload.pop(key, None)
    payload["schema_version"] = payload["version"] = 4
    source = copy.deepcopy(payload)

    snapshot = RunSnapshot.from_dict(payload)

    assert payload == source
    assert snapshot.schema_version == 5
    assert snapshot.run_lineage == {
        "root_run_id": "legacy-run",
        "parent_run_id": None,
        "depth": 0,
        "agent_path": [],
    }
    assert snapshot.execution_group_id == "legacy-run"
    assert snapshot.delegation_id is None
    assert snapshot.to_dict()["schema_version"] == 5


def test_production_does_not_adopt_a_legacy_unbound_definition() -> None:
    async def scenario() -> None:
        store = InMemoryCheckpointStore()
        payload = RunCheckpoint(
            run_id="legacy-run",
            session_id="legacy-session",
            messages=(),
        ).to_dict()
        for key in (
            "run_lineage",
            "execution_group_id",
            "agent_ref",
            "agent_definition_fingerprint",
            "delegation_id",
            "parent_tool_call_id",
            "budget_lease_id",
            "migrated_from_schema_version",
        ):
            payload.pop(key, None)
        payload["schema_version"] = payload["version"] = 4
        await store.save_snapshot(RunSnapshot.from_dict(payload))
        agent = Agent.create(
            model=_NoCallModel(),
            instructions="test",
            checkpoint_store=store,
        )
        agent.runtime.agent_definition = SimpleNamespace(
            fingerprint="sha256:" + "a" * 64,
            agent_id="agent",
            version="1.0.0",
        )
        agent.runtime.runtime_profile = SimpleNamespace(
            kind=SimpleNamespace(value="production")
        )

        with pytest.raises(StateMigrationError, match="exact AgentRef"):
            await agent.runtime._load_resume(
                RunRequest(
                    input="resume",
                    session_id="legacy-session",
                    resume_run_id="legacy-run",
                ),
                asyncio.get_running_loop().time() + 5,
                {},
            )

    asyncio.run(scenario())


def test_resume_rejects_a_store_that_returns_a_different_run() -> None:
    class SwappedCheckpointStore:
        async def load_snapshot(self, run_id):
            del run_id
            return RunCheckpoint(
                run_id="different-run",
                session_id="session-1",
                messages=(),
            ).to_snapshot()

        async def save_snapshot(self, snapshot):  # pragma: no cover - unused
            del snapshot

        async def delete(self, run_id):  # pragma: no cover - unused
            del run_id

    async def scenario() -> None:
        agent = Agent.create(
            model=_NoCallModel(),
            instructions="test",
            checkpoint_store=SwappedCheckpointStore(),
        )

        with pytest.raises(StateMigrationError, match="run_id does not match"):
            await agent.runtime._load_resume(
                RunRequest(
                    input="resume",
                    session_id="session-1",
                    resume_run_id="requested-run",
                ),
                asyncio.get_running_loop().time() + 5,
                {},
            )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "missing_field",
    (
        "run_lineage",
        "execution_group_id",
        "agent_ref",
        "agent_definition_fingerprint",
        "delegation_id",
        "parent_tool_call_id",
        "budget_lease_id",
        "migrated_from_schema_version",
    ),
)
def test_native_v5_checkpoint_requires_every_identity_field(
    missing_field: str,
) -> None:
    payload = RunCheckpoint(
        run_id="root-run",
        session_id="root-session",
        messages=(),
    ).to_dict()
    payload.pop(missing_field)

    with pytest.raises(ValueError, match="missing identity fields"):
        RunSnapshot.from_dict(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload["run_lineage"].update(
                {"root_run_id": "other-root"}
            ),
            "root_run_id must match run_id",
        ),
        (
            lambda payload: payload.update({"execution_group_id": "other-group"}),
            "execution_group_id must match",
        ),
        (
            lambda payload: payload.update({"delegation_id": "forged"}),
            "root snapshot cannot contain delegation_id",
        ),
        (
            lambda payload: payload.update({"budget_lease_id": "forged"}),
            "root snapshot cannot contain budget_lease_id",
        ),
    ),
)
def test_native_v5_root_identity_is_fail_closed(mutation, message: str) -> None:
    payload = RunCheckpoint(
        run_id="root-run",
        session_id="root-session",
        messages=(),
    ).to_dict()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        RunSnapshot.from_dict(payload)


def test_checkpoint_v5_round_trips_delegation_references() -> None:
    checkpoint = RunCheckpoint(
        run_id="child-run",
        session_id="child-session",
        messages=(),
        run_lineage={
            "root_run_id": "root-run",
            "parent_run_id": "parent-run",
            "depth": 1,
            "agent_path": ["supervisor@1.0.0", "researcher@2.0.0"],
        },
        execution_group_id="group-1",
        agent_ref={"agent_id": "researcher", "version": "2.0.0"},
        agent_definition_fingerprint="sha256:definition",
        delegation_id="delegation-1",
        parent_tool_call_id="call-1",
        budget_lease_id="lease-1",
    )

    restored = RunSnapshot.from_json(checkpoint.to_json())

    assert restored.run_lineage["parent_run_id"] == "parent-run"
    assert restored.execution_group_id == "group-1"
    assert restored.agent_ref == {
        "agent_id": "researcher",
        "version": "2.0.0",
    }
    assert restored.agent_definition_fingerprint == "sha256:definition"
    assert restored.delegation_id == "delegation-1"
    assert restored.parent_tool_call_id == "call-1"
    assert restored.budget_lease_id == "lease-1"


def test_native_v5_child_requires_a_budget_lease_identity() -> None:
    payload = RunCheckpoint(
        run_id="child-run",
        session_id="child-session",
        messages=(),
        run_lineage={
            "root_run_id": "root-run",
            "parent_run_id": "parent-run",
            "depth": 1,
            "agent_path": ["supervisor@1.0.0", "researcher@2.0.0"],
        },
        execution_group_id="group-1",
        agent_ref={"agent_id": "researcher", "version": "2.0.0"},
        agent_definition_fingerprint="sha256:definition",
        delegation_id="delegation-1",
        parent_tool_call_id="call-1",
        budget_lease_id="lease-1",
    ).to_dict()
    payload["budget_lease_id"] = None

    with pytest.raises(ValueError, match="budget_lease_id cannot be empty"):
        RunSnapshot.from_dict(payload)


def test_runtime_reserves_aggregate_model_budget_before_provider_io() -> None:
    async def scenario() -> None:
        ledger = _RecordingLedger()
        agent = Agent.create(
            model=_OneResponseModel(),
            instructions="test",
        )
        request = RunRequest(
            input="hello",
            session_id="child-session",
            budget_ledger=ledger,
            budget_lease=SimpleNamespace(
                lease_id="lease-1",
                execution_group_id="group-1",
            ),
        )

        result = await agent.runtime.execute(request)

        assert result.output == "done"
        assert ledger.model == [("group-1", 1)]
        assert ledger.tools == []

    asyncio.run(scenario())


def test_aggregate_model_budget_failure_has_a_stable_terminal_reason() -> None:
    async def scenario() -> None:
        model = _NoCallModel()
        agent = Agent.create(model=model, instructions="test")
        result = await agent.runtime.execute(
            RunRequest(
                input="hello",
                session_id="child-session",
                budget_ledger=_RejectingLedger("model"),
                budget_lease=SimpleNamespace(
                    lease_id="lease-1",
                    execution_group_id="group-1",
                ),
            )
        )

        assert result.finish_reason is FinishReason.MAX_MODEL_TURNS
        assert result.error_summary["category"] == "execution_group_budget"
        assert result.error_summary["code"] == ("execution_group_model_turns_exceeded")
        assert result.error_summary["resumable"] is False

    asyncio.run(scenario())


def test_aggregate_tool_budget_fails_before_the_tool_runs() -> None:
    async def scenario() -> None:
        calls = 0

        @function_tool
        def mutate_nothing() -> str:
            nonlocal calls
            calls += 1
            return "unexpected"

        model = _OneResponseModel()
        model.complete = lambda request: _tool_response(request)  # type: ignore[method-assign]
        agent = Agent.create(
            model=model,
            instructions="test",
            tools=(mutate_nothing,),
        )
        result = await agent.runtime.execute(
            RunRequest(
                input="hello",
                session_id="child-session",
                budget_ledger=_RejectingLedger("tool"),
                budget_lease=SimpleNamespace(
                    lease_id="lease-1",
                    execution_group_id="group-1",
                ),
            )
        )

        assert result.finish_reason is FinishReason.MAX_TOOL_CALLS
        assert result.error_summary["category"] == "execution_group_budget"
        assert result.error_summary["code"] == "execution_group_tool_calls_exceeded"
        assert calls == 0

    async def _tool_response(request):
        del request
        return ModelResponse(
            Message.assistant(""),
            tool_calls=(ToolCall("call-1", "mutate_nothing", {}),),
        )

    asyncio.run(scenario())
