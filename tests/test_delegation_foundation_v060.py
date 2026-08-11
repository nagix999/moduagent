from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, Field, RootModel

from moduagent.agent import Agent
from moduagent.config import RunLimits
from moduagent.definitions import (
    AgentDefinition,
    AgentEndpoint,
    AgentRef,
    AgentRegistry,
    DefinitionStatus,
    InMemoryAgentRegistry,
)
from moduagent.delegation import (
    DELEGATION_EVENT_CALLBACK_KEY,
    PARENT_DELEGATION_CONTEXT_KEY,
    BudgetExceeded,
    DelegatedAgentTool,
    DelegationContext,
    DelegationCoordinator,
    DelegationDecision,
    DelegationEvent,
    DelegationIdFactory,
    DelegationOutcome,
    DelegationOutcomeStatus,
    DelegationPolicy,
    DelegationReceipt,
    DelegationReceiptStatus,
    DurableBudgetLedger,
    EdgeDelegationAuthorizer,
    ExecutionGroupLimits,
    InMemoryBudgetLedger,
    InMemoryBudgetStateStore,
    InMemoryDelegationReceiptStore,
    LocalAgentInvoker,
    ParentDelegationContext,
    ReceiptManager,
    RunLineage,
    SessionKeyFactory,
    SessionStrategy,
    canonical_digest,
)
from moduagent.messages import Message
from moduagent.models import ModelResponse
from moduagent.persistence import InMemoryCheckpointStore
from moduagent.runtime import RunStatus
from moduagent.tools import ToolErrorType, ToolExecutionContext, ToolFailure


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str


class TaskAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str


class AliasTaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(alias="q")


class AliasTaskAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(alias="result")


class RootTaskRequest(RootModel[str]):
    pass


class RootTaskAnswer(RootModel[list[str]]):
    pass


class _EndpointHandler:
    def __init__(
        self,
        *,
        answer: str = "done",
        outcome_status: DelegationOutcomeStatus = DelegationOutcomeStatus.COMPLETED,
        error_code: str | None = None,
    ) -> None:
        self.answer = answer
        self.outcome_status = outcome_status
        self.error_code = error_code
        self.calls = 0
        self.received: tuple[Any, ...] | None = None

    async def _run_delegated(
        self,
        request: BaseModel,
        *,
        context,
        budget,
        budget_ledger,
        child_run_id: str,
    ) -> DelegationOutcome:
        self.calls += 1
        self.received = (request, context, budget, budget_ledger, child_run_id)
        if self.outcome_status is DelegationOutcomeStatus.COMPLETED:
            return DelegationOutcome(
                self.outcome_status,
                child_run_id,
                finish_reason="completed",
                output=TaskAnswer(answer=self.answer),
                usage={"input_tokens": 2, "output_tokens": 1},
            )
        return DelegationOutcome(
            self.outcome_status,
            child_run_id,
            finish_reason="error",
            error_code=self.error_code or "private_backend_failure",
            retryable=True,
            usage={"input_tokens": 2},
        )


class _AliasEndpointHandler:
    def __init__(self) -> None:
        self.calls = 0

    async def _run_delegated(
        self,
        request: BaseModel,
        *,
        context,
        budget,
        budget_ledger,
        child_run_id: str,
    ) -> DelegationOutcome:
        del context, budget, budget_ledger
        self.calls += 1
        assert isinstance(request, AliasTaskRequest)
        return DelegationOutcome(
            DelegationOutcomeStatus.COMPLETED,
            child_run_id,
            finish_reason="completed",
            output=AliasTaskAnswer(result=f"alias:{request.question}"),
        )


class _RemoteInvoker(LocalAgentInvoker):
    """Test adapter proving approval is enforced before an arbitrary invoker."""

    @staticmethod
    def _handler(endpoint: AgentEndpoint):
        return endpoint.handler


class _CountingAuthorizer:
    def __init__(self) -> None:
        self.calls = 0

    async def authorize(self, **kwargs):
        del kwargs
        self.calls += 1
        return DelegationDecision(True)


class _SplitViewRegistry:
    """Adversarial SPI returning different pinned views for one lookup."""

    def __init__(self, safe, resolved, *, descriptor=None) -> None:
        self.safe = safe
        self.resolved = resolved
        self.descriptor_override = descriptor

    def register(self, definition, endpoint, *, status=DefinitionStatus.DRAFT):
        return self.safe.register(definition, endpoint, status=status)

    def descriptor(self, ref):
        if self.descriptor_override is not None:
            return self.descriptor_override
        return self.safe.descriptor(ref)

    def resolve(self, ref):
        del ref
        return self.resolved


class _SwitchingRegistry:
    """Adversarial SPI that changes an exact ref after Tool composition."""

    def __init__(self, current: InMemoryAgentRegistry) -> None:
        self.current = current

    def register(self, definition, endpoint, *, status=DefinitionStatus.DRAFT):
        return self.current.register(definition, endpoint, status=status)

    def descriptor(self, ref):
        return self.current.descriptor(ref)

    def resolve(self, ref):
        return self.current.resolve(ref)


class _NoopCycleGuard:
    def __init__(self) -> None:
        self.calls = 0

    def validate(self, **kwargs) -> None:
        del kwargs
        self.calls += 1


class _LeakySessionKeyFactory(SessionKeyFactory):
    def create(self, *, parent_session_id, **kwargs):
        del kwargs
        return parent_session_id


class _SingleStructuredResponseModel:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request) -> ModelResponse:
        del request
        self.calls += 1
        return ModelResponse(
            Message.assistant('{"answer":"standard checkpoint result"}')
        )


class _CancellingEndpoint:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def _run_delegated(
        self,
        request: BaseModel,
        *,
        context,
        budget,
        budget_ledger,
        child_run_id: str,
    ) -> DelegationOutcome:
        del request, context, budget, budget_ledger, child_run_id
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class _SlowCancellationEndpoint:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.stopped = asyncio.Event()

    async def _run_delegated(
        self,
        request: BaseModel,
        *,
        context,
        budget,
        budget_ledger,
        child_run_id: str,
    ) -> DelegationOutcome:
        del request, context, budget, budget_ledger
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await asyncio.sleep(0.03)
            self.stopped.set()
            return DelegationOutcome(
                DelegationOutcomeStatus.COMPLETED,
                child_run_id,
                finish_reason="completed",
                output=TaskAnswer(answer="late"),
            )


class _BlockingSuccessEndpoint:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def _run_delegated(
        self,
        request: BaseModel,
        *,
        context,
        budget,
        budget_ledger,
        child_run_id: str,
    ) -> DelegationOutcome:
        del request, context, budget, budget_ledger
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return DelegationOutcome(
            DelegationOutcomeStatus.COMPLETED,
            child_run_id,
            finish_reason="completed",
            output=TaskAnswer(answer="single owner"),
        )


class _ResumableEndpoint:
    def __init__(self) -> None:
        self.run_calls = 0
        self.resume_calls = 0
        self.first_lease_id: str | None = None
        self.resume_lease_id: str | None = None
        self.resume_lease_was_active = False

    async def _run_delegated(
        self,
        request: BaseModel,
        *,
        context,
        budget,
        budget_ledger,
        child_run_id: str,
    ) -> DelegationOutcome:
        del request, context, budget_ledger
        self.run_calls += 1
        self.first_lease_id = budget.lease_id
        return DelegationOutcome(
            DelegationOutcomeStatus.FAILED,
            child_run_id,
            finish_reason="error",
            error_code="private_resumable_failure",
            resumable=True,
            usage={"input_tokens": 1},
        )

    async def _resume_delegated(
        self,
        request: BaseModel,
        *,
        context,
        budget,
        budget_ledger,
        child_run_id: str,
    ) -> DelegationOutcome:
        del request, context
        self.resume_calls += 1
        self.resume_lease_id = budget.lease_id
        state = await budget_ledger.snapshot(budget.execution_group_id)
        self.resume_lease_was_active = state.leases[budget.lease_id].status == "active"
        return DelegationOutcome(
            DelegationOutcomeStatus.COMPLETED,
            child_run_id,
            finish_reason="completed",
            output=TaskAnswer(answer="resumed"),
            usage={"output_tokens": 1},
        )


class _CompletionRaceEndpoint:
    def __init__(self) -> None:
        self.run_calls = 0
        self.reconcile_calls = 0

    async def _run_delegated(
        self,
        request: BaseModel,
        *,
        context,
        budget,
        budget_ledger,
        child_run_id: str,
    ) -> DelegationOutcome:
        del request, context, budget, budget_ledger
        self.run_calls += 1
        return DelegationOutcome(
            DelegationOutcomeStatus.COMPLETED,
            child_run_id,
            finish_reason="completed",
            output=TaskAnswer(answer="same terminal output"),
            usage={"input_tokens": 2, "output_tokens": 1},
        )

    async def _reconcile_delegated(
        self,
        *,
        context,
        budget,
        budget_ledger,
        child_run_id: str,
    ) -> DelegationOutcome:
        del context, budget, budget_ledger
        self.reconcile_calls += 1
        return DelegationOutcome(
            DelegationOutcomeStatus.COMPLETED,
            child_run_id,
            finish_reason="checkpoint_completed",
            output=TaskAnswer(answer="same terminal output"),
            usage={"input_tokens": 2, "output_tokens": 1},
        )


class _EventSink:
    def __init__(self) -> None:
        self.events: list[DelegationEvent] = []

    async def publish_delegation(self, event: DelegationEvent) -> None:
        self.events.append(event)


class _FailingEventSink(_EventSink):
    async def publish_delegation(self, event: DelegationEvent) -> None:
        del event
        raise RuntimeError("telemetry unavailable")


class _SimulatedProcessCrash(BaseException):
    pass


class _CrashBeforeReceiptStore(InMemoryDelegationReceiptStore):
    def __init__(self) -> None:
        super().__init__()
        self.crash_once = True

    async def claim(self, receipt: DelegationReceipt):
        if self.crash_once:
            self.crash_once = False
            raise _SimulatedProcessCrash
        return await super().claim(receipt)


class _CrashAfterReceiptClaimStore(InMemoryDelegationReceiptStore):
    def __init__(self) -> None:
        super().__init__()
        self.crash_once = True

    async def claim(self, receipt: DelegationReceipt):
        claimed = await super().claim(receipt)
        if self.crash_once:
            self.crash_once = False
            raise _SimulatedProcessCrash
        return claimed


class _CompletionRaceReceiptStore(InMemoryDelegationReceiptStore):
    def __init__(self) -> None:
        super().__init__()
        self.owner_at_terminal_cas = asyncio.Event()
        self.duplicate_stored_terminal = asyncio.Event()
        self.release_owner = asyncio.Event()
        self.release_duplicate = asyncio.Event()

    async def compare_and_swap(
        self,
        delegation_id: str,
        expected_revision: int,
        receipt: DelegationReceipt,
    ) -> bool:
        task = asyncio.current_task()
        task_name = task.get_name() if task is not None else ""
        is_completion = receipt.status is DelegationReceiptStatus.COMPLETED
        if is_completion and task_name == "delegation-owner":
            self.owner_at_terminal_cas.set()
            await self.release_owner.wait()
        if is_completion and task_name == "delegation-duplicate":
            changed = await super().compare_and_swap(
                delegation_id,
                expected_revision,
                receipt,
            )
            if changed:
                self.duplicate_stored_terminal.set()
                await self.release_duplicate.wait()
            return changed
        return await super().compare_and_swap(
            delegation_id,
            expected_revision,
            receipt,
        )


def _definition(ref: AgentRef, *, caller: AgentRef) -> AgentDefinition:
    return AgentDefinition(
        agent_id=ref.agent_id,
        version=ref.version,
        description=f"{ref.agent_id} delegated endpoint",
        instructions_ref=f"instructions/{ref.agent_id}",
        execution_profile="standard",
        model_route="default",
        tool_refs=(),
        skill_refs=(),
        input_contract_ref="contract/task-request",
        output_contract_ref="contract/task-answer",
        memory_policy_ref="memory/isolated",
        authorization_policy_ref="authorization/delegation",
        data_classification="internal",
        side_effect_level="none",
        approval_requirement="none",
        callable_by=frozenset({caller.agent_id}),
        limits=RunLimits(),
    )


async def _build_tool(
    handler: object,
    *,
    limits: ExecutionGroupLimits | None = None,
    allowed: bool = True,
    receipt_store: InMemoryDelegationReceiptStore | None = None,
    event_sink: _EventSink | None = None,
    budget_ledger: InMemoryBudgetLedger | None = None,
    cancellation_grace_seconds: float = 1.0,
    allow_resume: bool = False,
    cycle_guard: object | None = None,
    session_factory: SessionKeyFactory | None = None,
    max_result_bytes: int | None = None,
) -> tuple[
    DelegatedAgentTool,
    ParentDelegationContext,
    InMemoryBudgetLedger,
    InMemoryDelegationReceiptStore,
    AgentRef,
    AgentRef,
]:
    caller = AgentRef("parent-agent", "1.0.0")
    callee = AgentRef("child-agent", "1.0.0")
    registry = InMemoryAgentRegistry()
    registry.register(
        _definition(callee, caller=caller),
        AgentEndpoint(handler=handler, approved=True),
        status=DefinitionStatus.APPROVED,
    )
    ledger = (
        budget_ledger
        if budget_ledger is not None
        else InMemoryBudgetLedger(queue_poll_seconds=0.001)
    )
    receipts = receipt_store or InMemoryDelegationReceiptStore()
    coordinator = DelegationCoordinator(
        registry=registry,
        authorizer=EdgeDelegationAuthorizer({(caller, callee)} if allowed else set()),
        budget_ledger=ledger,
        receipt_store=receipts,
        id_factory=DelegationIdFactory(b"delegation-id-secret-for-tests-0001"),
        session_factory=(
            session_factory
            if session_factory is not None
            else SessionKeyFactory(b"session-key-secret-for-tests-00001")
        ),
        limits=limits,
        event_sink=event_sink,
        cancellation_grace_seconds=cancellation_grace_seconds,
        cycle_guard=cycle_guard,
    )
    tool = DelegatedAgentTool(
        coordinator=coordinator,
        caller=caller,
        callee=callee,
        input_model=TaskRequest,
        output_model=TaskAnswer,
        name="delegate_task",
        description="Delegate one typed task",
        allow_resume=allow_resume,
        max_result_bytes=max_result_bytes,
        expected_definition_fingerprint=(
            registry.descriptor(callee).definition_fingerprint
        ),
    )
    parent = ParentDelegationContext(
        lineage=RunLineage.root(run_id="root-run", agent=caller),
        execution_group_id="group-1",
        principal="principal-1",
        tenant="tenant-1",
        parent_session_id="parent-session",
        absolute_deadline=datetime.now(timezone.utc) + timedelta(seconds=10),
        limits=limits or ExecutionGroupLimits(),
    )
    return tool, parent, ledger, receipts, caller, callee


def _tool_context(
    parent: ParentDelegationContext,
    *,
    call_id: str = "call-1",
    run_id: str = "root-run",
):
    return ToolExecutionContext(
        run_id=run_id,
        session_id="parent-session",
        metadata={PARENT_DELEGATION_CONTEXT_KEY: parent},
        tool_call_id=call_id,
    )


def _receipt_context_digest(
    tool: DelegatedAgentTool,
    parent: ParentDelegationContext,
    caller: AgentRef,
    callee: AgentRef,
) -> str:
    return tool.coordinator.id_factory.context_digest(
        {
            "principal": parent.principal,
            "tenant": parent.tenant,
            "parent_session_id": parent.parent_session_id,
            "request_classification": "internal",
            "session_strategy": SessionStrategy.ISOLATED.value,
            "caller_agent_ref": str(caller),
            "callee_agent_ref": str(callee),
            "input_contract_digest": canonical_digest(TaskRequest.model_json_schema()),
            "output_contract_digest": canonical_digest(TaskAnswer.model_json_schema()),
            "max_result_bytes": tool.max_result_bytes,
        }
    )


def _tool_context_with_event_callback(
    parent: ParentDelegationContext,
    callback,
    *,
    call_id: str = "call-1",
):
    return ToolExecutionContext(
        run_id="root-run",
        session_id="parent-session",
        metadata={
            PARENT_DELEGATION_CONTEXT_KEY: parent,
            DELEGATION_EVENT_CALLBACK_KEY: callback,
        },
        tool_call_id=call_id,
    )


def test_delegation_exports_the_canonical_definition_types() -> None:
    from moduagent.delegation import AgentRef as DelegationAgentRef
    from moduagent.delegation import AgentRegistry as DelegationAgentRegistry

    assert DelegationAgentRef is AgentRef
    assert DelegationAgentRegistry is AgentRegistry


def test_lineage_accepts_semver_build_metadata_for_root_child_and_roundtrip() -> None:
    root_ref = AgentRef("parent-agent", "1.2.3+build.7")
    child_ref = AgentRef("child-agent", "2.0.0-rc.1+cuda.12")

    root = RunLineage.root(run_id="root-run", agent=root_ref)
    child = root.child(
        child=child_ref,
        parent_run_id="root-run",
        delegation_id="delegation-1",
        parent_tool_call_id="tool-call-1",
    )
    restored = RunLineage.from_dict(child.to_dict())

    assert root.agent_ref == root_ref
    assert root.agent_path == ("parent-agent@1.2.3+build.7",)
    assert child.agent_ref == child_ref
    assert child.agent_path == (
        "parent-agent@1.2.3+build.7",
        "child-agent@2.0.0-rc.1+cuda.12",
    )
    assert restored == child


def test_remote_endpoint_approval_is_rechecked_before_every_invocation() -> None:
    async def scenario() -> None:
        caller = AgentRef("parent-agent", "1.0.0")
        callee = AgentRef("child-agent", "1.0.0")
        definition = _definition(callee, caller=caller)
        handler = _EndpointHandler()
        registry = InMemoryAgentRegistry()
        registry.register(
            definition,
            AgentEndpoint(
                handler=handler,
                kind="remote",
                approved=False,
            ),
            status=DefinitionStatus.ACTIVE,
        )
        coordinator = DelegationCoordinator(
            registry=registry,
            authorizer=EdgeDelegationAuthorizer({(caller, callee)}),
            invoker=_RemoteInvoker(),
        )
        tool = coordinator.tool(
            caller=caller,
            callee=callee,
            input_model=TaskRequest,
            output_model=TaskAnswer,
        )
        parent = coordinator.parent_context(
            caller=caller,
            run_id="remote-root",
            session_id="parent-session",
            principal="principal-1",
            tenant="tenant-1",
        )

        with pytest.raises(ToolFailure) as unapproved:
            await tool.invoke(
                {"question": "first"},
                _tool_context(
                    parent,
                    call_id="remote-call-1",
                    run_id="remote-root",
                ),
            )
        assert unapproved.value.error.reason == (
            "delegation_remote_endpoint_not_approved"
        )
        assert handler.calls == 0

        registry.rebind_endpoint(
            callee,
            AgentEndpoint(handler=handler, kind="remote", approved=True),
            expected_fingerprint=definition.fingerprint,
        )
        approved = await tool.invoke(
            {"question": "second"},
            _tool_context(
                parent,
                call_id="remote-call-2",
                run_id="remote-root",
            ),
        )
        assert approved == TaskAnswer(answer="done")
        assert handler.calls == 1

        registry.rebind_endpoint(
            callee,
            AgentEndpoint(handler=handler, kind="remote", approved=False),
            expected_fingerprint=definition.fingerprint,
        )
        with pytest.raises(ToolFailure):
            await tool.invoke(
                {"question": "third"},
                _tool_context(
                    parent,
                    call_id="remote-call-3",
                    run_id="remote-root",
                ),
            )
        assert handler.calls == 1

    asyncio.run(scenario())


def test_custom_invoker_cannot_bypass_local_endpoint_approval() -> None:
    async def scenario() -> None:
        caller = AgentRef("parent-agent", "1.0.0")
        callee = AgentRef("local-child", "1.0.0")
        definition = _definition(callee, caller=caller)
        handler = _EndpointHandler()
        registry = InMemoryAgentRegistry()
        registry.register(
            definition,
            AgentEndpoint(handler=handler, kind="local", approved=False),
            status=DefinitionStatus.ACTIVE,
        )
        coordinator = DelegationCoordinator(
            registry=registry,
            authorizer=EdgeDelegationAuthorizer({(caller, callee)}),
            invoker=_RemoteInvoker(),
        )
        tool = coordinator.tool(
            caller=caller,
            callee=callee,
            input_model=TaskRequest,
            output_model=TaskAnswer,
        )
        parent = coordinator.parent_context(
            caller=caller,
            run_id="local-root",
            session_id="parent-session",
            principal="principal-1",
            tenant="tenant-1",
        )

        with pytest.raises(ToolFailure) as captured:
            await tool.invoke(
                {"question": "approval must be coordinator-owned"},
                _tool_context(
                    parent,
                    call_id="local-call",
                    run_id="local-root",
                ),
            )

        assert captured.value.error.reason == "delegation_endpoint_not_approved"
        assert handler.calls == 0

    asyncio.run(scenario())


def test_registry_split_views_are_rejected_before_authorization_or_invocation() -> None:
    async def scenario() -> None:
        caller = AgentRef("parent-agent", "1.0.0")
        requested = AgentRef("requested-child", "1.0.0")
        substituted = AgentRef("substituted-child", "1.0.0")
        requested_handler = _EndpointHandler(answer="requested")
        substituted_handler = _EndpointHandler(answer="substituted")
        backing = InMemoryAgentRegistry()
        backing.register(
            _definition(requested, caller=caller),
            AgentEndpoint(handler=requested_handler, approved=True),
            status=DefinitionStatus.ACTIVE,
        )
        substituted_definition = _definition(substituted, caller=caller)
        backing.register(
            substituted_definition,
            AgentEndpoint(handler=substituted_handler, approved=True),
            status=DefinitionStatus.ACTIVE,
        )
        registry = _SplitViewRegistry(
            backing,
            backing.resolve(substituted),
        )
        authorizer = _CountingAuthorizer()
        coordinator = DelegationCoordinator(
            registry=registry,
            authorizer=authorizer,
        )
        tool = coordinator.tool(
            caller=caller,
            callee=requested,
            input_model=TaskRequest,
            output_model=TaskAnswer,
        )
        parent = coordinator.parent_context(
            caller=caller,
            run_id="split-root",
            session_id="parent-session",
            principal="principal-1",
            tenant="tenant-1",
        )

        with pytest.raises(ToolFailure) as captured:
            await tool.invoke(
                {"question": "must stay pinned"},
                _tool_context(
                    parent,
                    call_id="split-call",
                    run_id="split-root",
                ),
            )

        assert captured.value.error.reason == "delegation_registry_integrity_failed"
        assert authorizer.calls == 0
        assert requested_handler.calls == 0
        assert substituted_handler.calls == 0

    asyncio.run(scenario())


def test_registry_cannot_weaken_descriptor_policy_or_contract_fields() -> None:
    async def scenario() -> None:
        caller = AgentRef("parent-agent", "1.0.0")
        callee = AgentRef("contract-child", "1.0.0")
        definition = replace(
            _definition(callee, caller=caller),
            semantic_digests={
                "input_contract": canonical_digest(TaskRequest.model_json_schema()),
                "output_contract": canonical_digest(TaskAnswer.model_json_schema()),
            },
        )
        handler = _EndpointHandler()
        backing = InMemoryAgentRegistry()
        backing.register(
            definition,
            AgentEndpoint(handler=handler, approved=True),
            status=DefinitionStatus.ACTIVE,
        )
        weak = replace(
            backing.descriptor(callee),
            input_contract_digest=None,
            output_contract_digest=None,
            data_classification="public",
            side_effect_level="advisory",
            callable_by=frozenset({"untrusted-caller"}),
        )
        registry = _SplitViewRegistry(
            backing,
            backing.resolve(callee),
            descriptor=weak,
        )
        authorizer = _CountingAuthorizer()
        coordinator = DelegationCoordinator(
            registry=registry,
            authorizer=authorizer,
        )
        tool = coordinator.tool(
            caller=caller,
            callee=callee,
            input_model=TaskRequest,
            output_model=TaskAnswer,
            description="fixed test description",
        )
        parent = coordinator.parent_context(
            caller=caller,
            run_id="weak-root",
            session_id="parent-session",
            principal="principal-1",
            tenant="tenant-1",
        )

        with pytest.raises(ToolFailure) as captured:
            await tool.invoke(
                {"question": "keep contracts pinned"},
                _tool_context(
                    parent,
                    call_id="weak-call",
                    run_id="weak-root",
                ),
            )

        assert captured.value.error.reason == "delegation_registry_integrity_failed"
        assert authorizer.calls == 0
        assert handler.calls == 0

    asyncio.run(scenario())


def test_delegated_tool_pins_definition_fingerprint_at_composition() -> None:
    async def scenario() -> None:
        caller = AgentRef("parent-agent", "1.0.0")
        callee = AgentRef("mutable-child", "1.0.0")
        safe_handler = _EndpointHandler(answer="safe")
        changed_handler = _EndpointHandler(answer="changed")
        safe_registry = InMemoryAgentRegistry()
        safe_registry.register(
            _definition(callee, caller=caller),
            AgentEndpoint(handler=safe_handler, approved=True),
            status=DefinitionStatus.ACTIVE,
        )
        changed_registry = InMemoryAgentRegistry()
        changed_registry.register(
            replace(
                _definition(callee, caller=caller),
                description="same exact ref, changed semantics",
                side_effect_level="write",
            ),
            AgentEndpoint(handler=changed_handler, approved=True),
            status=DefinitionStatus.ACTIVE,
        )
        registry = _SwitchingRegistry(safe_registry)
        authorizer = _CountingAuthorizer()
        coordinator = DelegationCoordinator(
            registry=registry,
            authorizer=authorizer,
        )
        tool = coordinator.tool(
            caller=caller,
            callee=callee,
            input_model=TaskRequest,
            output_model=TaskAnswer,
        )
        registry.current = changed_registry
        parent = coordinator.parent_context(
            caller=caller,
            run_id="drift-root",
            session_id="parent-session",
            principal="principal-1",
            tenant="tenant-1",
        )

        with pytest.raises(ToolFailure) as captured:
            await tool.invoke(
                {"question": "must remain pinned"},
                _tool_context(parent, call_id="drift-call", run_id="drift-root"),
            )

        assert captured.value.error.reason == "delegation_registry_integrity_failed"
        assert authorizer.calls == 0
        assert safe_handler.calls == 0
        assert changed_handler.calls == 0

    asyncio.run(scenario())


def test_coordinator_builds_root_and_child_runtime_parent_contexts() -> None:
    async def scenario() -> None:
        tool, _, _, _, caller, callee = await _build_tool(_EndpointHandler())
        root = tool.coordinator.parent_context(
            caller=caller,
            run_id="root-generated",
            session_id="root-session",
            principal="principal-1",
            tenant="tenant-1",
        )
        assert root.lineage == RunLineage.root(
            run_id="root-generated",
            agent=caller,
        )
        assert root.execution_group_id == "root-generated"
        incoming = tool.coordinator.session_factory.create(
            strategy=SessionStrategy.ISOLATED,
            tenant="tenant-1",
            parent_session_id="root-session",
            callee=callee,
            delegation_id="delegation-1",
        )
        child_context = DelegationContext(
            lineage=root.lineage.child(
                child=callee,
                parent_run_id="root-generated",
                delegation_id="delegation-1",
                parent_tool_call_id="call-1",
            ),
            execution_group_id=root.execution_group_id,
            principal=root.principal,
            tenant=root.tenant,
            absolute_deadline=root.absolute_deadline,
            child_session_id=incoming,
        )
        child_run_id = tool.coordinator.id_factory.child_run_id("delegation-1")
        child = tool.coordinator.parent_context(
            caller=callee,
            run_id=child_run_id,
            session_id=incoming,
            principal="principal-1",
            tenant="tenant-1",
            incoming=child_context,
        )
        assert child.lineage == child_context.lineage
        assert child.execution_group_id == root.execution_group_id
        assert child.absolute_deadline == root.absolute_deadline
        assert child.current_run_id == child_run_id

    asyncio.run(scenario())


def test_typed_tool_success_is_receipted_and_replayed_without_child_restart() -> None:
    async def scenario() -> None:
        handler = _EndpointHandler(answer="typed result")
        sink = _EventSink()
        tool, parent, ledger, receipts, _, _ = await _build_tool(
            handler,
            event_sink=sink,
        )

        first = await tool.invoke(
            {"question": "SENSITIVE-REQUEST"},
            _tool_context(parent),
        )
        replay = await tool.invoke(
            {"question": "SENSITIVE-REQUEST"},
            _tool_context(parent),
        )

        assert first == TaskAnswer(answer="typed result")
        assert replay == first
        assert handler.calls == 1
        assert tool.schema.parameters["additionalProperties"] is False
        properties = tool.schema.parameters["properties"]
        assert set(properties) == {"question"}
        assert not {
            "tenant",
            "principal",
            "session_id",
            "budget",
            "trace_id",
            "agent_version",
        } & set(properties)
        assert handler.received is not None
        _, child_context, _, child_ledger, child_run_id = handler.received
        assert child_context.lineage.depth == 1
        assert child_context.lineage.agent_path == (
            "parent-agent@1.0.0",
            "child-agent@1.0.0",
        )
        assert child_context.child_session_id.startswith("delegation:")
        assert "parent-session" not in child_context.child_session_id
        assert child_ledger is ledger
        assert type(child_context).from_dict(child_context.to_dict()) == child_context
        receipt = await receipts.get(child_context.lineage.delegation_id)
        assert receipt is not None
        assert receipt.status is DelegationReceiptStatus.COMPLETED
        assert receipt.child_run_id == child_run_id
        state = await ledger.snapshot("group-1")
        assert state.active_delegations == 0
        assert state.usage == {"input_tokens": 2, "output_tokens": 1}
        serialized_events = json.dumps([event.to_dict() for event in sink.events])
        assert "SENSITIVE-REQUEST" not in serialized_events
        assert "typed result" not in serialized_events

    asyncio.run(scenario())


def test_alias_only_input_and_output_contracts_survive_receipt_replay() -> None:
    async def scenario() -> None:
        caller = AgentRef("parent-agent", "1.0.0")
        callee = AgentRef("alias-child", "1.0.0")
        handler = _AliasEndpointHandler()
        registry = InMemoryAgentRegistry()
        registry.register(
            _definition(callee, caller=caller),
            AgentEndpoint(handler=handler, approved=True),
            status=DefinitionStatus.ACTIVE,
        )
        coordinator = DelegationCoordinator(
            registry=registry,
            authorizer=EdgeDelegationAuthorizer({(caller, callee)}),
        )
        tool = coordinator.tool(
            caller=caller,
            callee=callee,
            input_model=AliasTaskRequest,
            output_model=AliasTaskAnswer,
        )
        parent = coordinator.parent_context(
            caller=caller,
            run_id="alias-root",
            session_id="parent-session",
            principal="principal-1",
            tenant="tenant-1",
        )
        context = _tool_context(parent, call_id="alias-call", run_id="alias-root")

        first = await tool.invoke({"q": "hello"}, context)
        replay = await tool.invoke({"q": "hello"}, context)

        assert first.answer == "alias:hello"
        assert replay == first
        assert handler.calls == 1

    asyncio.run(scenario())


def test_delegated_result_limit_fails_before_receipt_storage_and_replays() -> None:
    async def scenario() -> None:
        secret_answer = "PRIVATE-한글-결과" * 40
        canonical_size = len(
            json.dumps(
                {"answer": secret_answer},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        handler = _EndpointHandler(answer=secret_answer)
        tool, parent, ledger, receipts, _, _ = await _build_tool(
            handler,
            max_result_bytes=canonical_size - 1,
        )
        context = _tool_context(parent)

        for _ in range(2):
            with pytest.raises(ToolFailure) as captured:
                await tool.invoke({"question": "same"}, context)
            error = captured.value.error
            assert error.type is ToolErrorType.RESULT_TOO_LARGE
            assert error.reason == "delegated_agent_result_too_large"
            assert error.details == {}
            assert error.retryable is False
            assert secret_answer not in json.dumps(
                error.to_dict(),
                ensure_ascii=False,
            )

        assert handler.calls == 1
        receipt = next(iter(receipts._receipts.values()))
        assert receipt.status is DelegationReceiptStatus.FAILED
        assert receipt.finish_reason == "result_too_large"
        assert receipt.error_code == "delegated_agent_result_too_large"
        assert receipt.result_payload is None
        assert receipt.result_digest is None
        assert secret_answer not in repr(receipt)
        state = await ledger.snapshot("group-1")
        assert state.active_delegations == 0
        assert state.leases[receipt.lease_id].status == "completed"

    asyncio.run(scenario())


def test_delegated_result_limit_accepts_exact_canonical_utf8_size() -> None:
    async def scenario() -> None:
        answer = "한글-🛡️"
        canonical_size = len(
            json.dumps(
                {"answer": answer},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        tool, parent, _, receipts, _, _ = await _build_tool(
            _EndpointHandler(answer=answer),
            max_result_bytes=canonical_size,
        )

        assert await tool.invoke(
            {"question": "same"},
            _tool_context(parent),
        ) == TaskAnswer(answer=answer)
        receipt = next(iter(receipts._receipts.values()))
        assert receipt.status is DelegationReceiptStatus.COMPLETED
        assert receipt.result_payload == {"answer": answer}

    asyncio.run(scenario())


def test_receipt_replay_is_bound_to_the_trusted_security_scope() -> None:
    async def scenario() -> None:
        handler = _EndpointHandler(answer="tenant scoped")
        tool, parent, _, _, _, _ = await _build_tool(handler)
        context = _tool_context(parent)
        assert await tool.invoke({"question": "same"}, context) == TaskAnswer(
            answer="tenant scoped"
        )
        changed_principal = ParentDelegationContext(
            lineage=parent.lineage,
            execution_group_id=parent.execution_group_id,
            principal="principal-2",
            tenant=parent.tenant,
            parent_session_id=parent.parent_session_id,
            absolute_deadline=parent.absolute_deadline,
            limits=parent.limits,
            current_run_id=parent.current_run_id,
        )
        with pytest.raises(ToolFailure) as captured:
            await tool.invoke(
                {"question": "same"},
                _tool_context(changed_principal),
            )
        assert captured.value.error.reason == "delegation_receipt_identity_mismatch"
        assert captured.value.error.details == {}
        assert handler.calls == 1

    asyncio.run(scenario())


def test_public_coordinator_tool_api_uses_descriptor_safe_defaults() -> None:
    async def scenario() -> None:
        caller = AgentRef("supervisor", "3.0.0")
        callee = AgentRef("researcher", "2.1.0")
        registry = InMemoryAgentRegistry()
        registry.register(
            _definition(callee, caller=caller),
            AgentEndpoint(handler=_EndpointHandler(), approved=True),
            status=DefinitionStatus.APPROVED,
        )
        coordinator = DelegationCoordinator(
            registry=registry,
            policy=DelegationPolicy(allowed_edges={"supervisor": {"researcher"}}),
            receipt_store=InMemoryDelegationReceiptStore(),
            execution_group_store=InMemoryBudgetStateStore(),
        )
        tool = coordinator.tool(
            caller=caller,
            callee=callee,
            input_model=TaskRequest,
            output_model=TaskAnswer,
        )
        assert tool.name == "delegate_to_researcher"
        assert tool.description == "researcher delegated endpoint"
        assert tool.request_classification == "internal"
        assert tool.coordinator is coordinator
        assert tool.max_result_bytes == 1_000_000

    asyncio.run(scenario())


def test_delegated_tool_rejects_root_models_and_non_integer_result_limits() -> None:
    async def scenario() -> None:
        tool, _, _, _, caller, callee = await _build_tool(_EndpointHandler())
        coordinator = tool.coordinator

        with pytest.raises(TypeError, match="object-root"):
            coordinator.tool(
                caller=caller,
                callee=callee,
                input_model=RootTaskRequest,
                output_model=TaskAnswer,
            )
        with pytest.raises(TypeError, match="object-root"):
            DelegatedAgentTool(
                coordinator=coordinator,
                caller=caller,
                callee=callee,
                input_model=TaskRequest,
                output_model=RootTaskAnswer,
                name="invalid_root_output",
                description="Must reject a scalar-root output contract",
                expected_definition_fingerprint=(
                    coordinator.registry.descriptor(callee).definition_fingerprint
                ),
            )
        with pytest.raises(TypeError, match="positive integer"):
            coordinator.tool(
                caller=caller,
                callee=callee,
                input_model=TaskRequest,
                output_model=TaskAnswer,
                max_result_bytes=True,
            )

    asyncio.run(scenario())


def test_runtime_event_callback_is_isolated_from_sink_and_callback_failures() -> None:
    async def scenario() -> None:
        observed: list[DelegationEvent] = []

        async def callback(event: DelegationEvent) -> None:
            observed.append(event)
            raise RuntimeError("core stream unavailable")

        tool, parent, _, _, _, _ = await _build_tool(
            _EndpointHandler(answer="still succeeds"),
            event_sink=_FailingEventSink(),
        )
        result = await tool.invoke(
            {"question": "safe"},
            _tool_context_with_event_callback(parent, callback),
        )
        assert result == TaskAnswer(answer="still succeeds")
        assert [event.type.value for event in observed] == [
            "delegation_requested",
            "delegation_authorized",
            "delegation_started",
            "delegation_completed",
        ]

    asyncio.run(scenario())


def test_success_and_replay_settle_each_reserved_lease_exactly_once() -> None:
    class StrictTerminalLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(queue_poll_seconds=0.001)
            self.terminal_lease_ids: set[str] = set()
            self.completions = 0
            self.releases = 0

        async def complete_lease(self, lease, *, usage=None) -> None:
            if lease.lease_id in self.terminal_lease_ids:
                raise AssertionError("lease was settled more than once")
            self.terminal_lease_ids.add(lease.lease_id)
            self.completions += 1
            await super().complete_lease(lease, usage=usage)

        async def release_lease(self, lease) -> None:
            if lease.lease_id in self.terminal_lease_ids:
                raise AssertionError("lease was settled more than once")
            self.terminal_lease_ids.add(lease.lease_id)
            self.releases += 1
            await super().release_lease(lease)

    async def scenario() -> None:
        ledger = StrictTerminalLedger()
        handler = _EndpointHandler(answer="once")
        tool, parent, _, _, _, _ = await _build_tool(
            handler,
            budget_ledger=ledger,
        )
        context = _tool_context(parent)
        assert await tool.invoke({"question": "same"}, context) == TaskAnswer(
            answer="once"
        )
        assert await tool.invoke({"question": "same"}, context) == TaskAnswer(
            answer="once"
        )
        assert handler.calls == 1
        assert ledger.completions == 1
        assert ledger.releases == 0

    asyncio.run(scenario())


def test_completed_receipt_replay_reconciles_a_failed_ledger_settlement() -> None:
    class FailFirstCompletionLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(queue_poll_seconds=0.001)
            self.complete_attempts = 0
            self.successful_reconciliations = 0
            self.releases = 0

        async def complete_lease(self, lease, *, usage=None) -> None:
            self.complete_attempts += 1
            if self.complete_attempts == 1:
                raise BudgetExceeded("budget_store_unavailable")
            await super().complete_lease(lease, usage=usage)

        async def reconcile_completed_lease(self, lease, *, usage=None) -> bool:
            changed = await super().reconcile_completed_lease(lease, usage=usage)
            self.successful_reconciliations += int(changed)
            return changed

        async def release_lease(self, lease) -> None:
            self.releases += 1
            await super().release_lease(lease)

    async def scenario() -> None:
        ledger = FailFirstCompletionLedger()
        handler = _EndpointHandler(answer="durably replayed")
        tool, parent, _, receipts, _, _ = await _build_tool(
            handler,
            budget_ledger=ledger,
        )
        context = _tool_context(parent)

        with pytest.raises(ToolFailure) as first:
            await tool.invoke({"question": "same"}, context)
        assert first.value.error.reason == "budget_store_unavailable"
        receipt = next(iter(receipts._receipts.values()))
        assert receipt.status is DelegationReceiptStatus.COMPLETED
        unsettled = await ledger.snapshot("group-1")
        assert unsettled.leases[receipt.lease_id].status == "active"
        assert ledger.releases == 0

        assert await tool.invoke({"question": "same"}, context) == TaskAnswer(
            answer="durably replayed"
        )
        settled = await ledger.snapshot("group-1")
        assert settled.leases[receipt.lease_id].status == "completed"
        assert handler.calls == 1
        assert ledger.complete_attempts == 1
        assert ledger.successful_reconciliations == 1
        assert ledger.releases == 0

    asyncio.run(scenario())


def test_identical_terminal_reconcile_race_joins_without_failing_owner() -> None:
    async def scenario() -> None:
        handler = _CompletionRaceEndpoint()
        receipts = _CompletionRaceReceiptStore()
        first_tool, parent, ledger, _, caller, callee = await _build_tool(
            handler,
            receipt_store=receipts,
        )
        second_coordinator = DelegationCoordinator(
            registry=first_tool.coordinator.registry,
            policy=EdgeDelegationAuthorizer({(caller, callee)}),
            budget_ledger=ledger,
            receipt_store=receipts,
            id_factory=DelegationIdFactory(b"delegation-id-secret-for-tests-0001"),
            session_factory=SessionKeyFactory(b"session-key-secret-for-tests-00001"),
            limits=parent.limits,
        )
        second_tool = second_coordinator.tool(
            caller=caller,
            callee=callee,
            input_model=TaskRequest,
            output_model=TaskAnswer,
            name="delegate_task",
        )
        context = _tool_context(parent)
        owner = asyncio.create_task(
            first_tool.invoke({"question": "same"}, context),
            name="delegation-owner",
        )
        await asyncio.wait_for(receipts.owner_at_terminal_cas.wait(), timeout=1)
        duplicate = asyncio.create_task(
            second_tool.invoke({"question": "same"}, context),
            name="delegation-duplicate",
        )
        await asyncio.wait_for(
            receipts.duplicate_stored_terminal.wait(),
            timeout=1,
        )

        receipts.release_owner.set()
        assert await asyncio.wait_for(owner, timeout=1) == TaskAnswer(
            answer="same terminal output"
        )
        receipts.release_duplicate.set()
        assert await asyncio.wait_for(duplicate, timeout=1) == TaskAnswer(
            answer="same terminal output"
        )
        assert handler.run_calls == 1
        assert handler.reconcile_calls == 1
        final_receipt = next(iter(receipts._receipts.values()))
        assert final_receipt.status is DelegationReceiptStatus.COMPLETED
        assert final_receipt.finish_reason == "checkpoint_completed"
        state = await ledger.snapshot("group-1")
        assert state.active_delegations == 0
        assert state.delegation_count == 1
        assert state.usage == {
            "input_tokens": 2,
            "output_tokens": 1,
        }

    asyncio.run(scenario())


def test_standard_child_checkpoint_survives_receipt_crash_window_then_is_gc() -> None:
    async def scenario() -> None:
        caller = AgentRef("parent-agent", "1.0.0")
        callee = AgentRef("child-agent", "1.0.0")
        definition = _definition(callee, caller=caller)
        checkpoints = InMemoryCheckpointStore()
        model = _SingleStructuredResponseModel()
        child = Agent.create(
            model=model,
            name=callee.agent_id,
            instructions="Return the typed answer.",
            output=TaskAnswer,
            checkpoint_store=checkpoints,
            definition=definition,
        )
        registry = InMemoryAgentRegistry()
        registry.register(
            definition,
            AgentEndpoint(handler=child, approved=True),
            status=DefinitionStatus.APPROVED,
        )
        limits = ExecutionGroupLimits(timeout_seconds=10)
        ledger = InMemoryBudgetLedger(queue_poll_seconds=0.001)
        receipts = _CompletionRaceReceiptStore()

        def coordinator() -> DelegationCoordinator:
            return DelegationCoordinator(
                registry=registry,
                policy=EdgeDelegationAuthorizer({(caller, callee)}),
                budget_ledger=ledger,
                receipt_store=receipts,
                id_factory=DelegationIdFactory(b"delegation-id-secret-for-tests-0001"),
                session_factory=SessionKeyFactory(
                    b"session-key-secret-for-tests-00001"
                ),
                limits=limits,
            )

        first_tool = coordinator().tool(
            caller=caller,
            callee=callee,
            input_model=TaskRequest,
            output_model=TaskAnswer,
            name="delegate_task",
        )
        second_tool = coordinator().tool(
            caller=caller,
            callee=callee,
            input_model=TaskRequest,
            output_model=TaskAnswer,
            name="delegate_task",
        )
        parent = ParentDelegationContext(
            lineage=RunLineage.root(run_id="root-run", agent=caller),
            execution_group_id="group-1",
            principal="principal-1",
            tenant="tenant-1",
            parent_session_id="parent-session",
            absolute_deadline=datetime.now(timezone.utc) + timedelta(seconds=10),
            limits=limits,
        )
        context = _tool_context(parent)
        owner = asyncio.create_task(
            first_tool.invoke({"question": "same"}, context),
            name="delegation-owner",
        )
        await asyncio.wait_for(receipts.owner_at_terminal_cas.wait(), timeout=2)
        running = next(iter(receipts._receipts.values()))
        assert running.status is DelegationReceiptStatus.RUNNING
        retained = await checkpoints.load(running.child_run_id)
        assert retained is not None
        assert retained.status is RunStatus.COMPLETED

        duplicate = asyncio.create_task(
            second_tool.invoke({"question": "same"}, context),
            name="delegation-duplicate",
        )
        await asyncio.wait_for(
            receipts.duplicate_stored_terminal.wait(),
            timeout=2,
        )
        receipts.release_duplicate.set()
        assert await asyncio.wait_for(duplicate, timeout=2) == TaskAnswer(
            answer="standard checkpoint result"
        )
        assert await checkpoints.load(running.child_run_id) is None

        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert model.calls == 1
        stored = await receipts.get(running.delegation_id)
        assert stored is not None
        assert stored.status is DelegationReceiptStatus.COMPLETED
        state = await ledger.snapshot("group-1")
        assert state.active_delegations == 0
        assert state.delegation_count == 1

    asyncio.run(scenario())


def test_crash_during_receipt_claim_cannot_orphan_a_budget_lease() -> None:
    async def scenario() -> None:
        handler = _EndpointHandler(answer="recovered")
        receipts = _CrashBeforeReceiptStore()
        limits = ExecutionGroupLimits(
            max_depth=2,
            max_delegations=2,
            max_parallel_delegations=1,
            max_delegations_per_agent=2,
            max_total_model_turns=4,
            max_total_tool_calls=4,
            timeout_seconds=10,
        )
        tool, parent, ledger, _, _, _ = await _build_tool(
            handler,
            limits=limits,
            receipt_store=receipts,
        )
        context = _tool_context(parent)
        with pytest.raises(_SimulatedProcessCrash):
            await tool.invoke({"question": "same"}, context)
        crashed = await ledger.snapshot("group-1")
        assert crashed.active_delegations == 0
        assert crashed.delegation_count == 0
        assert len(crashed.leases) == 0

        assert await tool.invoke({"question": "same"}, context) == TaskAnswer(
            answer="recovered"
        )
        recovered = await ledger.snapshot("group-1")
        assert recovered.active_delegations == 0
        assert recovered.delegation_count == 1
        assert len(recovered.leases) == 1
        assert handler.calls == 1
        stored = tuple(receipts._receipts.values())
        assert len(stored) == 1
        assert stored[0].lease_id == next(iter(recovered.leases))

    asyncio.run(scenario())


def test_two_coordinators_fence_one_receipt_owner_before_child_invocation() -> None:
    async def scenario() -> None:
        handler = _BlockingSuccessEndpoint()
        limits = ExecutionGroupLimits(
            max_depth=2,
            max_delegations=2,
            max_parallel_delegations=1,
            max_delegations_per_agent=2,
            max_total_model_turns=4,
            max_total_tool_calls=4,
            timeout_seconds=10,
        )
        first_tool, parent, ledger, receipts, caller, callee = await _build_tool(
            handler,
            limits=limits,
        )
        second_coordinator = DelegationCoordinator(
            registry=first_tool.coordinator.registry,
            policy=EdgeDelegationAuthorizer({(caller, callee)}),
            budget_ledger=ledger,
            receipt_store=receipts,
            id_factory=DelegationIdFactory(b"delegation-id-secret-for-tests-0001"),
            session_factory=SessionKeyFactory(b"session-key-secret-for-tests-00001"),
            limits=limits,
        )
        second_tool = second_coordinator.tool(
            caller=caller,
            callee=callee,
            input_model=TaskRequest,
            output_model=TaskAnswer,
            name="delegate_task",
        )
        context = _tool_context(parent)

        owner = asyncio.create_task(first_tool.invoke({"question": "same"}, context))
        await asyncio.wait_for(handler.started.wait(), timeout=1)
        with pytest.raises(ToolFailure) as captured:
            await second_tool.invoke({"question": "same"}, context)
        assert captured.value.error.reason == "delegation_in_progress"
        assert handler.calls == 1
        running_state = await ledger.snapshot("group-1")
        assert running_state.delegation_count == 1
        assert running_state.active_delegations == 1
        receipt = next(iter(receipts._receipts.values()))
        assert receipt.status is DelegationReceiptStatus.RUNNING
        assert receipt.owner_token is not None

        handler.release.set()
        assert await asyncio.wait_for(owner, timeout=1) == TaskAnswer(
            answer="single owner"
        )
        complete_state = await ledger.snapshot("group-1")
        assert complete_state.delegation_count == 1
        assert complete_state.active_delegations == 0
        assert handler.calls == 1

    asyncio.run(scenario())


def test_resumable_failure_claims_a_new_active_attempt_lease() -> None:
    async def scenario() -> None:
        handler = _ResumableEndpoint()
        tool, parent, ledger, receipts, _, _ = await _build_tool(
            handler,
            allow_resume=True,
        )
        context = _tool_context(parent)

        with pytest.raises(ToolFailure) as captured:
            await tool.invoke({"question": "same"}, context)
        assert captured.value.error.reason == "delegated_agent_failed"
        failed = next(iter(receipts._receipts.values()))
        assert failed.status is DelegationReceiptStatus.FAILED
        assert failed.attempt == 1
        first_lease_id = failed.lease_id
        assert first_lease_id is not None
        first_state = await ledger.snapshot("group-1")
        assert first_state.leases[first_lease_id].status == "completed"

        assert await tool.invoke({"question": "same"}, context) == TaskAnswer(
            answer="resumed"
        )
        completed = await receipts.get(failed.delegation_id)
        assert completed is not None
        assert completed.status is DelegationReceiptStatus.COMPLETED
        assert completed.attempt == 2
        assert completed.lease_id != first_lease_id
        assert handler.run_calls == 1
        assert handler.resume_calls == 1
        assert handler.first_lease_id == first_lease_id
        assert handler.resume_lease_id == completed.lease_id
        assert handler.resume_lease_was_active is True
        final_state = await ledger.snapshot("group-1")
        assert final_state.delegation_count == 2
        assert final_state.active_delegations == 0
        assert all(lease.status == "completed" for lease in final_state.leases.values())
        assert final_state.usage == {"input_tokens": 1, "output_tokens": 1}

    asyncio.run(scenario())


def test_crash_after_receipt_claim_fails_closed_without_an_orphan_lease() -> None:
    async def scenario() -> None:
        handler = _EndpointHandler(answer="must not run")
        receipts = _CrashAfterReceiptClaimStore()
        tool, parent, ledger, _, _, _ = await _build_tool(
            handler,
            receipt_store=receipts,
        )
        parent = ParentDelegationContext(
            lineage=parent.lineage,
            execution_group_id=parent.execution_group_id,
            principal=parent.principal,
            tenant=parent.tenant,
            parent_session_id=parent.parent_session_id,
            absolute_deadline=datetime.now(timezone.utc) + timedelta(seconds=0.2),
            limits=parent.limits,
            current_run_id=parent.current_run_id,
        )
        context = _tool_context(parent)

        with pytest.raises(_SimulatedProcessCrash):
            await tool.invoke({"question": "same"}, context)
        crashed_state = await ledger.snapshot("group-1")
        assert crashed_state.delegation_count == 0
        assert crashed_state.active_delegations == 0
        receipt = next(iter(receipts._receipts.values()))
        assert receipt.status is DelegationReceiptStatus.RESERVED
        assert receipt.owner_token is not None
        assert receipt.lease_id not in crashed_state.leases

        with pytest.raises(ToolFailure) as in_progress:
            await tool.invoke({"question": "same"}, context)
        assert in_progress.value.error.reason == "delegation_in_progress"
        assert handler.calls == 0

        await asyncio.sleep(0.21)
        with pytest.raises(ToolFailure) as reconciliation:
            await tool.invoke({"question": "same"}, context)
        assert reconciliation.value.error.reason == "delegation_reconciliation_required"
        final_receipt = await receipts.get(receipt.delegation_id)
        assert final_receipt is not None
        assert final_receipt.status is DelegationReceiptStatus.MANUAL_REQUIRED
        assert handler.calls == 0
        final_state = await ledger.snapshot("group-1")
        assert final_state.delegation_count == 0
        assert final_state.active_delegations == 0

    asyncio.run(scenario())


def test_reserved_active_lease_without_child_checkpoint_expires_to_manual() -> None:
    async def scenario() -> None:
        handler = _EndpointHandler(answer="must not restart")
        receipts = InMemoryDelegationReceiptStore()
        tool, original_parent, ledger, _, caller, callee = await _build_tool(
            handler,
            receipt_store=receipts,
        )
        deadline = datetime.now(timezone.utc) + timedelta(seconds=0.1)
        parent = ParentDelegationContext(
            lineage=original_parent.lineage,
            execution_group_id=original_parent.execution_group_id,
            principal=original_parent.principal,
            tenant=original_parent.tenant,
            parent_session_id=original_parent.parent_session_id,
            absolute_deadline=deadline,
            limits=original_parent.limits,
        )
        ids = tool.coordinator.id_factory
        request = TaskRequest(question="same")
        digest = ids.request_digest(request.model_dump(mode="json"))
        delegation_id = ids.delegation_id(
            parent_run_id="root-run",
            parent_tool_call_id="call-1",
            request_digest=digest,
        )
        receipt = DelegationReceipt(
            delegation_id=delegation_id,
            execution_group_id="group-1",
            root_run_id="root-run",
            parent_run_id="root-run",
            parent_tool_call_id="call-1",
            caller_agent_ref=caller,
            callee_agent_ref=callee,
            child_run_id=ids.child_run_id(delegation_id),
            request_digest=digest,
            context_digest=_receipt_context_digest(
                tool,
                parent,
                caller,
                callee,
            ),
            owner_token="owner:crashed-after-reserve",
            lease_id=ids.lease_id(delegation_id),
        )
        await receipts.claim(receipt)
        await ledger.ensure_group(
            "group-1",
            parent.limits,
            absolute_deadline=deadline,
        )
        await ledger.reserve_delegation(
            "group-1",
            callee,
            lease_id=receipt.lease_id,
        )
        await asyncio.sleep(0.11)

        with pytest.raises(ToolFailure) as captured:
            await tool.invoke({"question": "same"}, _tool_context(parent))
        assert captured.value.error.reason == "delegation_reconciliation_required"
        assert handler.calls == 0
        stored = await receipts.get(delegation_id)
        assert stored is not None
        assert stored.status is DelegationReceiptStatus.MANUAL_REQUIRED
        state = await ledger.snapshot("group-1")
        assert state.active_delegations == 1
        assert state.delegation_count == 1

    asyncio.run(scenario())


def test_nested_coordinator_trusts_runtime_run_id_not_its_own_id_factory() -> None:
    async def scenario() -> None:
        handler = _EndpointHandler(answer="nested")
        tool, _, _, _, caller, _ = await _build_tool(handler)
        upstream_factory = DelegationIdFactory(
            b"different-upstream-id-secret-tests-0001"
        )
        request_digest = upstream_factory.request_digest({"task": "upstream"})
        upstream_delegation_id = upstream_factory.delegation_id(
            parent_run_id="upstream-root",
            parent_tool_call_id="upstream-call",
            request_digest=request_digest,
        )
        upstream_child_run_id = upstream_factory.child_run_id(upstream_delegation_id)
        assert upstream_child_run_id != tool.coordinator.id_factory.child_run_id(
            upstream_delegation_id
        )
        grandparent = AgentRef("grandparent-agent", "1.0.0")
        incoming = DelegationContext(
            lineage=RunLineage.root(
                run_id="upstream-root",
                agent=grandparent,
            ).child(
                child=caller,
                parent_run_id="upstream-root",
                delegation_id=upstream_delegation_id,
                parent_tool_call_id="upstream-call",
            ),
            execution_group_id="nested-group",
            principal="principal-1",
            tenant="tenant-1",
            absolute_deadline=datetime.now(timezone.utc) + timedelta(seconds=10),
            child_session_id="nested-session",
        )
        parent = tool.coordinator.parent_context(
            caller=caller,
            run_id=upstream_child_run_id,
            session_id="nested-session",
            principal="principal-1",
            tenant="tenant-1",
            incoming=incoming,
        )
        assert parent.current_run_id == upstream_child_run_id
        context = ToolExecutionContext(
            run_id=upstream_child_run_id,
            session_id="nested-session",
            metadata={PARENT_DELEGATION_CONTEXT_KEY: parent},
            tool_call_id="downstream-call",
        )
        assert await tool.invoke({"question": "nested"}, context) == TaskAnswer(
            answer="nested"
        )
        assert handler.calls == 1

    asyncio.run(scenario())


def test_existing_receipt_must_match_deterministic_lease_identity() -> None:
    async def scenario() -> None:
        handler = _EndpointHandler()
        receipts = InMemoryDelegationReceiptStore()
        tool, parent, ledger, _, caller, callee = await _build_tool(
            handler,
            receipt_store=receipts,
        )
        ids = tool.coordinator.id_factory
        request = TaskRequest(question="same")
        request_digest = ids.request_digest(request.model_dump(mode="json"))
        delegation_id = ids.delegation_id(
            parent_run_id="root-run",
            parent_tool_call_id="call-1",
            request_digest=request_digest,
        )
        await receipts.claim(
            DelegationReceipt(
                delegation_id=delegation_id,
                execution_group_id="group-1",
                root_run_id="root-run",
                parent_run_id="root-run",
                parent_tool_call_id="call-1",
                caller_agent_ref=caller,
                callee_agent_ref=callee,
                child_run_id=ids.child_run_id(delegation_id),
                request_digest=request_digest,
                context_digest=_receipt_context_digest(
                    tool,
                    parent,
                    caller,
                    callee,
                ),
                owner_token="owner:forged",
                lease_id="lease:forged",
            )
        )

        with pytest.raises(ToolFailure) as captured:
            await tool.invoke({"question": "same"}, _tool_context(parent))
        assert captured.value.error.reason == "delegation_receipt_identity_mismatch"
        assert handler.calls == 0
        assert (await ledger.snapshot("group-1")).delegation_count == 0

    asyncio.run(scenario())


def test_durable_coordinator_requires_stable_identity_factories() -> None:
    class DurableTestStore(InMemoryBudgetStateStore):
        durable = True

    caller = AgentRef("supervisor", "3.0.0")
    callee = AgentRef("researcher", "2.1.0")
    registry = InMemoryAgentRegistry()
    registry.register(
        _definition(callee, caller=caller),
        AgentEndpoint(handler=_EndpointHandler(), approved=True),
        status=DefinitionStatus.APPROVED,
    )
    with pytest.raises(ValueError, match="stable hmac_secret"):
        DelegationCoordinator(
            registry=registry,
            policy=DelegationPolicy(allowed_edges={"supervisor": {"researcher"}}),
            execution_group_store=DurableTestStore(),
        )


@pytest.mark.parametrize(
    ("mode", "expected_reason"),
    [
        ("denied", "delegation_edge_denied"),
        ("cycle", "delegation_cycle_detected"),
        ("depth", "delegation_depth_exceeded"),
    ],
)
def test_topology_guards_reject_before_budget_or_child_invocation(
    mode: str,
    expected_reason: str,
) -> None:
    async def scenario() -> None:
        handler = _EndpointHandler()
        limits = ExecutionGroupLimits(max_depth=1)
        tool, parent, ledger, _, caller, callee = await _build_tool(
            handler,
            limits=limits,
            allowed=mode != "denied",
        )
        if mode == "cycle":
            parent = ParentDelegationContext(
                lineage=RunLineage(
                    root_run_id="root-run",
                    parent_run_id="previous-run",
                    delegation_id="previous-delegation",
                    parent_tool_call_id="previous-call",
                    caller_agent_id=callee.agent_id,
                    agent_id=caller.agent_id,
                    agent_version=caller.version,
                    agent_path=(str(callee), str(caller)),
                    depth=1,
                ),
                execution_group_id="group-1",
                principal="principal-1",
                tenant="tenant-1",
                parent_session_id="parent-session",
                absolute_deadline=parent.absolute_deadline,
                limits=limits,
                current_run_id=tool.coordinator.id_factory.child_run_id(
                    "previous-delegation"
                ),
            )
        elif mode == "depth":
            parent = ParentDelegationContext(
                lineage=RunLineage(
                    root_run_id="root-run",
                    parent_run_id="previous-run",
                    delegation_id="previous-delegation",
                    parent_tool_call_id="previous-call",
                    caller_agent_id="grandparent-agent",
                    agent_id=caller.agent_id,
                    agent_version=caller.version,
                    agent_path=("grandparent-agent@1.0.0", str(caller)),
                    depth=1,
                ),
                execution_group_id="group-1",
                principal="principal-1",
                tenant="tenant-1",
                parent_session_id="parent-session",
                absolute_deadline=parent.absolute_deadline,
                limits=limits,
                current_run_id=tool.coordinator.id_factory.child_run_id(
                    "previous-delegation"
                ),
            )
        run_id = (
            "root-run"
            if mode == "denied"
            else tool.coordinator.id_factory.child_run_id("previous-delegation")
        )
        with pytest.raises(ToolFailure) as captured:
            await tool.invoke(
                {"question": "PRIVATE"},
                _tool_context(parent, run_id=run_id),
            )
        assert captured.value.error.reason == expected_reason
        assert captured.value.error.details == {}
        assert captured.value.error.retryable is False
        assert handler.calls == 0
        assert await ledger.store.load("group-1") is None

    asyncio.run(scenario())


def test_canonical_topology_guard_cannot_be_replaced_by_a_noop_adapter() -> None:
    async def scenario() -> None:
        handler = _EndpointHandler()
        noop = _NoopCycleGuard()
        limits = ExecutionGroupLimits(max_depth=1)
        tool, parent, ledger, _, caller, _ = await _build_tool(
            handler,
            limits=limits,
            cycle_guard=noop,
        )
        parent = ParentDelegationContext(
            lineage=RunLineage(
                root_run_id="root-run",
                parent_run_id="previous-run",
                delegation_id="previous-delegation",
                parent_tool_call_id="previous-call",
                caller_agent_id="grandparent-agent",
                agent_id=caller.agent_id,
                agent_version=caller.version,
                agent_path=("grandparent-agent@1.0.0", str(caller)),
                depth=1,
            ),
            execution_group_id="group-1",
            principal="principal-1",
            tenant="tenant-1",
            parent_session_id="parent-session",
            absolute_deadline=parent.absolute_deadline,
            limits=limits,
            current_run_id=tool.coordinator.id_factory.child_run_id(
                "previous-delegation"
            ),
        )

        with pytest.raises(ToolFailure) as captured:
            await tool.invoke(
                {"question": "do not bypass depth"},
                _tool_context(
                    parent,
                    run_id=tool.coordinator.id_factory.child_run_id(
                        "previous-delegation"
                    ),
                ),
            )

        assert captured.value.error.reason == "delegation_depth_exceeded"
        assert noop.calls == 0
        assert handler.calls == 0
        assert await ledger.store.load("group-1") is None

    asyncio.run(scenario())


def test_isolated_session_factory_cannot_reuse_the_parent_session() -> None:
    async def scenario() -> None:
        handler = _EndpointHandler()
        tool, parent, _, _, _, _ = await _build_tool(
            handler,
            session_factory=_LeakySessionKeyFactory(
                b"session-key-secret-for-tests-00001"
            ),
        )

        with pytest.raises(ToolFailure) as captured:
            await tool.invoke(
                {"question": "keep sessions isolated"},
                _tool_context(parent),
            )

        assert captured.value.error.reason == "delegation_session_isolation_failed"
        assert handler.calls == 0

    asyncio.run(scenario())


def test_disabled_shared_session_fails_before_a_receipt_can_be_claimed() -> None:
    caller = AgentRef("parent-agent", "1.0.0")
    callee = AgentRef("child-agent", "1.0.0")
    registry = InMemoryAgentRegistry()
    registry.register(
        _definition(callee, caller=caller),
        AgentEndpoint(handler=_EndpointHandler(), approved=True),
        status=DefinitionStatus.ACTIVE,
    )
    receipts = InMemoryDelegationReceiptStore()
    coordinator = DelegationCoordinator(
        registry=registry,
        authorizer=EdgeDelegationAuthorizer({(caller, callee)}),
        receipt_store=receipts,
    )

    with pytest.raises(ValueError, match="shared delegation sessions are disabled"):
        coordinator.tool(
            caller=caller,
            callee=callee,
            input_model=TaskRequest,
            output_model=TaskAnswer,
            session_strategy=SessionStrategy.SHARED,
        )

    assert receipts._receipts == {}


def test_execution_group_ledger_aggregates_limits_and_backpressures() -> None:
    async def scenario() -> None:
        limits = ExecutionGroupLimits(
            max_delegations=3,
            max_parallel_delegations=1,
            max_delegations_per_agent=3,
            max_total_model_turns=2,
            max_total_tool_calls=1,
            timeout_seconds=5,
        )
        ledger = InMemoryBudgetLedger(queue_poll_seconds=0.001)
        deadline = datetime.now(timezone.utc) + timedelta(seconds=5)
        await ledger.ensure_group("budget-group", limits, absolute_deadline=deadline)
        callee = AgentRef("child", "1.0.0")
        first = await ledger.reserve_delegation("budget-group", callee)
        waiting = asyncio.create_task(ledger.reserve_delegation("budget-group", callee))
        await asyncio.sleep(0.01)
        assert not waiting.done()
        await ledger.complete_lease(first, usage={"total_tokens": 7})
        second = await asyncio.wait_for(waiting, timeout=1)
        await ledger.release_lease(second)
        await ledger.reserve_model_turn("budget-group", count=2)
        await ledger.reserve_tool_call("budget-group")
        with pytest.raises(BudgetExceeded, match="model_turns"):
            await ledger.reserve_model_turn("budget-group")
        with pytest.raises(BudgetExceeded, match="tool_calls"):
            await ledger.reserve_tool_call("budget-group")
        snapshot = await ledger.snapshot("budget-group")
        assert snapshot.active_delegations == 0
        assert snapshot.usage == {"total_tokens": 7}

        with pytest.raises(ValueError, match="durable"):
            DurableBudgetLedger(InMemoryBudgetStateStore())

        class DurableTestStore(InMemoryBudgetStateStore):
            durable = True

        assert isinstance(DurableBudgetLedger(DurableTestStore()), DurableBudgetLedger)

    asyncio.run(scenario())


def test_budget_reservation_failure_does_not_leave_a_live_receipt_owner() -> None:
    async def scenario() -> None:
        limits = ExecutionGroupLimits(
            max_delegations=1,
            max_parallel_delegations=1,
            max_delegations_per_agent=1,
            timeout_seconds=10,
        )
        handler = _EndpointHandler(answer="must not run")
        tool, parent, ledger, receipts, _, callee = await _build_tool(
            handler,
            limits=limits,
        )
        await ledger.ensure_group(
            parent.execution_group_id,
            limits,
            absolute_deadline=parent.absolute_deadline,
        )
        consumed = await ledger.reserve_delegation(
            parent.execution_group_id,
            callee,
        )
        await ledger.complete_lease(consumed)

        with pytest.raises(ToolFailure) as captured:
            await tool.invoke({"question": "new"}, _tool_context(parent))
        assert captured.value.error.reason == "delegation_count_exceeded"
        assert handler.calls == 0
        receipt = next(iter(receipts._receipts.values()))
        assert receipt.status is DelegationReceiptStatus.MANUAL_REQUIRED
        assert receipt.error_code == "delegation_budget_reservation_uncertain"
        state = await ledger.snapshot(parent.execution_group_id)
        assert state.delegation_count == 1
        assert state.active_delegations == 0

    asyncio.run(scenario())


def test_hmac_ids_and_session_namespaces_are_deterministic_and_opaque() -> None:
    factory = DelegationIdFactory(b"delegation-id-secret-for-tests-0001")
    request_digest = factory.request_digest({"question": "TOP-SECRET"})
    first = factory.delegation_id(
        parent_run_id="parent-run",
        parent_tool_call_id="tool-call",
        request_digest=request_digest,
    )
    second = factory.delegation_id(
        parent_run_id="parent-run",
        parent_tool_call_id="tool-call",
        request_digest=request_digest,
    )
    assert first == second
    assert "parent-run" not in first
    assert "tool-call" not in first
    assert "TOP-SECRET" not in first
    assert factory.child_run_id(first) == factory.child_run_id(first)
    lease_id = factory.lease_id(first)
    assert lease_id == factory.lease_id(first)
    assert first not in lease_id
    assert "parent-run" not in lease_id

    sessions = SessionKeyFactory(b"session-key-secret-for-tests-00001")
    callee = AgentRef("child", "1.0.0")
    isolated = sessions.create(
        strategy=SessionStrategy.ISOLATED,
        tenant="tenant-1",
        parent_session_id="parent-session",
        callee=callee,
        delegation_id=first,
    )
    per_parent = sessions.create(
        strategy=SessionStrategy.PER_PARENT_SESSION,
        tenant="tenant-1",
        parent_session_id="parent-session",
        callee=callee,
        delegation_id=first,
    )
    assert isolated != per_parent
    assert "tenant-1" not in isolated
    assert "parent-session" not in per_parent
    with pytest.raises(ValueError, match="shared"):
        sessions.create(
            strategy=SessionStrategy.SHARED,
            tenant="tenant-1",
            parent_session_id="parent-session",
            callee=callee,
            delegation_id=first,
        )


def test_child_failure_is_terminal_non_retryable_and_payload_free() -> None:
    async def scenario() -> None:
        handler = _EndpointHandler(
            outcome_status=DelegationOutcomeStatus.FAILED,
            error_code="private_backend_failure",
        )
        tool, parent, ledger, receipts, _, _ = await _build_tool(handler)
        with pytest.raises(ToolFailure) as captured:
            await tool.invoke({"question": "SECRET"}, _tool_context(parent))
        error = captured.value.error
        assert error.type is ToolErrorType.EXECUTION_ERROR
        assert error.reason == "delegated_agent_failed"
        assert error.retryable is False
        assert error.details == {}
        assert "SECRET" not in json.dumps(error.to_dict())
        assert "private_backend_failure" not in json.dumps(error.to_dict())
        assert handler.calls == 1
        state = await ledger.snapshot("group-1")
        assert state.active_delegations == 0
        assert state.usage == {"input_tokens": 2}
        delegation_id = handler.received[1].lineage.delegation_id
        receipt = await receipts.get(delegation_id)
        assert receipt is not None
        assert receipt.status is DelegationReceiptStatus.FAILED
        assert receipt.retryable is False

    asyncio.run(scenario())


def test_parent_cancellation_reaches_child_and_closes_receipt() -> None:
    async def scenario() -> None:
        handler = _CancellingEndpoint()
        tool, parent, ledger, receipts, _, _ = await _build_tool(handler)
        running = asyncio.create_task(
            tool.invoke({"question": "cancel me"}, _tool_context(parent))
        )
        await asyncio.wait_for(handler.started.wait(), timeout=1)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert handler.cancelled.is_set()
        stored = tuple(receipts._receipts.values())
        assert len(stored) == 1
        assert stored[0].status is DelegationReceiptStatus.CANCELLED
        assert (await ledger.snapshot("group-1")).active_delegations == 0

    asyncio.run(scenario())


def test_unconfirmed_cancellation_releases_retained_lease_once_after_stop() -> None:
    class StrictTerminalLedger(InMemoryBudgetLedger):
        def __init__(self) -> None:
            super().__init__(queue_poll_seconds=0.001)
            self.terminal_lease_ids: set[str] = set()
            self.releases = 0

        async def complete_lease(self, lease, *, usage=None) -> None:
            if lease.lease_id in self.terminal_lease_ids:
                raise AssertionError("lease was settled more than once")
            self.terminal_lease_ids.add(lease.lease_id)
            await super().complete_lease(lease, usage=usage)

        async def release_lease(self, lease) -> None:
            if lease.lease_id in self.terminal_lease_ids:
                raise AssertionError("lease was settled more than once")
            self.terminal_lease_ids.add(lease.lease_id)
            self.releases += 1
            await super().release_lease(lease)

    async def scenario() -> None:
        handler = _SlowCancellationEndpoint()
        ledger = StrictTerminalLedger()
        tool, parent, _, receipts, _, _ = await _build_tool(
            handler,
            budget_ledger=ledger,
            cancellation_grace_seconds=0.001,
        )
        running = asyncio.create_task(
            tool.invoke({"question": "cancel me"}, _tool_context(parent))
        )
        await asyncio.wait_for(handler.started.wait(), timeout=1)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        await asyncio.wait_for(handler.stopped.wait(), timeout=1)
        for _ in range(10):
            if ledger.releases == 1:
                break
            await asyncio.sleep(0)
        assert ledger.releases == 1
        assert (await ledger.snapshot("group-1")).active_delegations == 0
        stored = tuple(receipts._receipts.values())
        assert len(stored) == 1
        assert stored[0].status is DelegationReceiptStatus.MANUAL_REQUIRED

    asyncio.run(scenario())


def test_unknown_running_receipt_requires_manual_reconciliation() -> None:
    async def scenario() -> None:
        handler = _EndpointHandler()
        store = InMemoryDelegationReceiptStore()
        tool, parent, _, _, caller, callee = await _build_tool(
            handler,
            receipt_store=store,
        )
        parent = ParentDelegationContext(
            lineage=parent.lineage,
            execution_group_id=parent.execution_group_id,
            principal=parent.principal,
            tenant=parent.tenant,
            parent_session_id=parent.parent_session_id,
            absolute_deadline=datetime.now(timezone.utc) - timedelta(seconds=1),
            limits=parent.limits,
        )
        ids = tool.coordinator.id_factory
        request = TaskRequest(question="same")
        digest = ids.request_digest(request.model_dump(mode="json"))
        delegation_id = ids.delegation_id(
            parent_run_id="root-run",
            parent_tool_call_id="call-1",
            request_digest=digest,
        )
        receipt = DelegationReceipt(
            delegation_id=delegation_id,
            execution_group_id="group-1",
            root_run_id="root-run",
            parent_run_id="root-run",
            parent_tool_call_id="call-1",
            caller_agent_ref=caller,
            callee_agent_ref=callee,
            child_run_id=ids.child_run_id(delegation_id),
            request_digest=digest,
            context_digest=_receipt_context_digest(
                tool,
                parent,
                caller,
                callee,
            ),
            owner_token="owner:unknown-running",
            lease_id=ids.lease_id(delegation_id),
        )
        await store.claim(receipt)
        await ReceiptManager(store).transition(
            delegation_id,
            expected={DelegationReceiptStatus.RESERVED},
            status=DelegationReceiptStatus.RUNNING,
        )
        with pytest.raises(ToolFailure) as captured:
            await tool.invoke({"question": "same"}, _tool_context(parent))
        assert captured.value.error.reason == "delegation_reconciliation_required"
        assert handler.calls == 0
        stored = await store.get(delegation_id)
        assert stored is not None
        assert stored.status is DelegationReceiptStatus.MANUAL_REQUIRED

    asyncio.run(scenario())
