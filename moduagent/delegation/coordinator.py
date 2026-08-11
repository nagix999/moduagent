from __future__ import annotations

import asyncio
import hashlib
import math
import re
import secrets
import uuid
import weakref
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from moduagent.definitions import (
    AgentDescriptor,
    AgentDefinitionNotRunnableError,
    AgentNotFoundError,
    ResolvedAgentEndpoint,
)

from .budget import (
    BudgetExceeded,
    BudgetLedger,
    BudgetStateStore,
    DurableBudgetLedger,
    InMemoryBudgetLedger,
    LeaseStatus,
    StoreBackedBudgetLedger,
)
from .events import (
    DelegationEvent,
    DelegationEventSink,
    DelegationEventType,
    NoopDelegationEventSink,
)
from .guards import (
    CycleGuard,
    DelegationAuthorizer,
    DelegationRejected,
    authorize_or_reject,
)
from .models import (
    AgentRef,
    BudgetLease,
    DelegationContext,
    DelegationOutcome,
    DelegationOutcomeStatus,
    ExecutionGroupLimits,
    ParentDelegationContext,
    RunLineage,
    _classification,
    _utc,
)
from .receipts import (
    DelegationIdFactory,
    DelegationReceipt,
    DelegationReceiptStatus,
    DelegationReceiptStore,
    InMemoryDelegationReceiptStore,
    ReceiptAction,
    ReceiptClaim,
    ReceiptManager,
    ReceiptStoreError,
    _canonical_json,
    canonical_digest,
    receipt_action,
)
from .registry import (
    AgentEndpoint,
    AgentRegistry,
    AgentRegistryError,
    DelegationEndpointError,
    LocalAgentInvoker,
)
from .sessions import SessionKeyFactory, SessionStrategy

if TYPE_CHECKING:
    from .tool import DelegatedAgentTool, ParentContextResolver


_DEFAULT_DELEGATION_MAX_RESULT_BYTES = 1_000_000


def _resolve_max_result_bytes(value: int | None) -> int:
    resolved = _DEFAULT_DELEGATION_MAX_RESULT_BYTES if value is None else value
    if type(resolved) is not int:
        raise TypeError("max_result_bytes must be a positive integer")
    if resolved < 1:
        raise ValueError("max_result_bytes must be at least 1")
    return resolved


def _require_object_model(
    model: object,
    field_name: str,
) -> type[BaseModel]:
    if not isinstance(model, type) or not issubclass(model, BaseModel):
        raise TypeError(f"{field_name} must be a Pydantic BaseModel class")
    try:
        schema = model.model_json_schema()
    except Exception as exc:
        raise TypeError(
            f"{field_name} must provide an object-root Pydantic schema"
        ) from exc
    if (
        getattr(model, "__pydantic_root_model__", False) is True
        or not isinstance(schema, Mapping)
        or schema.get("type") != "object"
    ):
        raise TypeError(f"{field_name} must be an object-root Pydantic BaseModel class")
    return model


class DelegationFailure(Exception):
    """A model-safe failure classification emitted by the coordinator."""

    def __init__(
        self,
        code: str,
        *,
        kind: str = "execution_error",
        retryable: bool = False,
    ) -> None:
        _classification(code, "delegation failure code")
        if kind not in {
            "execution_error",
            "unauthorized",
            "timeout",
            "cancelled",
            "result_too_large",
        }:
            raise ValueError("delegation failure kind is invalid")
        if type(retryable) is not bool:
            raise TypeError("retryable must be a bool")
        super().__init__(code)
        self.code = code
        self.kind = kind
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class DelegationCall:
    caller: AgentRef
    callee: AgentRef
    request: BaseModel
    output_model: type[BaseModel]
    parent: ParentDelegationContext
    parent_run_id: str
    parent_tool_call_id: str
    expected_definition_fingerprint: str
    request_classification: str = "internal"
    session_strategy: SessionStrategy = SessionStrategy.ISOLATED
    allow_resume: bool = False
    event_callback: Callable[[DelegationEvent], Awaitable[None]] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    max_result_bytes: int = _DEFAULT_DELEGATION_MAX_RESULT_BYTES

    def __post_init__(self) -> None:
        for ref, name in ((self.caller, "caller"), (self.callee, "callee")):
            if not isinstance(ref, AgentRef):
                raise TypeError(f"{name} must be an AgentRef")
        if not isinstance(self.request, BaseModel):
            raise TypeError("request must be a Pydantic BaseModel")
        _require_object_model(type(self.request), "request")
        _require_object_model(self.output_model, "output_model")
        if not isinstance(self.parent, ParentDelegationContext):
            raise TypeError("parent must be ParentDelegationContext")
        if self.parent.lineage.agent_ref != self.caller:
            raise ValueError("caller must match the current lineage Agent")
        for value, name in (
            (self.parent_run_id, "parent_run_id"),
            (self.parent_tool_call_id, "parent_tool_call_id"),
            (self.expected_definition_fingerprint, "expected_definition_fingerprint"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} cannot be empty")
        if (
            re.fullmatch(r"sha256:[0-9a-f]{64}", self.expected_definition_fingerprint)
            is None
        ):
            raise ValueError(
                "expected_definition_fingerprint must be a canonical sha256 value"
            )
        _classification(self.request_classification, "request_classification")
        if not isinstance(self.session_strategy, SessionStrategy):
            object.__setattr__(
                self,
                "session_strategy",
                SessionStrategy(str(self.session_strategy)),
            )
        if type(self.allow_resume) is not bool:
            raise TypeError("allow_resume must be a bool")
        if self.event_callback is not None and not callable(self.event_callback):
            raise TypeError("event_callback must be callable or None")
        object.__setattr__(
            self,
            "max_result_bytes",
            _resolve_max_result_bytes(self.max_result_bytes),
        )


class DelegationCoordinator:
    """Apply delegation policy, budget, receipt, and invocation boundaries."""

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        policy: DelegationAuthorizer | None = None,
        authorizer: DelegationAuthorizer | None = None,
        budget_ledger: BudgetLedger | None = None,
        receipt_store: DelegationReceiptStore | None = None,
        execution_group_store: BudgetStateStore | None = None,
        id_factory: DelegationIdFactory | None = None,
        session_factory: SessionKeyFactory | None = None,
        hmac_secret: bytes | None = None,
        limits: ExecutionGroupLimits | None = None,
        invoker: LocalAgentInvoker | None = None,
        cycle_guard: CycleGuard | None = None,
        event_sink: DelegationEventSink | None = None,
        cancellation_grace_seconds: float = 1.0,
    ) -> None:
        if not isinstance(registry, AgentRegistry):
            raise TypeError("registry must implement AgentRegistry")
        if policy is not None and authorizer is not None:
            raise ValueError("use either policy or authorizer, not both")
        resolved_authorizer = policy if policy is not None else authorizer
        if not isinstance(resolved_authorizer, DelegationAuthorizer):
            raise TypeError("policy must implement DelegationAuthorizer")
        if budget_ledger is not None and execution_group_store is not None:
            raise ValueError(
                "use either budget_ledger or execution_group_store, not both"
            )
        if execution_group_store is not None:
            if not isinstance(execution_group_store, BudgetStateStore):
                raise TypeError("execution_group_store must implement BudgetStateStore")
            resolved_ledger: BudgetLedger = (
                DurableBudgetLedger(execution_group_store)
                if execution_group_store.durable
                else StoreBackedBudgetLedger(execution_group_store)
            )
        else:
            resolved_ledger = (
                budget_ledger if budget_ledger is not None else InMemoryBudgetLedger()
            )
        if not isinstance(resolved_ledger, BudgetLedger):
            raise TypeError("budget_ledger must implement BudgetLedger")
        resolved_receipt_store = (
            receipt_store
            if receipt_store is not None
            else InMemoryDelegationReceiptStore()
        )
        if not isinstance(resolved_receipt_store, DelegationReceiptStore):
            raise TypeError("receipt_store must implement DelegationReceiptStore")
        id_factory, session_factory = _resolve_identity_factories(
            id_factory=id_factory,
            session_factory=session_factory,
            hmac_secret=hmac_secret,
            durable=(
                resolved_receipt_store.durable
                or getattr(
                    execution_group_store
                    if execution_group_store is not None
                    else resolved_ledger,
                    "durable",
                    False,
                )
                is True
            ),
        )
        if not isinstance(id_factory, DelegationIdFactory):
            raise TypeError("id_factory must be DelegationIdFactory")
        if not isinstance(session_factory, SessionKeyFactory):
            raise TypeError("session_factory must be SessionKeyFactory")
        self.registry = registry
        self.authorizer = resolved_authorizer
        self.budget_ledger = resolved_ledger
        # Canonical deployment binding: a direct state store when supplied,
        # otherwise the replaceable ledger itself. Production validation uses
        # its explicit durability capability and never introspects `.store`.
        self.execution_group_binding = (
            execution_group_store
            if execution_group_store is not None
            else resolved_ledger
        )
        self.receipt_store = resolved_receipt_store
        self.receipts = ReceiptManager(resolved_receipt_store)
        self.id_factory = id_factory
        self.session_factory = session_factory
        self.limits = limits or ExecutionGroupLimits()
        if not isinstance(self.limits, ExecutionGroupLimits):
            raise TypeError("limits must be ExecutionGroupLimits")
        self.invoker = invoker or LocalAgentInvoker()
        self._canonical_cycle_guard = CycleGuard()
        if cycle_guard is not None and not callable(
            getattr(cycle_guard, "validate", None)
        ):
            raise TypeError("cycle_guard must provide validate()")
        self.cycle_guard = cycle_guard or self._canonical_cycle_guard
        self.event_sink = event_sink or NoopDelegationEventSink()
        if not isinstance(self.event_sink, DelegationEventSink):
            raise TypeError("event_sink must implement DelegationEventSink")
        if (
            isinstance(cancellation_grace_seconds, bool)
            or not isinstance(cancellation_grace_seconds, (int, float))
            or not math.isfinite(float(cancellation_grace_seconds))
            or cancellation_grace_seconds <= 0
        ):
            raise ValueError("cancellation_grace_seconds must be positive and finite")
        self.cancellation_grace_seconds = float(cancellation_grace_seconds)
        self._delegation_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._locks_guard = asyncio.Lock()
        self._active_tasks: dict[str, set[asyncio.Task[DelegationOutcome]]] = {}
        self._active_guard = asyncio.Lock()
        self._retained_lease_ids: set[str] = set()
        self._retained_release_events: dict[str, asyncio.Event] = {}
        self._settled_lease_ids: set[str] = set()

    def tool(
        self,
        *,
        caller: AgentRef,
        callee: AgentRef,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        name: str | None = None,
        description: str | None = None,
        request_classification: str | None = None,
        session_strategy: SessionStrategy = SessionStrategy.ISOLATED,
        context_resolver: ParentContextResolver | None = None,
        max_result_bytes: int | None = None,
        allow_resume: bool = False,
    ) -> DelegatedAgentTool:
        """Create the typed Agent-as-Tool adapter proposed by the public API."""

        from .tool import DelegatedAgentTool

        if not isinstance(caller, AgentRef):
            raise TypeError("caller must be an AgentRef")
        if not isinstance(callee, AgentRef):
            raise TypeError("callee must be an AgentRef")
        _require_object_model(input_model, "input_model")
        _require_object_model(output_model, "output_model")
        resolved_session_strategy = SessionStrategy(session_strategy)
        if (
            resolved_session_strategy is SessionStrategy.SHARED
            and self.session_factory.allow_shared is not True
        ):
            raise ValueError("shared delegation sessions are disabled")
        descriptor = self.registry.descriptor(callee)
        return DelegatedAgentTool(
            coordinator=self,
            caller=caller,
            callee=callee,
            input_model=input_model,
            output_model=output_model,
            name=_default_tool_name(callee) if name is None else name,
            description=(
                descriptor.description if description is None else description
            ),
            request_classification=(
                request_classification
                if request_classification is not None
                else descriptor.data_classification
            ),
            session_strategy=resolved_session_strategy,
            context_resolver=context_resolver,
            max_result_bytes=max_result_bytes,
            allow_resume=allow_resume,
            side_effect_level=descriptor.side_effect_level,
            expected_definition_fingerprint=descriptor.definition_fingerprint,
        )

    def parent_context(
        self,
        *,
        caller: AgentRef,
        run_id: str,
        session_id: str,
        principal: str,
        tenant: str,
        incoming: DelegationContext | None = None,
        execution_group_id: str | None = None,
        absolute_deadline: datetime | None = None,
    ) -> ParentDelegationContext:
        """Build the runtime-owned context placed in ToolExecutionContext metadata."""

        if not isinstance(caller, AgentRef):
            raise TypeError("caller must be an AgentRef")
        if incoming is None:
            maximum_deadline = datetime.now(timezone.utc) + timedelta(
                seconds=self.limits.timeout_seconds
            )
            deadline = (
                min(
                    _utc(absolute_deadline, "absolute_deadline"),
                    maximum_deadline,
                )
                if absolute_deadline is not None
                else maximum_deadline
            )
            lineage = RunLineage.root(run_id=run_id, agent=caller)
            group_id = execution_group_id or run_id
        else:
            if not isinstance(incoming, DelegationContext):
                raise TypeError("incoming must be a DelegationContext or None")
            if incoming.lineage.depth < 1 or incoming.lineage.delegation_id is None:
                raise ValueError("incoming context must describe a delegated child")
            if incoming.lineage.agent_ref != caller:
                raise ValueError("incoming lineage does not match caller")
            if principal != incoming.principal or tenant != incoming.tenant:
                raise ValueError("incoming security claims do not match")
            if session_id != incoming.child_session_id:
                raise ValueError("incoming child session does not match")
            if execution_group_id not in (None, incoming.execution_group_id):
                raise ValueError("incoming execution group does not match")
            if absolute_deadline not in (None, incoming.absolute_deadline):
                raise ValueError("incoming deadline does not match")
            deadline = incoming.absolute_deadline
            lineage = incoming.lineage
            group_id = incoming.execution_group_id
        return ParentDelegationContext(
            lineage=lineage,
            execution_group_id=group_id,
            principal=principal,
            tenant=tenant,
            parent_session_id=session_id,
            absolute_deadline=deadline,
            limits=self.limits,
            current_run_id=run_id,
        )

    async def delegate(self, call: DelegationCall) -> BaseModel:
        if not isinstance(call, DelegationCall):
            raise TypeError("call must be a DelegationCall")
        if call.parent.limits != self.limits:
            raise DelegationFailure("execution_group_contract_mismatch")
        self._validate_parent_run_identity(call)
        request_payload = call.request.model_dump(mode="json", by_alias=True)
        request_digest = self.id_factory.request_digest(request_payload)
        context_digest = self._context_digest(call)
        delegation_id = self.id_factory.delegation_id(
            parent_run_id=call.parent_run_id,
            parent_tool_call_id=call.parent_tool_call_id,
            request_digest=request_digest,
        )
        child_run_id = self.id_factory.child_run_id(delegation_id)
        await self._emit(
            DelegationEventType.REQUESTED,
            call,
            delegation_id,
            child_run_id=child_run_id,
        )

        lease = None
        owns_lease = False
        try:
            resolved = self.registry.resolve(call.callee)
            descriptor = self.registry.descriptor(call.callee)
            _validate_registry_resolution(
                call.callee,
                resolved,
                descriptor,
                expected_definition_fingerprint=call.expected_definition_fingerprint,
            )
            if resolved.endpoint.approved is not True:
                raise DelegationRejected(
                    (
                        "delegation_remote_endpoint_not_approved"
                        if resolved.endpoint.kind == "remote"
                        else "delegation_endpoint_not_approved"
                    ),
                    unauthorized=True,
                )
            _validate_contract_digests(call, descriptor)
            await authorize_or_reject(
                self.authorizer,
                caller=call.caller,
                callee=call.callee,
                descriptor=descriptor,
                parent=call.parent,
                request_classification=call.request_classification,
            )
            self._canonical_cycle_guard.validate(
                lineage=call.parent.lineage,
                callee=call.callee,
                limits=call.parent.limits,
            )
            if self.cycle_guard is not self._canonical_cycle_guard:
                self.cycle_guard.validate(
                    lineage=call.parent.lineage,
                    callee=call.callee,
                    limits=call.parent.limits,
                )
            await self._emit(
                DelegationEventType.AUTHORIZED,
                call,
                delegation_id,
                child_run_id=child_run_id,
            )
            await self.budget_ledger.ensure_group(
                call.parent.execution_group_id,
                call.parent.limits,
                absolute_deadline=call.parent.absolute_deadline,
            )
            child_lineage = call.parent.lineage.child(
                child=call.callee,
                parent_run_id=call.parent_run_id,
                delegation_id=delegation_id,
                parent_tool_call_id=call.parent_tool_call_id,
            )
            child_session_id = self.session_factory.create(
                strategy=call.session_strategy,
                tenant=call.parent.tenant,
                parent_session_id=call.parent.parent_session_id,
                callee=call.callee,
                delegation_id=delegation_id,
            )
            if (
                call.session_strategy is not SessionStrategy.SHARED
                and child_session_id == call.parent.parent_session_id
            ):
                raise DelegationFailure("delegation_session_isolation_failed")
            owner_token = f"owner:{uuid.uuid4().hex}"
            candidate = self._receipt_candidate(
                call=call,
                delegation_id=delegation_id,
                child_run_id=child_run_id,
                request_digest=request_digest,
                context_digest=context_digest,
                attempt=1,
                owner_token=owner_token,
                lease_id=self.id_factory.lease_id(delegation_id, attempt=1),
            )
            lock = await self._lock_for(delegation_id)
            async with lock:
                claim = await self.receipt_store.claim(candidate)
                receipt = claim.receipt
                self._validate_claim(
                    receipt,
                    call=call,
                    delegation_id=delegation_id,
                    child_run_id=child_run_id,
                    request_digest=request_digest,
                    context_digest=context_digest,
                )
                resumed = False
                if not claim.created:
                    lease = BudgetLease(
                        receipt.lease_id,
                        call.parent.execution_group_id,
                        call.callee,
                        call.parent.absolute_deadline,
                    )
                    await self._validate_existing_receipt_lease(
                        call=call,
                        receipt=receipt,
                        lease=lease,
                    )
                if (
                    not claim.created
                    and receipt_action(
                        receipt,
                        created=False,
                        allow_resume=call.allow_resume,
                    )
                    is ReceiptAction.RESUME
                ):
                    try:
                        receipt = await self._claim_resume_attempt(
                            receipt,
                            owner_token=owner_token,
                        )
                    except ReceiptStoreError as exc:
                        if exc.code == "delegation_receipt_state_conflict":
                            raise DelegationFailure("delegation_in_progress") from exc
                        raise
                    claim = ReceiptClaim(receipt, True)
                    resumed = True
                if claim.created:
                    try:
                        lease = await self.budget_ledger.reserve_delegation(
                            call.parent.execution_group_id,
                            call.callee,
                            lease_id=receipt.lease_id,
                        )
                    except Exception:
                        await self._mark_reservation_uncertain(call, receipt)
                        raise
                    owns_lease = True
                if lease is None:
                    raise DelegationFailure("delegation_budget_lease_missing")
                self._validate_lease(lease, call=call, receipt=receipt)
                context = DelegationContext(
                    lineage=child_lineage,
                    execution_group_id=call.parent.execution_group_id,
                    principal=call.parent.principal,
                    tenant=call.parent.tenant,
                    absolute_deadline=lease.absolute_deadline,
                    child_session_id=child_session_id,
                    request_classification=call.request_classification,
                )
                output = await self._handle_claim(
                    call=call,
                    endpoint=resolved.endpoint,
                    context=context,
                    lease=lease,
                    receipt=receipt,
                    created=claim.created,
                    resumed=resumed,
                )
            return output
        except asyncio.CancelledError:
            if owns_lease and lease is not None:
                await self._best_effort_release(lease)
            raise
        except DelegationRejected as exc:
            await self._emit(
                DelegationEventType.REJECTED,
                call,
                delegation_id,
                child_run_id=child_run_id,
                code=exc.code,
            )
            if owns_lease and lease is not None:
                await self._best_effort_release(lease)
            raise DelegationFailure(
                exc.code,
                kind="unauthorized" if exc.unauthorized else "execution_error",
            ) from exc
        except BudgetExceeded as exc:
            await self._emit(
                DelegationEventType.REJECTED,
                call,
                delegation_id,
                child_run_id=child_run_id,
                code=exc.code,
            )
            if owns_lease and lease is not None:
                await self._best_effort_release(lease)
            kind = (
                "timeout"
                if exc.code == "execution_group_timeout"
                else "execution_error"
            )
            raise DelegationFailure(exc.code, kind=kind) from exc
        except AgentRegistryError as exc:
            if owns_lease and lease is not None:
                await self._best_effort_release(lease)
            if isinstance(exc, AgentNotFoundError):
                code = "delegation_agent_not_found"
            elif isinstance(exc, AgentDefinitionNotRunnableError):
                code = "delegation_agent_not_runnable"
            else:
                code = "delegation_registry_failed"
            raise DelegationFailure(code) from exc
        except ReceiptStoreError as exc:
            if owns_lease and lease is not None:
                await self._best_effort_release(lease)
            raise DelegationFailure(exc.code) from exc
        except DelegationFailure:
            if owns_lease and lease is not None:
                await self._best_effort_release(lease)
            raise
        except Exception as exc:
            if owns_lease and lease is not None:
                await self._best_effort_release(lease)
            raise DelegationFailure("delegation_coordinator_failed") from exc
        finally:
            if lease is not None:
                retained_event = self._retained_release_events.get(lease.lease_id)
                if retained_event is not None:
                    retained_event.set()
                self._settled_lease_ids.discard(lease.lease_id)

    async def cancel_execution_group(self, execution_group_id: str) -> None:
        """Cancel active local child tasks and close the shared group budget."""

        async with self._active_guard:
            tasks = tuple(self._active_tasks.get(execution_group_id, ()))
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.wait(
                tasks,
                timeout=self.cancellation_grace_seconds,
            )
        cancel_group = getattr(self.budget_ledger, "cancel_group", None)
        if callable(cancel_group):
            await cancel_group(execution_group_id)

    async def _handle_claim(
        self,
        *,
        call: DelegationCall,
        endpoint: AgentEndpoint,
        context: DelegationContext,
        lease: BudgetLease,
        receipt: DelegationReceipt,
        created: bool,
        resumed: bool,
    ) -> BaseModel:
        action = receipt_action(
            receipt,
            created=created,
            allow_resume=call.allow_resume,
        )
        if action is ReceiptAction.REPLAY:
            output = self._replay(
                receipt,
                call.output_model,
                max_result_bytes=call.max_result_bytes,
            )
            await self._reconcile_completed_lease(lease)
            await self._best_effort_cleanup_checkpoint(
                endpoint=endpoint,
                context=context,
                lease=lease,
                child_run_id=receipt.child_run_id,
            )
            await self._emit(
                DelegationEventType.COMPLETED,
                call,
                receipt.delegation_id,
                child_run_id=receipt.child_run_id,
                status="replayed",
            )
            return output
        if action is ReceiptAction.TERMINAL_FAILURE:
            code = _receipt_failure_code(receipt)
            raise DelegationFailure(
                code,
                kind=(
                    "result_too_large"
                    if receipt.finish_reason == "result_too_large"
                    else "execution_error"
                ),
            )
        if action is ReceiptAction.TERMINAL_CANCELLED:
            raise DelegationFailure(
                "delegated_agent_cancelled",
                kind="cancelled",
            )
        if action is ReceiptAction.MANUAL_REQUIRED:
            raise DelegationFailure("delegation_reconciliation_required")
        if action is ReceiptAction.RECONCILE:
            try:
                reconciled = await self.invoker.reconcile(
                    endpoint,
                    context,
                    lease,
                    self.budget_ledger,
                    child_run_id=receipt.child_run_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if (
                    receipt.status
                    in {
                        DelegationReceiptStatus.RESERVED,
                        DelegationReceiptStatus.RUNNING,
                    }
                    and datetime.now(timezone.utc) >= context.absolute_deadline
                ):
                    await self.receipts.mark_manual_required(
                        receipt,
                        error_code="delegation_reconciliation_failed",
                    )
                    await self._emit(
                        DelegationEventType.RECONCILIATION_REQUIRED,
                        call,
                        receipt.delegation_id,
                        child_run_id=receipt.child_run_id,
                        code="delegation_reconciliation_failed",
                    )
                    raise DelegationFailure("delegation_reconciliation_failed") from exc
                raise DelegationFailure("delegation_in_progress") from exc
            if reconciled is not None:
                reconciled_receipt = receipt
                if receipt.status is DelegationReceiptStatus.RESERVED:
                    reconciled_receipt = await self.receipts.transition(
                        receipt.delegation_id,
                        expected={DelegationReceiptStatus.RESERVED},
                        status=DelegationReceiptStatus.RUNNING,
                        lease_id=lease.lease_id,
                        attempt=receipt.attempt,
                        owner_token=receipt.owner_token,
                    )
                return await self._finish_outcome(
                    call,
                    reconciled_receipt,
                    reconciled,
                    lease,
                    endpoint=endpoint,
                    context=context,
                )
            if (
                receipt.status
                in {
                    DelegationReceiptStatus.RESERVED,
                    DelegationReceiptStatus.RUNNING,
                }
                and datetime.now(timezone.utc) >= context.absolute_deadline
            ):
                await self.receipts.mark_manual_required(receipt)
                await self._emit(
                    DelegationEventType.RECONCILIATION_REQUIRED,
                    call,
                    receipt.delegation_id,
                    child_run_id=receipt.child_run_id,
                    code="delegation_reconciliation_required",
                )
                raise DelegationFailure("delegation_reconciliation_required")
            raise DelegationFailure("delegation_in_progress")

        if action is ReceiptAction.RESUME:
            raise DelegationFailure("delegation_resume_ownership_missing")
        running = await self.receipts.transition(
            receipt.delegation_id,
            expected={DelegationReceiptStatus.RESERVED},
            status=DelegationReceiptStatus.RUNNING,
            finish_reason=None,
            retryable=False,
            resumable=False,
            error_code=None,
            lease_id=lease.lease_id,
            attempt=receipt.attempt,
            owner_token=receipt.owner_token,
        )
        await self._emit(
            DelegationEventType.RESUMED if resumed else DelegationEventType.STARTED,
            call,
            running.delegation_id,
            child_run_id=running.child_run_id,
        )
        outcome = await self._invoke(
            endpoint=endpoint,
            call=call,
            context=context,
            lease=lease,
            child_run_id=running.child_run_id,
            resume=resumed,
        )
        return await self._finish_outcome(
            call,
            running,
            outcome,
            lease,
            endpoint=endpoint,
            context=context,
        )

    async def _invoke(
        self,
        *,
        endpoint: AgentEndpoint,
        call: DelegationCall,
        context: DelegationContext,
        lease: BudgetLease,
        child_run_id: str,
        resume: bool,
    ) -> DelegationOutcome:
        remaining = (
            lease.absolute_deadline - datetime.now(timezone.utc)
        ).total_seconds()
        if remaining <= 0:
            return DelegationOutcome(
                DelegationOutcomeStatus.FAILED,
                child_run_id,
                finish_reason="timeout",
                error_code="delegated_agent_timeout",
            )
        coroutine = (
            self.invoker.resume(
                endpoint,
                call.request,
                context,
                lease,
                self.budget_ledger,
                child_run_id=child_run_id,
            )
            if resume
            else self.invoker.invoke(
                endpoint,
                call.request,
                context,
                lease,
                self.budget_ledger,
                child_run_id=child_run_id,
            )
        )
        task = asyncio.create_task(coroutine)
        await self._track_task(call.parent.execution_group_id, task, add=True)
        try:
            done, _ = await asyncio.wait({task}, timeout=remaining)
            if task in done:
                return task.result()
            task.cancel()
            if not await self._confirm_task_stopped(task):
                self._retain_lease_until_task_stops(lease)
                return DelegationOutcome(
                    DelegationOutcomeStatus.MANUAL_REQUIRED,
                    child_run_id,
                    finish_reason="cancellation_unconfirmed",
                    error_code="delegation_cancellation_unconfirmed",
                )
            self._consume_task_result(task)
            return DelegationOutcome(
                DelegationOutcomeStatus.FAILED,
                child_run_id,
                finish_reason="timeout",
                error_code="delegated_agent_timeout",
            )
        except asyncio.CancelledError:
            task.cancel()
            confirmed = await self._confirm_task_stopped(task)
            if confirmed:
                self._consume_task_result(task)
            else:
                self._retain_lease_until_task_stops(lease)
            await self._terminal_receipt_after_cancellation(
                call,
                child_run_id=child_run_id,
                confirmed=confirmed,
            )
            raise
        except DelegationEndpointError as exc:
            return DelegationOutcome(
                DelegationOutcomeStatus.FAILED,
                child_run_id,
                finish_reason="error",
                error_code=exc.code,
            )
        except Exception:
            return DelegationOutcome(
                DelegationOutcomeStatus.FAILED,
                child_run_id,
                finish_reason="error",
                error_code="delegated_agent_invocation_failed",
            )
        finally:
            if task.done():
                await self._track_task(
                    call.parent.execution_group_id,
                    task,
                    add=False,
                )
            else:
                task.add_done_callback(
                    lambda completed: self._on_task_done(
                        call.parent.execution_group_id,
                        completed,
                        lease,
                    )
                )

    async def _finish_outcome(
        self,
        call: DelegationCall,
        receipt: DelegationReceipt,
        outcome: DelegationOutcome,
        lease: BudgetLease,
        *,
        endpoint: AgentEndpoint,
        context: DelegationContext,
    ) -> BaseModel:
        if outcome.child_run_id != receipt.child_run_id:
            protocol_failure = DelegationOutcome(
                DelegationOutcomeStatus.FAILED,
                receipt.child_run_id,
                finish_reason="error",
                error_code="delegation_child_run_id_mismatch",
                usage=outcome.usage,
            )
            await self._persist_failure(call, receipt, protocol_failure)
            self._settled_lease_ids.add(lease.lease_id)
            await self._complete_lease(lease, usage=outcome.usage)
            raise DelegationFailure("delegation_child_run_id_mismatch")
        if outcome.status is DelegationOutcomeStatus.COMPLETED:
            try:
                raw_output = (
                    outcome.output.model_dump(mode="python", by_alias=True)
                    if isinstance(outcome.output, BaseModel)
                    else outcome.output
                )
                output = call.output_model.model_validate(raw_output)
                raw_payload = output.model_dump(mode="json", by_alias=True)
                if not isinstance(raw_payload, Mapping):
                    raise TypeError("delegated output must serialize to an object")
                payload = dict(raw_payload)
                canonical_payload = _canonical_json(payload)
            except Exception as exc:
                await self._settle_output_failure(
                    call,
                    receipt,
                    lease,
                    usage=outcome.usage,
                    finish_reason="output_validation",
                    error_code="delegated_agent_output_validation_failed",
                )
                raise DelegationFailure(
                    "delegated_agent_output_validation_failed"
                ) from exc
            if len(canonical_payload) > call.max_result_bytes:
                await self._settle_output_failure(
                    call,
                    receipt,
                    lease,
                    usage=outcome.usage,
                    finish_reason="result_too_large",
                    error_code="delegated_agent_result_too_large",
                )
                raise DelegationFailure(
                    "delegated_agent_result_too_large",
                    kind="result_too_large",
                )
            result_digest = f"sha256:{hashlib.sha256(canonical_payload).hexdigest()}"
            try:
                await self.receipts.transition(
                    receipt.delegation_id,
                    expected={DelegationReceiptStatus.RUNNING},
                    status=DelegationReceiptStatus.COMPLETED,
                    finish_reason=outcome.finish_reason or "completed",
                    retryable=False,
                    resumable=False,
                    result_payload=payload,
                    result_digest=result_digest,
                    error_code=None,
                )
            except ReceiptStoreError as exc:
                if exc.code != "delegation_receipt_state_conflict":
                    raise
                return await self._recover_completed_race(
                    call=call,
                    prior=receipt,
                    expected_result_digest=result_digest,
                    lease=lease,
                    usage=outcome.usage,
                    cause=exc,
                    endpoint=endpoint,
                    context=context,
                )
            # The receipt is now the durable source of truth. If ledger settlement
            # fails, leave the lease ACTIVE so a replay can reconcile it; generic
            # exception cleanup must not rewrite it to RELEASED.
            self._settled_lease_ids.add(lease.lease_id)
            await self._complete_lease(
                lease,
                usage=outcome.usage,
            )
            await self._emit(
                DelegationEventType.COMPLETED,
                call,
                receipt.delegation_id,
                child_run_id=receipt.child_run_id,
                status=outcome.status.value,
            )
            await self._best_effort_cleanup_checkpoint(
                endpoint=endpoint,
                context=context,
                lease=lease,
                child_run_id=receipt.child_run_id,
            )
            return output
        await self._persist_failure(call, receipt, outcome)
        self._settled_lease_ids.add(lease.lease_id)
        if outcome.finish_reason != "cancellation_unconfirmed":
            await self._complete_lease(lease, usage=outcome.usage)
        kind = (
            "cancelled"
            if outcome.status is DelegationOutcomeStatus.CANCELLED
            else (
                "timeout" if outcome.finish_reason == "timeout" else "execution_error"
            )
        )
        raise DelegationFailure(_outcome_failure_code(outcome), kind=kind)

    async def _settle_output_failure(
        self,
        call: DelegationCall,
        receipt: DelegationReceipt,
        lease: BudgetLease,
        *,
        usage: Mapping[str, int | float],
        finish_reason: str,
        error_code: str,
    ) -> None:
        failure = DelegationOutcome(
            DelegationOutcomeStatus.FAILED,
            receipt.child_run_id,
            finish_reason=finish_reason,
            error_code=error_code,
            usage=usage,
        )
        await self._persist_failure(call, receipt, failure)
        self._settled_lease_ids.add(lease.lease_id)
        await self._complete_lease(lease, usage=usage)

    async def _recover_completed_race(
        self,
        *,
        call: DelegationCall,
        prior: DelegationReceipt,
        expected_result_digest: str,
        lease: BudgetLease,
        usage: Mapping[str, int | float],
        cause: ReceiptStoreError,
        endpoint: AgentEndpoint,
        context: DelegationContext,
    ) -> BaseModel:
        """Join an identical terminal completion won by another coordinator."""

        current = await self.receipt_store.get(prior.delegation_id)
        if (
            current is None
            or current.status is not DelegationReceiptStatus.COMPLETED
            or current.attempt != prior.attempt
            or current.result_digest != expected_result_digest
        ):
            raise DelegationFailure("delegation_terminal_race_mismatch") from cause
        try:
            self._validate_claim(
                current,
                call=call,
                delegation_id=prior.delegation_id,
                child_run_id=prior.child_run_id,
                request_digest=prior.request_digest,
                context_digest=prior.context_digest,
            )
            output = self._replay(
                current,
                call.output_model,
                max_result_bytes=call.max_result_bytes,
            )
        except (DelegationFailure, ReceiptStoreError) as exc:
            raise DelegationFailure("delegation_terminal_race_mismatch") from exc
        self._settled_lease_ids.add(lease.lease_id)
        await self._reconcile_completed_lease(lease, usage=usage)
        await self._emit(
            DelegationEventType.COMPLETED,
            call,
            current.delegation_id,
            child_run_id=current.child_run_id,
            status="reconciled",
        )
        await self._best_effort_cleanup_checkpoint(
            endpoint=endpoint,
            context=context,
            lease=lease,
            child_run_id=current.child_run_id,
        )
        return output

    async def _persist_failure(
        self,
        call: DelegationCall,
        receipt: DelegationReceipt,
        outcome: DelegationOutcome,
    ) -> None:
        status = {
            DelegationOutcomeStatus.FAILED: DelegationReceiptStatus.FAILED,
            DelegationOutcomeStatus.CANCELLED: DelegationReceiptStatus.CANCELLED,
            DelegationOutcomeStatus.MANUAL_REQUIRED: (
                DelegationReceiptStatus.MANUAL_REQUIRED
            ),
        }.get(outcome.status, DelegationReceiptStatus.FAILED)
        await self.receipts.transition(
            receipt.delegation_id,
            expected={DelegationReceiptStatus.RUNNING},
            status=status,
            finish_reason=outcome.finish_reason,
            retryable=False,
            resumable=outcome.resumable,
            error_code=_outcome_failure_code(outcome),
        )
        event_type = (
            DelegationEventType.RECONCILIATION_REQUIRED
            if status is DelegationReceiptStatus.MANUAL_REQUIRED
            else DelegationEventType.FAILED
        )
        await self._emit(
            event_type,
            call,
            receipt.delegation_id,
            child_run_id=receipt.child_run_id,
            status=status.value,
            code=_outcome_failure_code(outcome),
        )

    async def _terminal_receipt_after_cancellation(
        self,
        call: DelegationCall,
        *,
        child_run_id: str,
        confirmed: bool,
    ) -> None:
        receipt = await self.receipt_store.get(
            self.id_factory.delegation_id(
                parent_run_id=call.parent_run_id,
                parent_tool_call_id=call.parent_tool_call_id,
                request_digest=self.id_factory.request_digest(
                    call.request.model_dump(mode="json", by_alias=True)
                ),
            )
        )
        if receipt is None or receipt.status is not DelegationReceiptStatus.RUNNING:
            return
        try:
            status = (
                DelegationReceiptStatus.CANCELLED
                if confirmed
                else DelegationReceiptStatus.MANUAL_REQUIRED
            )
            code = (
                "delegated_agent_cancelled"
                if confirmed
                else "delegation_cancellation_unconfirmed"
            )
            await self.receipts.transition(
                receipt.delegation_id,
                expected={DelegationReceiptStatus.RUNNING},
                status=status,
                finish_reason=(
                    "cancelled" if confirmed else "cancellation_unconfirmed"
                ),
                retryable=False,
                resumable=False,
                error_code=code,
            )
            await self._emit(
                (
                    DelegationEventType.FAILED
                    if confirmed
                    else DelegationEventType.RECONCILIATION_REQUIRED
                ),
                call,
                receipt.delegation_id,
                child_run_id=child_run_id,
                status=status.value,
                code=code,
            )
        except ReceiptStoreError:
            return

    async def _confirm_task_stopped(
        self,
        task: asyncio.Task[DelegationOutcome],
    ) -> bool:
        if task.done():
            return True
        done, _ = await asyncio.wait(
            {task},
            timeout=self.cancellation_grace_seconds,
        )
        return task in done

    @staticmethod
    def _consume_task_result(task: asyncio.Task[DelegationOutcome]) -> None:
        try:
            task.exception()
        except asyncio.CancelledError:
            return

    def _on_task_done(
        self,
        execution_group_id: str,
        task: asyncio.Task[DelegationOutcome],
        lease: BudgetLease,
    ) -> None:
        self._consume_task_result(task)
        asyncio.create_task(
            self._finish_background_task(execution_group_id, task, lease)
        )

    async def _finish_background_task(
        self,
        execution_group_id: str,
        task: asyncio.Task[DelegationOutcome],
        lease: BudgetLease,
    ) -> None:
        await self._track_task(execution_group_id, task, add=False)
        if lease.lease_id in self._retained_lease_ids:
            delegate_finished = self._retained_release_events.get(lease.lease_id)
            if delegate_finished is not None:
                await delegate_finished.wait()
            self._retained_lease_ids.discard(lease.lease_id)
            await self._best_effort_release(lease)
            self._settled_lease_ids.discard(lease.lease_id)
            self._retained_release_events.pop(lease.lease_id, None)

    def _retain_lease_until_task_stops(self, lease: BudgetLease) -> None:
        self._retained_lease_ids.add(lease.lease_id)
        self._retained_release_events.setdefault(lease.lease_id, asyncio.Event())

    def _replay(
        self,
        receipt: DelegationReceipt,
        output_model: type[BaseModel],
        *,
        max_result_bytes: int,
    ) -> BaseModel:
        if receipt.result_payload is None:
            raise DelegationFailure("delegation_result_unavailable")
        try:
            canonical_payload = _canonical_json(receipt.result_payload)
        except ValueError as exc:
            raise DelegationFailure("delegation_result_digest_mismatch") from exc
        if len(canonical_payload) > max_result_bytes:
            raise DelegationFailure(
                "delegated_agent_result_too_large",
                kind="result_too_large",
            )
        result_digest = f"sha256:{hashlib.sha256(canonical_payload).hexdigest()}"
        if result_digest != receipt.result_digest:
            raise DelegationFailure("delegation_result_digest_mismatch")
        try:
            return output_model.model_validate(receipt.result_payload)
        except ValidationError as exc:
            raise DelegationFailure("delegation_replay_validation_failed") from exc

    def _validate_claim(
        self,
        actual: DelegationReceipt,
        *,
        call: DelegationCall,
        delegation_id: str,
        child_run_id: str,
        request_digest: str,
        context_digest: str,
    ) -> None:
        expected = {
            "delegation_id": delegation_id,
            "execution_group_id": call.parent.execution_group_id,
            "root_run_id": call.parent.lineage.root_run_id,
            "parent_run_id": call.parent_run_id,
            "parent_tool_call_id": call.parent_tool_call_id,
            "caller_agent_ref": call.caller,
            "callee_agent_ref": call.callee,
            "child_run_id": child_run_id,
            "request_digest": request_digest,
            "context_digest": context_digest,
            "lease_id": self.id_factory.lease_id(
                delegation_id,
                attempt=actual.attempt,
            ),
        }
        for name, expected_value in expected.items():
            if getattr(actual, name) != expected_value:
                raise ReceiptStoreError("delegation_receipt_identity_mismatch")

    @staticmethod
    def _receipt_candidate(
        *,
        call: DelegationCall,
        delegation_id: str,
        child_run_id: str,
        request_digest: str,
        context_digest: str,
        attempt: int,
        owner_token: str,
        lease_id: str,
    ) -> DelegationReceipt:
        return DelegationReceipt(
            delegation_id=delegation_id,
            execution_group_id=call.parent.execution_group_id,
            root_run_id=call.parent.lineage.root_run_id,
            parent_run_id=call.parent_run_id,
            parent_tool_call_id=call.parent_tool_call_id,
            caller_agent_ref=call.caller,
            callee_agent_ref=call.callee,
            child_run_id=child_run_id,
            request_digest=request_digest,
            context_digest=context_digest,
            attempt=attempt,
            owner_token=owner_token,
            lease_id=lease_id,
        )

    def _context_digest(self, call: DelegationCall) -> str:
        return self.id_factory.context_digest(
            {
                "principal": call.parent.principal,
                "tenant": call.parent.tenant,
                "parent_session_id": call.parent.parent_session_id,
                "request_classification": call.request_classification,
                "session_strategy": call.session_strategy.value,
                "caller_agent_ref": str(call.caller),
                "callee_agent_ref": str(call.callee),
                "input_contract_digest": canonical_digest(
                    type(call.request).model_json_schema()
                ),
                "output_contract_digest": canonical_digest(
                    call.output_model.model_json_schema()
                ),
                "max_result_bytes": call.max_result_bytes,
            }
        )

    async def _claim_resume_attempt(
        self,
        receipt: DelegationReceipt,
        *,
        owner_token: str,
    ) -> DelegationReceipt:
        next_attempt = receipt.attempt + 1
        return await self.receipts.transition(
            receipt.delegation_id,
            expected={DelegationReceiptStatus.FAILED},
            status=DelegationReceiptStatus.RESERVED,
            finish_reason=None,
            retryable=False,
            resumable=False,
            error_code=None,
            attempt=next_attempt,
            owner_token=owner_token,
            lease_id=self.id_factory.lease_id(
                receipt.delegation_id,
                attempt=next_attempt,
            ),
        )

    async def _validate_existing_receipt_lease(
        self,
        *,
        call: DelegationCall,
        receipt: DelegationReceipt,
        lease: BudgetLease,
    ) -> None:
        try:
            status = await self.budget_ledger.inspect_lease(lease)
        except BudgetExceeded as exc:
            await self._handle_missing_or_invalid_lease(
                call=call,
                receipt=receipt,
                code=exc.code,
            )
            raise AssertionError("unreachable") from exc
        if status == LeaseStatus.ACTIVE and receipt.status in {
            DelegationReceiptStatus.COMPLETED,
            DelegationReceiptStatus.FAILED,
            DelegationReceiptStatus.CANCELLED,
        }:
            await self.budget_ledger.reconcile_completed_lease(lease)
            status = LeaseStatus.COMPLETED
        allowed = {
            DelegationReceiptStatus.RESERVED: {LeaseStatus.ACTIVE},
            DelegationReceiptStatus.RUNNING: {LeaseStatus.ACTIVE},
            DelegationReceiptStatus.COMPLETED: {LeaseStatus.COMPLETED},
            DelegationReceiptStatus.FAILED: {
                LeaseStatus.COMPLETED,
                LeaseStatus.RELEASED,
            },
            DelegationReceiptStatus.CANCELLED: {
                LeaseStatus.COMPLETED,
                LeaseStatus.RELEASED,
                LeaseStatus.CANCELLED,
            },
            DelegationReceiptStatus.MANUAL_REQUIRED: {
                LeaseStatus.ACTIVE,
                LeaseStatus.COMPLETED,
                LeaseStatus.RELEASED,
                LeaseStatus.CANCELLED,
            },
        }
        if status not in allowed[receipt.status]:
            await self._handle_missing_or_invalid_lease(
                call=call,
                receipt=receipt,
                code="delegation_lease_state_invalid",
            )

    async def _handle_missing_or_invalid_lease(
        self,
        *,
        call: DelegationCall,
        receipt: DelegationReceipt,
        code: str,
    ) -> None:
        if (
            receipt.status is DelegationReceiptStatus.RESERVED
            and datetime.now(timezone.utc) < call.parent.absolute_deadline
        ):
            raise DelegationFailure("delegation_in_progress")
        if receipt.status in {
            DelegationReceiptStatus.RESERVED,
            DelegationReceiptStatus.RUNNING,
        }:
            try:
                await self.receipts.mark_manual_required(
                    receipt,
                    error_code="delegation_lease_reconciliation_required",
                )
                await self._emit(
                    DelegationEventType.RECONCILIATION_REQUIRED,
                    call,
                    receipt.delegation_id,
                    child_run_id=receipt.child_run_id,
                    code="delegation_lease_reconciliation_required",
                )
            except ReceiptStoreError as exc:
                raise DelegationFailure("delegation_in_progress") from exc
        raise DelegationFailure(
            "delegation_reconciliation_required"
        ) from BudgetExceeded(code)

    async def _mark_reservation_uncertain(
        self,
        call: DelegationCall,
        receipt: DelegationReceipt,
    ) -> None:
        """Fence an owned receipt when lease reservation has no atomic commit."""

        try:
            await self.receipts.mark_manual_required(
                receipt,
                error_code="delegation_budget_reservation_uncertain",
            )
            await self._emit(
                DelegationEventType.RECONCILIATION_REQUIRED,
                call,
                receipt.delegation_id,
                child_run_id=receipt.child_run_id,
                code="delegation_budget_reservation_uncertain",
            )
        except Exception:
            # A durable owner remains fenced as RESERVED and will fail closed at
            # its absolute deadline if the receipt store is also unavailable.
            return

    def _validate_parent_run_identity(self, call: DelegationCall) -> None:
        lineage = call.parent.lineage
        expected_run_id = call.parent.current_run_id
        if expected_run_id is None and lineage.depth == 0:
            expected_run_id = lineage.root_run_id
        if expected_run_id is None:
            raise DelegationFailure("delegation_parent_run_missing")
        if call.parent_run_id != expected_run_id:
            raise DelegationFailure("delegation_parent_run_mismatch")

    @staticmethod
    def _validate_lease(
        lease: BudgetLease,
        *,
        call: DelegationCall,
        receipt: DelegationReceipt,
    ) -> None:
        if lease.lease_id != receipt.lease_id:
            raise DelegationFailure("delegation_budget_lease_mismatch")
        if lease.execution_group_id != call.parent.execution_group_id:
            raise DelegationFailure("delegation_budget_group_mismatch")
        if lease.callee != call.callee:
            raise DelegationFailure("delegation_budget_callee_mismatch")
        if lease.absolute_deadline != call.parent.absolute_deadline:
            raise DelegationFailure("delegation_budget_deadline_mismatch")

    async def _lock_for(self, delegation_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            return self._delegation_locks.setdefault(delegation_id, asyncio.Lock())

    async def _track_task(
        self,
        execution_group_id: str,
        task: asyncio.Task[DelegationOutcome],
        *,
        add: bool,
    ) -> None:
        async with self._active_guard:
            active = self._active_tasks.setdefault(execution_group_id, set())
            if add:
                active.add(task)
            else:
                active.discard(task)
                if not active:
                    self._active_tasks.pop(execution_group_id, None)

    async def _emit(
        self,
        event_type: DelegationEventType,
        call: DelegationCall,
        delegation_id: str,
        *,
        child_run_id: str | None = None,
        status: str | None = None,
        code: str | None = None,
    ) -> None:
        event = DelegationEvent(
            type=event_type,
            execution_group_id=call.parent.execution_group_id,
            delegation_id=delegation_id,
            lineage=call.parent.lineage,
            caller=call.caller,
            callee=call.callee,
            parent_tool_call_id=call.parent_tool_call_id,
            child_run_id=child_run_id,
            status=status,
            code=code,
        )
        try:
            await self.event_sink.publish_delegation(event)
        except Exception:
            pass
        if call.event_callback is not None:
            try:
                await call.event_callback(event)
            except Exception:
                pass

    async def _best_effort_release(self, lease: BudgetLease) -> None:
        if lease.lease_id in self._retained_lease_ids or lease.lease_id in (
            self._settled_lease_ids
        ):
            return
        try:
            await self.budget_ledger.release_lease(lease)
        except Exception:
            return
        self._settled_lease_ids.add(lease.lease_id)

    async def _best_effort_cleanup_checkpoint(
        self,
        *,
        endpoint: AgentEndpoint,
        context: DelegationContext,
        lease: BudgetLease,
        child_run_id: str,
    ) -> None:
        try:
            await asyncio.wait_for(
                self.invoker.cleanup_checkpoint(
                    endpoint,
                    context,
                    lease,
                    self.budget_ledger,
                    child_run_id=child_run_id,
                ),
                timeout=self.cancellation_grace_seconds,
            )
        except Exception:
            # The receipt already owns the terminal result. A retained checkpoint
            # is a bounded storage leak, while making completion fail is unsafe.
            return

    async def _complete_lease(
        self,
        lease: BudgetLease,
        *,
        usage: Mapping[str, int | float] | None = None,
    ) -> None:
        await self.budget_ledger.complete_lease(lease, usage=usage)
        self._settled_lease_ids.add(lease.lease_id)

    async def _reconcile_completed_lease(
        self,
        lease: BudgetLease,
        *,
        usage: Mapping[str, int | float] | None = None,
    ) -> None:
        await self.budget_ledger.reconcile_completed_lease(lease, usage=usage)
        self._settled_lease_ids.add(lease.lease_id)


def _outcome_failure_code(outcome: DelegationOutcome) -> str:
    if outcome.status is DelegationOutcomeStatus.CANCELLED:
        return "delegated_agent_cancelled"
    if outcome.status is DelegationOutcomeStatus.MANUAL_REQUIRED:
        return "delegation_reconciliation_required"
    return {
        "timeout": "delegated_agent_timeout",
        "max_model_turns": "delegated_agent_max_model_turns",
        "max_tool_calls": "delegated_agent_max_tool_calls",
        "no_progress": "delegated_agent_no_progress",
        "output_validation": "delegated_agent_output_validation_failed",
        "result_too_large": "delegated_agent_result_too_large",
    }.get(outcome.finish_reason or "", "delegated_agent_failed")


def _receipt_failure_code(receipt: DelegationReceipt) -> str:
    if receipt.status is DelegationReceiptStatus.CANCELLED:
        return "delegated_agent_cancelled"
    if receipt.status is DelegationReceiptStatus.MANUAL_REQUIRED:
        return "delegation_reconciliation_required"
    return {
        "timeout": "delegated_agent_timeout",
        "max_model_turns": "delegated_agent_max_model_turns",
        "max_tool_calls": "delegated_agent_max_tool_calls",
        "no_progress": "delegated_agent_no_progress",
        "output_validation": "delegated_agent_output_validation_failed",
        "result_too_large": "delegated_agent_result_too_large",
    }.get(receipt.finish_reason or "", "delegated_agent_failed")


def _validate_registry_resolution(
    requested: AgentRef,
    resolved: object,
    descriptor: object,
    *,
    expected_definition_fingerprint: str,
) -> None:
    """Cross-check the replaceable Registry's two projections atomically enough.

    The 0.6 Registry SPI exposes ``resolve`` and ``descriptor`` separately. A
    custom implementation therefore has to prove both views refer to the same
    pinned definition before authorization or endpoint invocation can occur.
    """

    if not isinstance(resolved, ResolvedAgentEndpoint) or not isinstance(
        descriptor,
        AgentDescriptor,
    ):
        raise DelegationRejected("delegation_registry_integrity_failed")
    if (
        resolved.ref != requested
        or descriptor.ref != requested
        or resolved.definition.ref != requested
        or resolved.definition_fingerprint != expected_definition_fingerprint
        or resolved.definition_fingerprint != descriptor.definition_fingerprint
        or resolved.status != descriptor.status
        or not resolved.status.runnable_in_production
        or descriptor.description != resolved.definition.description
        or descriptor.callable_by != resolved.definition.callable_by
        or descriptor.side_effect_level != resolved.definition.side_effect_level
        or descriptor.data_classification != resolved.definition.data_classification
        or descriptor.input_contract_digest
        != resolved.definition.semantic_digests.get("input_contract")
        or descriptor.output_contract_digest
        != resolved.definition.semantic_digests.get("output_contract")
        or descriptor.supports_async != resolved.endpoint.supports_async
        or descriptor.supports_stream != resolved.endpoint.supports_stream
    ):
        raise DelegationRejected("delegation_registry_integrity_failed")


def _validate_contract_digests(
    call: DelegationCall,
    descriptor: AgentDescriptor,
) -> None:
    expected_input = descriptor.input_contract_digest
    if expected_input is not None:
        actual_input = canonical_digest(type(call.request).model_json_schema())
        if actual_input != expected_input:
            raise DelegationFailure("delegation_input_contract_mismatch")
    expected_output = descriptor.output_contract_digest
    if expected_output is not None:
        actual_output = canonical_digest(call.output_model.model_json_schema())
        if actual_output != expected_output:
            raise DelegationFailure("delegation_output_contract_mismatch")


def _resolve_identity_factories(
    *,
    id_factory: DelegationIdFactory | None,
    session_factory: SessionKeyFactory | None,
    hmac_secret: bytes | None,
    durable: bool,
) -> tuple[DelegationIdFactory, SessionKeyFactory]:
    if hmac_secret is not None and not isinstance(hmac_secret, bytes):
        raise TypeError("hmac_secret must be bytes")
    if id_factory is not None and not isinstance(id_factory, DelegationIdFactory):
        raise TypeError("id_factory must be DelegationIdFactory")
    if session_factory is not None and not isinstance(
        session_factory, SessionKeyFactory
    ):
        raise TypeError("session_factory must be SessionKeyFactory")
    if id_factory is not None and session_factory is not None:
        if hmac_secret is not None:
            raise ValueError(
                "hmac_secret cannot be combined with both explicit factories"
            )
        return id_factory, session_factory
    if (id_factory is None) != (session_factory is None) and hmac_secret is None:
        raise ValueError(
            "hmac_secret is required when only one identity factory is supplied"
        )
    if hmac_secret is None:
        if durable:
            raise ValueError(
                "durable delegation stores require a stable hmac_secret or "
                "both identity factories"
            )
        hmac_secret = secrets.token_bytes(32)
    resolved_id_factory = id_factory or DelegationIdFactory(hmac_secret)
    resolved_session_factory = session_factory or SessionKeyFactory(hmac_secret)
    return resolved_id_factory, resolved_session_factory


def _default_tool_name(callee: AgentRef) -> str:
    base = re.sub(r"[^A-Za-z0-9_-]", "_", f"delegate_to_{callee.agent_id}")
    if len(base) <= 64:
        return base
    suffix = hashlib.sha256(str(callee).encode("utf-8")).hexdigest()[:8]
    return f"{base[:55]}_{suffix}"
