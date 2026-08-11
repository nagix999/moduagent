from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from moduagent.definitions import (
    AgentEndpoint,
    AgentRegistry,
    AgentRegistryError,
    InMemoryAgentRegistry,
    ResolvedAgentEndpoint,
)

from .budget import BudgetLedger
from .models import BudgetLease, DelegationContext, DelegationOutcome


class DelegationEndpointError(Exception):
    """Stable failure raised by the private endpoint adapter."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@runtime_checkable
class DelegatedAgentEndpointHandler(Protocol):
    """Private runtime boundary implemented by an Agent endpoint handler."""

    async def _run_delegated(
        self,
        request: BaseModel,
        *,
        context: DelegationContext,
        budget: BudgetLease,
        budget_ledger: BudgetLedger,
        child_run_id: str,
    ) -> DelegationOutcome: ...


@runtime_checkable
class ReconciliableAgentEndpoint(Protocol):
    async def _reconcile_delegated(
        self,
        *,
        context: DelegationContext,
        budget: BudgetLease,
        budget_ledger: BudgetLedger,
        child_run_id: str,
    ) -> DelegationOutcome | None: ...


@runtime_checkable
class ResumableAgentEndpoint(Protocol):
    async def _resume_delegated(
        self,
        request: BaseModel,
        *,
        context: DelegationContext,
        budget: BudgetLease,
        budget_ledger: BudgetLedger,
        child_run_id: str,
    ) -> DelegationOutcome: ...


@runtime_checkable
class DelegatedCheckpointCleaner(Protocol):
    async def _cleanup_delegated_checkpoint(
        self,
        *,
        context: DelegationContext,
        budget: BudgetLease,
        budget_ledger: BudgetLedger,
        child_run_id: str,
    ) -> None: ...


class LocalAgentInvoker:
    """Invoke the handler inside a canonical definitions.AgentEndpoint."""

    @staticmethod
    def _handler(endpoint: AgentEndpoint) -> DelegatedAgentEndpointHandler:
        if not isinstance(endpoint, AgentEndpoint):
            raise TypeError("endpoint must be a definitions.AgentEndpoint")
        if endpoint.kind != "local":
            raise DelegationEndpointError("delegation_endpoint_not_local")
        if not endpoint.approved:
            raise DelegationEndpointError("delegation_endpoint_not_approved")
        if not endpoint.supports_async:
            raise DelegationEndpointError("delegation_endpoint_not_async")
        if not isinstance(endpoint.handler, DelegatedAgentEndpointHandler):
            raise DelegationEndpointError("delegation_endpoint_protocol_error")
        return endpoint.handler

    async def invoke(
        self,
        endpoint: AgentEndpoint,
        request: BaseModel,
        context: DelegationContext,
        budget: BudgetLease,
        budget_ledger: BudgetLedger,
        *,
        child_run_id: str,
    ) -> DelegationOutcome:
        self._validate_invocation_contract(
            context,
            budget,
            budget_ledger,
            child_run_id=child_run_id,
        )
        handler = self._handler(endpoint)
        outcome = await handler._run_delegated(
            request,
            context=context,
            budget=budget,
            budget_ledger=budget_ledger,
            child_run_id=child_run_id,
        )
        return self._validate_outcome(outcome, child_run_id)

    async def reconcile(
        self,
        endpoint: AgentEndpoint,
        context: DelegationContext,
        budget: BudgetLease,
        budget_ledger: BudgetLedger,
        *,
        child_run_id: str,
    ) -> DelegationOutcome | None:
        self._validate_invocation_contract(
            context,
            budget,
            budget_ledger,
            child_run_id=child_run_id,
        )
        handler = self._handler(endpoint)
        if not isinstance(handler, ReconciliableAgentEndpoint):
            return None
        outcome = await handler._reconcile_delegated(
            context=context,
            budget=budget,
            budget_ledger=budget_ledger,
            child_run_id=child_run_id,
        )
        if outcome is None:
            return None
        return self._validate_outcome(outcome, child_run_id)

    async def resume(
        self,
        endpoint: AgentEndpoint,
        request: BaseModel,
        context: DelegationContext,
        budget: BudgetLease,
        budget_ledger: BudgetLedger,
        *,
        child_run_id: str,
    ) -> DelegationOutcome:
        self._validate_invocation_contract(
            context,
            budget,
            budget_ledger,
            child_run_id=child_run_id,
        )
        handler = self._handler(endpoint)
        if not isinstance(handler, ResumableAgentEndpoint):
            raise DelegationEndpointError("delegation_not_resumable")
        outcome = await handler._resume_delegated(
            request,
            context=context,
            budget=budget,
            budget_ledger=budget_ledger,
            child_run_id=child_run_id,
        )
        return self._validate_outcome(outcome, child_run_id)

    async def cleanup_checkpoint(
        self,
        endpoint: AgentEndpoint,
        context: DelegationContext,
        budget: BudgetLease,
        budget_ledger: BudgetLedger,
        *,
        child_run_id: str,
    ) -> None:
        self._validate_invocation_contract(
            context,
            budget,
            budget_ledger,
            child_run_id=child_run_id,
        )
        handler = self._handler(endpoint)
        if not isinstance(handler, DelegatedCheckpointCleaner):
            return
        await handler._cleanup_delegated_checkpoint(
            context=context,
            budget=budget,
            budget_ledger=budget_ledger,
            child_run_id=child_run_id,
        )

    @staticmethod
    def _validate_outcome(
        outcome: object,
        child_run_id: str,
    ) -> DelegationOutcome:
        if not isinstance(outcome, DelegationOutcome):
            raise DelegationEndpointError("delegation_endpoint_protocol_error")
        if outcome.child_run_id != child_run_id:
            raise DelegationEndpointError("delegation_child_run_id_mismatch")
        return outcome

    @staticmethod
    def _validate_invocation_contract(
        context: DelegationContext,
        budget: BudgetLease,
        budget_ledger: BudgetLedger,
        *,
        child_run_id: str,
    ) -> None:
        """Reject mixed execution-group state before reaching an endpoint."""

        if not isinstance(context, DelegationContext):
            raise DelegationEndpointError("delegation_context_invalid")
        if not isinstance(budget, BudgetLease):
            raise DelegationEndpointError("delegation_budget_invalid")
        if not isinstance(budget_ledger, BudgetLedger):
            raise DelegationEndpointError("delegation_budget_ledger_invalid")
        if not isinstance(child_run_id, str) or not child_run_id:
            raise DelegationEndpointError("delegation_child_run_id_invalid")
        lineage = context.lineage
        if lineage.depth < 1 or lineage.delegation_id is None:
            raise DelegationEndpointError("delegation_lineage_invalid")
        if budget.execution_group_id != context.execution_group_id:
            raise DelegationEndpointError("delegation_budget_group_mismatch")
        if budget.callee != lineage.agent_ref:
            raise DelegationEndpointError("delegation_budget_callee_mismatch")
        if budget.absolute_deadline != context.absolute_deadline:
            raise DelegationEndpointError("delegation_budget_deadline_mismatch")


__all__ = [
    "AgentEndpoint",
    "AgentRegistry",
    "AgentRegistryError",
    "DelegatedAgentEndpointHandler",
    "DelegatedCheckpointCleaner",
    "DelegationEndpointError",
    "InMemoryAgentRegistry",
    "LocalAgentInvoker",
    "ReconciliableAgentEndpoint",
    "ResolvedAgentEndpoint",
    "ResumableAgentEndpoint",
]
