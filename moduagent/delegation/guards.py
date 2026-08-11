from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from moduagent.definitions import AgentDescriptor

from .models import (
    AgentRef,
    ExecutionGroupLimits,
    ParentDelegationContext,
    RunLineage,
    _classification,
)


class DelegationRejected(Exception):
    """Stable, payload-free pre-invocation rejection."""

    def __init__(self, code: str, *, unauthorized: bool = False) -> None:
        _classification(code, "delegation rejection code")
        if type(unauthorized) is not bool:
            raise TypeError("unauthorized must be a bool")
        super().__init__(code)
        self.code = code
        self.unauthorized = unauthorized


@dataclass(frozen=True, slots=True)
class DelegationDecision:
    allowed: bool
    reason_code: str = "delegation_allowed"

    def __post_init__(self) -> None:
        if type(self.allowed) is not bool:
            raise TypeError("allowed must be a bool")
        _classification(self.reason_code, "reason_code")


@runtime_checkable
class DelegationAuthorizer(Protocol):
    async def authorize(
        self,
        *,
        caller: AgentRef,
        callee: AgentRef,
        descriptor: AgentDescriptor,
        parent: ParentDelegationContext,
        request_classification: str,
    ) -> DelegationDecision: ...


class EdgeDelegationAuthorizer:
    """Deny-by-default authorization over application-owned topology edges."""

    def __init__(
        self,
        allowed_edges: set[tuple[AgentRef, AgentRef]]
        | frozenset[tuple[AgentRef, AgentRef]],
        *,
        allowed_tenants: Iterable[str] | None = None,
        allowed_principals: Iterable[str] | None = None,
    ) -> None:
        edges = frozenset(allowed_edges)
        if not all(
            isinstance(caller, AgentRef) and isinstance(callee, AgentRef)
            for caller, callee in edges
        ):
            raise TypeError("allowed_edges must contain AgentRef pairs")
        self.allowed_edges = edges
        self.allowed_tenants = _identity_grant(allowed_tenants, "allowed_tenants")
        self.allowed_principals = _identity_grant(
            allowed_principals,
            "allowed_principals",
        )

    async def authorize(
        self,
        *,
        caller: AgentRef,
        callee: AgentRef,
        descriptor: AgentDescriptor,
        parent: ParentDelegationContext,
        request_classification: str,
    ) -> DelegationDecision:
        if (caller, callee) not in self.allowed_edges:
            return DelegationDecision(False, "delegation_edge_denied")
        if (
            self.allowed_tenants is not None
            and parent.tenant not in self.allowed_tenants
        ):
            return DelegationDecision(False, "delegation_tenant_denied")
        if (
            self.allowed_principals is not None
            and parent.principal not in self.allowed_principals
        ):
            return DelegationDecision(False, "delegation_principal_denied")
        if descriptor.callable_by and not (
            "*" in descriptor.callable_by or caller.agent_id in descriptor.callable_by
        ):
            return DelegationDecision(False, "delegation_caller_denied")
        if request_classification != descriptor.data_classification:
            return DelegationDecision(False, "delegation_classification_denied")
        return DelegationDecision(True)


class DelegationPolicy:
    """Version-independent Agent-ID topology policy used by the public API."""

    def __init__(
        self,
        *,
        allowed_edges: Mapping[str, Iterable[str]],
        allowed_tenants: Iterable[str] | None = None,
        allowed_principals: Iterable[str] | None = None,
    ) -> None:
        if not isinstance(allowed_edges, Mapping):
            raise TypeError("allowed_edges must be a mapping of Agent IDs")
        normalized: dict[str, frozenset[str]] = {}
        for caller, raw_callees in allowed_edges.items():
            _classification(caller, "allowed_edges caller")
            if isinstance(raw_callees, (str, bytes)):
                raise TypeError("allowed_edges callees must be an iterable")
            callees = frozenset(raw_callees)
            for callee in callees:
                _classification(callee, "allowed_edges callee")
            normalized[caller] = callees
        self.allowed_edges = normalized
        self.allowed_tenants = _identity_grant(allowed_tenants, "allowed_tenants")
        self.allowed_principals = _identity_grant(
            allowed_principals,
            "allowed_principals",
        )

    async def authorize(
        self,
        *,
        caller: AgentRef,
        callee: AgentRef,
        descriptor: AgentDescriptor,
        parent: ParentDelegationContext,
        request_classification: str,
    ) -> DelegationDecision:
        if callee.agent_id not in self.allowed_edges.get(caller.agent_id, ()):
            return DelegationDecision(False, "delegation_edge_denied")
        if (
            self.allowed_tenants is not None
            and parent.tenant not in self.allowed_tenants
        ):
            return DelegationDecision(False, "delegation_tenant_denied")
        if (
            self.allowed_principals is not None
            and parent.principal not in self.allowed_principals
        ):
            return DelegationDecision(False, "delegation_principal_denied")
        if descriptor.callable_by and not (
            "*" in descriptor.callable_by or caller.agent_id in descriptor.callable_by
        ):
            return DelegationDecision(False, "delegation_caller_denied")
        if request_classification != descriptor.data_classification:
            return DelegationDecision(False, "delegation_classification_denied")
        return DelegationDecision(True)


def _identity_grant(
    values: Iterable[str] | None,
    field_name: str,
) -> frozenset[str] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of IDs")
    result = frozenset(values)
    if not result:
        raise ValueError(f"{field_name} cannot be empty; omit it for no restriction")
    for value in result:
        _classification(value, f"{field_name} item")
    return result


class CycleGuard:
    """Reject cyclic and over-depth paths before any child work starts."""

    def validate(
        self,
        *,
        lineage: RunLineage,
        callee: AgentRef,
        limits: ExecutionGroupLimits,
    ) -> None:
        if not isinstance(lineage, RunLineage):
            raise TypeError("lineage must be a RunLineage")
        if not isinstance(callee, AgentRef):
            raise TypeError("callee must be an AgentRef")
        if not isinstance(limits, ExecutionGroupLimits):
            raise TypeError("limits must be ExecutionGroupLimits")
        if str(callee) in lineage.agent_path:
            raise DelegationRejected("delegation_cycle_detected")
        if lineage.depth + 1 > limits.max_depth:
            raise DelegationRejected("delegation_depth_exceeded")


async def authorize_or_reject(
    authorizer: DelegationAuthorizer,
    *,
    caller: AgentRef,
    callee: AgentRef,
    descriptor: AgentDescriptor,
    parent: ParentDelegationContext,
    request_classification: str,
) -> None:
    try:
        decision = await authorizer.authorize(
            caller=caller,
            callee=callee,
            descriptor=descriptor,
            parent=parent,
            request_classification=request_classification,
        )
    except DelegationRejected:
        raise
    except Exception as exc:
        raise DelegationRejected(
            "delegation_authorizer_failed",
            unauthorized=True,
        ) from exc
    if not isinstance(decision, DelegationDecision):
        raise DelegationRejected(
            "delegation_authorizer_protocol_error",
            unauthorized=True,
        )
    if not decision.allowed:
        raise DelegationRejected(decision.reason_code, unauthorized=True)
