from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from .models import AgentRef, RunLineage, _classification


class DelegationEventType(str, Enum):
    REQUESTED = "delegation_requested"
    AUTHORIZED = "delegation_authorized"
    REJECTED = "delegation_rejected"
    STARTED = "delegation_started"
    RESUMED = "delegation_resumed"
    COMPLETED = "delegation_completed"
    FAILED = "delegation_failed"
    RECONCILIATION_REQUIRED = "delegation_reconciliation_required"


@dataclass(frozen=True, slots=True)
class DelegationEvent:
    """Bounded event projection; request, result, and exception data are absent."""

    type: DelegationEventType
    execution_group_id: str
    delegation_id: str
    lineage: RunLineage
    caller: AgentRef
    callee: AgentRef
    parent_tool_call_id: str
    child_run_id: str | None = None
    status: str | None = None
    code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.type, DelegationEventType):
            object.__setattr__(self, "type", DelegationEventType(str(self.type)))
        if not isinstance(self.lineage, RunLineage):
            raise TypeError("lineage must be a RunLineage")
        for ref, name in ((self.caller, "caller"), (self.callee, "callee")):
            if not isinstance(ref, AgentRef):
                raise TypeError(f"{name} must be an AgentRef")
        for value, name in (
            (self.execution_group_id, "execution_group_id"),
            (self.delegation_id, "delegation_id"),
            (self.parent_tool_call_id, "parent_tool_call_id"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} cannot be empty")
        if self.status is not None:
            _classification(self.status, "delegation event status")
        if self.code is not None:
            _classification(self.code, "delegation event code")

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type.value,
            "execution_group_id": self.execution_group_id,
            "delegation_id": self.delegation_id,
            "root_run_id": self.lineage.root_run_id,
            "parent_run_id": self.lineage.parent_run_id,
            "parent_tool_call_id": self.parent_tool_call_id,
            "caller_agent_id": self.caller.agent_id,
            "callee_agent_id": self.callee.agent_id,
            "child_run_id": self.child_run_id,
            "status": self.status,
            "code": self.code,
        }


@runtime_checkable
class DelegationEventSink(Protocol):
    async def publish_delegation(self, event: DelegationEvent) -> None: ...


class NoopDelegationEventSink:
    async def publish_delegation(self, event: DelegationEvent) -> None:
        del event
