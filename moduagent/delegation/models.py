from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel

from moduagent.definitions import AgentRef


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,255}$")
_CLASSIFICATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a non-empty stable identifier")
    return value


def _classification(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _CLASSIFICATION.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a stable classification code")
    return value


def _utc(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class RunLineage:
    """Durable parent/child identity that never becomes prompt content."""

    root_run_id: str
    parent_run_id: str | None
    delegation_id: str | None
    parent_tool_call_id: str | None
    caller_agent_id: str | None
    agent_id: str
    agent_version: str
    agent_path: tuple[str, ...]
    depth: int

    def __post_init__(self) -> None:
        _identifier(self.root_run_id, "root_run_id")
        for value, name in (
            (self.parent_run_id, "parent_run_id"),
            (self.delegation_id, "delegation_id"),
            (self.parent_tool_call_id, "parent_tool_call_id"),
            (self.caller_agent_id, "caller_agent_id"),
        ):
            if value is not None:
                _identifier(value, name)
        _identifier(self.agent_id, "agent_id")
        current_agent = AgentRef(self.agent_id, self.agent_version)
        if type(self.depth) is not int or self.depth < 0:
            raise ValueError("depth must be a non-negative integer")
        path = tuple(self.agent_path)
        if not path:
            raise ValueError("agent_path cannot be empty")
        parsed_path: list[AgentRef] = []
        for entry in path:
            parsed_path.append(AgentRef.parse(entry))
        if len(set(path)) != len(path):
            raise ValueError("agent_path cannot contain a cycle")
        if len(path) != self.depth + 1:
            raise ValueError("agent_path length must equal depth + 1")
        if path[-1] != str(current_agent):
            raise ValueError("agent_path must end with the current Agent ref")
        if self.depth == 0:
            if any(
                value is not None
                for value in (
                    self.parent_run_id,
                    self.delegation_id,
                    self.parent_tool_call_id,
                    self.caller_agent_id,
                )
            ):
                raise ValueError("root lineage cannot contain parent fields")
        else:
            for value, name in (
                (self.parent_run_id, "parent_run_id"),
                (self.delegation_id, "delegation_id"),
                (self.parent_tool_call_id, "parent_tool_call_id"),
                (self.caller_agent_id, "caller_agent_id"),
            ):
                if value is None:
                    raise ValueError(f"child lineage requires {name}")
            if parsed_path[-2].agent_id != self.caller_agent_id:
                raise ValueError("caller_agent_id must match the prior Agent path")
        object.__setattr__(self, "agent_path", path)

    @classmethod
    def root(cls, *, run_id: str, agent: AgentRef) -> RunLineage:
        return cls(
            root_run_id=run_id,
            parent_run_id=None,
            delegation_id=None,
            parent_tool_call_id=None,
            caller_agent_id=None,
            agent_id=agent.agent_id,
            agent_version=agent.version,
            agent_path=(str(agent),),
            depth=0,
        )

    @property
    def agent_ref(self) -> AgentRef:
        return AgentRef(self.agent_id, self.agent_version)

    def child(
        self,
        *,
        child: AgentRef,
        parent_run_id: str,
        delegation_id: str,
        parent_tool_call_id: str,
    ) -> RunLineage:
        return RunLineage(
            root_run_id=self.root_run_id,
            parent_run_id=parent_run_id,
            delegation_id=delegation_id,
            parent_tool_call_id=parent_tool_call_id,
            caller_agent_id=self.agent_id,
            agent_id=child.agent_id,
            agent_version=child.version,
            agent_path=(*self.agent_path, str(child)),
            depth=self.depth + 1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_run_id": self.root_run_id,
            "parent_run_id": self.parent_run_id,
            "delegation_id": self.delegation_id,
            "parent_tool_call_id": self.parent_tool_call_id,
            "caller_agent_id": self.caller_agent_id,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "agent_path": list(self.agent_path),
            "depth": self.depth,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunLineage:
        if not isinstance(value, Mapping):
            raise TypeError("lineage must be an object")
        raw_path = value.get("agent_path")
        if isinstance(raw_path, (str, bytes)) or not isinstance(
            raw_path, (list, tuple)
        ):
            raise ValueError("agent_path must be an array")

        def optional_text(name: str) -> str | None:
            item = value.get(name)
            if item is None:
                return None
            if not isinstance(item, str):
                raise TypeError(f"{name} must be a string or null")
            return item

        return cls(
            root_run_id=_required_text(value, "root_run_id"),
            parent_run_id=optional_text("parent_run_id"),
            delegation_id=optional_text("delegation_id"),
            parent_tool_call_id=optional_text("parent_tool_call_id"),
            caller_agent_id=optional_text("caller_agent_id"),
            agent_id=_required_text(value, "agent_id"),
            agent_version=_required_text(value, "agent_version"),
            agent_path=tuple(raw_path),
            depth=_required_int(value, "depth"),
        )


@dataclass(frozen=True, slots=True)
class ExecutionGroupLimits:
    max_depth: int = 2
    max_delegations: int = 8
    max_parallel_delegations: int = 3
    max_delegations_per_agent: int = 3
    max_total_model_turns: int = 30
    max_total_tool_calls: int = 24
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        for name in (
            "max_depth",
            "max_delegations",
            "max_parallel_delegations",
            "max_delegations_per_agent",
            "max_total_model_turns",
            "max_total_tool_calls",
        ):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
        ):
            raise TypeError("timeout_seconds must be a finite number")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))


@dataclass(frozen=True, slots=True)
class BudgetLease:
    lease_id: str
    execution_group_id: str
    callee: AgentRef
    absolute_deadline: datetime

    def __post_init__(self) -> None:
        _identifier(self.lease_id, "lease_id")
        _identifier(self.execution_group_id, "execution_group_id")
        if not isinstance(self.callee, AgentRef):
            raise TypeError("callee must be an AgentRef")
        object.__setattr__(
            self,
            "absolute_deadline",
            _utc(self.absolute_deadline, "absolute_deadline"),
        )


@dataclass(frozen=True, slots=True)
class DelegationContext:
    lineage: RunLineage
    execution_group_id: str
    principal: str
    tenant: str
    absolute_deadline: datetime
    child_session_id: str
    request_classification: str = "internal"

    def __post_init__(self) -> None:
        if not isinstance(self.lineage, RunLineage):
            raise TypeError("lineage must be a RunLineage")
        _identifier(self.execution_group_id, "execution_group_id")
        _identifier(self.principal, "principal")
        _identifier(self.tenant, "tenant")
        _identifier(self.child_session_id, "child_session_id")
        _classification(self.request_classification, "request_classification")
        object.__setattr__(
            self,
            "absolute_deadline",
            _utc(self.absolute_deadline, "absolute_deadline"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the content-free durable delegation projection."""

        return {
            "lineage": self.lineage.to_dict(),
            "execution_group_id": self.execution_group_id,
            "principal": self.principal,
            "tenant": self.tenant,
            "absolute_deadline": self.absolute_deadline.isoformat(),
            "child_session_id": self.child_session_id,
            "request_classification": self.request_classification,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DelegationContext:
        if not isinstance(value, Mapping):
            raise TypeError("delegation context must be an object")
        raw_lineage = value.get("lineage")
        if not isinstance(raw_lineage, Mapping):
            raise ValueError("delegation context lineage must be an object")
        raw_deadline = _required_text(value, "absolute_deadline")
        try:
            deadline = datetime.fromisoformat(raw_deadline)
        except ValueError as exc:
            raise ValueError("absolute_deadline must be an ISO datetime") from exc
        return cls(
            lineage=RunLineage.from_dict(raw_lineage),
            execution_group_id=_required_text(value, "execution_group_id"),
            principal=_required_text(value, "principal"),
            tenant=_required_text(value, "tenant"),
            absolute_deadline=deadline,
            child_session_id=_required_text(value, "child_session_id"),
            request_classification=_required_text(
                value,
                "request_classification",
            ),
        )


class DelegationOutcomeStatus(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    MANUAL_REQUIRED = "manual_required"


@dataclass(frozen=True, slots=True)
class DelegationOutcome:
    status: DelegationOutcomeStatus
    child_run_id: str
    finish_reason: str | None = None
    output: BaseModel | None = None
    error_code: str | None = None
    retryable: bool = False
    resumable: bool = False
    usage: Mapping[str, int | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, DelegationOutcomeStatus):
            object.__setattr__(
                self, "status", DelegationOutcomeStatus(str(self.status))
            )
        _identifier(self.child_run_id, "child_run_id")
        if self.finish_reason is not None:
            _classification(self.finish_reason, "finish_reason")
        if self.error_code is not None:
            _classification(self.error_code, "error_code")
        if type(self.retryable) is not bool or type(self.resumable) is not bool:
            raise TypeError("retryable and resumable must be bools")
        if not isinstance(self.usage, Mapping):
            raise TypeError("usage must be a mapping")
        normalized: dict[str, int | float] = {}
        for key, value in self.usage.items():
            _classification(key, "usage key")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("usage values must be numbers")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError("usage values must be finite and non-negative")
            normalized[key] = value
        object.__setattr__(self, "usage", copy.deepcopy(normalized))
        if self.status is DelegationOutcomeStatus.COMPLETED:
            if self.output is None:
                raise ValueError("completed outcome requires output")
            if self.error_code is not None:
                raise ValueError("completed outcome cannot contain error_code")
        elif self.output is not None:
            raise ValueError("non-completed outcome cannot contain output")
        if self.status is DelegationOutcomeStatus.MANUAL_REQUIRED and self.retryable:
            raise ValueError(
                "manual_required outcome cannot be automatically retryable"
            )


@dataclass(frozen=True, slots=True)
class ParentDelegationContext:
    """Run-owned state consumed by a DelegatedAgentTool, never by the model."""

    lineage: RunLineage
    execution_group_id: str
    principal: str
    tenant: str
    parent_session_id: str
    absolute_deadline: datetime
    limits: ExecutionGroupLimits = field(default_factory=ExecutionGroupLimits)
    current_run_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.lineage, RunLineage):
            raise TypeError("lineage must be a RunLineage")
        _identifier(self.execution_group_id, "execution_group_id")
        _identifier(self.principal, "principal")
        _identifier(self.tenant, "tenant")
        _identifier(self.parent_session_id, "parent_session_id")
        if not isinstance(self.limits, ExecutionGroupLimits):
            raise TypeError("limits must be ExecutionGroupLimits")
        if self.current_run_id is not None:
            _identifier(self.current_run_id, "current_run_id")
        object.__setattr__(
            self,
            "absolute_deadline",
            _utc(self.absolute_deadline, "absolute_deadline"),
        )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _required_text(value: Mapping[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str):
        raise TypeError(f"{name} must be a string")
    return item


def _required_int(value: Mapping[str, Any], name: str) -> int:
    item = value.get(name)
    if type(item) is not int:
        raise TypeError(f"{name} must be an integer")
    return item
