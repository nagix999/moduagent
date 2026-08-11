from __future__ import annotations

import asyncio
import copy
import math
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Protocol, TypeVar, runtime_checkable

from .models import AgentRef, BudgetLease, ExecutionGroupLimits, _identifier


class BudgetExceeded(Exception):
    """A stable, payload-free execution-group budget failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class LeaseStatus(str):
    ACTIVE = "active"
    COMPLETED = "completed"
    RELEASED = "released"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class StoredBudgetLease:
    lease_id: str
    callee_key: str
    status: str = LeaseStatus.ACTIVE

    def __post_init__(self) -> None:
        if not isinstance(self.lease_id, str) or not self.lease_id:
            raise ValueError("lease_id cannot be empty")
        if not isinstance(self.callee_key, str) or not self.callee_key:
            raise ValueError("callee_key cannot be empty")
        if self.status not in {
            LeaseStatus.ACTIVE,
            LeaseStatus.COMPLETED,
            LeaseStatus.RELEASED,
            LeaseStatus.CANCELLED,
        }:
            raise ValueError("stored lease status is invalid")


@dataclass(frozen=True, slots=True)
class ExecutionGroupBudgetState:
    execution_group_id: str
    limits: ExecutionGroupLimits
    absolute_deadline: datetime
    revision: int = 1
    delegation_count: int = 0
    model_turns: int = 0
    tool_calls: int = 0
    per_agent_delegations: Mapping[str, int] = field(default_factory=dict)
    leases: Mapping[str, StoredBudgetLease] = field(default_factory=dict)
    usage: Mapping[str, int | float] = field(default_factory=dict)
    cancelled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.execution_group_id, str) or not self.execution_group_id:
            raise ValueError("execution_group_id cannot be empty")
        if not isinstance(self.limits, ExecutionGroupLimits):
            raise TypeError("limits must be ExecutionGroupLimits")
        if not isinstance(self.absolute_deadline, datetime):
            raise TypeError("absolute_deadline must be a datetime")
        if self.absolute_deadline.tzinfo is None:
            raise ValueError("absolute_deadline must be timezone-aware")
        object.__setattr__(
            self,
            "absolute_deadline",
            self.absolute_deadline.astimezone(timezone.utc),
        )
        for name in ("revision", "delegation_count", "model_turns", "tool_calls"):
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name == "revision" else 0):
                raise ValueError(f"{name} is invalid")
        if type(self.cancelled) is not bool:
            raise TypeError("cancelled must be a bool")
        per_agent = copy.deepcopy(dict(self.per_agent_delegations))
        for key, value in per_agent.items():
            if not isinstance(key, str) or not key:
                raise ValueError("per-Agent budget keys cannot be empty")
            if type(value) is not int or value < 0:
                raise ValueError("per-Agent delegation counts cannot be negative")
        leases = copy.deepcopy(dict(self.leases))
        for key, lease in leases.items():
            if not isinstance(lease, StoredBudgetLease) or key != lease.lease_id:
                raise ValueError("stored budget leases are invalid")
        if sum(per_agent.values()) != self.delegation_count:
            raise ValueError("per-Agent counts must equal delegation_count")
        if len(leases) != self.delegation_count:
            raise ValueError("lease count must equal delegation_count")
        if self.model_turns > self.limits.max_total_model_turns:
            raise ValueError("model_turns exceed execution-group limits")
        if self.tool_calls > self.limits.max_total_tool_calls:
            raise ValueError("tool_calls exceed execution-group limits")
        if (
            sum(lease.status == LeaseStatus.ACTIVE for lease in leases.values())
            > self.limits.max_parallel_delegations
        ):
            raise ValueError("active leases exceed execution-group limits")
        usage = StoreBackedBudgetLedger._normalize_usage(self.usage)
        object.__setattr__(self, "per_agent_delegations", per_agent)
        object.__setattr__(self, "leases", leases)
        object.__setattr__(self, "usage", usage)

    @property
    def active_delegations(self) -> int:
        return sum(lease.status == LeaseStatus.ACTIVE for lease in self.leases.values())


@runtime_checkable
class BudgetStateStore(Protocol):
    """CAS boundary implemented by a durable database or an in-memory store."""

    durable: bool

    async def load(
        self, execution_group_id: str
    ) -> ExecutionGroupBudgetState | None: ...

    async def create(self, state: ExecutionGroupBudgetState) -> bool: ...

    async def compare_and_swap(
        self,
        execution_group_id: str,
        expected_revision: int,
        state: ExecutionGroupBudgetState,
    ) -> bool: ...


class InMemoryBudgetStateStore:
    durable = False

    def __init__(self) -> None:
        self._states: dict[str, ExecutionGroupBudgetState] = {}
        self._lock = asyncio.Lock()

    async def load(self, execution_group_id: str) -> ExecutionGroupBudgetState | None:
        async with self._lock:
            return copy.deepcopy(self._states.get(execution_group_id))

    async def create(self, state: ExecutionGroupBudgetState) -> bool:
        async with self._lock:
            if state.execution_group_id in self._states:
                return False
            self._states[state.execution_group_id] = copy.deepcopy(state)
            return True

    async def compare_and_swap(
        self,
        execution_group_id: str,
        expected_revision: int,
        state: ExecutionGroupBudgetState,
    ) -> bool:
        async with self._lock:
            current = self._states.get(execution_group_id)
            if current is None or current.revision != expected_revision:
                return False
            if state.execution_group_id != execution_group_id:
                raise ValueError("replacement state has a different group ID")
            if state.revision != expected_revision + 1:
                raise ValueError("replacement state must increment revision once")
            self._states[execution_group_id] = copy.deepcopy(state)
            return True


@runtime_checkable
class BudgetLedger(Protocol):
    async def load_group(
        self,
        execution_group_id: str,
    ) -> ExecutionGroupBudgetState | None: ...

    async def ensure_group(
        self,
        execution_group_id: str,
        limits: ExecutionGroupLimits,
        *,
        absolute_deadline: datetime | None = None,
    ) -> ExecutionGroupBudgetState: ...

    async def reserve_delegation(
        self,
        execution_group_id: str,
        callee: AgentRef,
        *,
        lease_id: str | None = None,
    ) -> BudgetLease: ...

    async def reserve_model_turn(
        self, execution_group_id: str, *, count: int = 1
    ) -> None: ...

    async def reserve_tool_call(
        self, execution_group_id: str, *, count: int = 1
    ) -> None: ...

    async def complete_lease(
        self,
        lease: BudgetLease,
        *,
        usage: Mapping[str, int | float] | None = None,
    ) -> None: ...

    async def reconcile_completed_lease(
        self,
        lease: BudgetLease,
        *,
        usage: Mapping[str, int | float] | None = None,
    ) -> bool: ...

    async def inspect_lease(self, lease: BudgetLease) -> str: ...

    async def release_lease(self, lease: BudgetLease) -> None: ...


T = TypeVar("T")


class StoreBackedBudgetLedger:
    """Atomic execution-group accounting over a CAS state store."""

    def __init__(
        self,
        store: BudgetStateStore,
        *,
        clock: Callable[[], datetime] | None = None,
        contention_attempts: int = 64,
        queue_poll_seconds: float = 0.01,
    ) -> None:
        if not isinstance(store, BudgetStateStore):
            raise TypeError("store must implement BudgetStateStore")
        if type(contention_attempts) is not int or contention_attempts < 1:
            raise ValueError("contention_attempts must be at least 1")
        if (
            isinstance(queue_poll_seconds, bool)
            or not isinstance(queue_poll_seconds, (int, float))
            or not math.isfinite(float(queue_poll_seconds))
            or queue_poll_seconds <= 0
        ):
            raise ValueError("queue_poll_seconds must be positive and finite")
        self.store = store
        self.durable = store.durable
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.contention_attempts = contention_attempts
        self.queue_poll_seconds = float(queue_poll_seconds)

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise TypeError("budget clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)

    async def load_group(
        self,
        execution_group_id: str,
    ) -> ExecutionGroupBudgetState | None:
        if not isinstance(execution_group_id, str) or not execution_group_id:
            raise ValueError("execution_group_id cannot be empty")
        return await self.store.load(execution_group_id)

    async def ensure_group(
        self,
        execution_group_id: str,
        limits: ExecutionGroupLimits,
        *,
        absolute_deadline: datetime | None = None,
    ) -> ExecutionGroupBudgetState:
        if not isinstance(execution_group_id, str) or not execution_group_id:
            raise ValueError("execution_group_id cannot be empty")
        if not isinstance(limits, ExecutionGroupLimits):
            raise TypeError("limits must be ExecutionGroupLimits")
        existing = await self.store.load(execution_group_id)
        if existing is not None:
            if existing.limits != limits:
                raise BudgetExceeded("execution_group_contract_mismatch")
            if absolute_deadline is not None:
                requested_deadline = self._as_deadline(absolute_deadline)
                if existing.absolute_deadline != requested_deadline:
                    raise BudgetExceeded("execution_group_contract_mismatch")
            return existing
        deadline = absolute_deadline or (
            self._now() + timedelta(seconds=limits.timeout_seconds)
        )
        deadline = self._as_deadline(deadline)
        candidate = ExecutionGroupBudgetState(
            execution_group_id=execution_group_id,
            limits=limits,
            absolute_deadline=deadline,
        )
        if await self.store.create(candidate):
            return candidate
        existing = await self.store.load(execution_group_id)
        if existing is None:
            raise BudgetExceeded("budget_store_contention")
        if existing.limits != limits or (
            absolute_deadline is not None and existing.absolute_deadline != deadline
        ):
            raise BudgetExceeded("execution_group_contract_mismatch")
        return existing

    async def reserve_delegation(
        self,
        execution_group_id: str,
        callee: AgentRef,
        *,
        lease_id: str | None = None,
    ) -> BudgetLease:
        if not isinstance(callee, AgentRef):
            raise TypeError("callee must be an AgentRef")
        resolved_lease_id = (
            f"lease:{uuid.uuid4().hex}"
            if lease_id is None
            else _identifier(lease_id, "lease_id")
        )
        while True:
            state = await self._require_state(execution_group_id)
            self._assert_live(state)
            adopted = self._adopt_active_lease(
                state,
                resolved_lease_id,
                callee,
            )
            if adopted is not None:
                return adopted
            if state.delegation_count >= state.limits.max_delegations:
                raise BudgetExceeded("delegation_count_exceeded")
            per_agent = state.per_agent_delegations.get(str(callee), 0)
            if per_agent >= state.limits.max_delegations_per_agent:
                raise BudgetExceeded("delegation_per_agent_exceeded")
            if state.active_delegations >= state.limits.max_parallel_delegations:
                remaining = (state.absolute_deadline - self._now()).total_seconds()
                if remaining <= 0:
                    raise BudgetExceeded("execution_group_timeout")
                await asyncio.sleep(min(self.queue_poll_seconds, remaining))
                continue

            def reserve(
                current: ExecutionGroupBudgetState,
            ) -> tuple[ExecutionGroupBudgetState, BudgetLease]:
                self._assert_live(current)
                existing = self._adopt_active_lease(
                    current,
                    resolved_lease_id,
                    callee,
                )
                if existing is not None:
                    return current, existing
                if current.delegation_count >= current.limits.max_delegations:
                    raise BudgetExceeded("delegation_count_exceeded")
                count = current.per_agent_delegations.get(str(callee), 0)
                if count >= current.limits.max_delegations_per_agent:
                    raise BudgetExceeded("delegation_per_agent_exceeded")
                if (
                    current.active_delegations
                    >= current.limits.max_parallel_delegations
                ):
                    raise _RetryReservation
                per_agent_counts = dict(current.per_agent_delegations)
                per_agent_counts[str(callee)] = count + 1
                leases = dict(current.leases)
                leases[resolved_lease_id] = StoredBudgetLease(
                    resolved_lease_id,
                    str(callee),
                )
                updated = replace(
                    current,
                    revision=current.revision + 1,
                    delegation_count=current.delegation_count + 1,
                    per_agent_delegations=per_agent_counts,
                    leases=leases,
                )
                return updated, BudgetLease(
                    resolved_lease_id,
                    execution_group_id,
                    callee,
                    current.absolute_deadline,
                )

            try:
                return await self._mutate(execution_group_id, reserve)
            except _RetryReservation:
                continue

    @staticmethod
    def _adopt_active_lease(
        state: ExecutionGroupBudgetState,
        lease_id: str,
        callee: AgentRef,
    ) -> BudgetLease | None:
        record = state.leases.get(lease_id)
        if record is None:
            return None
        if record.callee_key != str(callee):
            raise BudgetExceeded("budget_lease_identity_mismatch")
        if record.status != LeaseStatus.ACTIVE:
            raise BudgetExceeded("budget_lease_not_active")
        return BudgetLease(
            lease_id,
            state.execution_group_id,
            callee,
            state.absolute_deadline,
        )

    async def reserve_model_turn(
        self,
        execution_group_id: str,
        *,
        count: int = 1,
    ) -> None:
        self._validate_count(count)

        def reserve(
            current: ExecutionGroupBudgetState,
        ) -> tuple[ExecutionGroupBudgetState, None]:
            self._assert_live(current)
            if current.model_turns + count > current.limits.max_total_model_turns:
                raise BudgetExceeded("execution_group_model_turns_exceeded")
            return (
                replace(
                    current,
                    revision=current.revision + 1,
                    model_turns=current.model_turns + count,
                ),
                None,
            )

        await self._mutate(execution_group_id, reserve)

    async def reserve_tool_call(
        self,
        execution_group_id: str,
        *,
        count: int = 1,
    ) -> None:
        self._validate_count(count)

        def reserve(
            current: ExecutionGroupBudgetState,
        ) -> tuple[ExecutionGroupBudgetState, None]:
            self._assert_live(current)
            if current.tool_calls + count > current.limits.max_total_tool_calls:
                raise BudgetExceeded("execution_group_tool_calls_exceeded")
            return (
                replace(
                    current,
                    revision=current.revision + 1,
                    tool_calls=current.tool_calls + count,
                ),
                None,
            )

        await self._mutate(execution_group_id, reserve)

    async def complete_lease(
        self,
        lease: BudgetLease,
        *,
        usage: Mapping[str, int | float] | None = None,
    ) -> None:
        await self._complete_lease_state(lease, usage=usage)

    async def reconcile_completed_lease(
        self,
        lease: BudgetLease,
        *,
        usage: Mapping[str, int | float] | None = None,
    ) -> bool:
        """Close a crash-window lease, returning false when already completed."""

        return await self._complete_lease_state(lease, usage=usage)

    async def inspect_lease(self, lease: BudgetLease) -> str:
        """Validate persisted lease identity and return its durable status."""

        state = await self._require_state(lease.execution_group_id)
        if lease.absolute_deadline != state.absolute_deadline:
            raise BudgetExceeded("budget_lease_deadline_mismatch")
        return self._lease_record(state, lease).status

    async def _complete_lease_state(
        self,
        lease: BudgetLease,
        *,
        usage: Mapping[str, int | float] | None = None,
    ) -> bool:
        normalized_usage = self._normalize_usage(usage or {})

        def complete(
            current: ExecutionGroupBudgetState,
        ) -> tuple[ExecutionGroupBudgetState, bool]:
            record = self._lease_record(current, lease)
            if record.status == LeaseStatus.COMPLETED:
                return current, False
            if record.status != LeaseStatus.ACTIVE:
                raise BudgetExceeded("budget_lease_not_active")
            leases = dict(current.leases)
            leases[lease.lease_id] = replace(record, status=LeaseStatus.COMPLETED)
            aggregate = dict(current.usage)
            for key, value in normalized_usage.items():
                aggregate[key] = aggregate.get(key, 0) + value
            return (
                replace(
                    current,
                    revision=current.revision + 1,
                    leases=leases,
                    usage=aggregate,
                ),
                True,
            )

        return await self._mutate(lease.execution_group_id, complete)

    async def release_lease(self, lease: BudgetLease) -> None:
        def release(
            current: ExecutionGroupBudgetState,
        ) -> tuple[ExecutionGroupBudgetState, None]:
            record = self._lease_record(current, lease)
            if record.status in {
                LeaseStatus.RELEASED,
                LeaseStatus.CANCELLED,
                LeaseStatus.COMPLETED,
            }:
                return current, None
            leases = dict(current.leases)
            leases[lease.lease_id] = replace(record, status=LeaseStatus.RELEASED)
            return (
                replace(current, revision=current.revision + 1, leases=leases),
                None,
            )

        await self._mutate(lease.execution_group_id, release)

    async def cancel_group(self, execution_group_id: str) -> None:
        def cancel(
            current: ExecutionGroupBudgetState,
        ) -> tuple[ExecutionGroupBudgetState, None]:
            leases = {
                key: (
                    replace(value, status=LeaseStatus.CANCELLED)
                    if value.status == LeaseStatus.ACTIVE
                    else value
                )
                for key, value in current.leases.items()
            }
            return (
                replace(
                    current,
                    revision=current.revision + 1,
                    cancelled=True,
                    leases=leases,
                ),
                None,
            )

        await self._mutate(execution_group_id, cancel)

    async def snapshot(self, execution_group_id: str) -> ExecutionGroupBudgetState:
        return await self._require_state(execution_group_id)

    async def _require_state(
        self, execution_group_id: str
    ) -> ExecutionGroupBudgetState:
        state = await self.store.load(execution_group_id)
        if state is None:
            raise BudgetExceeded("execution_group_not_initialized")
        return state

    async def _mutate(
        self,
        execution_group_id: str,
        operation: Callable[
            [ExecutionGroupBudgetState], tuple[ExecutionGroupBudgetState, T]
        ],
    ) -> T:
        for _ in range(self.contention_attempts):
            current = await self._require_state(execution_group_id)
            updated, result = operation(current)
            if updated is current:
                return result
            if await self.store.compare_and_swap(
                execution_group_id,
                current.revision,
                updated,
            ):
                return result
            await asyncio.sleep(0)
        raise BudgetExceeded("budget_store_contention")

    def _assert_live(self, state: ExecutionGroupBudgetState) -> None:
        if state.cancelled:
            raise BudgetExceeded("execution_group_cancelled")
        if self._now() >= state.absolute_deadline:
            raise BudgetExceeded("execution_group_timeout")

    @staticmethod
    def _as_deadline(value: datetime) -> datetime:
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ValueError("absolute_deadline must be timezone-aware")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _validate_count(count: int) -> None:
        if type(count) is not int or count < 1:
            raise ValueError("reservation count must be a positive integer")

    @staticmethod
    def _normalize_usage(
        usage: Mapping[str, int | float],
    ) -> dict[str, int | float]:
        if not isinstance(usage, Mapping):
            raise TypeError("usage must be a mapping")
        normalized: dict[str, int | float] = {}
        for key, value in usage.items():
            if not isinstance(key, str) or not key:
                raise ValueError("usage keys cannot be empty")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("usage values must be numbers")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError("usage values must be finite and non-negative")
            normalized[key] = value
        return normalized

    @staticmethod
    def _lease_record(
        state: ExecutionGroupBudgetState,
        lease: BudgetLease,
    ) -> StoredBudgetLease:
        if not isinstance(lease, BudgetLease):
            raise TypeError("lease must be a BudgetLease")
        if lease.execution_group_id != state.execution_group_id:
            raise BudgetExceeded("budget_lease_group_mismatch")
        record = state.leases.get(lease.lease_id)
        if record is None or record.callee_key != str(lease.callee):
            raise BudgetExceeded("budget_lease_not_found")
        return record


class InMemoryBudgetLedger(StoreBackedBudgetLedger):
    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        queue_poll_seconds: float = 0.01,
    ) -> None:
        super().__init__(
            InMemoryBudgetStateStore(),
            clock=clock,
            queue_poll_seconds=queue_poll_seconds,
        )


class DurableBudgetLedger(StoreBackedBudgetLedger):
    """Ledger that refuses non-durable stores to prevent false guarantees."""

    def __init__(self, store: BudgetStateStore, **kwargs: object) -> None:
        if not getattr(store, "durable", False):
            raise ValueError("DurableBudgetLedger requires a durable CAS store")
        super().__init__(store, **kwargs)


class _RetryReservation(Exception):
    pass
