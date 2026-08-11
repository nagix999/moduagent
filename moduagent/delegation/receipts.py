from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .models import AgentRef, _classification, _identifier


class DelegationReceiptStatus(str, Enum):
    RESERVED = "reserved"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    MANUAL_REQUIRED = "manual_required"


_TERMINAL_RECEIPT_STATES = frozenset(
    {
        DelegationReceiptStatus.COMPLETED,
        DelegationReceiptStatus.FAILED,
        DelegationReceiptStatus.CANCELLED,
        DelegationReceiptStatus.MANUAL_REQUIRED,
    }
)


@dataclass(frozen=True, slots=True)
class DelegationReceipt:
    delegation_id: str
    execution_group_id: str
    root_run_id: str
    parent_run_id: str
    parent_tool_call_id: str
    caller_agent_ref: AgentRef
    callee_agent_ref: AgentRef
    child_run_id: str
    request_digest: str
    context_digest: str
    attempt: int = 1
    owner_token: str | None = None
    status: DelegationReceiptStatus = DelegationReceiptStatus.RESERVED
    finish_reason: str | None = None
    retryable: bool = False
    resumable: bool = False
    result_digest: str | None = None
    result_payload: Mapping[str, Any] | None = None
    result_ref: str | None = None
    error_code: str | None = None
    lease_id: str | None = None
    revision: int = 1

    def __post_init__(self) -> None:
        for value, name in (
            (self.delegation_id, "delegation_id"),
            (self.execution_group_id, "execution_group_id"),
            (self.root_run_id, "root_run_id"),
            (self.parent_run_id, "parent_run_id"),
            (self.parent_tool_call_id, "parent_tool_call_id"),
            (self.child_run_id, "child_run_id"),
        ):
            _identifier(value, name)
        for ref, name in (
            (self.caller_agent_ref, "caller_agent_ref"),
            (self.callee_agent_ref, "callee_agent_ref"),
        ):
            if not isinstance(ref, AgentRef):
                raise TypeError(f"{name} must be an AgentRef")
        if not _is_sha256(self.request_digest):
            raise ValueError("request_digest must use sha256")
        if not isinstance(
            self.context_digest, str
        ) or not self.context_digest.startswith("ctx:"):
            raise ValueError("context_digest must use the opaque context namespace")
        _identifier(self.context_digest, "context_digest")
        if type(self.attempt) is not int or self.attempt < 1:
            raise ValueError("attempt must be a positive integer")
        if self.owner_token is not None:
            _identifier(self.owner_token, "owner_token")
        if not isinstance(self.status, DelegationReceiptStatus):
            object.__setattr__(
                self, "status", DelegationReceiptStatus(str(self.status))
            )
        if self.finish_reason is not None:
            _classification(self.finish_reason, "finish_reason")
        if self.error_code is not None:
            _classification(self.error_code, "error_code")
        if self.result_ref is not None:
            _identifier(self.result_ref, "result_ref")
        if self.lease_id is not None:
            _identifier(self.lease_id, "lease_id")
        for name in ("retryable", "resumable"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        if self.result_payload is not None:
            if not isinstance(self.result_payload, Mapping):
                raise TypeError("result_payload must be a mapping or None")
            object.__setattr__(
                self,
                "result_payload",
                copy.deepcopy(dict(self.result_payload)),
            )
        if self.result_digest is not None and not _is_sha256(self.result_digest):
            raise ValueError("result_digest must use sha256")
        if self.status is DelegationReceiptStatus.COMPLETED:
            if self.result_digest is None:
                raise ValueError("completed receipt requires result_digest")
            if self.result_payload is None and self.result_ref is None:
                raise ValueError("completed receipt requires a payload or result_ref")
            if self.error_code is not None:
                raise ValueError("completed receipt cannot contain error_code")
        elif self.result_payload is not None or self.result_digest is not None:
            raise ValueError("only completed receipts may contain a result")
        if (
            self.status
            in {
                DelegationReceiptStatus.RESERVED,
                DelegationReceiptStatus.RUNNING,
            }
            and self.owner_token is None
        ):
            raise ValueError("reserved and running receipts require owner_token")
        if self.status is DelegationReceiptStatus.MANUAL_REQUIRED and self.retryable:
            raise ValueError("manual_required receipt cannot be retryable")


@dataclass(frozen=True, slots=True)
class ReceiptClaim:
    receipt: DelegationReceipt
    created: bool


class ReceiptStoreError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@runtime_checkable
class DelegationReceiptStore(Protocol):
    durable: bool

    async def get(self, delegation_id: str) -> DelegationReceipt | None: ...

    async def claim(self, receipt: DelegationReceipt) -> ReceiptClaim: ...

    async def compare_and_swap(
        self,
        delegation_id: str,
        expected_revision: int,
        receipt: DelegationReceipt,
    ) -> bool: ...


class InMemoryDelegationReceiptStore:
    durable = False

    def __init__(self) -> None:
        self._receipts: dict[str, DelegationReceipt] = {}
        self._lock = asyncio.Lock()

    async def get(self, delegation_id: str) -> DelegationReceipt | None:
        async with self._lock:
            return copy.deepcopy(self._receipts.get(delegation_id))

    async def claim(self, receipt: DelegationReceipt) -> ReceiptClaim:
        if not isinstance(receipt, DelegationReceipt):
            raise TypeError("receipt must be a DelegationReceipt")
        if receipt.status is not DelegationReceiptStatus.RESERVED:
            raise ValueError("new receipt must be reserved")
        async with self._lock:
            current = self._receipts.get(receipt.delegation_id)
            if current is None:
                self._receipts[receipt.delegation_id] = copy.deepcopy(receipt)
                return ReceiptClaim(copy.deepcopy(receipt), True)
            return ReceiptClaim(copy.deepcopy(current), False)

    async def compare_and_swap(
        self,
        delegation_id: str,
        expected_revision: int,
        receipt: DelegationReceipt,
    ) -> bool:
        async with self._lock:
            current = self._receipts.get(delegation_id)
            if current is None or current.revision != expected_revision:
                return False
            if receipt.delegation_id != delegation_id:
                raise ValueError("replacement receipt has a different ID")
            if receipt.revision != expected_revision + 1:
                raise ValueError("replacement receipt must increment revision once")
            self._receipts[delegation_id] = copy.deepcopy(receipt)
            return True


class ReceiptAction(str, Enum):
    START = "start"
    REPLAY = "replay"
    RECONCILE = "reconcile"
    RESUME = "resume"
    TERMINAL_FAILURE = "terminal_failure"
    TERMINAL_CANCELLED = "terminal_cancelled"
    MANUAL_REQUIRED = "manual_required"


def receipt_action(
    receipt: DelegationReceipt,
    *,
    created: bool,
    allow_resume: bool,
) -> ReceiptAction:
    if created:
        return ReceiptAction.START
    if receipt.status is DelegationReceiptStatus.COMPLETED:
        return ReceiptAction.REPLAY
    if receipt.status in {
        DelegationReceiptStatus.RESERVED,
        DelegationReceiptStatus.RUNNING,
    }:
        return ReceiptAction.RECONCILE
    if receipt.status is DelegationReceiptStatus.FAILED:
        if receipt.resumable and allow_resume:
            return ReceiptAction.RESUME
        return ReceiptAction.TERMINAL_FAILURE
    if receipt.status is DelegationReceiptStatus.CANCELLED:
        return ReceiptAction.TERMINAL_CANCELLED
    return ReceiptAction.MANUAL_REQUIRED


class ReceiptManager:
    def __init__(
        self,
        store: DelegationReceiptStore,
        *,
        contention_attempts: int = 64,
    ) -> None:
        if not isinstance(store, DelegationReceiptStore):
            raise TypeError("store must implement DelegationReceiptStore")
        if type(contention_attempts) is not int or contention_attempts < 1:
            raise ValueError("contention_attempts must be at least 1")
        self.store = store
        self.contention_attempts = contention_attempts

    async def transition(
        self,
        delegation_id: str,
        *,
        expected: set[DelegationReceiptStatus] | frozenset[DelegationReceiptStatus],
        status: DelegationReceiptStatus,
        **changes: Any,
    ) -> DelegationReceipt:
        expected_states = frozenset(expected)
        for _ in range(self.contention_attempts):
            current = await self.store.get(delegation_id)
            if current is None:
                raise ReceiptStoreError("delegation_receipt_not_found")
            if current.status not in expected_states:
                if current.status is status:
                    for field_name, expected_value in changes.items():
                        if getattr(current, field_name) != expected_value:
                            raise ReceiptStoreError("delegation_receipt_state_conflict")
                    return current
                raise ReceiptStoreError("delegation_receipt_state_conflict")
            resumable_restart = (
                current.status is DelegationReceiptStatus.FAILED
                and current.resumable
                and status is DelegationReceiptStatus.RESERVED
            )
            if resumable_restart:
                if changes.get("attempt") != current.attempt + 1:
                    raise ReceiptStoreError("delegation_receipt_attempt_invalid")
                if not isinstance(changes.get("owner_token"), str):
                    raise ReceiptStoreError("delegation_receipt_owner_invalid")
                if not isinstance(changes.get("lease_id"), str):
                    raise ReceiptStoreError("delegation_receipt_lease_invalid")
            if current.status in _TERMINAL_RECEIPT_STATES and not resumable_restart:
                raise ReceiptStoreError("delegation_receipt_terminal")
            updated = replace(
                current,
                revision=current.revision + 1,
                status=status,
                **changes,
            )
            if await self.store.compare_and_swap(
                delegation_id,
                current.revision,
                updated,
            ):
                return updated
            await asyncio.sleep(0)
        raise ReceiptStoreError("delegation_receipt_store_contention")

    async def mark_manual_required(
        self,
        receipt: DelegationReceipt,
        *,
        error_code: str = "delegation_reconciliation_required",
    ) -> DelegationReceipt:
        return await self.transition(
            receipt.delegation_id,
            expected={
                DelegationReceiptStatus.RESERVED,
                DelegationReceiptStatus.RUNNING,
            },
            status=DelegationReceiptStatus.MANUAL_REQUIRED,
            retryable=False,
            resumable=False,
            error_code=error_code,
        )


class DelegationIdFactory:
    """Create replay-stable, non-reversible delegation and child run IDs."""

    def __init__(self, secret: bytes) -> None:
        if not isinstance(secret, bytes):
            raise TypeError("delegation HMAC secret must be bytes")
        if len(secret) < 32:
            raise ValueError("delegation HMAC secret must contain at least 32 bytes")
        self._secret = secret

    def request_digest(self, request: Mapping[str, Any]) -> str:
        return canonical_digest(request)

    def delegation_id(
        self,
        *,
        parent_run_id: str,
        parent_tool_call_id: str,
        request_digest: str,
    ) -> str:
        if not _is_sha256(request_digest):
            raise ValueError("request_digest must use sha256")
        digest = self._digest(
            {
                "namespace": "moduagent.delegation.id.v1",
                "parent_run_id": parent_run_id,
                "parent_tool_call_id": parent_tool_call_id,
                "request_digest": request_digest,
            }
        )
        return f"dlg:{digest}"

    def child_run_id(self, delegation_id: str) -> str:
        return f"run:{self._digest({'namespace': 'moduagent.child.run.v1', 'delegation_id': delegation_id})}"

    def context_digest(self, context: Mapping[str, Any]) -> str:
        """Bind replay identity to security/session and typed-contract scope."""

        return f"ctx:{self._digest({'namespace': 'moduagent.delegation.context.v1', 'context': context})}"

    def lease_id(self, delegation_id: str, *, attempt: int = 1) -> str:
        if type(attempt) is not int or attempt < 1:
            raise ValueError("delegation attempt must be a positive integer")
        return f"lease:{self._digest({'namespace': 'moduagent.budget.lease.v1', 'delegation_id': delegation_id, 'attempt': attempt})}"

    def _digest(self, value: Mapping[str, Any]) -> str:
        payload = _canonical_json(value)
        raw = hmac.new(self._secret, payload, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def canonical_digest(value: Mapping[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(value)).hexdigest()}"


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("delegation payload must be canonical JSON") from exc


def _is_sha256(value: str) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
