from __future__ import annotations

import re
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from moduagent.definitions.models import (
    AgentDefinition,
    AgentRef,
    DefinitionStatus,
    _coerce_status,
    _validated_callers,
    _validated_code,
    _validated_text,
    validate_lifecycle_transition,
)


_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class AgentRegistryError(ValueError):
    """Base error for deterministic registry contract violations."""


class AgentNotFoundError(AgentRegistryError, LookupError):
    """An exact AgentRef is not registered."""


class AgentDefinitionConflictError(AgentRegistryError):
    """A pinned AgentRef is already bound to another registration."""


class AgentDefinitionNotRunnableError(AgentRegistryError):
    """A definition lifecycle state cannot be used for an operational run."""


@dataclass(frozen=True, slots=True)
class AgentEndpoint:
    """Opaque runtime target plus non-secret invocation capabilities."""

    handler: object = field(repr=False, compare=False)
    kind: str = "local"
    supports_async: bool = True
    supports_stream: bool = False
    approved: bool = False

    def __post_init__(self) -> None:
        if self.handler is None:
            raise ValueError("endpoint handler cannot be None")
        object.__setattr__(self, "kind", _validated_code(self.kind, "endpoint kind"))
        if self.kind not in {"local", "remote"}:
            raise ValueError("endpoint kind must be 'local' or 'remote'")
        for field_name in ("supports_async", "supports_stream", "approved"):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"endpoint {field_name} must be a bool")


@dataclass(frozen=True, slots=True)
class AgentDescriptor:
    """Minimal caller-facing metadata with no prompt, endpoint, or credential."""

    ref: AgentRef
    description: str
    input_contract_digest: str | None
    output_contract_digest: str | None
    side_effect_level: str
    data_classification: str
    callable_by: frozenset[str]
    supports_async: bool
    supports_stream: bool
    definition_fingerprint: str
    status: DefinitionStatus

    def __post_init__(self) -> None:
        if not isinstance(self.ref, AgentRef):
            raise TypeError("descriptor ref must be an AgentRef")
        _validated_text(self.description, "descriptor description", max_length=2048)
        for field_name in ("input_contract_digest", "output_contract_digest"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None
            ):
                raise ValueError(f"{field_name} must be a canonical sha256 value")
        object.__setattr__(self, "callable_by", _validated_callers(self.callable_by))
        object.__setattr__(
            self,
            "side_effect_level",
            _validated_code(self.side_effect_level, "descriptor side_effect_level"),
        )
        object.__setattr__(
            self,
            "data_classification",
            _validated_code(
                self.data_classification,
                "descriptor data_classification",
            ),
        )
        if (
            type(self.supports_async) is not bool
            or type(self.supports_stream) is not bool
        ):
            raise TypeError("descriptor endpoint capabilities must be bools")
        if (
            not isinstance(
                self.definition_fingerprint,
                str,
            )
            or _SHA256_PATTERN.fullmatch(self.definition_fingerprint) is None
        ):
            raise ValueError("definition_fingerprint must be a canonical sha256 value")
        if not isinstance(self.status, DefinitionStatus):
            object.__setattr__(self, "status", _coerce_status(self.status))

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.ref.agent_id,
            "version": self.ref.version,
            "description": self.description,
            "input_contract_digest": self.input_contract_digest,
            "output_contract_digest": self.output_contract_digest,
            "side_effect_level": self.side_effect_level,
            "data_classification": self.data_classification,
            "callable_by": sorted(self.callable_by),
            "supports_async": self.supports_async,
            "supports_stream": self.supports_stream,
            "definition_fingerprint": self.definition_fingerprint,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class ResolvedAgentEndpoint:
    ref: AgentRef
    definition: AgentDefinition
    definition_fingerprint: str
    endpoint: AgentEndpoint = field(repr=False, compare=False)
    status: DefinitionStatus

    def __post_init__(self) -> None:
        if self.definition.ref != self.ref:
            raise ValueError("resolved definition does not match AgentRef")
        if self.definition.fingerprint != self.definition_fingerprint:
            raise ValueError("resolved definition fingerprint does not match")
        if not isinstance(self.endpoint, AgentEndpoint):
            raise TypeError("resolved endpoint must be an AgentEndpoint")
        if not isinstance(self.status, DefinitionStatus):
            object.__setattr__(self, "status", _coerce_status(self.status))


@runtime_checkable
class AgentRegistry(Protocol):
    def register(
        self,
        definition: AgentDefinition,
        endpoint: AgentEndpoint,
        *,
        status: DefinitionStatus = DefinitionStatus.DRAFT,
    ) -> None: ...

    def resolve(self, ref: AgentRef) -> ResolvedAgentEndpoint: ...

    def descriptor(self, ref: AgentRef) -> AgentDescriptor: ...


@dataclass(slots=True)
class _RegistryRecord:
    definition: AgentDefinition
    endpoint: AgentEndpoint
    status: DefinitionStatus


class InMemoryAgentRegistry:
    """Process-local registry with exact version pinning and lifecycle guards."""

    def __init__(self) -> None:
        self._records: dict[AgentRef, _RegistryRecord] = {}
        self._lock = RLock()

    def register(
        self,
        definition: AgentDefinition,
        endpoint: AgentEndpoint,
        *,
        status: DefinitionStatus = DefinitionStatus.DRAFT,
    ) -> None:
        if not isinstance(definition, AgentDefinition):
            raise TypeError("definition must be an AgentDefinition")
        if not isinstance(endpoint, AgentEndpoint):
            raise TypeError("endpoint must be an AgentEndpoint")
        resolved_status = _coerce_status(status)
        ref = definition.ref
        with self._lock:
            if ref in self._records:
                raise AgentDefinitionConflictError(
                    f"AgentRef is already registered: {ref}"
                )
            self._records[ref] = _RegistryRecord(
                definition=definition,
                endpoint=endpoint,
                status=resolved_status,
            )

    def transition(
        self,
        ref: AgentRef,
        target: DefinitionStatus,
        *,
        expected_fingerprint: str | None = None,
    ) -> AgentDescriptor:
        resolved_ref = _require_ref(ref)
        resolved_target = _coerce_status(target)
        with self._lock:
            record = self._require_record(resolved_ref)
            _require_expected_fingerprint(record, expected_fingerprint)
            validate_lifecycle_transition(record.status, resolved_target)
            record.status = resolved_target
            return _descriptor(record)

    def rebind_endpoint(
        self,
        ref: AgentRef,
        endpoint: AgentEndpoint,
        *,
        expected_fingerprint: str,
    ) -> ResolvedAgentEndpoint:
        resolved_ref = _require_ref(ref)
        if not isinstance(endpoint, AgentEndpoint):
            raise TypeError("endpoint must be an AgentEndpoint")
        with self._lock:
            record = self._require_record(resolved_ref)
            _require_expected_fingerprint(record, expected_fingerprint)
            record.endpoint = endpoint
            return _resolved(resolved_ref, record)

    def resolve(self, ref: AgentRef) -> ResolvedAgentEndpoint:
        """Resolve an exact approved/active version; there is no latest alias."""

        resolved_ref = _require_ref(ref)
        with self._lock:
            record = self._require_record(resolved_ref)
            if not record.status.runnable_in_production:
                raise AgentDefinitionNotRunnableError(
                    f"AgentRef is not approved or active: {resolved_ref}"
                )
            return _resolved(resolved_ref, record)

    def resolve_registered(self, ref: AgentRef) -> ResolvedAgentEndpoint:
        """Inspect a pinned draft/staged registration without making it runnable."""

        resolved_ref = _require_ref(ref)
        with self._lock:
            return _resolved(resolved_ref, self._require_record(resolved_ref))

    def descriptor(self, ref: AgentRef) -> AgentDescriptor:
        resolved_ref = _require_ref(ref)
        with self._lock:
            return _descriptor(self._require_record(resolved_ref))

    def descriptors(
        self, *, agent_id: str | None = None
    ) -> tuple[AgentDescriptor, ...]:
        if agent_id is not None:
            AgentRef(agent_id, "0.0.0")
        with self._lock:
            records = (
                (ref, record)
                for ref, record in self._records.items()
                if agent_id is None or ref.agent_id == agent_id
            )
            return tuple(
                _descriptor(record)
                for _, record in sorted(
                    records,
                    key=lambda item: (item[0].agent_id, item[0].version),
                )
            )

    def _require_record(self, ref: AgentRef) -> _RegistryRecord:
        try:
            return self._records[ref]
        except KeyError as exc:
            raise AgentNotFoundError(f"unknown AgentRef: {ref}") from exc

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


def _require_ref(value: AgentRef) -> AgentRef:
    if not isinstance(value, AgentRef):
        raise TypeError("registry operations require an exact AgentRef")
    return value


def _require_expected_fingerprint(
    record: _RegistryRecord,
    expected: str | None,
) -> None:
    if expected is not None and expected != record.definition.fingerprint:
        raise AgentDefinitionConflictError("definition fingerprint precondition failed")


def _descriptor(record: _RegistryRecord) -> AgentDescriptor:
    definition = record.definition
    return AgentDescriptor(
        ref=definition.ref,
        description=definition.description,
        input_contract_digest=definition.semantic_digests.get("input_contract"),
        output_contract_digest=definition.semantic_digests.get("output_contract"),
        side_effect_level=definition.side_effect_level,
        data_classification=definition.data_classification,
        callable_by=definition.callable_by,
        supports_async=record.endpoint.supports_async,
        supports_stream=record.endpoint.supports_stream,
        definition_fingerprint=definition.fingerprint,
        status=record.status,
    )


def _resolved(ref: AgentRef, record: _RegistryRecord) -> ResolvedAgentEndpoint:
    return ResolvedAgentEndpoint(
        ref=ref,
        definition=record.definition,
        definition_fingerprint=record.definition.fingerprint,
        endpoint=record.endpoint,
        status=record.status,
    )


__all__ = [
    "AgentDefinitionConflictError",
    "AgentDefinitionNotRunnableError",
    "AgentDescriptor",
    "AgentEndpoint",
    "AgentNotFoundError",
    "AgentRegistry",
    "AgentRegistryError",
    "InMemoryAgentRegistry",
    "ResolvedAgentEndpoint",
]
