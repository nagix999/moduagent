from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from moduagent.definitions.registry import AgentRegistry
from moduagent.observability import DiagnosticSink, EventSink
from moduagent.persistence import CheckpointStore, ConversationStore
from moduagent.skills import SkillRegistry
from moduagent.tools import ToolAuthorizer, ToolRegistry


@runtime_checkable
class ModelRouter(Protocol):
    """Resolve a definition's logical model route without embedding secrets."""

    def resolve(self, model_route: str) -> Any: ...


@runtime_checkable
class SecretResolver(Protocol):
    """Resolve an opaque secret reference at the runtime boundary."""

    def resolve(self, secret_ref: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class RuntimeAttestation:
    """Application-owned deployment facts consumed by the Test profile.

    The digest binds the non-secret source reference and both declared facts.
    It detects accidental substitution; it is not a signature and does not
    make untrusted application code authoritative.
    """

    source_ref: str
    fingerprint: str
    external_io_enabled: bool
    deterministic_components: bool

    @classmethod
    def create(
        cls,
        *,
        source_ref: str,
        external_io_enabled: bool,
        deterministic_components: bool,
    ) -> "RuntimeAttestation":
        fingerprint = cls.fingerprint_for(
            source_ref=source_ref,
            external_io_enabled=external_io_enabled,
            deterministic_components=deterministic_components,
        )
        return cls(
            source_ref=source_ref,
            fingerprint=fingerprint,
            external_io_enabled=external_io_enabled,
            deterministic_components=deterministic_components,
        )

    @staticmethod
    def fingerprint_for(
        *,
        source_ref: str,
        external_io_enabled: bool,
        deterministic_components: bool,
    ) -> str:
        payload = {
            "schema_version": 1,
            "source_ref": source_ref,
            "external_io_enabled": external_io_enabled,
            "deterministic_components": deterministic_components,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_ref, str)
            or not self.source_ref.strip()
            or "?" in self.source_ref
            or "@" in self.source_ref
            or len(self.source_ref) > 512
        ):
            raise ValueError("source_ref must be a non-secret reference")
        if (
            not isinstance(self.fingerprint, str)
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                self.fingerprint,
            )
            is None
        ):
            raise ValueError("attestation fingerprint must use sha256")
        for field_name in ("external_io_enabled", "deterministic_components"):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a bool")
        expected = self.fingerprint_for(
            source_ref=self.source_ref,
            external_io_enabled=self.external_io_enabled,
            deterministic_components=self.deterministic_components,
        )
        if self.fingerprint != expected:
            raise ValueError("attestation fingerprint does not match its facts")


@dataclass(frozen=True, slots=True, repr=False)
class RuntimeBindings:
    """Replaceable environment objects excluded from definition fingerprints.

    Fields are optional so deployment assembly can be incremental. Runtime
    profiles decide which bindings are mandatory for a given environment.
    Unknown future delegation store/authorizer implementations are accepted by
    protocol shape rather than coupled to concrete adapters.
    """

    model_router: ModelRouter | None = None
    tool_registry: ToolRegistry | None = None
    skill_registry: SkillRegistry | None = None
    conversation_store: ConversationStore | None = None
    checkpoint_store: CheckpointStore | None = None
    delegation_receipt_store: object | None = None
    execution_group_store: object | None = None
    tool_authorizer: ToolAuthorizer | None = None
    delegation_authorizer: object | None = None
    secret_resolver: SecretResolver | None = None
    event_sink: EventSink | None = None
    diagnostic_sink: DiagnosticSink | None = None
    agent_registry: AgentRegistry | None = None
    tenant_context_provider: object | None = None
    principal_context_provider: object | None = None
    # Signed/pinned deployment facts used by TestProfile. The profile treats a
    # missing attestation conservatively instead of trusting caller booleans.
    runtime_attestation: RuntimeAttestation | None = None

    def __post_init__(self) -> None:
        if self.runtime_attestation is not None and not isinstance(
            self.runtime_attestation,
            RuntimeAttestation,
        ):
            raise TypeError("runtime_attestation must be RuntimeAttestation or None")

    def present_components(self) -> frozenset[str]:
        return frozenset(
            field_name
            for field_name in self.__dataclass_fields__
            if getattr(self, field_name) is not None
        )

    def require(self, *field_names: str) -> None:
        unknown = tuple(
            field_name
            for field_name in field_names
            if field_name not in self.__dataclass_fields__
        )
        if unknown:
            raise ValueError(f"unknown runtime binding: {', '.join(unknown)}")
        missing = tuple(
            field_name
            for field_name in field_names
            if getattr(self, field_name) is None
        )
        if missing:
            raise ValueError(f"missing runtime bindings: {', '.join(missing)}")

    def __repr__(self) -> str:
        components = ", ".join(sorted(self.present_components()))
        return f"RuntimeBindings(present=[{components}])"


__all__ = [
    "ModelRouter",
    "RuntimeAttestation",
    "RuntimeBindings",
    "SecretResolver",
]
