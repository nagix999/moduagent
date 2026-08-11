from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

from moduagent.config import RunLimits


_AGENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SEMVER_PATTERN = re.compile(
    r"^(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)\."
    r"(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
_SIDE_EFFECT_LEVELS = frozenset({"none", "read", "advisory", "write"})
_STABLE_CODE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_REFERENCE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,511}$")
_DIGEST_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_SHA256_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


REQUIRED_SEMANTIC_DIGEST_KEYS = frozenset(
    {
        "authorization_policy",
        "input_contract",
        "instructions",
        "memory_policy",
        "model_capabilities",
        "output_contract",
        "skills",
        "tools",
    }
)


@dataclass(frozen=True, slots=True)
class AgentRef:
    """An exact, version-pinned Agent identity."""

    agent_id: str
    version: str

    def __post_init__(self) -> None:
        _validate_agent_id(self.agent_id)
        _validate_version(self.version)

    @classmethod
    def parse(cls, value: str) -> AgentRef:
        if not isinstance(value, str):
            raise TypeError("AgentRef text must be a string")
        agent_id, separator, version = value.rpartition("@")
        if not separator or not agent_id or not version:
            raise ValueError("AgentRef text must use '<agent_id>@<version>'")
        return cls(agent_id, version)

    def __str__(self) -> str:
        return f"{self.agent_id}@{self.version}"


class DefinitionStatus(str, Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    APPROVED = "approved"
    STAGED = "staged"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"

    @property
    def runnable_in_production(self) -> bool:
        return self in {self.APPROVED, self.ACTIVE}


_LIFECYCLE_TRANSITIONS: Mapping[DefinitionStatus, DefinitionStatus] = {
    DefinitionStatus.DRAFT: DefinitionStatus.REVIEWED,
    DefinitionStatus.REVIEWED: DefinitionStatus.APPROVED,
    DefinitionStatus.APPROVED: DefinitionStatus.STAGED,
    DefinitionStatus.STAGED: DefinitionStatus.ACTIVE,
    DefinitionStatus.ACTIVE: DefinitionStatus.DEPRECATED,
    DefinitionStatus.DEPRECATED: DefinitionStatus.RETIRED,
}


def validate_lifecycle_transition(
    current: DefinitionStatus,
    target: DefinitionStatus,
) -> None:
    current = _coerce_status(current)
    target = _coerce_status(target)
    expected = _LIFECYCLE_TRANSITIONS.get(current)
    if target is not expected:
        expected_text = "none" if expected is None else expected.value
        raise ValueError(
            f"invalid definition lifecycle transition: {current.value} -> "
            f"{target.value}; expected {expected_text}"
        )


@dataclass(frozen=True, slots=True)
class AgentDefinition:
    """Immutable, deployable Agent semantics separated from runtime bindings.

    ``semantic_digests`` pins resolved artifacts that are not embedded in the
    definition itself. Development may omit them; the Production profile
    requires the complete digest set in ``REQUIRED_SEMANTIC_DIGEST_KEYS``.
    Runtime endpoints, credentials, stores, and telemetry objects cannot enter
    this object and therefore cannot affect its canonical fingerprint.
    """

    agent_id: str
    version: str
    description: str
    instructions_ref: str
    execution_profile: str
    model_route: str
    tool_refs: tuple[str, ...]
    skill_refs: tuple[str, ...]
    input_contract_ref: str
    output_contract_ref: str
    memory_policy_ref: str
    authorization_policy_ref: str
    data_classification: str
    side_effect_level: str
    approval_requirement: str
    callable_by: frozenset[str]
    limits: RunLimits
    semantic_digests: Mapping[str, str] = field(default_factory=dict, repr=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        ref = AgentRef(self.agent_id, self.version)
        object.__setattr__(self, "agent_id", ref.agent_id)
        object.__setattr__(self, "version", ref.version)
        object.__setattr__(
            self,
            "description",
            _validated_text(self.description, "description", max_length=2048),
        )
        for field_name in (
            "instructions_ref",
            "input_contract_ref",
            "output_contract_ref",
            "memory_policy_ref",
            "authorization_policy_ref",
        ):
            object.__setattr__(
                self,
                field_name,
                _validated_reference(getattr(self, field_name), field_name),
            )
        for field_name in (
            "execution_profile",
            "model_route",
            "data_classification",
            "side_effect_level",
            "approval_requirement",
        ):
            object.__setattr__(
                self,
                field_name,
                _validated_code(getattr(self, field_name), field_name),
            )
        if self.side_effect_level not in _SIDE_EFFECT_LEVELS:
            raise ValueError(
                "side_effect_level must be 'none', 'read', 'advisory', or 'write'"
            )
        object.__setattr__(
            self,
            "tool_refs",
            _validated_reference_tuple(self.tool_refs, "tool_refs"),
        )
        object.__setattr__(
            self,
            "skill_refs",
            _validated_reference_tuple(self.skill_refs, "skill_refs"),
        )
        object.__setattr__(
            self,
            "callable_by",
            _validated_callers(self.callable_by),
        )
        if not isinstance(self.limits, RunLimits):
            raise TypeError("limits must be a RunLimits")
        object.__setattr__(
            self,
            "semantic_digests",
            _validated_digests(self.semantic_digests),
        )
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        object.__setattr__(self, "fingerprint", f"sha256:{digest}")

    @property
    def ref(self) -> AgentRef:
        return AgentRef(self.agent_id, self.version)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "agent_id": self.agent_id,
            "version": self.version,
            "description": self.description,
            "instructions_ref": self.instructions_ref,
            "execution_profile": self.execution_profile,
            "model_route": self.model_route,
            "tool_refs": list(self.tool_refs),
            "skill_refs": list(self.skill_refs),
            "input_contract_ref": self.input_contract_ref,
            "output_contract_ref": self.output_contract_ref,
            "memory_policy_ref": self.memory_policy_ref,
            "authorization_policy_ref": self.authorization_policy_ref,
            "data_classification": self.data_classification,
            "side_effect_level": self.side_effect_level,
            "approval_requirement": self.approval_requirement,
            "callable_by": sorted(self.callable_by),
            "limits": _limits_payload(self.limits),
            "semantic_digests": dict(self.semantic_digests),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        value = self.canonical_payload()
        if include_fingerprint:
            value["fingerprint"] = self.fingerprint
        return value

    def __hash__(self) -> int:
        return hash((self.ref, self.fingerprint))


def _validate_agent_id(value: object) -> None:
    if not isinstance(value, str):
        raise TypeError("agent_id must be a string")
    if _AGENT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "agent_id must be 1-128 characters using letters, digits, '.', '_', or '-'"
        )


def _limits_payload(limits: RunLimits) -> dict[str, int | float | bool]:
    return {
        "max_steps": limits.max_steps,
        "max_tool_calls": limits.max_tool_calls,
        "timeout_seconds": float(limits.timeout_seconds),
        "parallel_tool_calls": limits.parallel_tool_calls,
        "max_parallel_tools": limits.max_parallel_tools,
        "max_step_attempts": limits.max_step_attempts,
        "max_replans": limits.max_replans,
        "max_tool_repair_attempts": limits.max_tool_repair_attempts,
        "max_model_turns": limits.max_model_turns,
        "no_progress_model_turn_threshold": limits.no_progress_model_turn_threshold,
    }


def _validate_version(value: object) -> None:
    if not isinstance(value, str):
        raise TypeError("version must be a string")
    match = _SEMVER_PATTERN.fullmatch(value)
    if len(value) > 128 or match is None:
        raise ValueError("version must be an exact Semantic Version")
    prerelease = match.group(4)
    if prerelease is not None and any(
        len(identifier) > 1 and identifier.startswith("0")
        for identifier in prerelease.split(".")
        if identifier.isdigit()
    ):
        raise ValueError("version must be an exact Semantic Version")


def _validated_text(value: object, field_name: str, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if value != value.strip() or not value:
        raise ValueError(f"{field_name} cannot be empty or padded")
    if len(value) > max_length or any(
        not character.isprintable() for character in value
    ):
        raise ValueError(
            f"{field_name} must be printable and at most {max_length} chars"
        )
    return value


def _validated_reference(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if _REFERENCE_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a stable non-secret reference")
    return value


def _validated_code(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if _STABLE_CODE_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a stable machine-readable code")
    return value


def _validated_reference_tuple(
    values: Iterable[str],
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of references")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of references") from exc
    normalized = tuple(
        _validated_reference(value, f"{field_name} item") for value in result
    )
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} cannot contain duplicates")
    return normalized


def _validated_callers(values: Iterable[str]) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError("callable_by must be an iterable of Agent IDs")
    try:
        callers = frozenset(values)
    except TypeError as exc:
        raise TypeError("callable_by must contain hashable Agent IDs") from exc
    for caller in callers:
        if caller == "*":
            continue
        _validate_agent_id(caller)
    return callers


def _validated_digests(values: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError("semantic_digests must be a mapping")
    normalized: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or _DIGEST_KEY_PATTERN.fullmatch(key) is None:
            raise ValueError("semantic digest keys must be stable lowercase codes")
        if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(
                f"semantic digest '{key}' must be a canonical sha256 value"
            )
        normalized[key] = value
    return MappingProxyType(dict(sorted(normalized.items())))


def _coerce_status(value: DefinitionStatus | str) -> DefinitionStatus:
    if isinstance(value, DefinitionStatus):
        return value
    if not isinstance(value, str):
        raise TypeError("definition status must be a DefinitionStatus")
    try:
        return DefinitionStatus(value)
    except ValueError as exc:
        raise ValueError(f"unknown definition status: {value}") from exc


__all__ = [
    "AgentDefinition",
    "AgentRef",
    "DefinitionStatus",
    "REQUIRED_SEMANTIC_DIGEST_KEYS",
    "validate_lifecycle_transition",
]
