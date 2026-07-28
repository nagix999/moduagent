from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


SKILL_PHASES = ("plan", "act", "finalize")
_SKILL_PHASE_SET = frozenset(SKILL_PHASES)


def freeze_mapping(value: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """Return a recursively immutable copy of a string-keyed mapping."""

    return MappingProxyType(
        {str(key): _freeze_value(item) for key, item in (value or {}).items()}
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return freeze_mapping(value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_skill_phases(value: Any) -> frozenset[str]:
    if isinstance(value, (str, bytes)):
        raise TypeError("applies_to must be an iterable of Skill phases")
    try:
        phases = tuple(value)
    except TypeError as exc:
        raise TypeError("applies_to must be an iterable of Skill phases") from exc
    if not phases:
        raise ValueError("applies_to must contain at least one Skill phase")
    if not all(isinstance(phase, str) for phase in phases):
        raise TypeError("applies_to must contain Skill phase strings")
    if len(set(phases)) != len(phases):
        raise ValueError("applies_to cannot contain duplicate Skill phases")
    unsupported = set(phases) - _SKILL_PHASE_SET
    if unsupported:
        expected = ", ".join(SKILL_PHASES)
        raise ValueError(f"applies_to must contain only: {expected}")
    return frozenset(phases)


@dataclass(frozen=True, slots=True)
class SkillLimits:
    """Limits applied while discovering, selecting, and loading skills."""

    max_active_skills: int = 3
    max_catalog_tokens: int = 2_048
    max_selection_tokens: int = 8_192
    max_instruction_tokens: int = 12_000
    max_resource_reads: int = 8
    max_resource_tokens: int = 8_192
    max_total_skill_tokens: int = 20_000
    max_resource_bytes_per_read: int = 64 * 1024
    max_resource_file_bytes: int = 1024 * 1024
    max_resource_search_bytes: int = 4 * 1024 * 1024
    max_resource_search_results: int = 20
    max_skill_bytes: int = 64 * 1024
    max_package_files: int = 256
    max_package_bytes: int = 50 * 1024 * 1024

    def __post_init__(self) -> None:
        positive = (
            "max_active_skills",
            "max_catalog_tokens",
            "max_selection_tokens",
            "max_instruction_tokens",
            "max_resource_tokens",
            "max_total_skill_tokens",
            "max_resource_bytes_per_read",
            "max_resource_file_bytes",
            "max_resource_search_bytes",
            "max_resource_search_results",
            "max_skill_bytes",
            "max_package_files",
            "max_package_bytes",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.max_resource_reads < 0:
            raise ValueError("max_resource_reads cannot be negative")


@dataclass(frozen=True, slots=True)
class SkillDescriptor:
    """Small, discovery-time representation of a skill."""

    name: str
    description: str
    source_id: str
    digest: str
    version: str | None = None
    license: str | None = None
    compatibility: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    applies_to: frozenset[str] = field(default_factory=lambda: _SKILL_PHASE_SET)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        object.__setattr__(self, "allowed_tools", frozenset(self.allowed_tools))
        object.__setattr__(self, "applies_to", _freeze_skill_phases(self.applies_to))

    @property
    def ref(self) -> "SkillRef":
        return SkillRef(
            name=self.name,
            version=self.version,
            digest=self.digest,
            source_id=self.source_id,
        )


@dataclass(frozen=True, slots=True)
class SkillRef:
    """Content-addressed reference to one immutable skill revision."""

    name: str
    digest: str
    source_id: str
    version: str | None = None

    @classmethod
    def from_descriptor(cls, descriptor: SkillDescriptor) -> "SkillRef":
        return descriptor.ref


@dataclass(frozen=True, slots=True)
class SkillArtifact:
    """Activated skill instructions and the index of packaged resources."""

    descriptor: SkillDescriptor
    instructions: str
    references: tuple[str, ...] = ()
    assets: tuple[str, ...] = ()
    scripts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "references", tuple(self.references))
        object.__setattr__(self, "assets", tuple(self.assets))
        object.__setattr__(self, "scripts", tuple(self.scripts))

    @property
    def ref(self) -> SkillRef:
        return self.descriptor.ref


@dataclass(frozen=True, slots=True)
class SkillActivation:
    """Effective, authorized view of a skill for one run."""

    ref: SkillRef
    selected_by: str = "explicit"
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    applies_to: frozenset[str] = field(default_factory=lambda: _SKILL_PHASE_SET)

    def __post_init__(self) -> None:
        if self.selected_by not in {"explicit", "model"}:
            raise ValueError("selected_by must be 'explicit' or 'model'")
        object.__setattr__(self, "allowed_tools", frozenset(self.allowed_tools))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))
        object.__setattr__(self, "applies_to", _freeze_skill_phases(self.applies_to))

    @property
    def name(self) -> str:
        return self.ref.name

    @property
    def version(self) -> str | None:
        return self.ref.version

    @property
    def digest(self) -> str:
        return self.ref.digest

    @property
    def source_id(self) -> str:
        return self.ref.source_id
