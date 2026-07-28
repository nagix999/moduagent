from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from types import MappingProxyType

from moduagent.skills.errors import (
    SkillDigestMismatchError,
    SkillNotFoundError,
    SkillValidationError,
)
from moduagent.skills.models import (
    SkillActivation,
    SkillArtifact,
    SkillDescriptor,
    SkillLimits,
    SkillRef,
)
from moduagent.skills.source import FilesystemSkillSource, SkillSource


class SkillRegistry:
    """Immutable catalog snapshot assembled from one or more skill sources."""

    def __init__(self, sources: Iterable[SkillSource] = ()) -> None:
        source_tuple = tuple(sources)
        descriptors: dict[str, SkillDescriptor] = {}
        descriptor_sources: dict[str, SkillSource] = {}
        source_ids: dict[str, SkillSource] = {}

        for source in source_tuple:
            for descriptor in source.discover():
                if descriptor.name in descriptors:
                    raise SkillValidationError(
                        f"duplicate skill in registry: {descriptor.name}"
                    )
                existing_source = source_ids.get(descriptor.source_id)
                if existing_source is not None and existing_source is not source:
                    raise SkillValidationError(
                        f"duplicate skill source id: {descriptor.source_id}"
                    )
                descriptors[descriptor.name] = descriptor
                descriptor_sources[descriptor.name] = source
                source_ids[descriptor.source_id] = source

        ordered = {name: descriptors[name] for name in sorted(descriptors)}
        self._sources = source_tuple
        self._descriptors: Mapping[str, SkillDescriptor] = MappingProxyType(ordered)
        self._descriptor_sources: Mapping[str, SkillSource] = MappingProxyType(
            {name: descriptor_sources[name] for name in ordered}
        )
        self._catalog_digest = _compute_catalog_digest(tuple(ordered.values()))

    @classmethod
    def from_paths(
        cls,
        paths: str | Path | Iterable[str | Path],
        *,
        strict: bool = True,
        limits: SkillLimits | None = None,
        lockfile: str | Path | None = None,
    ) -> "SkillRegistry":
        if isinstance(paths, (str, Path)):
            path_values = (paths,)
        else:
            path_values = tuple(paths)
        registry = cls(
            FilesystemSkillSource(path, strict=strict, limits=limits)
            for path in path_values
        )
        if lockfile is not None:
            registry.verify_lock(lockfile)
        return registry

    @classmethod
    def from_sources(cls, *sources: SkillSource) -> "SkillRegistry":
        return cls(sources)

    @property
    def sources(self) -> tuple[SkillSource, ...]:
        return self._sources

    @property
    def descriptors(self) -> tuple[SkillDescriptor, ...]:
        return tuple(self._descriptors.values())

    @property
    def catalog(self) -> tuple[SkillDescriptor, ...]:
        return self.descriptors

    @property
    def catalog_digest(self) -> str:
        return self._catalog_digest

    def get(self, name: str) -> SkillDescriptor | None:
        return self._descriptors.get(name)

    def require(self, name: str) -> SkillDescriptor:
        descriptor = self.get(name)
        if descriptor is None:
            raise SkillNotFoundError(f"unknown skill: {name}")
        return descriptor

    def ref(self, name: str) -> SkillRef:
        return self.require(name).ref

    def load(self, skill: str | SkillRef) -> SkillArtifact:
        descriptor = self.require(skill if isinstance(skill, str) else skill.name)
        ref = descriptor.ref if isinstance(skill, str) else skill
        if ref != descriptor.ref:
            raise SkillDigestMismatchError(
                f"skill reference does not match registry snapshot: {descriptor.name}"
            )
        artifact = self._descriptor_sources[descriptor.name].load(ref)
        if artifact.descriptor != descriptor:
            raise SkillDigestMismatchError(
                f"loaded skill does not match registry snapshot: {descriptor.name}"
            )
        return artifact

    def activation(
        self,
        name: str,
        *,
        selected_by: str = "explicit",
        allowed_tools: Iterable[str] | None = None,
    ) -> SkillActivation:
        descriptor = self.require(name)
        effective_tools = descriptor.allowed_tools
        if allowed_tools is not None:
            effective_tools = effective_tools.intersection(allowed_tools)
        return SkillActivation(
            ref=descriptor.ref,
            selected_by=selected_by,
            allowed_tools=effective_tools,
            metadata={"description": descriptor.description},
            applies_to=descriptor.applies_to,
        )

    def source_for(self, name: str) -> SkillSource:
        self.require(name)
        return self._descriptor_sources[name]

    def verify_lock(self, lockfile: str | Path) -> None:
        """Fail unless a generated lock file exactly matches this snapshot."""

        path = Path(lockfile)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SkillValidationError(f"cannot read Skill lock file: {path}") from exc
        if not isinstance(value, Mapping) or value.get("schema_version") != 1:
            raise SkillValidationError("unsupported Skill lock schema")
        if value.get("catalog_digest") != self.catalog_digest:
            raise SkillDigestMismatchError(
                "Skill lock catalog digest does not match the registry"
            )
        raw_skills = value.get("skills")
        if not isinstance(raw_skills, list):
            raise SkillValidationError("Skill lock skills must be an array")
        locked: dict[str, Mapping[str, object]] = {}
        for item in raw_skills:
            if not isinstance(item, Mapping):
                raise SkillValidationError("Skill lock entries must be objects")
            name = item.get("name")
            if not isinstance(name, str) or name in locked:
                raise SkillValidationError(
                    "Skill lock names must be unique non-empty strings"
                )
            locked[name] = item
        if set(locked) != set(self._descriptors):
            raise SkillDigestMismatchError("Skill lock names do not match the registry")
        for name, descriptor in self._descriptors.items():
            item = locked[name]
            if (
                item.get("version") != descriptor.version
                or item.get("digest") != descriptor.digest
                or item.get("source_id") != descriptor.source_id
            ):
                raise SkillDigestMismatchError(
                    f"Skill lock entry does not match the registry: {name}"
                )

    def __contains__(self, name: object) -> bool:
        return name in self._descriptors

    def __getitem__(self, name: str) -> SkillDescriptor:
        return self.require(name)

    def __iter__(self) -> Iterator[SkillDescriptor]:
        return iter(self._descriptors.values())

    def __len__(self) -> int:
        return len(self._descriptors)


def _compute_catalog_digest(descriptors: tuple[SkillDescriptor, ...]) -> str:
    payload = [
        {
            "name": descriptor.name,
            "description": descriptor.description,
            "source_id": descriptor.source_id,
            "digest": descriptor.digest,
            "version": descriptor.version,
            "license": descriptor.license,
            "compatibility": descriptor.compatibility,
            "metadata": dict(descriptor.metadata),
            "allowed_tools": sorted(descriptor.allowed_tools),
        }
        for descriptor in descriptors
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
