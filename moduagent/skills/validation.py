from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

import yaml

from moduagent.skills.errors import SkillValidationError
from moduagent.skills.models import (
    SkillArtifact,
    SkillDescriptor,
    SkillLimits,
)


_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_FRONTMATTER_KEYS = {
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
}
_PACKAGE_ROOTS = {"references", "assets", "scripts"}
_DIGEST_DOMAIN = b"moduagent-skill-package-v1\0"


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as exc:
            raise SkillValidationError(
                "YAML mapping keys must be scalar values"
            ) from exc
        if duplicate:
            raise SkillValidationError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def validate_skill_name(name: str) -> str:
    """Validate and return an Agent Skills compatible skill name."""

    if not isinstance(name, str) or not name:
        raise SkillValidationError("skill name must be a non-empty string")
    if len(name) > 64:
        raise SkillValidationError("skill name cannot exceed 64 characters")
    if not _NAME_PATTERN.fullmatch(name):
        raise SkillValidationError(
            "skill name must contain lowercase letters, digits, and single hyphens only"
        )
    return name


def validate_relative_path(path: str) -> str:
    """Return a normalized package path, rejecting traversal and ambiguity."""

    if not isinstance(path, str) or not path:
        raise SkillValidationError("skill package path must be a non-empty string")
    if "\\" in path or "\x00" in path:
        raise SkillValidationError(f"unsafe skill package path: {path!r}")
    pure = PurePosixPath(path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise SkillValidationError(f"unsafe skill package path: {path!r}")
    normalized = pure.as_posix()
    if normalized != path:
        raise SkillValidationError(f"skill package path is not normalized: {path!r}")
    return normalized


def compute_skill_digest(files: Mapping[str, bytes | str]) -> str:
    """Compute a stable digest over normalized package paths and bytes."""

    normalized: dict[str, bytes] = {}
    for raw_path, raw_content in files.items():
        path = validate_relative_path(raw_path)
        if path in normalized:
            raise SkillValidationError(f"duplicate skill package path: {path}")
        if isinstance(raw_content, str):
            content = raw_content.encode("utf-8")
        elif isinstance(raw_content, bytes):
            content = raw_content
        else:
            raise SkillValidationError(
                f"skill package file must be bytes or text: {path}"
            )
        normalized[path] = content

    digest = hashlib.sha256()
    digest.update(_DIGEST_DOMAIN)
    for path in sorted(normalized):
        encoded_path = path.encode("utf-8")
        content = normalized[path]
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return f"sha256:{digest.hexdigest()}"


def parse_skill_frontmatter(document: str) -> tuple[Mapping[str, Any], str]:
    """Parse the YAML frontmatter and Markdown body of ``SKILL.md``."""

    if document.startswith("\ufeff"):
        document = document[1:]
    lines = document.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise SkillValidationError("SKILL.md must start with YAML frontmatter")

    closing_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.rstrip("\r\n") == "---"
        ),
        None,
    )
    if closing_index is None:
        raise SkillValidationError("SKILL.md YAML frontmatter is not closed")

    yaml_text = "".join(lines[1:closing_index])
    try:
        if any(
            isinstance(token, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken))
            for token in yaml.scan(yaml_text)
        ):
            raise SkillValidationError(
                "YAML anchors and aliases are not allowed in SKILL.md"
            )
        value = yaml.load(yaml_text, Loader=_UniqueKeySafeLoader)
    except SkillValidationError:
        raise
    except yaml.YAMLError as exc:
        raise SkillValidationError(f"invalid SKILL.md YAML frontmatter: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SkillValidationError("SKILL.md frontmatter must be a YAML mapping")
    if not all(isinstance(key, str) for key in value):
        raise SkillValidationError("SKILL.md frontmatter keys must be strings")

    return dict(value), "".join(lines[closing_index + 1 :]).strip()


def validate_skill_package(
    files: Mapping[str, bytes | str],
    *,
    source_id: str,
    expected_name: str | None = None,
    strict: bool = True,
    limits: SkillLimits | None = None,
) -> SkillArtifact:
    """Validate a complete package and return its immutable artifact."""

    limits = limits or SkillLimits()
    package = _normalize_package_files(files)
    if len(package) > limits.max_package_files:
        raise SkillValidationError(
            f"skill package exceeds max_package_files ({limits.max_package_files})"
        )
    package_bytes = sum(len(content) for content in package.values())
    if package_bytes > limits.max_package_bytes:
        raise SkillValidationError(
            f"skill package exceeds max_package_bytes ({limits.max_package_bytes})"
        )
    oversized_resource = next(
        (
            path
            for path, content in sorted(package.items())
            if path.split("/", 1)[0] in {"references", "assets"}
            and len(content) > limits.max_resource_file_bytes
        ),
        None,
    )
    if oversized_resource is not None:
        raise SkillValidationError(
            "skill resource exceeds max_resource_file_bytes "
            f"({limits.max_resource_file_bytes}): {oversized_resource}"
        )
    if "SKILL.md" not in package:
        raise SkillValidationError("skill package must contain SKILL.md")
    if len(package["SKILL.md"]) > limits.max_skill_bytes:
        raise SkillValidationError(
            f"SKILL.md exceeds max_skill_bytes ({limits.max_skill_bytes})"
        )

    if strict:
        unsupported = [
            path
            for path in package
            if path != "SKILL.md"
            and ("/" not in path or path.split("/", 1)[0] not in _PACKAGE_ROOTS)
        ]
        if unsupported:
            raise SkillValidationError(
                f"unsupported skill package path: {sorted(unsupported)[0]}"
            )

    try:
        document = package["SKILL.md"].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillValidationError("SKILL.md must be valid UTF-8") from exc

    frontmatter, instructions = parse_skill_frontmatter(document)
    descriptor = _build_descriptor(
        frontmatter,
        source_id=source_id,
        digest=compute_skill_digest(package),
        expected_name=expected_name,
        strict=strict,
    )
    if not instructions:
        raise SkillValidationError("SKILL.md instructions cannot be empty")

    references, assets, scripts = _classify_paths(package)
    return SkillArtifact(
        descriptor=descriptor,
        instructions=instructions,
        references=references,
        assets=assets,
        scripts=scripts,
    )


def _normalize_package_files(
    files: Mapping[str, bytes | str],
) -> dict[str, bytes]:
    package: dict[str, bytes] = {}
    for raw_path, raw_content in files.items():
        path = validate_relative_path(raw_path)
        if path in package:
            raise SkillValidationError(f"duplicate skill package path: {path}")
        if isinstance(raw_content, str):
            content = raw_content.encode("utf-8")
        elif isinstance(raw_content, bytes):
            content = bytes(raw_content)
        else:
            raise SkillValidationError(
                f"skill package file must be bytes or text: {path}"
            )
        package[path] = content
    return package


def _build_descriptor(
    value: Mapping[str, Any],
    *,
    source_id: str,
    digest: str,
    expected_name: str | None,
    strict: bool,
) -> SkillDescriptor:
    if strict:
        unknown = sorted(set(value) - _FRONTMATTER_KEYS)
        if unknown:
            raise SkillValidationError(
                f"unknown SKILL.md frontmatter key: {unknown[0]}"
            )

    name = value.get("name")
    description = value.get("description")
    if not isinstance(name, str):
        raise SkillValidationError("SKILL.md frontmatter requires string field 'name'")
    name = validate_skill_name(name)
    if expected_name is not None and name != expected_name:
        raise SkillValidationError(
            f"skill name {name!r} does not match directory name {expected_name!r}"
        )
    if not isinstance(description, str) or not description.strip():
        raise SkillValidationError(
            "SKILL.md frontmatter requires non-empty string field 'description'"
        )
    description = description.strip()
    if len(description) > 1_024:
        raise SkillValidationError("skill description cannot exceed 1024 characters")

    license_value = _optional_string(value, "license")
    compatibility = _optional_string(value, "compatibility")
    if compatibility is not None and len(compatibility) > 500:
        raise SkillValidationError("skill compatibility cannot exceed 500 characters")

    raw_metadata = value.get("metadata", {})
    if raw_metadata is None:
        raw_metadata = {}
    if not isinstance(raw_metadata, Mapping):
        raise SkillValidationError("skill metadata must be a YAML mapping")
    metadata: dict[str, str] = {}
    for key, item in raw_metadata.items():
        if not isinstance(key, str) or not key:
            raise SkillValidationError("skill metadata keys must be non-empty strings")
        if not isinstance(item, str):
            raise SkillValidationError("skill metadata values must be strings")
        metadata[key] = item

    return SkillDescriptor(
        name=name,
        description=description,
        source_id=_required_string(source_id, "source_id"),
        digest=digest,
        version=metadata.get("version"),
        license=license_value,
        compatibility=compatibility,
        metadata=metadata,
        allowed_tools=_parse_allowed_tools(value.get("allowed-tools")),
    )


def _optional_string(value: Mapping[str, Any], key: str) -> str | None:
    raw = value.get(key)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise SkillValidationError(f"skill {key} must be a non-empty string")
    return raw.strip()


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SkillValidationError(f"{field} must be a non-empty string")
    return value.strip()


def _parse_allowed_tools(raw: Any) -> frozenset[str]:
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        values = raw.split()
    elif isinstance(raw, list):
        values = raw
    else:
        raise SkillValidationError("allowed-tools must be a string or YAML list")
    tools: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise SkillValidationError(
                "allowed-tools entries must be non-empty strings"
            )
        tool = value.strip()
        if any(ord(character) < 32 for character in tool):
            raise SkillValidationError("allowed-tools entries cannot contain controls")
        tools.add(tool)
    return frozenset(tools)


def _classify_paths(
    package: Mapping[str, bytes],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    references = tuple(
        sorted(path for path in package if path.startswith("references/"))
    )
    assets = tuple(sorted(path for path in package if path.startswith("assets/")))
    scripts = tuple(sorted(path for path in package if path.startswith("scripts/")))
    return references, assets, scripts
