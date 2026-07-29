from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict
from functools import partial
from typing import Any

from moduagent.runtime.context import SkillActivationState
from moduagent.skills.errors import SkillSelectionError
from moduagent.skills.resources import (
    SkillResourceBinaryError,
    SkillResourceCursorError,
    SkillResourceKind,
    SkillResourceLoader,
    SkillResourcePage,
    SkillResourceSearchResult,
    SkillResourceTooLargeError,
    _run_sync_in_daemon,
)
from moduagent.skills.runtime import SkillRuntime
from moduagent.skills.source import (
    FilesystemSkillSource,
    ResourceReadableSkillSource,
)
from moduagent.tools import ToolExecutionContext, ToolSchema


SKILL_READ_TOOL_NAME = "moduagent_skill_read"
SKILL_SEARCH_TOOL_NAME = "moduagent_skill_search"
SKILL_RESOURCE_TOOL_NAMES = frozenset({SKILL_READ_TOOL_NAME, SKILL_SEARCH_TOOL_NAME})


class SkillReadTool:
    name = SKILL_READ_TOOL_NAME
    description = (
        "Read one bounded UTF-8 page from references/ or assets/ of an active Skill."
    )
    idempotent = True
    timeout_seconds = 10.0
    max_result_bytes = 256 * 1024
    is_skill_resource_tool = True

    def __init__(self, skill_runtime: SkillRuntime) -> None:
        self.skill_runtime = skill_runtime

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            self.name,
            self.description,
            {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string"},
                    "path": {"type": "string"},
                    "cursor": {"type": "integer", "minimum": 0},
                    "max_bytes": {"type": "integer", "minimum": 1},
                    "expected_digest": {"type": "string"},
                },
                "required": ["skill_name", "path"],
                "additionalProperties": False,
            },
            counts_toward_tool_limit=False,
        )

    def validate_arguments(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {
            "skill_name",
            "path",
            "cursor",
            "max_bytes",
            "expected_digest",
        }
        _reject_extra(arguments, allowed)
        skill_name = _required_text(arguments, "skill_name")
        path = _required_text(arguments, "path")
        cursor = _non_negative_int(arguments.get("cursor", 0), "cursor")
        max_bytes = arguments.get("max_bytes")
        if max_bytes is not None:
            max_bytes = _positive_int(max_bytes, "max_bytes")
        expected_digest = arguments.get("expected_digest")
        if expected_digest is not None:
            expected_digest = _text(expected_digest, "expected_digest")
        return {
            "skill_name": skill_name,
            "path": path,
            "cursor": cursor,
            "max_bytes": max_bytes,
            "expected_digest": expected_digest,
        }

    async def invoke(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> Mapping[str, Any]:
        activation = _active_skill(context, str(arguments["skill_name"]))
        ref = self.skill_runtime.registry.ref(activation.name)
        if (
            ref.digest != activation.digest
            or ref.source_id != activation.source_id
            or ref.version != activation.version
        ):
            raise SkillSelectionError(
                f"active skill no longer matches the registry: {activation.name}"
            )
        source = self.skill_runtime.registry.source_for(activation.name)
        max_bytes = arguments.get("max_bytes")
        if max_bytes is None:
            max_bytes = self.skill_runtime.limits.max_resource_bytes_per_read
        if max_bytes > self.skill_runtime.limits.max_resource_bytes_per_read:
            raise ValueError("max_bytes exceeds the Skill read limit")

        if isinstance(source, FilesystemSkillSource):
            page = await _run_sync_in_daemon(
                partial(
                    _read_filesystem,
                    self.skill_runtime,
                    source,
                    ref,
                    str(arguments["path"]),
                    cursor=int(arguments.get("cursor", 0)),
                    max_bytes=int(max_bytes),
                    expected_digest=arguments.get("expected_digest"),
                )
            )
        elif isinstance(source, ResourceReadableSkillSource):
            page = await _run_sync_in_daemon(
                partial(
                    _read_embedded,
                    source,
                    ref,
                    str(arguments["path"]),
                    cursor=int(arguments.get("cursor", 0)),
                    max_bytes=int(max_bytes),
                    max_file_bytes=self.skill_runtime.limits.max_resource_file_bytes,
                    expected_digest=arguments.get("expected_digest"),
                )
            )
        else:
            raise TypeError("active Skill source does not support resources")
        return _page_dict(page)


class SkillSearchTool:
    name = SKILL_SEARCH_TOOL_NAME
    description = (
        "Search bounded UTF-8 content in references/ or assets/ of an active Skill."
    )
    idempotent = True
    timeout_seconds = 10.0
    max_result_bytes = 256 * 1024
    is_skill_resource_tool = True

    def __init__(self, skill_runtime: SkillRuntime) -> None:
        self.skill_runtime = skill_runtime

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            self.name,
            self.description,
            {
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string"},
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                    "cursor": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1},
                    "case_sensitive": {"type": "boolean"},
                },
                "required": ["skill_name", "query"],
                "additionalProperties": False,
            },
            counts_toward_tool_limit=False,
        )

    def validate_arguments(self, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        allowed = {
            "skill_name",
            "query",
            "path",
            "cursor",
            "max_results",
            "case_sensitive",
        }
        _reject_extra(arguments, allowed)
        result: dict[str, Any] = {
            "skill_name": _required_text(arguments, "skill_name"),
            "query": _required_text(arguments, "query"),
            "case_sensitive": bool(arguments.get("case_sensitive", False)),
        }
        if arguments.get("path") is not None:
            result["path"] = _text(arguments["path"], "path")
        if arguments.get("cursor") is not None:
            result["cursor"] = _text(arguments["cursor"], "cursor")
        if arguments.get("max_results") is not None:
            result["max_results"] = _positive_int(
                arguments["max_results"], "max_results"
            )
        return result

    async def invoke(
        self,
        arguments: Mapping[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> Mapping[str, Any]:
        activation = _active_skill(context, str(arguments["skill_name"]))
        ref = self.skill_runtime.registry.ref(activation.name)
        if (
            ref.digest != activation.digest
            or ref.source_id != activation.source_id
            or ref.version != activation.version
        ):
            raise SkillSelectionError(
                f"active skill no longer matches the registry: {activation.name}"
            )
        source = self.skill_runtime.registry.source_for(activation.name)
        if not isinstance(source, FilesystemSkillSource):
            raise TypeError(
                "Skill search currently requires a FilesystemSkillSource; "
                "embedded resources can be read directly"
            )
        result = await _run_sync_in_daemon(
            partial(
                _search_filesystem,
                self.skill_runtime,
                source,
                ref,
                str(arguments["query"]),
                path=arguments.get("path"),
                cursor=arguments.get("cursor"),
                max_results=arguments.get("max_results"),
                case_sensitive=bool(arguments.get("case_sensitive", False)),
            )
        )
        return _search_dict(result)


def _filesystem_loader(
    runtime: SkillRuntime,
    source: FilesystemSkillSource,
    ref: Any,
) -> SkillResourceLoader:
    root, expected_digests = source.resource_snapshot(ref)
    return SkillResourceLoader(
        root,
        max_file_bytes=runtime.limits.max_resource_file_bytes,
        max_read_bytes=runtime.limits.max_resource_bytes_per_read,
        max_search_bytes=runtime.limits.max_resource_search_bytes,
        max_search_results=runtime.limits.max_resource_search_results,
        expected_digests=expected_digests,
    )


def _read_filesystem(
    runtime: SkillRuntime,
    source: FilesystemSkillSource,
    ref: Any,
    path: str,
    *,
    cursor: int,
    max_bytes: int,
    expected_digest: str | None,
) -> SkillResourcePage:
    with _filesystem_loader(runtime, source, ref) as loader:
        return loader.read(
            path,
            cursor=cursor,
            max_bytes=max_bytes,
            expected_digest=expected_digest,
        )


def _search_filesystem(
    runtime: SkillRuntime,
    source: FilesystemSkillSource,
    ref: Any,
    query: str,
    *,
    path: str | None,
    cursor: str | None,
    max_results: int | None,
    case_sensitive: bool,
) -> SkillResourceSearchResult:
    with _filesystem_loader(runtime, source, ref) as loader:
        return loader.search(
            query,
            path=path,
            cursor=cursor,
            max_results=max_results,
            case_sensitive=case_sensitive,
        )


def _active_skill(
    context: ToolExecutionContext | None,
    name: str,
) -> SkillActivationState:
    if context is None:
        raise SkillSelectionError("Skill resource access requires a run context")
    raw_skills = context.metadata.get("active_skills", ())
    if isinstance(raw_skills, (str, bytes)) or not isinstance(
        raw_skills, (list, tuple)
    ):
        raise SkillSelectionError("active Skill metadata is invalid")
    for raw in raw_skills:
        activation = (
            raw
            if isinstance(raw, SkillActivationState)
            else SkillActivationState.from_dict(raw)
        )
        if activation.name == name:
            return activation
    raise SkillSelectionError(f"skill is not active for this run: {name}")


def _read_embedded(
    source: ResourceReadableSkillSource,
    ref: Any,
    path: str,
    *,
    cursor: int,
    max_bytes: int,
    max_file_bytes: int,
    expected_digest: str | None,
) -> SkillResourcePage:
    payload = source.read_resource(ref, path, max_bytes=max_file_bytes)
    if len(payload) > max_file_bytes:
        raise SkillResourceTooLargeError(
            f"Skill resource exceeds {max_file_bytes} bytes"
        )
    if b"\x00" in payload:
        raise SkillResourceBinaryError("Skill resource is not UTF-8 text")
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillResourceBinaryError("Skill resource is not UTF-8 text") from exc
    digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    if expected_digest is not None and expected_digest != digest:
        raise ValueError("Skill resource digest changed")
    if cursor > len(payload):
        raise SkillResourceCursorError("cursor is past the end of the Skill resource")
    try:
        payload[:cursor].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillResourceCursorError(
            "cursor does not point to a UTF-8 boundary"
        ) from exc
    end = min(len(payload), cursor + max_bytes)
    while end > cursor:
        try:
            content = payload[cursor:end].decode("utf-8")
            break
        except UnicodeDecodeError:
            end -= 1
    else:
        if cursor < len(payload):
            raise SkillResourceCursorError(
                "max_bytes is too small for the next UTF-8 character"
            )
        content = ""
    return SkillResourcePage(
        path=path,
        kind=(
            SkillResourceKind.REFERENCE
            if path.startswith("references/")
            else SkillResourceKind.ASSET
        ),
        content=content,
        cursor=cursor,
        next_cursor=end if end < len(payload) else None,
        size_bytes=len(payload),
        returned_bytes=end - cursor,
        digest=digest,
        truncated=end < len(payload),
    )


def _page_dict(page: SkillResourcePage) -> dict[str, Any]:
    value = asdict(page)
    value["kind"] = page.kind.value
    return value


def _search_dict(result: SkillResourceSearchResult) -> dict[str, Any]:
    return {
        "query": result.query,
        "matches": [
            {
                **asdict(match),
                "kind": match.kind.value,
            }
            for match in result.matches
        ],
        "cursor": result.cursor,
        "next_cursor": result.next_cursor,
        "scanned_files": result.scanned_files,
        "scanned_bytes": result.scanned_bytes,
        "digest": result.digest,
        "truncated": result.truncated,
    }


def _reject_extra(arguments: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = set(arguments).difference(allowed)
    if unknown:
        raise ValueError(f"unknown arguments: {', '.join(sorted(unknown))}")


def _required_text(arguments: Mapping[str, Any], key: str) -> str:
    if key not in arguments:
        raise ValueError(f"missing required argument: {key}")
    return _text(arguments[key], key)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
