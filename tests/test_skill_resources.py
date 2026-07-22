from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
from pathlib import Path
from threading import Event

import pytest

import moduagent.skills.resources as resource_module
import moduagent.skills.source as source_module
from moduagent.skills.resources import (
    SkillResourceBinaryError,
    SkillResourceChangedError,
    SkillResourceCursorError,
    SkillResourceKind,
    SkillResourceLoader,
    SkillResourcePathError,
    SkillResourceSecurityError,
    SkillResourceTooLargeError,
)
from moduagent.skills.runtime import SkillRuntime
from moduagent.skills.source import FilesystemSkillSource
from moduagent.skills.tools import SkillReadTool, SkillSearchTool
from moduagent.skills import (
    InMemorySkillSource,
    SkillLimits,
    SkillRegistry,
    SkillValidationError,
)
from moduagent.messages import ToolCall
from moduagent.runtime.context import SkillActivationState
from moduagent.tools import (
    ToolErrorType,
    ToolExecutionContext,
    ToolExecutor,
)


def _skill_root(tmp_path: Path) -> Path:
    root = tmp_path / "example-skill"
    (root / "references").mkdir(parents=True)
    (root / "assets").mkdir()
    (root / "scripts").mkdir()
    (root / "SKILL.md").write_text("# Example", encoding="utf-8")
    return root


def _catalog_skill(catalog: Path, name: str = "portable-skill") -> Path:
    skill = catalog / name
    (skill / "references").mkdir(parents=True)
    (skill / "assets").mkdir()
    (skill / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Use this Skill to check a policy.\n"
        "metadata:\n"
        '  version: "1.0.0"\n'
        "---\n\n"
        "# Policy\n\nRead the policy before answering.\n",
        encoding="utf-8",
    )
    (skill / "references" / "policy.md").write_text("safe policy", encoding="utf-8")
    return skill


def _active_context(registry: SkillRegistry, name: str) -> ToolExecutionContext:
    descriptor = registry.require(name)
    activation = SkillActivationState(
        name=descriptor.name,
        version=descriptor.version,
        digest=descriptor.digest,
        source_id=descriptor.source_id,
    )
    return ToolExecutionContext(metadata={"active_skills": (activation,)})


def test_read_pages_utf8_content_with_digest_and_metadata(tmp_path: Path) -> None:
    root = _skill_root(tmp_path)
    content = "첫 번째 규칙\nsecond rule\n"
    payload = content.encode("utf-8")
    (root / "references" / "policy.md").write_bytes(payload)
    loader = SkillResourceLoader(
        root,
        max_file_bytes=128,
        max_read_bytes=8,
        max_search_bytes=128,
    )

    pages = []
    cursor = 0
    while True:
        page = loader.read("references/policy.md", cursor=cursor)
        pages.append(page)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor

    assert "".join(page.content for page in pages) == content
    assert all(page.returned_bytes <= 8 for page in pages)
    assert pages[0].kind is SkillResourceKind.REFERENCE
    assert pages[0].cursor == 0
    assert pages[0].truncated is True
    assert pages[-1].truncated is False
    assert pages[-1].next_cursor is None
    assert {page.size_bytes for page in pages} == {len(payload)}
    expected_digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    assert {page.digest for page in pages} == {expected_digest}

    asset = root / "assets" / "template.md"
    asset.write_text("report template", encoding="utf-8")
    assert loader.read("assets/template.md").kind is SkillResourceKind.ASSET


@pytest.mark.parametrize(
    "resource_path",
    [
        "/etc/passwd",
        "../outside.txt",
        "references/../../outside.txt",
        "references/../scripts/run.py",
        "references//policy.md",
        "references/./policy.md",
        "references\\policy.md",
        "references/evil\x00.md",
        "SKILL.md",
        "scripts/run.py",
    ],
)
def test_read_rejects_unsafe_or_non_resource_paths(
    tmp_path: Path, resource_path: str
) -> None:
    root = _skill_root(tmp_path)
    (root / "references" / "policy.md").write_text("policy", encoding="utf-8")
    (root / "scripts" / "run.py").write_text("print('unsafe')", encoding="utf-8")
    loader = SkillResourceLoader(root)

    with pytest.raises(SkillResourcePathError):
        loader.read(resource_path)


def test_read_and_search_reject_symlinks(tmp_path: Path) -> None:
    root = _skill_root(tmp_path)
    outside = tmp_path / "secret.txt"
    outside.write_text("confidential needle", encoding="utf-8")
    link = root / "references" / "linked.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symbolic links are not available")

    loader = SkillResourceLoader(root)
    with pytest.raises(SkillResourceSecurityError):
        loader.read("references/linked.md")
    with pytest.raises(SkillResourceSecurityError):
        loader.search("needle")

    linked_root = tmp_path / "linked-skill"
    linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(SkillResourceSecurityError):
        SkillResourceLoader(linked_root)


def test_read_is_anchored_when_intermediate_directory_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _skill_root(tmp_path)
    nested = root / "references" / "nested"
    nested.mkdir()
    (nested / "policy.md").write_text("safe policy", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "policy.md").write_text("stolen secret", encoding="utf-8")
    loader = SkillResourceLoader(root)
    original_open = resource_module.os.open
    attacked = False

    def racing_open(
        path: str | bytes | int,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal attacked
        if path == "policy.md" and dir_fd is not None and not attacked:
            attacked = True
            nested.rename(root / "references" / "original-nested")
            nested.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(resource_module.os, "open", racing_open)

    # The final file is opened relative to the already pinned nested directory,
    # not by resolving the now-malicious pathname again.
    assert loader.read("references/nested/policy.md").content == "safe policy"
    with pytest.raises(SkillResourceSecurityError):
        loader.read("references/nested/policy.md")


def test_search_blocks_intermediate_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _skill_root(tmp_path)
    nested = root / "references" / "nested"
    nested.mkdir()
    (nested / "policy.md").write_text("safe needle", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "policy.md").write_text("secret needle", encoding="utf-8")
    loader = SkillResourceLoader(root)
    original_open = resource_module.os.open
    attacked = False

    def racing_open(
        path: str | bytes | int,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal attacked
        if path == "nested" and dir_fd is not None and not attacked:
            attacked = True
            nested.rename(root / "references" / "original-nested")
            nested.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(resource_module.os, "open", racing_open)

    with pytest.raises(SkillResourceSecurityError):
        loader.search("needle")


def test_binary_and_oversized_resources_are_blocked(tmp_path: Path) -> None:
    root = _skill_root(tmp_path)
    (root / "assets" / "image.bin").write_bytes(b"text\x00binary")
    (root / "references" / "large.md").write_bytes(b"x" * 17)
    loader = SkillResourceLoader(
        root,
        max_file_bytes=16,
        max_read_bytes=16,
        max_search_bytes=16,
    )

    with pytest.raises(SkillResourceBinaryError):
        loader.read("assets/image.bin")
    with pytest.raises(SkillResourceTooLargeError):
        loader.read("references/large.md")


def test_read_validates_cursor_limit_and_pinned_digest(tmp_path: Path) -> None:
    root = _skill_root(tmp_path)
    resource = root / "references" / "unicode.md"
    resource.write_text("가나다", encoding="utf-8")
    loader = SkillResourceLoader(
        root,
        max_file_bytes=32,
        max_read_bytes=8,
        max_search_bytes=32,
    )

    page = loader.read("references/unicode.md")
    assert page.next_cursor == 6
    assert page.content == "가나"

    with pytest.raises(SkillResourceCursorError):
        loader.read("references/unicode.md", cursor=1)
    with pytest.raises(SkillResourceTooLargeError):
        loader.read("references/unicode.md", max_bytes=9)

    resource.write_text("changed", encoding="utf-8")
    with pytest.raises(SkillResourceChangedError):
        loader.read("references/unicode.md", expected_digest=page.digest)


def test_search_is_paginated_and_never_visits_scripts(tmp_path: Path) -> None:
    root = _skill_root(tmp_path)
    (root / "assets" / "template.md").write_text("Needle in asset\n", encoding="utf-8")
    (root / "references" / "one.md").write_text(
        "needle one\nno match\nneedle two\nneedle three\n", encoding="utf-8"
    )
    (root / "scripts" / "secret.py").write_text(
        "needle must not be visible", encoding="utf-8"
    )
    loader = SkillResourceLoader(
        root,
        max_file_bytes=128,
        max_read_bytes=64,
        max_search_bytes=256,
        max_search_results=2,
    )

    first = loader.search("needle")
    assert first.truncated is True
    assert first.next_cursor is not None
    assert len(first.matches) == 2
    assert first.scanned_bytes <= 256
    assert first.digest.startswith("sha256:")

    second = loader.search("needle", cursor=first.next_cursor)
    assert second.truncated is False
    assert second.next_cursor is None
    matches = first.matches + second.matches
    assert len(matches) == 4
    assert all(match.path.startswith(("assets/", "references/")) for match in matches)
    assert all(match.digest.startswith("sha256:") for match in matches)
    assert [
        (match.line_number, match.column)
        for match in matches
        if match.path.endswith("one.md")
    ] == [
        (1, 1),
        (3, 1),
        (4, 1),
    ]

    with pytest.raises(SkillResourceCursorError):
        loader.search("different", cursor=first.next_cursor)


def test_search_scan_budget_resumes_at_the_next_file(tmp_path: Path) -> None:
    root = _skill_root(tmp_path)
    payloads = ["needle " + (str(index) * 32) for index in range(3)]
    for index, payload in enumerate(payloads):
        (root / "references" / f"{index}.md").write_text(payload, encoding="utf-8")
    loader = SkillResourceLoader(
        root,
        max_file_bytes=64,
        max_read_bytes=32,
        max_search_bytes=80,
        max_search_results=10,
    )

    first = loader.search("needle")
    assert first.truncated is True
    assert first.scanned_files == 2
    assert first.scanned_bytes <= 80
    second = loader.search("needle", cursor=first.next_cursor)
    assert second.truncated is False
    assert second.scanned_files == 1
    assert len(first.matches + second.matches) == 3


def test_async_read_and_search_use_the_same_contract(tmp_path: Path) -> None:
    root = _skill_root(tmp_path)
    (root / "references" / "policy.md").write_text("approval policy", encoding="utf-8")
    loader = SkillResourceLoader(root)

    async def scenario() -> None:
        page = await loader.aread("references/policy.md")
        result = await loader.asearch("approval", path="references")
        assert page.content == "approval policy"
        assert result.matches[0].path == "references/policy.md"

    asyncio.run(scenario())


def test_pinned_loader_rejects_regular_file_swap_and_new_resource(
    tmp_path: Path,
) -> None:
    root = _skill_root(tmp_path)
    resource = root / "references" / "policy.md"
    original = b"safe policy"
    resource.write_bytes(original)
    expected = {
        "references/policy.md": f"sha256:{hashlib.sha256(original).hexdigest()}"
    }
    loader = SkillResourceLoader(root, expected_digests=expected)

    replacement = root / "references" / "replacement.md"
    replacement.write_text("changed policy", encoding="utf-8")
    replacement.replace(resource)

    with pytest.raises(SkillResourceChangedError):
        loader.read("references/policy.md")
    (root / "assets" / "injected.md").write_text("secret", encoding="utf-8")
    with pytest.raises(SkillResourceChangedError):
        loader.search("policy")


def test_filesystem_source_ids_and_lock_survive_root_relocation(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first-mount"
    first_root.mkdir()
    _catalog_skill(first_root)
    first = SkillRegistry.from_paths(first_root)
    descriptor = first.require("portable-skill")
    assert descriptor.source_id == "filesystem://portable-skill"

    lockfile = tmp_path / "skills.lock.json"
    lockfile.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "catalog_digest": first.catalog_digest,
                "skills": [
                    {
                        "name": descriptor.name,
                        "version": descriptor.version,
                        "digest": descriptor.digest,
                        "source_id": descriptor.source_id,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    second_root = tmp_path / "second-mount"
    shutil.copytree(first_root, second_root)
    second = SkillRegistry.from_paths(second_root, lockfile=lockfile)
    assert second.catalog_digest == first.catalog_digest
    assert second.ref("portable-skill") == first.ref("portable-skill")

    namespaced = FilesystemSkillSource(second_root, source_id="finance/policies")
    assert (
        namespaced.discover()[0].source_id
        == "filesystem://finance/policies/portable-skill"
    )


def test_filesystem_source_snapshot_blocks_intermediate_symlink_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "skills"
    catalog.mkdir()
    skill = _catalog_skill(catalog)
    nested = skill / "references" / "nested"
    nested.mkdir()
    (nested / "detail.md").write_text("safe detail", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "detail.md").write_text("secret detail", encoding="utf-8")
    source = FilesystemSkillSource(catalog)
    original_open = source_module.os.open
    attacked = False

    def racing_open(
        path: str | bytes | int,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal attacked
        if path == "nested" and dir_fd is not None and not attacked:
            attacked = True
            nested.rename(skill / "references" / "original-nested")
            nested.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(source_module.os, "open", racing_open)

    with pytest.raises(SkillValidationError, match="safely open"):
        source.discover()


def test_filesystem_source_rechecks_resource_size_after_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "skills"
    catalog.mkdir()
    skill = _catalog_skill(catalog)
    resource = skill / "references" / "policy.md"
    resource.write_bytes(b"safe")
    limits = SkillLimits(max_resource_file_bytes=8)
    source = FilesystemSkillSource(catalog, limits=limits)
    original_fdopen = source_module.os.fdopen
    attacked = False

    def racing_fdopen(descriptor, *args, **kwargs):
        nonlocal attacked
        if (
            not attacked
            and source_module.os.fstat(descriptor).st_ino == resource.stat().st_ino
        ):
            attacked = True
            resource.write_bytes(b"x" * 9)
        return original_fdopen(descriptor, *args, **kwargs)

    monkeypatch.setattr(source_module.os, "fdopen", racing_fdopen)

    with pytest.raises(SkillValidationError, match="max_resource_file_bytes"):
        source.discover()


def test_resource_file_limit_is_shared_by_validation_and_runtime(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "skills"
    catalog.mkdir()
    skill = _catalog_skill(catalog)
    (skill / "references" / "policy.md").write_bytes(b"x" * 9)
    limits = SkillLimits(
        max_resource_file_bytes=8,
        max_resource_bytes_per_read=4,
        max_resource_search_bytes=16,
        max_resource_search_results=2,
    )

    with pytest.raises(SkillValidationError, match="max_resource_file_bytes"):
        SkillRegistry.from_paths(catalog, limits=limits)


def test_filesystem_tool_rejects_resource_changed_after_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "skills"
    catalog.mkdir()
    skill = _catalog_skill(catalog)
    source = FilesystemSkillSource(catalog)
    registry = SkillRegistry.from_sources(source)
    runtime = SkillRuntime(registry)
    tool = SkillReadTool(runtime)
    original_snapshot = source.resource_snapshot

    def swapped_snapshot(ref):
        root, digests = original_snapshot(ref)
        replacement = root / "references" / "replacement.md"
        replacement.write_text("changed policy", encoding="utf-8")
        replacement.replace(skill / "references" / "policy.md")
        return root, digests

    monkeypatch.setattr(source, "resource_snapshot", swapped_snapshot)
    call = ToolCall(
        "read-1",
        tool.name,
        {"skill_name": "portable-skill", "path": "references/policy.md"},
    )

    result = asyncio.run(
        ToolExecutor([tool]).execute(
            call,
            _active_context(registry, "portable-skill"),
        )
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.type is ToolErrorType.EXECUTION_ERROR
    assert result.error.details["exception_type"] == "SkillResourceChangedError"


def test_filesystem_search_rejects_resource_added_after_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "skills"
    catalog.mkdir()
    skill = _catalog_skill(catalog)
    source = FilesystemSkillSource(catalog)
    registry = SkillRegistry.from_sources(source)
    runtime = SkillRuntime(registry)
    tool = SkillSearchTool(runtime)
    original_snapshot = source.resource_snapshot

    def injected_snapshot(ref):
        root, digests = original_snapshot(ref)
        (skill / "assets" / "injected.md").write_text("secret policy", encoding="utf-8")
        return root, digests

    monkeypatch.setattr(source, "resource_snapshot", injected_snapshot)
    call = ToolCall(
        "search-1",
        tool.name,
        {"skill_name": "portable-skill", "query": "policy"},
    )

    result = asyncio.run(
        ToolExecutor([tool]).execute(
            call,
            _active_context(registry, "portable-skill"),
        )
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.details["exception_type"] == "SkillResourceChangedError"


def test_filesystem_package_scan_does_not_defeat_tool_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = tmp_path / "skills"
    catalog.mkdir()
    _catalog_skill(catalog)
    source = FilesystemSkillSource(catalog)
    registry = SkillRegistry.from_sources(source)
    runtime = SkillRuntime(registry)
    tool = SkillReadTool(runtime)
    tool.timeout_seconds = 0.02
    original_snapshot = source.resource_snapshot
    started = Event()
    release = Event()

    def blocking_snapshot(ref):
        started.set()
        release.wait(0.5)
        return original_snapshot(ref)

    monkeypatch.setattr(source, "resource_snapshot", blocking_snapshot)
    call = ToolCall(
        "read-timeout",
        tool.name,
        {"skill_name": "portable-skill", "path": "references/policy.md"},
    )

    async def scenario():
        try:
            return await ToolExecutor([tool]).execute(
                call,
                _active_context(registry, "portable-skill"),
            )
        finally:
            release.set()
            await asyncio.sleep(0.01)

    result = asyncio.run(scenario())

    assert started.is_set()
    assert result.success is False
    assert result.error is not None
    assert result.error.type is ToolErrorType.TIMEOUT
    assert result.duration_seconds < 0.2


def test_embedded_resource_larger_than_page_limit_is_paginated() -> None:
    content = "x" * (64 * 1024 + 123)
    source = InMemorySkillSource(
        {
            "large-reference": {
                "SKILL.md": (
                    "---\n"
                    "name: large-reference\n"
                    "description: Use this Skill to read a large reference.\n"
                    "---\n\n"
                    "# Large reference\n\nRead the reference.\n"
                ),
                "references/large.md": content,
            }
        }
    )
    registry = SkillRegistry.from_sources(source)
    tool = SkillReadTool(SkillRuntime(registry))
    context = _active_context(registry, "large-reference")

    async def scenario() -> tuple[dict, dict]:
        first = await tool.invoke(
            {
                "skill_name": "large-reference",
                "path": "references/large.md",
            },
            context,
        )
        second = await tool.invoke(
            {
                "skill_name": "large-reference",
                "path": "references/large.md",
                "cursor": first["next_cursor"],
            },
            context,
        )
        return dict(first), dict(second)

    first, second = asyncio.run(scenario())

    assert first["returned_bytes"] == 64 * 1024
    assert first["truncated"] is True
    assert second["returned_bytes"] == 123
    assert second["truncated"] is False
    assert first["content"] + second["content"] == content


def test_embedded_source_cannot_ignore_resource_file_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = InMemorySkillSource(
        {
            "bounded-embedded": {
                "SKILL.md": (
                    "---\n"
                    "name: bounded-embedded\n"
                    "description: Use this Skill to read bounded content.\n"
                    "---\n\n"
                    "# Bounded content\n\nRead the reference.\n"
                ),
                "references/policy.md": "safe",
            }
        }
    )
    limits = SkillLimits(
        max_resource_bytes_per_read=64,
        max_resource_file_bytes=128,
    )
    registry = SkillRegistry.from_sources(source)
    tool = SkillReadTool(SkillRuntime(registry, limits=limits))
    requested_limits: list[int | None] = []

    def oversized_resource(ref, path, *, max_bytes=None):
        requested_limits.append(max_bytes)
        return b"x" * 129

    monkeypatch.setattr(source, "read_resource", oversized_resource)

    async def scenario() -> None:
        with pytest.raises(SkillResourceTooLargeError):
            await tool.invoke(
                {
                    "skill_name": "bounded-embedded",
                    "path": "references/policy.md",
                },
                _active_context(registry, "bounded-embedded"),
            )

    asyncio.run(scenario())
    assert requested_limits == [128]
