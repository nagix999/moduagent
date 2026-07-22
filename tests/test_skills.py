from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from moduagent.messages import Message, Usage
from moduagent.models import ModelResponse
from moduagent.skills import (
    ExplicitSkillSelector,
    FilesystemSkillSource,
    HybridSkillSelector,
    InMemorySkillSource,
    ModelSkillSelector,
    SkillDigestMismatchError,
    SkillLimitError,
    SkillLimits,
    SkillNotFoundError,
    SkillRegistry,
    SkillSelectionError,
    SkillSelectionRequest,
    SkillValidationError,
    compute_skill_digest,
    validate_skill_package,
)


def _skill_markdown(
    name: str,
    *,
    description: str = "Use this skill to review an invoice.",
    body: str = "# Review\n\nCheck the invoice against policy.",
    extra: str = "",
) -> str:
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "license: MIT\n"
        "compatibility: Requires ModuAgent 0.2\n"
        "metadata:\n"
        '  version: "1.2.0"\n'
        '  owner: "finance"\n'
        "allowed-tools: lookup-invoice lookup-vendor\n"
        f"{extra}"
        "---\n\n"
        f"{body}\n"
    )


def test_in_memory_source_parses_official_frontmatter_and_indexes_files() -> None:
    source = InMemorySkillSource(
        {
            "invoice-review": {
                "SKILL.md": _skill_markdown("invoice-review"),
                "references/policy.md": "Policy text",
                "assets/template.md": "Template",
                "scripts/normalize.py": "raise RuntimeError('must not execute')",
            }
        },
        source_id="embedded",
    )

    descriptor = source.discover()[0]
    artifact = source.load(descriptor.ref)

    assert descriptor.name == "invoice-review"
    assert descriptor.version == "1.2.0"
    assert descriptor.license == "MIT"
    assert descriptor.metadata["owner"] == "finance"
    assert descriptor.allowed_tools == {"lookup-invoice", "lookup-vendor"}
    assert artifact.references == ("references/policy.md",)
    assert artifact.assets == ("assets/template.md",)
    assert artifact.scripts == ("scripts/normalize.py",)
    assert (
        source.read_resource(descriptor.ref, "references/policy.md") == b"Policy text"
    )
    with pytest.raises(SkillNotFoundError):
        source.read_resource(descriptor.ref, "scripts/normalize.py")
    with pytest.raises(TypeError):
        descriptor.metadata["owner"] = "other"  # type: ignore[index]


@pytest.mark.parametrize(
    ("markdown", "message"),
    [
        ("# No frontmatter", "frontmatter"),
        ("---\ndescription: Missing name\n---\nBody", "requires string field 'name'"),
        ("---\nname: Bad_Name\ndescription: bad\n---\nBody", "skill name"),
        (
            "---\nname: valid\ndescription: ok\nmetadata:\n  version: 2\n---\nBody",
            "metadata values",
        ),
        (
            "---\nname: valid\nname: duplicate\ndescription: ok\n---\nBody",
            "duplicate YAML key",
        ),
        (
            "---\nname: valid\ndescription: ok\nunknown: true\n---\nBody",
            "unknown SKILL.md frontmatter key",
        ),
    ],
)
def test_strict_skill_validation_rejects_malformed_packages(
    markdown: str,
    message: str,
) -> None:
    with pytest.raises(SkillValidationError, match=message):
        validate_skill_package(
            {"SKILL.md": markdown},
            source_id="memory://test/valid",
            expected_name="valid",
        )


def test_package_name_must_match_directory_and_unknown_paths_are_rejected() -> None:
    with pytest.raises(SkillValidationError, match="directory name"):
        InMemorySkillSource({"other-name": _skill_markdown("invoice-review")})

    with pytest.raises(SkillValidationError, match="unsupported skill package path"):
        InMemorySkillSource(
            {
                "invoice-review": {
                    "SKILL.md": _skill_markdown("invoice-review"),
                    "README.md": "not part of the strict package",
                }
            }
        )


def test_digest_is_order_independent_and_changes_with_any_file() -> None:
    first = {
        "SKILL.md": _skill_markdown("invoice-review"),
        "references/policy.md": "one",
    }
    reversed_order = dict(reversed(tuple(first.items())))
    changed = {**first, "references/policy.md": "two"}

    assert compute_skill_digest(first) == compute_skill_digest(reversed_order)
    assert compute_skill_digest(first).startswith("sha256:")
    assert compute_skill_digest(first) != compute_skill_digest(changed)


def test_filesystem_source_and_registry_pin_the_discovered_digest(
    tmp_path: Path,
) -> None:
    skill_dir = tmp_path / "invoice-review"
    skill_dir.mkdir()
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(_skill_markdown("invoice-review"), encoding="utf-8")
    source = FilesystemSkillSource(tmp_path)
    registry = SkillRegistry([source])

    initial_digest = registry["invoice-review"].digest
    assert registry.load("invoice-review").instructions.startswith("# Review")

    skill_file.write_text(
        _skill_markdown("invoice-review", body="Changed instructions"),
        encoding="utf-8",
    )
    assert registry["invoice-review"].digest == initial_digest
    with pytest.raises(SkillDigestMismatchError, match="content changed"):
        registry.load("invoice-review")


def test_filesystem_source_rejects_symlinks(tmp_path: Path) -> None:
    skill_dir = tmp_path / "invoice-review"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        _skill_markdown("invoice-review"), encoding="utf-8"
    )
    (skill_dir / "references").mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")
    (skill_dir / "references" / "secret.md").symlink_to(outside)

    with pytest.raises(SkillValidationError, match="symbolic links"):
        FilesystemSkillSource(tmp_path).discover()


def test_registry_is_sorted_immutable_and_rejects_duplicate_names() -> None:
    source = InMemorySkillSource(
        {
            "zeta": _skill_markdown("zeta"),
            "alpha": _skill_markdown("alpha"),
        }
    )
    registry = SkillRegistry.from_sources(source)
    equivalent = SkillRegistry.from_sources(source)

    assert tuple(item.name for item in registry) == ("alpha", "zeta")
    assert registry.catalog_digest == equivalent.catalog_digest
    assert registry.descriptors == registry.catalog
    assert not hasattr(registry, "register")

    duplicate = InMemorySkillSource(
        {"alpha": _skill_markdown("alpha")}, source_id="two"
    )
    with pytest.raises(SkillValidationError, match="duplicate skill"):
        SkillRegistry.from_sources(source, duplicate)


def test_resource_reads_reject_traversal_scripts_and_oversized_content() -> None:
    source = InMemorySkillSource(
        {
            "safe-skill": {
                "SKILL.md": _skill_markdown("safe-skill"),
                "references/large.txt": "0123456789",
                "scripts/run.py": "print('no')",
            }
        }
    )
    ref = source.discover()[0].ref

    with pytest.raises(SkillValidationError, match="unsafe"):
        source.read_resource(ref, "../secret")
    with pytest.raises(SkillNotFoundError):
        source.read_resource(ref, "scripts/run.py")
    with pytest.raises(SkillLimitError, match="read limit"):
        source.read_resource(ref, "references/large.txt", max_bytes=4)


class FakeSelectionModel:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return ModelResponse(
            Message.assistant(self.content),
            usage=Usage(20, 4, 24),
            finish_reason="stop",
        )


def _selection_registry() -> SkillRegistry:
    return SkillRegistry.from_sources(
        InMemorySkillSource(
            {
                "invoice-review": _skill_markdown("invoice-review"),
                "weather-guide": _skill_markdown(
                    "weather-guide",
                    description="Use this skill when weather guidance is requested.",
                    body="Instructions that must not appear in selection metadata.",
                ),
            }
        )
    )


def test_explicit_selector_validates_names_and_limits() -> None:
    async def scenario() -> None:
        registry = _selection_registry()
        request = SkillSelectionRequest(
            input="review this",
            catalog=registry.catalog,
            requested_skills=("invoice-review", "invoice-review"),
            max_skills=1,
        )
        result = await ExplicitSkillSelector().select(request)
        assert result.names == ("invoice-review",)
        assert result.selected_by == {"invoice-review": "explicit"}

        with pytest.raises(SkillSelectionError, match="unknown skill"):
            await ExplicitSkillSelector().select(
                SkillSelectionRequest(
                    input="x",
                    catalog=registry.catalog,
                    requested_skills=("missing",),
                )
            )

    asyncio.run(scenario())


def test_model_selector_uses_metadata_only_and_rejects_hallucinations() -> None:
    async def scenario() -> None:
        registry = _selection_registry()
        model = FakeSelectionModel('{"skills":["weather-guide"]}')
        result = await ModelSkillSelector(model).select(
            SkillSelectionRequest(input="Will it rain?", catalog=registry.catalog)
        )

        assert result.names == ("weather-guide",)
        assert result.usage.total_tokens == 24
        sent = model.requests[0]
        prompt = json.loads(sent.messages[1].content or "{}")
        assert {item["name"] for item in prompt["skills"]} == {
            "invoice-review",
            "weather-guide",
        }
        assert "Instructions that must not appear" not in sent.messages[1].content
        assert sent.tools == ()
        assert sent.output_schema["properties"]["skills"]["maxItems"] == 2

        hallucinating = ModelSkillSelector(FakeSelectionModel('{"skills":["admin"]}'))
        with pytest.raises(SkillSelectionError, match="unknown skill"):
            await hallucinating.select(
                SkillSelectionRequest(input="x", catalog=registry.catalog)
            )

    asyncio.run(scenario())


def test_hybrid_selector_prioritizes_explicit_and_fills_remaining_slots() -> None:
    async def scenario() -> None:
        registry = _selection_registry()
        automatic = ModelSkillSelector(
            FakeSelectionModel('{"skills":["weather-guide"]}')
        )
        selector = HybridSkillSelector(automatic, max_skills=2)
        result = await selector.select(
            SkillSelectionRequest(
                input="review this and check weather",
                catalog=registry.catalog,
                requested_skills=("invoice-review",),
                max_skills=2,
            )
        )

        assert result.names == ("invoice-review", "weather-guide")
        assert result.selected_by == {
            "invoice-review": "explicit",
            "weather-guide": "model",
        }
        assert result.usage.total_tokens == 24

    asyncio.run(scenario())


def test_skill_limits_are_strict_and_allow_disabling_resource_reads() -> None:
    assert SkillLimits(max_resource_reads=0).max_resource_reads == 0
    with pytest.raises(ValueError, match="max_active_skills"):
        SkillLimits(max_active_skills=0)
