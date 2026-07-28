from __future__ import annotations

import asyncio

import pytest

from moduagent.messages import Message
from moduagent.persistence.checkpoint import RunCheckpoint
from moduagent.runtime.context import RunContext, RunRequest, SkillActivationState
from moduagent.skills import (
    InMemorySkillSource,
    SkillRegistry,
    SkillRuntime,
    SkillValidationError,
    validate_skill_package,
)
from moduagent.skills.prompting import (
    SKILL_APPLIES_TO_METADATA_KEY,
    compose_skill_prompt,
    render_skill_messages,
)


def _skill_markdown(name: str, applies_to: str = "") -> str:
    return (
        "---\n"
        f"name: {name}\n"
        "description: Test phase-scoped instructions.\n"
        f"{applies_to}"
        "---\n\n"
        f"Instructions for {name}.\n"
    )


def test_applies_to_defaults_to_all_and_is_preserved_in_activation() -> None:
    registry = SkillRegistry.from_sources(
        InMemorySkillSource(
            {
                "all-phases": _skill_markdown("all-phases"),
                "execution-only": _skill_markdown(
                    "execution-only",
                    "applies-to:\n  - act\n",
                ),
            }
        )
    )

    all_phases = registry["all-phases"]
    execution_only = registry["execution-only"]

    assert all_phases.applies_to == {"plan", "act", "finalize"}
    assert execution_only.applies_to == {"act"}
    assert registry.activation("execution-only").applies_to == {"act"}


@pytest.mark.parametrize(
    ("applies_to", "message"),
    [
        ("applies-to: act\n", "YAML list"),
        ("applies-to:\n", "YAML list"),
        ("applies-to: []\n", "at least one"),
        ("applies-to: [act, act]\n", "duplicate"),
        ("applies-to: [act, verify]\n", "one of"),
        ("applies-to: [act, 1]\n", "phase strings"),
    ],
)
def test_applies_to_frontmatter_is_strict(
    applies_to: str,
    message: str,
) -> None:
    with pytest.raises(SkillValidationError, match=message):
        validate_skill_package(
            {"SKILL.md": _skill_markdown("phase-test", applies_to)},
            source_id="memory://phase-test",
            expected_name="phase-test",
        )


def test_render_and_compose_filter_skills_by_phase() -> None:
    artifacts = tuple(
        validate_skill_package(
            {"SKILL.md": markdown},
            source_id=f"memory://{name}",
            expected_name=name,
        )
        for name, markdown in (
            (
                "planning",
                _skill_markdown("planning", "applies-to: [plan]\n"),
            ),
            (
                "final-writing",
                _skill_markdown(
                    "final-writing",
                    "applies-to: [finalize]\n",
                ),
            ),
            ("all-phases", _skill_markdown("all-phases")),
        )
    )

    rendered = render_skill_messages(artifacts)
    assert len(rendered) == 3
    assert rendered[0].metadata[SKILL_APPLIES_TO_METADATA_KEY] == ("plan",)
    assert [
        message.metadata["moduagent.skill"]
        for message in render_skill_messages(artifacts, phase="plan")
    ] == ["planning", "all-phases"]

    base = (Message.system("framework"), Message.user("request"))
    finalized = compose_skill_prompt(base, rendered, phase="finalize")
    assert [message.metadata.get("moduagent.skill") for message in finalized] == [
        None,
        "final-writing",
        "all-phases",
        None,
    ]

    # Messages rendered before phase scopes existed remain compatible.
    legacy = Message.system(
        "legacy instructions",
        metadata={"moduagent.skill": "legacy"},
    )
    assert legacy in compose_skill_prompt(base, (legacy,), phase="act")

    with pytest.raises(ValueError, match="phase must be one of"):
        render_skill_messages(artifacts, phase="verify")
    with pytest.raises(ValueError, match="phase must be one of"):
        compose_skill_prompt(base, rendered, phase="verify")


def test_phase_scope_is_preserved_by_runtime_and_checkpoint() -> None:
    async def scenario() -> None:
        registry = SkillRegistry.from_sources(
            InMemorySkillSource(
                {
                    "execution-only": _skill_markdown(
                        "execution-only",
                        "applies-to: [act]\n",
                    )
                }
            )
        )
        context = RunContext(
            run_id="phase-scope",
            request=RunRequest(
                input="execute",
                session_id="phase-session",
                requested_skills=("execution-only",),
                skill_mode="explicit",
            ),
            messages=[Message.user("execute")],
        )

        await SkillRuntime(registry).activate(context)

        assert context.skill_state.active_skills[0].applies_to == ("act",)
        encoded = RunCheckpoint.from_context(context).to_dict()
        restored = RunCheckpoint.from_dict(encoded).to_context()
        assert restored.skill_state.active_skills[0].applies_to == ("act",)

    asyncio.run(scenario())

    # Checkpoints created before phase scoping keep the all-phases behavior.
    legacy = SkillActivationState.from_dict({"name": "legacy"})
    assert legacy.applies_to == ("plan", "act", "finalize")
