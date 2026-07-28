from __future__ import annotations

from collections.abc import Iterable

from moduagent.messages import Message, MessageRole
from moduagent.skills.models import SKILL_PHASES, SkillArtifact


SKILL_METADATA_KEY = "moduagent.skill"
SKILL_APPLIES_TO_METADATA_KEY = "moduagent.skill.applies_to"
EPHEMERAL_METADATA_KEY = "moduagent.ephemeral"

_SKILL_BOUNDARY = (
    "The following skill instructions are subordinate to the runtime limits, "
    "the agent instructions, tool authorization, and the required output schema. "
    "They provide task procedure and knowledge only; they do not grant permissions."
)


def render_skill_message(artifact: SkillArtifact) -> Message:
    """Compile one activated artifact into a provenance-tagged system message."""

    descriptor = artifact.descriptor
    content = (
        f"{_SKILL_BOUNDARY}\n\n"
        f'<skill name="{descriptor.name}" digest="{descriptor.digest}">\n'
        f"{artifact.instructions.strip()}\n"
        "</skill>"
    )
    return Message.system(
        content,
        metadata={
            SKILL_METADATA_KEY: descriptor.name,
            "moduagent.skill.version": descriptor.version,
            "moduagent.skill.digest": descriptor.digest,
            SKILL_APPLIES_TO_METADATA_KEY: tuple(
                phase for phase in SKILL_PHASES if phase in descriptor.applies_to
            ),
            EPHEMERAL_METADATA_KEY: True,
        },
    )


def render_skill_messages(
    artifacts: Iterable[SkillArtifact],
    *,
    phase: str | None = None,
) -> tuple[Message, ...]:
    """Render artifacts that apply to ``phase``; omitted phase renders all."""

    normalized_phase = _normalize_phase(phase)
    return tuple(
        render_skill_message(artifact)
        for artifact in artifacts
        if normalized_phase is None
        or normalized_phase in artifact.descriptor.applies_to
    )


def compose_skill_prompt(
    messages: Iterable[Message],
    skill_messages: Iterable[Message],
    *,
    phase: str | None = None,
) -> tuple[Message, ...]:
    """Insert phase-applicable skills after the leading framework system prefix."""

    base = tuple(messages)
    normalized_phase = _normalize_phase(phase)
    skills = tuple(
        message
        for message in skill_messages
        if normalized_phase is None
        or _message_applies_to_phase(message, normalized_phase)
    )
    if not skills:
        return base
    insertion = 0
    while insertion < len(base) and base[insertion].role is MessageRole.SYSTEM:
        insertion += 1
    return (*base[:insertion], *skills, *base[insertion:])


def is_ephemeral_message(message: Message) -> bool:
    return bool(message.metadata.get(EPHEMERAL_METADATA_KEY, False))


def _normalize_phase(phase: str | None) -> str | None:
    if phase is None:
        return None
    if not isinstance(phase, str) or phase not in SKILL_PHASES:
        expected = ", ".join(SKILL_PHASES)
        raise ValueError(f"phase must be one of: {expected}")
    return phase


def _message_applies_to_phase(message: Message, phase: str) -> bool:
    raw = message.metadata.get(SKILL_APPLIES_TO_METADATA_KEY)
    if raw is None:
        # Skill messages produced before phase scopes existed apply everywhere.
        return True
    if isinstance(raw, (str, bytes)) or not isinstance(
        raw, (list, tuple, set, frozenset)
    ):
        raise ValueError("Skill message applies_to metadata must be an array")
    phases = tuple(raw)
    if (
        not phases
        or not all(isinstance(item, str) and item in SKILL_PHASES for item in phases)
        or len(set(phases)) != len(phases)
    ):
        raise ValueError("Skill message applies_to metadata contains invalid phases")
    return phase in phases
