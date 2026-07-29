from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

from moduagent.messages import FinishReason, Message, MessageRole, Usage
from moduagent.models import ModelGateway


_SKILL_MODES = frozenset({"disabled", "explicit", "auto", "hybrid"})
_SKILL_PHASES = ("plan", "act", "finalize")


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    WAITING_FOR_MODEL = "waiting_for_model"
    WAITING_FOR_TOOLS = "waiting_for_tools"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class RunRequest:
    input: str
    session_id: str
    user_context: Mapping[str, Any] = field(default_factory=dict)
    resume_run_id: str | None = None
    requested_skills: tuple[str, ...] = ()
    skill_mode: str = "disabled"

    def __post_init__(self) -> None:
        requested_skills = tuple(self.requested_skills)
        if not all(
            isinstance(skill, str) and skill.strip() for skill in requested_skills
        ):
            raise ValueError("requested_skills must contain non-empty strings")
        if len(set(requested_skills)) != len(requested_skills):
            raise ValueError("requested_skills cannot contain duplicates")
        if self.skill_mode not in _SKILL_MODES:
            expected = ", ".join(sorted(_SKILL_MODES))
            raise ValueError(f"skill_mode must be one of: {expected}")
        object.__setattr__(self, "requested_skills", requested_skills)


@dataclass(frozen=True, slots=True)
class SkillActivationState:
    """Serializable identity and grant snapshot for an activated skill.

    Runtime state deliberately uses primitive values instead of depending on
    ``moduagent.skills`` domain objects. This keeps checkpoints readable when a
    custom registry implementation is used and lets older checkpoints migrate
    without importing the optional skill runtime.
    """

    name: str
    version: str | None = None
    digest: str = ""
    source_id: str = ""
    selected_by: str = "explicit"
    allowed_tools: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    applies_to: tuple[str, ...] = _SKILL_PHASES

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("skill name cannot be empty")
        if self.version is not None and not isinstance(self.version, str):
            raise TypeError("skill version must be a string or None")
        for value, field_name in (
            (self.digest, "digest"),
            (self.source_id, "source_id"),
            (self.selected_by, "selected_by"),
        ):
            if not isinstance(value, str):
                raise TypeError(f"skill {field_name} must be a string")
        allowed_tools = tuple(self.allowed_tools)
        if not all(isinstance(tool, str) and tool.strip() for tool in allowed_tools):
            raise ValueError("allowed_tools must contain non-empty strings")
        if len(set(allowed_tools)) != len(allowed_tools):
            raise ValueError("allowed_tools cannot contain duplicates")
        if isinstance(self.applies_to, (str, bytes)):
            raise TypeError("applies_to must be an array of Skill phases")
        applies_to = tuple(self.applies_to)
        if not applies_to:
            raise ValueError("applies_to must contain at least one Skill phase")
        if not all(isinstance(phase, str) for phase in applies_to):
            raise TypeError("applies_to must contain Skill phase strings")
        if len(set(applies_to)) != len(applies_to):
            raise ValueError("applies_to cannot contain duplicate Skill phases")
        if set(applies_to) - set(_SKILL_PHASES):
            expected = ", ".join(_SKILL_PHASES)
            raise ValueError(f"applies_to must contain only: {expected}")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("skill metadata must be a mapping")
        object.__setattr__(self, "allowed_tools", allowed_tools)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(
            self,
            "applies_to",
            tuple(phase for phase in _SKILL_PHASES if phase in applies_to),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "digest": self.digest,
            "source_id": self.source_id,
            "selected_by": self.selected_by,
            "allowed_tools": list(self.allowed_tools),
            "metadata": dict(self.metadata),
            "applies_to": list(self.applies_to),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SkillActivationState":
        if not isinstance(value, Mapping):
            raise ValueError("skill activation state must be an object")
        raw_allowed_tools = value.get("allowed_tools", ())
        if isinstance(raw_allowed_tools, (str, bytes)) or not isinstance(
            raw_allowed_tools, (list, tuple)
        ):
            raise ValueError("skill allowed_tools must be an array")
        raw_metadata = value.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise ValueError("skill metadata must be an object")
        raw_applies_to = value.get("applies_to", _SKILL_PHASES)
        if isinstance(raw_applies_to, (str, bytes)) or not isinstance(
            raw_applies_to, (list, tuple)
        ):
            raise ValueError("skill applies_to must be an array")
        raw_version = value.get("version")
        return cls(
            name=str(value.get("name", "")),
            version=None if raw_version is None else str(raw_version),
            digest=str(value.get("digest", "")),
            source_id=str(value.get("source_id", "")),
            selected_by=str(value.get("selected_by", "explicit")),
            allowed_tools=tuple(str(tool) for tool in raw_allowed_tools),
            metadata=dict(raw_metadata),
            applies_to=tuple(raw_applies_to),
        )


@dataclass(frozen=True, slots=True)
class SkillRunState:
    """Skill catalog snapshot pinned to a single agent run."""

    catalog_digest: str = ""
    active_skills: tuple[SkillActivationState, ...] = ()
    resource_reads: int = 0
    instruction_tokens: int = 0
    resource_tokens: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.catalog_digest, str):
            raise TypeError("catalog_digest must be a string")
        active_skills = tuple(self.active_skills)
        if not all(
            isinstance(activation, SkillActivationState) for activation in active_skills
        ):
            raise TypeError("active_skills must contain SkillActivationState instances")
        names = [activation.name for activation in active_skills]
        if len(set(names)) != len(names):
            raise ValueError("active_skills cannot contain duplicate names")
        for value, field_name in (
            (self.resource_reads, "resource_reads"),
            (self.instruction_tokens, "instruction_tokens"),
            (self.resource_tokens, "resource_tokens"),
        ):
            if value < 0:
                raise ValueError(f"{field_name} cannot be negative")
        object.__setattr__(self, "active_skills", active_skills)

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_digest": self.catalog_digest,
            "active_skills": [skill.to_dict() for skill in self.active_skills],
            "resource_reads": self.resource_reads,
            "instruction_tokens": self.instruction_tokens,
            "resource_tokens": self.resource_tokens,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SkillRunState":
        if not isinstance(value, Mapping):
            raise ValueError("skill run state must be an object")
        raw_active_skills = value.get("active_skills", ())
        if isinstance(raw_active_skills, (str, bytes)) or not isinstance(
            raw_active_skills, (list, tuple)
        ):
            raise ValueError("active_skills must be an array")
        return cls(
            catalog_digest=str(value.get("catalog_digest", "")),
            active_skills=tuple(
                SkillActivationState.from_dict(skill) for skill in raw_active_skills
            ),
            resource_reads=int(value.get("resource_reads", 0)),
            instruction_tokens=int(value.get("instruction_tokens", 0)),
            resource_tokens=int(value.get("resource_tokens", 0)),
        )


@dataclass(slots=True)
class RunContext:
    run_id: str
    request: RunRequest
    messages: list[Message]
    new_messages: list[Message] = field(default_factory=list)
    # Strict Plan-and-Execute keeps model/tool transcripts here so they can be
    # checkpointed without becoming public conversation history.
    internal_messages: list[Message] = field(default_factory=list, repr=False)
    execution_state: Any = None
    step: int = 0
    tool_call_count: int = 0
    status: RunStatus = RunStatus.CREATED
    policy_state: dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    metadata: dict[str, Any] = field(default_factory=dict)
    current_run_start: int = 0
    skill_state: SkillRunState = field(default_factory=SkillRunState)
    # Prompt-only instructions reconstructed from the pinned skill state. They
    # are deliberately omitted from checkpoints, conversation storage, and
    # AgentResult.messages; resume rebuilds them from the immutable digest.
    skill_messages: tuple[Message, ...] = field(default_factory=tuple, repr=False)
    # Ephemeral run-scoped provider boundary. It is intentionally absent from
    # snapshots and is rebound by the Coordinator on every run/resume.
    model_gateway: ModelGateway | None = field(default=None, repr=False)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc),
        repr=False,
    )
    # Diagnostics are operational observers, not durable execution state.
    # Appended after all 0.4.0 fields to preserve positional construction.
    # They are deliberately absent from checkpoint serializers.
    diagnostic_reporter: Any | None = field(default=None, repr=False)
    primary_failure: Mapping[str, Any] | None = field(default=None, repr=False)
    tool_failure_ids: dict[str, str] = field(default_factory=dict, repr=False)

    def add_message(self, message: Message, *, persist: bool = True) -> None:
        self.messages.append(message)
        if persist and message.role is not MessageRole.SYSTEM:
            self.new_messages.append(message)

    def add_internal_message(self, message: Message) -> None:
        self.internal_messages.append(message)

    def clear_internal_messages(self) -> None:
        self.internal_messages.clear()


@dataclass(frozen=True, slots=True)
class AgentResult:
    run_id: str
    output: Any
    messages: tuple[Message, ...]
    usage: Usage
    finish_reason: FinishReason
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def failure_id(self) -> str | None:
        """Return the terminal failure correlation ID when diagnostics are enabled."""

        summary = self.metadata.get("error_summary")
        if not isinstance(summary, Mapping):
            return None
        value = summary.get("failure_id")
        return value if isinstance(value, str) and value else None
