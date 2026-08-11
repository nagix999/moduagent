from __future__ import annotations

import uuid
import math
from collections.abc import Callable
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


EVENT_SCHEMA_VERSION = 2


class EventType(str, Enum):
    RUN_STARTED = "run_started"
    CHECKPOINT_LOADED = "checkpoint_loaded"
    CHECKPOINT_SAVED = "checkpoint_saved"
    SKILLS_DISCOVERED = "skills_discovered"
    SKILL_SELECTION_STARTED = "skill_selection_started"
    SKILL_SELECTION_COMPLETED = "skill_selection_completed"
    SKILL_SELECTED = "skill_selected"
    SKILL_ACTIVATED = "skill_activated"
    SKILL_RESOURCE_READ = "skill_resource_read"
    SKILL_SKIPPED = "skill_skipped"
    SKILL_DENIED = "skill_denied"
    SKILL_ERROR = "skill_error"
    MODEL_STARTED = "model_started"
    MODEL_DELTA = "model_delta"
    MODEL_COMPLETED = "model_completed"
    MODEL_FAILED = "model_failed"
    MEMORY_COMPACTED = "memory_compacted"
    DELEGATION_REQUESTED = "delegation_requested"
    DELEGATION_AUTHORIZED = "delegation_authorized"
    DELEGATION_REJECTED = "delegation_rejected"
    DELEGATION_STARTED = "delegation_started"
    DELEGATION_RESUMED = "delegation_resumed"
    DELEGATION_COMPLETED = "delegation_completed"
    DELEGATION_FAILED = "delegation_failed"
    DELEGATION_RECONCILIATION_REQUIRED = "delegation_reconciliation_required"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_REPAIR_SCHEDULED = "tool_repair_scheduled"
    TOOL_REPAIR_EXHAUSTED = "tool_repair_exhausted"
    POLICY_DECISION = "policy_decision"
    PLAN_CREATED = "plan_created"
    STEP_STARTED = "step_started"
    STEP_MODEL_DELTA = "step_model_delta"
    STEP_RESULT_CREATED = "step_result_created"
    STEP_VALIDATED = "step_validated"
    STEP_COMMITTED = "step_committed"
    STEP_RETRY = "step_retry"
    STEP_FAILED = "step_failed"
    PLAN_REVISED = "plan_revised"
    FINALIZATION_STARTED = "finalization_started"
    FINAL_DELTA = "final_delta"
    FINALIZATION_COMPLETED = "finalization_completed"
    RETRY = "retry"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


class EventVisibility(str, Enum):
    INTERNAL = "internal"
    PUBLIC = "public"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    type: EventType
    run_id: str
    data: Mapping[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    visibility: EventVisibility = EventVisibility.PUBLIC
    # Appended for compatibility with positional 0.3 AgentEvent construction.
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    event_schema_version: int = EVENT_SCHEMA_VERSION
    session_id: str | None = None
    engine_id: str | None = None
    sequence: int = 0
    # Event schema v2 adds delegation identity without moving content into the
    # envelope. Fields are appended so positional v1 construction remains
    # source-compatible.
    execution_group_id: str | None = None
    root_run_id: str | None = None
    parent_run_id: str | None = None
    child_run_id: str | None = None
    delegation_id: str | None = None
    agent_id: str | None = None
    agent_version: str | None = None
    depth: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.type, EventType):
            object.__setattr__(self, "type", EventType(str(self.type)))
        if not isinstance(self.visibility, EventVisibility):
            object.__setattr__(
                self,
                "visibility",
                EventVisibility(str(self.visibility)),
            )
        for value, field_name in (
            (self.run_id, "run_id"),
            (self.event_id, "event_id"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} cannot be empty")
        for value, field_name in (
            (self.session_id, "session_id"),
            (self.engine_id, "engine_id"),
            (self.execution_group_id, "execution_group_id"),
            (self.root_run_id, "root_run_id"),
            (self.parent_run_id, "parent_run_id"),
            (self.child_run_id, "child_run_id"),
            (self.delegation_id, "delegation_id"),
            (self.agent_id, "agent_id"),
            (self.agent_version, "agent_version"),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} cannot be empty")
        if not isinstance(self.data, Mapping):
            raise TypeError("event data must be a mapping")
        object.__setattr__(self, "data", dict(self.data))
        if self.event_schema_version not in {1, EVENT_SCHEMA_VERSION}:
            raise ValueError("event_schema_version must be 1 or 2")
        # v1 is an accepted input format, not an output mode. Normalize it to
        # the complete v2 root envelope so schema_version and wire shape never
        # contradict one another.
        if self.event_schema_version == 1:
            object.__setattr__(self, "event_schema_version", EVENT_SCHEMA_VERSION)
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("event sequence must be a non-negative integer")
        if type(self.depth) is not int or self.depth < 0:
            raise ValueError("event depth must be a non-negative integer")
        if not isinstance(self.occurred_at, datetime):
            raise TypeError("occurred_at must be a datetime")
        object.__setattr__(self, "occurred_at", _as_utc(self.occurred_at))
        # Root events created through the legacy constructor still receive a
        # complete v2 correlation envelope.
        if self.root_run_id is None:
            object.__setattr__(self, "root_run_id", self.run_id)
        if self.execution_group_id is None:
            object.__setattr__(self, "execution_group_id", self.root_run_id)

    @property
    def event_type(self) -> EventType:
        """Additive alias matching the v1 event envelope terminology."""

        return self.type

    @property
    def timestamp(self) -> datetime:
        """Additive alias; ``occurred_at`` remains source-compatible."""

        return self.occurred_at

    def to_dict(self) -> dict[str, Any]:
        event_type = self.type.value
        timestamp = self.occurred_at.isoformat()
        return {
            # Legacy wire keys remain during the 0.4 compatibility window.
            "type": event_type,
            "occurred_at": timestamp,
            # Canonical envelope keys.
            "event_id": self.event_id,
            "event_type": event_type,
            "event_schema_version": self.event_schema_version,
            "visibility": self.visibility.value,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "engine_id": self.engine_id,
            "sequence": self.sequence,
            "execution_group_id": self.execution_group_id,
            "root_run_id": self.root_run_id,
            "parent_run_id": self.parent_run_id,
            "child_run_id": self.child_run_id,
            "delegation_id": self.delegation_id,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "depth": self.depth,
            "timestamp": timestamp,
            "data": _json_safe(self.data),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AgentEvent":
        """Decode v1 or v2 event envelopes without mutating the source."""

        if not isinstance(value, Mapping):
            raise ValueError("event payload must be an object")
        raw_schema_version = value.get("event_schema_version", 1)
        try:
            schema_version = int(raw_schema_version)
        except (TypeError, ValueError) as exc:
            raise ValueError("event_schema_version must be an integer") from exc
        if schema_version == EVENT_SCHEMA_VERSION:
            _validate_native_v2_payload(value)
        event_type = value.get("event_type", value.get("type"))
        occurred_at = value.get("timestamp", value.get("occurred_at"))
        if occurred_at is None:
            parsed_at = datetime.now(timezone.utc)
        elif isinstance(occurred_at, datetime):
            parsed_at = occurred_at
        else:
            try:
                parsed_at = datetime.fromisoformat(
                    str(occurred_at).replace("Z", "+00:00")
                )
            except ValueError as exc:
                raise ValueError("event timestamp must be ISO-8601") from exc
        data = value.get("data", {})
        if not isinstance(data, Mapping):
            raise ValueError("event data must be an object")
        sequence = int(value.get("sequence", 0))
        # Legacy v1 envelopes did not require a durable sequence. Once decoded,
        # they are normalized to the native v2 wire shape, whose sequence must
        # be positive so that the migrated event can be serialized and decoded
        # again without silently changing its correlation identity.
        if schema_version == 1 and sequence == 0:
            sequence = 1
        return cls(
            type=(
                event_type
                if isinstance(event_type, EventType)
                else EventType(str(event_type))
            ),
            run_id=str(value.get("run_id", "")),
            data=dict(data),
            occurred_at=parsed_at,
            visibility=(
                value.get("visibility")
                if isinstance(value.get("visibility"), EventVisibility)
                else EventVisibility(str(value.get("visibility", "public")))
            ),
            event_id=str(value.get("event_id") or uuid.uuid4().hex),
            event_schema_version=schema_version,
            session_id=_optional_text(value.get("session_id")),
            engine_id=_optional_text(value.get("engine_id")),
            sequence=sequence,
            execution_group_id=_optional_text(value.get("execution_group_id")),
            root_run_id=_optional_text(value.get("root_run_id")),
            parent_run_id=_optional_text(value.get("parent_run_id")),
            child_run_id=_optional_text(value.get("child_run_id")),
            delegation_id=_optional_text(value.get("delegation_id")),
            agent_id=_optional_text(value.get("agent_id")),
            agent_version=_optional_text(value.get("agent_version")),
            depth=int(value.get("depth", 0)),
        )


def _validate_native_v2_payload(value: Mapping[str, Any]) -> None:
    required = {
        "event_id",
        "event_type",
        "event_schema_version",
        "visibility",
        "run_id",
        "session_id",
        "engine_id",
        "sequence",
        "execution_group_id",
        "root_run_id",
        "parent_run_id",
        "child_run_id",
        "delegation_id",
        "agent_id",
        "agent_version",
        "depth",
        "timestamp",
        "data",
    }
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(
            "native event v2 is missing envelope fields: " + ", ".join(missing)
        )
    for field_name in (
        "event_id",
        "event_type",
        "run_id",
        "execution_group_id",
        "root_run_id",
        "timestamp",
    ):
        field_value = value[field_name]
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"native event v2 {field_name} cannot be empty")
    if type(value["sequence"]) is not int or value["sequence"] < 1:
        raise ValueError("native event v2 sequence must be a positive integer")
    if type(value["depth"]) is not int or value["depth"] < 0:
        raise ValueError("native event v2 depth must be a non-negative integer")
    depth = value["depth"]
    event_type = str(value["event_type"])
    related = event_type.startswith("delegation_")
    child_run_id = value["child_run_id"]
    delegation_id = value["delegation_id"]
    if (child_run_id is None) != (delegation_id is None):
        raise ValueError(
            "native event v2 child_run_id and delegation_id must appear together"
        )
    if depth == 0:
        if value["root_run_id"] != value["run_id"]:
            raise ValueError("native root event v2 root_run_id must match run_id")
        if value["parent_run_id"] is not None:
            raise ValueError("native root event v2 cannot contain parent_run_id")
        if child_run_id is not None and not related:
            raise ValueError(
                "only delegation lifecycle events can reference a related child"
            )
    else:
        if not isinstance(value["parent_run_id"], str) or not value["parent_run_id"]:
            raise ValueError("native child event v2 requires parent_run_id")
        if delegation_id is None or (not related and child_run_id != value["run_id"]):
            raise ValueError(
                "native child event v2 requires its run and delegation identity"
            )
    if (value["agent_id"] is None) != (value["agent_version"] is None):
        raise ValueError(
            "native event v2 agent_id and agent_version must appear together"
        )


class EventPublisher:
    """Run-scoped factory that stamps identity and monotonic sequence values."""

    def __init__(
        self,
        *,
        run_id: str,
        session_id: str,
        engine_id: str,
        initial_sequence: int = 0,
        event_schema_version: int = EVENT_SCHEMA_VERSION,
        execution_group_id: str | None = None,
        root_run_id: str | None = None,
        parent_run_id: str | None = None,
        delegation_id: str | None = None,
        agent_id: str | None = None,
        agent_version: str | None = None,
        depth: int = 0,
        clock: Callable[[], datetime] | None = None,
        event_id_factory: Callable[[], str] | None = None,
    ) -> None:
        for value, field_name in (
            (run_id, "run_id"),
            (session_id, "session_id"),
            (engine_id, "engine_id"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} cannot be empty")
        if type(initial_sequence) is not int or initial_sequence < 0:
            raise ValueError("initial_sequence must be a non-negative integer")
        if type(event_schema_version) is not int or event_schema_version < 1:
            raise ValueError("event_schema_version must be a positive integer")
        self.run_id = run_id
        self.session_id = session_id
        self.engine_id = engine_id
        self.event_schema_version = event_schema_version
        self.root_run_id = run_id if root_run_id is None else root_run_id
        self.execution_group_id = (
            self.root_run_id if execution_group_id is None else execution_group_id
        )
        self.parent_run_id = parent_run_id
        self.delegation_id = delegation_id
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.depth = depth
        for value, field_name in (
            (self.execution_group_id, "execution_group_id"),
            (self.root_run_id, "root_run_id"),
            (self.parent_run_id, "parent_run_id"),
            (self.delegation_id, "delegation_id"),
            (self.agent_id, "agent_id"),
            (self.agent_version, "agent_version"),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} cannot be empty")
        if type(depth) is not int or depth < 0:
            raise ValueError("depth must be a non-negative integer")
        self._sequence = initial_sequence
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._event_id_factory = event_id_factory or (lambda: uuid.uuid4().hex)

    @property
    def last_sequence(self) -> int:
        return self._sequence

    def create(
        self,
        event_type: EventType,
        data: Mapping[str, Any] | None = None,
        *,
        visibility: EventVisibility = EventVisibility.PUBLIC,
        occurred_at: datetime | None = None,
    ) -> AgentEvent:
        self._sequence += 1
        return AgentEvent(
            type=event_type,
            run_id=self.run_id,
            data=dict(data or {}),
            occurred_at=occurred_at or self._clock(),
            visibility=visibility,
            event_id=self._event_id_factory(),
            event_schema_version=self.event_schema_version,
            session_id=self.session_id,
            engine_id=self.engine_id,
            sequence=self._sequence,
            execution_group_id=self.execution_group_id,
            root_run_id=self.root_run_id,
            parent_run_id=self.parent_run_id,
            child_run_id=(self.run_id if self.parent_run_id is not None else None),
            delegation_id=self.delegation_id,
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            depth=self.depth,
        )

    def stamp(
        self,
        event: AgentEvent,
        *,
        allow_related_delegation: bool = False,
    ) -> AgentEvent:
        """Stamp an Engine-created compatibility event before publication."""

        if not isinstance(event, AgentEvent):
            raise TypeError("event must be an AgentEvent")
        if event.run_id != self.run_id:
            raise ValueError("event run_id does not match the publisher")
        if event.session_id not in (None, self.session_id):
            raise ValueError("event session_id does not match the publisher")
        if event.engine_id not in (None, self.engine_id):
            raise ValueError("event engine_id does not match the publisher")
        legacy_root_placeholder = (
            event.sequence == 0
            and event.execution_group_id == event.run_id
            and event.root_run_id == event.run_id
            and event.parent_run_id is None
            and event.child_run_id is None
            and event.delegation_id is None
            and event.depth == 0
        )
        if event.execution_group_id not in (None, self.execution_group_id) and not (
            legacy_root_placeholder
        ):
            raise ValueError("event execution_group_id does not match the publisher")
        if type(allow_related_delegation) is not bool:
            raise TypeError("allow_related_delegation must be a bool")
        canonical_identity = {
            "root_run_id": self.root_run_id,
            "parent_run_id": self.parent_run_id,
            "agent_id": self.agent_id,
            "agent_version": self.agent_version,
            "depth": self.depth,
        }
        if not legacy_root_placeholder:
            for field_name, expected in canonical_identity.items():
                if getattr(event, field_name) != expected:
                    raise ValueError(f"event {field_name} does not match the publisher")
        canonical_child_run_id = self.run_id if self.parent_run_id is not None else None
        if allow_related_delegation:
            if (
                event.type.value.startswith("delegation_") is False
                or not isinstance(event.child_run_id, str)
                or not event.child_run_id
                or not isinstance(event.delegation_id, str)
                or not event.delegation_id
            ):
                raise ValueError(
                    "related delegation events require child and delegation identity"
                )
            stamped_child_run_id = event.child_run_id
            stamped_delegation_id = event.delegation_id
        else:
            if not legacy_root_placeholder and (
                event.child_run_id != canonical_child_run_id
                or event.delegation_id != self.delegation_id
            ):
                raise ValueError(
                    "event delegation correlation does not match the publisher"
                )
            stamped_child_run_id = canonical_child_run_id
            stamped_delegation_id = self.delegation_id
        if event.sequence == 0:
            self._sequence += 1
            sequence = self._sequence
        else:
            if event.sequence <= self._sequence:
                raise ValueError("event sequence must increase monotonically")
            self._sequence = event.sequence
            sequence = event.sequence
        return replace(
            event,
            session_id=self.session_id,
            engine_id=self.engine_id,
            sequence=sequence,
            event_schema_version=self.event_schema_version,
            execution_group_id=self.execution_group_id,
            root_run_id=self.root_run_id,
            parent_run_id=self.parent_run_id,
            child_run_id=stamped_child_run_id,
            delegation_id=stamped_delegation_id,
            agent_id=self.agent_id,
            agent_version=self.agent_version,
            depth=self.depth,
        )


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _json_safe(value: Any, _seen: set[int] | None = None) -> Any:
    seen = set() if _seen is None else _seen
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return _as_utc(value).isoformat()
    recursive = (
        isinstance(value, (Mapping, list, tuple, set, frozenset))
        or (is_dataclass(value) and not isinstance(value, type))
        or callable(getattr(value, "to_dict", None))
        or callable(getattr(value, "model_dump", None))
    )
    value_id = id(value)
    if recursive and value_id in seen:
        return {"unsupported_type": "recursive_reference"}
    if recursive:
        seen.add(value_id)
    try:
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item, seen) for key, item in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [_json_safe(item, seen) for item in value]
        if is_dataclass(value) and not isinstance(value, type):
            return {
                item.name: _json_safe(getattr(value, item.name), seen)
                for item in fields(value)
            }
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            return _json_safe(to_dict(), seen)
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return _json_safe(model_dump(mode="json"), seen)
    finally:
        if recursive:
            seen.remove(value_id)
    value_type = type(value)
    qualified_name = f"{value_type.__module__}.{value_type.__qualname__}"
    return {"unsupported_type": qualified_name[:256]}


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = [
    "AgentEvent",
    "EVENT_SCHEMA_VERSION",
    "EventPublisher",
    "EventType",
    "EventVisibility",
]
