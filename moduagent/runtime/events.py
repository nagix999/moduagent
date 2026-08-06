from __future__ import annotations

import uuid
import math
from collections.abc import Callable
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


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
    event_schema_version: int = 1
    session_id: str | None = None
    engine_id: str | None = None
    sequence: int = 0

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
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field_name} cannot be empty")
        if not isinstance(self.data, Mapping):
            raise TypeError("event data must be a mapping")
        object.__setattr__(self, "data", dict(self.data))
        if type(self.event_schema_version) is not int or self.event_schema_version < 1:
            raise ValueError("event_schema_version must be a positive integer")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("event sequence must be a non-negative integer")
        if not isinstance(self.occurred_at, datetime):
            raise TypeError("occurred_at must be a datetime")
        object.__setattr__(self, "occurred_at", _as_utc(self.occurred_at))

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
            "timestamp": timestamp,
            "data": _json_safe(self.data),
        }


class EventPublisher:
    """Run-scoped factory that stamps identity and monotonic sequence values."""

    def __init__(
        self,
        *,
        run_id: str,
        session_id: str,
        engine_id: str,
        initial_sequence: int = 0,
        event_schema_version: int = 1,
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
        )

    def stamp(self, event: AgentEvent) -> AgentEvent:
        """Stamp an Engine-created compatibility event before publication."""

        if not isinstance(event, AgentEvent):
            raise TypeError("event must be an AgentEvent")
        if event.run_id != self.run_id:
            raise ValueError("event run_id does not match the publisher")
        if event.session_id not in (None, self.session_id):
            raise ValueError("event session_id does not match the publisher")
        if event.engine_id not in (None, self.engine_id):
            raise ValueError("event engine_id does not match the publisher")
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
        )


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
    "EventPublisher",
    "EventType",
    "EventVisibility",
]
