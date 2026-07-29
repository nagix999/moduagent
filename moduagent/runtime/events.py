from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


class EventType(str, Enum):
    RUN_STARTED = "run_started"
    CHECKPOINT_LOADED = "checkpoint_loaded"
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
