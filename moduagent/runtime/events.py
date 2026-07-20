from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


class EventType(str, Enum):
    RUN_STARTED = "run_started"
    CHECKPOINT_LOADED = "checkpoint_loaded"
    MODEL_STARTED = "model_started"
    MODEL_DELTA = "model_delta"
    MODEL_COMPLETED = "model_completed"
    MEMORY_COMPACTED = "memory_compacted"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    POLICY_DECISION = "policy_decision"
    RETRY = "retry"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


@dataclass(frozen=True, slots=True)
class AgentEvent:
    type: EventType
    run_id: str
    data: Mapping[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
