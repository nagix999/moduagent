from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from moduagent.config import AgentConfig
from moduagent.messages import FinishReason, ToolCall
from moduagent.models import (
    ModelCapabilities,
    ModelChunk,
    ModelRequest,
    ModelResponse,
)
from moduagent.runtime.context import RunContext
from moduagent.runtime.events import AgentEvent, EventType, EventVisibility
from moduagent.tools import ToolResult, ToolSchema
from moduagent.execution.state import EngineSnapshot, EngineStateCodec

try:
    # These are public 0.4 Tool contracts. The direct-module fallback keeps this
    # package importable while the Tool package is upgraded independently.
    from moduagent.tools import ToolBatchOutcome, ToolRepairConstraint
except ImportError:  # pragma: no cover - exercised only during staggered upgrades
    from moduagent.tools.runtime import ToolBatchOutcome, ToolRepairConstraint


StateT = TypeVar("StateT")


class DurableBoundary(str, Enum):
    """Stable points at which an Engine requests a durable snapshot."""

    INITIALIZED = "initialized"
    BEFORE_MODEL = "before_model"
    TOOL_INVOCATION_PENDING = "tool_invocation_pending"
    AFTER_TOOL_OUTCOME = "after_tool_outcome"
    REPAIR_SCHEDULED = "repair_scheduled"
    REPLAN_COMPLETED = "replan_completed"
    STEP_RESULT_PENDING = "step_result_pending"
    STEP_COMMITTED = "step_committed"
    FINALIZATION_STARTED = "finalization_started"
    FINALIZATION_RESPONSE = "finalization_response"
    FINALIZATION_PERSISTED = "finalization_persisted"
    FINALIZATION_EMITTED = "finalization_emitted"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class ExecutionBudget:
    """Engine-neutral limits resolved once for the active run."""

    max_steps: int
    max_tool_calls: int
    max_step_attempts: int
    max_replans: int
    max_tool_repair_attempts: int
    parallel_tool_calls: bool = False
    max_parallel_tools: int = 4

    def __post_init__(self) -> None:
        for field_name in (
            "max_steps",
            "max_step_attempts",
            "max_parallel_tools",
        ):
            if type(getattr(self, field_name)) is not int:
                raise TypeError(f"{field_name} must be an integer")
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be at least 1")
        for field_name in (
            "max_tool_calls",
            "max_replans",
            "max_tool_repair_attempts",
        ):
            if type(getattr(self, field_name)) is not int:
                raise TypeError(f"{field_name} must be an integer")
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        if type(self.parallel_tool_calls) is not bool:
            raise TypeError("parallel_tool_calls must be a bool")

    @classmethod
    def from_config(cls, config: AgentConfig) -> ExecutionBudget:
        limits = config.limits
        return cls(
            max_steps=limits.max_steps,
            max_tool_calls=limits.max_tool_calls,
            max_step_attempts=limits.max_step_attempts,
            max_replans=limits.max_replans,
            max_tool_repair_attempts=limits.max_tool_repair_attempts,
            parallel_tool_calls=limits.parallel_tool_calls,
            max_parallel_tools=limits.max_parallel_tools,
        )


@dataclass(frozen=True, slots=True)
class EngineContext:
    """Common immutable inputs supplied to an ExecutionEngine."""

    run: RunContext
    config: AgentConfig
    stream_model: bool = False
    resolved_spec: Mapping[str, Any] = field(default_factory=dict)
    model_capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunContext):
            raise TypeError("run must be a RunContext")
        if not isinstance(self.config, AgentConfig):
            raise TypeError("config must be an AgentConfig")
        if type(self.stream_model) is not bool:
            raise TypeError("stream_model must be a bool")
        if not isinstance(self.resolved_spec, Mapping):
            raise TypeError("resolved_spec must be a mapping")
        if not isinstance(self.model_capabilities, ModelCapabilities):
            raise TypeError("model_capabilities must be ModelCapabilities")
        object.__setattr__(self, "resolved_spec", dict(self.resolved_spec))


@dataclass(frozen=True, slots=True)
class ResumeValidation:
    compatible: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if type(self.compatible) is not bool:
            raise TypeError("compatible must be a bool")
        if not isinstance(self.reason, str):
            raise TypeError("reason must be a string")
        if not self.compatible and not self.reason.strip():
            raise ValueError("incompatible resume validation requires a reason")

    @classmethod
    def accepted(cls) -> ResumeValidation:
        return cls(True)

    @classmethod
    def rejected(cls, reason: str) -> ResumeValidation:
        return cls(False, reason)


@dataclass(frozen=True, slots=True)
class EngineOutcome:
    """Terminal Engine result consumed by RunCoordinator."""

    finish_reason: FinishReason
    output: Any = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.finish_reason, FinishReason):
            object.__setattr__(
                self,
                "finish_reason",
                FinishReason(str(self.finish_reason)),
            )
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class EngineEmission:
    """One streamed event or the single terminal Engine outcome."""

    event: AgentEvent | None = None
    outcome: EngineOutcome | None = None

    def __post_init__(self) -> None:
        if (self.event is None) == (self.outcome is None):
            raise ValueError("an EngineEmission must contain exactly one value")
        if self.event is not None and not isinstance(self.event, AgentEvent):
            raise TypeError("event must be an AgentEvent")
        if self.outcome is not None and not isinstance(
            self.outcome,
            EngineOutcome,
        ):
            raise TypeError("outcome must be an EngineOutcome")


@dataclass(frozen=True, slots=True)
class FinalizationResult:
    """Validated output returned by the common ResultFinalizer service."""

    response: ModelResponse
    output: Any
    content: str
    buffered_deltas: tuple[str, ...] = ()
    persisted: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.response, ModelResponse):
            raise TypeError("response must be a ModelResponse")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("finalization content cannot be empty")
        if type(self.buffered_deltas) is not tuple:
            raise TypeError("buffered_deltas must be a tuple")
        if not all(isinstance(item, str) for item in self.buffered_deltas):
            raise TypeError("buffered_deltas must contain strings")
        if type(self.persisted) is not bool:
            raise TypeError("persisted must be a bool")


@runtime_checkable
class ExecutionServices(Protocol):
    """Engine-neutral operational services.

    Implementations own provider calls, Tool execution, persistence and sink
    isolation. Engines own request order, phase transitions and budgets.
    """

    def budget(self, context: EngineContext) -> ExecutionBudget: ...

    def remaining_seconds(self, context: EngineContext) -> float: ...

    def tool_schemas(
        self,
        context: EngineContext,
        names: frozenset[str] | None = None,
    ) -> tuple[ToolSchema, ...]: ...

    def output_schema(
        self,
        context: EngineContext,
    ) -> Mapping[str, Any] | None: ...

    def decode_output(
        self,
        context: EngineContext,
        response: ModelResponse,
    ) -> Any: ...

    async def prepare_model_request(
        self,
        context: EngineContext,
        request: ModelRequest,
        *,
        phase: str,
        skill_phase: str | None,
        protected_from: int | None = None,
    ) -> ModelRequest: ...

    async def request_model(
        self,
        context: EngineContext,
        request: ModelRequest,
        *,
        phase: str,
    ) -> ModelResponse: ...

    def stream_model(
        self,
        context: EngineContext,
        request: ModelRequest,
        *,
        phase: str,
        delta_event_type: EventType | None = EventType.MODEL_DELTA,
        delta_visibility: EventVisibility = EventVisibility.PUBLIC,
        delta_data: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[ModelChunk]: ...

    async def execute_tool_batch(
        self,
        context: EngineContext,
        calls: tuple[ToolCall, ...],
        *,
        allowed_tools: frozenset[str] | None,
        repair_constraint: ToolRepairConstraint | None,
    ) -> ToolBatchOutcome: ...

    async def record_tool_result(
        self,
        context: EngineContext,
        call: ToolCall,
        result: ToolResult,
    ) -> None:
        """Record one synthetic Tool result in the common audit trace."""

        ...

    async def finalize(
        self,
        context: EngineContext,
        request: ModelRequest,
        *,
        phase: str,
    ) -> FinalizationResult: ...

    async def persist_finalization(
        self,
        context: EngineContext,
        result: FinalizationResult,
    ) -> FinalizationResult:
        """Persist one public final message and return ``persisted=True``."""

        ...

    async def emit_finalization(
        self,
        context: EngineContext,
        result: FinalizationResult,
        *,
        phase: str,
    ) -> None:
        """Emit public final deltas through the common finalization boundary."""

        ...

    async def checkpoint(
        self,
        context: EngineContext,
        snapshot: EngineSnapshot,
        *,
        boundary: DurableBoundary,
    ) -> None: ...

    async def publish_event(
        self,
        context: EngineContext,
        event: AgentEvent,
    ) -> AgentEvent: ...

    async def defer_event(
        self,
        context: EngineContext,
        event: AgentEvent,
    ) -> None:
        """Publish an initialization/service event for later stream delivery."""

        ...


@runtime_checkable
class ExecutionEngine(Protocol[StateT]):
    """Explicit execution-algorithm contract selected by composition."""

    engine_id: str
    state_version: int
    state_codec: EngineStateCodec[StateT]
    configuration: Mapping[str, Any]
    required_capabilities: Mapping[str, bool]
    retain_terminal_checkpoint: bool

    async def initialize(
        self,
        context: EngineContext,
        services: ExecutionServices,
    ) -> StateT: ...

    def execute(
        self,
        context: EngineContext,
        state: StateT,
        services: ExecutionServices,
    ) -> AsyncIterator[EngineEmission]: ...

    def encode_state(self, state: StateT) -> Mapping[str, Any]: ...

    def decode_state(self, payload: Mapping[str, Any]) -> StateT: ...

    def migrate_state(
        self,
        from_version: int,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def validate_resume(
        self,
        snapshot: EngineSnapshot,
        resolved_spec: Mapping[str, Any],
    ) -> ResumeValidation: ...


class CodecBackedEngine(Generic[StateT]):
    """Reusable codec and resume behavior for concrete Engines."""

    engine_id: str
    state_version: int
    state_codec: EngineStateCodec[StateT]
    configuration: Mapping[str, Any] = MappingProxyType({})
    required_capabilities: Mapping[str, bool] = MappingProxyType({})
    retain_terminal_checkpoint: bool = False

    def encode_state(self, state: StateT) -> Mapping[str, Any]:
        return dict(self.state_codec.encode(state))

    def decode_state(self, payload: Mapping[str, Any]) -> StateT:
        return self.state_codec.decode(payload)

    def migrate_state(
        self,
        from_version: int,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return dict(self.state_codec.migrate(from_version, payload))

    def validate_resume(
        self,
        snapshot: EngineSnapshot,
        resolved_spec: Mapping[str, Any],
    ) -> ResumeValidation:
        if not isinstance(snapshot, EngineSnapshot):
            return ResumeValidation.rejected("snapshot must be an EngineSnapshot")
        if snapshot.engine_id != self.engine_id:
            return ResumeValidation.rejected(
                f"checkpoint engine {snapshot.engine_id!r} does not match "
                f"{self.engine_id!r}"
            )
        if not isinstance(resolved_spec, Mapping):
            return ResumeValidation.rejected("resolved_spec must be a mapping")
        configured_engine = resolved_spec.get("engine_id")
        if configured_engine is not None and configured_engine != self.engine_id:
            return ResumeValidation.rejected(
                "resolved Agent configuration selects a different Engine"
            )
        try:
            payload = (
                snapshot.state
                if snapshot.state_version == self.state_version
                else self.migrate_state(snapshot.state_version, snapshot.state)
            )
            self.decode_state(payload)
        except Exception as exc:
            return ResumeValidation.rejected(
                f"Engine state cannot be resumed: {type(exc).__name__}"
            )
        return ResumeValidation.accepted()

    async def _checkpoint(
        self,
        context: EngineContext,
        state: StateT,
        services: ExecutionServices,
        boundary: DurableBoundary,
    ) -> None:
        # ``checkpointing_enabled`` is an optional concrete-service capability,
        # not part of the required ExecutionServices protocol. Third-party
        # services that predate it keep the conservative 0.4 behavior, while the
        # built-in runtime can avoid encoding an otherwise discarded snapshot.
        if getattr(services, "checkpointing_enabled", True) is False:
            return
        await services.checkpoint(
            context,
            EngineSnapshot(
                engine_id=self.engine_id,
                state_version=self.state_version,
                state=self.encode_state(state),
            ),
            boundary=boundary,
        )

    @staticmethod
    async def _publish(
        context: EngineContext,
        services: ExecutionServices,
        event: AgentEvent,
    ) -> EngineEmission:
        published = await services.publish_event(context, event)
        if not isinstance(published, AgentEvent):
            raise TypeError("publish_event must return an AgentEvent")
        return EngineEmission(event=published)


__all__ = [
    "CodecBackedEngine",
    "DurableBoundary",
    "EngineContext",
    "EngineEmission",
    "EngineOutcome",
    "EngineSnapshot",
    "EngineStateCodec",
    "ExecutionBudget",
    "ExecutionEngine",
    "ExecutionServices",
    "FinalizationResult",
    "ResumeValidation",
]
