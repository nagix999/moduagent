from __future__ import annotations

import hashlib
import json
import re
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from types import MappingProxyType
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from moduagent.config import AgentConfig, RetryConfig, RunLimits
from moduagent.decision import (
    DecisionPolicy,
    PlanAndExecutePolicy,
    PlanGenerator,
    StandardDecisionPolicy,
    StepValidator,
    ToolFailureRecoveryConfig,
)
from moduagent.errors import CapabilityError, ConfigurationError
from moduagent.execution import (
    EngineStateCodec,
    ExecutionEngine,
    PlanExecutionEngine,
    StandardExecutionEngine,
)
from moduagent.memory import (
    ConversationMemoryPolicy,
    FullConversationMemoryPolicy,
    ModelConversationSummarizer,
    TokenBudgetConversationMemoryPolicy,
)
from moduagent.models import ModelCapabilities, ModelClient
from moduagent.observability import (
    DiagnosticReporter,
    DiagnosticSink,
    EventSink,
    NoopDiagnosticSink,
    NoopEventSink,
)
from moduagent.output import OutputCodec, TextOutputCodec
from moduagent.persistence import (
    CheckpointStore,
    ConversationStore,
    InMemoryConversationStore,
)
from moduagent.runtime.coordinator import RunCoordinator
from moduagent.skills import (
    HybridSkillSelector,
    ModelSkillSelector,
    SkillLimits,
    SkillRegistry,
    SkillSelector,
)
from moduagent.skills.runtime import SkillRuntime
from moduagent.skills.tools import SkillReadTool, SkillSearchTool
from moduagent.tools import (
    AllowAllAuthorizer,
    Tool,
    ToolAuthorizer,
    ToolExecutor,
    ToolRegistry,
)
from moduagent.tools.failure import ToolSafetyProfile, resolve_tool_safety_profile


@dataclass(frozen=True, slots=True)
class StandardExecutionProfile:
    """Configuration for the ordinary model/Tool execution engine.

    ``decision_policy`` is an advanced compatibility hook. Most applications
    should use the default policy.
    """

    decision_policy: DecisionPolicy | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    engine_id: str = field(default="standard", init=False)
    state_version: int = field(default=1, init=False)


@dataclass(frozen=True, slots=True)
class PlanExecutionProfile:
    """Configuration for strict, validated Plan-and-Execute execution."""

    plan_generator: PlanGenerator = field(repr=False, compare=False)
    step_validator: StepValidator | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    revise_on_tool_failure: bool = True
    max_step_attempts: int | None = None
    max_replans: int | None = None
    tool_failure_recovery: ToolFailureRecoveryConfig | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    engine_id: str = field(default="plan", init=False)
    state_version: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if not callable(getattr(self.plan_generator, "create", None)):
            raise TypeError("plan_generator must provide create()")
        if not callable(getattr(self.plan_generator, "revise", None)):
            raise TypeError("plan_generator must provide revise()")
        if not isinstance(self.revise_on_tool_failure, bool):
            raise TypeError("revise_on_tool_failure must be a bool")
        if self.max_step_attempts is not None and self.max_step_attempts < 1:
            raise ValueError("max_step_attempts must be at least 1")
        if self.max_replans is not None and self.max_replans < 0:
            raise ValueError("max_replans cannot be negative")

    def create_policy(self) -> PlanAndExecutePolicy:
        return PlanAndExecutePolicy(
            self.plan_generator,
            step_validator=self.step_validator,
            revise_on_tool_failure=self.revise_on_tool_failure,
            max_step_attempts=self.max_step_attempts,
            max_replans=self.max_replans,
            tool_failure_recovery=self.tool_failure_recovery,
        )


ExecutionProfile = StandardExecutionProfile | PlanExecutionProfile


@dataclass(frozen=True, slots=True)
class ResolvedExecutionProfile:
    """Secret-free description of the execution engine selected at composition."""

    kind: str
    engine_id: str
    state_version: int
    policy_type: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "details",
            _freeze_mapping(_redact_sensitive(self.details)),
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "kind": self.kind,
            "engine_id": self.engine_id,
            "state_version": self.state_version,
            "policy_type": self.policy_type,
        }
        if self.details:
            value["details"] = _plain(self.details)
        return value


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    schema_fingerprint: str
    safety_profile: ToolSafetyProfile

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "schema_fingerprint": self.schema_fingerprint,
            "safety_profile": {
                "same_call_retry_safe": (self.safety_profile.same_call_retry_safe),
                "changed_argument_repair_safe": (
                    self.safety_profile.changed_argument_repair_safe
                ),
                "timeout_retry_safe": self.safety_profile.timeout_retry_safe,
            },
        }


@dataclass(frozen=True, slots=True)
class AgentSpec:
    """Immutable, inspectable result of resolving an Agent configuration."""

    name: str
    instructions: str = field(repr=False)
    limits: RunLimits
    retry: RetryConfig
    model_adapter: str
    model_identity: Mapping[str, Any]
    model_capabilities: Mapping[str, Any]
    model_options: Mapping[str, Any]
    execution_profile: ResolvedExecutionProfile
    tools: tuple[ToolSpec, ...]
    output_contract: Mapping[str, Any]
    persistence_policy: Mapping[str, Any]
    stream_policy: Mapping[str, Any]
    skill_policy: Mapping[str, Any]
    compatibility_metadata: Mapping[str, Any]
    agent_fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("AgentSpec name cannot be empty")
        if not self.instructions.strip():
            raise ValueError("AgentSpec instructions cannot be empty")
        object.__setattr__(
            self,
            "model_identity",
            _freeze_mapping(_redact_sensitive(self.model_identity)),
        )
        object.__setattr__(
            self,
            "model_capabilities",
            _freeze_mapping(self.model_capabilities),
        )
        object.__setattr__(
            self,
            "model_options",
            _freeze_mapping(_redact_sensitive(self.model_options)),
        )
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(
            self,
            "output_contract",
            _freeze_mapping(self.output_contract),
        )
        object.__setattr__(
            self,
            "persistence_policy",
            _freeze_mapping(self.persistence_policy),
        )
        object.__setattr__(
            self,
            "stream_policy",
            _freeze_mapping(self.stream_policy),
        )
        object.__setattr__(
            self,
            "skill_policy",
            _freeze_mapping(self.skill_policy),
        )
        object.__setattr__(
            self,
            "compatibility_metadata",
            _freeze_mapping(self.compatibility_metadata),
        )
        payload = self._fingerprint_payload()
        object.__setattr__(
            self,
            "agent_fingerprint",
            f"sha256:{hashlib.sha256(_canonical_json(payload)).hexdigest()}",
        )

    @property
    def identity(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "name": self.name,
                "instructions": self.instructions,
            }
        )

    @property
    def capabilities(self) -> Mapping[str, Any]:
        return self.model_capabilities

    def to_dict(
        self,
        *,
        include_instructions: bool = True,
        include_fingerprint: bool = True,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "identity": {
                "name": self.name,
                "instructions": (
                    self.instructions if include_instructions else "[REDACTED]"
                ),
            },
            "limits": {
                "max_steps": self.limits.max_steps,
                "max_tool_calls": self.limits.max_tool_calls,
                "timeout_seconds": self.limits.timeout_seconds,
                "parallel_tool_calls": self.limits.parallel_tool_calls,
                "max_parallel_tools": self.limits.max_parallel_tools,
                "max_step_attempts": self.limits.max_step_attempts,
                "max_replans": self.limits.max_replans,
                "max_tool_repair_attempts": (self.limits.max_tool_repair_attempts),
                "max_model_turns": self.limits.max_model_turns,
                "no_progress_model_turn_threshold": (
                    self.limits.no_progress_model_turn_threshold
                ),
            },
            "retry": {
                "max_attempts": self.retry.max_attempts,
                "initial_delay": self.retry.initial_delay,
                "max_delay": self.retry.max_delay,
                "backoff_factor": self.retry.backoff_factor,
            },
            "model": {
                "adapter": self.model_adapter,
                "identity": _plain(self.model_identity),
                "capabilities": _plain(self.model_capabilities),
                "options": _plain(self.model_options),
            },
            "execution_profile": self.execution_profile.to_dict(),
            "tools": [tool.to_dict() for tool in self.tools],
            "output_contract": _plain(self.output_contract),
            "persistence_policy": _plain(self.persistence_policy),
            "stream_policy": _plain(self.stream_policy),
            "skill_policy": _plain(self.skill_policy),
            "compatibility_metadata": _plain(self.compatibility_metadata),
        }
        if include_fingerprint:
            value["agent_fingerprint"] = self.agent_fingerprint
        return value

    def _fingerprint_payload(self) -> Mapping[str, Any]:
        """Return only state-compatibility inputs, not replaceable operations.

        Provider adapters, credentials, stores, sinks and trace presentation can
        be replaced between resume attempts. Engine semantics, prompts, Tool
        contracts and output validation cannot.
        """

        return {
            "identity": {
                "name": self.name,
                "instructions": self.instructions,
            },
            "limits": self.to_dict(include_fingerprint=False)["limits"],
            "retry": self.to_dict(include_fingerprint=False)["retry"],
            "model": {
                "identity": _plain(self.model_identity),
                "capabilities": _plain(self.model_capabilities),
                "options": _plain(self.model_options),
            },
            "execution_profile": self.execution_profile.to_dict(),
            "tools": [tool.to_dict() for tool in self.tools],
            "output_contract": _plain(self.output_contract),
            "conversation_memory_policy": self.persistence_policy.get(
                "conversation_memory_policy"
            ),
            "skill_policy": _plain(self.skill_policy),
        }


@dataclass(slots=True)
class ResolvedAgentComposition:
    spec: AgentSpec
    runtime: RunCoordinator
    engine: ExecutionEngine[Any]
    decision_policy: DecisionPolicy
    tool_registry: ToolRegistry
    tool_executor: ToolExecutor
    conversation_memory_policy: ConversationMemoryPolicy
    skill_runtime: SkillRuntime | None


def compose_agent(
    *,
    config: AgentConfig,
    model: ModelClient,
    tools: Iterable[Tool] = (),
    conversation_store: ConversationStore | None = None,
    decision_policy: DecisionPolicy | None = None,
    execution_profile: ExecutionProfile | None = None,
    execution_engine: ExecutionEngine[Any] | None = None,
    output_codec: OutputCodec | None = None,
    event_sink: EventSink | None = None,
    diagnostic_sink: DiagnosticSink | None = None,
    diagnostic_timeout_seconds: float = 0.25,
    diagnostic_max_pending_deliveries: int = 1024,
    tool_authorizer: ToolAuthorizer | None = None,
    checkpoint_store: CheckpointStore | None = None,
    conversation_memory_policy: ConversationMemoryPolicy | None = None,
    skill_registry: SkillRegistry | None = None,
    skill_selector: SkillSelector | None = None,
    skill_limits: SkillLimits | None = None,
) -> ResolvedAgentComposition:
    """Resolve defaults, validate capabilities, and assemble one Agent."""

    if skill_selector is not None and skill_registry is None:
        raise ValueError("skill_selector requires skill_registry")
    policy, resolved_profile, compatibility = _resolve_execution(
        decision_policy=decision_policy,
        execution_profile=execution_profile,
        execution_engine=execution_engine,
    )
    if resolved_profile.kind == "plan" and config.finalization_mode == "disabled":
        raise ConfigurationError(
            "strict Plan-and-Execute requires finalization_mode to be enabled"
        )

    skill_runtime = (
        SkillRuntime(
            skill_registry,
            selector=skill_selector,
            limits=skill_limits,
        )
        if skill_registry is not None
        else None
    )
    registered_tools = tuple(tools)
    if skill_runtime is not None:
        registered_tools = (
            *registered_tools,
            SkillReadTool(skill_runtime),
            SkillSearchTool(skill_runtime),
        )
    tool_registry = ToolRegistry(registered_tools)
    memory_policy = (
        conversation_memory_policy
        if conversation_memory_policy is not None
        else FullConversationMemoryPolicy()
    )
    resolved_output = TextOutputCodec() if output_codec is None else output_codec
    resolved_conversation_store = (
        InMemoryConversationStore()
        if conversation_store is None
        else conversation_store
    )
    resolved_event_sink = NoopEventSink() if event_sink is None else event_sink
    resolved_diagnostic_sink = (
        NoopDiagnosticSink() if diagnostic_sink is None else diagnostic_sink
    )
    diagnostic_reporter = (
        None
        if diagnostic_sink is None
        else DiagnosticReporter(
            resolved_diagnostic_sink,
            timeout_seconds=diagnostic_timeout_seconds,
            max_pending_deliveries=diagnostic_max_pending_deliveries,
        )
    )
    tool_executor = ToolExecutor(
        tool_registry,
        authorizer=(
            AllowAllAuthorizer() if tool_authorizer is None else tool_authorizer
        ),
        retry=config.retry,
        diagnostic_reporter=diagnostic_reporter,
    )
    if checkpoint_store is not None and not _supports_idempotent_append(
        resolved_conversation_store
    ):
        raise ConfigurationError(
            "checkpointed Agents require a ConversationStore with atomic append_once()"
        )

    capabilities = _model_capabilities(model)
    if not capabilities.chat:
        raise CapabilityError("the configured model does not support chat")
    _validate_required_engine_capabilities(execution_engine, capabilities)
    if resolved_profile.kind == "plan" and not capabilities.structured_output:
        raise CapabilityError(
            "Plan execution requires model structured-output capability"
        )
    if resolved_profile.kind == "plan":
        _validate_plan_generator_capabilities(policy)
    _validate_auxiliary_model_capabilities(
        skill_selector=skill_selector,
        memory_policy=memory_policy,
    )
    output_schema = resolved_output.schema()
    if len(tool_registry) and not capabilities.tool_calling:
        raise CapabilityError("the configured model does not support tool calling")
    if (
        config.limits.parallel_tool_calls
        and len(tool_registry)
        and not capabilities.parallel_tool_calling
    ):
        raise CapabilityError(
            "parallel_tool_calls requires model parallel tool-calling capability"
        )
    if output_schema is not None and not capabilities.structured_output:
        raise CapabilityError("the configured model does not support structured output")

    tool_specs = tuple(_tool_spec(tool) for tool in tool_registry)
    spec = AgentSpec(
        name=config.name,
        instructions=config.instructions,
        limits=config.limits,
        retry=config.retry,
        model_adapter=_qualified_type_name(model),
        model_identity=_model_identity(model),
        model_capabilities=_capabilities_dict(capabilities),
        model_options=config.model_options,
        execution_profile=resolved_profile,
        tools=tool_specs,
        output_contract={
            "codec": _qualified_type_name(resolved_output),
            "structured": output_schema is not None,
            "schema_fingerprint": (
                _mapping_fingerprint(output_schema)
                if output_schema is not None
                else None
            ),
            "finalization_mode": config.finalization_mode,
            "staged_finalization": _uses_staged_standard_finalization(
                execution_kind=resolved_profile.kind,
                finalization_mode=config.finalization_mode,
                has_tools=bool(len(tool_registry)),
                has_output_schema=output_schema is not None,
                capabilities=capabilities,
            ),
        },
        persistence_policy={
            "conversation_store": _qualified_type_name(resolved_conversation_store),
            "idempotent_append": _supports_idempotent_append(
                resolved_conversation_store
            ),
            "checkpoint_store": (
                None
                if checkpoint_store is None
                else _qualified_type_name(checkpoint_store)
            ),
            "conversation_memory_policy": _memory_policy_spec(memory_policy),
        },
        stream_policy={
            "visibility": config.stream_visibility,
            "tool_trace_mode": config.tool_trace_mode,
            "event_sink": _qualified_type_name(resolved_event_sink),
            **(
                {}
                if diagnostic_sink is None
                else {
                    "diagnostic_sink": _qualified_type_name(resolved_diagnostic_sink),
                    "diagnostic_timeout_seconds": diagnostic_timeout_seconds,
                    "diagnostic_max_pending_deliveries": (
                        diagnostic_max_pending_deliveries
                    ),
                }
            ),
        },
        skill_policy={
            "enabled": skill_runtime is not None,
            "registry": (
                None if skill_registry is None else _qualified_type_name(skill_registry)
            ),
            "selector": (
                None if skill_selector is None else _selector_spec(skill_selector)
            ),
            "catalog_digest": (
                None if skill_registry is None else skill_registry.catalog_digest
            ),
            "limits": (
                None
                if skill_runtime is None
                else {
                    item.name: getattr(skill_runtime.limits, item.name)
                    for item in fields(skill_runtime.limits)
                }
            ),
        },
        compatibility_metadata=compatibility,
    )
    engine = execution_engine
    if engine is None:
        engine = (
            PlanExecutionEngine(policy)
            if resolved_profile.kind == "plan"
            else StandardExecutionEngine(policy)
        )
    runtime = RunCoordinator(
        config=config,
        model=model,
        decision_policy=policy,
        tool_executor=tool_executor,
        conversation_store=resolved_conversation_store,
        output_codec=resolved_output,
        event_sink=resolved_event_sink,
        diagnostic_reporter=diagnostic_reporter,
        checkpoint_store=checkpoint_store,
        conversation_memory_policy=memory_policy,
        skill_runtime=skill_runtime,
        engine=engine,
        resolved_spec={
            **resolved_profile.to_dict(),
            "execution_profile": resolved_profile.to_dict(),
            # Compatibility retention is an explicit lifecycle policy; the
            # Coordinator never interprets Plan state or phase values.
            "retain_terminal_checkpoint": (
                resolved_profile.kind == "plan" or engine.retain_terminal_checkpoint
            ),
        },
    )
    # Runtime owns persistence, but the immutable fingerprint is attached to
    # every run so checkpoint v4 can reject an incompatible Agent on resume.
    runtime.agent_spec = spec
    return ResolvedAgentComposition(
        spec=spec,
        runtime=runtime,
        engine=engine,
        decision_policy=policy,
        tool_registry=tool_registry,
        tool_executor=tool_executor,
        conversation_memory_policy=memory_policy,
        skill_runtime=skill_runtime,
    )


def _resolve_execution(
    *,
    decision_policy: DecisionPolicy | None,
    execution_profile: ExecutionProfile | None,
    execution_engine: ExecutionEngine[Any] | None,
) -> tuple[DecisionPolicy, ResolvedExecutionProfile, Mapping[str, Any]]:
    if execution_engine is not None:
        if decision_policy is not None or execution_profile is not None:
            raise ConfigurationError(
                "execution_engine cannot be combined with decision_policy "
                "or execution_profile"
            )
        _validate_custom_engine(execution_engine)
        policy = StandardDecisionPolicy()
        profile = ResolvedExecutionProfile(
            kind="custom",
            engine_id=execution_engine.engine_id,
            state_version=execution_engine.state_version,
            policy_type=_qualified_type_name(policy),
            details={
                "engine_type": _qualified_type_name(execution_engine),
                "state_codec": _qualified_type_name(execution_engine.state_codec),
                "configuration": dict(execution_engine.configuration),
                "configuration_fingerprint": _mapping_fingerprint(
                    _redact_sensitive(execution_engine.configuration)
                ),
                "required_capabilities": dict(execution_engine.required_capabilities),
            },
        )
        return policy, profile, {"resolved_from": "execution_engine"}
    if decision_policy is not None and execution_profile is not None:
        raise ConfigurationError(
            "use either decision_policy or execution_profile, not both"
        )
    compatibility: dict[str, Any] = {}
    if execution_profile is None:
        policy = (
            StandardDecisionPolicy() if decision_policy is None else decision_policy
        )
        is_plan = isinstance(policy, PlanAndExecutePolicy)
        profile = ResolvedExecutionProfile(
            kind="plan" if is_plan else "standard",
            engine_id="plan" if is_plan else "standard",
            state_version=1,
            policy_type=_qualified_type_name(policy),
            details=_policy_details(policy),
        )
        if decision_policy is not None:
            compatibility["resolved_from"] = "decision_policy"
            compatibility["decision_policy"] = _qualified_type_name(policy)
        else:
            compatibility["resolved_from"] = "default"
        return policy, profile, compatibility

    if isinstance(execution_profile, StandardExecutionProfile):
        policy = (
            StandardDecisionPolicy()
            if execution_profile.decision_policy is None
            else execution_profile.decision_policy
        )
        if isinstance(policy, PlanAndExecutePolicy):
            raise ConfigurationError(
                "StandardExecutionProfile cannot use PlanAndExecutePolicy"
            )
        profile = ResolvedExecutionProfile(
            kind="standard",
            engine_id=execution_profile.engine_id,
            state_version=execution_profile.state_version,
            policy_type=_qualified_type_name(policy),
            details=_policy_details(policy),
        )
    elif isinstance(execution_profile, PlanExecutionProfile):
        policy = execution_profile.create_policy()
        profile = ResolvedExecutionProfile(
            kind="plan",
            engine_id=execution_profile.engine_id,
            state_version=execution_profile.state_version,
            policy_type=_qualified_type_name(policy),
            details=_policy_details(policy),
        )
    else:
        raise TypeError(
            "execution_profile must be StandardExecutionProfile or PlanExecutionProfile"
        )
    compatibility["resolved_from"] = "execution_profile"
    return policy, profile, compatibility


def _validate_custom_engine(engine: ExecutionEngine[Any]) -> None:
    if not isinstance(engine, ExecutionEngine):
        raise TypeError("execution_engine must implement ExecutionEngine")
    if not isinstance(engine.engine_id, str) or not engine.engine_id.strip():
        raise ConfigurationError("custom execution engine_id cannot be empty")
    if engine.engine_id in {"standard", "plan"}:
        raise ConfigurationError(
            f"custom execution engine_id {engine.engine_id!r} is reserved"
        )
    if type(engine.state_version) is not int or engine.state_version < 1:
        raise ConfigurationError(
            "custom execution state_version must be a positive integer"
        )
    codec = engine.state_codec
    if not isinstance(codec, EngineStateCodec):
        raise TypeError("custom execution state_codec must implement EngineStateCodec")
    if codec.engine_id != engine.engine_id:
        raise ConfigurationError(
            "custom execution state_codec engine_id does not match the Engine"
        )
    if codec.state_version != engine.state_version:
        raise ConfigurationError(
            "custom execution state_codec version does not match the Engine"
        )
    retain_terminal = engine.retain_terminal_checkpoint
    if type(retain_terminal) is not bool:
        raise ConfigurationError(
            "custom execution retain_terminal_checkpoint must be a bool"
        )
    configuration = engine.configuration
    if not isinstance(configuration, Mapping):
        raise ConfigurationError(
            "custom execution engines must declare a configuration mapping"
        )
    _validate_declarative_mapping(configuration, "custom execution configuration")
    requirements = engine.required_capabilities
    if not isinstance(requirements, Mapping):
        raise ConfigurationError(
            "custom execution required_capabilities must be a mapping"
        )
    allowed_capabilities = {
        "chat",
        "streaming",
        "tool_calling",
        "parallel_tool_calling",
        "structured_output",
        "tool_calling_with_structured_output",
        "embeddings",
        "vision",
    }
    unknown = set(requirements).difference(allowed_capabilities)
    if unknown:
        raise ConfigurationError(
            "custom execution requires unknown model capabilities: "
            + ", ".join(sorted(str(item) for item in unknown))
        )
    if any(type(value) is not bool for value in requirements.values()):
        raise ConfigurationError(
            "custom execution capability requirements must be bool values"
        )


def _model_capabilities(model: ModelClient) -> ModelCapabilities:
    capabilities = getattr(model, "capabilities", None)
    if capabilities is None:
        # Pre-0.3 custom clients did not always publish capabilities. The
        # compatibility adapter preserves their previous permissive behavior.
        return ModelCapabilities()
    if not isinstance(capabilities, ModelCapabilities):
        raise TypeError("model capabilities must be a ModelCapabilities")
    return capabilities


def _supports_idempotent_append(store: ConversationStore) -> bool:
    method = getattr(store, "append_once", None)
    if not callable(method):
        return False
    return getattr(store, "supports_idempotent_append", True) is True


def _validate_required_engine_capabilities(
    engine: ExecutionEngine[Any] | None,
    capabilities: ModelCapabilities,
) -> None:
    if engine is None:
        return
    requirements = engine.required_capabilities
    available = _capabilities_dict(capabilities)
    missing = [
        str(name)
        for name, required in requirements.items()
        if required and not bool(available[str(name)])
    ]
    if missing:
        raise CapabilityError(
            "custom execution engine requires model capabilities: "
            + ", ".join(sorted(missing))
        )


def _validate_plan_generator_capabilities(policy: DecisionPolicy) -> None:
    if not isinstance(policy, PlanAndExecutePolicy):
        return
    model = getattr(policy.plan_generator, "model", None)
    if model is None:
        return
    capabilities = _model_capabilities(model)
    if not capabilities.chat:
        raise CapabilityError("the Plan generator model does not support chat")
    if not capabilities.structured_output:
        raise CapabilityError(
            "the Plan generator model does not support structured output"
        )


def _validate_auxiliary_model_capabilities(
    *,
    skill_selector: SkillSelector | None,
    memory_policy: ConversationMemoryPolicy,
) -> None:
    selectors: list[SkillSelector] = []
    if skill_selector is not None:
        selectors.append(skill_selector)
    while selectors:
        selector = selectors.pop()
        if isinstance(selector, ModelSkillSelector):
            capabilities = _model_capabilities(selector.model)
            if not capabilities.chat:
                raise CapabilityError("the Skill selector model does not support chat")
            if not capabilities.structured_output:
                raise CapabilityError(
                    "the Skill selector model does not support structured output"
                )
        elif isinstance(selector, HybridSkillSelector):
            selectors.append(selector.automatic)

    if not isinstance(memory_policy, TokenBudgetConversationMemoryPolicy):
        return
    summarizer = memory_policy.summarizer
    if not isinstance(summarizer, ModelConversationSummarizer):
        return
    if not _model_capabilities(summarizer.model).chat:
        raise CapabilityError("the conversation summarizer model does not support chat")


def _policy_details(policy: DecisionPolicy) -> dict[str, Any]:
    if not isinstance(policy, PlanAndExecutePolicy):
        return {}
    recovery = policy.tool_failure_recovery
    recovery_details: dict[str, Any] | None = None
    if recovery is not None:
        recovery_details = {
            "type": _qualified_type_name(recovery),
            "fallback": recovery.fallback,
            "require_repair_safe": recovery.require_repair_safe,
            "feedback_mode": recovery.feedback_mode,
        }
    return {
        "plan_generator": _plan_generator_spec(policy.plan_generator),
        "step_validator": _component_spec(policy.step_validator),
        "revise_on_tool_failure": policy.revise_on_tool_failure,
        "max_step_attempts": policy.max_step_attempts,
        "max_replans": policy.max_replans,
        "tool_failure_recovery": recovery_details,
    }


def _plan_generator_spec(generator: PlanGenerator) -> dict[str, Any]:
    spec = _component_spec(
        generator,
        scalar_fields=("max_steps", "history_limit"),
    )
    model = getattr(generator, "model", None)
    if model is not None:
        spec["model"] = {
            "type": _qualified_type_name(model),
            "identity": _model_identity(model),
        }
    return spec


def _memory_policy_spec(policy: ConversationMemoryPolicy) -> dict[str, Any]:
    spec = _component_spec(
        policy,
        scalar_fields=(
            "max_turns",
            "max_history_turns",
            "policy_fingerprint",
        ),
    )
    budget = getattr(policy, "budget", None)
    if budget is not None:
        spec["budget"] = {
            "context_window_tokens": getattr(
                budget,
                "context_window_tokens",
                None,
            ),
            "reserved_output_tokens": getattr(
                budget,
                "reserved_output_tokens",
                None,
            ),
            "safety_margin_tokens": getattr(
                budget,
                "safety_margin_tokens",
                None,
            ),
        }
    for name in ("token_counter", "summarizer"):
        component = getattr(policy, name, None)
        if component is not None:
            spec[name] = _component_spec(
                component,
                scalar_fields=("cache_fingerprint",),
            )
    return spec


def _selector_spec(selector: SkillSelector) -> dict[str, Any]:
    spec = _component_spec(
        selector,
        scalar_fields=("max_skills",),
        mapping_fields=("options", "provider_options"),
    )
    model = getattr(selector, "model", None)
    if model is not None:
        spec["model"] = {
            "type": _qualified_type_name(model),
            "identity": _model_identity(model),
        }
    automatic = getattr(selector, "automatic", None)
    if automatic is not None:
        spec["automatic"] = _selector_spec(automatic)
    return spec


def _component_spec(
    component: Any,
    *,
    scalar_fields: tuple[str, ...] = (),
    mapping_fields: tuple[str, ...] = (),
) -> dict[str, Any]:
    spec: dict[str, Any] = {"type": _qualified_type_name(component)}
    declared = getattr(component, "configuration", None)
    if isinstance(declared, Mapping):
        _validate_declarative_mapping(declared, "component configuration")
        spec["configuration"] = _redact_sensitive(declared)
    for name in scalar_fields:
        value = getattr(component, name, None)
        if value is not None and isinstance(value, (str, int, float, bool)):
            spec[name] = value
    for name in mapping_fields:
        value = getattr(component, name, None)
        if isinstance(value, Mapping):
            spec[name] = _redact_sensitive(value)
    return spec


def _validate_declarative_mapping(value: Mapping[str, Any], label: str) -> None:
    def visit(item: Any) -> bool:
        if isinstance(item, float):
            return math.isfinite(item)
        if item is None or isinstance(item, (str, int, bool, Enum)):
            return True
        if isinstance(item, Mapping):
            return all(
                isinstance(key, str) and visit(nested) for key, nested in item.items()
            )
        if isinstance(item, (list, tuple)):
            return all(visit(nested) for nested in item)
        return False

    if not visit(value):
        raise ConfigurationError(f"{label} must contain only JSON-like values")


def _capabilities_dict(capabilities: ModelCapabilities) -> dict[str, Any]:
    return {
        "chat": capabilities.chat,
        "streaming": capabilities.streaming,
        "tool_calling": capabilities.tool_calling,
        "parallel_tool_calling": capabilities.parallel_tool_calling,
        "structured_output": capabilities.structured_output,
        "tool_calling_with_structured_output": (
            capabilities.tool_calling_with_structured_output
        ),
        "embeddings": capabilities.embeddings,
        "vision": capabilities.vision,
        "limits": _redact_sensitive(capabilities.limits),
    }


def _uses_staged_standard_finalization(
    *,
    execution_kind: str,
    finalization_mode: str,
    has_tools: bool,
    has_output_schema: bool,
    capabilities: ModelCapabilities,
) -> bool:
    if execution_kind == "plan":
        return True
    if finalization_mode == "always":
        return True
    combined_contract = has_tools and has_output_schema
    return combined_contract and (
        finalization_mode == "structured_only"
        or not capabilities.tool_calling_with_structured_output
    )


def _tool_spec(tool: Tool) -> ToolSpec:
    schema = tool.schema
    return ToolSpec(
        name=schema.name,
        description=schema.description,
        schema_fingerprint=_mapping_fingerprint(schema.to_dict()),
        safety_profile=resolve_tool_safety_profile(tool),
    )


def _mapping_fingerprint(value: Mapping[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(value)).hexdigest()}"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        _plain(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _qualified_type_name(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "authorization",
        "bearer",
        "bearer_token",
        "client_secret",
        "cookie",
        "credential",
        "credentials",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "set_cookie",
        "token",
    }
)
_SENSITIVE_SUFFIXES = (
    "_api_key",
    "_authorization",
    "_credential",
    "_credentials",
    "_password",
    "_private_key",
    "_refresh_token",
    "_secret",
    "_token",
)
_HEADER_CONTAINER_KEYS = frozenset({"header", "headers", "http_headers"})


def _redact_sensitive(value: Any, *, key: str = "") -> Any:
    normalized_key = _normalize_key(key)
    if _is_sensitive_key(normalized_key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        if normalized_key in _HEADER_CONTAINER_KEYS:
            return {str(item_key): "[REDACTED]" for item_key in value}
        return {
            str(item_key): _redact_sensitive(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_redact_sensitive(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, str):
        return _redact_url_credentials(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return f"<{_qualified_type_name(value)}>"


def _model_identity(model: ModelClient) -> dict[str, Any]:
    """Return stable provider identity without credentials or live objects."""

    identity: dict[str, Any] = {}
    for attribute in ("model", "base_url"):
        value = getattr(model, attribute, None)
        if isinstance(value, (str, int, float, bool)) or value is None:
            if value is not None:
                identity[attribute] = value
    for attribute in ("default_options", "provider_options"):
        value = getattr(model, attribute, None)
        if isinstance(value, Mapping) and value:
            identity[attribute] = dict(value)
    return _redact_sensitive(identity)


def _normalize_key(value: str) -> str:
    text = re.sub(r"(?<!^)(?=[A-Z])", "_", str(value).strip())
    return text.lower().replace("-", "_").replace(" ", "_")


def _is_sensitive_key(normalized_key: str) -> bool:
    return normalized_key in _SENSITIVE_KEYS or normalized_key.endswith(
        _SENSITIVE_SUFFIXES
    )


def _redact_url_credentials(value: str) -> str:
    """Remove URL userinfo and credential-like query parameters."""

    if "://" not in value:
        return value
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if not parsed.scheme or not parsed.netloc:
        return value
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    netloc = f"{host}{port}"
    if parsed.username is not None or parsed.password is not None:
        netloc = f"[REDACTED]@{netloc}"
    query = urlencode(
        [
            (
                query_key,
                (
                    "[REDACTED]"
                    if _is_sensitive_key(_normalize_key(query_key))
                    else query_value
                ),
            )
            for query_key, query_value in parse_qsl(
                parsed.query,
                keep_blank_values=True,
            )
        ],
        doseq=True,
    )
    return urlunsplit((parsed.scheme, netloc, parsed.path, query, parsed.fragment))


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, ToolSafetyProfile):
        return {
            "same_call_retry_safe": value.same_call_retry_safe,
            "changed_argument_repair_safe": (value.changed_argument_repair_safe),
            "timeout_retry_safe": value.timeout_retry_safe,
        }
    return value


__all__ = [
    "AgentSpec",
    "ExecutionProfile",
    "PlanExecutionProfile",
    "ResolvedExecutionProfile",
    "StandardExecutionProfile",
    "ToolSpec",
    "compose_agent",
]
