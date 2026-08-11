from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from moduagent.config import RunLimits
from moduagent.definitions import (
    AgentDefinition,
    DefinitionStatus,
    REQUIRED_SEMANTIC_DIGEST_KEYS,
    ResolvedAgentEndpoint,
    RuntimeBindings,
)
from moduagent.errors import ConfigurationError
from moduagent.memory import (
    FullConversationMemoryPolicy,
    InMemoryContextMemoryStateStore,
    InMemoryMemoryStateStore,
    MemoryContextBound,
)
from moduagent.observability import (
    AuditEventSink,
    CompositeDiagnosticSink,
    CompositeEventSink,
    InMemoryDiagnosticSink,
    LoggingEventSink,
    MetricsEventSink,
    NoopDiagnosticSink,
    NoopEventSink,
)
from moduagent.persistence import InMemoryCheckpointStore, InMemoryConversationStore
from moduagent.tools import AgentTool, AllowAllAuthorizer


_POLICY_REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,511}$")
_VIOLATION_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,127}$")


class RuntimeProfileKind(str, Enum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


@dataclass(frozen=True, slots=True)
class ProfileViolation:
    code: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or (
            _VIOLATION_CODE_PATTERN.fullmatch(self.code) is None
        ):
            raise ValueError("profile violation code must be a stable lowercase code")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("profile violation message cannot be empty")


class RuntimeProfileError(ConfigurationError):
    """Fail-fast, payload-free rejection of an unsafe runtime composition."""

    def __init__(
        self,
        profile: RuntimeProfileKind,
        violations: Iterable[ProfileViolation],
    ) -> None:
        self.profile = profile
        self.violations = tuple(violations)
        if not self.violations:
            raise ValueError("RuntimeProfileError requires at least one violation")
        self.codes = tuple(violation.code for violation in self.violations)
        super().__init__(
            f"{profile.value} runtime profile rejected configuration: "
            + ", ".join(self.codes)
        )


@dataclass(frozen=True, slots=True)
class RuntimeValidationContext:
    """Non-secret facts needed to validate a resolved runtime composition."""

    bindings: RuntimeBindings
    definition: AgentDefinition | None = None
    definition_status: DefinitionStatus | None = None
    tenant_context: object | None = field(default=None, repr=False, compare=False)
    principal_context: object | None = field(default=None, repr=False, compare=False)
    memory_policy: object | None = field(default=None, repr=False, compare=False)
    tools: tuple[object, ...] = field(default_factory=tuple, repr=False, compare=False)
    delegation_enabled: bool = False
    delegation_limits: object | None = field(default=None, repr=False, compare=False)
    shared_parent_child_session: bool = False
    mutating_tool_names: frozenset[str] = field(default_factory=frozenset)
    approval_policy_refs: Mapping[str, str] = field(default_factory=dict, repr=False)
    idempotency_policy_refs: Mapping[str, str] = field(
        default_factory=dict,
        repr=False,
    )
    agent_idempotency_policy_ref: str | None = None
    resolved_endpoint: ResolvedAgentEndpoint | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    unapproved_plugin_refs: tuple[str, ...] = ()
    unapproved_remote_endpoint_refs: tuple[str, ...] = ()
    unapproved_delegation_endpoint_refs: tuple[str, ...] = ()
    content_telemetry_enabled: bool = False
    content_telemetry_policy_ref: str | None = None
    external_io_enabled: bool = False
    deterministic_components: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.bindings, RuntimeBindings):
            raise TypeError("bindings must be RuntimeBindings")
        if self.definition is not None and not isinstance(
            self.definition,
            AgentDefinition,
        ):
            raise TypeError("definition must be AgentDefinition or None")
        if self.definition_status is not None and not isinstance(
            self.definition_status,
            DefinitionStatus,
        ):
            try:
                object.__setattr__(
                    self,
                    "definition_status",
                    DefinitionStatus(self.definition_status),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("definition_status is invalid") from exc
        endpoint = self.resolved_endpoint
        if endpoint is not None and not isinstance(endpoint, ResolvedAgentEndpoint):
            raise TypeError("resolved_endpoint must be ResolvedAgentEndpoint or None")
        if endpoint is not None:
            if self.definition is None:
                object.__setattr__(self, "definition", endpoint.definition)
            elif self.definition.ref != endpoint.ref or (
                self.definition.fingerprint != endpoint.definition_fingerprint
            ):
                raise ValueError("definition does not match resolved_endpoint")
            if self.definition_status is None:
                object.__setattr__(self, "definition_status", endpoint.status)
            elif self.definition_status != endpoint.status:
                raise ValueError("definition_status does not match resolved_endpoint")
        for field_name in (
            "delegation_enabled",
            "shared_parent_child_session",
            "content_telemetry_enabled",
            "external_io_enabled",
            "deterministic_components",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a bool")
        if isinstance(self.tools, (str, bytes)):
            raise TypeError("tools must be an iterable of Tool objects")
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(
            self,
            "mutating_tool_names",
            _validated_name_set(self.mutating_tool_names, "mutating_tool_names"),
        )
        object.__setattr__(
            self,
            "approval_policy_refs",
            _validated_policy_mapping(self.approval_policy_refs),
        )
        object.__setattr__(
            self,
            "idempotency_policy_refs",
            _validated_policy_mapping(self.idempotency_policy_refs),
        )
        for field_name in (
            "agent_idempotency_policy_ref",
            "content_telemetry_policy_ref",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _validate_policy_ref(value, field_name)
        object.__setattr__(
            self,
            "unapproved_plugin_refs",
            _validated_reference_tuple(
                self.unapproved_plugin_refs,
                "unapproved_plugin_refs",
            ),
        )
        object.__setattr__(
            self,
            "unapproved_remote_endpoint_refs",
            _validated_reference_tuple(
                self.unapproved_remote_endpoint_refs,
                "unapproved_remote_endpoint_refs",
            ),
        )
        object.__setattr__(
            self,
            "unapproved_delegation_endpoint_refs",
            _validated_reference_tuple(
                self.unapproved_delegation_endpoint_refs,
                "unapproved_delegation_endpoint_refs",
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    kind: RuntimeProfileKind
    allow_in_memory: bool
    allow_allow_all_authorizer: bool
    external_io_allowed: bool
    deterministic_components_required: bool
    require_durable_stores: bool
    require_tenant_and_principal: bool
    require_telemetry: bool
    default_limits: RunLimits

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RuntimeProfileKind):
            try:
                object.__setattr__(self, "kind", RuntimeProfileKind(self.kind))
            except (TypeError, ValueError) as exc:
                raise ValueError("unknown runtime profile kind") from exc
        for field_name in (
            "allow_in_memory",
            "allow_allow_all_authorizer",
            "external_io_allowed",
            "deterministic_components_required",
            "require_durable_stores",
            "require_tenant_and_principal",
            "require_telemetry",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise TypeError(f"{field_name} must be a bool")
        if not isinstance(self.default_limits, RunLimits):
            raise TypeError("default_limits must be RunLimits")

    @classmethod
    def development(cls) -> RuntimeProfile:
        return DevelopmentProfile()

    @classmethod
    def test(cls) -> RuntimeProfile:
        return TestProfile()

    @classmethod
    def production(cls) -> RuntimeProfile:
        return ProductionProfile()

    def validate(self, context: RuntimeValidationContext) -> RuntimeValidationContext:
        if not isinstance(context, RuntimeValidationContext):
            raise TypeError("context must be a RuntimeValidationContext")
        violations: list[ProfileViolation] = []
        if self.kind is RuntimeProfileKind.TEST:
            if context.external_io_enabled:
                _add_violation(
                    violations,
                    "test_external_io_forbidden",
                    "Test profile forbids external I/O",
                )
            if self.deterministic_components_required and not (
                context.deterministic_components
            ):
                _add_violation(
                    violations,
                    "test_deterministic_components_required",
                    "Test profile requires deterministic components",
                )
        elif self.kind is RuntimeProfileKind.PRODUCTION:
            _validate_production(context, violations)
        if violations:
            raise RuntimeProfileError(self.kind, violations)
        return context


class DevelopmentProfile(RuntimeProfile):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            kind=RuntimeProfileKind.DEVELOPMENT,
            allow_in_memory=True,
            allow_allow_all_authorizer=True,
            external_io_allowed=True,
            deterministic_components_required=False,
            require_durable_stores=False,
            require_tenant_and_principal=False,
            require_telemetry=False,
            default_limits=RunLimits(),
        )


class TestProfile(RuntimeProfile):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            kind=RuntimeProfileKind.TEST,
            allow_in_memory=True,
            allow_allow_all_authorizer=True,
            external_io_allowed=False,
            deterministic_components_required=True,
            require_durable_stores=False,
            require_tenant_and_principal=False,
            require_telemetry=False,
            default_limits=RunLimits(
                max_steps=4,
                max_tool_calls=6,
                timeout_seconds=30,
                max_parallel_tools=1,
                max_step_attempts=1,
                max_replans=0,
                max_tool_repair_attempts=0,
                max_model_turns=8,
                no_progress_model_turn_threshold=2,
            ),
        )


class ProductionProfile(RuntimeProfile):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(
            kind=RuntimeProfileKind.PRODUCTION,
            allow_in_memory=False,
            allow_allow_all_authorizer=False,
            external_io_allowed=True,
            deterministic_components_required=False,
            require_durable_stores=True,
            require_tenant_and_principal=True,
            require_telemetry=True,
            default_limits=RunLimits(),
        )


def _validate_production(
    context: RuntimeValidationContext,
    violations: list[ProfileViolation],
) -> None:
    definition = context.definition
    bindings = context.bindings
    if definition is None:
        _add_violation(
            violations,
            "production_definition_required",
            "Production requires an AgentDefinition",
        )
    else:
        missing_digests = REQUIRED_SEMANTIC_DIGEST_KEYS.difference(
            definition.semantic_digests
        )
        if missing_digests:
            _add_violation(
                violations,
                "production_definition_digests_required",
                "Production requires all semantic artifact digests",
            )
        if context.definition_status not in {
            DefinitionStatus.APPROVED,
            DefinitionStatus.ACTIVE,
        }:
            _add_violation(
                violations,
                "production_definition_not_approved",
                "Production definitions must be approved or active",
            )
        if definition.side_effect_level.casefold() not in {
            "none",
            "read",
            "advisory",
        }:
            _add_violation(
                violations,
                "production_mutating_agent_not_supported",
                "Production mutating Agents require a future enforced approval plane",
            )
            if definition.approval_requirement.casefold() == "none":
                _add_violation(
                    violations,
                    "production_agent_approval_required",
                    "A mutating Agent requires an approval policy",
                )
            if context.agent_idempotency_policy_ref is None:
                _add_violation(
                    violations,
                    "production_agent_idempotency_required",
                    "A mutating Agent requires an idempotency policy",
                )

    _validate_required_bindings(bindings, violations)

    if not _context_present(context.tenant_context):
        if bindings.tenant_context_provider is None:
            _add_violation(
                violations,
                "production_tenant_context_required",
                "Production requires trusted tenant context or a provider",
            )
        elif not _valid_context_provider(bindings.tenant_context_provider):
            _add_violation(
                violations,
                "production_tenant_context_provider_invalid",
                "The tenant context provider does not implement its protocol",
            )
    if not _context_present(context.principal_context):
        if bindings.principal_context_provider is None:
            _add_violation(
                violations,
                "production_principal_context_required",
                "Production requires trusted principal context or a provider",
            )
        elif not _valid_context_provider(bindings.principal_context_provider):
            _add_violation(
                violations,
                "production_principal_context_provider_invalid",
                "The principal context provider does not implement its protocol",
            )

    if isinstance(bindings.tool_authorizer, AllowAllAuthorizer):
        _add_violation(
            violations,
            "production_allow_all_authorizer_forbidden",
            "Production forbids AllowAllAuthorizer",
        )
    if isinstance(bindings.conversation_store, InMemoryConversationStore):
        _add_violation(
            violations,
            "production_in_memory_conversation_store_forbidden",
            "Production requires a durable ConversationStore",
        )
    elif (
        bindings.conversation_store is not None
        and getattr(bindings.conversation_store, "durable", False) is not True
    ):
        _add_violation(
            violations,
            "production_conversation_store_not_durable",
            "Production requires an explicitly durable ConversationStore",
        )
    conversation_store = bindings.conversation_store
    if conversation_store is not None:
        if (
            getattr(
                conversation_store,
                "supports_tenant_agent_scope",
                False,
            )
            is not True
        ):
            _add_violation(
                violations,
                "production_scoped_conversation_store_required",
                "Production requires an explicitly tenant/Agent-scoped "
                "ConversationStore",
            )
        else:
            scoped_agent = getattr(conversation_store, "agent_id", None)
            if definition is not None and scoped_agent != definition.agent_id:
                _add_violation(
                    violations,
                    "production_conversation_store_agent_scope_mismatch",
                    "ConversationStore Agent scope must match AgentDefinition",
                )
            configured_tenant = _context_identity(
                context.tenant_context,
                ("tenant_id", "tenant", "id"),
            )
            scoped_tenant = getattr(conversation_store, "tenant_id", None)
            if configured_tenant is not None and scoped_tenant != configured_tenant:
                _add_violation(
                    violations,
                    "production_conversation_store_tenant_scope_mismatch",
                    "ConversationStore tenant scope must match trusted tenant context",
                )
    if isinstance(bindings.checkpoint_store, InMemoryCheckpointStore):
        _add_violation(
            violations,
            "production_in_memory_checkpoint_store_forbidden",
            "Production requires a durable CheckpointStore",
        )
    elif (
        bindings.checkpoint_store is not None
        and getattr(bindings.checkpoint_store, "durable", False) is not True
    ):
        _add_violation(
            violations,
            "production_checkpoint_store_not_durable",
            "Production requires an explicitly durable CheckpointStore",
        )
    if (
        bindings.checkpoint_store is not None
        and bindings.conversation_store is not None
    ):
        if not _supports_atomic_append(bindings.conversation_store):
            _add_violation(
                violations,
                "production_atomic_append_required",
                "Checkpointed Production runs require atomic append_once",
            )

    if _event_sink_is_unsafe(bindings.event_sink):
        _add_violation(
            violations,
            "production_event_telemetry_required",
            "Production requires a non-noop EventSink",
        )
    elif not _event_sink_has_sealed_content_projection(bindings.event_sink):
        _add_violation(
            violations,
            "production_event_sink_content_safety_required",
            "Production requires a content-safe EventSink projection",
        )
    if _diagnostic_sink_is_unsafe(bindings.diagnostic_sink):
        _add_violation(
            violations,
            "production_diagnostic_telemetry_required",
            "Production requires a non-local DiagnosticSink",
        )

    if context.memory_policy is None:
        _add_violation(
            violations,
            "production_memory_policy_required",
            "Production requires an explicit bounded Context Memory policy",
        )
    elif isinstance(context.memory_policy, FullConversationMemoryPolicy):
        _add_violation(
            violations,
            "production_full_conversation_memory_forbidden",
            "Production forbids unbounded FullConversationMemoryPolicy",
        )
    context_bound = getattr(context.memory_policy, "context_bound", None)
    if context.memory_policy is not None and not isinstance(
        context_bound,
        MemoryContextBound,
    ):
        _add_violation(
            violations,
            "production_memory_policy_bounded_capability_required",
            "Production Context Memory requires a typed finite bound",
        )
    state_store = getattr(context.memory_policy, "state_store", None)
    if isinstance(
        state_store,
        (InMemoryMemoryStateStore, InMemoryContextMemoryStateStore),
    ):
        _add_violation(
            violations,
            "production_process_local_memory_state_forbidden",
            "Production summary state must not be process-local",
        )
    elif state_store is not None and getattr(state_store, "durable", False) is not True:
        _add_violation(
            violations,
            "production_memory_state_store_not_durable",
            "Production summary state requires a durable state store",
        )

    authorization_fingerprint = getattr(
        bindings.tool_authorizer,
        "policy_fingerprint",
        None,
    )
    if (
        not isinstance(authorization_fingerprint, str)
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            authorization_fingerprint,
        )
        is None
    ):
        _add_violation(
            violations,
            "production_authorization_policy_fingerprint_required",
            "Production Tool authorization requires a canonical policy fingerprint",
        )

    if any(isinstance(tool, AgentTool) for tool in context.tools):
        _add_violation(
            violations,
            "production_legacy_agent_tool_forbidden",
            "Production forbids legacy AgentTool delegation",
        )
    mutating_tool_names = set(context.mutating_tool_names)
    for tool in context.tools:
        side_effect_level = getattr(tool, "side_effect_level", None)
        tool_name = getattr(tool, "name", None)
        if not isinstance(tool_name, str) or not tool_name:
            tool_name = type(tool).__name__
        if not isinstance(side_effect_level, str) or (
            side_effect_level.casefold() not in {"none", "read", "advisory", "write"}
        ):
            _add_violation(
                violations,
                "production_tool_side_effect_classification_required",
                "Every Production Tool requires an explicit side-effect classification",
            )
            # An unknown Tool must not be treated as a read merely because its
            # implementation omitted the optional Development capability.
            mutating_tool_names.add(tool_name)
            continue
        explicitly_mutating = getattr(tool, "mutating", False) is True
        if explicitly_mutating or side_effect_level.casefold() == "write":
            mutating_tool_names.add(tool_name)
    for tool_name in mutating_tool_names:
        _add_violation(
            violations,
            "production_mutating_tool_not_supported",
            "Production mutating Tools require a future enforced approval plane",
        )
        if tool_name not in context.approval_policy_refs:
            _add_violation(
                violations,
                "production_mutating_tool_approval_required",
                "Every mutating Tool requires an approval policy",
            )
        if tool_name not in context.idempotency_policy_refs:
            _add_violation(
                violations,
                "production_mutating_tool_idempotency_required",
                "Every mutating Tool requires an idempotency policy",
            )

    delegation_enabled = context.delegation_enabled or any(
        component is not None
        for component in (
            bindings.delegation_authorizer,
            bindings.delegation_receipt_store,
            bindings.execution_group_store,
        )
    )
    if delegation_enabled:
        for field_name in (
            "delegation_authorizer",
            "delegation_receipt_store",
            "execution_group_store",
        ):
            if getattr(bindings, field_name) is None:
                _add_violation(
                    violations,
                    "production_delegation_bindings_required",
                    "Production delegation requires authorizer and durable stores",
                )
        for field_name in (
            "delegation_receipt_store",
            "execution_group_store",
        ):
            store = getattr(bindings, field_name)
            if store is not None and getattr(store, "durable", False) is not True:
                _add_violation(
                    violations,
                    "production_delegation_stores_not_durable",
                    "Production delegation requires durable receipt and budget stores",
                )
        if not _valid_delegation_limits(context.delegation_limits):
            _add_violation(
                violations,
                "production_delegation_limits_required",
                "Production delegation requires depth, count, and deadline limits",
            )
    if context.shared_parent_child_session:
        _add_violation(
            violations,
            "production_shared_delegation_session_forbidden",
            "Production parent and child Agents require distinct session namespaces",
        )

    endpoint = context.resolved_endpoint
    if endpoint is not None and endpoint.endpoint.kind == "remote":
        if not endpoint.endpoint.approved:
            _add_violation(
                violations,
                "production_remote_endpoint_not_approved",
                "Production remote endpoints must be approved",
            )
    if context.unapproved_remote_endpoint_refs:
        _add_violation(
            violations,
            "production_remote_endpoint_not_approved",
            "Production remote endpoints must be approved",
        )
    if context.unapproved_delegation_endpoint_refs:
        _add_violation(
            violations,
            "production_delegation_endpoint_not_approved",
            "Production delegated endpoints must be approved",
        )
    if context.unapproved_plugin_refs:
        _add_violation(
            violations,
            "production_plugin_not_approved",
            "Production plugins must be approved",
        )
    if context.content_telemetry_enabled and (
        context.content_telemetry_policy_ref is None
    ):
        _add_violation(
            violations,
            "production_content_telemetry_policy_required",
            "Content telemetry requires an explicit policy",
        )


def _validate_required_bindings(
    bindings: RuntimeBindings,
    violations: list[ProfileViolation],
) -> None:
    required_methods = {
        "model_router": ("resolve",),
        "tool_registry": ("require",),
        "skill_registry": ("require",),
        "conversation_store": ("load", "append", "clear"),
        "checkpoint_store": ("load", "save", "delete"),
        "tool_authorizer": ("authorize",),
        "secret_resolver": ("resolve",),
        "event_sink": ("publish", "emit"),
        "diagnostic_sink": ("capture",),
    }
    for field_name, alternatives in required_methods.items():
        component = getattr(bindings, field_name)
        if component is None:
            _add_violation(
                violations,
                f"production_{field_name}_required",
                f"Production requires the {field_name} binding",
            )
            continue
        if field_name == "event_sink":
            valid = any(
                callable(getattr(component, method, None)) for method in alternatives
            )
        else:
            valid = all(
                callable(getattr(component, method, None)) for method in alternatives
            )
        if not valid:
            _add_violation(
                violations,
                f"production_{field_name}_invalid",
                f"Production {field_name} does not implement its protocol",
            )


def _supports_atomic_append(store: object) -> bool:
    return getattr(store, "supports_idempotent_append", False) is True and callable(
        getattr(store, "append_once", None)
    )


def _event_sink_is_unsafe(sink: object | None) -> bool:
    if sink is None or isinstance(sink, NoopEventSink):
        return True
    if isinstance(sink, CompositeEventSink):
        return not sink.sinks or all(
            _event_sink_is_unsafe(child) for child in sink.sinks
        )
    return False


def _event_sink_has_sealed_content_projection(sink: object | None) -> bool:
    """Accept only built-ins whose publish methods cannot be overridden."""

    if type(sink) in {LoggingEventSink, MetricsEventSink, AuditEventSink}:
        return True
    if type(sink) is CompositeEventSink:
        return bool(sink.sinks) and all(
            _event_sink_has_sealed_content_projection(child) for child in sink.sinks
        )
    return False


def _diagnostic_sink_is_unsafe(sink: object | None) -> bool:
    if sink is None or isinstance(sink, (NoopDiagnosticSink, InMemoryDiagnosticSink)):
        return True
    if isinstance(sink, CompositeDiagnosticSink):
        return not sink.sinks or all(
            _diagnostic_sink_is_unsafe(child) for child in sink.sinks
        )
    return False


def _valid_delegation_limits(value: object | None) -> bool:
    if value is None:
        return False
    for field_name in ("max_depth", "max_delegations"):
        item = getattr(value, field_name, None)
        if type(item) is not int or item < 1:
            return False
    timeout = getattr(value, "timeout_seconds", None)
    return (
        not isinstance(timeout, bool)
        and isinstance(timeout, (int, float))
        and math.isfinite(float(timeout))
        and timeout > 0
    )


def _context_present(value: object | None) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value)
    return True


def _context_identity(
    value: object | None,
    keys: tuple[str, ...],
) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if isinstance(value, Mapping):
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _valid_context_provider(value: object) -> bool:
    return callable(value) or callable(getattr(value, "resolve", None))


def _validated_name_set(values: Iterable[str], field_name: str) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of names")
    result = frozenset(values)
    for value in result:
        _validate_policy_ref(value, f"{field_name} item")
    return result


def _validated_policy_mapping(values: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError("policy references must be a mapping")
    result: dict[str, str] = {}
    for name, reference in values.items():
        _validate_policy_ref(name, "policy target")
        _validate_policy_ref(reference, "policy reference")
        result[name] = reference
    return MappingProxyType(dict(sorted(result.items())))


def _validate_policy_ref(value: object, field_name: str) -> None:
    if not isinstance(value, str) or _POLICY_REF_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a stable reference")


def _validated_reference_tuple(
    values: Iterable[str],
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of references")
    result = tuple(values)
    for value in result:
        _validate_policy_ref(value, f"{field_name} item")
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} cannot contain duplicates")
    return result


def _add_violation(
    violations: list[ProfileViolation],
    code: str,
    message: str,
) -> None:
    if any(existing.code == code for existing in violations):
        return
    violations.append(ProfileViolation(code, message))


__all__ = [
    "DevelopmentProfile",
    "ProductionProfile",
    "ProfileViolation",
    "RuntimeProfile",
    "RuntimeProfileError",
    "RuntimeProfileKind",
    "RuntimeValidationContext",
    "TestProfile",
]
