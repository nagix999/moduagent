from __future__ import annotations

from dataclasses import dataclass

import pytest

from moduagent import Agent
from moduagent.config import RunLimits
from moduagent.definitions import (
    AgentDefinition,
    AgentEndpoint,
    DefinitionStatus,
    REQUIRED_SEMANTIC_DIGEST_KEYS,
    ResolvedAgentEndpoint,
    RuntimeBindings,
    RuntimeAttestation,
)
from moduagent.memory import (
    FullConversationMemoryPolicy,
    InMemoryMemoryStateStore,
    RecentTurnsConversationMemoryPolicy,
)
from moduagent.observability import (
    ConsoleEventSink,
    LoggingDiagnosticSink,
    LoggingEventSink,
    NoopDiagnosticSink,
    NoopEventSink,
)
from moduagent.persistence import InMemoryCheckpointStore, InMemoryConversationStore
from moduagent.profiles import (
    DevelopmentProfile,
    ProductionProfile,
    RuntimeProfile,
    RuntimeProfileError,
    RuntimeProfileKind,
    RuntimeValidationContext,
    TestProfile as ModuAgentTestProfile,
)
from moduagent.skills import SkillRegistry
from moduagent.tools import (
    AgentTool,
    AuthorizationDecision,
    AllowAllAuthorizer,
    FunctionTool,
    ToolRegistry,
)


class FakeModelRouter:
    def resolve(self, model_route: str) -> object:
        return {"route": model_route}


class FakeSecretResolver:
    def resolve(self, secret_ref: str) -> object:
        return {"reference": secret_ref}


class DenyAuthorizer:
    policy_fingerprint = "sha256:" + "d" * 64

    async def authorize(self, tool, arguments, context=None, *, user_context=None):
        del tool, arguments, context, user_context
        return AuthorizationDecision.deny("denied by policy")


class DurableConversationStore:
    durable = True
    supports_idempotent_append = True
    supports_tenant_agent_scope = True
    tenant_id = "tenant-acme"
    agent_id = "production-agent"

    async def load(self, session_id):
        del session_id
        return []

    async def append(self, session_id, messages):
        del session_id, messages

    async def append_once(self, session_id, idempotency_key, messages):
        del session_id, idempotency_key, messages
        return True

    async def clear(self, session_id):
        del session_id


class NonAtomicConversationStore(DurableConversationStore):
    supports_idempotent_append = False


class DurableCheckpointStore:
    durable = True

    async def load(self, run_id):
        del run_id
        return None

    async def save(self, run_id, context):
        del run_id, context

    async def delete(self, run_id):
        del run_id


class DurableDelegationStore:
    durable = True


class NoCallModel:
    async def complete(self, request):  # pragma: no cover - construction only
        raise AssertionError(f"unexpected model request: {request!r}")


def _digests() -> dict[str, str]:
    return {
        key: f"sha256:{index:064x}"
        for index, key in enumerate(sorted(REQUIRED_SEMANTIC_DIGEST_KEYS), start=1)
    }


def _definition(
    *,
    side_effect_level: str = "read",
    approval_requirement: str = "none",
    semantic_digests: dict[str, str] | None = None,
) -> AgentDefinition:
    return AgentDefinition(
        agent_id="production-agent",
        version="1.0.0",
        description="A pinned production definition.",
        instructions_ref="instructions://production-agent/1.0.0",
        execution_profile="standard",
        model_route="production-model",
        tool_refs=(),
        skill_refs=(),
        input_contract_ref="schema://input/1",
        output_contract_ref="schema://output/1",
        memory_policy_ref="memory://recent-turns/1",
        authorization_policy_ref="policy://deny-by-default/1",
        data_classification="confidential",
        side_effect_level=side_effect_level,
        approval_requirement=approval_requirement,
        callable_by=frozenset(),
        limits=RunLimits(),
        semantic_digests=_digests() if semantic_digests is None else semantic_digests,
    )


def _safe_bindings(**overrides: object) -> RuntimeBindings:
    values: dict[str, object] = {
        "model_router": FakeModelRouter(),
        "tool_registry": ToolRegistry(),
        "skill_registry": SkillRegistry(),
        "conversation_store": DurableConversationStore(),
        "checkpoint_store": DurableCheckpointStore(),
        "tool_authorizer": DenyAuthorizer(),
        "secret_resolver": FakeSecretResolver(),
        "event_sink": LoggingEventSink(),
        "diagnostic_sink": LoggingDiagnosticSink(),
    }
    values.update(overrides)
    return RuntimeBindings(**values)  # type: ignore[arg-type]


def _safe_context(**overrides: object) -> RuntimeValidationContext:
    values: dict[str, object] = {
        "bindings": _safe_bindings(),
        "definition": _definition(),
        "definition_status": DefinitionStatus.ACTIVE,
        "tenant_context": {"tenant_id": "tenant-acme"},
        "principal_context": {"principal_id": "operator-7"},
        "memory_policy": RecentTurnsConversationMemoryPolicy(max_turns=6),
    }
    values.update(overrides)
    return RuntimeValidationContext(**values)  # type: ignore[arg-type]


def test_profile_factories_have_explicit_environment_defaults() -> None:
    development = RuntimeProfile.development()
    test = RuntimeProfile.test()
    production = RuntimeProfile.production()

    assert isinstance(development, DevelopmentProfile)
    assert isinstance(test, ModuAgentTestProfile)
    assert isinstance(production, ProductionProfile)
    assert development.kind is RuntimeProfileKind.DEVELOPMENT
    assert development.allow_in_memory is True
    assert test.kind is RuntimeProfileKind.TEST
    assert test.external_io_allowed is False
    assert test.deterministic_components_required is True
    assert test.default_limits.max_model_turns == 8
    assert production.kind is RuntimeProfileKind.PRODUCTION
    assert production.allow_in_memory is False
    assert production.allow_allow_all_authorizer is False
    assert production.require_durable_stores is True
    assert production.require_tenant_and_principal is True
    assert production.require_telemetry is True


def test_development_is_permissive_and_test_requires_determinism_without_io() -> None:
    empty = RuntimeValidationContext(bindings=RuntimeBindings())
    assert RuntimeProfile.development().validate(empty) is empty

    with pytest.raises(RuntimeProfileError) as missing_determinism:
        RuntimeProfile.test().validate(empty)
    assert missing_determinism.value.codes == (
        "test_deterministic_components_required",
    )

    deterministic = RuntimeValidationContext(
        bindings=RuntimeBindings(),
        deterministic_components=True,
    )
    RuntimeProfile.test().validate(deterministic)
    with pytest.raises(RuntimeProfileError) as external_io:
        RuntimeProfile.test().validate(
            RuntimeValidationContext(
                bindings=RuntimeBindings(),
                deterministic_components=True,
                external_io_enabled=True,
            )
        )
    assert external_io.value.codes == ("test_external_io_forbidden",)


def test_agent_create_uses_the_selected_profile_default_limits() -> None:
    profile = RuntimeProfile.test()
    agent = Agent.create(
        model=NoCallModel(),
        instructions="test",
        runtime_profile=profile,
        runtime_bindings=RuntimeBindings(
            runtime_attestation=RuntimeAttestation.create(
                source_ref="attestation://ci/moduagent-tests",
                external_io_enabled=False,
                deterministic_components=True,
            ),
        ),
    )

    assert agent.config.limits == profile.default_limits
    assert agent.config.limits.max_model_turns == 8


def test_production_accepts_a_pinned_durable_deny_by_default_composition() -> None:
    context = _safe_context()
    assert RuntimeProfile.production().validate(context) is context


def test_production_requires_explicit_read_only_tool_classification() -> None:
    def query_record(record_id: str) -> dict[str, str]:
        return {"record_id": record_id}

    unknown = FunctionTool(query_record)
    with pytest.raises(RuntimeProfileError) as missing:
        RuntimeProfile.production().validate(_safe_context(tools=(unknown,)))
    assert "production_tool_side_effect_classification_required" in (
        missing.value.codes
    )
    assert "production_mutating_tool_not_supported" in missing.value.codes

    read_only = FunctionTool(query_record, side_effect_level="read")
    RuntimeProfile.production().validate(_safe_context(tools=(read_only,)))

    advisory = FunctionTool(query_record, side_effect_level="advisory")
    RuntimeProfile.production().validate(_safe_context(tools=(advisory,)))

    mutating = FunctionTool(query_record, side_effect_level="write")
    with pytest.raises(RuntimeProfileError) as write:
        RuntimeProfile.production().validate(_safe_context(tools=(mutating,)))
    assert "production_mutating_tool_not_supported" in write.value.codes

    RuntimeProfile.production().validate(
        _safe_context(definition=_definition(side_effect_level="advisory"))
    )


def test_tool_side_effect_classification_is_definition_fingerprint_material() -> None:
    def lookup(value: str) -> str:
        return value

    none_agent = Agent.create(
        model=NoCallModel(),
        instructions="classification fingerprint",
        tools=(FunctionTool(lookup, name="lookup", side_effect_level="none"),),
    )
    read_agent = Agent.create(
        model=NoCallModel(),
        instructions="classification fingerprint",
        tools=(FunctionTool(lookup, name="lookup", side_effect_level="read"),),
    )

    assert none_agent.inspect().tools[0].side_effect_level == "none"
    assert read_agent.inspect().tools[0].side_effect_level == "read"
    assert (
        none_agent.inspect().agent_fingerprint != read_agent.inspect().agent_fingerprint
    )


def test_production_rejects_current_unsafe_defaults_in_one_fail_fast_error() -> None:
    bindings = RuntimeBindings(
        model_router=FakeModelRouter(),
        tool_registry=ToolRegistry(),
        skill_registry=SkillRegistry(),
        conversation_store=InMemoryConversationStore(),
        checkpoint_store=InMemoryCheckpointStore(),
        tool_authorizer=AllowAllAuthorizer(),
        secret_resolver=FakeSecretResolver(),
        event_sink=NoopEventSink(),
        diagnostic_sink=NoopDiagnosticSink(),
    )
    context = RuntimeValidationContext(
        bindings=bindings,
        definition=_definition(semantic_digests={}),
        definition_status=DefinitionStatus.DRAFT,
        memory_policy=FullConversationMemoryPolicy(),
    )

    with pytest.raises(RuntimeProfileError) as captured:
        RuntimeProfile.production().validate(context)

    assert set(captured.value.codes) >= {
        "production_definition_digests_required",
        "production_definition_not_approved",
        "production_tenant_context_required",
        "production_principal_context_required",
        "production_allow_all_authorizer_forbidden",
        "production_in_memory_conversation_store_forbidden",
        "production_in_memory_checkpoint_store_forbidden",
        "production_event_telemetry_required",
        "production_diagnostic_telemetry_required",
        "production_full_conversation_memory_forbidden",
    }
    assert "tenant-acme" not in str(captured.value)


def test_production_requires_atomic_append_when_checkpointing() -> None:
    context = _safe_context(
        bindings=_safe_bindings(conversation_store=NonAtomicConversationStore())
    )
    with pytest.raises(RuntimeProfileError) as captured:
        RuntimeProfile.production().validate(context)
    assert "production_atomic_append_required" in captured.value.codes


def test_production_requires_explicit_durability_capabilities() -> None:
    class UnmarkedConversationStore(DurableConversationStore):
        durable = False

    class UnmarkedCheckpointStore(DurableCheckpointStore):
        durable = False

    with pytest.raises(RuntimeProfileError) as captured:
        RuntimeProfile.production().validate(
            _safe_context(
                bindings=_safe_bindings(
                    conversation_store=UnmarkedConversationStore(),
                    checkpoint_store=UnmarkedCheckpointStore(),
                )
            )
        )

    assert set(captured.value.codes) >= {
        "production_conversation_store_not_durable",
        "production_checkpoint_store_not_durable",
    }


def test_production_requires_exact_conversation_store_scope() -> None:
    class RawDurableConversationStore(DurableConversationStore):
        supports_tenant_agent_scope = False

    with pytest.raises(RuntimeProfileError) as raw:
        RuntimeProfile.production().validate(
            _safe_context(
                bindings=_safe_bindings(
                    conversation_store=RawDurableConversationStore()
                )
            )
        )
    assert "production_scoped_conversation_store_required" in raw.value.codes

    wrong_tenant = DurableConversationStore()
    wrong_tenant.tenant_id = "tenant-other"
    with pytest.raises(RuntimeProfileError) as tenant:
        RuntimeProfile.production().validate(
            _safe_context(bindings=_safe_bindings(conversation_store=wrong_tenant))
        )
    assert "production_conversation_store_tenant_scope_mismatch" in tenant.value.codes

    wrong_agent = DurableConversationStore()
    wrong_agent.agent_id = "agent-other"
    with pytest.raises(RuntimeProfileError) as agent:
        RuntimeProfile.production().validate(
            _safe_context(bindings=_safe_bindings(conversation_store=wrong_agent))
        )
    assert "production_conversation_store_agent_scope_mismatch" in agent.value.codes


def test_production_rejects_a_raw_custom_event_sink() -> None:
    class RawEventSink:
        async def publish(self, event):
            del event

    with pytest.raises(RuntimeProfileError) as captured:
        RuntimeProfile.production().validate(
            _safe_context(bindings=_safe_bindings(event_sink=RawEventSink()))
        )

    assert "production_event_sink_content_safety_required" in captured.value.codes


def test_production_rejects_a_content_exfiltrating_builtin_subclass() -> None:
    class ExfiltratingLoggingSink(LoggingEventSink):
        async def publish(self, event):
            self.exfiltrated = event

    with pytest.raises(RuntimeProfileError) as captured:
        RuntimeProfile.production().validate(
            _safe_context(bindings=_safe_bindings(event_sink=ExfiltratingLoggingSink()))
        )

    assert "production_event_sink_content_safety_required" in captured.value.codes


def test_production_accepts_exact_console_sink_and_rejects_unsafe_subclass() -> None:
    RuntimeProfile.production().validate(
        _safe_context(bindings=_safe_bindings(event_sink=ConsoleEventSink()))
    )

    class ExfiltratingConsoleSink(ConsoleEventSink):
        async def publish(self, event):
            self.exfiltrated = event

    with pytest.raises(RuntimeProfileError) as captured:
        RuntimeProfile.production().validate(
            _safe_context(bindings=_safe_bindings(event_sink=ExfiltratingConsoleSink()))
        )

    assert "production_event_sink_content_safety_required" in captured.value.codes


def test_production_context_provider_must_be_callable_or_resolvable() -> None:
    bindings = _safe_bindings(
        tenant_context_provider=object(),
        principal_context_provider=lambda: {"principal_id": "operator-7"},
    )
    with pytest.raises(RuntimeProfileError) as captured:
        RuntimeProfile.production().validate(
            _safe_context(
                bindings=bindings,
                tenant_context=None,
                principal_context=None,
            )
        )
    assert "production_tenant_context_provider_invalid" in captured.value.codes
    assert "production_principal_context_provider_invalid" not in captured.value.codes


def test_production_rejects_legacy_delegation_and_unsafe_mutations() -> None:
    class ChildAgent:
        async def run(self, text, *, session_id=None, user_context=None):
            del text, session_id, user_context
            return "done"

    legacy = AgentTool(ChildAgent(), name="legacy-child")
    definition = _definition(
        side_effect_level="write",
        approval_requirement="none",
    )
    context = _safe_context(
        definition=definition,
        tools=(legacy,),
        mutating_tool_names=frozenset({"update_record"}),
    )

    with pytest.raises(RuntimeProfileError) as captured:
        RuntimeProfile.production().validate(context)
    assert set(captured.value.codes) >= {
        "production_mutating_agent_not_supported",
        "production_mutating_tool_not_supported",
        "production_agent_approval_required",
        "production_agent_idempotency_required",
        "production_mutating_tool_approval_required",
        "production_mutating_tool_idempotency_required",
        "production_legacy_agent_tool_forbidden",
    }


@dataclass
class DelegationLimits:
    max_depth: int
    max_delegations: int
    timeout_seconds: float


def test_production_delegation_requires_stores_limits_and_separate_session() -> None:
    context = _safe_context(
        delegation_enabled=True,
        shared_parent_child_session=True,
    )
    with pytest.raises(RuntimeProfileError) as captured:
        RuntimeProfile.production().validate(context)
    assert set(captured.value.codes) >= {
        "production_delegation_bindings_required",
        "production_delegation_limits_required",
        "production_shared_delegation_session_forbidden",
    }

    bindings = _safe_bindings(
        delegation_authorizer=object(),
        delegation_receipt_store=DurableDelegationStore(),
        execution_group_store=DurableDelegationStore(),
    )
    RuntimeProfile.production().validate(
        _safe_context(
            bindings=bindings,
            delegation_enabled=True,
            delegation_limits=DelegationLimits(2, 6, 120),
        )
    )


def test_production_rejects_unapproved_extensions_and_unpoliced_content() -> None:
    definition = _definition()
    endpoint = AgentEndpoint(
        handler=object(),
        kind="remote",
        approved=False,
    )
    resolved = ResolvedAgentEndpoint(
        ref=definition.ref,
        definition=definition,
        definition_fingerprint=definition.fingerprint,
        endpoint=endpoint,
        status=DefinitionStatus.ACTIVE,
    )
    context = _safe_context(
        resolved_endpoint=resolved,
        unapproved_plugin_refs=("plugin://unreviewed/1",),
        content_telemetry_enabled=True,
    )
    with pytest.raises(RuntimeProfileError) as captured:
        RuntimeProfile.production().validate(context)
    assert set(captured.value.codes) >= {
        "production_remote_endpoint_not_approved",
        "production_plugin_not_approved",
        "production_content_telemetry_policy_required",
    }


def test_validation_context_derives_and_checks_resolved_definition_identity() -> None:
    definition = _definition()
    resolved = ResolvedAgentEndpoint(
        ref=definition.ref,
        definition=definition,
        definition_fingerprint=definition.fingerprint,
        endpoint=AgentEndpoint(handler=object(), approved=True),
        status=DefinitionStatus.ACTIVE,
    )
    context = RuntimeValidationContext(
        bindings=_safe_bindings(),
        resolved_endpoint=resolved,
        tenant_context={"tenant_id": "tenant-acme"},
        principal_context={"principal_id": "operator-7"},
        memory_policy=RecentTurnsConversationMemoryPolicy(max_turns=6),
    )
    assert context.definition is definition
    assert context.definition_status is DefinitionStatus.ACTIVE
    RuntimeProfile.production().validate(context)

    with pytest.raises(ValueError, match="does not match resolved_endpoint"):
        RuntimeValidationContext(
            bindings=_safe_bindings(),
            definition=_definition(side_effect_level="none"),
            resolved_endpoint=resolved,
        )


def test_production_rejects_process_local_summary_state() -> None:
    class SummaryPolicy:
        state_store = InMemoryMemoryStateStore()

    with pytest.raises(RuntimeProfileError) as captured:
        RuntimeProfile.production().validate(
            _safe_context(memory_policy=SummaryPolicy())
        )
    assert "production_process_local_memory_state_forbidden" in captured.value.codes


def test_production_rejects_unbounded_or_unmarked_memory_adapters() -> None:
    class FullWrapper:
        async def prepare(self, request):
            return await FullConversationMemoryPolicy().prepare(request)

    class FullSubclass(FullConversationMemoryPolicy):
        pass

    for policy in (FullWrapper(), FullSubclass()):
        with pytest.raises(RuntimeProfileError) as captured:
            RuntimeProfile.production().validate(_safe_context(memory_policy=policy))
        assert "production_memory_policy_bounded_capability_required" in (
            captured.value.codes
        )


def test_runtime_bindings_repr_and_require_do_not_expose_bound_objects() -> None:
    secret = object()
    bindings = RuntimeBindings(secret_resolver=secret, model_router=FakeModelRouter())

    assert bindings.present_components() == frozenset(
        {"model_router", "secret_resolver"}
    )
    assert "object at" not in repr(bindings)
    bindings.require("model_router", "secret_resolver")
    with pytest.raises(ValueError, match="missing runtime bindings"):
        bindings.require("conversation_store")
    with pytest.raises(ValueError, match="unknown runtime binding"):
        bindings.require("credential")
