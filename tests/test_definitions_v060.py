from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from moduagent.config import RunLimits
from moduagent.definitions import (
    AgentDefinition,
    AgentDefinitionConflictError,
    AgentDefinitionNotRunnableError,
    AgentEndpoint,
    AgentRef,
    DefinitionStatus,
    InMemoryAgentRegistry,
    REQUIRED_SEMANTIC_DIGEST_KEYS,
    RuntimeBindings,
    RuntimeAttestation,
)


def _digests(*, reversed_order: bool = False) -> dict[str, str]:
    values = {
        key: f"sha256:{index:064x}"
        for index, key in enumerate(sorted(REQUIRED_SEMANTIC_DIGEST_KEYS), start=1)
    }
    return dict(reversed(tuple(values.items()))) if reversed_order else values


def _definition(
    *,
    version: str = "2.1.0",
    semantic_digests: dict[str, str] | None = None,
) -> AgentDefinition:
    return AgentDefinition(
        agent_id="researcher",
        version=version,
        description="Collects and verifies internal evidence.",
        instructions_ref="instructions://researcher/2.1.0",
        execution_profile="plan",
        model_route="internal-reasoning",
        tool_refs=("search_documents", "read_document"),
        skill_refs=("internal-research",),
        input_contract_ref="schema://research-request/1",
        output_contract_ref="schema://research-report/1",
        memory_policy_ref="memory://isolated/1",
        authorization_policy_ref="policy://research-read-only/1",
        data_classification="confidential",
        side_effect_level="read",
        approval_requirement="none",
        callable_by=frozenset({"supervisor", "audit-supervisor"}),
        limits=RunLimits(
            max_steps=5,
            max_tool_calls=8,
            max_model_turns=12,
            timeout_seconds=60,
        ),
        semantic_digests=_digests() if semantic_digests is None else semantic_digests,
    )


def test_agent_ref_requires_an_exact_semantic_version() -> None:
    assert AgentRef.parse("researcher@2.1.0") == AgentRef("researcher", "2.1.0")
    assert str(AgentRef("researcher", "2.1.0-rc.1+build.7")) == (
        "researcher@2.1.0-rc.1+build.7"
    )

    for invalid in ("latest", "2.1", "v2.1.0", "2.01.0", "2.1.0-01"):
        with pytest.raises(ValueError, match="Semantic Version"):
            AgentRef("researcher", invalid)
    with pytest.raises(ValueError, match="<agent_id>@<version>"):
        AgentRef.parse("researcher")
    with pytest.raises(TypeError, match="exact AgentRef"):
        InMemoryAgentRegistry().resolve("researcher@2.1.0")  # type: ignore[arg-type]


def test_definition_uses_the_pdf_side_effect_contract() -> None:
    advisory = replace(_definition(), side_effect_level="advisory")
    assert advisory.side_effect_level == "advisory"

    with pytest.raises(ValueError, match="side_effect_level"):
        replace(_definition(), side_effect_level="external")


def test_definition_fingerprint_is_canonical_immutable_and_binding_free() -> None:
    first = _definition()
    second = replace(
        first,
        callable_by=frozenset(reversed(sorted(first.callable_by))),
        limits=replace(first.limits, timeout_seconds=60.0),
        semantic_digests=_digests(reversed_order=True),
    )

    assert first.fingerprint == second.fingerprint
    assert first.fingerprint.startswith("sha256:")
    assert len(first.fingerprint) == 71
    assert first.canonical_json() == second.canonical_json()
    assert "credential" not in first.canonical_json()
    assert "endpoint" not in first.canonical_json()
    assert hash(first) == hash(second)

    before = first.fingerprint
    router = object()
    resolver = object()
    bindings = RuntimeBindings(model_router=router, secret_resolver=resolver)
    with pytest.raises(FrozenInstanceError):
        bindings.model_router = object()
    with pytest.raises(FrozenInstanceError):
        bindings.secret_resolver = object()
    assert bindings.model_router is router
    assert bindings.secret_resolver is resolver
    assert first.fingerprint == before
    assert "object" not in repr(bindings)

    with pytest.raises(FrozenInstanceError):
        first.version = "2.2.0"  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.semantic_digests["tools"] = "sha256:" + ("f" * 64)  # type: ignore[index]


def test_definition_fingerprint_changes_with_execution_semantics() -> None:
    definition = _definition()
    changed_limits = replace(
        definition,
        limits=replace(definition.limits, max_model_turns=13),
    )
    changed_digest_values = dict(definition.semantic_digests)
    changed_digest_values["tools"] = "sha256:" + ("f" * 64)
    changed_digest = replace(definition, semantic_digests=changed_digest_values)

    assert changed_limits.fingerprint != definition.fingerprint
    assert changed_digest.fingerprint != definition.fingerprint
    assert replace(definition, description="New description").fingerprint != (
        definition.fingerprint
    )


@pytest.mark.parametrize(
    ("field_name", "value", "error"),
    [
        ("agent_id", "bad agent", "agent_id"),
        ("description", " padded ", "description"),
        ("execution_profile", "not stable", "execution_profile"),
        ("tool_refs", ("duplicate", "duplicate"), "duplicates"),
        ("skill_refs", "one-skill", "iterable"),
        ("callable_by", "supervisor", "iterable"),
    ],
)
def test_definition_strictly_rejects_ambiguous_values(
    field_name: str,
    value: object,
    error: str,
) -> None:
    definition = _definition()
    with pytest.raises((TypeError, ValueError), match=error):
        replace(definition, **{field_name: value})


def test_definition_rejects_noncanonical_digests() -> None:
    with pytest.raises(ValueError, match="canonical sha256"):
        _definition(semantic_digests={"tools": "not-a-digest"})
    with pytest.raises(ValueError, match="lowercase"):
        _definition(semantic_digests={"Tool": "sha256:" + ("a" * 64)})
    with pytest.raises(ValueError, match="non-secret reference"):
        replace(
            _definition(),
            instructions_ref="https://user:password@example.test/prompt?token=secret",
        )


def test_registry_pins_versions_and_enforces_lifecycle() -> None:
    registry = InMemoryAgentRegistry()
    first = _definition(version="2.1.0")
    second = _definition(version="2.2.0")
    first_endpoint = AgentEndpoint(
        handler=object(),
        supports_async=True,
        supports_stream=True,
    )
    second_endpoint = AgentEndpoint(handler=object())
    registry.register(first, first_endpoint)
    registry.register(second, second_endpoint, status=DefinitionStatus.ACTIVE)

    with pytest.raises(AgentDefinitionNotRunnableError):
        registry.resolve(first.ref)
    with pytest.raises(ValueError, match="expected reviewed"):
        registry.transition(first.ref, DefinitionStatus.APPROVED)

    registry.transition(first.ref, DefinitionStatus.REVIEWED)
    approved = registry.transition(
        first.ref,
        DefinitionStatus.APPROVED,
        expected_fingerprint=first.fingerprint,
    )
    assert approved.status is DefinitionStatus.APPROVED
    assert registry.resolve(first.ref).definition is first

    registry.transition(first.ref, DefinitionStatus.STAGED)
    with pytest.raises(AgentDefinitionNotRunnableError):
        registry.resolve(first.ref)
    registry.transition(first.ref, DefinitionStatus.ACTIVE)
    assert registry.resolve(first.ref).endpoint is first_endpoint
    assert registry.resolve(second.ref).endpoint is second_endpoint
    assert [item.ref for item in registry.descriptors(agent_id="researcher")] == [
        first.ref,
        second.ref,
    ]


def test_registry_descriptor_is_minimal_and_endpoint_rebinding_preserves_pin() -> None:
    registry = InMemoryAgentRegistry()
    definition = _definition()
    secret_handler = {"credential": "must-not-appear", "url": "https://private"}
    endpoint = AgentEndpoint(handler=secret_handler, approved=True)
    registry.register(definition, endpoint, status=DefinitionStatus.ACTIVE)

    descriptor = registry.descriptor(definition.ref)
    encoded = descriptor.to_dict()
    assert encoded["agent_id"] == "researcher"
    assert (
        encoded["input_contract_digest"]
        == definition.semantic_digests["input_contract"]
    )
    assert "instructions" not in encoded
    assert "endpoint" not in encoded
    assert "credential" not in repr(endpoint)

    replacement = AgentEndpoint(handler=object(), kind="remote", approved=True)
    resolved = registry.rebind_endpoint(
        definition.ref,
        replacement,
        expected_fingerprint=definition.fingerprint,
    )
    assert resolved.endpoint is replacement
    assert resolved.definition_fingerprint == definition.fingerprint
    with pytest.raises(AgentDefinitionConflictError, match="precondition"):
        registry.rebind_endpoint(
            definition.ref,
            endpoint,
            expected_fingerprint="sha256:" + ("0" * 64),
        )
    with pytest.raises(AgentDefinitionConflictError, match="already registered"):
        registry.register(definition, endpoint)


def test_runtime_bindings_pin_identity_providers_after_composition() -> None:
    provider = object()
    bindings = RuntimeBindings(tenant_context_provider=provider)

    with pytest.raises(FrozenInstanceError):
        bindings.tenant_context_provider = None

    assert bindings.tenant_context_provider is provider


def test_runtime_attestation_requires_a_pinned_non_secret_source() -> None:
    attestation = RuntimeAttestation.create(
        source_ref="attestation://ci/build-42",
        external_io_enabled=False,
        deterministic_components=True,
    )
    assert attestation.deterministic_components is True

    with pytest.raises(ValueError, match="non-secret"):
        replace(attestation, source_ref="https://user@host?token=secret")
    with pytest.raises(ValueError, match="sha256"):
        replace(attestation, fingerprint="unsigned")
    with pytest.raises(ValueError, match="does not match"):
        replace(attestation, fingerprint="sha256:" + "0" * 64)
