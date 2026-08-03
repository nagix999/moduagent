from __future__ import annotations

from typing import Any

import pytest

from moduagent import (
    Agent,
    AgentConfig,
    AuthorizationDecision,
    ModelCapabilities,
    Plan,
    PlanExecutionProfile,
    PlanStep,
    RetryConfig,
    RunLimits,
    StandardDecisionPolicy,
    StandardExecutionProfile,
    function_tool,
)
from moduagent.execution import (
    CodecBackedEngine,
    EngineEmission,
    EngineOutcome,
    EngineStateCodec,
)
from moduagent.messages import FinishReason


class StaticModel:
    capabilities = ModelCapabilities(streaming=False)

    async def complete(self, request: Any) -> Any:
        raise AssertionError("the model should not be called")


class StaticPlanGenerator:
    async def create(self, context: Any) -> Plan:
        return Plan([PlanStep("inspect")])

    async def revise(self, context: Any, plan: Plan, feedback: str) -> Plan:
        return plan


class _Falsey:
    def __bool__(self) -> bool:
        return False


class FalseyOutputCodec(_Falsey):
    def schema(self) -> None:
        return None

    def decode(self, response: Any) -> str:
        del response
        return "decoded"


class FalseyConversationStore(_Falsey):
    supports_idempotent_append = True

    async def load(self, session_id: str) -> list[Any]:
        del session_id
        return []

    async def append(self, session_id: str, messages: Any) -> None:
        del session_id, messages

    async def append_once(
        self,
        session_id: str,
        idempotency_key: str,
        messages: Any,
    ) -> bool:
        del session_id, idempotency_key, messages
        return True

    async def clear(self, session_id: str) -> None:
        del session_id


class FalseyEventSink(_Falsey):
    async def publish(self, event: Any) -> None:
        del event


class FalseyDiagnosticSink(_Falsey):
    async def capture(self, record: Any) -> None:
        del record


class FalseyAuthorizer(_Falsey):
    async def authorize(self, *args: Any, **kwargs: Any) -> AuthorizationDecision:
        del args, kwargs
        return AuthorizationDecision.allow()


class FalseyDecisionPolicy(StandardDecisionPolicy):
    def __bool__(self) -> bool:
        return False


def test_agent_inspect_is_deterministic_and_secret_safe() -> None:
    @function_tool(
        idempotent=True,
        repair_safe=True,
        timeout_retry_safe=True,
    )
    def lookup(customer_id: int) -> str:
        return str(customer_id)

    config = AgentConfig(
        name="inspect-agent",
        instructions="Inspect safely.",
        model_options={
            "temperature": 0,
            "api_key": "must-not-leak",
            "nested": {"access_token": "also-secret"},
        },
    )
    first = Agent(config=config, model=StaticModel(), tools=[lookup])
    second = Agent(config=config, model=StaticModel(), tools=[lookup])

    inspected = first.inspect()
    payload = inspected.to_dict()

    assert inspected.agent_fingerprint == second.inspect().agent_fingerprint
    assert payload["model"]["options"]["api_key"] == "[REDACTED]"
    assert payload["model"]["options"]["nested"]["access_token"] == "[REDACTED]"
    assert "must-not-leak" not in repr(payload)
    assert payload["execution_profile"]["engine_id"] == "standard"
    assert payload["tools"][0]["safety_profile"] == {
        "same_call_retry_safe": True,
        "changed_argument_repair_safe": True,
        "timeout_retry_safe": True,
    }
    with pytest.raises(TypeError):
        inspected.model_options["temperature"] = 1  # type: ignore[index]


def test_explicit_execution_profiles_resolve_without_changing_core_api() -> None:
    standard = Agent(
        config=AgentConfig("standard", "Answer."),
        model=StaticModel(),
        execution_profile=StandardExecutionProfile(),
    )
    plan = Agent(
        config=AgentConfig("plan", "Plan and answer."),
        model=StaticModel(),
        execution_profile=PlanExecutionProfile(StaticPlanGenerator()),
    )

    assert standard.inspect().execution_profile.kind == "standard"
    assert plan.inspect().execution_profile.kind == "plan"
    assert plan.inspect().execution_profile.state_version == 1


def test_falsey_injected_components_are_not_replaced_by_defaults() -> None:
    output_codec = FalseyOutputCodec()
    conversation_store = FalseyConversationStore()
    event_sink = FalseyEventSink()
    diagnostic_sink = FalseyDiagnosticSink()
    authorizer = FalseyAuthorizer()
    policy = FalseyDecisionPolicy()

    agent = Agent(
        config=AgentConfig("falsey", "Preserve injected components."),
        model=StaticModel(),
        output_codec=output_codec,
        conversation_store=conversation_store,
        event_sink=event_sink,
        diagnostic_sink=diagnostic_sink,
        tool_authorizer=authorizer,
        decision_policy=policy,
    )

    assert agent.runtime.output_codec is output_codec
    assert agent.runtime.conversation_store is conversation_store
    assert agent.runtime.event_sink is event_sink
    assert agent.runtime.diagnostic_reporter is not None
    assert agent.runtime.diagnostic_reporter.sink is diagnostic_sink
    assert agent.tool_executor.authorizer is authorizer
    assert agent.runtime.decision_policy is policy


def test_execution_profile_and_legacy_policy_are_mutually_exclusive() -> None:
    from moduagent import StandardDecisionPolicy

    with pytest.raises(ValueError, match="either decision_policy or execution_profile"):
        Agent(
            config=AgentConfig("invalid", "Answer."),
            model=StaticModel(),
            decision_policy=StandardDecisionPolicy(),
            execution_profile=StandardExecutionProfile(),
        )


def test_composition_rejects_declared_capability_mismatch() -> None:
    class NoToolsModel(StaticModel):
        capabilities = ModelCapabilities(
            streaming=False,
            tool_calling=False,
        )

    @function_tool
    def lookup(value: str) -> str:
        return value

    with pytest.raises(ValueError, match="does not support tool calling"):
        Agent(
            config=AgentConfig("invalid", "Use a Tool."),
            model=NoToolsModel(),
            tools=[lookup],
        )


def test_inspect_reports_resolved_staged_finalization_for_unsupported_combo() -> None:
    class SeparateContractModel(StaticModel):
        capabilities = ModelCapabilities(
            streaming=False,
            tool_calling_with_structured_output=False,
        )

    class StructuredCodec:
        def schema(self) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            }

        def decode(self, response: Any) -> Any:
            return response

    @function_tool
    def lookup(value: str) -> str:
        return value

    agent = Agent(
        config=AgentConfig(
            "separated-contracts",
            "Use the Tool when needed.",
            finalization_mode="disabled",
        ),
        model=SeparateContractModel(),
        tools=[lookup],
        output_codec=StructuredCodec(),
    )

    spec = agent.inspect()
    assert spec.model_capabilities["tool_calling_with_structured_output"] is False
    assert spec.output_contract["staged_finalization"] is True


def test_fingerprint_tracks_semantic_model_options_and_retry_policy() -> None:
    first = Agent(
        config=AgentConfig(
            "fingerprint",
            "Answer.",
            retry=RetryConfig(max_attempts=1),
            model_options={"max_tokens": 16},
        ),
        model=StaticModel(),
    )
    second = Agent(
        config=AgentConfig(
            "fingerprint",
            "Answer.",
            retry=RetryConfig(max_attempts=2),
            model_options={"max_tokens": 4096},
        ),
        model=StaticModel(),
    )

    assert first.inspect().model_options["max_tokens"] == 16
    assert second.inspect().model_options["max_tokens"] == 4096
    assert first.inspect().agent_fingerprint != second.inspect().agent_fingerprint


def test_inspection_and_fingerprint_track_model_guard_limits() -> None:
    first = Agent(
        config=AgentConfig(
            "model-guard",
            "Answer.",
            limits=RunLimits(
                max_model_turns=8,
                no_progress_model_turn_threshold=2,
            ),
        ),
        model=StaticModel(),
    )
    second = Agent(
        config=AgentConfig(
            "model-guard",
            "Answer.",
            limits=RunLimits(
                max_model_turns=9,
                no_progress_model_turn_threshold=3,
            ),
        ),
        model=StaticModel(),
    )

    assert first.inspect().to_dict()["limits"]["max_model_turns"] == 8
    assert first.inspect().to_dict()["limits"]["no_progress_model_turn_threshold"] == 2
    assert first.inspect().agent_fingerprint != second.inspect().agent_fingerprint


def test_inspection_redacts_headers_url_credentials_and_repr_prompt() -> None:
    class EndpointModel(StaticModel):
        model = "company-model"
        base_url = "https://user:password@example.test/v1?access_token=must-not-leak"

    agent = Agent(
        config=AgentConfig(
            "redaction",
            "private system instruction",
            model_options={
                "headers": {"X-Service-Key": "must-not-leak"},
                "max_tokens": 128,
            },
        ),
        model=EndpointModel(),
    )

    spec = agent.inspect()
    payload = spec.to_dict()

    assert payload["model"]["options"]["max_tokens"] == 128
    assert payload["model"]["options"]["headers"] == {"X-Service-Key": "[REDACTED]"}
    assert "user:password" not in payload["model"]["identity"]["base_url"]
    assert "must-not-leak" not in repr(payload)
    assert "private system instruction" not in repr(spec)


def test_standard_profile_rejects_plan_policy_at_composition() -> None:
    from moduagent import PlanAndExecutePolicy

    with pytest.raises(ValueError, match="cannot use PlanAndExecutePolicy"):
        Agent(
            config=AgentConfig("invalid-standard", "Answer."),
            model=StaticModel(),
            execution_profile=StandardExecutionProfile(
                decision_policy=PlanAndExecutePolicy(StaticPlanGenerator())
            ),
        )


def test_custom_execution_engine_is_composed_inspected_and_executed() -> None:
    class CustomCodec(EngineStateCodec[dict[str, int]]):
        engine_id = "approval"
        state_version = 1

        def encode(self, state: dict[str, int]) -> dict[str, int]:
            return dict(state)

        def decode(self, payload: Any) -> dict[str, int]:
            return {"turn": int(payload.get("turn", 0))}

        def migrate(self, from_version: int, payload: Any) -> dict[str, int]:
            if from_version != 1:
                raise ValueError("unsupported state")
            return self.encode(self.decode(payload))

    class CustomEngine(CodecBackedEngine[dict[str, int]]):
        engine_id = "approval"
        state_version = 1
        configuration = {"mode": "single-approval"}
        required_capabilities = {"chat": True}

        def __init__(self) -> None:
            self.state_codec = CustomCodec()

        async def initialize(self, context: Any, services: Any) -> dict[str, int]:
            del context, services
            return {"turn": 0}

        async def execute(self, context: Any, state: Any, services: Any):
            del context, services
            state["turn"] += 1
            yield EngineEmission(
                outcome=EngineOutcome(
                    FinishReason.COMPLETED,
                    output=f"custom-{state['turn']}",
                )
            )

    async def scenario() -> None:
        engine = CustomEngine()
        agent = Agent(
            config=AgentConfig("custom", "Run the custom engine."),
            model=StaticModel(),
            execution_engine=engine,
        )

        inspected = agent.inspect()
        result = await agent.run("execute", session_id="custom-session")

        assert inspected.execution_profile.kind == "custom"
        assert inspected.execution_profile.engine_id == "approval"
        assert inspected.execution_profile.details["engine_type"].endswith(
            ".CustomEngine"
        )
        assert inspected.execution_profile.details["configuration"] == {
            "mode": "single-approval"
        }
        assert agent.runtime.engine is engine
        assert result.output == "custom-1"

    import asyncio

    asyncio.run(scenario())


def test_custom_engine_credentials_do_not_change_semantic_fingerprint() -> None:
    class CustomCodec(EngineStateCodec[dict[str, int]]):
        engine_id = "credential-safe"
        state_version = 1

        def encode(self, state: dict[str, int]) -> dict[str, int]:
            return dict(state)

        def decode(self, payload: Any) -> dict[str, int]:
            return {"value": int(payload.get("value", 0))}

    class CustomEngine(CodecBackedEngine[dict[str, int]]):
        engine_id = "credential-safe"
        state_version = 1
        required_capabilities = {"chat": True}

        def __init__(self, api_key: str) -> None:
            self.state_codec = CustomCodec()
            self.configuration = {
                "mode": "stable",
                "api_key": api_key,
                "clientSecret": f"client-{api_key}",
                "accessToken": f"access-{api_key}",
                "bearer": f"bearer-{api_key}",
                "headers": {"Authorization": f"Bearer {api_key}"},
            }

        async def initialize(self, context: Any, services: Any) -> dict[str, int]:
            del context, services
            return {"value": 0}

        async def execute(self, context: Any, state: Any, services: Any):
            del context, state, services
            yield EngineEmission(
                outcome=EngineOutcome(FinishReason.COMPLETED, output="done")
            )

    first = Agent(
        config=AgentConfig("credential-safe", "Run."),
        model=StaticModel(),
        execution_engine=CustomEngine("first-secret"),
    )
    second = Agent(
        config=AgentConfig("credential-safe", "Run."),
        model=StaticModel(),
        execution_engine=CustomEngine("rotated-secret"),
    )

    assert first.inspect().agent_fingerprint == second.inspect().agent_fingerprint
    assert (
        first.inspect().execution_profile.details["configuration_fingerprint"]
        == second.inspect().execution_profile.details["configuration_fingerprint"]
    )
    assert "first-secret" not in repr(first.inspect().to_dict())
    assert first.inspect().execution_profile.details["configuration"] == {
        "mode": "stable",
        "api_key": "[REDACTED]",
        "clientSecret": "[REDACTED]",
        "accessToken": "[REDACTED]",
        "bearer": "[REDACTED]",
        "headers": {"Authorization": "[REDACTED]"},
    }


def test_fingerprint_tracks_memory_and_skill_semantics() -> None:
    from moduagent import (
        InMemorySkillSource,
        RecentTurnsConversationMemoryPolicy,
        SkillLimits,
        SkillRegistry,
    )

    registry = SkillRegistry.from_sources(
        InMemorySkillSource(
            {
                "guide": {
                    "SKILL.md": (
                        "---\nname: guide\ndescription: Guide.\n---\nFollow the guide."
                    )
                }
            }
        )
    )
    first = Agent(
        config=AgentConfig("semantic", "Answer."),
        model=StaticModel(),
        conversation_memory_policy=RecentTurnsConversationMemoryPolicy(1),
        skill_registry=registry,
        skill_limits=SkillLimits(max_resource_reads=1),
    )
    second = Agent(
        config=AgentConfig("semantic", "Answer."),
        model=StaticModel(),
        conversation_memory_policy=RecentTurnsConversationMemoryPolicy(99),
        skill_registry=registry,
        skill_limits=SkillLimits(max_resource_reads=99),
    )

    assert first.inspect().agent_fingerprint != second.inspect().agent_fingerprint
    assert (
        first.inspect().persistence_policy["conversation_memory_policy"]["max_turns"]
        == 1
    )
    assert first.inspect().skill_policy["catalog_digest"] == registry.catalog_digest
    assert first.inspect().skill_policy["limits"]["max_resource_reads"] == 1


def test_custom_execution_engine_is_mutually_exclusive_with_profiles() -> None:
    class InvalidEngine:
        pass

    with pytest.raises(
        ValueError,
        match="execution_engine cannot be combined",
    ):
        Agent(
            config=AgentConfig("invalid-custom", "Answer."),
            model=StaticModel(),
            execution_profile=StandardExecutionProfile(),
            execution_engine=InvalidEngine(),  # type: ignore[arg-type]
        )
