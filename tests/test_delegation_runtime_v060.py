from __future__ import annotations

import asyncio

from pydantic import BaseModel, ConfigDict

from moduagent import (
    Agent,
    AgentDefinition,
    AgentEndpoint,
    DefinitionStatus,
    EventType,
    InMemoryDiagnosticSink,
    InMemoryAgentRegistry,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    RunLimits,
    RuntimeBindings,
    ToolCall,
    ToolExecutionContext,
    ToolExecutionIdentity,
    function_tool,
)
from moduagent.delegation import DelegationCoordinator, DelegationPolicy


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str


class ResearchAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str


class ChildModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return ModelResponse(Message.assistant('{"answer":"verified evidence"}'))


class ChildToolModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return ModelResponse(
                Message.assistant(""),
                tool_calls=(
                    ToolCall(
                        "child-tool-1",
                        "inspect_child_identity",
                        {},
                    ),
                ),
            )
        return ModelResponse(Message.assistant('{"answer":"identity verified"}'))


class ParentModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return ModelResponse(
                Message.assistant(""),
                tool_calls=(
                    ToolCall(
                        "delegate-1",
                        "ask_specialist",
                        {"question": "verify the deployment"},
                    ),
                ),
            )
        return ModelResponse(Message.assistant("parent completed"))


def _definition(
    agent_id: str,
    *,
    tool_refs: tuple[str, ...] = (),
    callable_by: frozenset[str] = frozenset(),
) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        version="1.0.0",
        description=f"{agent_id} test endpoint",
        instructions_ref=f"instructions/{agent_id}/1",
        execution_profile="standard",
        model_route=f"model/{agent_id}",
        tool_refs=tool_refs,
        skill_refs=(),
        input_contract_ref=f"contract/{agent_id}/input/1",
        output_contract_ref=f"contract/{agent_id}/output/1",
        memory_policy_ref="memory/full/development",
        authorization_policy_ref="policy/test/delegation",
        data_classification="internal",
        side_effect_level="none",
        approval_requirement="none",
        callable_by=callable_by,
        limits=RunLimits(),
    )


def test_real_parent_child_runtime_shares_budget_and_bridges_events() -> None:
    async def scenario() -> None:
        parent_definition = _definition(
            "supervisor",
            tool_refs=("ask_specialist",),
        )
        child_definition = _definition(
            "specialist",
            callable_by=frozenset({"supervisor"}),
        )
        child_model = ChildModel()
        child_diagnostics = InMemoryDiagnosticSink()
        child = Agent.create(
            name="specialist",
            model=child_model,
            instructions="Return verified evidence as JSON.",
            output=ResearchAnswer,
            definition=child_definition,
            diagnostic_sink=child_diagnostics,
        )
        registry = InMemoryAgentRegistry()
        registry.register(
            child_definition,
            AgentEndpoint(handler=child, approved=True),
            status=DefinitionStatus.ACTIVE,
        )
        coordinator = DelegationCoordinator(
            registry=registry,
            policy=DelegationPolicy(
                allowed_edges={"supervisor": {"specialist"}},
                allowed_tenants={"tenant-a"},
                allowed_principals={"analyst-1"},
            ),
        )
        delegated_tool = child.as_tool(
            coordinator=coordinator,
            caller=parent_definition.ref,
            input_model=ResearchRequest,
            output_model=ResearchAnswer,
            name="ask_specialist",
            max_result_bytes=4096,
        )
        assert delegated_tool.max_result_bytes == 4096
        parent_model = ParentModel()
        parent = Agent.create(
            name="supervisor",
            model=parent_model,
            instructions="Ask the specialist once, then answer.",
            tools=(delegated_tool,),
            definition=parent_definition,
            runtime_bindings=RuntimeBindings(
                tenant_context_provider=lambda: "tenant-a",
                principal_context_provider=lambda: "analyst-1",
            ),
        )

        events = [
            event
            async for event in parent.stream_all(
                "verify",
                session_id="parent-session",
            )
        ]
        result = events[-1].data["result"]

        assert result.output == "parent completed"
        assert len(parent_model.requests) == 2
        assert len(child_model.requests) == 1, [
            (
                record.code,
                record.exception_type,
                record.cause_types,
                dict(record.safe_details),
                tuple(
                    (frame.filename, frame.function, frame.lineno)
                    for frame in record.frames
                ),
            )
            for record in child_diagnostics.records
        ]
        state = await coordinator.budget_ledger.load_group(result.run_id)
        assert state is not None
        assert state.model_turns == 3
        assert state.tool_calls == 1
        assert state.delegation_count == 1
        delegation_events = [
            event for event in events if event.type.value.startswith("delegation_")
        ]
        assert [event.type for event in delegation_events] == [
            EventType.DELEGATION_REQUESTED,
            EventType.DELEGATION_AUTHORIZED,
            EventType.DELEGATION_STARTED,
            EventType.DELEGATION_COMPLETED,
        ]
        assert all(
            event.execution_group_id == result.run_id for event in delegation_events
        )
        assert all(event.depth == 0 for event in delegation_events)

    asyncio.run(scenario())


def test_delegated_child_tool_receives_only_runtime_owned_identity() -> None:
    async def scenario() -> None:
        observed_contexts: list[ToolExecutionContext] = []

        @function_tool(name="inspect_child_identity")
        async def inspect_child_identity(context: ToolExecutionContext) -> str:
            observed_contexts.append(context)
            return "identity checked"

        parent_definition = _definition(
            "supervisor",
            tool_refs=("ask_specialist",),
        )
        child_definition = _definition(
            "specialist",
            tool_refs=("inspect_child_identity",),
            callable_by=frozenset({"supervisor"}),
        )
        child_model = ChildToolModel()
        child = Agent.create(
            name="specialist",
            model=child_model,
            instructions="Inspect runtime identity, then return JSON.",
            tools=(inspect_child_identity,),
            output=ResearchAnswer,
            definition=child_definition,
        )
        registry = InMemoryAgentRegistry()
        registry.register(
            child_definition,
            AgentEndpoint(handler=child, approved=True),
            status=DefinitionStatus.ACTIVE,
        )
        coordinator = DelegationCoordinator(
            registry=registry,
            policy=DelegationPolicy(
                allowed_edges={"supervisor": {"specialist"}},
                allowed_tenants={"tenant-a"},
                allowed_principals={"analyst-1"},
            ),
        )
        parent = Agent.create(
            name="supervisor",
            model=ParentModel(),
            instructions="Ask the specialist once, then answer.",
            tools=(
                child.as_tool(
                    coordinator=coordinator,
                    caller=parent_definition.ref,
                    input_model=ResearchRequest,
                    output_model=ResearchAnswer,
                    name="ask_specialist",
                ),
            ),
            definition=parent_definition,
            runtime_bindings=RuntimeBindings(
                tenant_context_provider=lambda: "tenant-a",
                principal_context_provider=lambda: "analyst-1",
            ),
        )

        result = await parent.run(
            "verify",
            session_id="parent-identity-session",
            user_context={
                "arbitrary_parent_field": "parent-only",
                "secret": "PARENT_SECRET_MUST_NOT_CROSS",
                "tenant_id": "spoofed-tenant",
                "principal_id": "spoofed-principal",
            },
        )

        assert result.output == "parent completed"
        # Tool + typed-output draft + staged FINALIZE.
        assert len(child_model.requests) == 3
        assert observed_contexts and len(observed_contexts) == 1
        child_context = observed_contexts[0]
        assert child_context.trusted_identity == ToolExecutionIdentity(
            tenant_id="tenant-a",
            principal_id="analyst-1",
            delegated=True,
        )
        assert dict(child_context.user_context) == {}
        assert "PARENT_SECRET_MUST_NOT_CROSS" not in repr(child_model.requests)
        assert "parent-only" not in repr(child_model.requests)

    asyncio.run(scenario())
