from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel

from moduagent import (
    Agent,
    AgentConfig,
    DecisionKind,
    EventType,
    ExecutionDecision,
    InMemoryCheckpointStore,
    InMemoryConversationStore,
    Message,
    ModelCapabilities,
    ModelResponse,
    PydanticOutputCodec,
    ToolCall,
    function_tool,
)


class ScriptedModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[Any] = []

    async def complete(self, request: Any) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


class RecordingCheckpointStore(InMemoryCheckpointStore):
    def __init__(self) -> None:
        super().__init__()
        self.saved_payloads: list[str] = []

    async def save_snapshot(self, snapshot: Any) -> None:
        self.saved_payloads.append(snapshot.to_json())
        await super().save_snapshot(snapshot)


def test_standard_tool_protocol_is_model_only_and_secret_safe_publicly() -> None:
    async def scenario() -> None:
        argument_secret = "RAW-API-KEY-42"
        result_secret = "RAW-CUSTOMER-RESULT-73"
        invocations: list[str] = []

        @function_tool
        def customer_lookup(api_key: str) -> dict[str, str]:
            invocations.append(api_key)
            return {"customer_record": result_secret}

        call = ToolCall(
            "lookup-1",
            "customer_lookup",
            {"api_key": argument_secret},
        )
        model = ScriptedModel(
            [
                ModelResponse(Message.assistant(None, (call,)), (call,)),
                ModelResponse(Message.assistant("public answer")),
            ]
        )
        conversations = InMemoryConversationStore()
        checkpoints = RecordingCheckpointStore()
        agent = Agent(
            config=AgentConfig("privacy", "Use the customer Tool."),
            model=model,
            tools=[customer_lookup],
            conversation_store=conversations,
            checkpoint_store=checkpoints,
        )

        events = [
            event
            async for event in agent.stream_all(
                "look up the customer",
                session_id="privacy-session",
            )
        ]
        result = events[-1].data["result"]
        stored = await conversations.load("privacy-session")

        assert invocations == [argument_secret]
        assert result.output == "public answer"
        assert [message.role.value for message in model.requests[1].messages][-2:] == [
            "assistant",
            "tool",
        ]
        model_protocol = json.dumps(
            [message.to_dict() for message in model.requests[1].messages],
            ensure_ascii=False,
        )
        assert argument_secret in model_protocol
        assert result_secret in model_protocol

        assert [message.role.value for message in stored] == ["user", "assistant"]
        assert stored[-1].content == "public answer"
        assert [message.role.value for message in result.messages] == [
            "system",
            "user",
            "assistant",
        ]
        public_payload = json.dumps(
            {
                "conversation": [message.to_dict() for message in stored],
                "result": {
                    "messages": [message.to_dict() for message in result.messages],
                    "metadata": dict(result.metadata),
                },
                "events": [event.to_dict() for event in events],
                "checkpoints": checkpoints.saved_payloads,
            },
            ensure_ascii=False,
        )
        assert argument_secret not in public_payload
        assert result_secret not in public_payload
        assert all("arguments" not in entry for entry in result.metadata["tool_trace"])

        started = next(
            event for event in events if event.type is EventType.TOOL_STARTED
        )
        assert started.data["arguments_fingerprint"].startswith("sha256:")

    asyncio.run(scenario())


def test_always_finalization_keeps_act_draft_out_of_public_history() -> None:
    async def scenario() -> None:
        draft = "INTERNAL-ACT-DRAFT-ALWAYS"
        final = "PUBLIC-FINAL-ALWAYS"
        model = ScriptedModel(
            [
                ModelResponse(Message.assistant(draft)),
                ModelResponse(Message.assistant(final)),
            ]
        )
        conversations = InMemoryConversationStore()
        agent = Agent(
            config=AgentConfig(
                "always-finalize",
                "Return one public answer.",
                finalization_mode="always",
            ),
            model=model,
            conversation_store=conversations,
        )

        result = await agent.run("question", session_id="always-finalize")
        stored = await conversations.load("always-finalize")

        assert result.output == final
        assert [message.content for message in stored] == ["question", final]
        assert [message.content for message in result.messages] == [
            "Return one public answer.",
            "question",
            final,
        ]
        assert draft not in repr(stored)
        assert draft not in repr(result.messages)

    asyncio.run(scenario())


def test_structured_tool_finalization_keeps_all_act_drafts_internal() -> None:
    class Answer(BaseModel):
        answer: str

    async def scenario() -> None:
        tool_result = "INTERNAL-TOOL-RESULT"
        act_draft = "INTERNAL-POST-TOOL-DRAFT"
        final = '{"answer":"PUBLIC-STRUCTURED-FINAL"}'

        @function_tool
        def lookup(value: str) -> str:
            return tool_result

        call = ToolCall("lookup-structured", "lookup", {"value": "private-input"})
        model = ScriptedModel(
            [
                ModelResponse(Message.assistant(None, (call,)), (call,)),
                ModelResponse(Message.assistant(act_draft)),
                ModelResponse(Message.assistant(final)),
            ]
        )
        conversations = InMemoryConversationStore()
        agent = Agent(
            config=AgentConfig("structured", "Return structured output."),
            model=model,
            tools=[lookup],
            output_codec=PydanticOutputCodec(Answer),
            conversation_store=conversations,
        )

        result = await agent.run("question", session_id="structured-finalize")
        stored = await conversations.load("structured-finalize")

        assert result.output == Answer(answer="PUBLIC-STRUCTURED-FINAL")
        assert [message.content for message in stored] == ["question", final]
        assert [message.content for message in result.messages] == [
            "Return structured output.",
            "question",
            final,
        ]
        assert tool_result in repr(model.requests[1].messages)
        assert act_draft in repr(model.requests[2].messages)
        assert tool_result not in repr(stored)
        assert act_draft not in repr(stored)
        assert tool_result not in repr(result.messages)
        assert act_draft not in repr(result.messages)

    asyncio.run(scenario())


def test_custom_continue_work_response_is_not_public_conversation() -> None:
    class ContinueThenFinishPolicy:
        def __init__(self) -> None:
            self.turn = 0

        async def begin(self, context: Any) -> None:
            del context

        async def decide(
            self,
            context: Any,
            response: Any,
        ) -> ExecutionDecision:
            del context, response
            self.turn += 1
            if self.turn == 1:
                return ExecutionDecision(DecisionKind.CONTINUE)
            return ExecutionDecision(DecisionKind.FINISH)

        async def observe(self, context: Any, results: Any) -> None:
            del context, results

        def should_stop(self, context: Any) -> bool:
            del context
            return False

    async def scenario() -> None:
        work_response = "INTERNAL-CONTINUE-WORK"
        final = "PUBLIC-CUSTOM-FINAL"
        model = ScriptedModel(
            [
                ModelResponse(Message.assistant(work_response)),
                ModelResponse(Message.assistant(final)),
            ]
        )
        conversations = InMemoryConversationStore()
        agent = Agent(
            config=AgentConfig("continue-policy", "Continue once."),
            model=model,
            decision_policy=ContinueThenFinishPolicy(),
            conversation_store=conversations,
        )

        result = await agent.run("question", session_id="continue-policy")
        stored = await conversations.load("continue-policy")

        assert work_response in repr(model.requests[1].messages)
        assert [message.content for message in stored] == ["question", final]
        assert work_response not in repr(result.messages)
        assert work_response not in repr(stored)

    asyncio.run(scenario())


def test_should_stop_promotes_only_the_terminal_continue_response() -> None:
    class StopAfterContinuePolicy:
        stopped = False

        async def begin(self, context: Any) -> None:
            del context

        async def decide(
            self,
            context: Any,
            response: Any,
        ) -> ExecutionDecision:
            del context, response
            self.stopped = True
            return ExecutionDecision(DecisionKind.CONTINUE)

        async def observe(self, context: Any, results: Any) -> None:
            del context, results

        def should_stop(self, context: Any) -> bool:
            del context
            return self.stopped

    async def scenario() -> None:
        final = "PUBLIC-SHOULD-STOP-FINAL"
        conversations = InMemoryConversationStore()
        agent = Agent(
            config=AgentConfig("should-stop", "Stop after one response."),
            model=ScriptedModel([ModelResponse(Message.assistant(final))]),
            decision_policy=StopAfterContinuePolicy(),
            conversation_store=conversations,
        )

        result = await agent.run("question", session_id="should-stop")
        stored = await conversations.load("should-stop")

        assert result.output == final
        assert [message.content for message in stored] == ["question", final]
        assert [message.content for message in result.messages][-1] == final

    asyncio.run(scenario())
