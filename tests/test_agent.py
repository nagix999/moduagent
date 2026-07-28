from __future__ import annotations

import asyncio
import time

from moduagent import (
    Agent,
    AgentConfig,
    EventType,
    Message,
    ModelRequest,
    ModelResponse,
    ToolCall,
    RBACToolAuthorizer,
    RunLimits,
    function_tool,
)


class ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def response(text: str) -> ModelResponse:
    return ModelResponse(Message("assistant", text))


def test_run_limits_keeps_020_positional_argument_order() -> None:
    limits = RunLimits(3, 7, 45.0, True, 2)

    assert limits.max_steps == 3
    assert limits.max_tool_calls == 7
    assert limits.timeout_seconds == 45.0
    assert limits.parallel_tool_calls is True
    assert limits.max_parallel_tools == 2
    assert limits.max_step_attempts == 2
    assert limits.max_replans == 2


def test_agent_returns_final_model_message() -> None:
    async def scenario() -> None:
        model = ScriptedModel([response("안녕하세요")])
        agent = Agent(
            config=AgentConfig("assistant", "친절하게 답한다."),
            model=model,
        )

        result = await agent.run("안녕", session_id="session-1")

        assert result.output == "안녕하세요"
        assert result.finish_reason == "completed"
        assert model.requests[0].messages[0].role == "system"

    asyncio.run(scenario())


def test_agent_executes_function_tool() -> None:
    async def scenario() -> None:
        @function_tool
        def add(a: int, b: int) -> int:
            """두 수를 더한다."""

            return a + b

        call = ToolCall("call-1", "add", {"a": 2, "b": 3})
        model = ScriptedModel(
            [
                ModelResponse(Message("assistant", None, (call,)), (call,)),
                response("결과는 5입니다."),
            ]
        )
        agent = Agent(
            config=AgentConfig("calculator", "계산 도구를 사용한다."),
            model=model,
            tools=[add],
        )

        result = await agent.run("2+3은?", session_id="session-2")

        assert result.output == "결과는 5입니다."
        tool_message = model.requests[1].messages[-1]
        assert tool_message.role == "tool"
        assert '"value": 5' in (tool_message.content or "")

    asyncio.run(scenario())


def test_stream_emits_terminal_result() -> None:
    async def scenario() -> None:
        agent = Agent(
            config=AgentConfig("assistant", "답한다."),
            model=ScriptedModel([response("완료")]),
        )

        events = [event async for event in agent.stream("실행")]

        assert events[0].type is EventType.RUN_STARTED
        assert events[-1].type is EventType.RUN_COMPLETED
        assert events[-1].data["result"].output == "완료"

    asyncio.run(scenario())


def test_conversation_is_reused_by_session() -> None:
    async def scenario() -> None:
        model = ScriptedModel([response("첫 답변"), response("두 번째 답변")])
        agent = Agent(
            config=AgentConfig("assistant", "대화를 기억한다."),
            model=model,
        )

        await agent.run("첫 질문", session_id="same-session")
        await agent.run("두 번째 질문", session_id="same-session")

        second_messages = model.requests[1].messages
        assert [message.content for message in second_messages] == [
            "대화를 기억한다.",
            "첫 질문",
            "첫 답변",
            "두 번째 질문",
        ]

    asyncio.run(scenario())


def test_message_tool_calls_and_usage_are_normalized() -> None:
    async def scenario() -> None:
        @function_tool
        def echo(value: str) -> str:
            return value

        call = ToolCall("call-1", "echo", {"value": "ok"})
        model = ScriptedModel(
            [
                ModelResponse(
                    Message("assistant", None, (call,)),
                    usage={"input_tokens": 2},
                ),
                ModelResponse(
                    Message("assistant", "완료"),
                    usage={"input_tokens": 3},
                ),
            ]
        )
        agent = Agent(
            config=AgentConfig("assistant", "도구를 사용한다."),
            model=model,
            tools=[echo],
        )

        result = await agent.run("실행")

        assert result.output == "완료"
        assert result.usage.input_tokens == 5
        assert model.requests[1].messages[-1].role == "tool"

    asyncio.run(scenario())


def test_rbac_denial_is_returned_to_model() -> None:
    async def scenario() -> None:
        called = False

        @function_tool
        def private_data() -> str:
            nonlocal called
            called = True
            return "secret"

        call = ToolCall("call-1", "private_data", {})
        model = ScriptedModel(
            [
                ModelResponse(Message("assistant", None, (call,)), (call,)),
                response("권한이 없습니다."),
            ]
        )
        agent = Agent(
            config=AgentConfig("assistant", "권한을 지킨다."),
            model=model,
            tools=[private_data],
            tool_authorizer=RBACToolAuthorizer({"admin": {"private_data"}}),
        )

        result = await agent.run("조회", user_context={"roles": ["employee"]})

        assert result.output == "권한이 없습니다."
        assert called is False
        assert "not authorized" in (model.requests[1].messages[-1].content or "")

    asyncio.run(scenario())


def test_timed_out_sync_tool_keeps_valid_history() -> None:
    async def scenario() -> None:
        @function_tool
        def slow() -> str:
            time.sleep(0.2)
            return "late"

        call = ToolCall("call-1", "slow", {})
        model = ScriptedModel(
            [ModelResponse(Message("assistant", None, (call,)), (call,))]
        )
        agent = Agent(
            config=AgentConfig(
                "assistant",
                "도구를 사용한다.",
                RunLimits(max_steps=2, timeout_seconds=0.03),
            ),
            model=model,
            tools=[slow],
        )

        result = await agent.run("실행", session_id="timeout-session")

        assert result.finish_reason == "timeout"
        assert result.messages[-1].role == "tool"
        assert "timed out" in (result.messages[-1].content or "")

    asyncio.run(scenario())


def test_tool_limit_keeps_assistant_tool_message_pair() -> None:
    async def scenario() -> None:
        @function_tool
        def unused() -> str:
            return "unused"

        call = ToolCall("call-1", "unused", {})
        model = ScriptedModel(
            [
                ModelResponse(Message("assistant", None, (call,)), (call,)),
                response("다음 요청은 정상"),
            ]
        )
        agent = Agent(
            config=AgentConfig(
                "assistant",
                "도구를 사용한다.",
                RunLimits(max_tool_calls=0),
            ),
            model=model,
            tools=[unused],
        )

        first = await agent.run("첫 요청", session_id="limited")
        second = await agent.run("다음 요청", session_id="limited")

        assert first.finish_reason == "max_tool_calls"
        assert first.messages[-1].role == "tool"
        assert second.output == "다음 요청은 정상"
        roles = [message.role for message in model.requests[1].messages]
        assert roles == ["system", "user", "assistant", "tool", "user"]

    asyncio.run(scenario())
