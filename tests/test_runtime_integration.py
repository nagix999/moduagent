from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from moduagent import (
    Agent,
    AgentConfig,
    EventType,
    InMemoryCheckpointStore,
    InMemoryConversationStore,
    LLMPlanGenerator,
    Message,
    ModelCapabilities,
    ModelChunk,
    ModelRequest,
    ModelResponse,
    Plan,
    PlanAndExecutePolicy,
    PlanStep,
    PydanticOutputCodec,
    RetryConfig,
    RunLimits,
    TextOutputCodec,
    ToolCall,
    Usage,
    function_tool,
)
from moduagent.memory import (
    RecentTurnsConversationMemoryPolicy,
    SummarizingConversationMemoryPolicy,
    SummaryResult,
    TokenBudget,
    TokenBudgetConversationMemoryPolicy,
)


class ScriptedModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self, items: list[ModelResponse | Exception]) -> None:
        self.items = items
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def stream(self, request: ModelRequest):
        raise AssertionError("stream should not be called")
        yield


class StreamingModel:
    capabilities = ModelCapabilities(streaming=True)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("complete should not be called")

    async def stream(self, request: ModelRequest):
        yield ModelChunk(delta="안녕")
        yield ModelChunk(delta="하세요")
        yield ModelChunk(response=ModelResponse(Message.assistant("안녕하세요")))


def test_runtime_streams_model_deltas_and_terminal_result() -> None:
    async def scenario() -> None:
        agent = Agent(
            config=AgentConfig("streamer", "간단히 답한다."),
            model=StreamingModel(),
        )

        events = [event async for event in agent.stream("인사해줘")]

        deltas = [
            event.data["delta"]
            for event in events
            if event.type is EventType.MODEL_DELTA
        ]
        assert deltas == ["안녕", "하세요"]
        assert events[-1].type is EventType.RUN_COMPLETED
        assert events[-1].data["result"].output == "안녕하세요"

    asyncio.run(scenario())


def test_runtime_retries_model_without_emitted_tokens() -> None:
    async def scenario() -> None:
        model = ScriptedModel(
            [RuntimeError("temporary"), ModelResponse(Message.assistant("복구"))]
        )
        agent = Agent(
            config=AgentConfig(
                "retry",
                "답한다.",
                retry=RetryConfig(max_attempts=2, initial_delay=0),
            ),
            model=model,
        )

        events = [event async for event in agent.stream("실행")]

        assert sum(event.type is EventType.RETRY for event in events) == 1
        assert events[-1].data["result"].output == "복구"
        assert len(model.requests) == 2

    asyncio.run(scenario())


def test_standard_finalization_mode_always_adds_a_text_finalizer() -> None:
    async def scenario() -> None:
        conversations = InMemoryConversationStore()
        model = ScriptedModel(
            [
                ModelResponse(Message.assistant("internal draft")),
                ModelResponse(Message.assistant("public final")),
            ]
        )
        agent = Agent(
            config=AgentConfig(
                "always-finalize",
                "Answer accurately.",
                finalization_mode="always",
            ),
            model=model,
            conversation_store=conversations,
        )

        result = await agent.run("answer this", session_id="always-finalize")

        assert result.output == "public final"
        assert len(model.requests) == 2
        assert model.requests[1].tools == ()
        assert model.requests[1].output_schema is None
        assert [
            message.content for message in await conversations.load("always-finalize")
        ] == ["answer this", "internal draft", "public final"]

    asyncio.run(scenario())


def test_standard_finalization_mode_disabled_skips_structured_staging() -> None:
    async def scenario() -> None:
        @function_tool
        def unused() -> str:
            return "unused"

        model = ScriptedModel(
            [ModelResponse(Message.assistant('{"answer":"direct","confidence":1.0}'))]
        )
        agent = Agent(
            config=AgentConfig(
                "disabled-finalize",
                "Answer accurately.",
                finalization_mode="disabled",
            ),
            model=model,
            tools=[unused],
            output_codec=PydanticOutputCodec(StructuredAnswer),
        )

        result = await agent.run("answer this")

        assert result.output == StructuredAnswer(answer="direct", confidence=1.0)
        assert len(model.requests) == 1
        assert model.requests[0].tools
        assert model.requests[0].output_schema["title"] == "StructuredAnswer"

    asyncio.run(scenario())


def test_recent_turns_policy_limits_model_view_but_preserves_full_history() -> None:
    async def scenario() -> None:
        conversations = InMemoryConversationStore()
        model = ScriptedModel(
            [
                ModelResponse(Message.assistant("answer 1")),
                ModelResponse(Message.assistant("answer 2")),
                ModelResponse(Message.assistant("answer 3")),
            ]
        )
        agent = Agent(
            config=AgentConfig("memory", "Remember the conversation."),
            model=model,
            conversation_store=conversations,
            conversation_memory_policy=RecentTurnsConversationMemoryPolicy(max_turns=1),
        )

        await agent.run("question 1", session_id="recent-session")
        await agent.run("question 2", session_id="recent-session")
        events = [
            event
            async for event in agent.stream("question 3", session_id="recent-session")
        ]

        assert [message.content for message in model.requests[2].messages] == [
            "Remember the conversation.",
            "question 2",
            "answer 2",
            "question 3",
        ]
        assert [message.content for message in events[-1].data["result"].messages] == [
            "Remember the conversation.",
            "question 1",
            "answer 1",
            "question 2",
            "answer 2",
            "question 3",
            "answer 3",
        ]
        assert [
            message.content for message in await conversations.load("recent-session")
        ] == [
            "question 1",
            "answer 1",
            "question 2",
            "answer 2",
            "question 3",
            "answer 3",
        ]
        memory_events = [
            event for event in events if event.type is EventType.MEMORY_COMPACTED
        ]
        assert len(memory_events) == 1
        assert memory_events[0].data["dropped_messages"] == 2

    asyncio.run(scenario())


def test_summarizing_policy_compacts_model_view_and_preserves_raw_history() -> None:
    async def scenario() -> None:
        class RecordingSummarizer:
            def __init__(self) -> None:
                self.calls: list[tuple[Message, ...]] = []

            async def summarize(
                self,
                messages: tuple[Message, ...],
                *,
                previous_summary: str | None = None,
            ) -> SummaryResult:
                assert previous_summary is None
                self.calls.append(messages)
                return SummaryResult(
                    "question 1 was answered with answer 1",
                    Usage(input_tokens=3, output_tokens=2, total_tokens=5),
                )

        conversations = InMemoryConversationStore()
        await conversations.append(
            "summary-session",
            (
                Message.user("question 1"),
                Message.assistant("answer 1"),
                Message.user("question 2"),
                Message.assistant("answer 2"),
            ),
        )
        summarizer = RecordingSummarizer()
        model = ScriptedModel(
            [
                ModelResponse(
                    Message.assistant("answer 3"),
                    usage=Usage(input_tokens=7, output_tokens=4, total_tokens=11),
                )
            ]
        )
        agent = Agent(
            config=AgentConfig("summarizing-memory", "Remember the conversation."),
            model=model,
            conversation_store=conversations,
            conversation_memory_policy=SummarizingConversationMemoryPolicy(
                budget=TokenBudget(100_000),
                summarizer=summarizer,
                max_history_turns=1,
            ),
        )

        result = await agent.run("question 3", session_id="summary-session")

        assert [[message.content for message in call] for call in summarizer.calls] == [
            ["question 1", "answer 1"]
        ]
        assert [message.content for message in model.requests[0].messages] == [
            "Remember the conversation.",
            "Summary of earlier conversation:\nquestion 1 was answered with answer 1",
            "question 2",
            "answer 2",
            "question 3",
        ]
        assert model.requests[0].messages[1].metadata["moduagent.memory"] == "summary"
        assert [message.content for message in result.messages] == [
            "Remember the conversation.",
            "question 1",
            "answer 1",
            "question 2",
            "answer 2",
            "question 3",
            "answer 3",
        ]
        assert [
            message.content for message in await conversations.load("summary-session")
        ] == [
            "question 1",
            "answer 1",
            "question 2",
            "answer 2",
            "question 3",
            "answer 3",
        ]
        assert result.usage == Usage(
            input_tokens=10,
            output_tokens=6,
            total_tokens=16,
        )

    asyncio.run(scenario())


def test_memory_preparation_uses_the_run_deadline() -> None:
    async def scenario() -> None:
        class BlockingMemoryPolicy:
            async def prepare(self, request: Any) -> Any:
                await asyncio.Event().wait()

        model = ScriptedModel([ModelResponse(Message.assistant("unreachable"))])
        agent = Agent(
            config=AgentConfig(
                "memory-timeout",
                "Answer.",
                limits=RunLimits(timeout_seconds=0.02),
            ),
            model=model,
            conversation_memory_policy=BlockingMemoryPolicy(),
        )

        result = await agent.run("question")

        assert result.finish_reason == "timeout"
        assert result.error == "run timed out"
        assert model.requests == []

    asyncio.run(scenario())


def test_memory_overflow_fails_before_the_model_is_called() -> None:
    async def scenario() -> None:
        class FixedCounter:
            async def count_request(self, request: ModelRequest) -> int:
                return 100

        model = ScriptedModel([ModelResponse(Message.assistant("unreachable"))])
        agent = Agent(
            config=AgentConfig("memory-overflow", "Answer."),
            model=model,
            conversation_memory_policy=TokenBudgetConversationMemoryPolicy(
                budget=TokenBudget(10),
                token_counter=FixedCounter(),
            ),
        )

        result = await agent.run("question")

        assert result.finish_reason == "error"
        assert "exceeds the token budget" in (result.error or "")
        assert model.requests == []

    asyncio.run(scenario())


def test_checkpoint_resumes_failed_run() -> None:
    async def scenario() -> None:
        checkpoints = InMemoryCheckpointStore()
        conversations = InMemoryConversationStore()
        model = ScriptedModel(
            [
                RuntimeError("model unavailable"),
                ModelResponse(Message.assistant("완료")),
            ]
        )
        agent = Agent(
            config=AgentConfig("resumable", "답한다."),
            model=model,
            checkpoint_store=checkpoints,
            conversation_store=conversations,
        )

        failed = await agent.run("작업", session_id="resume-session")
        assert failed.error == "model unavailable"
        assert await checkpoints.load(failed.run_id) is not None

        resumed = await agent.resume(failed.run_id, session_id="resume-session")

        assert resumed.run_id == failed.run_id
        assert resumed.output == "완료"
        assert await checkpoints.load(failed.run_id) is None
        assert [
            message.content for message in await conversations.load("resume-session")
        ] == [
            "작업",
            "완료",
        ]

    asyncio.run(scenario())


class StaticPlanGenerator:
    async def create(self, context: Any) -> Plan:
        return Plan(
            [
                PlanStep(
                    step_id="research",
                    objective="자료 조사",
                    completion_criteria=["자료 조사 결과가 있다"],
                ),
                PlanStep(
                    step_id="summarize",
                    objective="조사 결과 정리",
                    completion_criteria=["조사 결과가 정리됐다"],
                    dependencies=["research"],
                ),
            ]
        )

    async def revise(self, context: Any, plan: Plan, feedback: str) -> Plan:
        return plan


def test_plan_and_execute_policy_advances_all_steps() -> None:
    async def scenario() -> None:
        model = ScriptedModel(
            [
                ModelResponse(
                    Message.assistant(
                        '{"step_id":"research","status":"completed",'
                        '"facts":["자료를 찾았다"],'
                        '"completion_evidence":["자료 조사 결과가 있다"]}'
                    )
                ),
                ModelResponse(
                    Message.assistant(
                        '{"step_id":"summarize","status":"completed",'
                        '"facts":["결과를 정리했다"],'
                        '"completion_evidence":["조사 결과가 정리됐다"]}'
                    )
                ),
                ModelResponse(Message.assistant("최종 결과")),
            ]
        )
        agent = Agent(
            config=AgentConfig("planner", "계획에 따라 답한다."),
            model=model,
            decision_policy=PlanAndExecutePolicy(StaticPlanGenerator()),
        )

        result = await agent.run("두 단계로 처리해줘")

        assert result.output == "최종 결과"
        assert len(model.requests) == 3
        assert all(request.tools == () for request in model.requests)
        assert [
            request.output_schema and request.output_schema.get("title")
            for request in model.requests
        ] == ["StepResult", "StepResult", None]
        assert result.metadata["plan"]["current_index"] == 2
        assert result.metadata["plan_usage"]["committed_steps"] == 2

    asyncio.run(scenario())


class StructuredAnswer(BaseModel):
    answer: str
    confidence: float


def test_pydantic_output_codec_returns_domain_object() -> None:
    async def scenario() -> None:
        model = ScriptedModel(
            [ModelResponse(Message.assistant('{"answer":"서울","confidence":0.9}'))]
        )
        agent = Agent(
            config=AgentConfig("structured", "JSON으로 답한다."),
            model=model,
            output_codec=PydanticOutputCodec(StructuredAnswer),
        )

        result = await agent.run("한국의 수도는?")

        assert result.output == StructuredAnswer(answer="서울", confidence=0.9)
        assert model.requests[0].output_schema["title"] == "StructuredAnswer"

    asyncio.run(scenario())


def test_structured_plan_and_tool_requests_use_separate_model_phases() -> None:
    async def scenario() -> None:
        @function_tool
        def add(a: int, b: int) -> int:
            """Add two integers."""

            return a + b

        class PhaseAwareModel:
            capabilities = ModelCapabilities(streaming=False)

            def __init__(self) -> None:
                self.requests: list[ModelRequest] = []
                self.act_count = 0

            async def complete(self, request: ModelRequest) -> ModelResponse:
                self.requests.append(request)

                # PLAN uses its own schema and cannot expose application tools.
                if (
                    request.output_schema is not None
                    and "steps" in request.output_schema.get("properties", {})
                ):
                    assert request.tools == ()
                    return ModelResponse(
                        Message.assistant(
                            '{"steps":[{"step_id":"calculate",'
                            '"objective":"add the numbers",'
                            '"completion_criteria":["the sum is verified"],'
                            '"expected_output":"verified sum",'
                            '"dependencies":[],"allowed_tools":["add"]}]}'
                        )
                    )

                if request.output_schema is not None and (
                    request.output_schema.get("title") == "StepResult"
                ):
                    assert request.tools == ()
                    assert any(
                        message.role.value == "tool" for message in request.messages
                    )
                    return ModelResponse(
                        Message.assistant(
                            '{"step_id":"calculate","status":"completed",'
                            '"facts":["add returned 5"],'
                            '"completion_evidence":["the sum is verified"]}'
                        )
                    )

                # FINALIZE applies only the public output schema after execution.
                if request.output_schema is not None:
                    assert request.output_schema["title"] == "StructuredAnswer"
                    assert request.tools == ()
                    assert all(
                        message.role.value != "tool" for message in request.messages
                    )
                    assert "committed_results" in (request.messages[-1].content or "")
                    return ModelResponse(
                        Message.assistant('{"answer":"5","confidence":1.0}')
                    )

                # ACT_TOOL can call tools, but no output schema constrains it.
                assert request.tools
                self.act_count += 1
                call = ToolCall("call-add", "add", {"a": 2, "b": 3})
                return ModelResponse(Message.assistant(None, (call,)), (call,))

            async def stream(self, request: ModelRequest):
                raise AssertionError("stream should not be called")
                yield

        conversations = InMemoryConversationStore()
        await conversations.append(
            "structured-memory",
            (
                Message.user("obsolete question"),
                Message.assistant("obsolete answer"),
                Message.user("latest question"),
                Message.assistant("latest answer"),
            ),
        )
        model = PhaseAwareModel()
        agent = Agent(
            config=AgentConfig("calculator", "Use add for arithmetic."),
            model=model,
            tools=[add],
            decision_policy=PlanAndExecutePolicy(LLMPlanGenerator(model, max_steps=1)),
            output_codec=PydanticOutputCodec(StructuredAnswer),
            conversation_store=conversations,
            conversation_memory_policy=RecentTurnsConversationMemoryPolicy(max_turns=1),
        )

        result = await agent.run("What is 2 + 3?", session_id="structured-memory")

        assert result.output == StructuredAnswer(answer="5", confidence=1.0)
        assert result.messages[-1].role.value == "assistant"
        assert result.messages[-1].content == '{"answer":"5","confidence":1.0}'
        assert result.metadata["plan"]["current_index"] == 1
        assert len(model.requests) == 4
        plan_request, act_request, step_result_request, finalize_request = (
            model.requests
        )
        assert plan_request.tools == ()
        assert plan_request.output_schema is not None
        assert "steps" in plan_request.output_schema["properties"]
        assert act_request.tools and act_request.output_schema is None
        assert step_result_request.tools == ()
        assert step_result_request.output_schema["title"] == "StepResult"
        assert finalize_request.tools == ()
        assert finalize_request.output_schema["title"] == "StructuredAnswer"
        for request in (act_request, step_result_request, finalize_request):
            contents = [message.content for message in request.messages]
            assert "obsolete question" not in contents
            assert "latest question" not in contents
        assert all(
            not (request.tools and request.output_schema is not None)
            for request in model.requests
        )

    asyncio.run(scenario())


def test_text_plan_and_tool_path_extracts_step_result_then_finalizes() -> None:
    async def scenario() -> None:
        calls: list[tuple[int, int]] = []

        class OneStepPlanGenerator:
            async def create(self, context: Any) -> Plan:
                return Plan(
                    [
                        PlanStep(
                            step_id="calculate",
                            objective="add the numbers",
                            completion_criteria=["the sum is verified"],
                            allowed_tools=["add"],
                        )
                    ]
                )

            async def revise(self, context: Any, plan: Plan, feedback: str) -> Plan:
                return plan

        @function_tool
        def add(a: int, b: int) -> int:
            calls.append((a, b))
            return a + b

        call = ToolCall("call-add", "add", {"a": 2, "b": 3})
        model = ScriptedModel(
            [
                ModelResponse(Message.assistant(None, (call,)), (call,)),
                ModelResponse(
                    Message.assistant(
                        '{"step_id":"calculate","status":"completed",'
                        '"facts":["add returned 5"],'
                        '"completion_evidence":["the sum is verified"]}'
                    )
                ),
                ModelResponse(Message.assistant("The answer is 5.")),
            ]
        )
        agent = Agent(
            config=AgentConfig("calculator", "Use add for arithmetic."),
            model=model,
            tools=[add],
            decision_policy=PlanAndExecutePolicy(OneStepPlanGenerator()),
            output_codec=TextOutputCodec(),
        )

        result = await agent.run("What is 2 + 3?")

        assert result.output == "The answer is 5."
        assert calls == [(2, 3)]
        assert len(model.requests) == 3
        act_request, step_result_request, finalize_request = model.requests
        assert act_request.tools and act_request.output_schema is None
        assert step_result_request.tools == ()
        assert step_result_request.output_schema["title"] == "StepResult"
        assert finalize_request.tools == ()
        assert finalize_request.output_schema is None
        assert all(
            not (request.tools and request.output_schema is not None)
            for request in model.requests
        )

    asyncio.run(scenario())


def test_structured_finalization_resume_does_not_repeat_act() -> None:
    async def scenario() -> None:
        @function_tool
        def add(a: int, b: int) -> int:
            return a + b

        checkpoints = InMemoryCheckpointStore()
        conversations = InMemoryConversationStore()
        model = ScriptedModel(
            [
                ModelResponse(Message.assistant("Draft answer: 5")),
                RuntimeError("finalizer unavailable"),
                ModelResponse(Message.assistant('{"answer":"5","confidence":1.0}')),
            ]
        )
        agent = Agent(
            config=AgentConfig("calculator", "Use add when needed."),
            model=model,
            tools=[add],
            output_codec=PydanticOutputCodec(StructuredAnswer),
            checkpoint_store=checkpoints,
            conversation_store=conversations,
        )

        failed = await agent.run("What is 2 + 3?", session_id="structured-resume")

        assert failed.error == "finalizer unavailable"
        assert len(model.requests) == 2
        assert model.requests[0].tools and model.requests[0].output_schema is None
        assert model.requests[1].tools == ()
        assert model.requests[1].output_schema is not None
        assert await checkpoints.load(failed.run_id) is not None

        resumed = await agent.resume(failed.run_id, session_id="structured-resume")

        assert resumed.output == StructuredAnswer(answer="5", confidence=1.0)
        assert len(model.requests) == 3
        assert sum(bool(request.tools) for request in model.requests) == 1
        assert model.requests[2].tools == ()
        assert model.requests[2].output_schema is not None
        assert await checkpoints.load(failed.run_id) is None
        assert [
            message.content for message in await conversations.load("structured-resume")
        ] == [
            "What is 2 + 3?",
            "Draft answer: 5",
            '{"answer":"5","confidence":1.0}',
        ]

    asyncio.run(scenario())


def test_structured_finalizer_never_executes_returned_tool_calls() -> None:
    async def scenario() -> None:
        executions: list[tuple[int, int]] = []

        @function_tool
        def add(a: int, b: int) -> int:
            executions.append((a, b))
            return a + b

        unexpected = ToolCall("unexpected", "add", {"a": 2, "b": 3})
        model = ScriptedModel(
            [
                ModelResponse(Message.assistant("Draft answer: 5")),
                ModelResponse(
                    Message.assistant(None, (unexpected,)),
                    (unexpected,),
                ),
            ]
        )
        agent = Agent(
            config=AgentConfig("calculator", "Use add when needed."),
            model=model,
            tools=[add],
            output_codec=PydanticOutputCodec(StructuredAnswer),
        )

        result = await agent.run("What is 2 + 3?")

        assert result.finish_reason == "error"
        assert result.error == "finalization returned tool calls"
        assert executions == []

    asyncio.run(scenario())


def test_structured_tool_stream_labels_act_and_finalize_deltas() -> None:
    async def scenario() -> None:
        @function_tool
        def add(a: int, b: int) -> int:
            return a + b

        class StructuredStreamingModel:
            capabilities = ModelCapabilities(streaming=True)

            def __init__(self) -> None:
                self.requests: list[ModelRequest] = []

            async def complete(self, request: ModelRequest) -> ModelResponse:
                raise AssertionError("complete should not be called")

            async def stream(self, request: ModelRequest):
                self.requests.append(request)
                if request.output_schema is None:
                    yield ModelChunk(delta="Draft answer: 5")
                    yield ModelChunk(
                        response=ModelResponse(Message.assistant("Draft answer: 5"))
                    )
                    return
                yield ModelChunk(delta='{"answer":"5","confidence":1.0}')
                yield ModelChunk(
                    response=ModelResponse(
                        Message.assistant('{"answer":"5","confidence":1.0}')
                    )
                )

        model = StructuredStreamingModel()
        agent = Agent(
            config=AgentConfig("calculator", "Use add when needed."),
            model=model,
            tools=[add],
            output_codec=PydanticOutputCodec(StructuredAnswer),
        )

        events = [event async for event in agent.stream("What is 2 + 3?")]

        deltas = [
            (event.data["phase"], event.data["delta"])
            for event in events
            if event.type is EventType.MODEL_DELTA
        ]
        assert deltas == [
            ("act", "Draft answer: 5"),
            ("finalize", '{"answer":"5","confidence":1.0}'),
        ]
        assert events[-1].data["result"].output == StructuredAnswer(
            answer="5", confidence=1.0
        )
        assert len(model.requests) == 2
        assert model.requests[0].tools and model.requests[0].output_schema is None
        assert model.requests[1].tools == ()
        assert model.requests[1].output_schema is not None

    asyncio.run(scenario())


def test_memory_policy_prepares_act_and_finalize_without_mutating_requests() -> None:
    async def scenario() -> None:
        class CompactingMemoryPolicy:
            def __init__(self) -> None:
                self.requests: list[Any] = []

            async def prepare(self, request: Any) -> Any:
                self.requests.append(request)
                original = request.model_request.messages
                selected = (original[0], *original[request.protected_from :])
                return SimpleNamespace(
                    messages=selected,
                    usage=Usage(input_tokens=1, total_tokens=1),
                    original_tokens=len(original) * 10,
                    selected_tokens=len(selected) * 10,
                    summarized_messages=0,
                    dropped_messages=len(original) - len(selected),
                    metadata={"budget_tokens": 100},
                )

        @function_tool
        def unused_tool() -> str:
            return "unused"

        conversations = InMemoryConversationStore()
        await conversations.append(
            "memory-session",
            (Message.user("old question"), Message.assistant("old answer")),
        )
        memory_policy = CompactingMemoryPolicy()
        model = ScriptedModel(
            [
                ModelResponse(Message.assistant("Draft answer")),
                ModelResponse(Message.assistant('{"answer":"final","confidence":1.0}')),
            ]
        )
        agent = Agent(
            config=AgentConfig(
                "memory-agent",
                "Answer briefly.",
                model_options={"temperature": 0.1},
            ),
            model=model,
            tools=[unused_tool],
            output_codec=PydanticOutputCodec(StructuredAnswer),
            conversation_store=conversations,
            conversation_memory_policy=memory_policy,
        )

        events = [
            event
            async for event in agent.stream("new question", session_id="memory-session")
        ]

        assert [request.phase.value for request in memory_policy.requests] == [
            "act",
            "finalize",
        ]
        assert [request.protected_from for request in memory_policy.requests] == [3, 3]
        assert [message.content for message in model.requests[0].messages] == [
            "Answer briefly.",
            "new question",
        ]
        assert [message.content for message in model.requests[1].messages[:-1]] == [
            "Answer briefly.",
            "new question",
            "Draft answer",
        ]
        for original, prepared in zip(memory_policy.requests, model.requests):
            model_request = original.model_request
            assert prepared.tools == model_request.tools
            assert prepared.output_schema == model_request.output_schema
            assert prepared.options == model_request.options
            assert prepared.provider_options == model_request.provider_options

        memory_events = [
            event for event in events if event.type is EventType.MEMORY_COMPACTED
        ]
        assert [event.data["phase"] for event in memory_events] == [
            "act",
            "finalize",
        ]
        assert all(event.data["dropped_messages"] == 2 for event in memory_events)
        assert all(event.data["budget_tokens"] == 100 for event in memory_events)
        model_starts = [
            index
            for index, event in enumerate(events)
            if event.type is EventType.MODEL_STARTED
        ]
        memory_indices = [
            index
            for index, event in enumerate(events)
            if event.type is EventType.MEMORY_COMPACTED
        ]
        assert all(
            memory < started for memory, started in zip(memory_indices, model_starts)
        )

        result = events[-1].data["result"]
        assert result.output == StructuredAnswer(answer="final", confidence=1.0)
        assert result.usage.total_tokens == 2
        assert [message.content for message in result.messages] == [
            "Answer briefly.",
            "old question",
            "old answer",
            "new question",
            "Draft answer",
            '{"answer":"final","confidence":1.0}',
        ]

    asyncio.run(scenario())


class ConcurrentModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.active = 0
        self.max_active = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.03)
            latest_user = next(
                message.content
                for message in reversed(request.messages)
                if message.role.value == "user"
            )
            return ModelResponse(Message.assistant(f"answer:{latest_user}"))
        finally:
            self.active -= 1

    async def stream(self, request: ModelRequest):
        raise AssertionError("stream should not be called")
        yield


def test_same_session_is_serialized_but_different_sessions_can_overlap() -> None:
    async def scenario() -> None:
        model = ConcurrentModel()
        agent = Agent(
            config=AgentConfig("concurrent", "답한다."),
            model=model,
        )

        first = asyncio.create_task(agent.run("first", session_id="same"))
        await asyncio.sleep(0.005)
        second = asyncio.create_task(agent.run("second", session_id="same"))
        first_result, second_result = await asyncio.gather(first, second)

        assert first_result.output == "answer:first"
        assert second_result.output == "answer:second"
        assert model.max_active == 1
        assert [message.content for message in model.requests[1].messages] == [
            "답한다.",
            "first",
            "answer:first",
            "second",
        ]

        model.max_active = 0
        await asyncio.gather(
            agent.run("a", session_id="session-a"),
            agent.run("b", session_id="session-b"),
        )
        assert model.max_active == 2

    asyncio.run(scenario())


class BlockingStreamingModel:
    capabilities = ModelCapabilities(streaming=True)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("complete should not be called")

    async def stream(self, request: ModelRequest):
        await asyncio.Event().wait()
        yield ModelChunk(delta="unreachable")


def test_stream_cancellation_keeps_checkpoint_for_resume() -> None:
    async def scenario() -> None:
        checkpoints = InMemoryCheckpointStore()
        agent = Agent(
            config=AgentConfig("cancel", "답한다."),
            model=BlockingStreamingModel(),
            checkpoint_store=checkpoints,
        )
        stream = agent.stream("long task", session_id="cancel-session")

        started = await anext(stream)
        assert started.type is EventType.RUN_STARTED
        model_started = await anext(stream)
        assert model_started.type is EventType.MODEL_STARTED

        pending = asyncio.create_task(anext(stream))
        await asyncio.sleep(0.01)
        pending.cancel()
        try:
            await pending
        except asyncio.CancelledError:
            pass

        checkpoint = await checkpoints.load(started.run_id)
        assert checkpoint is not None
        assert checkpoint.session_id == "cancel-session"

    asyncio.run(scenario())
