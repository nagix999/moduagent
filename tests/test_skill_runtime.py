from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

from pydantic import BaseModel

from moduagent import (
    Agent,
    AgentConfig,
    EventType,
    FinishReason,
    InMemoryCheckpointStore,
    InMemoryConversationStore,
    InMemorySkillSource,
    LLMPlanGenerator,
    Message,
    ModelCapabilities,
    ModelResponse,
    ModelSkillSelector,
    PlanAndExecutePolicy,
    PydanticOutputCodec,
    RBACToolAuthorizer,
    RecentTurnsConversationMemoryPolicy,
    RunLimits,
    SkillLimits,
    SkillRegistry,
    SkillSelectionResult,
    StandardDecisionPolicy,
    ToolCall,
    ToolErrorType,
    Usage,
    function_tool,
)


def _skill(
    name: str,
    body: str,
    *,
    description: str = "Use this skill for the matching task.",
    allowed_tools: tuple[str, ...] = (),
) -> str:
    allowed = f"allowed-tools: {' '.join(allowed_tools)}\n" if allowed_tools else ""
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "metadata:\n"
        '  version: "1.0.0"\n'
        f"{allowed}"
        "---\n\n"
        f"{body}\n"
    )


class RecordingModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        return self.responses.pop(0)

    async def stream(self, request):
        raise AssertionError("streaming is not expected")
        yield


def test_explicit_skill_is_prompt_only_and_not_persisted() -> None:
    async def scenario() -> None:
        registry = SkillRegistry.from_sources(
            InMemorySkillSource(
                {
                    "brief-answer": _skill(
                        "brief-answer",
                        "Always answer in one short sentence.",
                    )
                }
            )
        )
        model = RecordingModel(
            [
                ModelResponse(
                    Message.assistant("짧은 답변입니다."),
                    usage=Usage(10, 3, 13),
                )
            ]
        )
        conversations = InMemoryConversationStore()
        agent = Agent(
            config=AgentConfig("skilled", "Answer accurately."),
            model=model,
            skill_registry=registry,
            conversation_store=conversations,
        )

        events = [
            event
            async for event in agent.stream(
                "간단히 답해줘",
                session_id="skill-session",
                skills=["brief-answer"],
            )
        ]
        result = events[-1].data["result"]

        request_contents = [
            message.content or "" for message in model.requests[0].messages
        ]
        assert any(
            "Always answer in one short sentence." in content
            for content in request_contents
        )
        assert all(
            "Always answer in one short sentence." not in (message.content or "")
            for message in result.messages
        )
        assert [message.content for message in result.messages] == [
            "Answer accurately.",
            "간단히 답해줘",
            "짧은 답변입니다.",
        ]
        assert [
            message.content for message in await conversations.load("skill-session")
        ] == ["간단히 답해줘", "짧은 답변입니다."]
        assert result.metadata["skills"][0]["name"] == "brief-answer"
        event_types = [event.type for event in events]
        assert EventType.SKILLS_DISCOVERED in event_types
        assert EventType.SKILL_SELECTED in event_types
        assert EventType.SKILL_ACTIVATED in event_types

    asyncio.run(scenario())


class StructuredAnswer(BaseModel):
    answer: str


def test_skill_applies_to_plan_act_and_finalize_with_tool() -> None:
    async def scenario() -> None:
        @function_tool
        def add(a: int, b: int) -> int:
            """Add two integers."""

            return a + b

        registry = SkillRegistry.from_sources(
            InMemorySkillSource(
                {
                    "addition-workflow": _skill(
                        "addition-workflow",
                        "Use add and report the verified result.",
                        allowed_tools=("add",),
                    )
                }
            )
        )

        class PhaseModel:
            capabilities = ModelCapabilities(streaming=False)

            def __init__(self) -> None:
                self.requests = []
                self.act_calls = 0

            async def complete(self, request):
                self.requests.append(request)
                contents = "\n".join(
                    message.content or "" for message in request.messages
                )
                assert "Use add and report the verified result." in contents
                if request.output_schema and "steps" in request.output_schema.get(
                    "properties", {}
                ):
                    assert request.tools == ()
                    return ModelResponse(
                        Message.assistant(
                            '{"steps":[{"description":"calculate once"}]}'
                        )
                    )
                if request.output_schema is not None:
                    assert request.tools == ()
                    return ModelResponse(Message.assistant('{"answer":"5"}'))
                assert [schema.name for schema in request.tools] == ["add"]
                self.act_calls += 1
                if self.act_calls == 1:
                    call = ToolCall("add-1", "add", {"a": 2, "b": 3})
                    return ModelResponse(Message.assistant(None, (call,)), (call,))
                return ModelResponse(Message.assistant("Verified result is 5."))

            async def stream(self, request):
                raise AssertionError("streaming is not expected")
                yield

        model = PhaseModel()
        agent = Agent(
            config=AgentConfig("calculator", "Use verified calculations."),
            model=model,
            tools=[add],
            skill_registry=registry,
            decision_policy=PlanAndExecutePolicy(LLMPlanGenerator(model, max_steps=1)),
            output_codec=PydanticOutputCodec(StructuredAnswer),
        )

        result = await agent.run(
            "2 + 3은?",
            skills=["addition-workflow"],
        )

        assert result.output == StructuredAnswer(answer="5")
        assert len(model.requests) == 4
        assert all(
            not (request.tools and request.output_schema is not None)
            for request in model.requests
        )

    asyncio.run(scenario())


def test_model_skill_selection_is_a_separate_schema_only_phase() -> None:
    async def scenario() -> None:
        registry = SkillRegistry.from_sources(
            InMemorySkillSource(
                {
                    "weather-guide": _skill(
                        "weather-guide",
                        "Mention that forecasts can change.",
                        description="Use for weather and forecast questions.",
                    )
                }
            )
        )

        class SelectionModel:
            capabilities = ModelCapabilities(streaming=False)

            def __init__(self) -> None:
                self.requests = []

            async def complete(self, request):
                self.requests.append(request)
                schema = request.output_schema or {}
                if "skills" in schema.get("properties", {}):
                    assert request.tools == ()
                    return ModelResponse(
                        Message.assistant('{"skills":["weather-guide"]}'),
                        usage=Usage(5, 2, 7),
                    )
                contents = "\n".join(
                    message.content or "" for message in request.messages
                )
                assert "Mention that forecasts can change." in contents
                return ModelResponse(
                    Message.assistant("예보는 바뀔 수 있습니다."),
                    usage=Usage(8, 3, 11),
                )

            async def stream(self, request):
                raise AssertionError("streaming is not expected")
                yield

        model = SelectionModel()
        agent = Agent(
            config=AgentConfig("weather", "Answer weather questions."),
            model=model,
            skill_registry=registry,
            skill_selector=ModelSkillSelector(model),
        )

        result = await agent.run("내일 비가 와?", skill_mode="auto")

        assert result.output == "예보는 바뀔 수 있습니다."
        assert result.usage.total_tokens == 18
        assert len(model.requests) == 2
        assert model.requests[0].tools == ()
        assert model.requests[0].output_schema is not None
        assert model.requests[1].output_schema is None

    asyncio.run(scenario())


def test_reference_read_is_bounded_ephemeral_and_separately_counted(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        skill_dir = tmp_path / "policy-guide"
        references = skill_dir / "references"
        references.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            _skill(
                "policy-guide",
                "Read references/policy.md before answering policy questions.",
            ),
            encoding="utf-8",
        )
        (references / "policy.md").write_text(
            "Annual leave requires manager approval.",
            encoding="utf-8",
        )
        registry = SkillRegistry.from_paths(tmp_path)
        call = ToolCall(
            "read-1",
            "moduagent_skill_read",
            {
                "skill_name": "policy-guide",
                "path": "references/policy.md",
            },
        )
        model = RecordingModel(
            [
                ModelResponse(Message.assistant(None, (call,)), (call,)),
                ModelResponse(Message.assistant("관리자 승인이 필요합니다.")),
            ]
        )
        conversations = InMemoryConversationStore()
        agent = Agent(
            config=AgentConfig("policy", "Use the policy source."),
            model=model,
            skill_registry=registry,
            conversation_store=conversations,
        )

        events = [
            event
            async for event in agent.stream(
                "연차 승인 조건은?",
                session_id="resource-session",
                skills=["policy-guide"],
            )
        ]
        result = events[-1].data["result"]

        assert result.output == "관리자 승인이 필요합니다."
        assert len(model.requests) == 2
        assert {schema.name for schema in model.requests[0].tools} == {
            "moduagent_skill_read",
            "moduagent_skill_search",
        }
        assert "Annual leave requires manager approval." in (
            model.requests[1].messages[-1].content or ""
        )
        assert result.messages == (
            Message.system("Use the policy source."),
            Message.user("연차 승인 조건은?"),
            Message.assistant("관리자 승인이 필요합니다."),
        )
        assert [
            message.content for message in await conversations.load("resource-session")
        ] == ["연차 승인 조건은?", "관리자 승인이 필요합니다."]
        resource_events = [
            event for event in events if event.type is EventType.SKILL_RESOURCE_READ
        ]
        assert len(resource_events) == 1
        assert resource_events[0].data["returned_bytes"] > 0
        tool_completed = [
            event for event in events if event.type is EventType.TOOL_COMPLETED
        ][0]
        assert "result" not in tool_completed.data

    asyncio.run(scenario())


def test_resource_token_budget_failure_is_observed_by_policy(tmp_path: Path) -> None:
    async def scenario() -> None:
        skill_dir = tmp_path / "bounded-reference"
        references = skill_dir / "references"
        references.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            _skill(
                "bounded-reference",
                "Read references/policy.md before answering.",
            ),
            encoding="utf-8",
        )
        (references / "policy.md").write_text(
            "This resource is deliberately larger than one estimated token.",
            encoding="utf-8",
        )

        call = ToolCall(
            "read-over-budget",
            "moduagent_skill_read",
            {
                "skill_name": "bounded-reference",
                "path": "references/policy.md",
            },
        )
        model = RecordingModel(
            [
                ModelResponse(Message.assistant(None, (call,)), (call,)),
                ModelResponse(Message.assistant("자료 한도를 초과했습니다.")),
            ]
        )

        class RecordingPolicy(StandardDecisionPolicy):
            def __init__(self) -> None:
                self.observed = []

            async def observe(self, context, results) -> None:
                self.observed.append(tuple(results))
                await super().observe(context, results)

        policy = RecordingPolicy()
        agent = Agent(
            config=AgentConfig("bounded-resource", "Respect resource limits."),
            model=model,
            decision_policy=policy,
            skill_registry=SkillRegistry.from_paths(tmp_path),
            skill_limits=SkillLimits(max_resource_tokens=1),
        )

        result = await agent.run(
            "정책을 확인해줘",
            skills=["bounded-reference"],
        )

        assert result.output == "자료 한도를 초과했습니다."
        assert len(policy.observed) == 1
        observed = policy.observed[0][0]
        assert observed.success is False
        assert observed.error is not None
        assert observed.error.type is ToolErrorType.RESULT_TOO_LARGE
        assert observed.error.message == "Skill resource token budget exceeded"

        model_payload = json.loads(model.requests[1].messages[-1].content or "")
        assert model_payload["success"] is False
        assert model_payload["error"]["type"] == "result_too_large"
        assert (
            model_payload["error"]["message"] == "Skill resource token budget exceeded"
        )

    asyncio.run(scenario())


def test_checkpoint_resume_rejects_changed_skill_digest(tmp_path: Path) -> None:
    async def scenario() -> None:
        skill_dir = tmp_path / "stable-workflow"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            _skill("stable-workflow", "Use revision one."),
            encoding="utf-8",
        )
        registry = SkillRegistry.from_paths(tmp_path)

        class FailingModel:
            capabilities = ModelCapabilities(streaming=False)

            async def complete(self, request):
                raise RuntimeError("model unavailable")

            async def stream(self, request):
                raise AssertionError("streaming is not expected")
                yield

        checkpoints = InMemoryCheckpointStore()
        agent = Agent(
            config=AgentConfig("resumable-skill", "Follow the workflow."),
            model=FailingModel(),
            skill_registry=registry,
            checkpoint_store=checkpoints,
        )

        failed = await agent.run(
            "실행해줘",
            session_id="skill-resume",
            skills=["stable-workflow"],
        )
        assert failed.error == "model unavailable"
        assert await checkpoints.load(failed.run_id) is not None

        skill_file.write_text(
            _skill("stable-workflow", "Use revision two."),
            encoding="utf-8",
        )
        resumed = await agent.resume(failed.run_id, session_id="skill-resume")

        assert resumed.finish_reason == "error"
        assert "skill content changed" in (resumed.error or "")

    asyncio.run(scenario())


def test_checkpoint_resume_retries_incomplete_auto_selection() -> None:
    async def scenario() -> None:
        registry = SkillRegistry.from_sources(
            InMemorySkillSource(
                {
                    "retry-guide": _skill(
                        "retry-guide",
                        "Use the retry guide after selection succeeds.",
                    )
                }
            )
        )

        class FlakySelector:
            def __init__(self) -> None:
                self.calls = 0

            async def select(self, request):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("selector temporarily unavailable")
                return SkillSelectionResult(
                    names=("retry-guide",),
                    selected_by={"retry-guide": "model"},
                )

        selector = FlakySelector()
        checkpoints = InMemoryCheckpointStore()
        model = RecordingModel([ModelResponse(Message.assistant("완료"))])
        agent = Agent(
            config=AgentConfig("retry-selection", "Answer accurately."),
            model=model,
            skill_registry=registry,
            skill_selector=selector,
            checkpoint_store=checkpoints,
        )

        failed = await agent.run(
            "처리해줘",
            session_id="selection-resume",
            skill_mode="auto",
        )
        assert failed.error == "selector temporarily unavailable"

        resumed = await agent.resume(failed.run_id, session_id="selection-resume")

        assert resumed.output == "완료"
        assert selector.calls == 2
        assert resumed.metadata["skills"][0]["name"] == "retry-guide"

    asyncio.run(scenario())


def test_filesystem_activation_and_restore_are_off_loop_and_reuse_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    skill_dir = tmp_path / "cached-reference"
    references = skill_dir / "references"
    references.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        _skill(
            "cached-reference",
            "Use references/policy.md when needed.",
        ),
        encoding="utf-8",
    )
    (references / "policy.md").write_text("Policy text.", encoding="utf-8")
    registry = SkillRegistry.from_paths(tmp_path)
    original_load = registry.load
    load_threads: list[tuple[int, bool]] = []

    def tracked_load(skill):
        load_threads.append((threading.get_ident(), threading.current_thread().daemon))
        return original_load(skill)

    monkeypatch.setattr(registry, "load", tracked_load)

    class ResumeModel:
        capabilities = ModelCapabilities(streaming=False)

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("pause for resume")
            return ModelResponse(Message.assistant("복구 완료"))

        async def stream(self, request):
            raise AssertionError("streaming is not expected")
            yield

    async def scenario() -> None:
        event_loop_thread = threading.get_ident()
        checkpoints = InMemoryCheckpointStore()
        agent = Agent(
            config=AgentConfig("cached-skill", "Use the selected Skill."),
            model=ResumeModel(),
            skill_registry=registry,
            checkpoint_store=checkpoints,
        )

        failed = await agent.run(
            "실행해줘",
            session_id="async-skill-load",
            skills=["cached-reference"],
        )
        assert failed.error == "pause for resume"
        # Activation loads once. has_resources() and supports_resource_search()
        # must use the activated artifact instead of scanning twice more.
        assert len(load_threads) == 1

        resumed = await agent.resume(
            failed.run_id,
            session_id="async-skill-load",
        )

        assert resumed.output == "복구 완료"
        # Async restore validates once, then both schema checks reuse its artifact.
        assert len(load_threads) == 2
        assert all(
            thread_id != event_loop_thread and daemon
            for thread_id, daemon in load_threads
        )

    asyncio.run(scenario())


def test_filesystem_activation_scan_is_cancellable_and_does_not_block_loop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    skill_dir = tmp_path / "slow-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        _skill("slow-skill", "Follow the slow Skill."),
        encoding="utf-8",
    )
    registry = SkillRegistry.from_paths(tmp_path)
    original_load = registry.load
    scan_started = threading.Event()
    scan_done = threading.Event()
    load_threads: list[tuple[int, bool]] = []

    def slow_load(skill):
        load_threads.append((threading.get_ident(), threading.current_thread().daemon))
        scan_started.set()
        try:
            time.sleep(0.3)
            return original_load(skill)
        finally:
            scan_done.set()

    monkeypatch.setattr(registry, "load", slow_load)

    async def scenario() -> tuple[object, float, int, int]:
        event_loop_thread = threading.get_ident()
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0.001)

        ticker_task = asyncio.create_task(ticker())
        agent = Agent(
            config=AgentConfig(
                "timed-skill",
                "Respect the run timeout.",
                limits=RunLimits(timeout_seconds=0.05),
            ),
            model=RecordingModel([ModelResponse(Message.assistant("unused"))]),
            skill_registry=registry,
        )
        started_at = asyncio.get_running_loop().time()
        result = await agent.run("실행해줘", skills=["slow-skill"])
        elapsed = asyncio.get_running_loop().time() - started_at
        ticker_task.cancel()
        try:
            await ticker_task
        except asyncio.CancelledError:
            pass
        return result, elapsed, ticks, event_loop_thread

    result, elapsed, ticks, event_loop_thread = asyncio.run(scenario())

    assert scan_started.is_set()
    assert result.finish_reason is FinishReason.TIMEOUT
    assert result.error == "run timed out"
    assert elapsed < 0.2
    assert ticks >= 2
    assert len(load_threads) == 1
    assert load_threads[0][1] is True
    assert load_threads[0][0] != event_loop_thread
    assert scan_done.wait(timeout=1.0)


def test_mixed_resource_and_business_rejection_resumes_complete_protocol(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        executions: list[str] = []

        @function_tool
        def lookup_record(record_id: str) -> str:
            """Look up one business record."""

            executions.append(record_id)
            return "record"

        skill_dir = tmp_path / "mixed-workflow"
        references = skill_dir / "references"
        references.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            _skill(
                "mixed-workflow",
                "Read the policy and then look up the record.",
                allowed_tools=("lookup_record",),
            ),
            encoding="utf-8",
        )
        (references / "policy.md").write_text("Policy.", encoding="utf-8")
        resource_call = ToolCall(
            "resource-1",
            "moduagent_skill_read",
            {
                "skill_name": "mixed-workflow",
                "path": "references/policy.md",
            },
        )
        business_call = ToolCall(
            "business-1",
            "lookup_record",
            {"record_id": "42"},
        )
        model = RecordingModel(
            [
                ModelResponse(
                    Message.assistant(None, (resource_call, business_call)),
                    (resource_call, business_call),
                ),
                ModelResponse(Message.assistant("분리 호출로 복구했습니다.")),
            ]
        )
        checkpoints = InMemoryCheckpointStore()
        agent = Agent(
            config=AgentConfig("mixed-calls", "Use valid Tool sequences."),
            model=model,
            tools=[lookup_record],
            skill_registry=SkillRegistry.from_paths(tmp_path),
            checkpoint_store=checkpoints,
            conversation_memory_policy=RecentTurnsConversationMemoryPolicy(max_turns=1),
        )

        failed = await agent.run(
            "자료와 레코드를 확인해줘",
            session_id="mixed-resume",
            skills=["mixed-workflow"],
        )

        assert failed.finish_reason is FinishReason.ERROR
        assert "cannot mix" in (failed.error or "")
        assert executions == []
        checkpoint = await checkpoints.load(failed.run_id)
        assert checkpoint is not None
        assert [
            message.tool_call_id
            for message in checkpoint.messages
            if message.role.value == "tool"
        ] == ["resource-1", "business-1"]

        resumed = await agent.resume(failed.run_id, session_id="mixed-resume")

        assert resumed.finish_reason is FinishReason.COMPLETED
        assert resumed.output == "분리 호출로 복구했습니다."
        assert resumed.error is None

    asyncio.run(scenario())


def test_resource_read_quota_rejection_resumes_complete_protocol() -> None:
    async def scenario() -> None:
        registry = SkillRegistry.from_sources(
            InMemorySkillSource(
                {
                    "quota-workflow": {
                        "SKILL.md": _skill(
                            "quota-workflow",
                            "Read references/policy.md before answering.",
                        ),
                        "references/policy.md": "Policy.",
                    }
                }
            )
        )
        resource_call = ToolCall(
            "quota-read-1",
            "moduagent_skill_read",
            {
                "skill_name": "quota-workflow",
                "path": "references/policy.md",
            },
        )
        model = RecordingModel(
            [
                ModelResponse(
                    Message.assistant(None, (resource_call,)),
                    (resource_call,),
                ),
                ModelResponse(Message.assistant("읽기 없이 복구했습니다.")),
            ]
        )
        checkpoints = InMemoryCheckpointStore()
        agent = Agent(
            config=AgentConfig("quota-calls", "Respect resource quotas."),
            model=model,
            skill_registry=registry,
            skill_limits=SkillLimits(max_resource_reads=0),
            checkpoint_store=checkpoints,
            conversation_memory_policy=RecentTurnsConversationMemoryPolicy(max_turns=1),
        )

        failed = await agent.run(
            "정책을 확인해줘",
            session_id="quota-resume",
            skills=["quota-workflow"],
        )

        assert failed.finish_reason is FinishReason.ERROR
        assert failed.error == "Skill resource read limit exceeded"
        checkpoint = await checkpoints.load(failed.run_id)
        assert checkpoint is not None
        assert checkpoint.skill_state.resource_reads == 0
        assert [
            message.tool_call_id
            for message in checkpoint.messages
            if message.role.value == "tool"
        ] == ["quota-read-1"]

        resumed = await agent.resume(failed.run_id, session_id="quota-resume")

        assert resumed.finish_reason is FinishReason.COMPLETED
        assert resumed.output == "읽기 없이 복구했습니다."
        assert resumed.error is None

    asyncio.run(scenario())


def test_restore_rejects_checkpoint_resource_reads_above_current_limit() -> None:
    async def scenario() -> None:
        registry = SkillRegistry.from_sources(
            InMemorySkillSource(
                {
                    "restore-quota": {
                        "SKILL.md": _skill(
                            "restore-quota",
                            "Read references/policy.md before answering.",
                        ),
                        "references/policy.md": "Policy.",
                    }
                }
            )
        )
        resource_call = ToolCall(
            "restore-read-1",
            "moduagent_skill_read",
            {
                "skill_name": "restore-quota",
                "path": "references/policy.md",
            },
        )

        class FailingAfterReadModel:
            capabilities = ModelCapabilities(streaming=False)

            def __init__(self) -> None:
                self.calls = 0

            async def complete(self, request):
                self.calls += 1
                if self.calls == 1:
                    return ModelResponse(
                        Message.assistant(None, (resource_call,)),
                        (resource_call,),
                    )
                raise RuntimeError("pause after resource read")

            async def stream(self, request):
                raise AssertionError("streaming is not expected")
                yield

        checkpoints = InMemoryCheckpointStore()
        first_agent = Agent(
            config=AgentConfig("restore-quota", "Use the reference."),
            model=FailingAfterReadModel(),
            skill_registry=registry,
            skill_limits=SkillLimits(max_resource_reads=1),
            checkpoint_store=checkpoints,
        )
        failed = await first_agent.run(
            "정책을 확인해줘",
            session_id="restore-quota",
            skills=["restore-quota"],
        )
        assert failed.error == "pause after resource read"
        checkpoint = await checkpoints.load(failed.run_id)
        assert checkpoint is not None
        assert checkpoint.skill_state.resource_reads == 1

        stricter_agent = Agent(
            config=AgentConfig("restore-quota", "Use the reference."),
            model=RecordingModel([ModelResponse(Message.assistant("unused"))]),
            skill_registry=registry,
            skill_limits=SkillLimits(max_resource_reads=0),
            checkpoint_store=checkpoints,
        )
        resumed = await stricter_agent.resume(
            failed.run_id,
            session_id="restore-quota",
        )

        assert resumed.finish_reason is FinishReason.ERROR
        assert resumed.error == "checkpoint reads exceed max_resource_reads"

    asyncio.run(scenario())


def test_hallucinated_resource_tool_without_registry_resumes_complete_protocol() -> (
    None
):
    async def scenario() -> None:
        hallucinated_call = ToolCall(
            "hallucinated-read-1",
            "moduagent_skill_read",
            {
                "skill_name": "missing-skill",
                "path": "references/missing.md",
            },
        )
        model = RecordingModel(
            [
                ModelResponse(
                    Message.assistant(None, (hallucinated_call,)),
                    (hallucinated_call,),
                ),
                ModelResponse(Message.assistant("등록된 도구 없이 복구했습니다.")),
            ]
        )
        checkpoints = InMemoryCheckpointStore()
        agent = Agent(
            config=AgentConfig("no-skills", "Use only registered Tools."),
            model=model,
            checkpoint_store=checkpoints,
            conversation_memory_policy=RecentTurnsConversationMemoryPolicy(max_turns=1),
        )

        failed = await agent.run(
            "처리해줘",
            session_id="hallucinated-resource-resume",
        )

        assert failed.finish_reason is FinishReason.ERROR
        assert failed.error == "Skill resource tools are not configured"
        checkpoint = await checkpoints.load(failed.run_id)
        assert checkpoint is not None
        assert [
            message.tool_call_id
            for message in checkpoint.messages
            if message.role.value == "tool"
        ] == ["hallucinated-read-1"]

        resumed = await agent.resume(
            failed.run_id,
            session_id="hallucinated-resource-resume",
        )

        assert resumed.finish_reason is FinishReason.COMPLETED
        assert resumed.output == "등록된 도구 없이 복구했습니다."
        assert resumed.error is None

    asyncio.run(scenario())


def test_skill_allowed_tools_never_bypass_tool_authorizer() -> None:
    async def scenario() -> None:
        executions: list[int] = []

        @function_tool
        def dangerous_write(value: int) -> int:
            executions.append(value)
            return value

        registry = SkillRegistry.from_sources(
            InMemorySkillSource(
                {
                    "write-workflow": _skill(
                        "write-workflow",
                        "Use dangerous_write when explicitly authorized.",
                        allowed_tools=("dangerous_write",),
                    )
                }
            )
        )
        call = ToolCall(
            "write-1",
            "dangerous_write",
            {"value": 7},
        )
        model = RecordingModel(
            [
                ModelResponse(Message.assistant(None, (call,)), (call,)),
                ModelResponse(Message.assistant("권한이 없어 실행하지 않았습니다.")),
            ]
        )
        agent = Agent(
            config=AgentConfig("secured-skill", "Respect authorization."),
            model=model,
            tools=[dangerous_write],
            skill_registry=registry,
            tool_authorizer=RBACToolAuthorizer(
                role_permissions={"writer": {"dangerous_write"}}
            ),
        )

        result = await agent.run(
            "값을 기록해줘",
            skills=["write-workflow"],
            user_context={"roles": ["reader"]},
        )

        assert result.output == "권한이 없어 실행하지 않았습니다."
        assert executions == []
        tool_messages = [
            message for message in result.messages if message.role.value == "tool"
        ]
        assert len(tool_messages) == 1
        assert "unauthorized" in (tool_messages[0].content or "")

    asyncio.run(scenario())
