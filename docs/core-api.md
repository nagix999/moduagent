# ModuAgent Core API

이 문서는 0.4에서 Agent를 만들고 실행하는 데 필요한 최소 API를 설명합니다. custom Engine, 저수준 Tool 복구, 저장 schema는 [Advanced API](advanced-api.md), 배포 설정은 [Operations](operations.md)를 참고하세요.

## 설치

```bash
python -m pip install moduagent==0.4.0
```

Python 3.10 이상이 필요합니다. Redis adapter를 사용하면 `redis` 패키지도 설치합니다.

## Standard Agent

`Agent`, `AgentConfig`, 모델만 있으면 Standard Agent를 만들 수 있습니다. Tool은 타입 힌트로 입력 schema를 만들고 docstring을 모델 설명으로 사용합니다.

```python
import asyncio

from moduagent import (
    Agent,
    AgentConfig,
    StandardExecutionProfile,
    VLLMClient,
    function_tool,
)


@function_tool(idempotent=True)
def add(a: int, b: int) -> int:
    """두 정수를 더한다."""
    return a + b


async def main() -> None:
    model = VLLMClient(
        base_url="http://localhost:8000/v1",
        model="company-model",
        api_key="token",
    )
    agent = Agent(
        config=AgentConfig(
            name="calculator",
            instructions="필요할 때 계산 Tool을 사용해 간결하게 답한다.",
        ),
        model=model,
        tools=[add],
        execution_profile=StandardExecutionProfile(),
    )

    result = await agent.run("12와 30을 더해줘", session_id="demo")
    if result.error:
        raise RuntimeError(result.error)
    print(result.output)


asyncio.run(main())
```

`execution_profile`을 생략해도 Standard가 기본값입니다. Standard의 일반 호출 수와 최종화 규칙은 0.3.2 동작을 유지합니다.

## Plan-and-Execute Agent

여러 단계가 필요하고 각 결과를 검증·커밋해야 하는 작업은 `PlanExecutionProfile`을 사용합니다.

```python
from moduagent import (
    Agent,
    AgentConfig,
    LLMPlanGenerator,
    PlanExecutionProfile,
    RunLimits,
)

planning_agent = Agent(
    config=AgentConfig(
        name="research-agent",
        instructions="검증된 단계 결과만 사용해 답한다.",
        limits=RunLimits(
            max_steps=4,
            max_step_attempts=2,
            max_replans=1,
            max_tool_calls=8,
            timeout_seconds=120,
        ),
    ),
    model=model,
    tools=[add],
    execution_profile=PlanExecutionProfile(
        plan_generator=LLMPlanGenerator(model=model, max_steps=4),
        revise_on_tool_failure=True,
    ),
)
```

strict Plan의 기본 흐름은 다음과 같습니다.

```text
PLAN → ACT_TOOL → STEP_RESULT → STEP_VALIDATE/COMMIT → VERIFY → FINALIZE
```

`max_steps`는 계획 단계 수, `max_step_attempts`는 단계 검증 재시도, `max_replans`는 미완료 계획 수정, `max_tool_calls`는 전체 Tool 호출 예산입니다. 모든 모델·Tool·저장소 작업은 하나의 `timeout_seconds` deadline을 공유합니다.

기존 `decision_policy=PlanAndExecutePolicy(...)`도 지원됩니다. 새 코드에는 실행 방식을 더 명시적으로 보여 주는 `execution_profile`을 권장합니다.

## 구성 확인

`Agent.inspect()`는 조립이 끝난 불변 `AgentSpec`을 반환하며 외부 I/O를 수행하지 않습니다.

```python
spec = planning_agent.inspect()

print(spec.execution_profile.kind)       # plan
print(spec.execution_profile.engine_id)  # plan
print(spec.agent_fingerprint)
print(spec.to_dict(include_instructions=False))
```

inspect 결과에는 다음 정보가 들어갑니다.

- 적용된 `RunLimits`와 retry 설정
- 모델 adapter와 capability
- 선택된 실행 Profile과 Engine state version
- Tool schema fingerprint와 `ToolSafetyProfile`
- 출력, 대화·체크포인트, stream, Skill 정책
- 기존 API에서 변환된 경우 compatibility metadata

API key, password, token처럼 민감한 이름의 모델 옵션은 정제됩니다. `include_instructions=False`를 사용하면 system instruction도 숨길 수 있습니다.

## 구조화 출력

`PydanticOutputCodec`을 사용하면 성공한 `result.output`이 Pydantic 객체가 됩니다.

```python
from pydantic import BaseModel, Field

from moduagent import PydanticOutputCodec


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)


agent = Agent(
    config=AgentConfig(
        name="structured",
        instructions="요청된 출력 형식을 지킨다.",
    ),
    model=model,
    tools=[add],
    output_codec=PydanticOutputCodec(model=Answer),
)
```

Tool 선택 요청에는 Tool schema만, 최종화 요청에는 출력 schema만 전달합니다. vLLM에서 Tool Calling과 구조화 출력 schema가 같은 요청에서 충돌하지 않도록 두 경계를 분리합니다.

## 대화와 스트리밍

같은 `session_id`를 사용하면 `ConversationStore`의 이전 대화가 이어집니다. 개발용 단일 프로세스 저장소는 다음처럼 구성합니다.

```python
from moduagent import (
    InMemoryConversationStore,
    RecentTurnsConversationMemoryPolicy,
)

agent = Agent(
    config=AgentConfig(name="assistant", instructions="간결하게 답한다."),
    model=model,
    conversation_store=InMemoryConversationStore(ttl_seconds=3600),
    conversation_memory_policy=RecentTurnsConversationMemoryPolicy(max_turns=6),
)
```

`Agent.stream()`은 공개 이벤트만 전달합니다. strict Plan의 내부 단계와 진단 이벤트가 필요할 때만 `Agent.stream_all()`을 사용하세요. Memory 정책은 모델에 전달하는 view만 제한하며 저장된 대화 원문을 수정하지 않습니다.

## 다음 문서

- Tool 안전과 custom 확장: [Advanced API](advanced-api.md)
- timeout, checkpoint, observability: [Operations](operations.md)
- 0.3.2에서 업그레이드: [0.4 마이그레이션](migration-0.4.md)
