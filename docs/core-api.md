# ModuAgent Core API

이 문서는 0.5에서 Agent를 만들고 실행하는 데 필요한 최소 API를 설명합니다. `Agent.create()`는 기존 구성 요소를 대신하는 별도 런타임이 아니라 일반적인 조립 코드를 줄이는 얇은 facade입니다. custom Engine, 저수준 Tool 복구, 저장 schema는 [Advanced API](advanced-api.md), 배포 설정은 [Operations](operations.md)를 참고하세요.

## 설치

```bash
python -m pip install moduagent==0.5.2
```

Python 3.10 이상이 필요합니다. Redis adapter를 사용하면 `redis` 패키지도 설치합니다.

## Quick API

사용자는 모델, 지침, Tool과 출력 계약을 정의하고 프레임워크는 `AgentConfig`, 실행 Profile과 output codec을 조립합니다.

```python
import asyncio

from moduagent import Agent, VLLMClient, tool


@tool(idempotent=True)
def add(a: int, b: int) -> int:
    """두 정수를 더한다."""
    return a + b


async def main() -> None:
    async with VLLMClient.from_env(
        default_options={"temperature": 0, "max_tokens": 256},
    ) as model:
        agent = Agent.create(
            model=model,
            name="calculator",
            instructions="계산에는 add Tool을 사용해 간결하게 답한다.",
            tools=[add],
        )
        answer = await agent.ask("12와 30을 더해줘", session_id="demo")
        print(answer)


asyncio.run(main())
```

`VLLMClient.from_env()`는 `VLLM_MODEL`을 필수로 읽고 `VLLM_BASE_URL`, `VLLM_API_KEY`, `VLLM_TIMEOUT`을 선택적으로 읽습니다. endpoint를 코드나 다른 설정 시스템에서 가져오려면 기존 생성자를 사용합니다. 내장 HTTP client는 `async with` 또는 `aclose()`로 종료합니다.

`tool`은 `function_tool`의 짧은 별칭이며 동작과 안전 기본값이 같습니다. `idempotent`, `repair_safe`, timeout과 오류 분류를 함수 이름이나 구현에서 추론하지 않으므로 필요한 항목은 사용자가 명시해야 합니다.

### 조립 규칙

`Agent.create()`의 입력은 다음 기존 객체로 결정적으로 변환됩니다.

| Quick API 입력 | 해석 |
|---|---|
| `name`, `instructions`, `limits`, `retry` | 하나의 `AgentConfig` |
| `execution="standard"` | `StandardExecutionProfile()` |
| `execution="plan"` | 같은 모델과 `limits.max_steps`를 사용하는 `PlanExecutionProfile(LLMPlanGenerator(...))` |
| `execution=profile` | 사용자가 만든 `StandardExecutionProfile` 또는 `PlanExecutionProfile`을 그대로 사용 |
| `output=None` | 기본 `TextOutputCodec` |
| `output=SomePydanticModel` | `PydanticOutputCodec(SomePydanticModel)` |
| `output=codec` | 사용자가 만든 `OutputCodec`을 그대로 사용 |
| `memory=policy` | `ConversationMemoryPolicy`로 전달 |
| `conversation_store=store` | 대화 저장소로 전달 |
| `checkpoint_store=store` | 중단 및 재개용 체크포인트 저장소로 전달 |
| `event_sink=sink`, `diagnostic_sink=sink` | 실행 이벤트와 실패 진단 sink로 전달 |
| `diagnostic_timeout_seconds`, `diagnostic_max_pending_deliveries` | 진단 sink 전달의 시간과 대기열 상한 설정 |
| `tool_authorizer=authorizer` | 모든 Tool 실행 전 애플리케이션 인가를 적용 |
| `skill_registry`, `skill_selector`, `skill_limits` | 선택적 Skill 실행 경계를 구성 |
| `model_options`, `metadata`, `finalization_mode`, `stream_visibility` | 기존 `AgentConfig` 필드로 전달 |
| `tool_trace_mode="off"|"summary"|"arguments"` | 결과에 포함할 Tool trace 수준을 선택 |

Plan Quick API는 `max_steps`를 `RunLimits`에 한 번만 지정하면 Planner와 Engine에 같은 값이 적용됩니다. 별도 planning model, custom `PlanGenerator`, custom 정책 또는 custom Engine이 필요하면 아래의 명시적 생성자를 사용합니다.

### 고급 확장 경로

Quick API는 모든 constructor 인자를 숨겨 전달하는 `**kwargs` 통로를 제공하지 않습니다. 지원하지 않는 custom 실행 의미가 필요하면 기존 `Agent(config=..., model=..., ...)`에서 관련 객체를 명시적으로 조립합니다. 프레임워크는 사용자의 지침, Tool 설명, 데이터 의미와 출력 모델 내용을 대신 생성하지 않습니다.

### `ask()`와 `run()`

`ask()`는 성공한 decoded output을 바로 반환하고 실패하면 안전한 `AgentRunError`를 발생시킵니다.

```python
from moduagent import AgentRunError

try:
    answer = await agent.ask("12와 30을 더해줘")
except AgentRunError as exc:
    print(exc.finish_reason, exc.code, exc.run_id)
```

usage, 메시지와 metadata가 필요한 운영 코드는 `run()`을 사용합니다.

```python
result = await agent.run("12와 30을 더해줘")
print(result.explain())
result.raise_for_error()
answer = result.unwrap()
print(result.usage.total_tokens)
print(dict(result.run_usage))
for trace in result.tool_trace:
    print(dict(trace))
```

`run_usage`에는 Coordinator가 관찰한 `model_turns`, `tool_calls`, `duration_seconds`가 성공과 실패 모두에 포함됩니다. `error_summary`, `tool_trace`, `run_usage` 편의 속성은 임의 metadata를 제외한 immutable projection입니다. 실패한 실행은 `dict(result.error_summary)`로 정제된 분류를 확인할 수 있습니다. `tool_trace_mode="arguments"`는 비밀정보로 알려진 키를 마스킹하지만 일반 비즈니스 입력은 노출할 수 있으므로 접근 통제된 환경에서만 사용합니다.

provider가 `timeout`, `length`, `max_tokens`로 부분 응답을 끝내면
`error_summary`의 code는 `model_output_incomplete`이고
`provider_finish_reason`에는 해당 allowlist 값만 포함됩니다. 응답 본문이나
provider metadata는 공개 오류에 포함되지 않습니다.

`raise_for_error()`는 `FinishReason.COMPLETED`가 아니거나 error가 있으면 예외를 발생시키고, `unwrap()`은 같은 검사를 한 뒤 output을 반환합니다. `AgentRunError`와 `explain()`은 allowlist된 요약만 유지하며 원본 프롬프트, 모델 출력, Tool 인자나 임의 metadata를 복제하지 않습니다. 전체 진단은 원래 `AgentResult`와 접근 통제된 sink에서 확인합니다.

0.5에는 `FinishReason.MAX_MODEL_TURNS` (`"max_model_turns"`)와 `FinishReason.NO_PROGRESS` (`"no_progress"`)가 추가됩니다. 둘 다 `RUN_FAILED`이며 각각 전체 provider 시도 예산 소진과 동일 상태·동일 의미 응답의 반복 차단을 뜻합니다.

## 명시적 Standard Agent

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
    async with VLLMClient(
        base_url="http://localhost:8000/v1",
        model="company-model",
        api_key="token",
    ) as model:
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
        result.raise_for_error()
        print(result.output)


asyncio.run(main())
```

`execution_profile`을 생략해도 Standard가 기본값입니다. Standard의 일반 호출 수와 최종화 규칙은 0.3.2 동작을 유지합니다.
`pd.read_sql`처럼 blocking I/O를 수행하는 동기 Tool은 운영 환경에서
`SyncToolScheduler`를 공유하도록 설정하세요. 자세한 구성은
[Operations](operations.md)를 참고합니다.

## Plan-and-Execute Agent

여러 단계가 필요하고 각 결과를 검증·커밋해야 하는 작업은 `PlanExecutionProfile`을 사용합니다.

일반적인 구성은 Quick API로 시작할 수 있습니다.

```python
from moduagent import Agent, RunLimits

planning_agent = Agent.create(
    name="research-agent",
    instructions="검증된 단계 결과만 사용해 답한다.",
    model=model,
    tools=[add],
    execution="plan",
    limits=RunLimits(
        max_steps=4,
        max_step_attempts=2,
        max_replans=1,
        max_tool_calls=8,
        timeout_seconds=120,
    ),
)
```

별도 Planner나 상세 복구 설정이 필요하면 같은 구성을 명시적으로 확장합니다.

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

`max_steps`는 계획 단계 수, `max_step_attempts`는 단계 검증 재시도, `max_replans`는 미완료 계획 수정, `max_tool_calls`는 전체 Tool 호출 예산입니다. `max_model_turns`는 retry를 포함해 공통 `ModelGateway`를 통과하는 실제 provider 시도 예산이고 `no_progress_model_turn_threshold`는 동일한 의미 상태와 응답의 연속 반복 한도입니다. 모든 모델·Tool·저장소 작업은 하나의 `timeout_seconds` deadline을 공유합니다. 자세한 의미와 custom 구성 요소의 gateway 계약은 [Operations](operations.md)를 참고하세요.

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

Quick API에서는 Pydantic 모델 클래스를 직접 전달합니다.

```python
from pydantic import BaseModel, Field

class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)


agent = Agent.create(
    name="structured",
    instructions="요청된 출력 형식을 지킨다.",
    model=model,
    tools=[add],
    output=Answer,
)
```

명시적 구성에서는 codec을 직접 전달합니다.

```python
from moduagent import PydanticOutputCodec

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
정확한 원격 tokenizer를 반복 호출하면 `CachingTokenCounter`로
`VLLMTokenCounter`를 감싸 bounded TTL/LRU cache를 적용할 수 있습니다.

## 다음 문서

- Tool 안전과 custom 확장: [Advanced API](advanced-api.md)
- timeout, checkpoint, observability: [Operations](operations.md)
- 0.4에서 업그레이드: [0.5 마이그레이션](migration-0.5.md)
- 0.3.2에서 업그레이드: [0.4 마이그레이션](migration-0.4.md)
