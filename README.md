# ModuAgent

PDF 기술 설계에 따라 구현한 합성형 Python AI Agent 프레임워크입니다. `Agent`/`AgentRuntime`, vLLM·Ollama, Function Tool, 토큰 스트리밍, 구조화 출력, RBAC, 대화·체크포인트 저장, Plan-and-Execute, 관측성 컴포넌트를 제공합니다.

- [ConversationMemoryPolicy 설계](docs/conversation-memory-policy.md)

## 설치

Python 3.10 이상이 필요합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

Redis 저장소를 사용할 때는 비동기 Redis 클라이언트를 별도로 설치합니다.

```bash
python -m pip install redis
```

## 기본 사용

타입 힌트는 Tool 입력 스키마가 되고 docstring은 설명이 됩니다. `idempotent=True`인 Tool만 설정된 횟수만큼 재시도됩니다.

```python
import asyncio

from moduagent import (
    Agent,
    AgentConfig,
    InMemoryConversationStore,
    RecentTurnsConversationMemoryPolicy,
    RetryConfig,
    VLLMClient,
    function_tool,
)


@function_tool(idempotent=True, timeout_seconds=5.0, max_result_bytes=4096)
def add(a: int, b: int) -> int:
    """두 정수를 더한다."""
    return a + b


async def main() -> None:
    model = VLLMClient(
        base_url="http://vllm.internal:8000/v1",
        model="company-model",
        api_key="internal-token",
    )
    conversations = InMemoryConversationStore(ttl_seconds=3600)
    agent = Agent(
        config=AgentConfig(
            name="calculator",
            instructions="필요하면 계산 도구를 사용해 간결하게 답한다.",
            retry=RetryConfig(max_attempts=2),
        ),
        model=model,
        tools=[add],
        conversation_store=conversations,
        conversation_memory_policy=RecentTurnsConversationMemoryPolicy(max_turns=6),
    )

    result = await agent.run("12와 30을 더해줘", session_id="demo-session")
    if result.error:
        raise RuntimeError(result.error)
    print(result.output)


asyncio.run(main())
```

`InMemoryConversationStore`는 단일 프로세스 메모리에 대화 원문을 저장합니다. 동일한 `session_id`를 사용하면 이전 대화가 이어지며, 위 예제에서는 마지막 접근 후 1시간이 지나면 대화가 만료됩니다. `RecentTurnsConversationMemoryPolicy(max_turns=6)`는 저장된 원문을 지우지 않고, 모델 요청에만 최근 완료 대화 6개와 현재 실행 내용을 전달합니다. Tool Call과 그 결과는 하나의 대화로 묶어서 유지됩니다.

### 긴 대화의 토큰 제한

최근 대화 개수와 입력 토큰 예산을 함께 제한하려면 `TokenBudgetConversationMemoryPolicy`를 사용합니다.

```python
from moduagent import (
    Agent,
    AgentConfig,
    InMemoryConversationStore,
    TokenBudget,
    TokenBudgetConversationMemoryPolicy,
    VLLMTokenCounter,
)

memory_policy = TokenBudgetConversationMemoryPolicy(
    budget=TokenBudget(
        context_window_tokens=32_768,
        reserved_output_tokens=2_048,
        safety_margin_tokens=2_048,
    ),
    token_counter=VLLMTokenCounter(model),
    max_history_turns=6,
)

agent = Agent(
    config=AgentConfig(
        name="bounded-assistant",
        instructions="대화 맥락을 활용해 간결하게 답한다.",
    ),
    model=model,
    conversation_store=InMemoryConversationStore(ttl_seconds=3600),
    conversation_memory_policy=memory_policy,
)
```

`input_budget` 값은 `context_window_tokens - reserved_output_tokens - safety_margin_tokens`입니다. 예산을 넘으면 가장 오래된 완전한 대화부터 모델 요청에서 제외하며, system instruction과 현재 실행은 항상 유지합니다. 이 필수 내용만으로도 예산을 넘으면 모델을 호출하지 않고 `ConversationMemoryOverflowError`로 종료합니다.

`VLLMTokenCounter`는 vLLM의 `/tokenize` API를 사용해 실제 chat template 기준으로 계산합니다. 해당 API를 사용할 수 없는 환경에서는 `ApproximateTokenCounter`를 대신 주입하고 충분한 `safety_margin_tokens`를 설정하세요. 메모리 Policy는 모델에 전달하는 view만 제한하며 `ConversationStore`와 `AgentResult.messages`의 원문은 변경하지 않습니다.

### 오래된 대화 요약

오래된 내용을 버리지 않고 짧은 메모리로 남기려면 `SummarizingConversationMemoryPolicy`를 사용합니다. 요약 모델은 기존 vLLM 모델을 그대로 재사용할 수 있습니다.

```python
from moduagent import (
    InMemoryMemoryStateStore,
    ModelConversationSummarizer,
    SummarizingConversationMemoryPolicy,
    TokenBudget,
    VLLMTokenCounter,
)

token_counter = VLLMTokenCounter(model)
memory_policy = SummarizingConversationMemoryPolicy(
    budget=TokenBudget(
        context_window_tokens=32_768,
        reserved_output_tokens=2_048,
        safety_margin_tokens=1_024,
    ),
    token_counter=token_counter,
    summarizer=ModelConversationSummarizer(
        model=model,
        token_counter=token_counter,
        max_input_tokens=8_192,
        max_output_tokens=512,
    ),
    state_store=InMemoryMemoryStateStore(),
    max_history_turns=6,
)

agent = Agent(
    config=AgentConfig(
        name="summary-assistant",
        instructions="대화 맥락을 활용해 간결하게 답한다.",
    ),
    model=model,
    conversation_store=InMemoryConversationStore(ttl_seconds=3600),
    conversation_memory_policy=memory_policy,
)
```

최근 6개 대화는 원문으로 유지하고 그보다 오래된 연속 구간은 요약합니다. 6개 이하라도 입력 토큰 예산을 넘으면 오래된 대화부터 추가로 요약합니다. 하나의 메시지가 매우 커도 summarizer가 제한된 크기의 여러 요청으로 나누어 요약을 누적합니다.

요약은 모델 요청에만 삽입되는 파생 cache이므로 `ConversationStore`와 `AgentResult.messages`의 원문은 그대로 유지됩니다. `InMemoryMemoryStateStore`의 cache는 프로세스 재시작 시 사라집니다. cache가 없거나 요약 범위가 늘어나면 추가 모델 호출이 발생하며, 그 시간과 토큰 사용량도 현재 run의 timeout과 `Usage`에 포함됩니다. 요약 호출이 실패하면 토큰 예산에 맞는 최근 대화만 사용해 계속합니다.

Tool 선택·실행 단계에서는 최종 응답용 구조화 출력 스키마를 적용하지 않습니다. 따라서 `PydanticOutputCodec`을 함께 사용해도 JSON Schema가 Tool Calling을 방해하지 않습니다.

Ollama는 모델 객체만 바꾸면 됩니다. `VLLMClient`, `OllamaClient`, `Agent` 생성자는 키워드 인자를 사용합니다.

```python
from moduagent import OllamaClient

model = OllamaClient(
    base_url="http://localhost:11434",
    model="qwen3:14b",
)
```

## 토큰 스트리밍

`Agent.stream()`은 토큰뿐 아니라 모델·Tool·정책·종료 이벤트를 순서대로 전달합니다. 토큰은 `MODEL_DELTA`, 최종 `AgentResult`는 마지막 `RUN_COMPLETED` 또는 `RUN_FAILED` 이벤트에 들어 있습니다.

```python
from moduagent import EventType

result = None
async for event in agent.stream("짧게 소개해줘", session_id="stream-session"):
    if event.type is EventType.MODEL_DELTA:
        print(event.data["delta"], end="", flush=True)
    elif event.type in (EventType.RUN_COMPLETED, EventType.RUN_FAILED):
        result = event.data["result"]

print()
if result is not None and result.error:
    raise RuntimeError(result.error)
```

## Pydantic 구조화 출력

모델이 JSON Schema 구조화 출력을 지원해야 합니다. 성공 시 `result.output`은 문자열이 아니라 지정한 Pydantic 객체입니다.

```python
from pydantic import BaseModel, Field

from moduagent import Agent, AgentConfig, PydanticOutputCodec


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)


structured_agent = Agent(
    config=AgentConfig(
        name="structured-answer",
        instructions="반드시 요청된 JSON 스키마로 답한다.",
    ),
    model=model,
    output_codec=PydanticOutputCodec(model=Answer),
)

result = await structured_agent.run("한국의 수도는?")
print(result.output.answer, result.output.confidence)
```

`tools`가 등록된 Agent에서는 실행과 최종 구조화를 분리합니다. ACT 요청에는 Tool 스키마만 전달하고 최종 출력 스키마는 전달하지 않습니다. Policy가 종료를 결정하면 런타임이 Tool 없이 Pydantic 스키마만 적용한 FINALIZE 요청을 한 번 더 보내 `result.output`을 만듭니다. 실제 Tool이 호출되지 않았더라도 이 추가 모델 호출은 발생합니다.

## RBAC Tool 권한

기본 Authorizer는 모든 Tool을 허용합니다. 운영 환경에서는 역할별 Tool allowlist를 명시하고, 실행 시 `user_context.roles`를 전달합니다. 등록되지 않은 역할과 Tool은 거부됩니다.

```python
from moduagent import Agent, AgentConfig, RBACToolAuthorizer

authorizer = RBACToolAuthorizer(
    role_permissions={
        "calculator_user": {"add"},
        "admin": {"*"},
    }
)

secured_agent = Agent(
    config=AgentConfig(
        name="secured-calculator",
        instructions="허용된 계산 도구만 사용한다.",
    ),
    model=model,
    tools=[add],
    tool_authorizer=authorizer,
)

result = await secured_agent.run(
    "1과 2를 더해줘",
    session_id="rbac-session",
    user_context={"roles": ["calculator_user"]},
)
```

## Redis 대화·체크포인트와 재개

Redis 클라이언트는 동기·비동기 방식 모두 주입할 수 있습니다. 아래 예시는 `redis.asyncio`를 사용합니다. 대화 TTL과 체크포인트 TTL은 서로 독립적입니다.

```python
from redis.asyncio import Redis

from moduagent import (
    Agent,
    AgentConfig,
    RedisCheckpointStore,
    RedisConversationStore,
)

redis = Redis.from_url("redis://localhost:6379/0", decode_responses=True)
conversations = RedisConversationStore(
    client=redis,
    ttl_seconds=7 * 24 * 60 * 60,
)
checkpoints = RedisCheckpointStore(
    client=redis,
    ttl_seconds=24 * 60 * 60,
)

resumable_agent = Agent(
    config=AgentConfig(
        name="resumable-agent",
        instructions="요청을 끝까지 수행한다.",
    ),
    model=model,
    conversation_store=conversations,
    checkpoint_store=checkpoints,
)

failed = await resumable_agent.run("작업을 수행해줘", session_id="session-42")
if failed.error and await checkpoints.load(failed.run_id) is not None:
    resumed = await resumable_agent.resume(
        failed.run_id,
        session_id="session-42",
    )
```

재개 시에는 원래 실행과 같은 `session_id`, 호환되는 Agent 설정·Policy·Tool 구성을 사용해야 합니다. 완료된 실행의 체크포인트는 자동 삭제됩니다.

## Plan-and-Execute

복수 단계 작업은 `LLMPlanGenerator`와 `PlanAndExecutePolicy`를 조합합니다. 계획은 `policy_state`에 저장되므로 체크포인트와 함께 복구됩니다.

```python
from moduagent import (
    Agent,
    AgentConfig,
    LLMPlanGenerator,
    PlanAndExecutePolicy,
)

planning_agent = Agent(
    config=AgentConfig(
        name="research-agent",
        instructions="현재 계획 단계에 맞춰 실행하고 간결하게 답한다.",
    ),
    model=model,
    tools=[add],
    decision_policy=PlanAndExecutePolicy(
        plan_generator=LLMPlanGenerator(model=model, max_steps=4),
        revise_on_tool_failure=True,
    ),
)
```

Pydantic 구조화 출력과 Tool을 함께 쓰면 모델 호출은 `PLAN → ACT → FINALIZE`로 분리됩니다.

- PLAN: 계획 전용 스키마로 실행 계획을 생성합니다.
- ACT: 최종 출력 스키마 없이 Tool을 선택·실행하며 계획을 진행합니다.
- FINALIZE: Tool 없이 최종 Pydantic 스키마로 결과를 구조화합니다.

세 단계와 재시도는 모두 하나의 `RunLimits.timeout_seconds` 실행 제한을 공유합니다. FINALIZE는 추가 모델 호출이므로 모델 지연 시간과 계획 단계 수를 포함해 전체 제한을 설정해야 합니다.

## 관측성

실행 이벤트는 여러 Sink로 동시에 보낼 수 있습니다. 기본 로깅·감사 Sink는 비밀 키를 재귀적으로 마스킹하며, Sink 실패는 Agent 실행을 중단하지 않습니다.

```python
from moduagent import (
    Agent,
    AgentConfig,
    AuditEventSink,
    CompositeEventSink,
    InMemoryMetricRecorder,
    LoggingEventSink,
    MetricsEventSink,
)

metrics = InMemoryMetricRecorder()
audit = AuditEventSink()  # 운영에서는 writer=...를 주입
events = CompositeEventSink(
    sinks=[
        LoggingEventSink(),
        MetricsEventSink(recorder=metrics),
        audit,
    ]
)

observed_agent = Agent(
    config=AgentConfig(name="observed-agent", instructions="간결하게 답한다."),
    model=model,
    event_sink=events,
)
```

`LoggingEventSink`, `MetricsEventSink`, `AuditEventSink` 외에도 `EventSink.publish(event)` 계약을 구현해 사내 로그·메트릭·감사 시스템을 연결할 수 있습니다.

## 테스트

네트워크 없이 전체 단위·런타임 통합 테스트를 실행합니다.

```bash
python3 -m pytest -q tests --ignore=tests/integration
```

실제 모델 서버 연동 테스트는 환경 변수를 설정한 뒤 선택 실행합니다.

```bash
VLLM_BASE_URL=http://localhost:8000/v1 \
VLLM_MODEL=company-model \
VLLM_API_KEY=token \
python3 -m pytest -q tests/integration/test_live_models.py -k vllm

OLLAMA_BASE_URL=http://localhost:11434 \
OLLAMA_MODEL=qwen3:14b \
python3 -m pytest -q tests/integration/test_live_models.py -k ollama
```

환경 변수가 없으면 해당 live 테스트는 skip됩니다.

SQLite repository 계약은 별도 서비스 없이 실행되며, Redis는 `REDIS_URL`이
설정된 경우에만 연결합니다.

```bash
python3 -m pytest -q tests/integration/test_live_persistence.py -k database

REDIS_URL=redis://localhost:6379/15 \
python3 -m pytest -q tests/integration/test_live_persistence.py -k redis
```

## 현재 보장 범위

- 체크포인트는 실행 상태 재개를 지원하지만 Tool 부작용의 exactly-once 실행을 보장하지 않습니다. 쓰기 Tool은 자체 idempotency key와 중복 처리 방어를 구현해야 합니다.
- `idempotent=True`는 프레임워크의 Tool 재시도를 허용한다는 의미이며 exactly-once 보장이 아닙니다.
- 동일 세션 직렬화는 한 `AgentRuntime` 프로세스 안에서만 적용됩니다. 분산 lock, 분산 실행 큐, worker scheduler는 현재 범위에 포함되지 않습니다.
- Redis 대화 저장은 list 명령을 지원하는 클라이언트에서 append 원자성을 사용하지만, 전체 Agent 실행의 분산 트랜잭션을 제공하지 않습니다.
