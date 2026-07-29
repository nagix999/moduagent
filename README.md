# ModuAgent

PDF 기술 설계에 따라 구현한 합성형 Python AI Agent 프레임워크입니다. `Agent`/`AgentRuntime`, vLLM·Ollama, Function Tool, Agent Skills, 토큰 스트리밍, 구조화 출력, RBAC, 대화·체크포인트 저장, Plan-and-Execute, 관측성 컴포넌트를 제공합니다.

- [ConversationMemoryPolicy 설계](https://github.com/nagix999/moduagent/blob/main/docs/conversation-memory-policy.md)
- [Agent Skills 사용과 보안](https://github.com/nagix999/moduagent/blob/main/docs/skills.md)
- [Plan-and-Execute 상태 머신](https://github.com/nagix999/moduagent/blob/main/docs/plan-and-execute.md)

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

## Agent Skills

Skill은 모델에 업무 절차와 지식을 제공하고, Tool은 실제 작업을 실행합니다. Skill을 활성화해도 Tool 권한은 추가되지 않습니다.

0.3.0은 운영자가 검토한 로컬·agent-scoped Skill catalog를 신뢰 경계로 사용합니다. 서로 신뢰하지 않는 tenant가 같은 catalog를 공유해야 한다면 tenant별 Registry 또는 별도 Skill 접근 제어 계층을 둡니다.

공식 Agent Skills 형식처럼 Skill마다 `SKILL.md`를 둡니다.

```text
skills/
└── invoice-review/
    ├── SKILL.md
    ├── references/
    │   └── policy.md
    └── assets/
        └── report-template.md
```

```markdown
---
name: invoice-review
description: 회사 정책에 따라 청구서를 검토한다. 청구서 오류와 승인 여부를 확인할 때 사용한다.
metadata:
  version: "1.0.0"
allowed-tools: lookup_invoice lookup_vendor
applies-to:
  - plan
  - act
---

# Invoice review

1. 필요하면 `references/policy.md`를 읽는다.
2. 등록된 조회 Tool로 원본을 확인한다.
3. 확인되지 않은 내용은 추측하지 않는다.
```

명시적으로 Skill을 선택하는 방식이 기본입니다.

```python
from moduagent import Agent, AgentConfig, SkillRegistry

skills = SkillRegistry.from_paths("./skills")
agent = Agent(
    config=AgentConfig(
        name="invoice-agent",
        instructions="근거가 확인된 내용만 답한다.",
    ),
    model=model,
    tools=[lookup_invoice, lookup_vendor],
    skill_registry=skills,
)

result = await agent.run(
    "이 청구서를 검토해줘",
    session_id="invoice-42",
    skills=["invoice-review"],
)
```

모델이 `name`과 `description`만 보고 자동 선택하게 하려면 별도 Selector를 설정합니다. 선택 단계는 Tool 없이 구조화 출력만 사용하는 독립 호출이므로 Tool Calling과 최종 `PydanticOutputCodec`을 방해하지 않습니다.

```python
from moduagent import ModelSkillSelector

agent = Agent(
    config=AgentConfig(name="assistant", instructions="정확하게 답한다."),
    model=model,
    skill_registry=skills,
    skill_selector=ModelSkillSelector(model=model, max_skills=2),
)

result = await agent.run("이 청구서를 검토해줘", skill_mode="auto")
```

`skill_mode="hybrid"`와 `skills=[...]`를 함께 전달하면 명시 Skill을 먼저 고정하고 남은 범위를 자동 선택합니다.

`applies-to`는 Skill 지침을 `plan`, `act`, `finalize` 중 필요한 단계에만 적용합니다. 생략하면 세 단계 모두 적용됩니다. Skill 본문은 대화 저장소와 `AgentResult.messages`에는 저장되지 않습니다. `references/`와 text `assets/`는 모델이 내부 `moduagent_skill_read`·`moduagent_skill_search` Tool로 필요한 부분만 읽습니다. Resource 호출은 business Tool용 `max_tool_calls` 대신 `SkillLimits.max_resource_reads`를 사용합니다.

`allowed-tools`는 권한을 주는 설정이 아닙니다. 실제 사용 범위는 등록된 Tool과 Skill 선언의 교집합이며, 실행 직전에 `ToolAuthorizer`를 다시 적용합니다. RBAC을 사용할 때는 필요한 내부 Resource Tool도 정책에 허용해야 합니다. `scripts/`는 패키지 digest에는 포함되지만 0.3.0에서도 자동 실행하거나 읽지 않습니다.

Skill 작성과 검증:

```bash
moduagent skills init invoice-review --path ./skills
moduagent skills validate ./skills
moduagent skills inspect ./skills --json
moduagent skills lock ./skills
```

운영 환경에서는 생성된 lock을 `SkillRegistry.from_paths("./skills", lockfile="./skills/skills.lock.json")`로 검증하면 시작 시 catalog·version·source·digest 불일치를 차단합니다.

보안 경로 검증을 위해 `FilesystemSkillSource`는 POSIX `openat`/`O_NOFOLLOW`를 요구합니다. 이를 제공하지 않는 플랫폼에서는 `InMemorySkillSource` 또는 같은 격리 계약을 구현한 custom source를 사용하세요.

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

`PlanAndExecutePolicy`에서 기본 `Agent.stream()`은 내부 ACT token을 숨기고 FINALIZE token만 `FINAL_DELTA`로 공개합니다. 최종 `AgentResult`는 마지막 `RUN_COMPLETED` 또는 `RUN_FAILED` 이벤트에 들어 있습니다.

```python
from moduagent import EventType

result = None
async for event in planning_agent.stream(
    "요청을 수행해줘",
    session_id="stream-session",
):
    if event.type is EventType.FINAL_DELTA:
        print(event.data["delta"], end="", flush=True)
    elif event.type in (EventType.RUN_COMPLETED, EventType.RUN_FAILED):
        result = event.data["result"]

print()
if result is not None and result.error:
    raise RuntimeError(result.error)
```

계획, 단계, Tool, 검증, `STEP_MODEL_DELTA`를 함께 확인할 때만 `planning_agent.stream_all(...)`을 사용합니다. 일반 `StandardDecisionPolicy`의 직접 응답 token은 기존처럼 `MODEL_DELTA`입니다.

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

일반 `StandardDecisionPolicy`에서 Tool과 구조화 출력을 함께 쓰면 ACT 요청에는 Tool schema만 전달하고, Tool 없이 Pydantic schema만 적용하는 최종화 요청을 분리합니다. 이 최종화의 공급자 원문도 공개 assistant 메시지로 `ConversationStore`에 저장됩니다.

`AgentConfig.finalization_mode`의 기본값은 0.2의 Standard 호출 수와 호환되는 `structured_only`입니다.

| 값 | `StandardDecisionPolicy` 동작 |
|---|---|
| `structured_only` | Tool과 구조화 출력이 함께 있을 때만 별도 최종화 |
| `always` | 텍스트와 구조화 출력 모두 별도 최종화 |
| `disabled` | 분리 최종화를 끔 |

strict `PlanAndExecutePolicy`는 이 설정과 별개로 모든 단계가 커밋된 뒤 FINALIZE를 정상 실행당 한 번 수행합니다. 텍스트 출력은 schema 없이, 구조화 출력은 공개 Pydantic schema로 실행하며 둘 다 FINALIZE에는 Tool을 전달하지 않습니다. strict Policy와 `finalization_mode="disabled"`를 함께 설정하면 Agent 생성 시 `ValueError`가 발생합니다.

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

재개 시에는 원래 실행과 같은 `session_id`, 호환되는 Agent 설정·Policy·Tool 구성을 사용해야 합니다. 일반 실행의 완료 checkpoint는 삭제됩니다. strict Plan-and-Execute는 FINALIZE 중복 억제를 위해 terminal `DONE` 상태를 checkpoint에 남길 수 있으므로 운영 TTL을 설정합니다.

## Plan-and-Execute

복수 단계 작업은 `LLMPlanGenerator`와 strict `PlanAndExecutePolicy`를 조합합니다. 0.3.2의 `PlanAndExecutePolicy`는 일반 ACT 응답을 단계 완료로 간주하지 않고, strict `StepResult`의 검증과 명시적 커밋을 요구합니다.

```python
from moduagent import (
    Agent,
    AgentConfig,
    LLMPlanGenerator,
    PlanAndExecutePolicy,
    RunLimits,
)

planning_agent = Agent(
    config=AgentConfig(
        name="research-agent",
        instructions="검증된 단계 결과만 사용해 간결하게 답한다.",
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
    decision_policy=PlanAndExecutePolicy(
        plan_generator=LLMPlanGenerator(model=model, max_steps=4),
        revise_on_tool_failure=True,
    ),
)
```

실행 경계는 다음과 같습니다.

```text
PLAN → ACT_TOOL → STEP_RESULT → STEP_VALIDATE/COMMIT → VERIFY → FINALIZE
```

- `ACT_TOOL`: 현재 단계의 허용 Tool schema만 전달합니다.
- `STEP_RESULT`: Tool 없이 내부 `StepResult` schema만 전달합니다.
- `FINALIZE`: Tool 없이 텍스트 또는 공개 Pydantic schema로 응답합니다.

vLLM에도 Tool과 출력 schema를 한 요청에 함께 전달하지 않습니다. `RunLimits.max_steps`는 strict 정책에서 모델 호출 수가 아니라 계획 단계 수이며, `max_step_attempts`와 `max_replans`가 재시도와 재계획을 별도로 제한합니다. 모든 호출과 저장 작업은 하나의 `timeout_seconds`를 공유합니다.

### Tool 실패 복구

0.3.2의 corrected-arguments repair는 명시적으로 활성화하는 일반화된 기능입니다. DB 조회뿐 아니라 검색식, 파일 선택 조건, 외부 API query parameter처럼 모델이 Tool 인자를 고쳐 다시 호출할 수 있는 오류에 적용할 수 있습니다.

```python
from moduagent import (
    RetryConfig,
    ToolError,
    ToolErrorType,
    ToolFailureRecoveryConfig,
    ToolRecoveryAction,
    function_tool,
)


def map_query_error(exc: Exception) -> ToolError | None:
    # 운영 코드에서는 교정 가능한 예외 타입만 allowlist로 분류합니다.
    if not isinstance(exc, ValueError):
        return None
    return ToolError(
        type=ToolErrorType.EXECUTION_ERROR,
        reason="invalid_query_syntax",
        message="The query syntax is invalid; correct the Tool arguments.",
        retryable=False,
        recovery=ToolRecoveryAction.REPAIR_CALL,
    )


@function_tool(
    idempotent=True,
    repair_safe=True,
    error_mapper=map_query_error,
)
def search_records(query: str, limit: int = 100) -> dict:
    return backend.search(query=query, limit=limit)


planning_agent = Agent(
    config=AgentConfig(
        name="repairing-agent",
        instructions="Tool 오류 피드백을 반영해 안전한 인자만 교정한다.",
        retry=RetryConfig(max_attempts=2),
        limits=RunLimits(
            max_tool_repair_attempts=1,
            max_replans=1,
            max_tool_calls=8,
        ),
    ),
    model=model,
    tools=[search_records],
    decision_policy=PlanAndExecutePolicy(
        LLMPlanGenerator(model),
        tool_failure_recovery=ToolFailureRecoveryConfig(
            fallback="replan",
            require_repair_safe=True,
            feedback_mode="safe_message",
        ),
    ),
)
```

세 제한은 서로 다른 동작입니다.

- `RetryConfig`, `idempotent=True`, `ToolError.retryable=True`: 분류된 일시적 오류에 동일한 Tool call과 동일 인자를 다시 실행합니다.
- `max_tool_repair_attempts`와 `repair_safe=True`: `REPAIR_CALL` 오류를 받은 ACT가 같은 Tool을 정확히 한 번, 새 call ID와 교정된 인자로 다시 호출합니다.
- `max_replans`: repair가 불가능하거나 소진됐을 때 미완료 계획을 수정합니다.

`repair_safe`는 `idempotent`와 같은 뜻이 아닙니다. 전자는 달라진 인자로 모델이 다시 호출해도 안전하다는 선언이고, 후자는 같은 호출을 그대로 재실행해도 안전하다는 선언입니다. 기본 Tool은 repair-safe가 아니며 `ToolFailureRecoveryConfig`도 지정하지 않으면 기존 `revise_on_tool_failure` 동작을 유지합니다.

`error_mapper: Callable[[Exception], ToolError | None]`는 예상한 예외만 `RETRY_CALL`, `REPAIR_CALL`, `REPLAN`, `FAIL` 중 하나로 분류합니다. `RETRY_CALL`은 `idempotent=True`, `retryable=True`, 남은 `RetryConfig` 시도가 모두 충족될 때만 같은 호출을 다시 실행합니다. 단, 취소할 수 없는 동기 Tool timeout은 daemon 작업이 계속 실행되므로 기본적으로 재시도하지 않고 `FAIL`로 분류합니다. DB driver의 statement timeout·취소 완료를 보장하는 통합만 `timeout_retry_safe=True`를 명시할 수 있습니다. `REPAIR_CALL`은 opt-in repair, `REPLAN`은 즉시 계획 수정, `FAIL`은 terminal 실패를 요청합니다. `None`을 반환하면 generic 오류 처리로 돌아갑니다. Tool 구현이 이미 안전한 오류를 구성할 수 있다면 `raise ToolFailure(ToolError(...))`로 같은 계약을 전달할 수도 있습니다.

repair 응답은 실행 전에 동일 Tool 1개, 새 call ID, 이전과 다른 canonical JSON 인자인지 검사하고, Tool validation·기본값 적용 뒤의 유효 인자 hash도 다시 비교합니다. 키 순서나 숫자 표현만 바꾸거나 같은 유효 인자를 반복하거나 다른 Tool을 선택하면 Tool 본체를 실행하지 않고 fallback합니다. 여러 Tool을 병렬 호출한 batch가 부분 성공한 경우에는 이미 발생한 side effect를 자동 재실행하지 않도록 `fallback="replan"`이어도 terminal 실패로 끝냅니다.

`feedback_mode`의 기본값은 `"type_only"`이며 strict repair 요청에는 오류 type과 `reason`만 넣습니다. `"safe_message"`는 호출자가 모델 공개에 안전하다고 보증한 `ToolError.message`를 제어문자 제거·길이 제한 후 추가하며, 비밀정보를 자동 탐지하거나 redaction하는 옵션은 아닙니다. strict 실패 Tool 메시지에서는 항상 `message`와 `details`를 제외하지만, 공개 `ToolResult.model_content()`와 non-strict 경로의 호환성은 유지되므로 `error_mapper`와 `ToolFailure`에는 원본 예외, SQL, 접속 문자열, 토큰, 고객 데이터, 내부 경로·schema를 넣지 마세요. 공개 `tool_trace`와 recovery 이벤트에도 원본 오류 메시지나 Tool 결과가 포함되지 않습니다.

`result.metadata["plan"]["steps"][].allowed_tools`는 해당 단계에서 호출할 수 있었던 Tool 목록이며 실제 호출 이력이 아닙니다. 0.3.1부터 실제 호출 시도는 순서가 보존된 `result.metadata["tool_trace"]`에서 확인할 수 있습니다. 기본 `tool_trace_mode="summary"`는 단계·call ID·Tool 이름·성공 여부·시도 횟수·실행 시간과 정제된 오류 분류만 남깁니다. 오류에는 `type`, `retryable`과 선택적인 `reason`, `recovery`가 들어가며, 교정 호출에는 `recovery_of_call_id`가 추가됩니다. `off`는 trace를 만들지 않고, `arguments`는 민감한 키를 재귀적으로 마스킹한 호출 인자를 추가합니다. 실행된 Tool은 validation·형 변환·기본값 적용 후 실제 인자를 `arguments_source="validated"`로 기록하고, 실행 전에 거부된 호출은 요청 인자를 `requested`로 기록합니다.

Tool repair가 발생하면 `result.metadata["plan_usage"]["tool_repairs"]`에 전체 교정 횟수가 추가됩니다. 복구가 terminal 실패로 끝나면 `result.metadata["failure"]`에 정제·길이 제한된 step/Tool 식별자, 오류 분류, fallback 원인만 보존하며 원본 예외나 Tool 결과는 포함하지 않습니다.

```python
planning_agent = Agent(
    config=AgentConfig(
        name="research-agent",
        instructions="검증된 단계 결과만 사용해 간결하게 답한다.",
        tool_trace_mode="arguments",
    ),
    model=model,
    tools=[add],
    decision_policy=PlanAndExecutePolicy(LLMPlanGenerator(model=model)),
)

result = await planning_agent.run("요청")
for call in result.metadata.get("tool_trace", []):
    print(call["step_id"], call["tool_name"], call.get("arguments"))
```

이 trace는 크기가 제한되고 Tool 결과 값이나 원본 오류 메시지를 포함하지 않는 운영 감사용 요약입니다. strict checkpoint에도 내부 실행 metadata로 저장되어 resume 후 이어지며 checkpoint schema는 계속 v3입니다. 인자를 저장해야 할 명확한 운영 목적이 없다면 기본 `summary`를 사용하세요.

Tool이 pandas `DataFrame` 같은 표 형식 값을 반환하면 런타임은 이를 모델이 읽을 수 있는 JSON-safe record 구조로 정규화합니다. 직접 Tool을 구현할 때도 가능한 한 문자열·숫자·불리언·`None`·list·dict로 구성된 값을 반환하고, 대용량 조회는 Tool에서 행과 열을 제한하는 것이 좋습니다.

검증 재시도, Tool 실패 복구 또는 재계획 한도를 소진해 Policy가 terminal 실패를 결정하면 현재 단계는 `failed`로 확정됩니다. 일시적인 transport 오류나 전체 run timeout처럼 checkpoint에서 재개할 수 있는 중단은 `in_progress`로 남을 수 있습니다.

`pd.read_sql` 같은 동기 Tool은 worker thread에서 실행되므로 timeout 결과가 반환되어도 Python이 실행 중인 DB 호출을 강제로 중단하지는 못합니다. Tool timeout과 별도로 DB driver의 query/connection timeout 및 서버 측 statement timeout을 설정하세요.

0.2의 암묵적 단계 완료 동작이 잠시 필요하면 `LegacyPlanAndExecutePolicy`를 명시적으로 사용해야 하며 생성 시 `DeprecationWarning`이 발생합니다. 신규 코드는 strict 정책을 사용하세요. 상태, 스트림, vLLM 분리, checkpoint v3와 마이그레이션 내용은 [Plan-and-Execute 문서](https://github.com/nagix999/moduagent/blob/main/docs/plan-and-execute.md)에 정리했습니다.

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
- strict Plan-and-Execute의 `final_emitted`는 동일 run의 중복 FINALIZE와 중복 방출을 억제하지만 end-to-end exactly-once 전달은 아닙니다. 손실·중복 없는 전달에는 durable outbox, 안정적인 event ID와 소비자 idempotency가 필요합니다.
- 동일 세션 직렬화는 한 `AgentRuntime` 프로세스 안에서만 적용됩니다. 분산 lock, 분산 실행 큐, worker scheduler는 현재 범위에 포함되지 않습니다.
- Redis 대화 저장은 list 명령을 지원하는 클라이언트에서 append 원자성을 사용하지만, 전체 Agent 실행의 분산 트랜잭션을 제공하지 않습니다.
