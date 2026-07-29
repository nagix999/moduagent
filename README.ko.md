# ModuAgent

[English](https://github.com/nagix999/moduagent/blob/main/README.md) |
[한국어](https://github.com/nagix999/moduagent/blob/main/README.ko.md)

ModuAgent는 자체 모델 엔드포인트와 Python 함수를 바탕으로 AI 에이전트를
구축할 수 있는 조합형 Python 런타임입니다.

일반 모델 호출이나 도구 호출 루프로 시작할 수 있습니다. 애플리케이션에
필요해지는 시점에 범위가 제한된 대화 메모리, 검증된 Pydantic 출력, 엄격한
계획-실행(Plan-and-Execute), 체크포인트 복구, 스킬, 관측성 기능을 추가할 수
있습니다.

> 현재 버전: **0.4.2 (Alpha)** · Python **3.10+** · **MIT License**

ModuAgent를 처음 사용한다면 먼저 1단계와 2단계를 완료합니다. 그다음 사용
사례에 필요할 때만 메모리, 구조화 출력 또는 계획-실행을 추가합니다.

## ModuAgent가 제공하는 기능

에이전트는 작고 명시적인 구성 요소를 조합하여 만듭니다.

```text
User input
    │
    ▼
Agent ──► Execution profile ──► Model
  │              │                │
  │              └──────────────► Tools
  │
  ├── Conversation store + memory policy
  ├── Output codec
  ├── Checkpoint store
  ├── Skills and authorization
  └── Events, diagnostics, and metrics
```

- `AgentConfig`는 지침, 재시도 동작, 실행 제한을 정의합니다.
- 모델 클라이언트는 vLLM, Ollama 또는 그 밖의 지원되는 엔드포인트에 연결합니다.
- 도구는 모델이 호출할 수 있는 타입이 지정된 Python 함수입니다.
- 실행 프로필은 작업의 진행 방식을 제어합니다.
- 출력 코덱은 텍스트 또는 검증된 Pydantic 객체를 반환합니다.
- 대화 저장소는 기록을 보관하고, 메모리 정책은 모델에 전달할 범위를 선택합니다.
- 체크포인트 저장소는 중단된 실행을 저장하여 안전하게 복구할 수 있게 합니다.

대부분의 애플리케이션은 모델과 소수의 도구로 시작하는 것이 좋습니다.
요구 사항이 생길 때 나머지 구성 요소를 추가합니다.

### 실행 방식 선택

| | Standard 실행 | 엄격한 계획-실행 |
|---|---|---|
| 선택 방식 | 기본값 | 명시적으로 활성화 |
| 적합한 용도 | 채팅, 직접적인 도구 사용, 짧은 루프 | 의존 관계가 있고 감사 가능한 다단계 작업 |
| 흐름 | 모델 → 선택적 도구 → 답변 | 계획 → 실행 → 검증/반영 → 답변 |
| 비용 | 지연 시간이 짧고 호출 수가 적음 | 더 강한 제어를 위해 호출 수가 많음 |
| 중간 상태 | 가벼운 상태 | 버전 관리되고 검증된 단계 상태 |

중간 단계를 각각 검증하거나 안전하게 재개해야 하는 경우가 아니라면
Standard 실행을 사용합니다.

## 설치

ModuAgent에는 Python 3.10 이상이 필요합니다. 접근 가능한 모델 서버도
필요하며, ModuAgent 자체는 모델을 호스팅하지 않습니다.

패키지를 설치합니다.

```bash
python -m pip install "moduagent==0.4.2"
```

패키지 인덱스에 아직 `0.4.2`가 없고 이미 0.4 소스를 체크아웃했다면 저장소
루트에서 설치합니다.

```bash
cd /path/to/moduagent
python -m pip install -e .
```

기여자는 개발 도구도 함께 설치할 수 있습니다.

```bash
python -m pip install -e '.[dev]'
```

선택적 통합 기능은 별도로 설치합니다.

```bash
python -m pip install redis       # Redis conversation/checkpoint stores
python -m pip install matplotlib  # report automation example
python -m pip install "psycopg[binary]>=3.2,<4"  # PostgreSQL report example
```

## 1단계: 첫 에이전트 실행

아래 예시는 vLLM의 OpenAI 호환 엔드포인트에 연결하고 모델만 사용하는
에이전트를 실행합니다.

```python
import asyncio
import os

from moduagent import Agent, AgentConfig, VLLMClient


def create_model() -> VLLMClient:
    return VLLMClient(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=os.environ["VLLM_MODEL"],
        api_key=os.getenv("VLLM_API_KEY"),
        timeout=60,
    )


async def main() -> None:
    agent = Agent(
        config=AgentConfig(
            name="assistant",
            instructions="Answer accurately and concisely.",
        ),
        model=create_model(),
    )

    result = await agent.run(
        "Explain what an AI agent is in one paragraph.",
        session_id="getting-started",
    )

    if result.error:
        raise RuntimeError(result.error)

    print(result.output)
    print(f"Tokens used: {result.usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
```

파일을 실행하기 전에 엔드포인트를 설정합니다.

```bash
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL="your-model-name"
export VLLM_API_KEY="optional-token"
python getting_started.py
```

엔드포인트와 선택한 모델은 에이전트가 사용하는 도구 호출이나 JSON 스키마
출력 같은 기능을 지원해야 합니다.
도구 예제를 사용하려면 선택한 모델에 맞는 vLLM 채팅 템플릿과 도구 파서를
설정합니다.

`Agent.run()`은 `AgentResult`를 반환합니다. 가장 유용한 필드는 다음과 같습니다.

| 필드 | 의미 |
|---|---|
| `output` | 최종 텍스트 또는 검증된 객체 |
| `error` | 외부에 공개해도 안전한 오류 메시지 또는 `None` |
| `finish_reason` | `completed`, `timeout`, `max_steps` 등의 종료 이유 |
| `usage` | 누적된 모델 토큰 사용량 |
| `run_id` | 체크포인트 복구에 사용하는 식별자 |
| `messages` | 외부에 공개되는 대화 메시지 |
| `metadata` | 안전하게 정리된 도구 추적 정보, 계획 요약, 오류 범주 |

Ollama도 동일한 에이전트 API를 사용합니다.

```python
from moduagent import OllamaClient

model = OllamaClient(
    base_url="http://localhost:11434",
    model="qwen3:14b",
)
```

## 2단계: 도구 추가

`@function_tool`을 사용하여 타입이 지정된 Python 함수를 노출합니다.
이 예시는 1단계의 `create_model()`을 다시 사용합니다.

```python
import asyncio

from moduagent import Agent, AgentConfig, RetryConfig, function_tool


@function_tool(
    idempotent=True,
    timeout_seconds=5,
    max_result_bytes=4096,
)
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def main() -> None:
    calculator = Agent(
        config=AgentConfig(
            name="calculator",
            instructions=(
                "Use the add Tool whenever addition is required. "
                "Do not invent a calculated result."
            ),
            retry=RetryConfig(max_attempts=2),
        ),
        model=create_model(),
        tools=[add],
    )

    result = await calculator.run(
        "What is 12 plus 30?",
        session_id="calculator-demo",
    )

    if result.error:
        raise RuntimeError(result.error)

    print(result.output)
    print(result.metadata.get("tool_trace", []))


asyncio.run(main())
```

함수의 타입 힌트는 입력 스키마가 되고 독스트링은 모델에 표시되는 설명이
됩니다. `tools=[...]`로 전달한 도구만 호출할 수 있습니다.

`idempotent=True`는 검증된 동일 호출을 반복해도 안전하다는 선언입니다.
트랜잭션이나 정확히 한 번(exactly-once) 실행을 보장하는 기능은 아닙니다.
쓰기 도구에는 애플리케이션 수준의 멱등성 키와 중복 방지가 여전히 필요합니다.

`pandas.read_sql()` 같은 동기 함수는 이벤트 루프 밖에서 실행됩니다. 운영
환경에서는 동기 도구가 공통 bounded scheduler를 사용하게 하여 timeout된
호출이 백그라운드 스레드를 무제한 생성하지 않도록 합니다.

```python
from moduagent import SyncToolScheduler, function_tool

blocking_tools = SyncToolScheduler(max_workers=8, max_queue=32)

@function_tool(sync_scheduler=blocking_tools, timeout_seconds=10)
def query_db(sql: str) -> list[dict]:
    return run_read_only_query(sql)
```

원본 어시스턴트 도구 호출과 원본 도구 결과는 내부 프로토콜 메시지입니다.
실행 중에는 모델이 사용할 수 있지만 `ConversationStore`나
`AgentResult.messages`에는 추가되지 않습니다. 기본 공개 도구 추적 정보는
크기가 제한되며 비밀 정보를 노출하지 않는 요약입니다.

## 3단계: 대화 메모리 추가

동일한 `session_id`를 사용하면 대화를 이어갈 수 있습니다.

이후의 예제 코드는 새로 추가하는 설정에 집중하며 앞 단계에서 정의한
`create_model()`과 `add`를 다시 사용합니다. 각 비동기 예제는
애플리케이션 진입점에서 호출할 수 있는 함수로 감쌌습니다.

```python
from moduagent import (
    Agent,
    AgentConfig,
    InMemoryConversationStore,
    RecentTurnsConversationMemoryPolicy,
)

conversations = InMemoryConversationStore(ttl_seconds=3600)

memory_agent = Agent(
    config=AgentConfig(
        name="memory-assistant",
        instructions="Use relevant conversation context when answering.",
    ),
    model=create_model(),
    conversation_store=conversations,
    conversation_memory_policy=RecentTurnsConversationMemoryPolicy(max_turns=6),
)

async def demonstrate_memory() -> None:
    remembered = await memory_agent.run(
        "Remember that my deployment region is Seoul.",
        session_id="user-42",
    )
    if remembered.error:
        raise RuntimeError(remembered.error)

    result = await memory_agent.run(
        "Which deployment region did I choose?",
        session_id="user-42",
    )
    if result.error:
        raise RuntimeError(result.error)
    print(result.output)
```

저장소와 메모리 정책의 역할은 서로 다릅니다.

| 구성 요소 | 역할 |
|---|---|
| `ConversationStore` | 외부에 공개되는 전체 대화를 저장합니다. |
| `ConversationMemoryPolicy` | 모델에 전달할 대화 범위를 선택합니다. |

`RecentTurnsConversationMemoryPolicy`는 저장된 메시지를 삭제하지 않습니다.
모델에는 가장 최근의 완전한 대화 턴만 전달합니다. 인메모리 저장소는 단일
프로세스 개발 및 테스트 용도입니다.

엄격한 토큰 제한과 자동 요약은
[Conversation Memory 가이드](https://github.com/nagix999/moduagent/blob/main/docs/conversation-memory-policy.md)를
참고합니다.
vLLM의 정확한 토큰 계산을 반복해서 사용한다면 `VLLMTokenCounter`를
`CachingTokenCounter`로 감쌉니다. cache에는 크기가 제한된 keyed digest와
성공한 토큰 수만 저장됩니다.

## 4단계: 검증된 구조화 출력 반환

`PydanticOutputCodec`은 모델의 최종 응답을 검증하고 해당 모델 객체를
`result.output`으로 반환합니다.

```python
from pydantic import BaseModel, Field

from moduagent import Agent, AgentConfig, PydanticOutputCodec


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)


structured_agent = Agent(
    config=AgentConfig(
        name="structured-assistant",
        instructions=(
            "Use the add Tool for every arithmetic operation, then return "
            "the answer in the requested format."
        ),
    ),
    model=create_model(),
    tools=[add],
    output_codec=PydanticOutputCodec(model=Answer),
)

async def demonstrate_structured_output() -> None:
    result = await structured_agent.run(
        "What is 20 plus 22?",
        session_id="structured-demo",
    )

    if result.error:
        raise RuntimeError(result.error)

    answer: Answer = result.output
    print(answer.answer, answer.confidence)
```

도구와 구조화 출력을 함께 사용할 수 있습니다. ModuAgent는 요청 단계를
다음과 같이 분리합니다.

```text
ACT:      model receives Tool schemas, without the final output schema
FINALIZE: model receives the Pydantic schema, without Tools
```

이 방식은 도구 호출과 구조화 출력을 같은 요청에 넣었을 때 흔히 발생하는
vLLM 충돌을 방지합니다. `VLLMClient`는 기본적으로 이 조합을 지원하지 않는
것으로 선언하므로 런타임이 분리 방식을 사용합니다.

## 5단계: 엄격한 계획-실행 사용

작업에 서로 의존하는 단계가 있고 각 중간 결과가 최종 답변에 반영되기 전에
검증되어야 한다면 계획-실행을 사용합니다.

```python
from moduagent import (
    Agent,
    AgentConfig,
    InMemoryCheckpointStore,
    InMemoryConversationStore,
    LLMPlanGenerator,
    PlanExecutionProfile,
    RunLimits,
)

model = create_model()

planning_agent = Agent(
    config=AgentConfig(
        name="planning-agent",
        instructions=(
            "Use the add Tool for every arithmetic operation. "
            "Complete multi-step requests using only validated and committed "
            "step results."
        ),
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
        plan_generator=LLMPlanGenerator(
            model=model,
            max_steps=4,
        ),
        revise_on_tool_failure=True,
    ),
    conversation_store=InMemoryConversationStore(),
    checkpoint_store=InMemoryCheckpointStore(),
)

async def demonstrate_plan_execution() -> None:
    result = await planning_agent.run(
        "Calculate 10 + 20, then add 5 to that verified result.",
        session_id="plan-demo",
    )
    if result.error:
        raise RuntimeError(result.error)
    print(result.output)
```

엄격한 실행 흐름은 다음과 같습니다.

```text
PLAN → ACT_TOOL → STEP_RESULT → VALIDATE/COMMIT → VERIFY → FINALIZE
```

- `max_steps`는 모델 호출 횟수가 아니라 생성되는 계획 단계 수를 제한합니다.
- `max_step_attempts`는 한 단계의 검증 재시도 횟수를 제한합니다.
- `max_replans`는 완료되지 않은 작업을 수정하는 횟수를 제한합니다.
- `max_tool_calls`는 전체 실행에서 업무 도구 호출 횟수를 제한합니다.
- `timeout_seconds`는 계획 수립부터 모델 호출, 도구 실행, 최종 응답 생성,
  영속화까지 모든 작업이 공유하는 하나의 실행 기한입니다.

채팅, 직접적인 도구 사용, 짧은 워크플로에는 Standard 실행이 더 나은
기본값입니다.

### 전체 리포트 자동화 예제

저장소에는 단 두 개의 도구만 사용하는 완전한 계획-실행 에이전트가
포함되어 있습니다.

- `query_db`: SQLite(기본값) 또는 PostgreSQL에서 제한된 읽기 전용 쿼리를
  실행합니다.
- `plot_graph`: 해당 실행에 속한 쿼리 산출물을 읽고 PNG 차트를 생성합니다.

[examples/report_automation_agent.py](https://github.com/nagix999/moduagent/blob/main/examples/report_automation_agent.py)를
참고합니다.

```bash
python -m pip install matplotlib
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL="your-tool-capable-model"
python examples/report_automation_agent.py
```

같은 예제를 PostgreSQL에서 실행하려면 위 모델 환경변수를 유지하고 다음을
추가합니다.

```bash
python -m pip install "psycopg[binary]>=3.2,<4"
export REPORT_DB_BACKEND="postgresql"
export REPORT_DATABASE_URL="postgresql://report_reader@localhost:5432/reporting"
python examples/report_automation_agent.py
```

`CONNECT`, 스키마 `USAGE`, `SELECT` 권한만 가진 전용 데이터베이스 역할을
사용하세요. 예제도 읽기 전용 트랜잭션을 시작하고 statement 및 lock
timeout을 적용합니다.

## 다섯 단계를 마친 후

이제 대부분의 에이전트를 만들 수 있습니다. 다음 기능은 선택 사항이며,
스트리밍, 복구, 재사용 가능한 도메인 절차 또는 배포 제어가 필요할 때
추가합니다.

## 결과 스트리밍

사용자에게 보여 줄 출력에는 `stream()`을 사용합니다.

```python
from moduagent import EventType

async def stream_result() -> None:
    result = None

    async for event in planning_agent.stream(
        "Run the task and stream the final answer.",
        session_id="stream-demo",
    ):
        if event.type in (EventType.MODEL_DELTA, EventType.FINAL_DELTA):
            print(event.data["delta"], end="", flush=True)
        elif event.type in (EventType.RUN_COMPLETED, EventType.RUN_FAILED):
            result = event.data["result"]

    if result is not None and result.error:
        raise RuntimeError(result.error)
```

두 가지 공개 델타를 모두 처리합니다. Standard의 직접 응답은
`MODEL_DELTA`를 사용하고, 계획-실행을 포함한 단계별 최종 응답 생성은
`FINAL_DELTA`를 사용합니다.

`stream_all()`은 진단용 내부 이벤트와 중간 모델 델타도 노출합니다.
사용자에게 직접 전달하지 말고 접근이 통제된 진단 경로에서만 사용합니다.

## 스텝과 오류 원인 확인

실행 과정은 `EventSink` 또는 `stream_all()`로 확인합니다. 개발자가 예외의
정제된 원인도 확인해야 한다면 `DiagnosticSink`를 추가합니다.

```python
from moduagent import InMemoryDiagnosticSink, LoggingEventSink

diagnostics = InMemoryDiagnosticSink(max_records=1_000)

observable_agent = Agent(
    config=AgentConfig(
        name="observable-agent",
        instructions="Complete the request using the available Tools.",
    ),
    model=create_model(),
    tools=[add],
    event_sink=LoggingEventSink(),
    diagnostic_sink=diagnostics,
    diagnostic_timeout_seconds=0.25,
    diagnostic_max_pending_deliveries=1_024,
)

result = await observable_agent.run("Use add for 20 + 22.")

if result.failure_id is not None:
    failure = diagnostics.get(result.failure_id)
    if failure is not None:
        print(failure.to_dict())

for failure in diagnostics.for_run(result.run_id):
    print(failure.failure_id, failure.component, failure.operation)
```

`result.metadata["tool_trace"]`에는 실제 실행된 도구와 연관 ID가 기록됩니다.
`result.failure_id`는 종료된 실행의 근본 오류를 가리킵니다. 연관된 도구
레코드의 `terminal`은 최종 결과가 아니라 수집 당시 복구 가능성을 뜻하므로,
이후 Plan 정책이 중단을 결정한 경우에도 `False`일 수 있습니다. 복구된 도구
오류는 도구 추적 정보와 `diagnostics.for_run()`에만 남을 수 있습니다.

진단은 기본적으로 꺼져 있습니다. `diagnostic_sink`를 생략하거나
`NoopDiagnosticSink`를 사용하면 0.4 동작을 유지합니다. 전달은 최선 노력
방식이며
`diagnostic_timeout_seconds`와 `diagnostic_max_pending_deliveries`로
제한됩니다. 표준 라이브러리 로깅은 제한된 데몬 워커 풀에서 실행되며 이미
실행 중인 동기 핸들러는 강제로 취소할 수 없습니다. 사용자 정의 비동기
수집기는 취소를 준수해야 합니다.

진단 필드는 크기가 제한되며 원본 예외 메시지, SQL, 프롬프트, 도구 인자와
결과, 공급자 응답 본문, 소스 코드 줄, 지역 변수를 수집하지 않습니다. 실제
`OSError.errno`와 즉시 읽을 수 있는 허용 속성은 보존할 수 있지만 Pydantic
동적 키는 숨깁니다. 잘린 트레이스백은 가장 안쪽 프레임을 유지합니다. 내장
이벤트 로그는 페이로드와 자유 형식 사유를 생략하고 스텝·도구 연관 ID를
해시합니다.

엄격한 Plan 검증은 프레임워크가 소유하는 `validation_code`,
`validation_location`, 선택적 `validation_cause_code`를 제공하므로 사유
문자열을 파싱하지 말고 이 값을 확인합니다. 사용자 정의 Engine의
`EngineOutcome.error`는 공개되는 신뢰 텍스트로 취급해야 합니다.
`AgentResult.metadata["error_summary"]`는 런타임 예약 필드이므로 Engine
메타데이터로 덮어쓸 수 없습니다. 로깅, 영속 사용자 정의 수집기, 보안 지침은
[진단 가이드](https://github.com/nagix999/moduagent/blob/main/docs/diagnostics.md)를
참고합니다.

## 성능 지표

`MetricsEventSink`는 phase별 `model.calls`와 모델 실행 시간뿐 아니라 메모리
준비, 도구, 체크포인트, 전체 실행, 동일 세션 대기 시간을 기록합니다. Noop
관측성은 큐와 복사 경로를 완전히 건너뜁니다. 이벤트 전달 큐에는 상한이 있어
느린 sink가 payload를 무제한 보관하는 대신 backpressure를 적용합니다.

실행 또는 영속화 코드를 변경한 뒤 소스 트리의 microbenchmark를 실행합니다.

```bash
python benchmarks/performance_v042.py --pretty
```

## 체크포인트와 안전한 재개

중단된 작업을 이어서 실행해야 한다면 체크포인트 저장소를 추가합니다.

```python
from moduagent import InMemoryCheckpointStore

checkpoints = InMemoryCheckpointStore()

resumable_agent = Agent(
    config=AgentConfig(
        name="resumable-agent",
        instructions="Complete the request safely.",
    ),
    model=create_model(),
    tools=[add],
    conversation_store=InMemoryConversationStore(),
    checkpoint_store=checkpoints,
)

async def resume_if_safe() -> None:
    failed = await resumable_agent.run(
        "Run the task.",
        session_id="resume-demo",
    )

    error_summary = failed.metadata.get("error_summary", {})

    if failed.error and error_summary.get("resumable") is True:
        resumed = await resumable_agent.resume(
            failed.run_id,
            session_id="resume-demo",
        )
        print(resumed.output)
```

원래의 `run_id`, 동일한 `session_id`, 호환되는 에이전트 설정으로 재개합니다.
`error_summary["resumable"]`이 `true`일 때만 재개합니다.
`InMemoryCheckpointStore`는 동일한 프로세스 안에서의 복구만 보여 주는
예제입니다. 프로세스가 종료되면 모든 체크포인트가 사라집니다.

- `retryable`은 새 실행을 시도할 수 있다는 의미입니다.
- `resumable`은 안전하지 않은 부작용을 재실행하지 않고 저장된 실행을
  이어갈 수 있다는 의미입니다.

체크포인트가 존재하더라도 `resumable`은 `false`일 수 있습니다. 특히 도구가
시작되었지만 그 결과가 영속적으로 확정되지 않았을 수 있습니다. ModuAgent는
해당 도구를 자동으로 다시 실행하지 않고 실행을 중단하여 수동 검토를 요구합니다.

체크포인트를 사용하는 에이전트에는 원자적 `append_once()`를 지원하는
`ConversationStore`가 필요합니다. 내장 인메모리 저장소와 지원되는 Redis
저장소는 이 계약을 구현합니다. 프로덕션에서는 Redis 또는 사용자 정의
영속 어댑터를 사용합니다.

## 스킬로 도메인 지식 추가

스킬은 재사용 가능한 지침과 크기가 제한된 텍스트 리소스를 제공합니다.
도구 권한을 부여하지는 않습니다.

```text
skills/
└── invoice-review/
    ├── SKILL.md
    ├── references/
    │   └── policy.md
    └── assets/
        └── report-template.md
```

```python
from moduagent import SkillRegistry, function_tool


@function_tool(idempotent=True)
def lookup_invoice(invoice_id: str) -> dict[str, object]:
    """Look up an invoice by ID."""
    return {
        "invoice_id": invoice_id,
        "amount": 125_000,
        "evidence_attached": True,
        "approved": False,
    }


skills = SkillRegistry.from_paths("./examples/skills")

agent = Agent(
    config=AgentConfig(
        name="invoice-agent",
        instructions="Use verified evidence only.",
    ),
    model=create_model(),
    tools=[lookup_invoice],
    skill_registry=skills,
)

async def review_invoice() -> None:
    result = await agent.run(
        "Review invoice INV-100.",
        session_id="invoice-42",
        skills=["invoice-review"],
    )
    if result.error:
        raise RuntimeError(result.error)
    print(result.output)
```

예제 도구는 시연을 위해 고정된 데이터를 반환합니다. 함수 본문을 권한이
부여된 자체 데이터 접근 코드로 교체합니다.

유효한 도구 범위는 등록된 도구, 스킬의 `allowed-tools`, 설정된
`ToolAuthorizer`의 교집합입니다. 스킬의 `scripts/`는 자동으로 실행되지
않습니다.

작성법, 잠금 파일, 리소스 제한, 자동 선택은
[에이전트 스킬 가이드](https://github.com/nagix999/moduagent/blob/main/docs/skills.md)를
참고합니다.

## 확정된 에이전트 설정 확인

`Agent.inspect()`는 외부 요청을 보내지 않으며, 변경할 수 없고 비밀 정보를
노출하지 않는 `AgentSpec`을 반환합니다.

```python
spec = planning_agent.inspect()

print(spec.execution_profile.kind)  # plan
print(spec.agent_fingerprint)
print(spec.to_dict(include_instructions=False))
```

이 명세에는 확정된 모델 기능, 도구 스키마 지문과 안전 프로필, 출력 동작,
영속성 정책, 호환성 메타데이터가 포함됩니다. API 키와 토큰은 마스킹됩니다.

## 프로덕션 적용 전

- 인메모리 대화, 체크포인트, 요약 저장소를 영속 저장소로
  교체합니다.
- 모델, 도구, 데이터베이스, 전체 실행의 시간 제한을 각각 설정합니다.
- 동기 Python 도구에 적용한 시간 제한은 실행 중인 스레드를 강제로 멈출 수
  없습니다. 드라이버 또는 서버 측 명령문 시간 제한도 설정합니다.
- 데이터베이스 행, 도구 결과 바이트, 컨텍스트 토큰, 출력 토큰, 계획 단계,
  도구 호출 수를 제한합니다.
- 도구의 부작용을 검토한 뒤에만 재시도 또는 변경된 인자를 사용한 복구가
  안전하다고 선언합니다.
- 쓰기 도구에는 애플리케이션 수준의 멱등성 키와 중복 처리를 적용합니다.
- `ToolAuthorizer` 또는 RBAC를 사용합니다. 스킬의 `allowed-tools`는 범위를
  좁힐 뿐입니다.
- 인자 로깅에 명확하고 검토된 목적이 없다면 기본 요약 도구 추적 정보를
  유지합니다.
- 원본 예외, SQL, 자격 증명, 고객 데이터, 내부 경로를 모델에 보이는
  오류 메시지에 포함하지 않습니다.
- 대화, 체크포인트, 이벤트, 생성된 산출물에 암호화, 테넌트 격리, 접근 제어,
  보존 기간, TTL을 설정합니다.
- 배포 설정과 함께 `agent.inspect()` 결과를 기록하고, 운영 중인 에이전트를
  업그레이드하기 전에 재개 동작을 테스트합니다.
- 공개 스트림은 사용자에게 보내고 내부 이벤트는 보호된 `EventSink`로
  보냅니다. 오류 진단은 별도로 접근이 통제된 `DiagnosticSink`에 저장합니다.

ModuAgent는 분산 잠금, 작업자 큐, 스케줄러, 영속 아웃박스, 종단 간 정확히
한 번 도구 실행을 제공하지 않습니다. 필요한 경우 애플리케이션 인프라를
통해 추가합니다.

## 자주 묻는 질문

### Pydantic 출력을 사용했을 때 도구가 실행되지 않은 이유는 무엇인가요?

0.4에서는 도구 선택과 최종 구조화 출력이 서로 다른 모델 단계입니다. 도구가
호출되지 않았다면 모델의 도구 호출 지원 여부, 채팅 템플릿과 파서 설정,
도구 설명, 에이전트 지침을 확인합니다.

### 계획-실행이 `max_steps`로 종료된 이유는 무엇인가요?

엄격한 계획-실행에서 `max_steps`는 생성되는 계획 단계의 최대
개수입니다. 전체 모델 요청 횟수가 아닙니다. 작업에 독립적으로 검증할 단계가
실제로 더 필요한 경우에만 값을 늘립니다.

### 실제로 호출된 도구는 어디에서 확인하나요?

`result.metadata["tool_trace"]`를 사용합니다. 계획 단계의 `allowed_tools`는
실행된 도구가 아니라 허용된 도구 목록입니다.

### `InMemoryConversationStore`는 프로덕션 저장소인가요?

아닙니다. 프로세스 로컬 저장소이며 예제, 테스트, 개발을 위한 용도입니다.
다중 프로세스 환경이나 재시작 후에도 데이터를 유지해야 하는 시스템에서는
Redis 또는 영속 사용자 정의 저장소를 사용합니다.

## 공개 API 안내

| 목적 | 주요 API |
|---|---|
| 구성 및 실행 | `Agent`, `AgentConfig`, `RunLimits`, `RetryConfig` |
| 모델 연결 | `VLLMClient`, `OllamaClient` |
| 도구 추가 | `function_tool`, `ToolSafetyProfile`, `ToolAuthorizer` |
| 실행 방식 선택 | `StandardExecutionProfile`, `PlanExecutionProfile` |
| 출력 검증 | `PydanticOutputCodec`, `TextOutputCodec` |
| 대화 유지 | `ConversationStore`, `RecentTurnsConversationMemoryPolicy` |
| 작업 재개 | `CheckpointStore`, `Agent.resume()` |
| 도메인 절차 추가 | `SkillRegistry`, `SkillSelector` |
| 실행 관측 | `Agent.stream_all()`, `EventSink`, `DiagnosticSink`, `failure_id` |
| 설정 확인 | `Agent.inspect()`, `AgentSpec` |

## 예제

- [리포트 자동화 에이전트](https://github.com/nagix999/moduagent/blob/main/examples/report_automation_agent.py):
  `query_db`와 `plot_graph`만 사용하며 SQLite와 PostgreSQL 쿼리 백엔드를
  지원하는 엄격한 계획-실행 예제입니다.
- [청구서 검토 스킬](https://github.com/nagix999/moduagent/tree/main/examples/skills/invoice-review):
  스킬 지침, 참고 자료, 자산 예제입니다.

## 문서

상세 가이드는 현재 한국어로 작성되어 있습니다.

- [Core API](https://github.com/nagix999/moduagent/blob/main/docs/core-api.md):
  에이전트 구성 및 공통 API를 설명합니다.
- [Advanced API](https://github.com/nagix999/moduagent/blob/main/docs/advanced-api.md):
  사용자 정의 `Engine`, 도구 실패 계약, 확장 지점을 설명합니다.
- [계획-실행](https://github.com/nagix999/moduagent/blob/main/docs/plan-and-execute.md):
  엄격한 상태 머신과 복구 세부 정보를 설명합니다.
- [Conversation Memory](https://github.com/nagix999/moduagent/blob/main/docs/conversation-memory-policy.md):
  최근 대화 턴, 토큰 예산, 요약을 설명합니다.
- [에이전트 스킬](https://github.com/nagix999/moduagent/blob/main/docs/skills.md):
  재사용 가능한 절차와 리소스 접근을 설명합니다.
- [Operations](https://github.com/nagix999/moduagent/blob/main/docs/operations.md):
  보안, 시간 제한, 저장소, 이벤트, 배포를 설명합니다.
- [Diagnostics](https://github.com/nagix999/moduagent/blob/main/docs/diagnostics.md):
  스텝 타임라인, 오류 연관 관계, 정제된 상세 정보, 사용자 정의 수집기를
  설명합니다.
- [0.4 마이그레이션](https://github.com/nagix999/moduagent/blob/main/docs/migration-0.4.md):
  소스 호환성과 체크포인트 마이그레이션을 설명합니다.
- [Changelog](https://github.com/nagix999/moduagent/blob/main/CHANGELOG.md)

## 개발

오프라인 테스트 모음을 실행합니다.

```bash
python -m pytest -q tests --ignore=tests/integration
```

실제 vLLM, Ollama, Redis 테스트는 `tests/integration`에 있으며 해당 환경
변수가 설정되지 않으면 건너뜁니다.

```bash
ruff check .
ruff format --check .
```

## 라이선스

ModuAgent는
[MIT License](https://github.com/nagix999/moduagent/blob/main/LICENSE)로
제공됩니다.
