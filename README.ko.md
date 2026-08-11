# ModuAgent

[English](https://github.com/nagix999/moduagent/blob/main/README.md) |
[한국어](https://github.com/nagix999/moduagent/blob/main/README.ko.md)

ModuAgent는 자체 모델 엔드포인트와 Python 함수를 바탕으로 AI 에이전트를
구축할 수 있는 조합형 Python 런타임입니다.

일반 모델 호출이나 도구 호출 루프로 시작할 수 있습니다. 애플리케이션에
필요해지는 시점에 범위가 제한된 Context Memory, 검증된 Pydantic 출력, 엄격한
계획-실행(Plan-and-Execute), 체크포인트 복구, 스킬, 관측성 기능을 추가할 수
있습니다.

> 현재 버전: **0.5.3** · 상태: **Alpha** · Python **3.10+** · **MIT License**

ModuAgent를 처음 사용한다면 아래의 짧은 다섯 단계를 따라갑니다. 이 단계는
0.5 Quick API를 사용하며, 고급 조합에는 명시적인 구성 요소 API를 그대로
사용할 수 있습니다.
한 번에 개념 하나씩 추가하는 실행 가능한 파일은
[초급 예제](https://github.com/nagix999/moduagent/blob/main/examples/README.md)부터
시작하세요.
여러 도구를 연결하는 작업을 시작할 준비가 되었다면
[중급 예제](https://github.com/nagix999/moduagent/blob/main/examples/INTERMEDIATE.ko.md)로
이어가세요.

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

- `Agent.create()`는 일반적인 설정을 대신 구성합니다.
- `AgentConfig`는 명시적인 조합이 필요할 때 지침, 재시도 동작, 실행 제한을
  제공합니다.
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
python -m pip install "moduagent==0.5.3"
```

패키지 인덱스에 아직 `0.5.3`이 없고 이미 0.5 소스를 체크아웃했다면 저장소
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

from moduagent import Agent, VLLMClient


async def main() -> None:
    async with VLLMClient.from_env(
        default_options={"temperature": 0, "max_tokens": 256},
    ) as model:
        agent = Agent.create(
            model=model,
            instructions="Answer accurately and concisely.",
        )

        answer = await agent.ask(
            "Explain what an AI agent is in one paragraph.",
            session_id="getting-started",
        )
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())
```

파일을 실행하기 전에 엔드포인트를 설정합니다.

```bash
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL="your-model-name"
# export VLLM_API_KEY="your-token-if-required"
export VLLM_TIMEOUT="60"
python getting_started.py
```

`VLLMClient.from_env()`는 다음과 같이 문서에 명시된 환경변수만 읽습니다.

| 환경변수 | 필수 | 의미 |
|---|---|---|
| `VLLM_MODEL` | 예 | vLLM이 제공하는 모델 이름 |
| `VLLM_BASE_URL` | 아니요 | OpenAI 호환 기본 URL. 기본값은 `http://localhost:8000/v1` |
| `VLLM_API_KEY` | 아니요 | Bearer 토큰 |
| `VLLM_TIMEOUT` | 아니요 | 양수인 요청 timeout(초). 기본값은 `60` |

다른 곳에서 설정값을 가져온다면 일반 `VLLMClient(...)` 생성자를 사용합니다.
`from_env()`에 직접 전달한 `timeout=`은 `VLLM_TIMEOUT`보다 우선합니다.

엔드포인트와 선택한 모델은 에이전트가 사용하는 도구 호출이나 JSON 스키마
출력 같은 기능을 지원해야 합니다.
도구 예제를 사용하려면 선택한 모델에 맞는 vLLM 채팅 템플릿과 도구 파서를
설정합니다.

`ask()`는 가장 짧은 실행 경로입니다. 디코딩한 출력을 반환하고 실행이 완료되지
않으면 비밀 정보를 노출하지 않는 `AgentRunError`를 발생시킵니다. 전체 결과가
필요한 운영 코드에서는 `run()`을 사용합니다. 두 방식은 5단계에서 설명합니다.

Ollama도 동일한 에이전트 API를 사용합니다.

```python
from moduagent import OllamaClient

model = OllamaClient(
    base_url="http://localhost:11434",
    model="qwen3:14b",
)
```

OpenAI 호환 및 Ollama 클라이언트는 검증된 임베딩 경계도 제공합니다. 임베딩
엔드포인트용 클라이언트를 구성하고 `embed()`를 직접 호출합니다. 결정적인
벡터 생성에는 에이전트가 필요하지 않습니다.

```python
from moduagent import ModelCapabilities, VLLMClient

async with VLLMClient(
    base_url="http://localhost:8001/v1",
    model="BAAI/bge-m3",
    capabilities=ModelCapabilities(
        chat=False,
        streaming=False,
        tool_calling=False,
        parallel_tool_calling=False,
        structured_output=False,
        embeddings=True,
        tool_calling_with_structured_output=False,
    ),
) as embedding_model:
    vectors = await embedding_model.embed(["첫 번째 문서", "두 번째 문서"])
```

반환된 배치 수는 입력 수와 같아야 합니다. 벡터는 비어 있지 않고 모든 값이
유한하며 차원이 일관되어야 합니다. OpenAI 호환 응답 index는 정확히 한 번씩
나타나는 `0..N-1`이어야 합니다. 잘못된 응답은 문서나 provider 응답 내용을
포함하지 않는 `ModelProtocolError`로 종료됩니다.

## 2단계: 도구 추가

`@tool`을 사용하여 타입이 지정된 Python 함수를 노출합니다.

```python
import asyncio

from moduagent import Agent, VLLMClient, tool


@tool(timeout_seconds=5, max_result_bytes=4096)
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def main() -> None:
    async with VLLMClient.from_env(
        default_options={"temperature": 0, "max_tokens": 256},
    ) as model:
        calculator = Agent.create(
            model=model,
            instructions=(
                "Use the add Tool whenever addition is required. "
                "Do not invent a calculated result."
            ),
            tools=[add],
        )
        answer = await calculator.ask(
            "What is 12 plus 30?",
            session_id="calculator-demo",
        )
        print(answer)


asyncio.run(main())
```

함수의 타입 힌트는 입력 스키마가 되고 독스트링은 모델에 표시되는 설명이
됩니다. `tools=[...]`로 전달한 도구만 호출할 수 있습니다.

`@tool`은 기존 `@function_tool` 어댑터의 짧은 이름입니다. 두 방식 모두 도구의
재시도 또는 수정 후 재실행 안전성을 추측하지 않습니다. 예를 들어
`idempotent=True`는 검증된 동일 호출을 반복해도 안전하다는 선언일 뿐,
트랜잭션이나 정확히 한 번(exactly-once) 실행을 보장하지 않습니다. 쓰기
도구에는 애플리케이션 수준의 멱등성 키와 중복 방지가 여전히 필요합니다.

`pandas.read_sql()` 같은 동기 함수는 이벤트 루프 밖에서 실행됩니다. 운영
환경에서는 동기 도구가 공통 bounded scheduler를 사용하게 하여 timeout된
호출이 백그라운드 스레드를 무제한 생성하지 않도록 합니다.

```python
from moduagent import SyncToolScheduler, tool

blocking_tools = SyncToolScheduler(max_workers=8, max_queue=32)

@tool(sync_scheduler=blocking_tools, timeout_seconds=10)
def query_db(sql: str) -> list[dict]:
    return run_read_only_query(sql)
```

원본 어시스턴트 도구 호출과 원본 도구 결과는 내부 프로토콜 메시지입니다.
실행 중에는 모델이 사용할 수 있지만 `ConversationStore`나
`AgentResult.messages`에는 추가되지 않습니다. 기본 공개 도구 추적 정보는
크기가 제한되며 비밀 정보를 노출하지 않는 요약입니다.

`AgentTool(child_agent)`은 한 프로세스 안에서 다른 에이전트를 도구로 노출하는
legacy 방식입니다. 실패한 child 실행은 도구 실패이며 성공한 `None` 값으로
반환되지 않습니다. Child terminal 실패는 이 legacy 경계에서 재시도할 수 없는
실패로 분류됩니다. 일반 Tool retry 횟수, 변경 인자 repair, timeout retry 또는
`idempotent=True`만으로는 실패한 child를 다시 실행하지 않습니다. 사용자 정의
Agent-like 객체가 이미 안전하게 분류한 `ToolFailure`를 명시적으로 발생시키면
그 계약은 보존됩니다. 이 어댑터에는 root 예산, cycle/depth 차단, receipt,
parent/child session namespace가 없습니다. Parent의 `session_id`를 그대로
전달하므로 parent와 child가 같은 `ConversationStore`
객체를 사용하면 경고가 발생합니다. legacy 위임에서는 저장소를 분리하고 이를
프로덕션 격리 경계로 간주하지 마세요.

## 3단계: 검증된 구조화 출력 반환

Pydantic 모델 클래스를 `output=`으로 전달합니다. Quick API는 내부적으로
기존 `PydanticOutputCodec`을 만들며, `ask()`는 검증된 모델 객체를 반환합니다.

```python
import asyncio

from pydantic import BaseModel, Field

from moduagent import Agent, VLLMClient, tool


@tool
def add(a: int, b: int) -> int:
    """두 정수를 더합니다."""
    return a + b


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)


async def main() -> None:
    async with VLLMClient.from_env(
        default_options={"temperature": 0, "max_tokens": 256},
    ) as model:
        structured_agent = Agent.create(
            model=model,
            instructions=(
                "Use the add Tool for every arithmetic operation, then return "
                "the answer in the requested format."
            ),
            tools=[add],
            output=Answer,
        )
        answer: Answer = await structured_agent.ask(
            "What is 20 plus 22?",
            session_id="structured-demo",
        )
        print(answer.answer, answer.confidence)


if __name__ == "__main__":
    asyncio.run(main())
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

## 4단계: 엄격한 계획-실행 사용

작업에 서로 의존하는 단계가 있고 각 중간 결과가 최종 답변에 반영되기 전에
검증되어야 한다면 계획-실행을 사용합니다.

```python
import asyncio

from pydantic import BaseModel, Field

from moduagent import Agent, RunLimits, VLLMClient, tool


@tool
def add(a: int, b: int) -> int:
    """두 정수를 더합니다."""
    return a + b


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)

async def main() -> None:
    async with VLLMClient.from_env(
        default_options={"temperature": 0, "max_tokens": 512},
    ) as model:
        planning_agent = Agent.create(
            model=model,
            instructions=(
                "Use the add Tool for every arithmetic operation. "
                "Complete multi-step requests using only validated and committed "
                "step results."
            ),
            tools=[add],
            output=Answer,
            execution="plan",
            limits=RunLimits(
                max_steps=4,
                max_step_attempts=2,
                max_replans=1,
                max_tool_calls=8,
                timeout_seconds=120,
            ),
        )
        answer: Answer = await planning_agent.ask(
            "Calculate 10 + 20, then add 5 to that verified result.",
            session_id="plan-demo",
        )
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())
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

`execution="plan"`을 사용하면 Quick API가 같은 `model`로
`LLMPlanGenerator`를 만들고, generator의 `max_steps`를
`limits.max_steps`와 맞춥니다. 사용자 정의 planner, validator 또는 복구
정책이 필요하면 명시적인 `PlanExecutionProfile`을 `execution=`에 전달합니다.

채팅, 직접적인 도구 사용, 짧은 워크플로에는 Standard 실행이 더 나은
기본값입니다.

## 5단계: 결과 확인과 모델 호출 제한

`ask()`는 `run()`을 호출한 뒤 `unwrap()`을 호출하는 것과 같습니다. 성공한
경우 디코딩된 출력만 필요하다면 다음과 같이 사용합니다. 이 단계의 예제는
이미 구성한 `planning_agent`가 있다고 가정하며, 호출하는 동안 모델 client의
비동기 context를 열어 두어야 합니다.

```python
import asyncio


async def main() -> None:
    answer = await planning_agent.ask("Complete the task.")
    print(answer)


asyncio.run(main())
```

사용량, 추적 정보, 종료 이유 또는 복구 metadata도 필요하면 `run()`을
사용합니다.

```python
import asyncio


async def main() -> None:
    result = await planning_agent.run(
        "Complete the task.",
        session_id="operations-demo",
    )
    print(result.explain())  # 간결하고 정제된 종료 요약
    answer = result.unwrap()  # completed가 아니면 AgentRunError 발생
    print(answer)


asyncio.run(main())
```

또는 `unwrap()` 줄을 다음의 명시적 형식으로 바꿔도 같습니다.

```python
result.raise_for_error()
answer = result.output
```

주요 `AgentResult` 필드는 다음과 같습니다.

| 필드 | 의미 |
|---|---|
| `output` | 최종 텍스트 또는 검증된 객체 |
| `error` | 외부에 공개해도 안전한 오류 메시지 또는 `None` |
| `finish_reason` | 안정적으로 분류된 종료 이유 |
| `usage` | 누적된 모델 토큰 사용량 |
| `run_id` | 오류 연관 관계 및 체크포인트 식별자 |
| `messages` | 외부에 공개되는 대화 메시지 |
| `metadata` | 크기가 제한된 도구 추적 정보, 계획 요약, 안전한 오류 범주 |
| `run_usage` | 변경할 수 없는 모델 turn·도구 호출·경과시간 요약 |
| `tool_trace` | 실행된 도구의 변경할 수 없고 크기가 제한된 projection |
| `error_summary` | 변경할 수 없는 안전한 종료 오류 분류 |

종료 이유는 다음과 같습니다.

| 종료 이유 | 의미 |
|---|---|
| `completed` | 출력이 정상적으로 완료됨 |
| `max_steps` | 계획 또는 실행 스텝 예산 소진 |
| `max_tool_calls` | 도구 호출 예산 소진 |
| `max_model_turns` | 전체 실행의 모델 시도 예산 소진 |
| `no_progress` | 같은 의미 상태와 응답이 반복되어 circuit breaker 동작 |
| `timeout` | 전체 실행 기한 만료 |
| `cancelled` | 호출자가 실행 취소 |
| `error` | 그 밖의 종료 오류 발생 |

`ask()`와 `unwrap()`은 `completed` 이외의 모든 종료 이유에 대해 비밀 정보를
노출하지 않는 `AgentRunError`를 발생시킵니다.

```python
import asyncio

from moduagent import AgentRunError


async def main() -> None:
    try:
        answer = await planning_agent.ask("Complete the task.")
        print(answer)
    except AgentRunError as exc:
        print(exc.run_id, exc.finish_reason, exc.code)
        print(exc.retryable, exc.resumable, exc.failure_id)


asyncio.run(main())
```

이 예외는 prompt, 출력, 도구 인자, 원본 provider body 또는 임의의 결과
metadata를 보관하지 않습니다.

### 엄격한 모델 재시도 계약

`RetryConfig.max_attempts`는 첫 호출을 포함하고 기본값은 `1`입니다. 따라서
재시도는 명시적으로 활성화해야 합니다.

```python
from moduagent import RetryConfig

retrying_agent = Agent.create(
    model=model,
    instructions="Answer accurately.",
    retry=RetryConfig(max_attempts=2),
)
```

모델 호출은 다음 allowlist에 해당할 때만 재시도합니다.

- timeout 오류
- 연결 또는 네트워크 오류
- HTTP `408`
- HTTP `5xx`

다음 오류는 재시도하지 않습니다.

- HTTP `429`를 포함한 그 밖의 모든 HTTP `4xx`
- 잘못된 JSON, 유효하지 않은 도구 인자 또는 provider 프로토콜/파싱 오류
- provider 출력이 `timeout`, `length`, `max_tokens`로 끝난 경우
- 구조화 출력 검증 오류
- 잘못된 요청, capability 불일치, `TypeError` 또는 프로그래밍 오류

스트리밍 모델 호출은 공개 delta가 하나라도 전달된 후에는 재시도하지 않습니다.
도구 재시도와 repair는 별도 계약이며 도구가 선언한 안전성 프로필도 충족해야
합니다. 모델 재시도를 활성화했다는 이유만으로 도구 재실행이 안전해지지
않습니다.

provider의 부분 응답은 `model_output_incomplete` code로 실패합니다.
`result.error_summary["provider_finish_reason"]` 또는
`AgentRunError.provider_finish_reason`에서 `timeout`, `length`, `max_tokens`를
구분할 수 있으며, 부분 출력과 provider metadata는 보존하지 않습니다.

### 전체 실행 모델 보호 장치

모든 실행에는 서로 독립적인 두 가지 모델 보호 장치가 있습니다.

```python
limits = RunLimits(
    max_model_turns=32,
    no_progress_model_turn_threshold=3,
)
```

- `max_model_turns=32`는 계획, 실행, 메모리 요약, Skill 선택, repair, 최종
  출력에 걸친 프레임워크 관리 모델 시도 횟수를 제한합니다. 전송 오류
  재시도도 이 예산을 사용합니다.
- `no_progress_model_turn_threshold=3`은 동일한 의미 상태와 정규화된 응답이
  연속 세 번 관찰되면 중단합니다. 성공한 도구 outcome은 run별 salt가 적용된
  fingerprint가 새로울 때만 무진행 횟수를 초기화합니다. 동일한 성공 outcome의
  반복은 보호 장치를 우회하지 않습니다. 새 입력을 소비한 각 메모리 요약
  batch와 커밋된 각 계획 스텝도 진전으로 처리합니다. 어떤 경우에도 전체 모델
  호출 횟수는 초기화하지 않습니다.

이에 해당하는 종료 이유는 `max_model_turns`와 `no_progress`이며 둘 다
자동으로 재시도하거나 안전하게 재개하지 않습니다. 따라서 `error_summary`와
`AgentRunError`는 `retryable=False`, `resumable=False`와 함께 크기가 제한된
카운터와 안전한 분류 필드만 포함합니다.

기본 제공 구성 요소는 보조 호출도 run의 `ModelGateway`를 통과시킵니다.
사용자 정의 메모리 정책, selector, planner와 model client도 이 경계를
지켜야 합니다. provider를 직접 호출하거나 client 내부에서 숨겨진 재시도를
수행하면 프레임워크가 각 요청을 별도 turn으로 계산할 수 없습니다.

체크포인트 저장소가 있으면 provider 재시도를 포함한 모든 프레임워크 관리
모델 시도를 실제 provider I/O 직전에 영속적으로 예약합니다. 따라서 요청 중
프로세스가 강제 종료되어도 재개 시 이미 소비한 turn이 복원되지 않습니다.
모델 보호 장치의 체크포인트에는 숫자 카운터, run별 무작위 salt,
HMAC-SHA-256 관찰 digest만 저장합니다. 성공한 도구 진전 역시 run별 salt가
적용된 fingerprint로 표현합니다. 원본 프롬프트, 모델 출력, 도구 인자와 결과,
provider metadata, provider가 만든 call ID는 보호 장치 상태에 저장하지
않습니다.

## 프레임워크의 책임 범위

ModuAgent 0.5에는 도메인 Recipe, Workflow DSL, 데이터베이스 추상화, SQL 생성
또는 보고서 전용 동작이 포함되지 않습니다. 프레임워크는 구성 요소를 조합하고
실행하며, 애플리케이션은 다음을 직접 담당합니다.

- 지침과 업무 규칙
- 도구 구현과 입출력 스키마
- 기준이 되는 데이터베이스 또는 서비스 스키마
- 도구 멱등성, repair 및 timeout 안전성 선언
- 데이터베이스 역할, 트랜잭션, 쿼리 제한을 포함한 실제 보안 경계

Quick API는 반복되는 프레임워크 연결 코드만 줄입니다. 도메인 의미나 도구
안전성을 추론하지 않습니다.

## 고급 조합: Context Memory

동일한 `session_id`를 사용하면 대화를 이어갈 수 있습니다. `Agent.create()`는
일반적인 저장소, 메모리 정책, 인가, 체크포인트, 스킬과 관측성 구성 요소를
받습니다. 별도 planning model, 사용자 정의 planner·정책, 세부 Tool recovery 또는
사용자 정의 Engine처럼 저수준 조합이 필요할 때 명시적인 생성자를 사용합니다.

```python
from moduagent import Agent, InMemoryConversationStore, RecentTurnsConversationMemoryPolicy

conversations = InMemoryConversationStore(
    ttl_seconds=3600,
    max_sessions=1_000,
    max_total_bytes=16_000_000,
)

memory_agent = Agent.create(
    name="memory-assistant",
    instructions="Use relevant conversation context when answering.",
    model=model,
    conversation_store=conversations,
    memory=RecentTurnsConversationMemoryPolicy(max_turns=6),
)

async def demonstrate_memory() -> None:
    first = await memory_agent.run(
        "Remember that my deployment region is Seoul.",
        session_id="user-42",
    )
    first.raise_for_error()

    result = await memory_agent.run(
        "Which deployment region did I choose?",
        session_id="user-42",
    )
    print(result.unwrap())
```

저장소와 메모리 정책의 역할은 서로 다릅니다.

| 구성 요소 | 역할 |
|---|---|
| `ConversationStore` | 외부에 공개되는 전체 대화를 저장합니다. |
| `ConversationMemoryPolicy` | 모델에 전달할 대화 범위를 선택합니다. |

이 구성 요소는 현재 session에서 사용할 **Context Memory**를 만들고 각 모델
요청의 입력 범위를 제한합니다. 여러 session에서 사실, 선호 또는 episode를
검색하는 **Long-Term Memory**는 제공하지 않습니다.

`RecentTurnsConversationMemoryPolicy`는 저장된 메시지를 삭제하지 않습니다.
모델에는 가장 최근의 완전한 대화 턴만 전달합니다. 인메모리 저장소는 단일
프로세스 개발 및 테스트 용도입니다. 이 정책은 token을 계산하지 않으므로
`MEMORY_COMPACTED`의 `original_tokens=0`, `selected_tokens=0`은 실제 입력이
0 token이라는 뜻이 아니라 “미계수”를 의미합니다.

호환 기본값인 `FullConversationMemoryPolicy`는 입력 길이를 제한하지 않으므로
session이 길어지면 운영 endpoint의 context window를 초과할 수 있습니다.
프로덕션에서는 배포 모델의 exact counter(예: `VLLMTokenCounter`)를 사용하는
`TokenBudgetConversationMemoryPolicy`를 권장하며, 오래된 문맥을 보존해야 할
때만 summarizer를 추가합니다. 자세한 내용은
[Context Memory 가이드](https://github.com/nagix999/moduagent/blob/main/docs/conversation-memory-policy.md)를
참고합니다. vLLM의 exact token 계산을 반복한다면 counter를
`CachingTokenCounter`로 감쌉니다. cache에는 크기가 제한된 keyed digest와
성공한 token 수만 저장됩니다.

### 애플리케이션 예제: 리포트 자동화

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

이 코드는 사용자가 prompt, 스키마, 도구와 안전장치를 정의하는 방법을 보여
주는 애플리케이션 예제입니다. 프레임워크에 내장된 Recipe나 데이터베이스
추상화가 아닙니다.

## Quick API 이후

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

    if result is not None:
        result.raise_for_error()
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
import asyncio
import logging

from moduagent import Agent, InMemoryDiagnosticSink, LoggingEventSink

logging.basicConfig(level=logging.INFO)
diagnostics = InMemoryDiagnosticSink(max_records=1_000)

observable_agent = Agent.create(
    name="observable-agent",
    instructions="Complete the request using the available Tools.",
    model=model,
    tools=[add],
    event_sink=LoggingEventSink(),
    diagnostic_sink=diagnostics,
)


async def main() -> None:
    result = await observable_agent.run("Use add for 20 + 22.")
    print(dict(result.run_usage))
    for trace in result.tool_trace:
        print(dict(trace))
    if result.error_summary:
        print(dict(result.error_summary))

    if result.failure_id is not None:
        failure = diagnostics.get(result.failure_id)
        if failure is not None:
            print(failure.to_dict())

    for failure in diagnostics.for_run(result.run_id):
        print(failure.failure_id, failure.component, failure.operation)


asyncio.run(main())
```

`result.tool_trace`에는 실제 실행된 도구와 연관 ID가 기록됩니다.
`result.failure_id`는 종료된 실행의 근본 오류를 가리킵니다. 연관된 도구
레코드의 `terminal`은 최종 결과가 아니라 수집 당시 복구 가능성을 뜻하므로,
이후 Plan 정책이 중단을 결정한 경우에도 `False`일 수 있습니다. 복구된 도구
오류는 도구 추적 정보와 `diagnostics.for_run()`에만 남을 수 있습니다.

진단은 기본적으로 꺼져 있습니다. `diagnostic_sink`를 생략하거나
`NoopDiagnosticSink`를 사용하면 기본 동작을 유지합니다. 전달은 최선 노력
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

resumable_agent = Agent.create(
    name="resumable-agent",
    instructions="Complete the request safely.",
    model=model,
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
`max_model_turns`와 `no_progress`도 항상 `resumable=false`인 terminal 보호
결정입니다. 재개로 소비한 turn 예산을 늘리거나 동작한 circuit을 다시 열 수
없습니다.

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
from moduagent import SkillRegistry, tool


@tool(idempotent=True)
def lookup_invoice(invoice_id: str) -> dict[str, object]:
    """Look up an invoice by ID."""
    return {
        "invoice_id": invoice_id,
        "amount": 125_000,
        "evidence_attached": True,
        "approved": False,
    }


skills = SkillRegistry.from_paths("./examples/skills")

agent = Agent.create(
    name="invoice-agent",
    instructions="Use verified evidence only.",
    model=model,
    tools=[lookup_invoice],
    skill_registry=skills,
)

async def review_invoice() -> None:
    result = await agent.run(
        "Review invoice INV-100.",
        session_id="invoice-42",
        skills=["invoice-review"],
    )
    print(result.unwrap())
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

`Agent.inspect()`는 외부 요청을 보내지 않으며, 변경할 수 없고 credential이
마스킹된 `AgentSpec`을 반환합니다.

```python
spec = planning_agent.inspect()

print(spec.execution_profile.kind)  # plan
print(spec.agent_fingerprint)
print(spec.to_dict(include_instructions=False))
```

이 명세에는 확정된 모델 기능, 도구 스키마 지문과 안전 프로필, 출력 동작,
영속성 정책, 호환성 메타데이터가 포함됩니다. API 키와 토큰은 마스킹됩니다.
원본 instruction은 객체에 남아 있으므로 instruction에 비밀 정보를 넣지 말고,
명세를 로깅하거나 외부로 내보낼 때는 `include_instructions=False`를 사용합니다.

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

ModuAgent에서는 도구 선택과 최종 구조화 출력이 서로 다른 모델 단계입니다.
도구가
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
| 빠른 구성과 출력 | `Agent.create()`, `Agent.ask()`, `AgentRunError` |
| 실행 운영 | `Agent.run()`, `AgentResult`, `RunLimits`, `RetryConfig` |
| 명시적 조합 | `Agent`, `AgentConfig` |
| 모델 연결 | `VLLMClient`, `OllamaClient` |
| 도구 추가 | `tool`, `function_tool`, `ToolSafetyProfile`, `ToolAuthorizer` |
| 실행 방식 선택 | `StandardExecutionProfile`, `PlanExecutionProfile` |
| 출력 검증 | `PydanticOutputCodec`, `TextOutputCodec` |
| 제한된 session 문맥 유지 | `ConversationStore`, `RecentTurnsConversationMemoryPolicy` |
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
- [프로덕션 제어](https://github.com/nagix999/moduagent/blob/main/examples/PRODUCTION.ko.md):
  인가된 멱등 쓰기, 제한된 메모리, 영속 재개, 취소와 동시 세션 운영
  가이드입니다.

## 문서

상세 가이드는 현재 한국어로 작성되어 있습니다.

- [Core API](https://github.com/nagix999/moduagent/blob/main/docs/core-api.md):
  에이전트 구성 및 공통 API를 설명합니다.
- [Advanced API](https://github.com/nagix999/moduagent/blob/main/docs/advanced-api.md):
  사용자 정의 `Engine`, 도구 실패 계약, 확장 지점을 설명합니다.
- [계획-실행](https://github.com/nagix999/moduagent/blob/main/docs/plan-and-execute.md):
  엄격한 상태 머신과 복구 세부 정보를 설명합니다.
- [Context Memory](https://github.com/nagix999/moduagent/blob/main/docs/conversation-memory-policy.md):
  제한된 session 문맥, token 예산, 요약을 설명합니다.
- [에이전트 스킬](https://github.com/nagix999/moduagent/blob/main/docs/skills.md):
  재사용 가능한 절차와 리소스 접근을 설명합니다.
- [Operations](https://github.com/nagix999/moduagent/blob/main/docs/operations.md):
  보안, 시간 제한, 저장소, 이벤트, 배포를 설명합니다.
- [Diagnostics](https://github.com/nagix999/moduagent/blob/main/docs/diagnostics.md):
  스텝 타임라인, 오류 연관 관계, 정제된 상세 정보, 사용자 정의 수집기를
  설명합니다.
- [0.4 마이그레이션](https://github.com/nagix999/moduagent/blob/main/docs/migration-0.4.md):
  소스 호환성과 체크포인트 마이그레이션을 설명합니다.
- [0.5 마이그레이션](https://github.com/nagix999/moduagent/blob/main/docs/migration-0.5.md):
  Quick API, 안전성 변경과 migration이 필요 없는 0.5.3 PATCH를 설명합니다.
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
