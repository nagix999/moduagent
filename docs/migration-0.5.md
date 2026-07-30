# ModuAgent 0.5 마이그레이션

이 문서는 ModuAgent 0.4.x 애플리케이션을 0.5로 옮기는 방법을 설명합니다.

## 요약

0.5는 기존 `Agent(config=..., model=..., ...)`, 실행 Profile, `function_tool`, `run()`과 저장 계약을 유지하면서 일반적인 구성을 위한 Quick API를 추가합니다. 도메인 Tool, 프롬프트, schema, 안전 정책과 운영 컴포넌트는 계속 애플리케이션이 결정합니다.

```bash
python -m pip install --upgrade "moduagent==0.5.0"
```

대부분의 코드는 생성 방식을 즉시 바꿀 필요가 없습니다. 다만 모델 retry가 strict allowlist로 바뀌고 모든 run에 유한한 model turn/no-progress guard가 적용되므로, 먼저 기존 API 그대로 회귀 테스트한 뒤 Quick API를 점진적으로 적용하세요.

## 변경 요약

| 0.4 방식 | 0.5 추가 또는 변경 | 호환성 |
|---|---|---|
| `Agent(config=..., model=...)` | 그대로 유지, `Agent.create(...)` 추가 | additive |
| `function_tool` | 그대로 유지, 동일 함수인 `tool` 별칭 추가 | additive |
| `Agent.run()`과 `AgentResult` 직접 검사 | 그대로 유지, `ask()`, `raise_for_error()`, `unwrap()`, `explain()` 추가 | additive |
| `PydanticOutputCodec(Model)` | 그대로 유지, Quick API의 `output=Model` 추가 | additive |
| `PlanExecutionProfile(LLMPlanGenerator(...))` | 그대로 유지, `execution="plan"` 추가 | additive |
| `VLLMClient(...)` | 그대로 유지, `VLLMClient.from_env()` 추가 | additive |
| broad한 model 예외 retry 가능성 | transient transport allowlist만 retry | 의도적인 동작 강화 |
| `RunLimits`의 step/Tool/time 예산 | `max_model_turns=32`, `no_progress_model_turn_threshold=3` 추가 | 필드는 additive, 기본 실행에 새 안전 상한 적용 |
| 기존 terminal reason | `max_model_turns`, `no_progress` 추가 | event consumer 갱신 필요 |

새 `RunLimits` 필드는 dataclass의 마지막에 추가되어 기존 위치 인자의 의미를 바꾸지 않습니다. 신규 코드에서는 앞으로도 keyword 인자를 권장합니다.

## Quick API로 점진적 전환

0.4의 일반 Standard 구성:

```python
agent = Agent(
    config=AgentConfig(
        name="assistant",
        instructions="정확하고 간결하게 답한다.",
    ),
    model=model,
    tools=tools,
)
```

동일한 0.5 Quick 구성:

```python
agent = Agent.create(
    name="assistant",
    instructions="정확하고 간결하게 답한다.",
    model=model,
    tools=tools,
)
```

`Agent.create()`는 별도 런타임이나 축약된 실행 의미를 만들지 않습니다. 다음 기존 객체로 변환한 뒤 같은 `Agent`와 composition 경계를 사용합니다.

| 인자 | resolved 구성 |
|---|---|
| `execution="standard"` | `StandardExecutionProfile()` |
| `execution="plan"` | 같은 model과 `limits.max_steps`를 쓰는 `PlanExecutionProfile`과 `LLMPlanGenerator` |
| 명시적 `execution=profile` | 전달한 Profile을 그대로 사용 |
| `output=None` | `TextOutputCodec` |
| `output=PydanticModel` | `PydanticOutputCodec(PydanticModel)` |
| `output=codec` | 전달한 `OutputCodec`을 그대로 사용 |
| `memory=policy` | 전달한 `ConversationMemoryPolicy` |

Quick Plan 구성은 중복되던 Planner와 Engine의 `max_steps`를 한 곳에 둡니다.

```python
agent = Agent.create(
    name="planner",
    instructions="검증된 단계 결과만 사용한다.",
    model=model,
    tools=tools,
    execution="plan",
    output=PublicResult,
    limits=RunLimits(max_steps=4, max_replans=1),
)
```

다음 기능이 필요하면 기존 명시적 생성자를 유지합니다.

- 별도 planning model 또는 custom `PlanGenerator`
- custom `DecisionPolicy`나 `ExecutionEngine`
- `ConversationStore`, `CheckpointStore`
- event/diagnostic sink
- `ToolAuthorizer`
- Skill registry와 selector
- 상세 Tool failure recovery

Quick API와 명시적 API를 한 Agent에서 중첩 조립하지 말고 필요한 제어 수준에 맞는 진입점 하나를 선택하세요. `Agent.inspect()`로 resolved Profile, Tool/output 계약과 안전 한도를 배포 전에 확인합니다.

## Tool 별칭

```python
from moduagent import tool


@tool(idempotent=True, repair_safe=True)
def lookup(query: str) -> dict:
    """검증된 외부 정보를 조회한다."""
    ...
```

`tool`은 `function_tool`의 alias이므로 안전 기본값도 같습니다. 0.5가 함수 본문이나 이름에서 `idempotent`, `repair_safe`, timeout retry 가능성을 추론하지 않습니다. 기존 decorator를 일괄 변경할 필요도 없습니다.

## `ask()`와 실패 처리

간단한 호출은 decoded output을 직접 받을 수 있습니다.

```python
answer = await agent.ask("질문")
```

`ask()`는 내부적으로 `run()`과 `AgentResult.unwrap()`을 사용합니다. 완료되지 않은 결과에는 `AgentRunError`를 발생시키므로 `MAX_STEPS`, `MAX_TOOL_CALLS`, `MAX_MODEL_TURNS`, `NO_PROGRESS`, timeout과 cancellation을 성공 output으로 오인하지 않습니다.

usage, messages, metadata나 재개 판단이 필요하면 기존 `run()`을 유지합니다.

```python
result = await agent.run("질문")
print(result.explain())
result.raise_for_error()
answer = result.unwrap()
```

`AgentRunError`와 `explain()`은 정제된 run ID, finish reason, category/code, retry/resume 여부와 허용된 카운터만 포함합니다. 원본 프롬프트, 모델 출력, Tool arguments와 임의 metadata는 예외에 복제하지 않습니다.

## 모델 retry 정책 확인

0.5는 다음 오류만 자동 retry합니다.

- timeout과 HTTP 408
- connection/network 오류
- HTTP 5xx

다음 오류는 첫 실패에서 종료합니다.

- HTTP 4xx와 429
- 잘못된 요청 값
- model client의 type/contract 오류
- 응답 JSON, Tool arguments JSON과 stream protocol 오류
- 분류되지 않은 일반 실행 오류

custom model adapter가 transient 오류를 감싸면 cause chain을 보존하세요.

```python
try:
    return await provider_call()
except ConnectionError as exc:
    raise RuntimeError("provider call failed") from exc
```

원인 없이 새 예외만 발생시키면 retry 가능한 transport 유형을 복구할 수 없습니다. 반대로 provider 응답을 해석하지 못한 경우에는 `ModelProtocolError`를 사용해 deterministic parse 실패임을 표시합니다. raw provider body나 credential을 예외 메시지로 복사하지 마세요.

stream은 아직 delta를 내보내지 않은 transient 오류만 retry합니다. 일부 출력 후 재시작하면 사용자가 중복 token을 받을 수 있으므로, delta가 하나라도 방출된 stream 실패는 terminal입니다.

Plan JSON이 잘못됐을 때는 더 이상 catch-all 실행 단계를 만들지 않습니다. schema-only `StepResult`의 JSON/Pydantic schema가 잘못되면 같은 요청을 반복하지 않고 현재 단계를 즉시 실패시키며 `step_result_schema_invalid`를 기록합니다. 업무 근거의 의미 검증 재시도와 protocol/schema 실패를 구분해 alert를 구성하세요.

## model turn과 no-progress 상한

기존 `max_steps`는 Plan 단계 수이고 모델 호출 수가 아닙니다. 0.5에서는 `ModelGateway`를 통과하는 모든 실제 provider 시도를 `max_model_turns`로 별도 제한합니다.

```python
limits = RunLimits(
    max_steps=6,
    max_tool_calls=10,
    max_model_turns=32,
    no_progress_model_turn_threshold=3,
)
```

provider retry도 각각 한 turn을 소비하며 실패한 시도도 이미 외부 요청을 보냈으므로 카운트합니다. streaming chunk 수는 세지 않고 stream 시도 하나를 셉니다. 기본 32회보다 많은 모델 호출이 정상적으로 필요한 Agent는 phase별 호출 수와 retry 최악값을 측정한 뒤 상한을 명시적으로 높이세요.

기본 planner, memory summarizer와 Skill selector는 public `Agent`에서 공통 gateway를 사용합니다. custom 정책·selector·planner가 전달된 gateway를 무시하거나 custom model client가 내부 retry를 숨기면 프레임워크가 그 요청을 별도 turn으로 계산할 수 없습니다. 전체 예산이 필요한 확장 구성 요소에는 gateway 사용 계약 테스트를 추가하세요.

no-progress guard는 같은 Engine/phase/logical step에서 같은 content, finish reason, Tool 이름과 arguments를 연속 관찰합니다. provider call ID, usage와 provider metadata는 비교에서 제외합니다. 기본 threshold 3은 세 번째 동일 관찰에서 종료합니다.

모든 성공 Tool 호출이 streak를 초기화하는 것은 아닙니다. Tool 이름, 실제 검증된 인자와 정규화된 결과로 만든 run-salted fingerprint가 직전 성공 outcome과 다를 때만 새 진전입니다. 같은 성공 outcome을 call ID, 실행 시간 또는 retry 횟수만 바꿔 반복하면 초기화하지 않습니다. 새 record를 소비한 각 `memory_summary` batch와 Plan step commit은 명시적인 진전으로 처리합니다.

guard checkpoint에는 숫자 카운터, run별 무작위 salt와 HMAC-SHA-256 관찰 digest만 저장하며, 성공 Tool outcome도 raw payload가 아닌 run-salted fingerprint로 비교합니다. 원본 상태, 응답, Tool 인자·결과, usage와 provider metadata는 guard 상태에 저장하지 않습니다.

새 terminal 결과를 처리하도록 enum 분기를 갱신하세요.

```python
from moduagent import FinishReason


if result.finish_reason is FinishReason.MAX_MODEL_TURNS:
    alert("model turn budget exhausted")
elif result.finish_reason is FinishReason.NO_PROGRESS:
    alert("model produced no semantic progress")
```

두 결과는 모두 `RUN_FAILED`이고 `error_summary.retryable=False`,
`error_summary.resumable=False`입니다.

## Checkpoint와 rolling 배포

0.5는 outer checkpoint schema v4와 Engine state version을 유지하고 model guard의 digest·카운터 상태를 additive compatibility state로 저장합니다. 0.5에서 생성한 checkpoint를 resume하면 이미 소비한 model turn과 no-progress streak가 이어집니다.

`CheckpointStore`가 있으면 첫 호출과 각 provider retry의 turn 예약을 실제 provider I/O 직전 `before_model` durable boundary에 기록합니다. 저장이 실패하면 provider를 호출하지 않습니다. provider 요청 중 hard crash가 발생해도 resume은 예약된 turn을 다시 사용할 수 없습니다. 이 안전성 때문에 모델 호출마다 checkpoint write가 추가되므로 배포 전 저장소 latency와 처리량을 측정하세요.

새 guard 한도는 Agent fingerprint에 포함됩니다. 따라서 일반적인 0.4 active checkpoint는 0.5 worker의 fingerprint 검사에서 fail closed하며, 과거 호출 횟수를 알 수 없는 상태로 카운터를 0부터 다시 시작하지 않습니다. 배포 전에 0.4 worker에서 active run을 drain하거나 버전별 worker/store namespace에 고정하세요. 예외적으로 `legacy-unbound` checkpoint나 custom 호환 경로를 운영한다면 실제 fixture로 재개 정책을 별도 검증해야 합니다. 반대로 0.4 worker도 0.5가 기록한 예산 상태를 강제하지 못하므로 0.5 checkpoint를 0.4 worker로 되돌리지 마세요.

다음 항목을 배포 전에 시험하세요.

- provider retry 한 번마다 model turn이 증가함
- 각 provider 호출 직전 예약된 turn이 checkpoint에 먼저 저장되고 저장 실패 시 호출되지 않음
- `max_model_turns`를 넘는 provider 요청이 실제로 전송되지 않음
- call ID만 바꾼 동일 Tool 요청이 no-progress로 종료됨
- 새로운 성공 Tool outcome은 streak를 초기화하지만 동일 성공 outcome 반복은 초기화하지 않음
- 각 성공 `memory_summary` batch와 step commit 후 streak가 초기화됨
- checkpoint resume 후 model turn과 streak가 이어짐
- `max_model_turns`와 `no_progress` terminal checkpoint가 `resumable=False`임
- 새 finish reason을 event/metric/alert consumer가 실패로 처리함
- malformed Plan, Tool arguments JSON과 `StepResult`가 반복 요청되지 않음
- `AgentRunError`와 진단 event에 원문 payload가 나타나지 않음

## 0.5의 비목표

0.5 Quick API는 도메인 Recipe, DB schema 추론, 자동 SQL 안전 판정, 고정 Workflow DSL이나 Tool 안전성 자동 추론을 추가하지 않습니다. 애플리케이션은 Tool, 프롬프트, 데이터 계약, 권한과 복구 안전성을 계속 직접 정의합니다.
