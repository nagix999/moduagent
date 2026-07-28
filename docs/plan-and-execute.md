# Plan-and-Execute

상태: ModuAgent 0.3.0의 `PlanAndExecutePolicy`는 검증된 단계 결과와 공개 최종 응답을 분리하는 strict 상태 머신이다. `Agent`의 기본 Policy는 계속 `StandardDecisionPolicy`이며, Plan-and-Execute가 필요한 Agent에 명시적으로 설정한다.

## 핵심 계약

한 번의 정상 실행은 다음 경계를 따른다.

```text
PLAN
  → STEP_PREPARE
  → ACT_TOOL       # 현재 단계에 Tool이 있을 때
  → STEP_RESULT    # Tool 없음 + StepResult schema
  → STEP_VALIDATE
      ↳ RETRY_STEP
      ↳ REPLAN
      ↳ STEP_COMMIT
  → VERIFY
  → FINALIZE
  → DONE
```

구현상 `ACT_TOOL`과 `STEP_RESULT`는 모두 `RunPhase.ACT`이며 `ExecutionState.awaiting_step_result`로 구분된다. Tool이 없는 단계는 `ACT_TOOL`을 건너뛰고 바로 `STEP_RESULT`를 생성한다.

다음 불변 조건은 런타임과 Policy가 함께 강제한다.

- PLAN은 최종 답변 단계를 만들지 않고 안정적인 `step_id`와 완료 조건을 가진 실행 단계만 만든다.
- ACT는 현재 단계 하나만 실행한다. 다른 단계나 공개 최종 답변을 만들지 않는다.
- Tool 요청에는 Tool schema만, `StepResult` 요청에는 출력 schema만 전달한다.
- `StepResult`가 schema와 완료 조건 검증을 통과해야만 단계를 커밋한다.
- 커밋된 결과는 대화 메시지가 아니라 `ExecutionState.committed_results`에 보관한다.
- 완료된 계획만 VERIFY를 통과하며 FINALIZE에는 Tool을 전달하지 않는다.
- `TextOutputCodec`과 `PydanticOutputCodec` 모두 별도의 FINALIZE 호출을 정상 실행당 한 번 수행한다.
- 공개 대화에는 사용자 입력과 FINALIZE 응답만 저장한다. ACT 응답과 단계 지시는 저장하지 않는다.
- `DONE` 이후에는 모델이나 Tool을 다시 호출하지 않는다.

기본 `StepValidator`는 단계 ID, terminal status, 완료 조건별 비어 있지 않은 근거 수를 결정적으로 검사한다. 사실의 의미적 진위를 판정하지는 않으므로 더 강한 검증이 필요하면 `StepValidator`를 확장해 주입한다.

## 기본 사용

```python
from moduagent import (
    Agent,
    AgentConfig,
    InMemoryCheckpointStore,
    LLMPlanGenerator,
    PlanAndExecutePolicy,
    RunLimits,
)

planning_agent = Agent(
    config=AgentConfig(
        name="planning-agent",
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
    tools=tools,
    decision_policy=PlanAndExecutePolicy(
        plan_generator=LLMPlanGenerator(model, max_steps=4),
    ),
    checkpoint_store=InMemoryCheckpointStore(),
)

result = await planning_agent.run(
    "요청을 여러 검증 가능한 단계로 수행해줘",
    session_id="plan-session",
)
```

`LLMPlanGenerator.max_steps`는 모델이 반환한 계획을 제한하고, `RunLimits.max_steps`는 custom `PlanGenerator`를 포함한 최종 계획의 단계 수를 검사한다. strict Plan-and-Execute에서 `max_steps`는 모델 호출 횟수가 아니라 계획 단계 수다.

추가 제한은 서로 독립적이다.

| 설정 | 의미 |
|---|---|
| `max_step_attempts` | schema 오류 또는 검증 실패로 같은 단계를 다시 만들 수 있는 최대 시도 수 |
| `max_replans` | 미완료 계획을 수정할 수 있는 최대 횟수 |
| `max_tool_calls` | 한 run에서 실행할 business Tool 호출 수 |
| `timeout_seconds` | PLAN, 모든 ACT/검증, 재계획, FINALIZE, 저장 작업을 포함한 전체 시간 |

Tool 왕복 자체는 단계 시도 횟수를 증가시키지 않는다. Tool 실패 시 `revise_on_tool_failure=True`이면 완료된 단계와 결과를 보존한 채 미완료 범위를 재계획하며 `max_replans`를 소비한다.

한 계획 단계는 모델이 한 응답에서 선택한 Tool call batch를 실행한 뒤 `STEP_RESULT`로 넘어간다. 두 번째 Tool의 인자를 첫 번째 Tool 결과로 정해야 하는 순차 작업은 Planner가 각각 독립적으로 검증 가능한 의존 단계로 나눠야 한다. 같은 batch의 여러 호출은 `parallel_tool_calls` 설정에 따라 병렬 또는 순차 실행할 수 있지만 서로의 결과를 인자로 참조할 수는 없다.

### `finalization_mode`

`AgentConfig.finalization_mode`의 기본값은 `structured_only`다. PDF의 목표안은 `always`를 제안하지만, 일반 `StandardDecisionPolicy`를 쓰던 0.2 애플리케이션의 모델 호출 수를 유지하기 위해 0.3.0 기본값은 호환적으로 선택했다.

| 값 | `StandardDecisionPolicy` |
|---|---|
| `always` | 텍스트와 구조화 출력 모두 실행 응답 뒤에 Tool 없는 finalizer를 호출 |
| `structured_only` | Tool과 구조화 출력이 함께 있을 때만 Tool 없는 finalizer를 호출 |
| `disabled` | 분리 finalizer를 사용하지 않음 |

일반 finalizer가 생성한 실제 공급자 원문은 공개 assistant 메시지로 `ConversationStore`에 저장된다. `disabled`에서 Tool과 출력 schema를 같은 요청에 넣을 수 있는지는 모델 adapter의 제약을 따른다. `VLLMClient`는 이 조합을 거부한다.

strict `PlanAndExecutePolicy`는 FINALIZE가 불변 조건이므로 `always`와 `structured_only`에서 모두 항상 FINALIZE한다. `disabled`와 함께 구성하면 `Agent` 생성 시 `ValueError`가 발생한다.

## 계획과 단계 결과

`PlanStep`은 예상 결과와 실제 결과를 분리한다.

```python
from moduagent import Plan, PlanStep

plan = Plan(
    [
        PlanStep(
            step_id="S1",
            objective="필요한 사실을 확인한다.",
            completion_criteria=["확인 근거가 하나 이상 있다."],
            expected_output="검증 가능한 사실",
            dependencies=[],
            allowed_tools=["lookup"],
        )
    ]
)
```

ACT의 내부 출력은 다음 `StepResult` 필드만 허용한다.

- `step_id`
- `status`: `completed`, `blocked`, `failed`
- `facts`
- `artifacts`
- `uncertainties`
- `missing_inputs`
- `completion_evidence`

`extra="forbid"`가 적용되므로 `final_answer` 같은 단계 밖의 필드를 넣으면 커밋되지 않는다. 커밋된 결과는 canonical JSON의 SHA-256 `result_ref`와 함께 저장된다. 재계획은 기존 완료 단계의 ID, 상태, 결과 참조와 `committed_results`를 보존한다.

## vLLM과 출력 schema

vLLM을 포함한 일부 OpenAI 호환 서버는 한 요청에 Tool Calling과 구조화 출력 schema를 함께 보내면 Tool을 호출하지 않거나 요청을 거부한다. 0.3.0은 두 계약을 요청 수준에서 분리한다.

```text
ACT_TOOL
  tools=(현재 단계의 허용 Tool)
  output_schema=None

STEP_RESULT
  tools=()
  output_schema=StepResult schema

FINALIZE
  tools=()
  output_schema=공개 Pydantic schema 또는 None
```

따라서 `PydanticOutputCodec`은 Tool 선택을 방해하지 않는다. `VLLMClient`도 `tools`와 `output_schema`가 동시에 들어온 요청을 명시적으로 거부해 잘못된 조합을 조기에 발견한다.

## 공개 스트림과 진단 스트림

strict Plan-and-Execute에서 ACT token은 `STEP_MODEL_DELTA`라는 internal 이벤트다. 기본 `Agent.stream()`은 이를 숨기고 FINALIZE token을 `FINAL_DELTA`로 공개한다.

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
```

계획, 단계, Tool, 검증과 내부 token을 진단할 때만 `stream_all()`을 사용한다.

```python
async for event in planning_agent.stream_all("요청을 수행해줘"):
    print(event.visibility.value, event.type.value, event.data)
```

`AgentConfig(stream_visibility="all")`도 `stream()`에 internal 이벤트를 포함하지만, 사용자 응답 UI와 진단 소비자를 코드 수준에서 분리하기 위해 `stream_all()`을 명시적으로 호출하는 방식을 권장한다. 일반 `StandardDecisionPolicy`의 직접 응답 token은 기존처럼 `MODEL_DELTA`다.

## Text와 Pydantic 최종화

텍스트 출력도 ACT의 마지막 문장을 재사용하지 않고 별도 FINALIZE를 수행한다.

```python
from pydantic import BaseModel

from moduagent import PydanticOutputCodec


class PublicResult(BaseModel):
    answer: str


structured_agent = Agent(
    config=config,
    model=model,
    tools=tools,
    decision_policy=PlanAndExecutePolicy(LLMPlanGenerator(model)),
    output_codec=PydanticOutputCodec(PublicResult),
)
```

FINALIZE는 원래 사용자 목표와 검증된 `committed_results`만 받고 Tool 없이 실행된다. Pydantic 응답은 전체 schema 검증이 끝난 뒤에만 공개 delta와 대화 저장이 진행된다. 성공 결과의 `AgentResult.output`은 `TextOutputCodec`이면 문자열, `PydanticOutputCodec`이면 해당 모델 객체다.

## Skill 단계 범위

0.3.0 Skill은 `SKILL.md`의 `applies-to` 확장으로 지침을 적용할 단계를 제한할 수 있다.

```yaml
---
name: controlled-procedure
description: 검증된 절차가 필요한 요청에 사용한다.
applies-to:
  - plan
  - act
allowed-tools: lookup
---
```

지원 값은 `plan`, `act`, `finalize`다. 필드를 생략하면 세 단계 모두에 적용된다. PLAN 전용 지침은 ACT와 FINALIZE에, FINALIZE 전용 지침은 ACT에 전달되지 않는다. `VERIFY`는 결정적 런타임 검사 단계이므로 Skill 적용 대상이 아니다.

`applies-to`는 지침의 prompt 범위만 제한한다. `allowed-tools`, Tool 등록 범위, `ToolAuthorizer`의 권한 검사는 별도로 적용된다. FINALIZE에는 어떤 Skill을 적용하더라도 Tool이 제공되지 않는다.

## Checkpoint와 resume

Checkpoint schema v3는 다음 strict 상태를 저장한다.

- phase, 계획 버전, 현재 단계
- 커밋된 `StepResult`와 결과 참조
- 단계별 시도 횟수와 재계획 횟수
- pending 결과와 내부 Tool protocol 메시지
- FINALIZE 원문, 저장 여부, 공개 방출 여부와 호출 횟수
- phase-scoped Skill 활성화 상태

v3 checkpoint를 재개하면 완료 단계는 다시 실행하지 않고 현재 ACT 또는 FINALIZE 경계에서 계속한다. v1과 v2 payload도 읽을 수 있지만 strict `ExecutionState`가 없으므로 진행 중인 0.2 Plan-and-Execute를 단계 단위로 그대로 이어 주는 마이그레이션은 제공하지 않는다. 배포 전 진행 중인 run을 끝내거나, 재실행이 안전한 요청만 새 strict 계획으로 시작해야 한다.

실패한 checkpoint에는 Tool 결과나 Skill resource 원문이 일시적으로 포함될 수 있다. 운영 저장소는 암호화, 접근 제어와 짧은 TTL을 적용해야 한다.

새 strict run의 Planner는 같은 session에 저장된 최근 공개 대화를 함께 보므로 “그 결과를 요약해줘” 같은 후속 요청은 이전 FINALIZE 응답을 참조할 수 있다. `LLMPlanGenerator(history_limit=8)`의 기본값은 최근 8개 메시지이며 `0`으로 끌 수 있다. ACT에는 과거 대화 전체를 다시 전달하지 않고 Planner가 만든 현재 단계와 필요한 선행 결과만 전달한다.

## 0.2 마이그레이션

0.3.0의 `PlanAndExecutePolicy`는 strict 동작이 기본이다. 0.2의 암묵적 단계 완료 동작이 잠시 필요하면 이름을 명시적으로 바꾼다.

```python
from moduagent import LegacyPlanAndExecutePolicy

legacy_policy = LegacyPlanAndExecutePolicy(
    plan_generator=LLMPlanGenerator(model),
)
```

`LegacyPlanAndExecutePolicy`는 생성 시 `DeprecationWarning`을 내며, 검증된 `StepResult`, ACT 비영속화, 별도 텍스트 FINALIZE와 공개/internal 스트림 분리를 보장하지 않는다. 신규 코드는 사용하지 않는다.

strict 전환 시 다음 차이를 반영한다.

- 계획 단계마다 Tool 선택 요청과 `StepResult` 요청이 분리될 수 있다.
- 텍스트와 구조화 출력 모두 FINALIZE 모델 호출이 추가된다.
- Planner가 만든 `allowed_tools`는 실제 등록된 Tool 이름이어야 한다.
- 완료 근거가 부족한 출력은 자동 완료되지 않고 제한 안에서 재시도 또는 재계획된다.
- 대화 기록에서 과거 ACT 초안이 사라지고 공개 최종 응답만 남는다.

## 전달 보장과 Tool 부작용

`ExecutionState.final_response`, `final_persisted`, `final_emitted`와 `RunPhase.DONE`은 checkpoint resume 중 FINALIZE 모델 호출과 공개 방출의 중복을 막는다. 이 범위는 동일 run과 신뢰 가능한 `CheckpointStore` 안의 런타임 중복 억제다.

이미 `DONE`인 run을 명시적으로 resume하면 저장된 결과를 담은 terminal `RUN_COMPLETED`를 다시 관찰할 수 있지만 FINALIZE 모델과 `FINAL_DELTA`는 다시 실행·방출하지 않는다. 소비자는 `run_id`를 terminal event의 중복 제거 키로 사용할 수 있다.

다음은 프레임워크 단독으로 보장하지 않는다.

- 프로세스가 checkpoint 기록과 클라이언트 수신 사이에서 중단될 때의 end-to-end exactly-once 이벤트 전달
- 네트워크 재시도, worker 중복 실행 또는 외부 시스템 장애 중 Tool 부작용의 exactly-once 실행
- 여러 프로세스가 같은 run을 동시에 소유하지 못하게 하는 분산 lease

최종 이벤트가 반드시 한 번 전달되어야 하는 시스템은 checkpoint 전이와 같은 내구성 경계에 outbox record를 기록하고, 안정적인 event ID와 소비자 idempotency를 사용한다. 변경 Tool은 run ID와 tool call ID 등으로 idempotency key를 만들고 대상 시스템에서 중복을 제거해야 한다. `@function_tool(idempotent=True)`는 프레임워크 재시도를 허용한다는 뜻이며 exactly-once 보장이 아니다.
