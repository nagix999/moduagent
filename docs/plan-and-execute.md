# Plan-and-Execute

상태: ModuAgent 0.5의 `PlanExecutionEngine`은 검증된 단계 결과와 공개 최종 응답을 분리하는 strict 상태 머신이다. 기존 `PlanAndExecutePolicy`는 호환 facade로 유지된다. `Agent`의 기본 실행은 계속 Standard이며, Plan-and-Execute가 필요한 Agent에 명시적으로 설정한다.

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

같은 모델로 계획과 실행을 수행하는 일반적인 경우에는 Quick API를 사용할 수 있다.

```python
from moduagent import Agent, RunLimits

planning_agent = Agent.create(
    name="planning-agent",
    instructions="검증된 단계 결과만 사용해 간결하게 답한다.",
    model=model,
    tools=tools,
    execution="plan",
    limits=RunLimits(
        max_steps=4,
        max_step_attempts=2,
        max_replans=1,
        max_tool_calls=8,
        max_model_turns=32,
        no_progress_model_turn_threshold=3,
        timeout_seconds=120,
    ),
)

result = await planning_agent.run(
    "요청을 여러 검증 가능한 단계로 수행해줘",
    session_id="plan-session",
)
```

`execution="plan"`은 같은 `model`과 `limits.max_steps`를 사용하는 `LLMPlanGenerator`와 `PlanExecutionProfile`로 변환된다. 별도 planning model, custom `PlanGenerator`, checkpoint 또는 상세 복구 설정이 필요하면 다음 명시적 API를 사용한다.

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
| `max_tool_repair_attempts` | 모델이 실패한 Tool call의 인자를 교정해 새 call을 만들 수 있는 최대 횟수 |
| `max_replans` | 미완료 계획을 수정할 수 있는 최대 횟수 |
| `max_tool_calls` | 한 run에서 실행할 business Tool 호출 수 |
| `max_model_turns` | provider retry를 포함한 실제 모델 시도 수. 기본값 32 |
| `no_progress_model_turn_threshold` | 동일 의미 상태·동일 의미 응답의 연속 관찰 threshold. 기본값 3 |
| `timeout_seconds` | PLAN, 모든 ACT/검증, 재계획, FINALIZE, 저장 작업을 포함한 전체 시간 |

Tool 왕복 자체는 단계 시도 횟수를 증가시키지 않는다. Tool 실패 시 `revise_on_tool_failure=True`이면 완료된 단계와 결과를 보존한 채 미완료 범위를 재계획하며 `max_replans`를 소비한다.

## 모델 retry와 실행 loop 차단

`RetryConfig.max_attempts`의 각 provider 시도는 성공 여부와 관계없이 `max_model_turns`를 하나씩 소비한다. PLAN, ACT_TOOL, `StepResult`, 재계획과 FINALIZE 호출도 같은 run 예산을 공유한다. streaming에서는 chunk가 아니라 stream 시도 하나를 한 turn으로 센다. 다음 시도가 상한을 넘기기 전에 `FinishReason.MAX_MODEL_TURNS`와 `max_model_turns_exceeded`로 종료한다.

`CheckpointStore`가 있으면 각 turn은 실제 provider 호출 직전 `before_model` durable boundary에 예약된다. retry 시도도 각각 먼저 저장되며, 저장이 실패하면 provider I/O를 시작하지 않는다. provider 요청 중 hard crash가 발생해도 resume은 저장된 turn 수에서 이어진다.

provider retry는 timeout·HTTP 408, connection/network 오류와 HTTP 5xx만 허용한다. HTTP 4xx/429, 잘못된 요청, client contract 오류, JSON decoding과 protocol 오류는 같은 요청을 반복해도 수정되지 않으므로 즉시 실패한다. stream이 일부 delta를 이미 방출한 뒤 끊기면 중복 출력을 방지하기 위해 transient 오류도 재시도하지 않는다. 전체 분류표는 [Operations](operations.md)를 참고한다.

`no_progress_model_turn_threshold`는 동일한 Engine ID, phase와 logical step ID에서 동일한 의미 응답이 반복되는 경우를 센다. 의미 응답에는 content, finish reason, Tool 이름과 arguments가 포함되며 provider call ID, usage와 provider metadata는 제외된다. 기본값 3은 첫 응답을 1로 계산하여 세 번째 동일 응답에서 `FinishReason.NO_PROGRESS`와 `model_no_progress`로 종료한다.

성공한 Tool outcome은 Tool 이름, 실제 검증된 인자와 정규화된 결과의 run-salted fingerprint가 직전 성공 outcome과 다를 때만 진전이다. 같은 Tool이 같은 인자와 같은 결과로 성공하는 loop는 streak를 초기화하지 않는다. provider call ID, 실행 시간이나 retry 횟수만 바뀐 경우도 동일 outcome이다. 반면 새 source record를 소비한 각 `memory_summary` batch와 Plan step commit은 명시적인 진전이므로 streak를 초기화한다.

비교를 위해 원본 프롬프트, 출력이나 Tool 인자·결과를 보관하지 않는다. model guard checkpoint에는 run별 무작위 salt, HMAC-SHA-256 관찰 digest와 숫자 카운터만 저장하고, 성공 outcome도 raw payload가 아닌 run-salted fingerprint로 비교한다. resume 후에도 같은 run의 예산과 streak를 이어간다. `MAX_MODEL_TURNS`와 `NO_PROGRESS` terminal은 모두 `retryable=False`, `resumable=False`다.

Planner 응답이 비어 있거나 잘못된 JSON·schema·Tool 이름을 포함하면 catch-all 단계를 생성하지 않고 `ModelProtocolError`로 즉시 실패한다. FINALIZE가 Tool call, 빈 응답 또는 잘린 응답을 반환한 경우도 protocol 실패다. 이러한 실패는 provider retry 대상이 아니다.

## Tool 실패 복구

Tool 실패에는 서로 다른 세 가지 복구 경계가 있다.

| 경계 | 설정 | 동작 |
|---|---|---|
| 동일-call retry | `RetryConfig`, `idempotent=True`, `retryable=True` | 같은 call ID의 validation 완료 인자를 그대로 다시 실행 |
| corrected-arguments repair | `ToolFailureRecoveryConfig`, `repair_safe=True`, `max_tool_repair_attempts` | 모델이 같은 Tool에 새 call ID와 교정된 인자를 생성 |
| replan | `max_replans`, recovery의 `fallback="replan"` | 완료 결과를 보존하고 미완료 계획을 수정 |

corrected-arguments repair는 기본적으로 꺼져 있다. `PlanAndExecutePolicy.tool_failure_recovery`를 생략하면 0.3.1과 같은 `revise_on_tool_failure` 동작을 유지한다. 활성화할 때는 Tool이 교정된 다른 인자로 다시 실행되어도 안전한지 검토하고 `repair_safe=True`를 명시한다. `require_repair_safe=True`가 기본이므로 선언되지 않은 Tool은 repair하지 않고 fallback한다.

```python
from moduagent import (
    Agent,
    AgentConfig,
    LLMPlanGenerator,
    PlanAndExecutePolicy,
    RetryConfig,
    RunLimits,
    ToolError,
    ToolErrorType,
    ToolFailure,
    ToolFailureRecoveryConfig,
    ToolRecoveryAction,
    function_tool,
)


def map_filter_error(exc: Exception) -> ToolError | None:
    # 예상하고 안전하게 설명할 수 있는 오류만 모델 repair 대상으로 분류한다.
    if not isinstance(exc, ValueError):
        return None
    return ToolError(
        type=ToolErrorType.EXECUTION_ERROR,
        reason="invalid_filter",
        message="The filter expression is invalid; correct the Tool arguments.",
        retryable=False,
        recovery=ToolRecoveryAction.REPAIR_CALL,
    )


@function_tool(
    idempotent=True,
    repair_safe=True,
    error_mapper=map_filter_error,
)
def search_catalog(filter: str, limit: int = 100) -> dict:
    return catalog.search(filter=filter, limit=limit)


agent = Agent(
    config=AgentConfig(
        name="catalog-agent",
        instructions="검증된 catalog 결과만 사용한다.",
        retry=RetryConfig(max_attempts=2),
        limits=RunLimits(
            max_tool_repair_attempts=1,
            max_replans=1,
            max_tool_calls=8,
        ),
    ),
    model=model,
    tools=[search_catalog],
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

`error_mapper`의 계약은 `Callable[[Exception], ToolError | None]`이다. 알려진 오류에는 `ToolError.recovery`로 `RETRY_CALL`, `REPAIR_CALL`, `REPLAN`, `FAIL` 중 하나를 지정하고, 분류하지 않은 예외에는 `None`을 반환해 generic 처리로 되돌린다. Tool 내부에서 이미 typed failure를 알 수 있으면 `raise ToolFailure(ToolError(...))`를 사용해 mapper 없이 전달할 수 있다. 두 경로 모두 raw exception을 전달하는 통로가 아니라 모델 공개용 오류를 구성하는 계약이다.

| `ToolRecoveryAction` | 의미 |
|---|---|
| `RETRY_CALL` | `idempotent=True`, `retryable=True`, 남은 `RetryConfig` 시도가 모두 충족되면 동일 call과 동일 인자를 executor가 다시 실행 |
| `REPAIR_CALL` | opt-in recovery가 동일 Tool 1개에 교정된 새 인자와 새 call ID를 요청 |
| `REPLAN` | call repair 없이 미완료 계획을 즉시 수정 |
| `FAIL` | 복구하지 않고 run을 terminal 실패로 전이 |

`RETRY_CALL`은 같은 인자의 executor retry이고 `REPAIR_CALL`은 모델이 다른 인자를 만들도록 요청하는 복구다. 단, 취소할 수 없는 동기 Tool timeout은 worker가 백그라운드에서 계속 실행되므로 동일-call retry를 막고 `FAIL`로 분류한다. DB driver의 statement timeout·취소 완료를 보장하는 통합만 `function_tool(timeout_retry_safe=True)`를 명시할 수 있다.

`REPAIR_CALL` 응답은 실행 전에 정확히 하나의 동일 Tool, 이전에 쓰지 않은 call ID, 이전과 다른 canonical JSON arguments인지 검사한다. 실행기는 validation·형 변환·기본값 적용 후의 유효 arguments hash도 비교하므로 `1`을 `1.0`으로 바꾸는 식의 표현상 차이는 Tool 본체를 다시 실행하지 않는다. 키 순서만 바꾼 동일 인자, 다른 Tool, 복수 호출도 실행하지 않고 fallback한다. 단계 검증의 `max_step_attempts`는 소비하지 않지만 `max_tool_repair_attempts`, 전체 `max_tool_calls`와 run timeout은 적용받는다. repair가 불가능하거나 한도를 소진하면 config의 `fallback`에 따라 `replan` 또는 `fail`로 전이한다. 여러 Tool batch가 부분 성공한 경우에는 이미 발생한 side effect의 중복 실행을 막기 위해 fallback 설정과 무관하게 terminal 실패한다.

이 기능은 DB 전용이 아니다. 검색 문법, 변환 옵션, 파일 선택 조건, 외부 API parameter처럼 모델이 인자를 안전하게 교정할 수 있는 Tool에 적용한다. 반대로 결제·삭제·전송 같은 변경 Tool은 `repair_safe=True`로 선언하지 않는 것이 기본이며, 원 시스템의 idempotency와 중복 방지 없이 자동 복구를 켜면 안 된다.

`feedback_mode="type_only"`가 기본이며 strict repair prompt에는 오류 type과 `reason`만 전달한다. `"safe_message"`는 호출자가 모델 공개에 안전하다고 보증한 `ToolError.message`를 제어문자 제거·길이 제한 후 추가한다. 이는 secret redaction 기능이 아니다. strict 실패 Tool 메시지는 `message`와 `details`를 항상 제외하지만, 공개 `ToolResult.model_content()`와 non-strict 경로의 호환성은 유지되므로 `error_mapper`와 `ToolFailure`에는 원본 예외, SQL, 접속 문자열, 인증정보, 고객 데이터, 내부 경로·schema를 넣지 않는다. `TOOL_REPAIR_SCHEDULED`, `TOOL_REPAIR_EXHAUSTED`, `STEP_FAILED`는 internal 이벤트이며 공개 결과에는 원본 오류나 Tool 결과를 추가하지 않는다.

공개 `tool_trace.error`에는 `type`, `retryable`과 선택적인 `reason`, `recovery`가 기록된다. 교정된 호출에는 `recovery_of_call_id`가 붙는다. repair가 한 번 이상 예약되면 `plan_usage.tool_repairs`가 추가되고, terminal 복구 실패에는 정제된 `metadata.failure`이 추가된다. 두 metadata 모두 원본 예외 메시지와 Tool 결과는 저장하지 않는다.

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

0.5에서는 schema-only `StepResult`가 JSON 문법 오류, 빈 응답, 허용되지 않은 필드 또는 Pydantic schema 불일치를 가지면 현재 단계를 즉시 `failed`로 전이한다. 이 경우 `validation_code="step_result_schema_invalid"`, `validation_location="step_result"`가 기록되며 같은 응답을 `max_step_attempts`만큼 반복 생성하지 않는다. `max_step_attempts`는 schema를 통과한 결과의 완료 근거 검증처럼 다시 시도해 의미가 있는 단계 검증에 사용한다.

검증 재시도, Tool 실패 복구 또는 재계획 한도를 소진해 Policy가 terminal 실패를 결정하면 현재 `PlanStep.status`도 `failed`로 전이한다. 다음 단계는 실행되지 않았으므로 `pending`으로 남는다. 일시적인 transport 오류나 전체 run timeout처럼 checkpoint에서 재개 가능한 중단은 `in_progress`로 남을 수 있지만, 이는 동기 DB 조회나 Tool 함수가 백그라운드에서 계속 실행 중이라는 뜻이 아니다.

## 허용 Tool과 실제 Tool 호출

`PlanStep.allowed_tools`는 Planner가 단계별로 정한 허용 범위다. 목록에 Tool이 있다는 사실은 해당 Tool이 실제로 호출됐다는 뜻이 아니다. 실제 호출의 운영 감사 요약은 완료 또는 실패 결과의 `AgentResult.metadata["tool_trace"]`에 순서대로 기록된다.

`AgentConfig.tool_trace_mode`는 다음 세 가지 모드를 지원한다.

| 값 | 저장 내용 |
|---|---|
| `off` | `tool_trace`를 만들지 않음 |
| `summary` | 기본값. `step_id`, `call_id`, `tool_name`, `success`, `attempts`, `duration_seconds`, 정제된 `error`만 저장 |
| `arguments` | `summary`에 민감한 키를 재귀적으로 마스킹한 `arguments`와 `arguments_source`를 추가 |

`arguments_source="validated"`는 validation·형 변환·기본값 적용 후 Tool에 실제 전달된 인자이고, `"requested"`는 validation 또는 실행 전에 거부된 호출의 모델 요청 인자다. `error`는 원본 메시지 대신 `type`, `retryable`과 선택적인 `reason`, `recovery`만 가진다. trace 전체 크기도 제한되며 Tool 결과 값은 어떤 모드에도 넣지 않는다. live `AgentResult`는 요청한 trace mode를 유지하지만 v4 checkpoint에는 raw arguments 대신 `arguments_fingerprint`만 저장한다. resume 후 trace도 fingerprint projection으로 이어진다.

```python
agent = Agent(
    config=AgentConfig(
        name="audited-agent",
        instructions="검증된 결과만 사용한다.",
        tool_trace_mode="arguments",
    ),
    model=model,
    tools=tools,
    decision_policy=PlanAndExecutePolicy(LLMPlanGenerator(model)),
)

result = await agent.run("조회해줘")
for call in result.metadata.get("tool_trace", []):
    print(call["step_id"], call["tool_name"], call.get("arguments"))
```

인자도 업무 데이터가 될 수 있으므로 일반 운영은 기본 `summary`를 권장한다. `stream_all()`도 안전하게 투영된 내부 이벤트만 제공하며 원본 Tool 결과, 원본 예외, provider metadata를 공개하지 않는다. 원문 진단이 꼭 필요하면 프레임워크 외부의 별도 접근 통제 sink를 사용한다.

Tool이 pandas `DataFrame` 또는 그 밖의 표 형식 값을 반환하면 런타임은 JSON-safe record 구조로 정규화한 뒤 모델과 진단 경계에 전달한다. 이 변환은 `Timestamp`, NumPy scalar, 결측값처럼 일반 JSON encoder가 직접 처리하기 어려운 셀도 JSON 호환 값으로 바꾼다. 다만 전체 DataFrame을 반환하면 prompt가 매우 커질 수 있으므로 조회 Tool 자체에서 필요한 열과 행을 제한하는 것이 좋다.

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

`stream()`은 `MODEL_DELTA`, `FINAL_DELTA`, terminal 결과만 전달한다. `stream_all()`은 phase, Tool 식별자, 복구 분류, checkpoint boundary를 추가하지만 모든 event data는 JSON-safe projection이다.

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

0.4의 `RunSnapshot` outer schema v4는 공통 상태와 독립적으로 versioned Plan Engine state를 저장한다.

- phase, 계획 버전, 현재 단계
- 커밋된 `StepResult`와 결과 참조
- 단계별 시도 횟수와 재계획 횟수
- Tool repair 횟수, pending repair 대상, 사용된 call ID와 원문 없는 arguments hash
- 크기와 민감 정보가 정제된 실제 `tool_trace`
- pending 결과와 원문을 제외한 Tool 복구 상태
- FINALIZE 원문, 저장 여부, 공개 방출 여부와 호출 횟수
- phase-scoped Skill 활성화 상태
- 실제 model turn 수, no-progress streak, run별 salt와 원문 없는 HMAC-SHA-256 관찰 digest

0.4는 v1-v3 checkpoint를 copy-on-migrate 방식으로 읽고 검증된 v4 snapshot으로 변환한다. v4를 v3로 downgrade하지 않으므로 rolling 배포에서는 store/key namespace를 버전별로 분리하거나 진행 중 run을 0.4 worker에 고정한다.

v3 checkpoint를 변환해 재개하면 완료 단계와 원래 실패한 repair call을 다시 실행하지 않는다. 의미가 불명확한 partial-success 또는 실행 중 Tool 상태는 자동 replay하지 않고 `manual_required`로 fail closed한다. 자세한 절차는 [0.4 마이그레이션](migration-0.4.md)을 참고한다. 0.4 애플리케이션을 0.5로 올릴 때의 additive API와 새 terminal reason 대응은 [0.5 마이그레이션](migration-0.5.md)을 참고한다.

checkpoint에는 모델이 만든 원본 Tool 인자와 결과를 복제하지 않고 call identity와 arguments fingerprint, 성공 outcome의 run-salted fingerprint, 안전한 실패 code만 보존한다. conversation과 checkpoint 저장소에는 암호화, 접근 제어와 TTL을 적용한다.

모든 모델 시도는 provider I/O 직전 `before_model` boundary에서 turn을 durable하게 예약한다. 이 저장이 성공해야 provider를 호출하며, retry도 동일하다. 따라서 호출 도중 프로세스가 종료되어도 resume이 이미 소비한 turn을 다시 제공하지 않는다. model guard가 `MAX_MODEL_TURNS` 또는 `NO_PROGRESS`로 terminal 전이한 checkpoint는 자동 resume 대상이 아니다.

동기 Tool은 event loop를 막지 않도록 worker thread에서 실행된다. Python thread는 강제로 안전하게 취소할 수 없으므로 `pd.read_sql` 같은 blocking 호출은 Tool 또는 run timeout 결과가 반환된 뒤에도 해당 worker에서 끝날 때까지 계속될 수 있다. 0.4는 Tool 본체 호출 전에 `tool_invocation_pending` checkpoint를 `manual_required`로 저장하고, 저장 실패 시 Tool을 호출하지 않는다. 이 상태는 자동 resume되지 않는다. DB driver의 query timeout, connection timeout과 서버 측 statement timeout도 함께 설정해야 한다.

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
