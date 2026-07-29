# ModuAgent 0.4 마이그레이션

이 문서는 0.3.2 애플리케이션과 checkpoint를 0.4로 옮기는 방법을 설명합니다.

## 요약

0.4는 실행 내부 구조를 분리하지만 기존 `Agent(config=..., model=..., ...)`, `decision_policy`, `PlanAndExecutePolicy`, Tool 안전 boolean과 주요 top-level import를 유지합니다. 우선 의존성만 올려 기존 테스트를 실행한 뒤 실행 Profile, `Agent.inspect()`, 명시적 Tool 안전 Profile을 점진적으로 적용할 수 있습니다.

```bash
python -m pip install --upgrade "moduagent==0.4.2"
```

0.4.2는 checkpoint schema와 기본 실행 의미를 변경하지 않는 성능 릴리스입니다.
checkpoint store가 없으면 Engine snapshot을 만들지 않으며, 내장 v4 store는
legacy 상태 왕복 없이 저장합니다. `CachingTokenCounter`와
`SyncToolScheduler`는 명시적으로 구성할 때만 적용됩니다. 동기 Redis/DB
adapter는 이벤트 루프 대신 bounded worker에서 실행됩니다.

## 실행 Profile로 전환

기존 Standard Agent는 변경 없이 동작합니다.

```python
agent = Agent(
    config=config,
    model=model,
    tools=tools,
)
```

새 코드에서 Standard를 명시하려면 다음처럼 작성합니다.

```python
from moduagent import StandardExecutionProfile

agent = Agent(
    config=config,
    model=model,
    tools=tools,
    execution_profile=StandardExecutionProfile(),
)
```

0.3.2 Plan 구성:

```python
agent = Agent(
    config=config,
    model=model,
    tools=tools,
    decision_policy=PlanAndExecutePolicy(
        plan_generator=LLMPlanGenerator(model=model),
    ),
)
```

0.4 권장 구성:

```python
from moduagent import PlanExecutionProfile

agent = Agent(
    config=config,
    model=model,
    tools=tools,
    execution_profile=PlanExecutionProfile(
        plan_generator=LLMPlanGenerator(model=model),
    ),
)
```

`decision_policy`와 `execution_profile`을 동시에 지정하면 구성 오류가 발생합니다. 기존 custom `DecisionPolicy`가 필요하면 당분간 `decision_policy`를 유지하고 `Agent.inspect().compatibility_metadata`에서 resolver 결과를 확인하세요.

## AgentSpec 확인

Agent 생성 직후 resolved configuration을 저장해 배포 간 차이를 확인할 수 있습니다.

```python
spec = agent.inspect()
inspection = spec.to_dict(include_instructions=False)

print(spec.agent_fingerprint)
print(inspection["execution_profile"])
print(inspection["tools"])
```

0.4 resume은 Agent fingerprint와 Engine 정보를 호환성 판단에 사용합니다. 모델, Tool schema, 실행 Profile, 출력·저장 정책이 의도치 않게 바뀌지 않았는지 배포 전에 비교하세요.

모델 adapter가 Tool 호출과 구조화 출력을 한 요청에서 함께 처리하지
못한다면 `ModelCapabilities.tool_calling_with_structured_output=False`로
선언하세요. Standard Engine은 ACT와 FINALIZE를 분리하며, `VLLMClient`는
이 값을 기본적으로 `False`로 설정합니다.

0.4부터 Standard/Plan 실행 중의 assistant Tool call과 Tool result
transcript는 내부 실행 문맥으로만 사용됩니다. 원본 인자와 결과는
`ConversationStore` 또는 `AgentResult.messages`에 저장되지 않으며, 공개
`tool_trace`에는 fingerprint와 안전한 분류만 남습니다.

## Tool 안전 설정

기존 설정은 유지됩니다.

```python
@function_tool(
    idempotent=True,
    repair_safe=True,
    timeout_retry_safe=False,
)
def search(query: str) -> list[dict]:
    ...
```

새 설정은 같은 의미를 value object로 모읍니다.

```python
from moduagent.tools import ToolSafetyProfile


@function_tool(
    safety_profile=ToolSafetyProfile(
        same_call_retry_safe=True,
        changed_argument_repair_safe=True,
        timeout_retry_safe=False,
    ),
)
def search(query: str) -> list[dict]:
    ...
```

두 방식을 한 Tool에 함께 지정할 수 없습니다. custom Tool이 명시적 Profile이나 기존 boolean을 제공하지 않으면 자동 복구는 fail closed합니다.

`error_mapper`와 `ToolError`는 계속 지원됩니다. 새 통합은 `failure_classifier`로 `ToolFailureClassification`을 반환할 수 있습니다. `recovery_directive=None`은 0.3.2의 fallback 의미를 유지합니다.

## Checkpoint v3 → v4

0.4 snapshot은 다음처럼 분리됩니다.

```text
RunSnapshot v4
├── common_state
├── engine
│   ├── engine_id
│   ├── state_version
│   └── state
├── finalization_markers
├── skill_state
└── sanitized_runtime_metadata
```

outer `schema_version`은 `4`이고 compatibility guard인 `version`도 `4`로 기록됩니다. Plan state는 `plan_progress`, `step_execution`, `tool_recovery`, `finalization` 하위 상태로 이동합니다.

v1-v3 payload는 읽을 때 deep copy한 뒤 완전히 검증하여 v4 object로 바꿉니다.

```python
import json

from moduagent.persistence.migration import migrate_checkpoint_payload

with open("checkpoint-v3.json", encoding="utf-8") as file:
    original = json.load(file)

snapshot = migrate_checkpoint_payload(
    original,
    agent_fingerprint=agent.inspect().agent_fingerprint,
)
payload_v4 = snapshot.to_dict()

assert original["version"] == 3
assert payload_v4["schema_version"] == 4
assert payload_v4["version"] == 4
```

운영 store를 한꺼번에 덮어쓰기 전에 실제 v3 fixture로 다음 상황을 검증하세요.

- Standard 실행과 strict Plan 실행
- step 시작 전·후, pending `StepResult`, step commit 후
- pending Tool repair와 repair budget
- FINALIZE 응답 생성·저장·방출 marker
- 완료, terminal 실패, timeout 중단

Migration은 raw Tool arguments를 복제하지 않고 one-way fingerprint와 정제된 failure만 보존합니다. pending repair는 원래 실패 call을 replay하지 않고 repair 요청부터 재개합니다. 이미 일부 Tool이 성공한 batch처럼 의미가 불명확한 상태는 자동 재실행하지 않고 fail closed합니다.

checkpoint를 켠 Agent는 conversation append도 crash-safe해야 합니다. 기존 custom store가 `load/append/clear`만 제공한다면 0.4 조립 단계에서 거부됩니다. DB의 unique constraint나 Redis Lua transaction을 사용해 다음 선택 계약을 추가하고 capability를 명시하세요.

```python
class ProductionConversationStore:
    supports_idempotent_append = True

    async def append_once(
        self,
        session_id: str,
        idempotency_key: str,
        messages: list,
    ) -> bool:
        # 같은 key+digest는 False, 같은 key+다른 digest는 오류.
        # key 기록과 message append는 반드시 한 transaction이어야 한다.
        ...
```

다음 payload는 migration 오류로 거부됩니다.

- outer `execution_state`와 `policy_state["execution_state"]`가 서로 다름
- outer finalization marker와 Plan 내부 mirror가 서로 다름
- current step, attempt count, plan step 상태가 일관되지 않음
- fingerprint 형식이 잘못되었거나 active call ID가 seen ID에 없음
- 지원하지 않는 outer schema 또는 Engine state version

v4 snapshot을 v3로 downgrade하지 않습니다. 롤백이 필요하면 0.3.2 프로세스용 store/key namespace를 분리하고 전환 전 v3 데이터를 보존하세요.

## 이벤트 소비자

기존 `EventType`과 `event.type`, `event.occurred_at` 사용법은 유지됩니다. 0.4 event envelope에는 event ID, schema version, session ID, Engine ID, run 내 sequence가 추가됩니다.

이벤트를 JSON으로 저장하는 소비자는 새 필드를 무시할 수 있도록 additive parsing을 사용하세요. 순서 판단에는 수신 시각 대신 `(run_id, sequence)`를 사용하고, `event_id`로 중복 전달을 방어하세요. public stream의 terminal `data["result"]`는 기존처럼 typed `AgentResult`일 수 있으므로 wire 저장에는 event의 JSON-safe 변환 경계를 사용합니다.

## 동작 호환성 확인

업그레이드 후 다음 회귀 테스트를 권장합니다.

- Standard text 응답의 모델 호출 수
- Standard Tool + Pydantic의 ACT/FINALIZE 분리
- strict Plan의 단계 상태와 commit 결과
- same-call retry, changed-argument repair, replan 예산의 독립성
- partial-success Tool batch의 fail-closed 처리
- public stream에 내부 ACT token이 노출되지 않음
- 대화에 공개 최종 응답이 한 번만 저장됨
- terminal result와 event가 한 번만 방출됨

## 0.4의 비목표

이번 버전은 Graph, 분산 queue/scheduler, peer multi-agent protocol, 범용 Human approval workflow를 추가하지 않습니다. 해당 기능을 기대하는 migration은 별도 orchestrator 또는 후속 Engine 설계가 필요합니다.
