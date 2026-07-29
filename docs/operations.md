# ModuAgent Operations

이 문서는 ModuAgent 0.4를 사내 운영 환경에 배포할 때 확인할 항목을 정리합니다.

## Deadline과 예산

한 run의 `RunLimits.timeout_seconds`는 모델, Memory 요약, Tool, 대화·checkpoint 저장을 포함한 전체 deadline입니다.

```python
from moduagent import AgentConfig, RetryConfig, RunLimits

config = AgentConfig(
    name="production-agent",
    instructions="확인된 정보만 답한다.",
    retry=RetryConfig(max_attempts=2),
    limits=RunLimits(
        max_steps=6,
        max_tool_calls=10,
        max_step_attempts=2,
        max_replans=1,
        max_tool_repair_attempts=1,
        timeout_seconds=120,
    ),
)
```

모델 client timeout, Tool timeout, DB statement timeout은 전체 deadline보다 짧게 설정해 정리와 checkpoint 저장 시간을 남기세요. 출력 token 상한과 provider의 context window도 별도로 설정해야 합니다.

## 동기 Tool과 timeout

`pd.read_sql` 같은 동기 함수는 worker thread에서 실행됩니다. 프레임워크 timeout이 발생해도 이미 실행 중인 Python thread나 DB query를 강제로 중단할 수 없습니다.

- DB driver의 query/connection timeout을 설정합니다.
- 서버 측 statement timeout과 query resource limit을 설정합니다.
- 쓰기 Tool에는 application idempotency key와 중복 방어를 구현합니다.
- 실행 취소가 확인되지 않은 동기 Tool은 `timeout_retry_safe=False`로 둡니다.
- 결과 행·열과 `max_result_bytes`를 제한합니다.

`ToolSafetyProfile`은 자동 복구 허용 범위이지 transaction 또는 exactly-once 선언이 아닙니다.

Tool 호출 직전 checkpoint는 `resume_safety=manual_required`로 저장됩니다. Tool outcome과 Engine commit이 안전한 durable boundary에 도달해야 다시 `resumable`이 됩니다. terminal `error_summary.retryable`은 요청을 새로 시도할 수 있는지, `resumable`은 저장된 동일 run을 replay 위험 없이 이어갈 수 있는지를 각각 뜻합니다.

## 대화와 Memory

`InMemoryConversationStore`와 `InMemoryMemoryStateStore`는 단일 프로세스 개발·테스트용입니다. 프로세스 재시작과 여러 worker 사이에 상태를 공유하지 않습니다.

운영에서는 durable `ConversationStore`를 사용하고 다음을 정합니다.

- tenant별 key namespace와 접근 제어
- 대화 및 요약 cache의 TTL
- 보존·삭제 정책과 개인정보 처리
- context window, 예약 출력 token, safety margin
- 요약 모델 실패 시 fallback과 추가 호출 비용

Memory Policy는 모델에 전달할 view만 줄이고 원문 저장소를 삭제하지 않습니다.

`checkpoint_store`를 사용하는 Agent의 `ConversationStore`는 `IdempotentConversationStore` 계약을 만족해야 합니다. 즉 `supports_idempotent_append=True`를 선언하고 `append_once(session_id, idempotency_key, messages)`를 원자적으로 구현해야 합니다. 같은 key와 같은 batch의 재호출은 `False`, 같은 key에 다른 batch를 넣으면 오류를 반환해야 합니다. 프로세스 로컬 lock만으로는 여러 worker 사이의 원자성을 보장하지 못합니다.

## Redis 대화와 checkpoint

```python
from redis.asyncio import Redis

from moduagent import RedisCheckpointStore, RedisConversationStore

redis = Redis.from_url(
    "redis://localhost:6379/0",
    decode_responses=True,
)
conversations = RedisConversationStore(
    client=redis,
    ttl_seconds=7 * 24 * 60 * 60,
)
checkpoints = RedisCheckpointStore(
    client=redis,
    ttl_seconds=24 * 60 * 60,
)
```

대화 TTL과 checkpoint TTL은 독립적입니다. resume 대상 run보다 checkpoint TTL이 짧지 않게 설정하세요.

`RedisConversationStore`는 list mode와 `EVAL`을 제공하는 client에서 atomic `append_once`를 지원합니다. `get/set` fallback 또는 `EVAL`이 없는 client는 checkpointed Agent에 사용할 수 없습니다. DB adapter는 repository의 단일 transaction/unique constraint로 `append_messages_once(session_id, idempotency_key, rows, digest) -> bool`을 구현하세요.

0.4는 outer schema v4와 Engine별 `state_version`을 분리합니다. v1-v3 checkpoint는 읽을 때 복사·검증 후 v4로 migration하며 원본 payload를 변경하지 않습니다. pending repair는 원래 실패 call을 재실행하지 않고 repair turn에서 재개하고, 의미가 불명확한 partial batch는 fail closed합니다. v4를 v3로 downgrade하지 않습니다.

같은 `run_id`와 `session_id`, 호환되는 Agent fingerprint, Engine ID와 state version으로만 resume하세요. 배포 전 [0.4 마이그레이션](migration-0.4.md)의 fixture 검증을 실행하는 것이 좋습니다.

## 이벤트와 관측성

운영 로그, metric, audit는 `CompositeEventSink`로 분리할 수 있습니다.

```python
from moduagent import (
    AuditEventSink,
    CompositeEventSink,
    InMemoryMetricRecorder,
    LoggingEventSink,
    MetricsEventSink,
)

metrics = InMemoryMetricRecorder()
events = CompositeEventSink(
    sinks=[
        LoggingEventSink(),
        MetricsEventSink(recorder=metrics),
        AuditEventSink(writer=audit_writer),
    ]
)
```

이벤트 sink 실패는 run을 실패시키지 않습니다. 각 이벤트는 run 안에서 단조 증가하는 sequence와 event/session/Engine 식별자를 가지므로 소비자는 `(run_id, sequence)` 또는 `event_id`로 중복을 방어할 수 있습니다.

기본 public stream에는 사용자에게 필요한 token과 terminal result만 전달됩니다. Plan phase, Tool, recovery, checkpoint 진단은 `stream_all()` 또는 내부 sink에서 확인합니다. `stream_all()` 역시 JSON-safe 최소 projection이며 원본 exception, Tool 결과, provider metadata를 제공하지 않습니다.

`RUN_FAILED`만 보인다면 terminal result의 안전한 오류와 내부 sink의 직전 이벤트를 함께 확인합니다. `RUN_STARTED` 다음 즉시 실패하면 구성 capability, checkpoint 호환성, 첫 모델 요청 전 Memory overflow를 우선 점검합니다.

## Tool trace와 민감정보

`AgentConfig.tool_trace_mode`는 다음 값을 지원합니다.

| 값 | 저장 내용 |
|---|---|
| `off` | Tool trace 없음 |
| `summary` | Tool, call ID, 성공 여부, 시도와 정제된 오류 분류 |
| `arguments` | summary와 masking된 validation 인자 |

운영 기본값은 `summary`를 권장합니다. `arguments` masking은 알려진 민감 key를 줄이는 방어선이지 데이터 유출을 완전히 탐지하는 기능이 아닙니다. Tool 결과와 backend 원문 오류는 별도 접근 통제된 로컬 진단 시스템에만 보관하세요.

## Skill 보안

- 운영자가 검토한 catalog와 lockfile을 사용합니다.
- 서로 신뢰하지 않는 tenant는 Registry 또는 접근 제어 계층을 분리합니다.
- `allowed-tools`는 권한 부여가 아니며 `ToolAuthorizer`를 우회하지 않습니다.
- filesystem source는 지원되는 POSIX 보안 경로 계약을 사용합니다.
- Skill의 `scripts/`는 자동 실행되지 않습니다.

## 배포 점검표

- 모델의 tool calling, parallel tool calling, structured output capability 확인
- `Agent.inspect()` 결과와 `agent_fingerprint`를 배포 artifact에 기록
- Tool별 안전 Profile과 예외 allowlist 검토
- model/Tool/DB/전체 run timeout의 계층화
- Conversation·checkpoint TTL, backup, 암호화, tenant 격리 확인
- v3 fixture의 v4 migration과 resume 테스트
- public/internal event 분리와 sink 장애 격리 테스트
- 쓰기 Tool idempotency key와 downstream 중복 방어 확인
- 프로세스 내부 session lock에 의존하지 않는 분산 동시성 제어

ModuAgent는 분산 lock, 작업 queue, scheduler, durable outbox를 제공하지 않습니다. 필요한 경우 애플리케이션 인프라에서 구성합니다.
