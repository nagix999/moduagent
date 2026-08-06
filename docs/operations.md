# ModuAgent Operations

이 문서는 ModuAgent 0.5를 사내 운영 환경에 배포할 때 확인할 항목을 정리합니다.

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
        max_model_turns=32,
        no_progress_model_turn_threshold=3,
        timeout_seconds=120,
    ),
)
```

모델 client timeout, Tool timeout, DB statement timeout은 전체 deadline보다 짧게 설정해 정리와 checkpoint 저장 시간을 남기세요. 출력 token 상한과 provider의 context window도 별도로 설정해야 합니다.

### 모델 retry 분류

`RetryConfig.max_attempts`는 모든 예외를 반복하는 옵션이 아닙니다. 0.5는 원인 chain을 allowlist로 분류하고 다음 일시적 실패만 재시도합니다.

| 분류 | retry | 안전한 진단 code |
|---|---:|---|
| timeout, HTTP 408 | 예 | `model_timeout` |
| connection/network 오류 | 예 | `model_connection_error` |
| HTTP 5xx | 예 | `model_http_5xx` |
| HTTP 4xx와 429 | 아니요 | `model_http_4xx` |
| 잘못된 요청 값 | 아니요 | `model_request_invalid` |
| client type/contract 오류 | 아니요 | `model_client_contract_error` |
| 응답 protocol·JSON decoding 오류 | 아니요 | `model_protocol_error` |
| provider가 `timeout`·`length`·`max_tokens`로 출력을 중단 | 아니요 | `model_output_incomplete` |
| 분류되지 않은 실행 오류 | 아니요 | `model_invocation_failed` |

provider가 HTTP 200을 반환했더라도 body, `choices`, Tool call, Tool arguments JSON, stream chunk 또는 terminal response가 adapter 계약에 맞지 않으면 `ModelProtocolError`다. 같은 응답을 다시 받아도 해결되지 않는 결정적 실패이므로 즉시 종료하고 provider retry를 소비하지 않습니다. Plan JSON이 잘못된 경우도 catch-all 계획으로 바꾸지 않습니다.

provider 응답의 종료 이유가 `timeout`, `length`, `max_tokens`이면 부분 출력을
공개하거나 Pydantic으로 decode하지 않고 `model_output_incomplete`로 종료합니다.
이때 `error_summary["provider_finish_reason"]`과 `AgentRunError.provider_finish_reason`
에는 세 값 중 하나만 보존되며, 응답 본문과 provider metadata는 보존하지
않습니다. `length`나 `max_tokens`는 재시도보다 출력 예산·스키마·프롬프트를
검토해야 하므로 자동 재시도하지 않습니다.

strict Plan의 schema-only `StepResult`가 잘못된 JSON이거나 Pydantic schema에 맞지 않으면 현재 단계를 즉시 `failed`로 전이하고 `step_result_schema_invalid`를 반환합니다. 같은 `StepResult` 요청을 반복하거나 `max_step_attempts`만큼 재생성하지 않습니다. 최종 Pydantic output 검증 실패도 provider retry가 아니라 terminal output validation 실패입니다.

stream에서 아직 delta를 공개하지 않은 transient 오류만 재시도할 수 있습니다. 하나 이상의 delta를 이미 내보낸 뒤 연결이 끊기면 중복 출력을 막기 위해 retryable transport 오류라도 같은 stream을 다시 시작하지 않습니다. `MODEL_STARTED`, `MODEL_FAILED`, `RETRY` 이벤트로 model turn, attempt, phase, 요청의 message·Tool 개수, structured-output 사용 여부와 지연시간을 확인할 수 있습니다. 이 이벤트에는 prompt·message content·Tool arguments·provider body를 넣지 않습니다. `RETRY`와 `error_summary`에도 정제된 category와 code만 기록됩니다.

`LoggingEventSink`는 기본적으로 `MODEL_DELTA`, `STEP_MODEL_DELTA`,
`FINAL_DELTA`를 기록하지 않습니다. 개발 중 chunk 단위 이벤트 개수가
필요한 경우에만 `LoggingEventSink(include_deltas=True)`를 사용하세요.
이 경우에도 built-in 로그에는 delta 본문이 아니라 문자 수와 byte 수만
남습니다. `MetricsEventSink`는 실패한 model attempt 수를
`model.calls.failed`, 실패 지연시간을 `model.failed_duration_seconds`로
기록합니다.

### 모델 turn 예산과 무진전 차단

`max_steps`는 계획 단계 수이며 모델 호출 수가 아닙니다. 0.5의 `max_model_turns`는 한 run에서 `ModelGateway`를 통과하는 실제 provider 시도 수입니다.

- 기본값은 `32`입니다.
- PLAN, ACT, `StepResult`, FINALIZE와 ModelGateway를 통과하는 보조 모델 요청을 합산합니다.
- `RetryConfig`의 첫 시도와 재시도를 각각 한 turn으로 셉니다.
- 실패해서 normalized response를 만들지 못한 시도도 provider를 호출했으므로 예산을 소비합니다.
- streaming은 chunk 수가 아니라 stream provider 시도 하나를 한 turn으로 셉니다.
- 다음 시도가 상한을 초과하기 직전에 종료하므로 provider 호출 수는 설정값을 넘지 않습니다.

예를 들어 `max_model_turns=1`, `RetryConfig(max_attempts=3)`이면 첫 호출이 connection 오류로 실패해도 두 번째 provider 호출 전에 `FinishReason.MAX_MODEL_TURNS`로 종료합니다.

`CheckpointStore`가 설정되어 있으면 첫 시도와 각 retry의 turn 예약을 실제 provider I/O 직전 `before_model` durable boundary에 저장합니다. 예약 저장에 실패하면 provider를 호출하지 않습니다. 요청 중 프로세스가 강제 종료되더라도 resume은 저장된 카운터에서 이어지므로 이미 전송했을 수 있는 시도를 다시 사용하지 않습니다.

기본 planner, memory summarizer와 Skill selector는 public `Agent`에서 이 gateway를 사용합니다. custom 정책·selector·planner도 `MemoryRequest.model_gateway`, `SkillSelectionRequest.model_gateway` 또는 `RunContext.model_gateway`를 사용해야 합니다. 이를 무시하고 provider를 직접 호출하거나 custom `ModelClient` 내부에서 자체 retry하면 프레임워크는 숨겨진 요청을 별도 turn으로 셀 수 없습니다. 전체 예산 보장이 필요한 확장 구성 요소의 계약 테스트에 이 조건을 포함하세요.

`no_progress_model_turn_threshold`는 같은 실행 의미 상태에서 모델이 같은 의미 응답을 연속 반환하는 loop를 차단합니다. 기본값 `3`은 첫 관찰을 1로 세며 세 번째 동일 관찰에서 종료합니다. 즉 한 번의 중복 응답은 허용하지만 그다음 동일 응답은 실행하지 않습니다.

비교 대상은 다음과 같습니다.

- 상태: Engine ID, phase, 현재 logical step ID
- 응답: content, finish reason, Tool 이름과 JSON arguments

provider가 매번 새로 만드는 Tool call ID, token usage와 provider metadata는 의미 비교에서 제외됩니다. 따라서 call ID만 바꾼 동일 Tool 요청은 진전으로 보지 않습니다.

별도로 검증되는 진전 경계는 다음과 같습니다.

- 성공한 Tool outcome의 Tool 이름, 실제 검증된 인자와 정규화된 결과로 만든 run-salted fingerprint가 직전 성공 outcome과 다를 때
- 새 대화 record batch를 소비한 `memory_summary` 호출이 성공했을 때
- Plan step이 커밋되었을 때

같은 Tool이 같은 유효 인자와 같은 결과로 성공하는 반복은 fingerprint가 같으므로 streak를 초기화하지 않습니다. provider call ID, 실행 시간이나 retry 횟수만 달라진 경우도 새 outcome이 아닙니다. Tool 결과가 실제로 바뀌면 새 fingerprint이므로 진전으로 인정합니다.

회로 차단기는 원본 프롬프트, 모델 출력 또는 Tool arguments/results를 상태에 저장하지 않습니다. checkpoint에는 숫자 카운터, run별 무작위 salt와 HMAC-SHA-256 관찰 digest만 저장하며, 성공한 Tool outcome 비교값도 raw payload가 아닌 run-salted fingerprint로 저장합니다. resume은 같은 run의 이 상태를 이어갑니다. terminal 결과는 다음처럼 구분됩니다.

| `FinishReason` | error summary code | 의미 |
|---|---|---|
| `MAX_MODEL_TURNS` (`"max_model_turns"`) | `max_model_turns_exceeded` | 다음 provider 시도가 전체 turn 예산을 초과함 |
| `NO_PROGRESS` (`"no_progress"`) | `model_no_progress` | 동일 상태·동일 의미 응답이 설정한 threshold에 도달함 |

두 결과 모두 `RUN_FAILED`이며 `retryable=False`, `resumable=False`입니다. guard가 이미 소비한 turn 예산이나 terminal circuit 상태를 resume으로 되돌릴 수 없습니다. `AgentResult.metadata["error_summary"]`에는 원문 대신 `model_turns`, `max_model_turns`, `no_progress_model_turns`, `no_progress_model_turn_threshold`가 포함됩니다.

## 동기 Tool과 timeout

`pd.read_sql` 같은 동기 함수는 worker thread에서 실행됩니다. 프레임워크 timeout이 발생해도 이미 실행 중인 Python thread나 DB query를 강제로 중단할 수 없습니다.

동기 도구가 여러 개라면 bounded scheduler를 공유합니다.

```python
from moduagent import SyncToolScheduler, function_tool

db_workers = SyncToolScheduler(max_workers=8, max_queue=32)

@function_tool(sync_scheduler=db_workers, timeout_seconds=10)
def query_db(sql: str) -> list[dict]:
    return run_read_only_query(sql)
```

대기열이 가득 차면 `SyncToolSchedulerOverloaded`로 빠르게 실패합니다.
취소되거나 timeout된 작업도 실제 함수가 반환할 때까지 worker capacity를
차지하므로 DB 자체 timeout이 반드시 필요합니다. 동기 Redis/DB persistence
adapter도 0.4.2부터 공용 bounded worker에서 실행되어 event loop를 막지
않습니다.

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

0.4는 outer schema v4와 Engine별 `state_version`을 분리합니다. v1-v3 checkpoint는 읽을 때 복사·검증 후 v4로 migration하며 원본 payload를 변경하지 않습니다. pending repair는 원래 실패 call을 재실행하지 않고 repair turn에서 재개하고, 의미가 불명확한 partial batch는 fail closed합니다. v4를 v3로 downgrade하지 않습니다. 0.5는 outer schema를 올리지 않고 model guard의 카운터, run별 salt와 HMAC-SHA-256 digest를 compatibility policy state에 추가하므로 0.5에서 생성한 checkpoint를 resume하면 이미 소비한 model turn과 no-progress streak를 이어갑니다. 새 guard 한도는 Agent fingerprint에도 포함되므로 일반적인 0.4 active checkpoint는 0.5 worker에서 fail closed합니다. 먼저 0.4 worker에서 active run을 drain하거나 버전별 worker/store namespace에 고정하세요. checkpointing이 활성화된 모델 시도는 provider 호출 직전 durable하게 예약되며, 저장에 실패한 시도는 provider에 전송되지 않습니다.

같은 `run_id`와 `session_id`, 호환되는 Agent fingerprint, Engine ID와 state version으로만 resume하세요. 배포 전 [0.4 마이그레이션](migration-0.4.md)의 fixture 검증과 [0.5 마이그레이션](migration-0.5.md)의 model guard rolling 배포 점검을 실행하는 것이 좋습니다.

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

0.4.2의 내장 metric은 다음 성능 구간을 분리합니다.

| metric | 의미 |
|---|---|
| `moduagent.model.calls{phase}` | PLAN/ACT/STEP_RESULT/FINALIZE 등의 모델 호출 수 |
| `moduagent.model.duration_seconds{phase}` | 성공한 모델 호출 시간 |
| `moduagent.memory.prepare_seconds{phase}` | compact된 요청의 토큰 계산·요약 시간 |
| `moduagent.tool.duration_seconds{tool}` | 도구 실행 시간 |
| `moduagent.checkpoint.duration_seconds` | durable snapshot 저장 시간 |
| `moduagent.run.queue_wait_seconds` | 같은 세션 직렬화 대기 시간 |
| `moduagent.run.duration_seconds{status}` | 전체 run 시간 |

Noop sink는 queue와 deepcopy를 만들지 않습니다. 다른 sink의 전달 queue와
Runtime 내부 이벤트 handoff는 bounded이며, 가득 차면 메모리를 계속 늘리는
대신 실행에 backpressure를 적용합니다.

기본 public stream에는 사용자에게 필요한 token과 terminal result만 전달됩니다. Plan phase, Tool, recovery, checkpoint 진단은 `stream_all()` 또는 내부 sink에서 확인합니다. `stream_all()` 역시 JSON-safe 최소 projection이며 원본 exception, Tool 결과, provider metadata를 제공하지 않습니다.

`RUN_FAILED`만 보인다면 terminal result의 안전한 오류와 내부 sink의 직전 이벤트를 함께 확인합니다. `RUN_STARTED` 다음 즉시 실패하면 구성 capability, checkpoint 호환성, 첫 모델 요청 전 Memory overflow를 우선 점검합니다.

0.5의 terminal `finish_reason`에는 기존 `completed`, `max_steps`, `max_tool_calls`, `timeout`, `cancelled`, `error`에 `max_model_turns`와 `no_progress`가 추가됩니다. event consumer는 알려지지 않은 enum 값을 받을 수 있는 additive parsing을 사용하고 두 값도 실패 terminal로 처리해야 합니다.

0.4.1부터는 선택적인 `DiagnosticSink`를 설정하여 terminal
`result.failure_id`나 도구 추적의 `failure_id`로 정제된 예외 원인을 연결할 수
있습니다. 원본 예외, SQL, 프롬프트, 도구 입출력은 수집하지 않습니다. 구성과
운영 보안 지침은 [진단 가이드](diagnostics.md)를 참고하세요.

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
- `RetryConfig`가 일시적 transport 오류에만 적용되는지 fault injection으로 확인
- `max_model_turns`와 `no_progress_model_turn_threshold`의 업무별 상한 검토
- `max_model_turns`, `no_progress` terminal event와 alert routing 확인
- Conversation·checkpoint TTL, backup, 암호화, tenant 격리 확인
- v3 fixture의 v4 migration과 resume 테스트
- public/internal event 분리와 sink 장애 격리 테스트
- `python benchmarks/performance_v042.py --pretty` 기준 결과 보관 및 비교
- 쓰기 Tool idempotency key와 downstream 중복 방어 확인
- 프로세스 내부 session lock에 의존하지 않는 분산 동시성 제어

ModuAgent는 분산 lock, 작업 queue, scheduler, durable outbox를 제공하지 않습니다. 필요한 경우 애플리케이션 인프라에서 구성합니다.
