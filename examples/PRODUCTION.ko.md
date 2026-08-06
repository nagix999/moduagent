# 프로덕션 제어 예제

[`20_production_controls.py`](20_production_controls.py)는 학습 예제에서 운영
Agent로 넘어가기 위한 작은 예제입니다. 변경 요청 하나를 조회하고 조건을
충족할 때 승인 쓰기 한 번을 수행합니다. 신원, 권한, 멱등성 키와 저장소는
모델이 아니라 애플리케이션이 소유합니다.

OpenAI 호환 vLLM endpoint를 설정한 뒤 실행합니다.

```bash
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL="your-tool-capable-model"
export VLLM_API_KEY="..."  # 토큰이 필요 없으면 생략합니다.
python examples/20_production_controls.py
```

모듈을 import할 때는 자격증명을 읽거나 network client를 생성하지 않습니다.
`VLLMClient.from_env()`는 `main()` 안에서만 호출됩니다.

## 예제가 강제하는 경계

- `get_change_request`와 `approve_change`는 하나의 authoritative repository를
  공유합니다. 둘 다 controller가 발급한 대상/tenant 범위를 검사하며, 쓰기는
  조회 결과의 정확한 version을 사용합니다.
- `ScopedChangeAuthorizer`는 먼저 deny-by-default RBAC를 적용하고, 요청한
  변경 건과 tenant가 controller가 발급한 범위와 일치해야 승인합니다. 쓰기
  Tool도 애플리케이션 경계에서 같은 범위를 다시 확인합니다.
- 역할, `user_id`, 대상 범위와 멱등성 키는 신뢰할 수 있는 `user_context`로
  전달합니다. prompt 문장이 권한을 부여해서는 안 됩니다. context는 보호된
  event와 checkpoint에 포함될 수 있으므로 JWT, credential, 개인정보를 넣지
  않습니다.
- controller가 run 전에 멱등성 키 하나를 생성합니다. 쓰기 Tool은 키와
  payload를 원자적으로 묶는 애플리케이션 저장소에 전달합니다. 키는
  model-visible schema와 prompt 밖의 신뢰된 context에서 주입합니다. 동일한
  요청을 다시 보내면 기존 receipt를 반환하고, 같은 키에 다른 payload를
  사용하면 실패합니다.
- eligibility, expected version, 상태 전이, uniqueness 검사와 receipt 생성을
  repository의 한 critical section에서 처리합니다. 운영 DB 구현은 사전 조회 후
  쓰기가 아니라 하나의 transaction과 조건부 갱신(예:
  `UPDATE ... WHERE version=? AND status='pending' RETURNING ...`)을 사용해야 합니다.
- 구조화 출력은 형태만 검증하며 업무 성공의 증거는 아닙니다. controller가
  모델 출력과 source of truth인 애플리케이션 receipt를 대조합니다. 모델 turn,
  Tool 호출, 전체 시간, 출력 크기와 최근 대화 turn을 제한합니다.
- checkpoint, conversation, diagnostic component를 `Agent.create()`에 직접
  전달합니다. 보호된 진단 정보는 공개 출력과 분리됩니다.

예제의 `InMemoryApprovalStore`, `InMemoryConversationStore`,
`InMemoryCheckpointStore`, `InMemoryDiagnosticSink`는 파일을 쉽게 실행하기
위한 것입니다. 영속적이지 않고 여러 process가 공유하지도 않습니다. 대화
TTL, session 수, 직렬화 byte 제한은 보관된 메시지 payload의 회계 상한이며
Python 객체 overhead나 process RSS 상한은 아닙니다. eviction도 보관 기능이
아닙니다. 실제 승인에는 멱등성 키와 tenant/업무 작업 양쪽에 unique 제약이
있는 DB transaction을 사용하고 payload digest와 receipt를 함께 저장해야 합니다.
version과 eligibility도 receipt 기록과 같은 조건부 쓰기 안에서 검사합니다.

## 긴 대화 요약하기

실행 예제는 최근의 완전한 여섯 turn만 사용합니다. 이전 문맥의 의미를
유지해야 한다면 token budget과 summarizer를 구성합니다.

```python
from moduagent import (
    InMemoryMemoryStateStore,
    ModelConversationSummarizer,
    SummarizingConversationMemoryPolicy,
    TokenBudget,
    VLLMTokenCounter,
)

memory = SummarizingConversationMemoryPolicy(
    budget=TokenBudget(
        context_window_tokens=32_768,
        reserved_output_tokens=1_024,
        safety_margin_tokens=1_024,
    ),
    token_counter=VLLMTokenCounter(model),
    summarizer=ModelConversationSummarizer(
        model=model,
        max_input_tokens=8_192,
        max_output_tokens=512,
    ),
    state_store=InMemoryMemoryStateStore(),
    max_history_turns=40,
)
agent = build_agent(
    model,
    conversation_store=conversations,
    memory=memory,
)
```

요약은 손실이 있는 모델 출력이며 model turn을 소비합니다. 또한 모델에
전달하는 view만 줄이고 원본 대화 record를 삭제하지 않습니다. 재시작 후에도
요약이 필요하면 in-memory summary state를 tenant별 durable 저장소로 교체해야
합니다.

## 영속 checkpoint와 resume

예제의 checkpoint 저장소는 API 설명용이지 장애 복구용이 아닙니다. Redis로
대화와 checkpoint를 재시작 안전하게 구성할 수 있습니다.

```python
from redis.asyncio import Redis
from moduagent import RedisCheckpointStore, RedisConversationStore

redis = Redis.from_url("redis://localhost:6379/0", decode_responses=True)
conversations = RedisConversationStore(
    redis,
    key_prefix="moduagent:conversation:",
    ttl_seconds=7 * 24 * 60 * 60,
)
checkpoints = RedisCheckpointStore(
    redis,
    key_prefix="moduagent:checkpoint:",
    ttl_seconds=24 * 60 * 60,
)
agent = build_agent(
    model,
    conversation_store=conversations,
    checkpoint_store=checkpoints,
)

result = await agent.resume(saved_run_id, session_id=saved_session_id)
```

resume에는 동일 session과 호환되는 Agent fingerprint가 필요합니다.
`resume_safety`가 `manual_required`인 checkpoint는 자동 재생하면 안 됩니다.
승인 record와 멱등성 키도 함께 영속화해야 합니다. Agent checkpoint만
영속적이라고 process-local 쓰기가 exactly-once가 되지는 않습니다.
resume된 쓰기마다 외부 인가 backend에서 현재 신원, 역할, tenant와 대상 권한을
다시 확인해야 합니다. 오래된 checkpoint의 역할을 현재 권한으로 신뢰하지 않습니다.

## Stream을 안전하게 취소하기

`RUN_STARTED`에서 `run_id`를 보관하고 client 연결이 끊기면 async generator를
항상 닫습니다. generator를 닫거나 consumer task를 cancel한 뒤 await하면
ModuAgent가 cancellation 정리를 수행하고 가능한 경우 안전한 checkpoint를
남길 수 있습니다.

```python
from moduagent import EventType

stream = agent.stream_all(command, session_id=session_id, user_context=identity)
run_id = None
try:
    async for event in stream:
        if event.type is EventType.RUN_STARTED:
            run_id = event.run_id
        elif event.type is EventType.MODEL_STARTED:
            break  # 예: model 요청 중 downstream client 연결 종료
finally:
    await stream.aclose()

checkpoint = await checkpoints.load(run_id) if run_id else None
if checkpoint is not None and checkpoint.resume_safety == "resumable":
    result = await agent.resume(run_id, session_id=session_id)
```

`stream_all()`은 신뢰된 server-side 진단용입니다. 내부 event를 최종 사용자에게
그대로 전달하지 말고 명시적인 공개 event schema로 투영하고 redaction합니다.

취소가 동기 driver나 원격 side effect까지 멈춘다고 가정하면 안 됩니다. driver
timeout과 애플리케이션 멱등성을 적용하고, resume 전에 `resume_safety`를
확인해야 합니다.

## 여러 session 동시 실행

서로 다른 session ID는 Agent 인스턴스 하나에서 동시에 실행할 수 있습니다.

```python
first, second = await asyncio.gather(
    agent.run(first_command, session_id="ticket-4815", user_context=first_user),
    agent.run(second_command, session_id="ticket-4816", user_context=second_user),
)
```

같은 session ID의 호출은 한 runtime 안에서 직렬화되어 대화 순서를 유지합니다.
이 lock은 process-local이므로 multi-worker 환경에는 애플리케이션 차원의 분산
조정이 필요합니다. provider 동시성과 Tool worker queue를 제한하고, 쓰기
경합은 durable 멱등성 제약으로 방어해야 합니다.

## 배포 점검표

1. 모델 밖에서 인증하고 credential, JWT, 개인정보가 없는 최소한의 신뢰된
   `user_context`를 구성합니다.
2. Agent를 시작하기 전에 command와 멱등성 키를 영속화합니다.
3. conversation, checkpoint, summary, diagnostic, approval 저장소를 tenant별
   durable 구현으로 교체하고 retention을 정합니다.
4. 실행/resume 시 쓰기 범위를 다시 인가하고, DB statement timeout,
   조건부 version/eligibility 갱신, transaction, 업무 작업 및 멱등성 키
   uniqueness constraint와 최소 권한 credential을 설정합니다.
5. 거부, 중복 전달, process 장애, 취소, resume, 동시 쓰기를 실제 배포 전에
   테스트합니다.
