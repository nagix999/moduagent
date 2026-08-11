# Context Memory: ConversationMemoryPolicy 설계

상태: 구현 완료. 0.6.0은 기존 정책을 유지하면서 durable cursor pagination,
tenant-bound summary v2, CAS state store와 bounded Context assembly를 추가했다.
Long-Term Memory는 여전히 범위 밖이다.

## 목적

대화 원문은 보존하면서 모델에 전달하는 요청만 context window 안으로 제한한다. 긴 세션에서도 vLLM의 context 초과와 입력 증가에 따른 timeout을 예방하고, Tool Calling·구조화 출력·체크포인트 동작은 유지한다.

이 문서의 Memory는 현재 session의 요청 문맥을 구성하는 **Context Memory**다.
여러 session에서 사실, 선호, episode 또는 장기 작업 상태를 검색·수정하는
**Long-Term Memory**는 0.6.0에 포함되지 않는다. Context Memory의 summary도
원문에서 다시 만들 수 있는 cache이며 장기 지식의 source of truth가 아니다.

이 정책은 다음 두 대상을 구분한다.

- `ConversationStore`: 사용자·assistant·Tool 메시지 원문을 보존하는 source of truth
- `ConversationMemoryPolicy`: 각 모델 요청에 사용할 제한된 메시지 view를 생성하는 계층

```text
ConversationStore ── canonical history ──> History loader
                                                │
                                  full history 또는 summary + tail
                                                ▼
                                           RunContext
                                                │
                                                ▼
                                  ConversationMemoryPolicy
                                                │ bounded messages
                                                ▼
                                            ModelClient
```

기존 정책의 `prepare()`는 `RunContext.messages`, `new_messages`와
`ConversationStore`를 수정하지 않는다. 0.6 durable loader는 이미 compact된 원문
prefix를 RunContext에 다시 싣지 않고 summary boundary와 tail만 넣는다. 따라서
원문 전체의 source of truth는 항상 `ConversationStore`이며, durable 경로의
checkpoint와 `AgentResult.messages`가 과거 원문 전체를 복제한다고 가정하면 안
된다.

## 공개 API

새 모듈은 `moduagent.memory`에 둔다.

```python
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from moduagent.messages import Message, Usage
from moduagent.models import ModelGateway, ModelRequest


class MemoryPhase(str, Enum):
    PLAN = "plan"
    ACT = "act"
    STEP_RESULT = "step_result"
    VERIFY = "verify"
    FINALIZE = "finalize"


@dataclass(frozen=True, slots=True)
class MemoryRequest:
    run_id: str
    session_id: str
    phase: MemoryPhase
    model_request: ModelRequest
    protected_from: int
    user_context: Mapping[str, object] = field(default_factory=dict)
    model_gateway: ModelGateway | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class MemoryResult:
    messages: tuple[Message, ...]
    usage: Usage = field(default_factory=Usage)
    original_tokens: int = 0
    selected_tokens: int = 0
    summarized_messages: int = 0
    dropped_messages: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)


@runtime_checkable
class ConversationMemoryPolicy(Protocol):
    async def prepare(self, request: MemoryRequest) -> MemoryResult: ...
```

정책은 메시지만 선택할 수 있다. Runtime은 원래 `ModelRequest`의 `tools`, `output_schema`, `options`, `provider_options`를 그대로 유지하고 `messages`만 `MemoryResult.messages`로 교체한다.

`Agent`에는 기존 API와 호환되는 keyword-only 인자를 사용한다.

```python
conversation_memory_policy: ConversationMemoryPolicy | None = None
```

`Agent.create()`에서는 `context_memory=policy`를 권장하며 `memory=policy`도
호환 alias로 유지한다. 명시적 생성자에서는
`context_memory_policy=policy`와 기존 `conversation_memory_policy=policy` 중
하나만 지정한다.

`None`은 기존 동작과 같은 `FullConversationMemoryPolicy`를 사용한다. 운영 환경에서는 `TokenBudgetConversationMemoryPolicy`를 명시한다.

0.6의 `BoundedConversationMemoryPolicy`는 `context_bound`를 통해 감사 가능한
`MemoryContextBound(kind="turns" | "tokens", limit=...)`를 노출한다. Built-in
RecentTurns와 TokenBudget 계열은 이 값을 제공한다. Production profile은 typed
finite bound가 없는 custom 정책과 unbounded Full 정책을 구성 시점에 거부한다.

## 기본 구현

### FullConversationMemoryPolicy

입력 메시지를 그대로 반환한다. 기존 사용자와 테스트의 동작을 보존하고 점진적으로 배포하기 위한 identity policy다. Token 상한을 보장하지 않으므로 session이 길어지는 프로덕션 환경에서는 권장하지 않는다. 운영 기본값으로 간주하지 말고 배포 모델의 exact counter를 사용하는 `TokenBudgetConversationMemoryPolicy`를 명시한다.

### RecentTurnsConversationMemoryPolicy

tokenizer나 요약 모델 없이 최근 완료된 대화 turn N개만 선택하는 단순한 정책이다.

```python
policy = RecentTurnsConversationMemoryPolicy(max_turns=6)
```

`max_turns`는 메시지 수가 아니라 하나의 user 메시지부터 다음 user 메시지 직전까지를 뜻한다. assistant Tool Call과 모든 Tool 결과도 같은 turn에 포함되므로 중간에서 분리되지 않는다.

최종 view는 다음으로 구성한다.

```text
Agent system instruction
→ 최근 완료된 history turn 최대 N개
→ 현재 user 메시지와 현재 실행 전체
→ phase 전용 임시 프롬프트
```

- `max_turns=0`이면 과거 기록 없이 system instruction과 현재 실행만 사용한다.
- 현재 실행은 `max_turns`에 포함하지 않고 항상 유지한다.
- 요약 호출이나 별도 상태 저장소가 필요 없다.
- 원본 `ConversationStore`와 `RunContext`는 변경하지 않는다.
- 손상된 Tool 블록에 대한 원자성 검사는 TokenBudget 정책과 동일하게 적용한다.

이 정책은 token을 세지 않는다. 따라서 compaction이 발생했을 때
`MEMORY_COMPACTED.original_tokens`와 `selected_tokens`가 모두 `0`이면 실제
요청이 0 token이라는 뜻이 아니라 **미계수**를 뜻한다. Message 수와 제외된
turn 수는 별도 필드로 확인한다.

이 정책은 동작과 비용이 단순하지만 context window를 보장하지 않는다. 최근 한 turn이나 현재 Tool 결과 하나가 매우 클 수 있기 때문이다. 엄격한 토큰 제한이 필요하면 아래 `TokenBudgetConversationMemoryPolicy`에 `max_history_turns`를 함께 설정한다.

### TokenBudgetConversationMemoryPolicy

전체 요청의 토큰 수를 계산하고, 초과할 때 오래된 완전한 대화 turn을 요약하거나 제외한다.

```python
policy = TokenBudgetConversationMemoryPolicy(
    budget=TokenBudget(
        context_window_tokens=32_768,
        reserved_output_tokens=2_048,
        safety_margin_tokens=1_024,
    ),
    token_counter=token_counter,
    summarizer=summarizer,          # 선택
    state_store=memory_state_store, # summarizer 사용 시 권장
    max_history_turns=6,            # 선택, 최근 N개를 먼저 제한
)
```

`max_history_turns=None`이면 토큰 예산에 들어오는 만큼 최근 turn을 유지한다. 정수를 지정하면 먼저 최근 N개로 제한한 후 토큰 예산을 적용하므로 대화 횟수와 토큰 수를 동시에 제한할 수 있다.

요약을 항상 사용하는 구성을 명확히 표현하려면 `SummarizingConversationMemoryPolicy`와 `ModelConversationSummarizer`를 사용한다.

```python
token_counter = VLLMTokenCounter(model)
policy = SummarizingConversationMemoryPolicy(
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
```

`ModelConversationSummarizer`는 대화 record를 JSON으로 직렬화하고 제한된 크기의 batch를 순서대로 요약한다. 단일 record가 batch 예산보다 크면 fragment로 나누고, 각 응답을 다음 batch의 `previous_summary`로 전달한다. 요약 요청에는 Tool과 애플리케이션 output schema를 전달하지 않으며 빈 응답이나 Tool Call 응답은 실패로 처리한다.

토큰 예산은 다음과 같다.

```text
input_budget = context_window_tokens
             - reserved_output_tokens
             - safety_margin_tokens
```

최종 요청은 반드시 다음 조건을 만족해야 한다.

```text
count(messages + tool schema + output schema + chat-template overhead)
    <= input_budget
```

`Usage.input_tokens`는 모델 응답 이후 값이므로 사전 제한에 사용하지 않는다. `TokenCounter`는 `ModelRequest` 전체를 세어야 한다.

```python
@runtime_checkable
class TokenCounter(Protocol):
    async def count_request(self, request: ModelRequest) -> int: ...
```

운영 환경에서는 실제 배포 모델의 tokenizer를 사용한다. 제공되는 `VLLMTokenCounter`는 vLLM의 chat 메시지와 Tool을 받는 `/tokenize` API를 사용한다. 로컬 휴리스틱은 `ApproximateTokenCounter`로 명확히 이름 붙이고 큰 safety margin을 적용한다. Exact token 제한이 성공 기준이라면 `ApproximateTokenCounter`만으로 상한을 보장했다고 판단하지 않는다.

동일한 요청이 여러 실행 단계에서 반복되면 exact tokenizer 호출도 재사용할 수
있다. `CachingTokenCounter`는 delegate의 계산 방식을 바꾸지 않으며, 성공한
결과만 bounded LRU에 저장하고 동시에 들어온 동일 요청은 하나의 호출로
합친다.

```python
from moduagent.memory import CachingTokenCounter, VLLMTokenCounter

token_counter = CachingTokenCounter(
    VLLMTokenCounter(model),
    max_entries=2_048,
    ttl_seconds=300,
)
```

cache key에는 전체 `ModelRequest`의 메시지, Tool schema, output schema,
options, provider options가 포함된다. 프로세스별 keyed digest만 저장하므로
원문 prompt나 Tool 인자는 cache에 남지 않는다. 실패나 취소 결과는 저장하지
않으며 `max_entries`를 넘으면 가장 오래 사용하지 않은 항목부터 제거한다.
vLLM의 exact count 의미는 delegate가 그대로 결정한다.

`TokenBudgetConversationMemoryPolicy`도 한 번의 `prepare()` 안에서 이미 계산한
동일 요청을 임시로 재사용한다. 이 request-local memo는 `prepare()`가 끝나면
폐기되며, 최종 선택 view가 이미 검증된 경우 같은 요청을 다시 계산하지 않는다.

- vLLM context limit: <https://docs.vllm.ai/en/stable/features/context_extension/>
- vLLM tokenizer API: <https://docs.vllm.ai/en/stable/serving/openai_compatible_server/>

## 메시지 선택 규칙

Runtime은 과거 기록과 현재 실행의 경계를 `RunContext.current_run_start`에 기록하고 체크포인트에도 저장한다. 마지막 `user` 메시지를 검색해 경계를 추측하면 FINALIZE의 임시 user 프롬프트 때문에 잘못된 범위를 보호할 수 있으므로 사용하지 않는다.

최종 모델 view의 순서는 다음과 같다.

```text
Agent system instruction
→ 제외된 과거 prefix의 요약
→ 예산에 들어오는 최근의 완전한 history turn
→ 현재 user 메시지와 현재 실행의 전체 메시지
→ FINALIZE 전용 임시 프롬프트
```

알고리즘은 다음 순서로 동작한다.

1. 원본 `ModelRequest`의 토큰 수를 계산한다.
2. 예산 이하면 메시지를 그대로 반환한다.
3. 초과하면 `current_run_start` 이전 메시지를 완전한 user turn으로 묶는다.
4. 최신 turn부터 역순으로 연속된 suffix를 선택한다. 중간 turn을 건너뛰지 않는다.
5. 제외되는 오래된 연속 prefix는 summarizer가 있으면 요약하고, 없으면 제거한다.
6. 실제 최종 요청을 다시 계산한다. 초과하면 가장 오래된 선택 turn부터 추가로 제외한다.
7. 필수 메시지만으로도 초과하면 Provider를 호출하지 않고 `ConversationMemoryOverflowError`를 발생시킨다.

항상 유지하는 메시지는 다음과 같다.

- 최초 Agent system instruction
- 현재 user 입력
- 현재 실행에서 생성한 assistant·Tool·정책 메시지
- 현재 phase의 임시 프롬프트

현재 user 입력이나 Tool/output schema 자체가 너무 크면 의미를 바꾸는 자동 절단을 하지 않는다. 오류에는 `required_tokens`, `available_tokens`, `message_tokens`, `tool_tokens`, `schema_tokens`를 포함한다.

## Tool 메시지 원자성

assistant Tool Call과 대응하는 모든 Tool 결과는 하나의 블록으로 취급한다.

```text
assistant(tool_calls=[A, B])
→ tool(call_id=A)
→ tool(call_id=B)
```

불변 조건은 다음과 같다.

- Tool Call 블록은 전체를 유지하거나 전체를 제외한다.
- 포함된 Tool 결과에는 같은 view 안에 대응하는 assistant Tool Call이 있어야 한다.
- 병렬 Tool Call의 모든 결과를 함께 유지한다.
- `call_id`, Tool 이름, 인자, 메시지 순서는 변경하지 않는다.
- 손상된 과거 Tool 블록은 해당 turn 전체를 제외하고 이벤트를 남긴다.
- 현재 실행의 Tool 블록이 손상됐으면 `MemoryIntegrityError`로 실패한다.

현재 실행의 Tool 결과가 너무 큰 경우에도 기본값은 오류다. 0.6.0에는 Tool 결과를 의미 보존 방식으로 축약하는 public reducer API가 없으므로, Tool 자체의 pagination과 `max_result_bytes`를 사용해 결과 크기를 제한한다.

## 요약과 상태 저장: legacy 호환 경로

요약은 원문이 아니라 재생성 가능한 파생 cache다. `ConversationStore`에 synthetic summary 메시지를 추가하지 않는다.

```python
@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    summary: str
    covered_message_count: int
    covered_prefix_digest: str
    policy_fingerprint: str


@runtime_checkable
class MemoryStateStore(Protocol):
    async def load(self, session_id: str) -> MemorySnapshot | None: ...
    async def save(self, session_id: str, snapshot: MemorySnapshot) -> None: ...
    async def clear(self, session_id: str) -> None: ...
```

요약 규칙은 다음과 같다.

- 완전한 과거 turn으로 구성된 연속 prefix만 요약한다.
- 기존 요약에 새로 제외된 turn만 합치는 incremental summarization을 사용한다.
- `covered_prefix_digest`가 원문과 다르면 cache를 폐기한다.
- `policy_fingerprint`가 다르면 cache를 재생성한다.
- 긴 요약 입력은 bounded batch로 나눠 fold하고, 단일 메시지도 필요하면 fragment로 나눈다.
- 숫자, 날짜, 식별자, 사용자 선호, 결정 사항, 미해결 항목, Tool 관찰을 보존한다.
- 과거 사용자 입력을 system 권한으로 승격하지 않도록 요약은 Tool Call이 없는 assistant 메시지로 model view에만 삽입한다.
- 요약 호출도 기존 global deadline과 usage 집계에 포함한다.
- 요약 실패 시 최근 turn만으로 예산을 만족하면 recent-only로 진행하고, 그렇지 않으면 실행을 실패시킨다.

동일한 모델을 summarizer로 사용해도 요약 요청에는 MemoryPolicy를 다시 적용하지 않는다. 대신 summarizer 자체에 작은 고정 입력·출력 예산을 적용해 재귀 호출을 막는다.

기존 정책은 `InMemoryMemoryStateStore`와 `MemorySnapshot`을 계속 지원한다. 이
경로는 0.5 source compatibility를 위한 것이며 전체 session을 읽는 저장소의
비용을 해결하지 않는다. 긴 영속 session에는 다음 0.6 경로를 사용한다.

## 0.6 durable Context Memory

`DurableSummarizingConversationMemoryPolicy`는 summary가 덮은 absolute message
cursor 이후의 tail만 읽는다. Runtime은 정책이 소유한
`DurableContextHistoryLoader`를 bootstrap에 사용하므로 전체 conversation blob을
materialize하지 않는다.

```python
from moduagent import (
    Agent,
    DurableSummarizingConversationMemoryPolicy,
    InMemoryContextMemoryStateStore,
    InMemoryConversationStore,
    ModelConversationSummarizer,
    ScopedConversationStore,
    TokenBudget,
    VLLMTokenCounter,
)

token_counter = VLLMTokenCounter(model)
conversation_store = ScopedConversationStore(
    InMemoryConversationStore(),
    tenant_id="tenant-a",
    agent_id="support-agent",
)
context_memory = DurableSummarizingConversationMemoryPolicy(
    budget=TokenBudget(
        context_window_tokens=32_768,
        reserved_output_tokens=2_048,
        safety_margin_tokens=1_024,
    ),
    summarizer=ModelConversationSummarizer(
        model=model,
        token_counter=token_counter,
        max_input_tokens=8_192,
        max_output_tokens=512,
    ),
    token_counter=token_counter,
    state_store=InMemoryContextMemoryStateStore(),
    tenant_id="tenant-a",
    agent_id="support-agent",
    history_page_size=256,
    max_uncompacted_messages=2_048,
)

agent = Agent.create(
    name="support-agent",
    model=model,
    instructions="Answer from the bounded conversation context.",
    conversation_store=conversation_store,
    context_memory=context_memory,
)
```

위 인메모리 저장소는 동작 확인용이다. 운영에서는
`RedisMemoryStateStore` 또는 `DatabaseMemoryStateStore`와 영속 conversation
store를 사용한다. Conversation store에는 다음 native pagination 계약과 exact
tenant/Agent scope binding이 필요하다.

```python
async def load_tail(
    session_id: str,
    after_sequence: int = 0,
    limit: int = 100,
) -> ConversationPage: ...

supports_bounded_load_tail = True
supports_tenant_agent_scope = True
tenant_id = "tenant-a"
agent_id = "support-agent"
```

Raw ConversationStore SPI는 `session_id`만 받으므로 tenant A/B가 같은 session ID를
쓰면 충돌한다. `ScopedConversationStore`의 안전한 기본값
`key_mode="shared"`는 tenant+Agent+session을 합친 storage key로 격리한다. 기존
raw key를 유지하는 `key_mode="isolated_legacy"`는 backend namespace 전체가 해당
tenant/Agent 하나에만 전용일 때만 명시한다. Scope 누락/불일치는 history load와
clear를 포함한 store I/O보다 먼저 실패한다.

- `InMemoryConversationStore`는 이 계약을 지원하지만 프로세스 로컬이다.
- `RedisConversationStore`는 Redis list mode에서만 bounded pagination을
  광고한다. get/set blob mode의 `load_tail()`은 호환 편의 기능일 뿐 durable
  loader가 거부한다.
- `DatabaseConversationStore`는 repository가 `load_messages_page(session_id,
  after_sequence, limit)`를 구현할 때만 bounded pagination을 광고한다.

Summary snapshot schema v2의 key는 `tenant_id`, `agent_id`, `session_id`,
`policy_fingerprint`를 delimiter-safe하게 묶는다. Snapshot은 다음을 저장한다.

```text
ConversationSummarySnapshot v2
├── covered_through_sequence + covered_prefix_digest
├── structured_summary
│   ├── summary
│   ├── facts / decisions / preferences / open_items
│   └── tool_observations
├── bounded source_message_ids
└── CAS version
```

모든 문자열, 항목 수와 직렬화 크기에 public 상한이 적용된다. CAS는 writer
race에서 cursor regression을 거부하며, loader는 저장된 source ID window와
현재 conversation prefix를 다시 검증한다. 이 검사는 일반 append/clear race를
탐지하지만 conversation SPI에 session generation token이 없으므로 모든 독립
writer의 ABA clear/recreate를 선형화하지는 못한다. Session 삭제는 summary를
먼저 지우는 다음 경로로 조정한다.

```python
await context_memory.clear_history(conversation_store, session_id)
```

0.5 `MemorySnapshot`은 tenant/Agent identity가 없으므로 automatic migration
source를 반드시 `ScopedLegacyMemoryStateStore`로 scope-bind한다. V2 miss 시
loader는 canonical scoped conversation prefix를 bounded page로 두 번 scan하고
count, legacy digest, source IDs와 v2 digest가 모두 일치할 때만 CAS로 backfill한다.

```python
from moduagent import ScopedLegacyMemoryStateStore

legacy_source = ScopedLegacyMemoryStateStore(
    legacy_memory_state_store,
    tenant_id="tenant-a",
    agent_id="support-agent",
)
context_memory = DurableSummarizingConversationMemoryPolicy(
    # 기존 인자 생략
    tenant_id="tenant-a",
    agent_id="support-agent",
    legacy_state_store=legacy_source,
    max_legacy_migration_messages=100_000,
)
```

Legacy namespace 하나를 여러 scope adapter로 포장하면 안 된다. Prefix가 migration
cap을 넘거나 두 scan이 달라지면 state를 쓰지 않고 fail closed한다. 한 번
성공한 뒤에는 이후 run이 legacy state store를 읽지 않는다.

명시적 copy가 필요하면 canonical store에서 얻은 실제 cursor와 source message
ID를 제공한다.

```python
from moduagent import MemoryStateKey, migrate_memory_snapshot

snapshot_v2 = migrate_memory_snapshot(
    legacy_snapshot,
    key=MemoryStateKey(
        tenant_id="tenant-a",
        agent_id="support-agent",
        session_id=session_id,
        policy_fingerprint=legacy_snapshot.policy_fingerprint,
    ),
    covered_through_sequence=cursor,
    source_message_ids=authoritative_message_ids,
)
```

실제 ID 없이 만든 legacy provenance marker는 즉시 신뢰하지 않는다. 첫 load에서
동일한 bounded two-pass 검증으로 authoritative ID를 backfill할 수 있을 때만 v2로
승격하며, 그렇지 않으면 `ContextHistoryCursorInvalidatedError`로 종료한다. Summary
marker는 `migrate_memory_snapshot()`이 생성한 exact singleton
`legacy-prefix:<covered_prefix_digest>`만 인정하므로 같은 문자열 prefix를 쓰는 일반
v2 source ID와 혼동하지 않는다. Summary v2 decoder는 future schema를 거부하고
v1으로 downgrade하지 않는다.

### ContextAssembler v1

Durable policy는 다음 source를 `ContextItem`으로 만들어 하나의 exact request token
budget 안에서 조립한다.

- required: system/승인된 Skill, current task/run, Tool call+모든 result protocol
  block, Tool schema와 output schema;
- optional: 검증된 conversation summary와 최근 complete turn.

Assembler가 priority와 atomic group을 먼저 선택하고, provider chat template가
비가산적인 경우 전체 request를 다시 세어 newest contiguous turn suffix만
보수적으로 남긴다. 새 summary는 그 exact request에서 실제 선택된 뒤에만 CAS로
cursor를 전진시킨다. 선택되지 않은 candidate는 state를 쓰지 않으며, 이미 저장된
summary나 CAS winner가 현재 request에 들어가지 않으면 compacted prefix를 건너뛴
raw history를 다시 읽지 않고 summary를 이번 request에서 생략한 뒤 이미 검증된
required/recent-only view로 계속한다. `MEMORY_COMPACTED`에는 source별 개수와
token/count만 기록하며 summary 본문, Tool 인자/결과와 source ID는 넣지 않는다.

## Runtime 통합

PLAN/replan, ACT, STEP_RESULT와 FINALIZE 요청은 공통
`_prepare_model_request()`를 거친다. `MemoryPhase.VERIFY`도 public phase 계약에
포함되지만 현재 built-in 검증기는 별도 VERIFY model 요청을 만들지 않는다.

```python
async def _prepare_model_request(
    self,
    context: RunContext,
    request: ModelRequest,
    *,
    phase: MemoryPhase,
    deadline: float,
) -> tuple[ModelRequest, AgentEvent | None]:
    request = replace(
        request,
        messages=compose_skill_prompt(request.messages, context.skill_messages),
    )
    usage_before_memory = context.usage
    memory = await self._within(
        deadline,
        lambda: self.conversation_memory_policy.prepare(
            MemoryRequest(
                run_id=context.run_id,
                session_id=context.request.session_id,
                phase=phase,
                model_request=request,
                protected_from=(
                    context.current_run_start + len(context.skill_messages)
                ),
                user_context=context.request.user_context,
                model_gateway=context.model_gateway,
            )
        ),
    )
    # model_gateway를 사용한 summary 호출은 gateway가 usage를 이미 집계한다.
    # 직접 집계되지 않은 custom summarizer usage만 한 번 더한다.
    if context.usage == usage_before_memory:
        context.usage = context.usage + memory.usage
    prepared = replace(request, messages=tuple(memory.messages))
    compacted = (
        prepared.messages != request.messages
        or memory.summarized_messages > 0
        or memory.dropped_messages > 0
    )
    event = (
        AgentEvent(EventType.MEMORY_COMPACTED, context.run_id, dict(memory.metadata))
        if compacted
        else None
    )
    return prepared, event
```

위 코드는 흐름을 설명한 축약 예시다. 실제 이벤트에는 token·요약·제외 수치도 포함하며, view가 바뀐 경우에만 생성한다.

Legacy 정책의 호출 순서는 다음과 같다.

```text
ConversationStore.load
→ full RunContext 생성 및 current_run_start 기록
→ checkpoint 저장
→ phase별 ModelRequest 생성
→ ConversationMemoryPolicy.prepare
→ bounded ModelRequest 전송
```

Durable 정책은 bootstrap 앞부분만 다음과 같이 바꾼다.

```text
ContextMemoryStateStore.load(composite key)
→ ConversationStore.load_tail(summary cursor 이후, bounded pages)
→ source anchor/state/tail 재검증
→ summary boundary + uncompacted tail로 RunContext 생성
→ phase별 ConversationMemoryPolicy.prepare
```

Checkpoint v5 envelope은 실행 메시지, `current_run_start`, 누적 usage, Engine state와
delegation identity를 저장한다. Built-in Engine state는 v1이다. Summary cache는
checkpoint와 독립된 schema v2 state이며 본문을 checkpoint, event 또는 public
result metadata에 복제하지 않는다. Event envelope v2의 `MEMORY_COMPACTED`도
content-free count와 duration만 기록한다.

Checkpoint v1-v4는 `migrate_checkpoint_payload()`를 통해 v5 root run으로 읽을 수
있지만 0.5.x는 v5를 읽지 못한다. Legacy `MemorySnapshot`에서 summary v2로의
전환은 위의 scope-bound automatic lazy migration, authoritative ID를 제공한
explicit copy, 또는 재생성 중 하나를 사용한다.

0.5.2부터 `LLMPlanGenerator`의 PLAN과 replan 요청도 Runtime의 동일한
request-preparation 경로를 사용한다. `history_limit`가 먼저 공개 대화 후보 수를
제한하고, `ConversationMemoryPolicy`가 그 후보를 token 또는 최근 turn 기준으로
다시 선택한다. 현재 사용자 요청과 Plan protocol 지침은 protected 영역이므로
제거되지 않으며, compaction이 발생하면 `phase="plan"`인
`MEMORY_COMPACTED` 이벤트가 기록된다. 이 계약은 0.6.0에서도 그대로 유지된다.

## 오류와 관측성

Legacy 정책의 주요 예외는 다음과 같다.

- `ConversationMemoryError`: 공통 기반 예외
- `ConversationMemoryOverflowError`: 필수 입력이 예산을 초과함
- `MemoryIntegrityError`: Tool 메시지 관계가 손상됨

Durable 경로는 원인을 문자열로 파싱하지 않도록 `ContextMemoryError` 아래에
cursor invalidation, pagination required, tail overflow, serialization, integrity,
cursor regression과 CAS write conflict 유형을 구분한다. Integrity/cursor 오류는
recent-only fallback으로 숨기지 않고 원래 유형으로 종료한다.

Summary model의 protocol 오류와 model guard 실패는 recent-only fallback으로
변환하지 않고 run을 종료한다. Provider가 `timeout`, `length`,
`max_tokens`로 끝낸 부분 summary도 `model_output_incomplete`로 실패하며,
부분 content를 model context나 `MemoryStateStore` cache에 넣지 않는다.
요약 호출의 일반 transport 실패는 summary boundary가 이미 있는 경우에도 state를
변경하지 않고 예산을 만족한 required/recent-only view로 진행한다. 반대로 모든
`ContextMemoryError`는 무결성/동시성 실패이므로 fallback으로 숨기지 않는다.

`EventType.MEMORY_COMPACTED` 이벤트에는 원문이나 요약 본문을 넣지 않는다.

```python
{
    "phase": "act",
    "original_tokens": 41_000,
    "selected_tokens": 27_500,
    "summarized_messages": 42,
    "dropped_messages": 0,
    "budget_tokens": 29_696,
    "cache_hit": True,
    "duration_seconds": 0.012,
}
```

`original_tokens`와 `selected_tokens`는 provider의 사후 usage가 아니라 설정한
`TokenCounter`가 계산한 전체 `ModelRequest` 크기다. `duration_seconds`는 token
계산과 선택적 요약을 포함한 Context Memory 준비 wall time이다. 이벤트는
PLAN/ACT/FINALIZE에서 같은 필드 의미를 사용하며 원문이나 summary 본문을
포함하지 않는다. Token을 계산하지 않는 RecentTurns 정책의 `0/0`은 미계수다.

권장 metric은 `memory.context_tokens`, `memory.compactions`, `memory.summary_tokens`, `memory.compression_ratio`, `memory.overflow`다. `session_id`는 metric label로 사용하지 않는다.

## 저장소 메모리와의 경계

ConversationMemoryPolicy는 모델 입력 토큰만 제한한다. `InMemoryConversationStore`의 Python 메모리 사용량과 전체 기록 조회 비용은 별도 문제다.

저장소 용량 제어와 0.6 cursor pagination은 다음과 같이 분리한다.

- `InMemoryConversationStore`: `max_sessions`, `max_total_bytes`, lazy periodic TTL sweep, 세션 단위 LRU eviction을 제공한다. 메시지는 caller가 중첩 mapping을 변경해 byte 회계를 우회하지 못하도록 canonical JSON row로 보관하며, `stats()`는 본문 없이 세션 수와 직렬화된 메시지 byte 수만 반환하고 `sweep_expired()`로 즉시 정리할 수 있다. 이 수치는 Python 객체 overhead나 process RSS의 상한이 아니다.
- `RedisConversationStore`: list mode에서 bounded `LRANGE` pagination을 제공한다.
  Blob mode는 전체 기록을 읽으므로 durable loader에 사용할 수 없다.
- `DatabaseConversationStore`: repository의 `load_messages_page()`가 있을 때만
  bounded pagination을 제공한다. Repository는 stable append 순서와 원자적
  cursor 의미를 보장해야 한다.
- 운영 원문 보존: Redis 또는 DB 사용

메시지 수를 임의로 잘라 모델 문제와 저장 문제를 동시에 처리하지 않는다. 원문 retention과 모델 token budget은 서로 다른 설정이어야 한다.
단일 세션 자체가 `max_total_bytes`를 초과하는 append는 기존 데이터를
변경하지 않고 `ConversationStoreCapacityError`로 실패한다. 다른 세션을
eviction하여 공간을 만들 수 있을 때만 현재 append를 유지한다.

## 파일 구조

```text
moduagent/memory/
├── __init__.py
├── base.py          # Protocol, DTO, 예외
├── token.py         # TokenBudget, TokenCounter
├── policies.py      # Full/TokenBudget policy
├── summarizer.py    # ConversationSummarizer
├── state.py         # Legacy MemorySnapshot, MemoryStateStore
└── context/
    ├── models.py    # Summary v2 schema와 composite key
    ├── history.py   # Bounded cursor loader
    ├── policies.py  # Durable summarizing policy
    ├── stores.py    # In-memory/Redis/DB CAS stores
    └── assembler.py # Priority/atomic-group Context assembly
```

통합 대상은 다음과 같다.

- `moduagent/agent.py`: policy 주입
- `moduagent/runtime/runtime.py`: phase별 공통 요청 준비 경로
- `moduagent/runtime/context.py`: `current_run_start`와 memory metadata
- `moduagent/persistence/conversation.py`: pagination SPI와 stable source ID
- `moduagent/persistence/snapshot.py`: checkpoint v5 (summary 본문은 저장하지 않음)
- `moduagent/runtime/events.py`: event v2의 content-free `MEMORY_COMPACTED`
- `moduagent/__init__.py`: 공개 export

## 구현 순서

1. `ConversationMemoryPolicy` 계약과 `FullConversationMemoryPolicy`를 추가하고 Runtime 호출 경로만 분리한다.
2. `RecentTurnsConversationMemoryPolicy`와 정확한 `TokenCounter` 기반의 요약 없는 토큰 윈도우를 추가한다.
3. `ConversationSummarizer`와 `MemoryStateStore`를 추가해 오래된 정보 손실을 줄인다.
4. `InMemoryConversationStore` 용량 제한을 추가하고 durable store pagination은 별도 작업으로 유지한다.
5. PLAN과 replan 모델 호출을 공통 준비 경로로 이동한다.

## 완료 조건

- 예산 이하 요청은 메시지와 호출 순서가 기존과 동일하다.
- RecentTurns 정책은 최근 완료 turn N개와 현재 실행만 정확히 선택한다.
- TokenBudget 정책을 거친 모든 built-in phase 요청이 설정한 input budget 이하이다.
- Tool Call과 모든 결과가 절대 분리되지 않는다.
- system instruction과 현재 실행 메시지는 제거되지 않는다.
- 원본 store, `RunContext`, `AgentResult.messages`는 변경되지 않는다.
- 필수 입력만으로 초과하면 모델 호출 전에 token breakdown과 함께 실패한다.
- 요약 cache hit와 digest 불일치 재생성이 동작한다.
- 요약 usage와 시간이 전체 run usage·timeout에 포함된다.
- Pydantic FINALIZE와 streaming event 순서가 유지된다.
- resume 후 Tool을 중복 실행하지 않는다.
- policy 미설정 시 기존 전체 테스트가 그대로 통과한다.
