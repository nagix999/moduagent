# ConversationMemoryPolicy 설계

상태: 구현 완료. PLAN 공통 준비 경로와 저장소 용량·pagination은 후속 범위다.

## 목적

대화 원문은 보존하면서 모델에 전달하는 요청만 context window 안으로 제한한다. 긴 세션에서도 vLLM의 context 초과와 입력 증가에 따른 timeout을 예방하고, Tool Calling·구조화 출력·체크포인트 동작은 유지한다.

이 정책은 다음 두 대상을 구분한다.

- `ConversationStore`: 사용자·assistant·Tool 메시지 원문을 보존하는 source of truth
- `ConversationMemoryPolicy`: 각 모델 요청에 사용할 제한된 메시지 view를 생성하는 계층

```text
ConversationStore ── 전체 원문 ──> RunContext
                                      │
                                      │ ModelRequest
                                      ▼
                          ConversationMemoryPolicy
                                      │
                                      │ bounded messages
                                      ▼
                                  ModelClient
```

정책은 `RunContext.messages`, `new_messages`, `ConversationStore`를 수정하지 않는다. 따라서 감사 기록, `AgentResult.messages`, 체크포인트에는 원문이 남는다.

## 공개 API

새 모듈은 `moduagent.memory`에 둔다.

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Protocol, runtime_checkable

from moduagent.messages import Message, Usage
from moduagent.models import ModelRequest


class MemoryPhase(str, Enum):
    ACT = "act"
    FINALIZE = "finalize"


@dataclass(frozen=True, slots=True)
class MemoryRequest:
    run_id: str
    session_id: str
    phase: MemoryPhase
    model_request: ModelRequest
    protected_from: int
    user_context: Mapping[str, object] = field(default_factory=dict)


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

`Agent`에는 기존 API와 호환되는 keyword-only 인자를 추가한다.

```python
conversation_memory_policy: ConversationMemoryPolicy | None = None
```

`None`은 기존 동작과 같은 `FullConversationMemoryPolicy`를 사용한다. 운영 환경에서는 `TokenBudgetConversationMemoryPolicy`를 명시한다.

## 기본 구현

### FullConversationMemoryPolicy

입력 메시지를 그대로 반환한다. 기존 사용자와 테스트의 동작을 보존하고 점진적으로 배포하기 위한 identity policy다.

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

운영 환경에서는 실제 모델 tokenizer를 사용한다. 제공되는 `VLLMTokenCounter`는 vLLM의 chat 메시지와 Tool을 받는 `/tokenize` API를 사용한다. 로컬 휴리스틱은 `ApproximateTokenCounter`로 명확히 이름 붙이고 큰 safety margin을 적용한다.

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

현재 실행의 Tool 결과가 너무 큰 경우에도 기본값은 오류다. 0.2.0에는 Tool 결과를 의미 보존 방식으로 축약하는 public reducer API가 없으므로, Tool 자체의 pagination과 `max_result_bytes`를 사용해 결과 크기를 제한한다.

## 요약과 상태 저장

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

초기 구현은 `InMemoryMemoryStateStore`를 제공한다. Redis/DB 대화 저장소를 사용하는 다중 인스턴스 환경에서는 대응하는 durable state store와 prefix digest 검증이 필요하다.

## Runtime 통합

ACT와 FINALIZE 요청은 모두 공통 `_prepare_model_request()`를 거친다.

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
            )
        ),
    )
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

호출 순서는 다음과 같다.

```text
ConversationStore.load
→ full RunContext 생성 및 current_run_start 기록
→ checkpoint 저장
→ phase별 ModelRequest 생성
→ ConversationMemoryPolicy.prepare
→ bounded ModelRequest 전송
```

체크포인트는 전체 실행 메시지와 `current_run_start`, 누적 usage를 저장한다. 요약 cache의 `policy_fingerprint`와 covered-prefix digest는 `MemoryStateStore`의 `MemorySnapshot`이 관리하며 0.2.0 checkpoint에는 snapshot 식별자를 중복 저장하지 않는다. Resume 후 cache가 현재 Policy나 원문 prefix와 맞지 않으면 MemoryPolicy가 cache를 사용하지 않고 다시 요약한다.

`LLMPlanGenerator`는 현재 과거 대화 전체가 아니라 현재 `request.input`만 모델에 보내므로 긴 history에 의한 context 초과 대상은 아니다. 다만 “아까 말한 내용”처럼 과거 참조를 포함하는 계획의 품질은 떨어질 수 있다. 후속 단계에서 PLAN도 동일한 request-preparation 경로를 사용하도록 Model 호출 책임을 Runtime으로 이동한다.

## 오류와 관측성

새 예외는 다음 세 개로 제한한다.

- `ConversationMemoryError`: 공통 기반 예외
- `ConversationMemoryOverflowError`: 필수 입력이 예산을 초과함
- `MemoryIntegrityError`: Tool 메시지 관계가 손상됨

`EventType.MEMORY_COMPACTED`를 추가한다. 이벤트에는 원문이나 요약 본문을 넣지 않는다.

```python
{
    "phase": "act",
    "original_tokens": 41_000,
    "selected_tokens": 27_500,
    "summarized_messages": 42,
    "dropped_messages": 0,
    "budget_tokens": 29_696,
    "cache_hit": True,
}
```

권장 metric은 `memory.context_tokens`, `memory.compactions`, `memory.summary_tokens`, `memory.compression_ratio`, `memory.overflow`다. `session_id`는 metric label로 사용하지 않는다.

## 저장소 메모리와의 경계

ConversationMemoryPolicy는 모델 입력 토큰만 제한한다. `InMemoryConversationStore`의 Python 메모리 사용량과 전체 기록 조회 비용은 별도 문제다.

후속 저장소 개선은 다음과 같이 분리한다.

- `InMemoryConversationStore`: `max_sessions`, `max_total_bytes`, 주기적 TTL sweep, 세션 단위 LRU eviction
- `RedisConversationStore`/`DatabaseConversationStore`: summary cursor 이후 tail 조회와 pagination
- 운영 원문 보존: Redis 또는 DB 사용

메시지 수를 임의로 잘라 모델 문제와 저장 문제를 동시에 처리하지 않는다. 원문 retention과 모델 token budget은 서로 다른 설정이어야 한다.

## 파일 구조

```text
moduagent/memory/
├── __init__.py
├── base.py          # Protocol, DTO, 예외
├── token.py         # TokenBudget, TokenCounter
├── policies.py      # Full/TokenBudget policy
├── summarizer.py    # ConversationSummarizer
└── state.py         # MemorySnapshot, MemoryStateStore
```

통합 대상은 다음과 같다.

- `moduagent/agent.py`: policy 주입
- `moduagent/runtime/runtime.py`: ACT/FINALIZE 공통 준비 경로
- `moduagent/runtime/context.py`: `current_run_start`와 memory metadata
- `moduagent/persistence/checkpoint.py`: resume용 memory 필드
- `moduagent/runtime/events.py`: `MEMORY_COMPACTED`
- `moduagent/__init__.py`: 공개 export

## 구현 순서

1. `ConversationMemoryPolicy` 계약과 `FullConversationMemoryPolicy`를 추가하고 Runtime 호출 경로만 분리한다.
2. `RecentTurnsConversationMemoryPolicy`와 정확한 `TokenCounter` 기반의 요약 없는 토큰 윈도우를 추가한다.
3. `ConversationSummarizer`와 `MemoryStateStore`를 추가해 오래된 정보 손실을 줄인다.
4. `InMemoryConversationStore` 용량 제한과 durable store pagination을 별도 작업으로 추가한다.
5. PLAN 모델 호출을 공통 준비 경로로 이동한다.

## 완료 조건

- 예산 이하 요청은 메시지와 호출 순서가 기존과 동일하다.
- RecentTurns 정책은 최근 완료 turn N개와 현재 실행만 정확히 선택한다.
- 모든 ACT/FINALIZE 요청이 설정한 input budget 이하이다.
- Tool Call과 모든 결과가 절대 분리되지 않는다.
- system instruction과 현재 실행 메시지는 제거되지 않는다.
- 원본 store, `RunContext`, `AgentResult.messages`는 변경되지 않는다.
- 필수 입력만으로 초과하면 모델 호출 전에 token breakdown과 함께 실패한다.
- 요약 cache hit와 digest 불일치 재생성이 동작한다.
- 요약 usage와 시간이 전체 run usage·timeout에 포함된다.
- Pydantic FINALIZE와 streaming event 순서가 유지된다.
- resume 후 Tool을 중복 실행하지 않는다.
- policy 미설정 시 기존 전체 테스트가 그대로 통과한다.
