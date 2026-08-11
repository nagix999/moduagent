# ModuAgent 0.6 마이그레이션

이 문서는 ModuAgent 0.5.3 애플리케이션을 0.6.0으로 옮길 때 확인할 source,
저장 schema와 운영 제한을 설명한다.

```bash
python -m pip install --upgrade "moduagent==0.6.0"
```

## 요약

기존 `Agent.create()`, 명시적 `Agent(...)`, Standard/Plan profile, Tool, Skill,
`ConversationMemoryPolicy`와 legacy `AgentTool`은 유지된다. 새
`AgentDefinition`, runtime profile, durable Context Memory와 typed delegation은
opt-in이다. 따라서 먼저 기존 API로 회귀 테스트한 뒤 기능별로 전환할 수 있다.

단, 저장 wire schema를 사용하는 시스템은 다음 변경을 먼저 반영해야 한다.

| 저장 경계 | 0.5.3 | 0.6.0 | 읽기 호환성 |
|---|---:|---:|---|
| Checkpoint outer envelope | v4 | v5 | `migrate_checkpoint_payload()`가 v1-v4를 v5 root run으로 변환 |
| Built-in Engine state | v1 | v1 | 변경 없음 |
| Event envelope | v1 | v2 | `AgentEvent.from_dict()`가 v1을 v2 root event로 정규화 |
| Durable Context summary | 없음/legacy `MemorySnapshot` | v2 | scope가 확인된 legacy source는 bounded lazy migration, 그 외에는 명시적 copy/rebuild |
| Conversation message rows | v1 | v1 + cursor pagination SPI | 기존 row 형식 유지 |

0.5.x는 checkpoint v5를 읽지 못한다. 배포 전 v4 백업 또는 별도 namespace를
보존하고, 0.6 writer를 연 뒤에는 0.5 worker가 같은 checkpoint namespace를
소비하지 않게 한다.

## 1. 기존 Agent를 그대로 검증

기본 profile은 Development이므로 definition을 추가하지 않은 기존 Agent는 계속
실행된다.

```python
agent = Agent.create(
    name="assistant",
    model=model,
    instructions="Answer accurately.",
    tools=tools,
)
```

업그레이드 직후에는 다음을 회귀 테스트한다.

- Standard와 strict Plan 성공/실패, Tool retry/repair, cancellation;
- checkpoint가 없는 실행과 v4 checkpoint resume;
- custom event decoder가 additive v2 field를 허용하는지;
- Redis/DB conversation adapter의 append와 idempotent append;
- custom Tool/Skill/Engine이 새 runtime metadata를 payload로 오인하지 않는지.

## 2. AgentDefinition과 profile 도입

`AgentDefinition`은 instruction, model route, Tool/Skill, 입출력 계약, memory,
authorization와 실행 제한의 배포 의미를 정확한 SemVer로 고정한다.
`RuntimeBindings`에는 실제 model/store/authorizer/telemetry 객체를 둔다.

Development에서는 `semantic_digests`를 생략할 수 있다. Production profile은
전체 semantic digest, APPROVED/ACTIVE definition, tenant/principal provider,
영속 저장소, 명시적 authorizer와 telemetry를 요구하며 unsafe 조합을 생성 시점에
거부한다. Context Memory도 `MemoryContextBound`를 노출하는 explicit bounded
policy여야 하며 unbounded Full policy는 허용되지 않는다.

```python
from moduagent import Agent, ProductionProfile, RuntimeBindings

agent = Agent.create(
    name="approved-agent",
    model=model,
    instructions=instructions,
    definition=definition,
    runtime_profile=ProductionProfile(),
    runtime_bindings=RuntimeBindings(
        conversation_store=conversation_store,
        checkpoint_store=checkpoint_store,
        event_sink=event_sink,
        diagnostic_sink=diagnostic_sink,
        tenant_context_provider=tenant_provider,
        principal_context_provider=principal_provider,
        # 다른 필수 binding도 배포 구성에 맞게 지정한다.
    ),
)
```

Production profile을 한 번에 켜기보다 Development에서 definition mismatch를 먼저
제거하고, Test profile에서 deterministic 구성 검증 후 Production 요구를 채운다.
Test profile의 `RuntimeAttestation`은 애플리케이션이 선언한 외부 I/O·결정성 사실을
canonical digest로 묶는 배포 메타데이터다. 이는 서명이나 독립적인 보안 검증이
아니므로 prompt/요청에서 만들지 말고 신뢰된 CI 또는 composition 코드에서
`RuntimeAttestation.create(...)`로 생성한다.

## 3. Legacy AgentTool에서 typed delegation으로 전환

legacy `AgentTool(child)`은 자동 변환되지 않는다. 자식에게 `AgentDefinition`을
부여하고 exact-version registry에 등록한 뒤 같은 `DelegationCoordinator`에서
`child.as_tool(...)`을 만든다. 전체 예제는 [Agent 위임](delegation.md)에 있다.

전환 시 다음 차이가 의도적으로 적용된다.

- 모델이 선택하는 인자는 Pydantic `input_model`에 한정된다. lineage, tenant,
  principal, deadline과 child session은 runtime-owned metadata다.
- caller/callee topology, `callable_by`, data classification, cycle/depth와 execution
  group 전체 budget을 자식 model I/O 전에 검사한다.
- 기본 child session은 parent와 분리된다. legacy의 동일 `session_id` 공유에
  의존했다면 필요한 정보는 typed request에 명시적으로 넣는다.
- 실패한 delegation은 안정적인 payload-free Tool failure가 된다. 원본 child
  예외나 output을 분기 조건으로 사용하지 않는다.

다중 worker에서는 `BudgetStateStore`와 `DelegationReceiptStore`의 원자적 CAS,
`durable=True`, 모든 worker에서 동일한 32-byte 이상 HMAC secret과 namespace가
필수다. 0.6은 이 SPI와 인메모리 기준 구현을 제공하지만 특정 분산 DB schema나
queue를 제공하지 않는다. active execution group이 있는 동안 secret을 제자리
회전하지 말고 이전 namespace를 drain한 뒤 새 namespace로 전환한다.

Receipt는 중복 실행을 fence하고 검증된 완료 결과를 replay하지만 외부 시스템의
side effect와 하나의 transaction을 만들지 않는다. 쓰기 작업은 애플리케이션의
idempotency key/outbox/업무 receipt가 여전히 필요하다.

## 4. Checkpoint v5

v5는 v4 공통/Engine/finalization 상태를 유지하면서 다음 content-free identity를
추가한다.

- `run_lineage`, `execution_group_id`;
- exact `agent_ref`와 `agent_definition_fingerprint`;
- `delegation_id`, `parent_tool_call_id`, `budget_lease_id`;
- tenant/principal의 raw 값이 아닌 SHA-256 scope digest.

`migrate_checkpoint_payload()`는 v1-v3를 기존 Engine state로 정규화한 뒤 v5로
만들고, v4를 depth 0 root run으로 copy-migrate한다. `RunSnapshot.from_dict()`는
outer v4/v5만 직접 받으므로, 버전을 모르는 외부 payload에는 항상 migration
entry point를 사용한다.

```python
from moduagent.persistence import migrate_checkpoint_payload

snapshot = migrate_checkpoint_payload(payload)
assert snapshot.schema_version == 5
```

과거 checkpoint에는 exact AgentRef, tenant/principal scope 또는 delegation lease가
없으므로 migration이 이를 추측하지 않는다. 모두 root/unbound 값으로 시작한다.
따라서 0.5에서 중단된 run을 0.6 delegated child로 바꾸어 resume하지 말고, 기존
호환 Agent로 Development 환경에서 완료하거나 새 root run을 시작한다. Production
profile은 exact AgentRef가 없는 legacy checkpoint를 자동 채택하지 않고 fail
closed한다.

## 5. Event v2

v2는 기존 `type`, `run_id`, `data`, timestamp와 sequence를 유지하고 다음 상관관계
필드를 추가한다.

```text
execution_group_id, root_run_id, parent_run_id, child_run_id,
delegation_id, agent_id, agent_version, depth
```

v1 event를 읽으면 `root_run_id=run_id`, `execution_group_id=run_id`, `depth=0`인 v2
root event가 된다. Consumer는 알 수 없는 additive field와 새 `DELEGATION_*`
event type을 허용하고 `(run_id, sequence)` 또는 `event_id` 기준 중복 처리 정책을
유지해야 한다. 상관관계 envelope에는 prompt, Tool 인자/결과와 raw identity를
복사하지 않는다.

## 6. Durable Context Memory summary v2

0.6의 `DurableSummarizingConversationMemoryPolicy`는 전체 session을 매번 읽지 않고
summary cursor 이후의 bounded tail만 읽는다. 사용하려면 conversation store가
`supports_bounded_load_tail=True`인 `PaginatedConversationStore`이면서 tenant와
Agent에 명시적으로 묶여 있어야 한다. 기존 SPI는 public `session_id`만 key로
사용하므로 raw store를 그대로 공유하면 같은 session ID가 tenant 사이에서
충돌한다.

```python
from moduagent import ScopedConversationStore

conversation_store = ScopedConversationStore(
    raw_conversation_store,
    tenant_id="tenant-a",
    agent_id="support-agent",
)  # key_mode="shared"가 안전한 기본값
```

기본 `key_mode="shared"`는 tenant, Agent, public session을 합친 별도 storage key를
사용한다. 기존 raw session key를 그대로 읽어야 할 때만
`key_mode="isolated_legacy"`를 명시할 수 있으며, 이 모드는 backend namespace
전체가 해당 tenant/Agent 하나에만 전용이라는 운영 보장이 있어야 한다. Scope가
없거나 policy와 일치하지 않으면 durable loader는 conversation/state read보다 먼저
fail closed한다.

- `InMemoryConversationStore`: 지원하지만 개발/테스트 전용;
- `RedisConversationStore`: Redis list mode에서만 지원;
- `DatabaseConversationStore`: repository가 `load_messages_page()`를 구현할 때만
  지원. 전체 blob을 읽어 page처럼 자르는 fallback은 거부된다.

Summary v2 key는 tenant, Agent, session, policy fingerprint를 함께 묶는다. Snapshot은
absolute cursor, chained prefix digest, bounded source message IDs, CAS version과
크기가 제한된 `ConversationSummary` 구조를 저장한다. 원문 conversation이 source of
truth이며 summary는 재생성 가능한 cache다.

기존 `MemorySnapshot`은 tenant/Agent 정보가 없으므로 session-only store를 직접
자동 migration source로 사용할 수 없다. 전용 legacy namespace임을 명시하는
`ScopedLegacyMemoryStateStore`로 감싸고 durable policy의
`legacy_state_store=`에 전달하면 v2 miss 시 한 번만 lazy migration한다.

```python
from moduagent import (
    DurableSummarizingConversationMemoryPolicy,
    ScopedLegacyMemoryStateStore,
)

legacy_source = ScopedLegacyMemoryStateStore(
    legacy_memory_state_store,
    tenant_id="tenant-a",
    agent_id="support-agent",
)
context_memory = DurableSummarizingConversationMemoryPolicy(
    # budget, summarizer, state_store, token_counter 생략
    tenant_id="tenant-a",
    agent_id="support-agent",
    legacy_state_store=legacy_source,
    max_legacy_migration_messages=100_000,
)
```

Lazy migration은 canonical `ScopedConversationStore.load_tail()`로 legacy covered
prefix를 page 단위로 두 번 읽는다. 두 pass의 message count, legacy digest,
store-issued source ID와 v2 chained digest가 모두 같을 때만 CAS로 v2 snapshot을
쓴다. 전체 `ConversationStore.load()`는 호출하지 않는다. Prefix가 설정한
`max_legacy_migration_messages`를 넘거나 digest/count가 다르거나 CAS winner가
검증되지 않으면 fail closed한다. 같은 unpartitioned legacy state store를 여러
tenant/Agent scope adapter로 포장해서는 안 된다.

명시적 copy가 필요하면 `migrate_memory_snapshot()`에 canonical store에서 얻은
`covered_through_sequence`와 `source_message_ids`를 함께 제공한다.

```python
from moduagent import MemoryStateKey, migrate_memory_snapshot

summary_v2 = migrate_memory_snapshot(
    legacy_snapshot,
    key=MemoryStateKey(
        tenant_id="tenant-a",
        agent_id="support-agent",
        session_id="session-42",
        policy_fingerprint=legacy_snapshot.policy_fingerprint,
    ),
    covered_through_sequence=cursor,
    source_message_ids=authoritative_message_ids,
)
```

Source ID 없이 만든 `legacy-prefix:*` marker는 즉시 신뢰되지 않는다. 첫 loader가
동일한 bounded two-pass canonical 검증으로 실제 source IDs를 backfill할 수 있을
때만 사용 가능하며, 검증할 수 없으면 `ContextHistoryCursorInvalidatedError`로
fail closed한다. Legacy marker로 인식하는 값은
`migrate_memory_snapshot()`이 만든 exact singleton
`legacy-prefix:<covered_prefix_digest>`뿐이다. 같은 문자열 prefix를 쓰는 일반 v2
source ID는 legacy state로 오인하지 않는다. Conversation을 clear할 때는 scope를 먼저 검증하는
`policy.clear_history(store, session_id)`로 v2/legacy summary state를 먼저 지운다.

모델 request에서는 ContextAssembler v1이 system/Skill, task projection,
conversation summary, 최근 complete turn, 현재 run과 Tool/protocol block, Tool/output
schema를 하나의 exact token budget으로 조립한다. 현재 run과 protocol/schema는
required atomic group이며 summary와 최근 turn만 우선순위에 따라 빠질 수 있다.
새 summary는 exact request에서 실제 선택된 뒤에만 CAS 저장된다. 이미 저장된
summary 또는 CAS winner가 현재 required request와 함께 들어갈 수 없으면 raw
prefix를 다시 읽지 않고 해당 summary를 이번 request에서 생략한 뒤, 이미 예산을
만족한 required/recent-only view로 계속한다. 일반 summary transport 실패도 같은
bounded fallback을 사용하지만 protocol/model-guard/`ContextMemoryError`는 원래
typed error로 종료한다.

Summary v2 decoder는 future schema를 받지 않으며 v2를 v1로 downgrade하지 않는다.
Long-Term Memory나 cross-session 검색도 추가하지 않는다.

## 배포 순서

1. 0.6 reader로 v1-v4 checkpoint와 v1 event fixture를 검증한다.
2. 소비자가 event v2 additive field와 새 delegation event를 허용하게 한다.
3. 기존 Agent를 Development profile로 배포하고 definition mismatch를 제거한다.
4. Context Memory를 새 namespace에서 canary하고 cursor/reset 회귀를 검증한다.
5. Delegation은 읽기 전용 자식부터 시작하고 aggregate budget과 cancellation을
   부하 테스트한다.
6. durable store/HMAC namespace, backup과 rollback cutover가 준비된 뒤 v5 writer와
   Production profile을 활성화한다.

Rollback 시 이미 만들어진 v5 checkpoint, v2 summary와 active delegation receipt를
0.5가 처리하게 하지 않는다. 기존 namespace를 유지한 채 0.6 실행을 drain 또는
수동 조정하고, 검증된 v4 backup을 사용하는 것이 안전한 경계다.
