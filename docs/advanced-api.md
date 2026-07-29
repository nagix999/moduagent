# ModuAgent Advanced API

이 문서는 0.4의 확장 계약을 설명합니다. 대부분의 애플리케이션은 [Core API](core-api.md)의 `Agent`와 실행 Profile만 사용하면 됩니다.

## 실행 경계

0.4의 실행 구조는 세 역할로 나뉩니다.

```text
Agent → RunCoordinator → ExecutionEngine → RuntimeServices
```

- `Agent`는 `run`, `stream`, `stream_all`, `resume`, `inspect`를 제공하는 facade입니다.
- `RunCoordinator`는 session lock, deadline, 공통 상태, terminal 처리와 영속화를 담당합니다.
- Standard와 Plan Engine은 모델 요청 순서와 상태 전이를 소유합니다.
- 공통 service는 모델, Tool, Context, 최종화, 저장, 이벤트 경계를 제공합니다.

Engine은 전체 deadline과 Tool 예산, 공개/내부 이벤트 구분, 공개 응답의 단일 저장·방출 불변 조건을 지켜야 합니다. Plan phase나 Tool repair 의미를 Coordinator에 추가하면 안 됩니다.

저수준 계약은 `moduagent.execution` 하위 namespace에 있습니다. custom Engine은 해당 버전의 `ExecutionEngine`, `EngineStateCodec`, `EngineContext`, `ExecutionServices` 계약을 구현하고 `engine_id`, `state_version`, `state_codec`, `configuration`, `required_capabilities`, `retain_terminal_checkpoint`를 명시해야 합니다. 조립 단계에서는 이 선언과 모델 capability를 fail-fast로 검증합니다. 이 API는 Advanced이며 Core보다 더 엄격한 version 검토가 필요합니다.

## Tool 안전 Profile

`ToolSafetyProfile`은 프레임워크가 자동 수행할 수 있는 복구 범위만 표현합니다.

```python
from moduagent import function_tool
from moduagent.tools import ToolSafetyProfile


@function_tool(
    safety_profile=ToolSafetyProfile(
        same_call_retry_safe=True,
        changed_argument_repair_safe=True,
        timeout_retry_safe=False,
    ),
)
def search_records(query: str, limit: int = 100) -> list[dict]:
    return backend.search(query=query, limit=limit)
```

| 필드 | 의미 |
|---|---|
| `same_call_retry_safe` | 같은 Tool과 같은 유효 인자를 자동 재실행할 수 있음 |
| `changed_argument_repair_safe` | 모델이 교정한 다른 유효 인자로 같은 Tool을 재실행할 수 있음 |
| `timeout_retry_safe` | timeout 뒤에도 이전 실행과 겹치지 않게 안전하게 재시도할 수 있음 |

세 값의 기본값은 모두 `False`입니다. 기존 `idempotent`, `repair_safe`, `timeout_retry_safe` 인자는 같은 Profile로 변환되지만 명시적 `safety_profile`과 혼용할 수 없습니다. 어떤 설정도 transaction이나 end-to-end exactly-once를 보장하지 않습니다.

## 실패 분류와 Projection

custom Tool은 예상한 예외만 안정적인 분류로 변환할 수 있습니다.

```python
from moduagent.tools import (
    ToolErrorType,
    ToolFailureClassification,
    ToolRecoveryAction,
)


def classify_search_error(
    exc: Exception,
) -> ToolFailureClassification | None:
    if not isinstance(exc, ValueError):
        return None
    return ToolFailureClassification(
        error_type=ToolErrorType.EXECUTION_ERROR,
        stable_reason="invalid_query_syntax",
        retryable=False,
        recovery_directive=ToolRecoveryAction.REPAIR_CALL,
        safe_message="The query is invalid; correct its arguments.",
    )


@function_tool(
    safety_profile=ToolSafetyProfile(
        changed_argument_repair_safe=True,
    ),
    failure_classifier=classify_search_error,
)
def search_records(query: str) -> list[dict]:
    return backend.search(query)
```

`stable_reason`은 backend 원문 대신 운영자가 정의한 제한된 식별자를 사용하세요. `safe_message`는 호출자가 모델 공개에 안전하다고 보증한 경우에만 넣습니다.

내부의 `InternalToolFailure`는 `FailureProjector`를 통해 `SafeToolFailureView`로만 모델·checkpoint 경계를 통과합니다. 원본 exception, SQL, 접속 문자열, token, 고객 데이터, 내부 경로·schema를 분류 message에 넣지 마세요. `diagnostic_ref`는 로컬 진단용이며 projection 대상이 아닙니다.

## ToolRuntime과 복구 소유권

`ToolRuntime`은 다음 작업만 소유합니다.

- registry 조회와 인자 validation
- authorization
- 실행 timeout과 결과 정규화
- Profile이 허용한 동일 호출 retry
- `ToolBatchOutcome` 생성

`ToolBatchOutcome`에는 요청 call, 각 결과, 내부 failure와 정제된 failure view가 일대일로 대응됩니다. `success_count`, `failure_count`, `partial_success`, `retry_exhausted`로 batch 상태를 판정할 수 있습니다.

strict Plan의 `ToolRecoveryController`는 outcome을 보고 same-step repair, replan, terminal failure를 결정합니다. repair는 다음 조건을 모두 만족해야 합니다.

- 실패 call과 같은 Tool 하나만 요청
- 이전에 사용하지 않은 새 call ID
- 요청 인자와 validation 후 유효 인자가 모두 이전과 다름
- 단계의 허용 Tool scope 안에 있음
- partial-success batch가 아님
- `changed_argument_repair_safe`와 남은 repair 예산이 있음

동일 호출 retry, 변경 인자 repair, replan은 서로의 예산을 소비하지 않습니다.

## Memory와 Skill

Memory는 모델 요청 view를 만들며 Engine state를 소유하지 않습니다. 긴 대화에는 `RecentTurnsConversationMemoryPolicy`, `TokenBudgetConversationMemoryPolicy`, `SummarizingConversationMemoryPolicy`를 선택적으로 사용합니다. 자세한 내용은 [ConversationMemoryPolicy 설계](conversation-memory-policy.md)를 참고하세요.

Skill은 phase별 instruction과 제한된 resource Tool을 제공하지만 business Tool 권한을 추가하지 않습니다. 유효 Tool 범위는 등록 Tool, Skill의 `allowed-tools`, `ToolAuthorizer`가 모두 허용한 교집합입니다. 자세한 내용은 [Agent Skills 사용과 보안](skills.md)을 참고하세요.

## Checkpoint state codec

checkpoint outer envelope와 Engine state version은 독립적입니다. codec은 JSON-safe mapping만 저장하며 persistence 계층은 Plan 도메인 객체를 import하지 않습니다.

```python
from moduagent.persistence.snapshot import EngineStateCodec
```

codec은 `engine_id`, `state_version`과 `encode`, `decode`, `migrate` 계약을 제공합니다. 저장 경계는 encode/decode roundtrip으로 상태를 검증합니다. migration은 입력 payload를 변경하지 않고 완전히 검증된 새 state를 반환해야 합니다. Engine-specific state의 의미가 바뀌면 outer schema가 같아도 `state_version`을 올립니다.

custom Engine도 모델 호출은 공통 ModelGateway, Tool 호출은 `ExecutionServices.execute_tool_batch()`, 공개 최종 응답은 ResultFinalizer 서비스를 사용해야 합니다. Tool 서비스는 마지막 Engine snapshot을 이용해 `manual_required` invocation intent를 먼저 저장하며, 이 저장이 실패하면 Tool을 호출하지 않습니다. 저수준 Tool executor를 직접 우회하는 Engine은 동일한 authorization, failure projection, deadline, pre-invocation checkpoint 불변 조건을 직접 책임져야 합니다.

## 의도적으로 포함하지 않은 기능

0.4 코어는 다음 기능을 구현하지 않습니다.

- 범용 DAG/Graph 실행
- 분산 queue와 scheduler
- 여러 Agent의 자유 대화, 투표, 토론 protocol
- 범용 Human approval workflow
- Tool side effect의 end-to-end exactly-once

필요하면 외부 orchestrator 또는 별도 Engine에서 구현하되 ModuAgent의 deadline, Tool authorization, failure projection, checkpoint 안전 계약을 유지해야 합니다.
