# Agent 위임

ModuAgent 0.6의 권장 Agent-to-Agent 경로는 자식 Agent를 타입이 있는 Tool로
노출하는 `Agent.as_tool()`이다. 자식은 정확한 `AgentRef` 버전으로 고정되고,
`DelegationCoordinator`가 자식 모델 호출 전에 topology, 인가, cycle/depth,
deadline, 전체 실행 예산, session namespace와 receipt를 검사한다.

기존 `AgentTool`은 0.x 호환을 위해 남아 있지만 실행 그룹 전체 예산, cycle/depth
검사, 격리된 자식 session과 durable receipt를 제공하지 않는다. Production
profile은 legacy `AgentTool`을 거부한다.

## 실행 가능한 최소 예제

다음 예제는 네트워크를 사용하지 않는다. 실제 애플리케이션에서는 두 model
class만 vLLM 등의 `ModelClient`로 교체한다.

```python
import asyncio

from pydantic import BaseModel, ConfigDict

from moduagent import (
    Agent,
    AgentDefinition,
    AgentEndpoint,
    DefinitionStatus,
    InMemoryAgentRegistry,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    RunLimits,
    RuntimeBindings,
    ToolCall,
)
from moduagent.delegation import DelegationCoordinator, DelegationPolicy


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str


class ResearchAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str


class ChildModel:
    capabilities = ModelCapabilities(streaming=False)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(Message.assistant('{"answer":"verified evidence"}'))


class ParentModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self) -> None:
        self.turn = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.turn += 1
        if self.turn == 1:
            return ModelResponse(
                Message.assistant(""),
                tool_calls=(
                    ToolCall(
                        "delegate-1",
                        "ask_specialist",
                        {"question": "verify the deployment"},
                    ),
                ),
            )
        return ModelResponse(Message.assistant("parent completed"))


def definition(
    agent_id: str,
    *,
    tool_refs: tuple[str, ...] = (),
    callable_by: frozenset[str] = frozenset(),
) -> AgentDefinition:
    return AgentDefinition(
        agent_id=agent_id,
        version="1.0.0",
        description=f"{agent_id} endpoint",
        instructions_ref=f"instructions/{agent_id}/1",
        execution_profile="standard",
        model_route=f"model/{agent_id}",
        tool_refs=tool_refs,
        skill_refs=(),
        input_contract_ref=f"contract/{agent_id}/input/1",
        output_contract_ref=f"contract/{agent_id}/output/1",
        memory_policy_ref="memory/full/development",
        authorization_policy_ref="policy/delegation/1",
        data_classification="internal",
        side_effect_level="none",
        approval_requirement="none",
        callable_by=callable_by,
        limits=RunLimits(),
    )


async def main() -> None:
    parent_definition = definition(
        "supervisor",
        tool_refs=("ask_specialist",),
    )
    child_definition = definition(
        "specialist",
        callable_by=frozenset({"supervisor"}),
    )
    child = Agent.create(
        name="specialist",
        model=ChildModel(),
        instructions="Return verified evidence as JSON.",
        output=ResearchAnswer,
        definition=child_definition,
    )

    registry = InMemoryAgentRegistry()
    registry.register(
        child_definition,
        AgentEndpoint(handler=child, approved=True),
        status=DefinitionStatus.ACTIVE,
    )
    coordinator = DelegationCoordinator(
        registry=registry,
        policy=DelegationPolicy(
            allowed_edges={"supervisor": {"specialist"}},
            allowed_tenants={"tenant-a"},
            allowed_principals={"analyst-1"},
        ),
    )

    ask_specialist = child.as_tool(
        coordinator=coordinator,
        caller=parent_definition.ref,
        input_model=ResearchRequest,
        output_model=ResearchAnswer,
        name="ask_specialist",
    )
    parent = Agent.create(
        name="supervisor",
        model=ParentModel(),
        instructions="Ask the specialist once, then answer.",
        tools=(ask_specialist,),
        definition=parent_definition,
        runtime_bindings=RuntimeBindings(
            tenant_context_provider=lambda: "tenant-a",
            principal_context_provider=lambda: "analyst-1",
        ),
    )

    result = await parent.run("verify", session_id="parent-session")
    print(result.unwrap())


asyncio.run(main())
```

출력은 `parent completed`다. 실제 parent model은 `ask_specialist`의 JSON schema로
`ResearchRequest`를 만들고, Tool 결과로 검증된 `ResearchAnswer`만 받는다. lineage,
tenant/principal, deadline, lease와 child session ID는 모델 인자가 아니며 runtime이
Tool metadata로 전달한다.

## 정의와 인가

- `AgentDefinition.version`은 정확한 SemVer다. registry는 `latest` 별칭 없이
  `AgentRef(agent_id, version)`를 해석한다.
- coordinator의 `allowed_edges`가 허용해야 하며, 자식 definition의
  `callable_by`가 비어 있지 않으면 caller도 그 목록에 있어야 한다.
  tenant/principal 허용 목록과 요청/자식의 data classification도 일치해야 한다.
- `DefinitionStatus.APPROVED` 또는 `ACTIVE`인 endpoint만 실행할 수 있다.
- Development에서는 `semantic_digests`를 생략할 수 있다. Production에서는
  instruction, model capability, Tool, Skill, 입출력 계약, memory와 authorization
  정책의 canonical SHA-256 digest를 모두 고정해야 한다.

## 실행 그룹 예산과 session

```python
from moduagent.delegation import ExecutionGroupLimits, SessionStrategy

limits = ExecutionGroupLimits(
    max_depth=2,
    max_delegations=8,
    max_parallel_delegations=3,
    max_delegations_per_agent=3,
    max_total_model_turns=30,
    max_total_tool_calls=24,
    timeout_seconds=120,
)

coordinator = DelegationCoordinator(
    registry=registry,
    policy=policy,
    limits=limits,
)

delegated_tool = child.as_tool(
    coordinator=coordinator,
    caller=parent_definition.ref,
    input_model=ResearchRequest,
    output_model=ResearchAnswer,
    session_strategy=SessionStrategy.ISOLATED,
)
```

root와 모든 child의 실제 model attempt와 Tool call이 같은 execution-group budget을
소비한다. Provider retry도 model turn을 하나 소비한다. 한 parent의 delegated
Tool들은 동일한 coordinator를 사용해야 한다.

`ISOLATED`가 기본값이며 delegation마다 opaque child session을 만든다.
`PER_PARENT_SESSION`은 같은 tenant, parent session, 자식 버전 조합에서 session을
재사용한다. `SHARED`는 `SessionKeyFactory(..., allow_shared=True)`로 명시적으로
열기 전에는 거부되며 Production profile에서도 금지된다.

## 운영 저장소와 복구 경계

예제의 registry, budget ledger와 receipt store는 프로세스 로컬이다. 다중 worker와
재시작 복구가 필요하면 다음을 모두 애플리케이션 인프라로 구현한다.

- exact-version `AgentRegistry`;
- `BudgetStateStore`의 원자적 create/CAS와 `durable=True`;
- `DelegationReceiptStore`의 원자적 claim/CAS와 `durable=True`;
- 모든 worker에서 동일한 32-byte 이상 `hmac_secret`과 동일한 저장 namespace;
- durable checkpoint/conversation store와 endpoint별 idempotency 경계.

```python
coordinator = DelegationCoordinator(
    registry=durable_registry,
    policy=production_authorizer,
    execution_group_store=durable_budget_store,
    receipt_store=durable_receipt_store,
    hmac_secret=deployment_secret,
    limits=limits,
)
```

Receipt는 중복 소유자를 fence하고, 검증된 완료 결과를 replay하며, 확인할 수 없는
crash window는 fail closed 또는 `manual_required`로 남긴다. 이는 외부 side effect와
receipt를 하나의 transaction으로 묶거나 end-to-end exactly-once를 보장하지 않는다.
쓰기 작업은 별도의 idempotency key, transactional outbox 또는 업무 receipt가
필요하다. `allow_resume=True`는 자식 checkpoint가 안전하게 resumable인 경우에만
사용한다.

Checkpoint가 구성된 delegated child는 terminal checkpoint를 receipt와
budget lease가 settle될 때까지 보존하고 이후 best-effort로 삭제한다. 삭제 실패로
terminal checkpoint가 남을 수 있으므로 receipt가 replay/reconciliation의 기준이며,
checkpoint 존재만 보고 자식을 다시 호출하면 안 된다.

## 이벤트와 실패 확인

`Agent.stream_all()`은 다음 content-free 이벤트를 parent stream에 연결한다.

```text
DELEGATION_REQUESTED → DELEGATION_AUTHORIZED → DELEGATION_STARTED
→ DELEGATION_COMPLETED | DELEGATION_FAILED
```

거부, 재개와 수동 조정이 필요할 때는 `DELEGATION_REJECTED`,
`DELEGATION_RESUMED`, `DELEGATION_RECONCILIATION_REQUIRED`도 발생한다. Event
schema v2의 `execution_group_id`, root/parent/child run ID, delegation ID,
Agent ID/version과 depth로 상관관계를 만든다. 원본 요청, 자식 출력, Tool 인자,
tenant/principal 값은 이벤트 envelope에 넣지 않는다.

호출 실패는 고정된 `DelegationFailure.code`로 Tool failure에 매핑된다. 사용자에게
원본 예외나 payload를 노출하지 말고 보호된 diagnostic/event sink에서 failure ID와
분류를 확인한다.
