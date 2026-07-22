# Agent Skills

상태: ModuAgent 0.2.0에서 Agent Skills의 `SKILL.md` 핵심 형식, 로컬·InMemory source, 명시적·자동·혼합 선택, 제한된 resource 접근, digest 기반 checkpoint 복구를 지원한다. 0.2.0은 검토된 agent-scoped catalog를 신뢰 경계로 삼는 로컬 실행 프로필이다.

## 개념

Skill은 모델에 업무 절차와 필요한 지식을 제공한다. Tool은 조회·계산·변경 같은 실제 작업을 실행한다.

```text
Skill  ── 업무 절차·사용할 자료·허용 Tool 범위
Tool   ── 타입이 지정된 실제 실행 기능
Agent  ── Skill 선택, Tool 인가, 실행 한도와 대화 상태 관리
```

Skill을 활성화해도 새로운 Tool 권한은 생기지 않는다. Skill 본문은 모델 요청에만 임시로 추가되고 `ConversationStore`나 `AgentResult.messages`에는 저장되지 않는다.

## 패키지 형식

ModuAgent는 [Agent Skills specification](https://agentskills.io/specification)의 `SKILL.md` 패키지 형식을 사용한다.

```text
skills/
└── invoice-review/
    ├── SKILL.md                  # 필수
    ├── references/              # 선택: 필요할 때 읽는 참고 문서
    │   └── policy.md
    ├── assets/                  # 선택: 필요할 때 읽는 텍스트 자산
    │   └── report-template.md
    └── scripts/                 # 선택: 0.2.0에서는 실행·노출하지 않음
```

동작 확인용 패키지는 [`examples/skills/invoice-review`](../examples/skills/invoice-review)에 있다.

`SKILL.md`는 UTF-8 Markdown이며 YAML frontmatter로 시작해야 한다.

```markdown
---
name: invoice-review
description: 회사 정책에 따라 청구서를 검토한다. 청구 금액, 증빙, 승인 여부를 확인할 때 사용한다.
license: MIT
compatibility: ModuAgent 0.2.0
metadata:
  version: "1.0.0"
allowed-tools: lookup_invoice
---

# Invoice review

1. `lookup_invoice`로 원본 청구서를 조회한다.
2. `references/policy.md`에서 적용할 정책을 확인한다.
3. 확인된 사실과 미확인 항목을 구분해 답한다.
```

지원하는 frontmatter 필드는 다음과 같다.

| 필드 | 필수 | 규칙 |
|---|---:|---|
| `name` | 예 | 소문자·숫자·단일 하이픈만 사용, 최대 64자, 디렉터리명과 일치 |
| `description` | 예 | 언제 이 Skill을 사용할지도 포함하는 비어 있지 않은 문자열, 최대 1,024자 |
| `license` | 아니요 | 라이선스 이름 또는 설명 |
| `compatibility` | 아니요 | 실행 환경 요구사항, 최대 500자 |
| `metadata` | 아니요 | 문자열 key와 문자열 value로 구성된 mapping. `version`이 Skill 버전이 됨 |
| `allowed-tools` | 아니요 | 공백으로 구분한 문자열 또는 YAML 문자열 목록 |

ModuAgent의 기본 strict 프로필은 다른 frontmatter 필드와 `references/`, `assets/`, `scripts/` 밖의 패키지 파일을 거부한다. 이는 추가 파일을 허용하는 Agent Skills 기본 형식보다 의도적으로 좁다. Skill 이름, YAML 중복 key, anchor/alias, 절대경로, `..`, 역슬래시, symlink도 거부한다.

`compatibility`와 임의 `metadata`는 0.2.0에서 정보로 보존하지만 환경 의존성을 자동 설치하거나 호환성을 자동 판정하지 않는다. YAML 목록 형태의 `allowed-tools`는 ModuAgent 확장이고, 이식 가능한 Skill은 공식 형식인 공백 구분 문자열을 사용한다.

기본 패키지 제한은 `SKILL.md` 64 KiB, 파일 256개, 전체 50 MiB다. 본문은 비어 있을 수 없다.

`FilesystemSkillSource`의 경로 격리는 POSIX `openat`/directory FD와 `O_NOFOLLOW`를 요구한다. 이를 제공하지 않는 플랫폼에서는 안전하지 않은 fallback을 사용하지 않고 생성 단계에서 실패한다. 해당 환경에서는 `InMemorySkillSource`나 동일한 보안 계약을 구현한 custom source를 사용한다.

## 기본 사용

다음 예제는 실제 0.2.0 API와 source checkout에 포함된 예제 Skill을 사용한다. PyPI wheel 설치 환경에서는 같은 구조의 자체 catalog 경로를 지정한다.

```python
import asyncio

from moduagent import (
    Agent,
    AgentConfig,
    RunLimits,
    SkillLimits,
    SkillRegistry,
    VLLMClient,
    function_tool,
)


@function_tool(idempotent=True)
def lookup_invoice(invoice_id: str) -> dict[str, object]:
    """청구서 원본과 증빙 상태를 조회한다."""
    return {
        "invoice_id": invoice_id,
        "amount_krw": 850_000,
        "receipt_attached": True,
        "manager_approved": False,
    }


async def main() -> None:
    model = VLLMClient(
        base_url="http://localhost:8000/v1",
        model="company-model",
    )
    skill_limits = SkillLimits(
        max_active_skills=2,
        max_resource_reads=4,
        max_resource_tokens=4_096,
    )
    registry = SkillRegistry.from_paths(
        "./examples/skills",
        limits=skill_limits,
    )
    agent = Agent(
        config=AgentConfig(
            name="invoice-agent",
            instructions="근거가 확인된 내용만 답한다.",
            # reference 한 번을 읽은 뒤 답하려면 ACT 단계가 최소 두 번 필요하다.
            limits=RunLimits(max_steps=6, max_tool_calls=4),
        ),
        model=model,
        tools=[lookup_invoice],
        skill_registry=registry,
        skill_limits=skill_limits,
    )

    result = await agent.run(
        "INV-2026-0042를 검토해줘",
        session_id="invoice-42",
        skills=["invoice-review"],
    )
    if result.error:
        raise RuntimeError(result.error)
    print(result.output)


asyncio.run(main())
```

`SkillRegistry`는 생성 시 source를 eager 검증하고 정렬된 immutable catalog identity와 `catalog_digest`를 만든다. 큰 catalog나 네트워크 파일시스템에서는 시작 비용을 고려해야 한다. 파일을 수정했다면 기존 Registry를 갱신하는 대신 새 Registry와 Agent를 생성한다.

여러 디렉터리나 source도 합칠 수 있다.

```python
from moduagent import InMemorySkillSource, SkillRegistry

embedded = InMemorySkillSource(
    {
        "brief-answer": {
            "SKILL.md": """---
name: brief-answer
description: 한 문장으로 짧게 답해야 할 때 사용한다.
---

# Brief answer

답변은 한 문장으로 작성한다.
""",
        }
    },
    source_id="embedded",
)

registry = SkillRegistry.from_sources(embedded)
```

Registry 전체에서 Skill 이름과 `source_id`는 중복될 수 없다.

Filesystem descriptor의 기본 `source_id`는 배포 절대경로가 아니라 `filesystem://<skill-name>`이므로 같은 catalog를 다른 mount 경로로 옮겨도 lock identity가 유지된다. 여러 배포 namespace를 명시하려면 source를 직접 구성한다.

```python
from moduagent import FilesystemSkillSource, SkillRegistry

registry = SkillRegistry.from_sources(
    FilesystemSkillSource("./skills", source_id="finance-prod")
)
```

## 선택 모드

`Agent.run()`과 `Agent.stream()`은 동일한 `skills`, `skill_mode` 인자를 받는다.

| 모드 | 호출 | 동작 |
|---|---|---|
| `disabled` | 아무 인자도 전달하지 않음 | 기본값. Registry가 있어도 Skill을 사용하지 않음 |
| `explicit` | `skills=["invoice-review"]` | 지정한 이름만 활성화. `skill_mode`를 생략하면 자동으로 explicit가 됨 |
| `auto` | `skill_mode="auto"` | 모델이 catalog의 `name`, `description`, `version`만 보고 선택 |
| `hybrid` | `skills=[...], skill_mode="hybrid"` | 명시 Skill을 먼저 고정하고 남은 개수를 모델이 선택 |

### 명시 선택

명시 선택에는 별도 Selector가 필요 없다.

```python
result = await agent.run(
    "INV-2026-0042를 검토해줘",
    skills=["invoice-review"],
)
```

알 수 없는 이름, 중복 이름, `max_active_skills`를 넘는 선택은 모델 호출 전에 실패한다.

### 자동 선택

자동 선택은 `ModelSkillSelector`를 Agent에 설정해야 한다.

```python
from moduagent import Agent, ModelSkillSelector

selector = ModelSkillSelector(
    model,
    max_skills=2,
    options={"temperature": 0},
)

agent = Agent(
    config=config,
    model=model,
    tools=tools,
    skill_registry=registry,
    skill_selector=selector,
)

result = await agent.run(
    "INV-2026-0042를 검토해줘",
    skill_mode="auto",
)
```

선택 요청은 business Tool 없이 JSON schema만 사용하는 독립적인 모델 호출이다. 모델은 `{"skills":["invoice-review"]}` 형태로만 답해야 하며 catalog에 없는 이름을 반환하면 실행이 실패한다. 선택 호출의 token usage도 `AgentResult.usage`에 합산된다.

### 혼합 선택

```python
result = await agent.run(
    "INV-2026-0042를 검토하고 한 문장으로 정리해줘",
    skills=["invoice-review"],
    skill_mode="hybrid",
)
```

`hybrid`에도 `skill_selector`가 필요하다. 일반적으로 `ModelSkillSelector`를 그대로 설정하면 Runtime이 명시 선택과 자동 선택을 조합한다. 선택 동작을 직접 구성해야 하는 경우 `ExplicitSkillSelector`와 `HybridSkillSelector`도 공개되어 있다.

`skills`와 `skill_mode="auto"`를 함께 사용할 수 없다. 이 경우 `hybrid`를 사용한다. Resume 시에는 checkpoint의 Skill 상태가 복원되므로 `skills`나 `skill_mode`를 다시 전달하지 않는다.

## 모델 컨텍스트에 적용되는 범위

모델 컨텍스트에는 다음 순서로 Skill이 적용된다.

```text
Agent system instruction
→ 선택된 SKILL.md 본문
→ Conversation history와 현재 실행
→ phase 전용 메시지
```

- 자동 선택 단계에는 catalog metadata만 전달한다.
- 선택된 `SKILL.md` 본문은 PLAN, ACT, FINALIZE에 적용한다.
- `references/`와 text `assets/` 내용은 모델이 요청할 때만 추가한다.
- Skill system 메시지는 prompt 전용이며 대화 저장소, 결과 메시지, checkpoint에 원문으로 저장하지 않는다.
- Skill instruction은 ConversationMemoryPolicy가 제거할 수 없는 보호 메시지이며 token budget에는 포함된다.

## References와 assets

활성 Skill에 `references/` 또는 `assets/` 파일이 있으면 Agent가 내부 Tool을 자동으로 등록·노출한다.

| Tool | 기능 |
|---|---|
| `moduagent_skill_read` | UTF-8 파일 한 페이지를 byte cursor 기반으로 읽음 |
| `moduagent_skill_search` | filesystem Skill의 허용된 파일을 제한된 범위에서 검색 |

두 Tool 모두 활성 Skill의 `references/`와 `assets/`만 접근할 수 있다. `SKILL.md`, `scripts/`, 비활성 Skill, 절대경로, traversal, symlink, binary 파일에는 접근할 수 없다. InMemory source의 resource는 `read`할 수 있지만 현재 `search`는 `FilesystemSkillSource`만 지원한다.

Resource 결과는 ephemeral 메시지다. 실행 중 다음 모델 호출에는 제공되지만 `ConversationStore`와 최종 `AgentResult.messages`에는 남지 않는다. 이벤트에도 원문 대신 경로, digest, byte 수만 기록한다.

실행 중이거나 실패한 run의 checkpoint에는 정확한 Tool Call protocol을 복구하기 위해 resource 결과가 임시로 포함될 수 있다. 운영 `CheckpointStore`는 resource 원문과 같은 민감도로 암호화·접근 제어하고 짧은 TTL을 적용한다. 정상 완료 시 해당 checkpoint는 삭제된다.

### `max_steps`와의 관계

Resource Tool 호출은 business Tool용 `RunLimits.max_tool_calls`를 소비하지 않고 `SkillLimits.max_resource_reads`를 소비한다. 그러나 resource를 읽기로 결정하는 모델 호출은 하나의 ACT step이다. 읽은 결과로 답을 만들려면 다음 ACT step이 추가로 필요하다.

```text
ACT step 1: moduagent_skill_read 호출
→ reference 결과
ACT step 2: 결과를 사용해 답변 또는 business Tool 호출
```

따라서 reference 한 번을 읽는 단순 요청도 `max_steps >= 2`가 필요하다. PLAN, 여러 페이지 읽기, 검색 후 읽기, business Tool 호출을 함께 사용하는 Skill은 여유 있게 설정한다. `max_resource_reads`를 높여도 `max_steps`는 자동으로 증가하지 않는다.

## `allowed-tools`와 권한

`allowed-tools`는 Skill이 사용할 수 있다고 선언한 business Tool의 최대 범위이며 권한 부여가 아니다.

```text
모델에 노출되는 business Tool
  = Agent에 등록된 Tool
  ∩ 활성 Skill들의 allowed-tools 합집합

실제 실행
  = 위 범위
  ∩ ToolAuthorizer의 실행 시점 승인
```

- 선언한 Tool이 Agent에 등록되어 있지 않으면 Skill 활성화가 실패한다.
- 활성 Skill의 `allowed-tools`가 비어 있으면 business Tool은 노출되지 않는다.
- 여러 Skill이 활성화되면 각 Skill 선언의 합집합을 사용한다.
- Skill은 `ToolAuthorizer`를 우회하거나 권한을 추가할 수 없다.
- 제한적인 `ToolAuthorizer`를 사용한다면 필요한 business Tool과 내부 `moduagent_skill_read`, `moduagent_skill_search`도 정책에서 허용해야 한다.

변경 작업은 `AllowAllAuthorizer` 기본값에 의존하지 말고 운영 환경의 신뢰 가능한 identity에서 권한을 계산하는 `ToolAuthorizer`를 설정한다.

## Scripts

`scripts/`는 공식 패키지 구조와 digest 고정을 위해 인식하지만 0.2.0에서는 다음 동작을 모두 하지 않는다.

- 자동 실행
- subprocess 또는 shell 실행
- 모델 context에 파일 내용 추가
- `moduagent_skill_read` 또는 `moduagent_skill_search`를 통한 노출
- dependency 설치

스크립트와 같은 동작이 필요하면 검토한 Python 함수를 `FunctionTool`로 등록하고 `allowed-tools`에 그 Tool 이름을 선언한다. `scripts/`의 파일을 변경하면 실행되지는 않아도 package digest가 변경된다.

## 한도

`SkillLimits`의 기본값은 다음과 같다.

| 필드 | 기본값 | 적용 범위 |
|---|---:|---|
| `max_active_skills` | 3 | 한 run에서 활성화할 최대 Skill 수 |
| `max_catalog_tokens` | 2,048 | auto/hybrid 선택에 제공할 catalog metadata |
| `max_selection_tokens` | 8,192 | 현재 입력과 최근 대화로 구성한 자동 선택 입력 |
| `max_instruction_tokens` | 12,000 | Skill 하나의 instruction |
| `max_resource_reads` | 8 | 한 run의 read/search 호출 수 |
| `max_resource_tokens` | 8,192 | 한 run에서 읽은 resource token 누계 |
| `max_total_skill_tokens` | 20,000 | instruction과 resource를 합친 Skill context |
| `max_resource_bytes_per_read` | 65,536 | read 한 번의 반환 byte 수 |
| `max_resource_file_bytes` | 1,048,576 | 읽기·검색 가능한 resource 파일 하나의 최대 크기 |
| `max_resource_search_bytes` | 4,194,304 | search 한 번의 전체 scan byte 수 |
| `max_resource_search_results` | 20 | search 한 번에 반환할 최대 match 수 |
| `max_skill_bytes` | 65,536 | `SKILL.md` 파일 크기 |
| `max_package_files` | 256 | 패키지 파일 수 |
| `max_package_bytes` | 52,428,800 | 패키지 전체 크기 |

같은 `SkillLimits`를 `SkillRegistry.from_paths(..., limits=limits)`와 `Agent(..., skill_limits=limits)`에 전달하면 패키지 검증과 실행 한도를 한 설정으로 관리할 수 있다.

`*_tokens` 한도는 UTF-8 크기에 기반한 보수적 근사치다. 실제 모델의 전체 context window를 강제하려면 `TokenBudgetConversationMemoryPolicy`와 `VLLMTokenCounter`를 함께 사용한다. 이 Policy는 Skill system 메시지, Tool schema, 최종 출력 schema를 포함한 실제 `ModelRequest`를 계산한다.

## Checkpoint와 digest

각 package digest는 정규화한 상대경로와 모든 파일 byte로 계산된다. 따라서 `SKILL.md`, `references/`, `assets/`, `scripts/` 중 하나라도 바뀌면 digest가 바뀐다. `catalog_digest`는 Registry에 포함된 모든 Skill descriptor와 package digest로 계산된다.

Checkpoint에는 다음 상태가 저장된다.

- `catalog_digest`
- 활성 Skill의 name, version, source ID, package digest
- 선택 출처(`explicit` 또는 `model`)
- 유효 Tool 범위
- instruction/resource token과 resource read 누계

Resume은 Skill을 다시 선택하지 않고 이 상태를 복원한다.

```python
from moduagent import InMemoryCheckpointStore

checkpoints = InMemoryCheckpointStore()
agent = Agent(
    config=config,
    model=model,
    skill_registry=registry,
    checkpoint_store=checkpoints,
)

failed = await agent.run(
    "작업을 실행해줘",
    session_id="invoice-42",
    skills=["invoice-review"],
)

resumed = await agent.resume(failed.run_id, session_id="invoice-42")
```

Resume 시 Registry 전체의 catalog digest나 활성 Skill의 digest·source·Tool 범위가 달라지면 실행을 최신 내용으로 조용히 바꾸지 않고 실패한다. 활성화하지 않은 다른 Skill의 변경도 catalog digest를 바꾸므로 진행 중 checkpoint가 있으면 배포 단위 전체를 고정해야 한다. 과거 checkpoint v1은 Skills가 비활성화된 상태로 읽는다.

## CLI

설치 후 `moduagent skills` 명령으로 패키지를 만들고 검사한다.

### init

```bash
moduagent skills init invoice-review --path ./skills
```

`./skills/invoice-review/SKILL.md`와 기본 `references/`, `assets/` 디렉터리를 만든다. 기존 `SKILL.md`를 덮어쓰지 않는다. 만들 디렉터리는 선택할 수 있다.

```bash
moduagent skills init invoice-review \
  --path ./skills \
  --resources references,assets,scripts
```

### validate

```bash
moduagent skills validate ./skills
moduagent skills validate ./skills/invoice-review
```

단일 Skill 경로나 여러 Skill이 들어 있는 디렉터리를 strict mode로 검증하고 Skill 수와 `catalog_digest`를 출력한다. 이식성 검사가 아닌 확장 패키지를 조사할 때만 `--no-strict`를 사용한다.

```bash
moduagent skills validate ./skills --no-strict
```

### inspect

```bash
moduagent skills inspect ./skills
moduagent skills inspect ./skills --json
```

기본 출력에는 이름, 버전, package digest, 설명이 표시된다. JSON에는 `schema_version`, `catalog_digest`, source ID, `allowed_tools`도 포함된다.

### lock

```bash
moduagent skills lock ./skills
moduagent skills lock ./skills --output deploy/skills.lock.json
```

기본 파일은 catalog 디렉터리 아래의 `skills.lock.json`이다. `--output`이 상대경로이면 catalog 디렉터리를 기준으로 저장한다. 단일 Skill 디렉터리를 지정한 경우에는 그 부모 디렉터리를 기준으로 저장한다.

Lock 파일은 배포 검토와 artifact 비교를 위한 catalog snapshot이다. Runtime이 파일을 자동 탐색하지는 않으므로 운영 코드에서 명시적으로 전달한다.

```python
registry = SkillRegistry.from_paths(
    "./skills",
    lockfile="./skills/skills.lock.json",
)
```

이 경우 시작 시 catalog, version, source ID, package digest가 lock과 정확히 일치하지 않으면 실패한다. 실제 checkpoint resume 일관성도 Runtime이 저장한 digest로 다시 강제한다.

## 운영 체크리스트

- 자동 선택을 켜기 전에 사내 요청 세트로 Skill `description`의 선택 정확도를 평가한다.
- 0.2.0 catalog에는 운영자가 검토한 로컬 Skill만 넣는다. 서로 신뢰하지 않는 tenant가 같은 Registry를 공유해야 한다면 tenant별 Registry 또는 body load 전 별도 접근 제어 계층을 둔다.
- 변경 Tool에는 명시적인 `ToolAuthorizer`를 설정한다.
- `SkillLimits`와 `RunLimits.max_steps`를 함께 조정한다.
- 배포할 Skill catalog와 `skills.lock.json`을 같은 release artifact로 고정한다.
- 실행 중 checkpoint가 있으면 같은 catalog digest를 가진 artifact로 resume한다.
- `scripts/`를 실행 파일 저장소로 간주하지 않는다.
- Skill 본문과 references에는 secret을 넣지 않는다.
