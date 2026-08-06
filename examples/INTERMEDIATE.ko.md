# 중급 예제

이 예제는 ModuAgent `0.5.2`를 기준으로 합니다. 먼저 [`README.md`](README.md)의
초급 과정을 마친 다음 `10` → `11` → `12` 순서로 진행하세요. 각 예제는 독립적으로
실행할 수 있고 애플리케이션이 소유한 결정론적 데이터를 사용합니다. 따라서 Agent
패턴을 바꾸지 않고 샘플 Tool을 자체 연동 코드로 교체할 수 있습니다.

## 설치 및 설정

정확한 릴리스 버전을 설치합니다.

```bash
python -m pip install "moduagent==0.5.2"
```

또는 이 소스 체크아웃에서 현재 작업 트리를 사용합니다.

```bash
python -m pip install -e .
```

OpenAI 호환 vLLM 연결 정보를 환경변수로 설정합니다. 자격 증명을 예제 파일에
넣거나 소스 관리 시스템에 커밋하지 마세요.

```bash
export VLLM_BASE_URL="<your vLLM base URL>"
export VLLM_MODEL="<your tool-capable model>"
export VLLM_API_KEY="<optional token>"
```

## 학습 경로

| 순서 | 시나리오 | Tool 수 | 핵심 학습 내용 | 안전 경계 |
| --- | --- | ---: | --- | --- |
| 10 | [인시던트 조사](10_incident_investigation.py) | 5 | 병렬 증거를 연관 분석하고 중첩된 리포트를 검증 | 읽기 전용 조사이며 완화, 롤백 또는 배포를 실행하지 않음 |
| 11 | [고객 케이스 해결](11_customer_case_resolution.py) | 5 | 순차 조회와 계산 과정에 검증된 값을 전달 | 제안만 제공하며 환불, 반품, 메시지 전송 또는 케이스 갱신을 수행하지 않음 |
| 12 | [릴리스 준비 상태](12_release_readiness.py) | 5 | 여러 증거 게이트를 적용하고 일관된 결정을 강제 | 출시/보류를 추천할 뿐 배포하거나 시스템을 변경하지 않음 |

세 예제 모두 Standard 실행, 구조화된 Pydantic 출력, 제한된 `RunLimits`,
요약 전용 Tool 추적, 제한된 모델 출력과 비동기 모델 클라이언트 컨텍스트
관리자를 사용합니다.

## 10 — 인시던트 조사

실행:

```bash
python examples/10_incident_investigation.py
```

Agent는 먼저 인시던트를 읽은 다음 메트릭, 배포, 로그와 의존 서비스 상태를
수집합니다. 처음 조회 이후의 네 가지 읽기 작업은 서로 독립적이므로, 이 예제는
범위가 제한된 병렬 Tool 실행을 활성화합니다. `IncidentReport`는 중첩 모델과
필드 검증, 그리고 완화된 인시던트를 해결 완료로 보고하지 못하게 하는 필드 간
규칙을 보여 줍니다. 최종 `runbook_actions` 객체에는 생성된 운영 명령 대신
애플리케이션이 소유한 작업 코드가 들어갑니다. Agent는 이를 추천할 수 있지만
실행할 수는 없습니다.

디버깅 지점:

- `run usage`에는 실행 시간, 모델 turn 수와 전체 Tool 호출 수가 표시됩니다.
- `tool trace`와 `observed calls`에는 다섯 가지 증거 출처가 모두 있어야 합니다.
- 실패하면 `error_summary`와 인메모리 진단 레코드로 provider body나 자격 증명을
  기록하지 않고도 실패한 컴포넌트를 찾을 수 있습니다.
- 증거 출처 누락, 잘못된 시간 범위 또는 유효하지 않은 리포트 필드는 데이터 흐름
  문제입니다. 제한값을 늘리기 전에 호출 로그부터 확인하세요.
- 필수 `runbook_actions` 키 네 개는 instruction에 이름이 명시되어 있습니다.
  모델이 필수 객체를 너무 일찍 닫으려 할 때 발생한 guided-decoding 공백 루프를
  방지하기 위한 것입니다. `max_tokens`만 늘려서는 이 실패 유형을 해결할 수 없습니다.

## 11 — 고객 케이스 해결

실행:

```bash
python examples/11_customer_case_resolution.py
```

Agent는 케이스 → 주문 → 정책 → 적격성 → 환불 견적의 의존 관계를 따릅니다.
타입이 지정된 인자를 사용하므로 잘못된 카테고리와 통화 값은 Tool 경계에서
실패합니다. 최종 스키마는 사람의 승인을 필수로 요구하고
`write_action_performed=true`를 허용하지 않으므로 이 워크플로는 제안 범위에
머뭅니다.

디버깅 지점:

- 출력된 `tools` 목록에는 다섯 호출이 의존 관계 순서대로 나타나야 합니다.
- `tool trace`는 요약만 제공합니다. 개발 중 비밀 정보가 아닌 일부 인자를
  확인해야 한다면 로컬 `CALL_LOG`를 사용하세요.
- 조회 결과가 `not_found`이면 Agent는 누락된 값을 지어내지 않고
  `manual_review`를 선택해야 합니다.
- 계산 검증이 실패하면 광범위한 모델 재시도를 추가하기보다 각 인자를 바로 앞의
  Tool 결과와 비교하세요.

## 12 — 릴리스 준비 상태

실행:

```bash
python examples/12_release_readiness.py
```

Agent는 릴리스 manifest의 정확한 commit 및 change-set 식별자를 CI, 보안, 위험,
용량 검사로 전달합니다. `ReleaseDecision`은 model validator를 사용하여 다섯 가지
증거 유형을 모두 요구하고 `ship`/`hold` 결정과 차단 사유의 일관성을 유지합니다.
예제 데이터에는 차단 대상 보안 취약점이 있으므로 증거에 기반한 결과는 `hold`여야
합니다.

디버깅 지점:

- `checks`와 `tool trace`에는 다섯 Tool 이름이 모두 있어야 합니다.
- 이 예제에서 `ship` 결과가 나오면 증거를 건너뛰었거나 보안 결과를 적용하지 않은
  것입니다. `CALL_LOG`와 최종 검증 오류를 확인하세요.
- 같은 manifest commit이 CI와 보안 Tool 양쪽에 전달되고, 같은 change-set ID가
  위험 Tool에 전달되는지 확인하세요.
- 배포나 승인 변경은 별도로 권한이 부여된 애플리케이션 워크플로에 두세요.
  디버깅을 위해 이 평가 Agent에 추가해서는 안 됩니다.

## 라이브 검증

일반 테스트 스위트는 네트워크 접근 없이 이 예제를 import하고 테스트합니다.
설정된 vLLM endpoint를 대상으로 세 예제를 모두 실행하려면 다음 명령을 사용합니다.

```bash
MODUAGENT_RUN_LIVE_INTERMEDIATE=1 \
python -m pytest -q tests/integration/test_live_intermediate_scenarios_v051a.py
```

라이브 스위트는 구조화된 값, 정확한 Tool 집합과 의존 순서, 중복 호출 방지,
성공한 Tool 추적, 모델 turn 제한과 Tool 호출 제한을 검증합니다. 연결 자격 증명을
포함하거나 출력하지 않습니다.

## 예제 응용

먼저 인메모리 데이터셋을 자체 읽기 전용 어댑터로 교체하세요. 타입이 지정된 Tool
인자, 결과 크기 및 timeout 제한, 구조화 출력 검증과 명시적인 쓰기 금지 경계는
그대로 유지합니다. 쓰기 가능한 Tool은 애플리케이션 권한 부여, 멱등성, 감사 저장소와
사람의 승인 단계를 도입한 이후에만 추가하세요.
