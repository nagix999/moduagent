# 중급 예제

이 예제는 ModuAgent `0.6.0`을 기준으로 합니다. 먼저 [`README.md`](README.md)의
초급 과정을 마친 다음 `10` → `11` → `12` → `13` 순서로 진행하세요. 각 예제는
독립적으로 실행할 수 있습니다. 예제 `10`–`12`는 애플리케이션이 소유한 결정론적
데이터를 사용하고, 예제 `13`은 운영자가 승인한 로컬 파일과 Docling Serve endpoint를
사용합니다. Agent 패턴을 바꾸지 않고 제한된 Tool을 자체 연동 코드로 교체할 수
있습니다.

## 설치 및 설정

정확한 릴리스 버전을 설치합니다.

```bash
python -m pip install "moduagent==0.6.0"
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
| 13 | [문서 질의응답 및 리포트](13_document_qa_and_report.py) | 3 | 요청을 라우팅한 뒤 출처 위치를 보존하면서 답변하거나 목차형 Markdown 리포트를 작성 | 승인된 로컬 파일과 읽기 전용 증거 Tool만 사용하며 임의 경로·URL·파일 덮어쓰기를 허용하지 않음 |

네 예제 모두 모델 작업에 제한을 두며 Agent Tool은 읽기 전용 또는 제안 범위에
머뭅니다. 예제 `10`–`12`는 Standard 실행, 구조화된 Pydantic 출력, 요약 전용 Tool
추적과 `RunLimits`를 사용합니다. 예제 `13`은 여기에 제한된 문서 변환, 인용을
보존하는 여러 단계의 Markdown 조립, 제한된 요청 라우팅과 선택적인 애플리케이션
소유 결과물을 더합니다.

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

## 13 — 문서 질의응답 및 근거 기반 리포트

이 예제는 파일 경로를 받아 별도로 배포된 Docling Serve 컨테이너에 파일을
업로드하고, 크기가 제한된 인메모리 증거 corpus를 구성합니다. 실행 전 애플리케이션이
관리하는 루트 디렉터리를 설정합니다.

```bash
export DOCUMENT_ROOT="/srv/approved-documents"
export DOCLING_SERVE_URL="http://localhost:5001"
# export DOCLING_SERVE_API_KEY="<선택적 Docling Serve 키>"
# export DOCLING_SERVE_DO_OCR="true"  # 기본값: false
```

질문하기:

```bash
python examples/13_document_qa_and_report.py \
  --file /srv/approved-documents/policy.pdf \
  --file /srv/approved-documents/appendix.docx \
  --prompt "무엇이 변경되었으며 어떤 근거가 이를 뒷받침합니까?"
```

Markdown 리포트를 새 결과 파일로 원자적으로 저장하기:

```bash
python examples/13_document_qa_and_report.py \
  --file /srv/approved-documents/policy.pdf \
  --file /srv/approved-documents/appendix.docx \
  --prompt "운영 영향을 분석하고 다음 조치를 제안하세요." \
  --output ./policy-impact.md
```

`--mode`는 선택 사항이며 기본값은 `auto`입니다. 제한된 Tool-free intent Agent는
검증된 신뢰할 수 없는 요청 문장만 보고 구조화된 `RequestIntent`를 반환합니다.
파일 경로, 파싱된 문서, corpus 근거와 retrieval Tool은 라우터에 노출하지 않습니다.
독립된 리포트·제안서·검토서 또는 현황 분석과 개선안·권고를 함께 요구하는 복합
산출물은 `report`로 보냅니다. 직접 질문·설명·요약·추출·비교와 단일 사실의 간단한
분석은 `question`으로 보냅니다. 의도가 모호하면 안전한 기본값인 `question`을
선택합니다.

호스트 애플리케이션이 워크플로를 이미 알고 있다면 `--mode question` 또는
`--mode report`를 hard override로 사용할 수 있습니다. 이 경우 분류 모델 호출을
건너뛰며, `--mode auto`는 기본 동작을 명시적으로 선택합니다. `--output`은 결과물
저장 여부만 제어하며 리포트 모드를 선택하지 않습니다.

자동 라우팅에는 제한된 모델 호출이 한 번 추가됩니다. 구조화된 `RequestIntent`가
유효하지 않으면 경로를 임의로 추측하지 않고 실패하므로, 경로 확정성이나 가장 낮은
지연 시간이 중요하면 명시적 override를 사용하세요.

애플리케이션 코드에서는 `run_document_request()`로 같은 자동 또는 override
라우팅을 사용할 수 있습니다. 호스트가 각 단계를 직접 관리한다면 하위 수준의
`classify_intent()`, `run_question()`, `run_report()`도 그대로 사용할 수 있습니다.

`DOCLING_SERVE_URL`의 기본값은 `http://localhost:5001`이고, `DOCUMENT_ROOT`의
기본값은 현재 작업 디렉터리입니다. 두 값 모두 최종 사용자의 입력이 아닌 신뢰된
배포 설정으로 취급하세요. `DOCLING_SERVE_TIMEOUT`은 각 HTTP 요청을 제한합니다.
`DOCLING_SERVE_MAX_WAIT_SECONDS`는 작업 제출, polling, 재시도와 결과 조회를 모두
포함하는 하나의 전체 deadline입니다. 최종 Docling 결과는 `status=success`인 경우만
수용하며 `partial_success`와 실패 결과는 거부합니다. OCR은 기본적으로 꺼져 있습니다.
원문에 필요하고 Docling 배포 환경에 적절한 OCR 자원이 있을 때만
`DOCLING_SERVE_DO_OCR`을 활성화하세요.

클라이언트는 승인된 각 파일을 multipart 데이터로 Docling Serve 비동기 파일 변환
API에 제출하고, 제한 시간 안에서 작업 상태를 polling한 뒤 Markdown과 무손실 JSON
결과를 가져옵니다. 즉 Python HTTP I/O와 Docling 작업이 모두 비동기이며, Docling이
호출자를 대신해 임의 경로나 URL을 가져오게 하지 않습니다. 선택적 키는 Docling
Serve의 `X-Api-Key` 헤더로 전달합니다. 서버 계약은
[Docling Serve REST API](https://docling-project.github.io/docling/usage/api_server/rest_api/)를
참고하세요.

Agent에는 읽기 전용 Tool 세 개(`list_documents`, `search_evidence`,
`read_evidence`)만 제공됩니다. 질문 모드는 인용이 포함된 Markdown 답변을
반환합니다. 리포트 모드는 먼저 제한된 세부 목차를 만들고, 각 파트를 증거 corpus에
맞춰 작성하고, 인용을 검증한 다음 결정론적 애플리케이션 코드로 파트를 합칩니다.
하나의 모델 응답에 전체 리포트 작성과 병합을 맡기지 않습니다.

관련 근거가 없으면 `QuestionAnswer`는 답변이나 인용을 만들어 내지 않고
`status=insufficient_evidence`, 빈 citations와 구체적인 limitations로 명시적으로
기권할 수 있습니다.

검증을 통과한 각 인용에는 Markdown 출처 위치 각주와 애플리케이션이 소유한 근거
원문 발췌 블록을 나란히 붙입니다. 페이지와 bounding box는 Docling provenance에서
가져옵니다. UTF-8 텍스트 계열
원문은 일치하는 원문 Line No를 표시하고, 그 밖의 형식은 확인 가능한 경우 일치하는
Docling-Markdown Line No임을 구분해 표시합니다. 좌표가 없다면 `페이지 확인 불가`와
Docling item 참조 및 heading 경로 같은 구조적 위치를 남기며 위치를 만들어 내지
않습니다. 모델은 불투명한 citation ID만 선택할 수 있고 원문과 위치는 항상
애플리케이션의 불변 corpus에서 가져옵니다. 존재하지 않는 ID는 렌더링 전에
실패하므로 근거로 조용히 통과하지 않습니다.

표 근거는 셀 값을 하나의 연속된 원문으로 오인하지 않도록
`Docling 표 셀 직렬화(연속 원문 아님)`로 명시합니다.

안전 및 디버깅 경계:

- 애플리케이션은 업로드 전에 `DOCUMENT_ROOT` 아래의 중복되지 않는 정규 파일을
  canonical 경로로 해석합니다. 심볼릭 링크, 정규 파일이 아닌 대상, 지원하지 않는
  확장자와 루트 바깥 경로는 거부합니다.
- 네트워크 I/O 전에 파일 수와 크기를 확인합니다. 최대 10개, 파일당 50 MiB,
  총 200 MiB입니다. Docling 요청·응답 크기, polling 시간, corpus 크기, Tool 결과,
  목차 크기, 파트 수, 모델 turn과 출력도 예제 안에서 제한합니다.
- 파일과 변환된 텍스트는 Agent 지시가 아니라 신뢰할 수 없는 증거입니다. 모델은
  경로나 변환 URL을 선택할 수 없고 문서 URL 입력도 받지 않습니다. SSRF 경계를
  유지하려면 설정된 Docling endpoint를 애플리케이션 네트워크 allowlist 안에 두세요.
- 원문 내용과 선택된 증거는 설정된 모델로 전송됩니다. 실행 전에 접근 제어와
  redaction을 적용하세요. 인용 검증은 기밀성 통제가 아닙니다.
- Markdown은 항상 stdout에 출력합니다. `--output`을 지정하면 같은 Markdown을
  추가로 저장하고 생성 경로는 stderr에 표시합니다. 원자적 writer는 기존 대상과
  심볼릭 링크를 거부하고 덮어쓰지 않습니다.
- 기본 검색은 한 실행 안에서 동작하는 제한된 결정론적 lexical retrieval입니다.
  크거나 반복해서 사용하는 corpus는 `EvidenceRetriever` adapter를 통해 Vector Store
  구현으로 교체할 수 있습니다. 불투명한 evidence ID를 유지하고, 반환된 모든 ID를
  불변 run corpus와 다시 대조한 뒤 증거로 노출하세요.

## 라이브 검증

일반 테스트 스위트는 네트워크 접근 없이 네 예제를 import하고 테스트합니다. 현재
선택적 vLLM 스위트는 예제 `13`을 제외한 `10`–`12`를 검증합니다. 설정된 vLLM
endpoint를 대상으로 이 세 예제를 실행하려면 다음 명령을 사용합니다.

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
