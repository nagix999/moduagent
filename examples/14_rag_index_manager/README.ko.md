# RAG 인덱스 관리 Agent

[English](README.md) | [한국어](README.ko.md)

이 예제는 사내 어시스턴트가 검색에 사용할 RAG 인덱스를 **구축하고 유지하는
관리 Agent**다. 애플리케이션이 승인한 문서 디렉터리를 스캔하고, 변경된 파일만
Docling Serve로 파싱한 뒤, Gemma의 텍스트·비전 분석 결과를 검색용 메타데이터로
추가한다. 구조를 보존한 청크를 BGE-M3로 임베딩하고, 검증된 새 Milvus 세대를
게시한다.

기본 구성은 다음과 같다.

- 관리 판단, 텍스트 분석, 이미지 분석, 페이지 레이아웃 보정:
  `gemma-4-26B-A4B-it`
- dense embedding: `BGE-M3`
- 문서 파서: Docling Serve 비동기 파일 변환 API
- 검색 인덱스: Milvus dense 검색과 제한된 lexical reranking을 결합하고 원문 청크와
  provenance를 보존하는 hybrid retrieval
- 작업·세대 상태의 기준: 로컬 SQLite manifest
- 중간 산출물: 파일 시스템의 재시작 가능한 artifact cache

이 예제는 질문에 답하는 RAG 검색 Agent 자체가 아니다. **검색 Agent가 사용할
인덱스를 안전하고 증분 방식으로 만드는 제어 평면(control plane)**이다. 실제
질의 시의 검색, reranking, 답변 생성과 ACL 적용은 별도의 serving 경로에서
구성한다.

## 무엇을 보여주는 예제인가

핵심 목표는 다음 일곱 가지다.

1. 승인된 디렉터리의 문서를 안전하게 발견하고 같은 바이트를 해시·파싱한다.
2. 원문 변경과 파이프라인 설정 변경을 구분하고 필요한 단계부터만 다시 실행한다.
3. Docling의 문서 구조와 provenance를 보존하면서 검색하기 좋은 블록과 청크로
   재구성한다.
4. PPT/PDF처럼 읽기 순서가 자주 깨지는 문서는 페이지 캡처와 파싱 결과를 Gemma
   VLM에 함께 보여 주고, 제한된 구조 패치만 적용한다.
5. BGE-M3 임베딩을 새 Milvus staging collection에 적재하고 검증 후 alias를
   전환한다.
6. 자연어 관리 요청을 정확히 하나의 제한된 Tool 호출로 변환하고, 실패 시 어느
   단계에서 왜 실패했는지 내용 비노출 로그로 진단한다.
7. 파일이 계속 유입되는 환경에서 안정화, 자동 동기화, 재시도와 격리를 내구
   상태로 관리하고 프로세스 재시작 뒤 이어서 실행한다.

## 전체 구조

```text
파일 유입 ------------> ContinuousIngestionSupervisor ----+
                       - snapshot 안정화                   |
                       - 자동 sync / 재시도 / 격리          |
                                                           +-->
사용자 자연어 요청 ---> Management Agent (Gemma) ----------+   RAGIndexManager
                       - status / preview                      (결정론적 Python
                       - sync / rebuild / rollback               orchestration)
        |
        +--> 안전한 디렉터리 스캔 + SHA-256
        +--> SQLite manifest와 비교하여 증분 계획
        +--> DoclingDocument JSON 파싱/캐시
        +--> 구조 보존 블록 재구성
        +--> 페이지 이미지 + Gemma VLM 레이아웃 보정
        +--> Gemma 텍스트/이미지 retrieval metadata 생성
        +--> 결정론적 청킹 + provenance
        +--> BGE-M3 임베딩
        +--> Milvus staging 적재 및 검증
        +--> active alias 게시 + manifest commit
```

중요한 설계 원칙은 **LLM이 작업 대상을 결정하지 않는 것**이다. 모델은 파일
경로, source hash, chunk ID, collection 이름, 삭제 대상, alias target을 만들거나
변경하지 않는다. 이 값들은 모두 애플리케이션 코드가 결정한다.

일상적인 파일 유입도 LLM이 판단하지 않는다. `ContinuousIngestionSupervisor`가
디렉터리를 반복 스캔하고 동일 content snapshot이 안정화 시간 동안 유지됐을 때만
증분 동기화를 실행한다. 실패한 동일 snapshot은 지수 백오프로 제한 재시도한 뒤
격리하며, 파일 내용이 다시 바뀌면 자동으로 격리를 해제한다. 주기적인 전체
reconciliation도 수행하므로 filesystem notification 하나에 의존하지 않는다.

## AI Agent의 역할

이 예제에서는 같은 Gemma 배포를 세 가지 서로 다른 역할로 사용한다.

### 1. 관리 Agent

`agent.py`의 `build_management_agent()`가 생성한다. 사용자의 자연어 요청을 읽고
다음 Tool 중 정확히 하나를 선택한다.

| Tool | 의미 | 쓰기 권한 필요 |
|---|---|---:|
| `inspect_index_status` | manifest와 Milvus의 현재 상태 확인 | 아니요 |
| `preview_incremental_sync` | 변경 파일과 재처리 단계를 dry-run으로 계산 | 아니요 |
| `apply_incremental_sync` | 변경분을 처리하고 검증된 새 세대를 게시 | 예 |
| `rebuild_entire_index` | 모든 현재 문서를 처음부터 재처리 | 예 |
| `rollback_previous_generation` | 직전 보존 세대로 복구 | 예 |

Agent는 `max_tool_calls=1`, `parallel_tool_calls=False`,
`tool_choice="required"`로 구성된다. 성공한 Tool 호출이 정확히 하나 있어야만
구조화된 `ManagementResponse`를 최종 결과로 만들 수 있다. 이 제한은 한 요청에서
동기화와 롤백 같은 관리 작업이 함께 실행되는 것을 막고, 최종 응답의 건수와 세대
ID가 실제 Tool 결과와 같은지 애플리케이션이 다시 검증할 수 있게 한다.

`allow_writes=False`이면 쓰기 Tool 자체가 Agent에 제공되지 않는다. 따라서
“동기화해 줘”라는 요청도 계획 확인까지만 수행한다. 쓰기 허용 여부는 프롬프트가
아니라 애플리케이션 코드가 결정한다.

상시 supervisor와 관리 Agent의 역할은 분리되어 있다. supervisor는 정상적인 신규,
수정, 삭제 문서를 자동 반영한다. 관리 Agent는 현재 세대와 자동 수집 상태 설명,
dry-run, 전체 재구축, rollback처럼 운영 의도가 필요한 control-plane 작업을
담당한다. 상태 Tool은 자동 수집이 초기화 중인지, 정상인지, 재시도 중인지 또는
동일 snapshot이 격리됐는지를 함께 전달한다.

### 2. 페이지 레이아웃 보정 VLM

`VLLMLayoutRefinementClient`는 Docling의 페이지 캡처와 해당 페이지의 opaque
block descriptor를 Gemma VLM에 전달한다. 특히 PPT/PPTX와 고정 레이아웃 PDF에서
열 순서, 제목·본문 관계, 그룹 구조가 어긋난 경우를 보정한다.

VLM이 할 수 있는 일은 제한되어 있다.

- 기존 block ID의 순서 제안
- 기존 block의 문서 역할 제안
- 같은 페이지 안의 부모·자식 그룹 제안
- 기존 heading block ID를 이용한 section ancestry 제안
- 명시적으로 허용된 경우에만 장식 요소/반복 머리말 제외 사유 제안

VLM은 원문 텍스트, provenance, block ID, source 위치를 수정할 수 없다. 알 수 없는
ID, 누락·중복 ID, 다른 페이지의 ID, 순환 부모 관계는 모두 거부된다. 사람이 읽는
section path도 모델이 임의 문자열로 만드는 것이 아니라, 검증된 heading ID가
가리키는 Docling 원문에서 애플리케이션이 생성한다.

기본값은 `allow_exclusions=False`다. 페이지 이미지가 없거나 한 페이지의 블록이
기본 상한 32개를 넘으면 VLM을 호출하지 않고 Docling의 기존 순서로 안전하게
fallback한다.

일부 Docling 버전은 PPT/PPTX와 Office 문서에서 page-image 옵션을 켜도 전체 페이지
이미지를 주지 않는다. `.env`의 `RAG_OFFICE_PAGE_CAPTURE=true`를 사용하면 이미
검증한 Office 파일을 headless LibreOffice로 PDF 변환하고 `pdftocairo`로 PNG를
만드는 제한된 fallback을 사용한다. Agent host에 `soffice`와 `pdftocairo`가 있어야
하며, subprocess 시간·페이지 수·이미지 크기·전체 출력 크기는 모두 제한된다.

하나의 Office text box가 여러 Docling block으로 분리되면서 동일 bbox를 공유하는
경우, 페이지 이미지만으로 opaque ID와 각 줄을 정확히 대응시킬 수 없다. 이때는
공유 bbox 내부의 Docling 원래 순서를 유지하고 VLM이 제안한 역할·부모·그룹·제외와
모호한 heading 관계를 제거한다. 유효한 같은 페이지 ID라도 heading이 아닌 block을
section ancestry로 지목하면 그 선택적 관계만 버린다. 알 수 없는·중복·누락·다른
페이지 ID는 여전히 전체 patch를 거부한다.

### 3. 검색 메타데이터 생성 모델

`VLLMEnrichmentClient`는 각 구조화 블록을 분석해 요약, 키워드, 이미지 설명과
embedding용 보조 텍스트를 만든다. 모델 출력은 closed JSON schema로 검증되며,
다음 원칙을 지킨다.

- Docling 원문은 canonical source로 계속 보존한다.
- 모델 생성 요약·태그·이미지 설명은 inferred metadata로만 저장한다.
- 문서 안의 명령문은 프롬프트가 아니라 분석 대상 데이터로 취급한다.
- 이미지 입력과 응답 크기, 동시성, timeout, retry 횟수는 제한된다.

BGE-M3는 추론 Agent가 아니라 `VLLMEmbeddingClient`가 호출하는 임베딩 모델이다.
청크 순서, 벡터 개수, 차원, finite 값과 모델 fingerprint가 모두 검증된다.

## 파이프라인 단계

```text
승인된 문서 디렉터리
  -> bounded scan + SHA-256 revision
  -> fingerprint-aware incremental plan
  -> DoclingDocument JSON
  -> structure-preserving blocks
  -> optional whole-page layout refinement
  -> Gemma text / picture metadata
  -> deterministic chunks + provenance
  -> BGE-M3 embeddings
  -> Milvus staging generation
  -> count / dimension / content validation
  -> Milvus alias publication
  -> SQLite manifest commit
```

### 1. 안전한 스캔

`scanner.py`는 일반 파일만 일정한 상대 경로 순서로 스캔한다. 심볼릭 링크,
hard-link 중복, 비정상 파일, traversal, 디렉터리 교체, 파일 수·깊이·개별 크기·총
크기 초과를 거부한다. 스캔 뒤 파싱하기 직전에도 동일한 파일 identity와 SHA-256을
재검증한다. Docling에는 호스트 경로를 넘기지 않고 검증된 바이트를 multipart로
업로드한다.

기본 지원 확장자는 PDF, DOCX, PPTX, XLSX, HTML, Markdown, TXT, CSV, JSON, XML,
ODT, EPUB와 일반 이미지 형식이다.

### 2. 변경 감지와 단계별 재시작

`planner.py`는 현재 스캔 결과와 게시된 manifest를 비교해 각 문서를 다음 중 하나로
분류한다.

- `new`: 새 문서이므로 parse부터 수행
- `modified`: 내용 SHA-256이 달라졌으므로 parse부터 수행
- `pipeline_changed`: 내용은 같지만 특정 처리 설정이 달라져 해당 단계부터 재수행
- `deleted`: 문서 디렉터리에서 사라진 source를 새 세대에서 제거
- `unchanged`: 원문과 모든 fingerprint가 같아 기존 결과 재사용

파이프라인은 parser → restructuring → layout refinement → enrichment → chunking →
embedding/dimension → indexing 순서의 fingerprint를 가진다. 예를 들어 chunk 크기만
바뀌면 Docling과 Gemma 분석 artifact를 재사용하고 chunk 단계부터 다시 실행한다.
BGE-M3 모델이나 embedding dimension만 바뀌면 cached chunk를 다시 임베딩한다.

### 3. Docling 재구조화와 provenance

`restructure.py`는 DoclingDocument JSON을 `StructuredBlock`으로 변환한다. 텍스트,
표, 이미지, key-value를 구분하고 heading hierarchy, page number, bounding box,
char span, Docling `self_ref`를 보존한다. 표는 구조를 잃지 않도록 Markdown 형태의
canonical block text로 직렬화한다.

`chunking.py`는 구조와 modality를 고려해 결정론적으로 청크를 만든다. 기본값은
최대 2,400자, 240자 overlap이며, 원문 청크와 모델이 만든 embedding hint를 서로
분리한다. `chunk_id`는 source revision과 구조를 바탕으로 안정적으로 생성된다.

### 4. Milvus blue/green 게시

`RAGIndexManager.sync()`는 새 generation ID와 staging collection을 만든다. 변경되지
않은 source는 가능하면 현재 active collection에서 복사하고, 변경된 source의 이전
row는 staging에서 제거한 뒤 새 청크를 upsert한다. 모든 문서 처리가 끝난 뒤 다음을
검증한다.

- 기대한 모든 chunk ID가 존재하는가
- row 수가 manifest와 같은가
- 벡터 차원이 설정과 같은가
- 모든 벡터와 metadata가 유효한가

검증이 끝난 뒤에만 Milvus active alias를 새 collection으로 전환하고 SQLite
manifest를 commit한다. 직전 alias target은 bounded rollback 대상으로 남는다.

Milvus alias 변경과 SQLite commit은 서로 다른 저장소에 걸쳐 있으므로 하나의 DB
transaction이 될 수 없다. 게시 이후 manifest commit이 실패하면 이전 alias 복구를
시도한다. 복구 결과를 확정할 수 없으면 추측해서 진행하지 않고 fail-closed한다.
다중 worker 운영에서는 durable transition journal과 운영자 승인 reconcile 절차를
추가해야 한다.

## 주요 구현 파일

| 파일 | 책임 |
|---|---|
| `agent.py` | 자연어 관리 Agent, Tool 노출, 단일 작업 감사, 최종 응답 검증 |
| `pipeline.py` | 증분 처리, staging, 게시, rollback을 조율하는 `RAGIndexManager` |
| `scanner.py` | 승인된 문서 트리의 안전한 스캔과 동일 바이트 재검증 |
| `planner.py` | manifest와 fingerprint를 이용한 순수 증분 계획 |
| `backends.py` | Docling, Gemma text/vision, layout VLM, BGE-M3 HTTP adapter |
| `restructure.py` | DoclingDocument를 provenance 보존 블록으로 변환 |
| `chunking.py` | 구조 인식 결정론적 청킹과 embedding text 생성 |
| `artifacts.py` | Docling/레이아웃/분석/청크 중간 산출물 캐시와 무결성 검증 |
| `catalog.py` | SQLite run/generation/document/chunk manifest |
| `stores.py` | 실제 Milvus adapter와 테스트용 `InMemoryMilvusStore` |
| `diagnostics.py` | 내용 비노출 단계 로그와 Agent run 실패 상관관계 |
| `supervisor.py` | 지속 polling, 안정화, 자동 sync, 백오프, 격리, 단일 process lease |
| `models.py` | source, block, provenance, chunk, plan, fingerprint 데이터 계약 |
| `cli.py` | 환경 변수 기반 live 구성과 CLI 진입점 |

## 필요한 서비스와 설치

실제 실행 전에 다음 서비스가 필요하다.

- Docling Serve
- `gemma-4-26B-A4B-it`를 제공하는 OpenAI-compatible vLLM endpoint
- `BGE-M3`를 제공하는 vLLM embedding endpoint
- Milvus 2.5+

Milvus Python client만 별도로 설치한다.

```bash
python3 -m pip install -r examples/14_rag_index_manager/requirements.txt
```

### 가장 빠른 로컬 구성

저장소 루트에서 환경 템플릿을 복사한다.

```bash
cp examples/14_rag_index_manager/.env.example .env
```

`.env`에서 필수로 확인할 값은 네 개다.

```dotenv
RAG_VLLM_BASE_URL=http://your-gemma-vllm:8000/v1
RAG_TEXT_MODEL=google/gemma-4-26B-A4B-it

RAG_EMBEDDING_BASE_URL=http://your-bge-vllm:8001/v1
RAG_EMBEDDING_MODEL=BAAI/bge-m3
```

두 endpoint에 인증이 있다면 `RAG_VLLM_API_KEY`와
`RAG_EMBEDDING_API_KEY`도 입력한다. BGE-M3를 다른 차원으로 제공했다면
`RAG_EMBEDDING_DIMENSION`을 실제 dense vector 차원으로 바꾼다. 하나의 vLLM
프로세스가 두 모델을 동시에 제공하지 않는 구성이라면 Gemma와 BGE-M3용 URL이
각각 필요하다.

Docling Serve와 Milvus는 포함된 Compose 파일로 로컬에 실행할 수 있다.

```bash
docker compose \
  --env-file .env \
  -f examples/14_rag_index_manager/compose.yaml \
  up -d
```

Compose 기본값은 CPU Docling Serve `v1.21.0`과 Milvus `v2.5.27`을
`127.0.0.1`에만 노출한다. Docling 처리 속도가 중요하고 NVIDIA 환경이 준비되어
있다면 `.env`의 `DOCLING_SERVE_IMAGE`를 호스트 CUDA와 맞는 공식 이미지로
교체한다. vLLM 서비스는 Compose에 포함하지 않는다.

### NIST 사이버보안 문서 100개 준비

다음 명령은 NIST의 현재 Final SP 800 목록에서 100개를 고정 선택하고 PDF를
내려받는다.

```bash
python3 -m examples.14_rag_index_manager.sample_data \
  --env-file .env
```

출력은 기본적으로 다음 위치에 생성된다.

```text
examples/14_rag_index_manager/.runtime/nist-cybersecurity/
├── documents/             # Agent 입력 PDF 100개
├── selection.json         # 고정한 상세 페이지/PDF URL
└── corpus-manifest.json   # 제목, URL, 파일명, 크기, SHA-256
```

다운로드가 중단되어도 같은 `selection.json`으로 재개하며, 완성된 파일은 manifest의
크기와 SHA-256을 다시 확인한다. NIST 이외 host로 향하는 redirect, PDF가 아닌
응답, 100 MiB가 넘는 개별 파일과 2 GiB가 넘는 전체 corpus는 거부한다. 공개
자료라도 원 출처 표시를 유지하고 문서별 제3자 저작권 표시는 별도로 확인한다.

처음에는 `.env`의 `RAG_SAMPLE_DOCUMENT_COUNT=5`로 smoke test한 뒤 100으로 늘리는
것을 권장한다. 페이지 이미지와 VLM 보정까지 켜면 100개 전체 처리는 상당한 GPU
시간이 필요할 수 있다.

환경 변수 예시는 다음과 같다.

```bash
export RAG_DOCUMENT_ROOT=/srv/assistant-documents
export RAG_STATE_DIR=/var/lib/rag-index-manager
export RAG_KB_ID=corporate-assistant

export DOCLING_SERVE_URL=http://localhost:5001
export DOCLING_SERVE_REVISION='<고정한 이미지 버전 또는 digest>'
# export DOCLING_SERVE_API_KEY='...'
# export DOCLING_SERVE_DO_OCR=true

export RAG_VLLM_BASE_URL=http://localhost:8000/v1
export RAG_TEXT_MODEL=google/gemma-4-26B-A4B-it
# export RAG_LAYOUT_MODEL=gemma-4-26B-A4B-it
# export RAG_VLLM_API_KEY='...'

export RAG_EMBEDDING_BASE_URL=http://localhost:8001/v1
export RAG_EMBEDDING_MODEL=BAAI/bge-m3
export RAG_EMBEDDING_DIMENSION=1024
# export RAG_EMBEDDING_API_KEY='...'

export RAG_MILVUS_URI=http://localhost:19530
# export RAG_MILVUS_TOKEN='...'
```

`DOCLING_SERVE_REVISION`에는 배포한 Docling 이미지 tag나 digest처럼 안정적인 값을
넣는 것이 좋다. 이 값은 parser fingerprint에 포함되므로, 파서 버전을 변경하면
필요한 문서가 자동으로 parse 단계부터 다시 처리된다.

## CLI 실행

CLI는 현재 디렉터리의 `.env`를 자동으로 읽는다. 셸에서 이미 설정한 환경변수는
`.env`보다 우선한다. 다른 파일은 `--env-file path/to/file`로 지정한다. `.env`
값은 literal로만 읽으며 `$VAR`, command substitution 또는 셸 코드를 실행하지
않는다.

상태 또는 변경 계획 확인은 쓰기 권한 없이 실행한다.

```bash
python3 -m examples.14_rag_index_manager \
  --env-file .env \
  --request "변경된 문서를 확인하고 인덱싱 계획을 보여줘"
```

실제 새 세대를 게시하려면 애플리케이션이 명시적으로 `--apply`를 제공해야 한다.

```bash
python3 -m examples.14_rag_index_manager \
  --env-file .env \
  --request "변경된 문서를 반영해서 인덱스를 동기화해줘" \
  --apply \
  --verbose
```

지원되는 요청은 상태 확인, 증분 계획, 증분 동기화, 전체 재구축, 직전 세대
rollback이다. `--documents`, `--state-dir`, `--kb-id`,
`--embedding-dimension`으로 기본 구성을 덮어쓸 수 있다.

`--verbose`는 Agent의 모델·Tool 실행과 그 아래의 Docling 파싱, VLM 레이아웃
보정, 청킹, BGE-M3 임베딩, Milvus 게시 단계를 계층형 진행 로그로 stderr에
즉시 출력한다. 최종 `ManagementResponse` JSON만 stdout으로 출력한다.
로그 수집용 JSON이 필요하면 `--log-format json`을 추가한다. 예쁜 로그의
표시 언어는 기본 `ko`이며 `--log-language en`으로 바꿀 수 있다. `.env`에서는
각각 `RAG_LOG_FORMAT`, `RAG_LOG_LANGUAGE`로 설정할 수 있다.

### 파일 유입을 계속 자동 반영하기

다음 명령은 장시간 실행되는 ingestion supervisor를 시작한다. `--watch` 자체가
자동 쓰기 실행에 대한 애플리케이션의 명시적 선택이므로 `--apply`와 자연어 요청은
사용하지 않는다.

```bash
python3 -m examples.14_rag_index_manager \
  --env-file .env \
  --watch
```

동작은 다음과 같다.

1. 기본 5초마다 문서 트리를 안전하게 스캔한다.
2. content snapshot이 기본 15초 동안 바뀌지 않아야 처리를 시작한다.
3. 신규·수정·삭제 및 pipeline fingerprint 변경분만 자동 동기화한다.
4. 실패하면 최대 5회까지 5초에서 300초 사이의 지수 백오프로 재시도한다.
5. 동일 snapshot이 계속 실패하면 격리하고, 새 revision이 들어오면 자동 해제한다.
6. 기본 5분마다 unchanged 상태도 전체 reconciliation한다.

`supervisor-state.json`은 state directory에 원자적으로 저장되므로 프로세스 재시작
후에도 성공 snapshot, 재시도 횟수와 격리 상태를 이어간다. lock file은 같은 state
directory를 쓰는 supervisor가 두 개 동시에 동작하는 것을 막는다. 운영에서는 이
명령을 systemd, Docker restart policy 또는 Kubernetes Deployment처럼 프로세스를
재시작해 주는 서비스 관리자 아래에 둔다.

watcher는 `rag_ingestion_supervisor` JSON과 기존 `rag_index_progress` JSON을
stderr에 출력한다. 문서 경로, 파일명, 원문, URL, API key와 backend 응답 본문은
포함하지 않는다. 설정값은 `.env`의 `RAG_WATCH_*` 항목이나 대응 CLI 옵션으로
조정할 수 있다.

파일 생산자는 가능하면 같은 filesystem에서 `.tmp`처럼 지원하지 않는 확장자로
완성한 뒤 최종 이름으로 원자적 rename한다. stability window와 파싱 직전 identity
재검증이 부분 파일을 방어하지만, 원자적 handoff가 가장 명확한 운영 계약이다.

VLM endpoint가 특정 페이지를 HTTP 오류로 거부하면 그 페이지만 canonical Docling
순서를 유지하고, 성공한 다른 페이지의 patch는 계속 적용한다. 이때
`rag_layout_fallback` warning에는 opaque source ID, page 번호, HTTP status만 남는다.
잘못된 모델 JSON, 위조·중복 reference, 손상된 page capture는 availability 문제로
간주하지 않고 계속 fail-closed 처리하며, 안전하지 않은 선택적 local 관계는
문서 구조로 채택하지 않고 제거한다.

## Jupyter Notebook에서 Python 코드로 실행

디렉터리 이름이 숫자로 시작하므로 일반 `from examples.14_... import ...` 문법은
사용할 수 없다. `importlib.import_module()`을 사용한다. 아래 코드는 관리 Agent와
모든 backend를 직접 구성하는 전체 예시다.

```python
import importlib
import logging
import os
from pathlib import Path

from moduagent import (
    AgentRunError,
    InMemoryDiagnosticSink,
    LoggingEventSink,
    VLLMClient,
)

rag = importlib.import_module("examples.14_rag_index_manager")
rag.load_environment_file(".env", required=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")

document_root = Path("/srv/assistant-documents")
state_root = Path("/var/lib/rag-index-manager")
embedding_dimension = 1024

# stderr/stdout 대신 노트북 셀에서 바로 진행 로그를 확인한다.
execution_log = rag.PipelineExecutionLog.pretty(include_timestamp=True)
diagnostics = InMemoryDiagnosticSink(max_records=100)

parser = rag.DoclingServeClient(
    base_url=os.getenv("DOCLING_SERVE_URL", "http://localhost:5001"),
    api_key=rag.environment_secret("DOCLING_SERVE_API_KEY"),
    parser_revision=os.getenv("DOCLING_SERVE_REVISION", "local-dev"),
    do_ocr=False,
    generate_page_images=True,
    images_scale=1.0,
)
office_renderer = (
    rag.OfficePageCaptureRenderer()
    if os.getenv("RAG_OFFICE_PAGE_CAPTURE", "false").lower()
    in {"1", "true", "yes", "on"}
    else None
)
layout_refiner = rag.VLLMLayoutRefinementClient(
    base_url=os.getenv("RAG_VLLM_BASE_URL", "http://localhost:8000/v1"),
    api_key=rag.environment_secret("RAG_VLLM_API_KEY"),
    model="google/gemma-4-26B-A4B-it",
    allow_exclusions=False,
    page_capture_renderer=office_renderer,
)
enricher = rag.VLLMEnrichmentClient(
    base_url=os.getenv("RAG_VLLM_BASE_URL", "http://localhost:8000/v1"),
    api_key=rag.environment_secret("RAG_VLLM_API_KEY"),
    model="google/gemma-4-26B-A4B-it",
)
embedder = rag.VLLMEmbeddingClient(
    base_url=os.getenv("RAG_EMBEDDING_BASE_URL", "http://localhost:8001/v1"),
    api_key=rag.environment_secret("RAG_EMBEDDING_API_KEY"),
    model="BAAI/bge-m3",
)
milvus = rag.MilvusStore(
    uri=os.getenv("RAG_MILVUS_URI", "http://localhost:19530"),
    token=rag.environment_secret("RAG_MILVUS_TOKEN"),
    alias="corporate_assistant_active",
    collection_prefix="corporate_assistant_chunks",
)

chunking = rag.ChunkingConfig(
    max_chars=2_400,
    overlap_chars=240,
)
pipeline_fingerprint = rag.build_pipeline_fingerprint(
    parser=parser,
    refiner=layout_refiner,
    enricher=enricher,
    embedder=embedder,
    restructuring_revision=rag.RESTRUCTURING_FINGERPRINT,
    chunking_revision=chunking.fingerprint,
    indexing_revision=milvus.fingerprint,
    embedding_dimension=embedding_dimension,
)

catalog = rag.ManifestCatalog(state_root / "manifest.sqlite3")
artifacts = rag.ArtifactStore(state_root / "artifacts")
supervisor_state = rag.SupervisorStateStore(state_root / "supervisor-state.json")
manager = rag.RAGIndexManager(
    config=rag.ManagerConfig(
        document_root=document_root,
        kb_id="corporate-assistant",
        embedding_dimension=embedding_dimension,
        chunking=chunking,
    ),
    pipeline=pipeline_fingerprint,
    catalog=catalog,
    artifacts=artifacts,
    parser=parser,
    refiner=layout_refiner,
    enricher=enricher,
    embedder=embedder,
    vector_store=milvus,
    execution_log=execution_log,
    write_lease=supervisor_state.operation_lease,
)

management_model = VLLMClient(
    base_url=os.getenv("RAG_VLLM_BASE_URL", "http://localhost:8000/v1"),
    model="google/gemma-4-26B-A4B-it",
    api_key=os.getenv("RAG_VLLM_API_KEY"),
    timeout=90,
    default_options={"temperature": 0, "max_tokens": 1_024},
)
```

Jupyter에서는 다음 셀처럼 top-level `await`로 실행한다. 먼저 dry-run을 권장한다.

```python
async def run_request(request: str, *, allow_writes: bool = False):
    try:
        return await rag.run_management_request(
            management_model,
            manager,
            request,
            allow_writes=allow_writes,
            event_sink=LoggingEventSink(),
            diagnostic_sink=diagnostics,
            supervisor_state_provider=lambda: supervisor_state.load(
                kb_id=manager.config.kb_id,
                pipeline_digest=manager.pipeline.digest,
            ),
        )
    except AgentRunError as error:
        print(
            rag.format_management_failure(
                error,
                execution_log=execution_log,
                diagnostic_sink=diagnostics,
            )
        )
        raise
```

하나의 async lifecycle 안에서 계획을 확인하고, 확인 후 실제 동기화를 수행한다.
실제 게시가 필요하지 않다면 두 번째 호출은 생략한다.

```python
try:
    async with (
        parser,
        layout_refiner,
        enricher,
        embedder,
        milvus,
        management_model,
    ):
        preview = await run_request(
            "현재 문서 변경분과 처리 계획을 보여줘",
            allow_writes=False,
        )
        display(preview.model_dump())

        result = await run_request(
            "변경된 문서를 반영해 새 인덱스 세대를 게시해줘",
            allow_writes=True,
        )
        display(result.model_dump())
finally:
    catalog.close()
```

종료된 client 인스턴스를 다음 셀에서 다시 사용하지 않는다. 새 작업을 실행하려면
backend와 manager를 다시 구성하거나, 여러 요청을 위와 같은 하나의 async
lifecycle 안에서 실행한다. SQLite catalog는 `finally`에서 닫는다.

AI Agent를 거치지 않고 결정론적 관리 API를 직접 호출할 수도 있다.

```python
status = await manager.status()
plan = await manager.preview()

# 실제 쓰기 작업
report = await manager.sync()

# 전체 재구축
rebuild = await manager.sync(force_rebuild=True)

# 직전 성공 세대로 복구
rollback = await manager.rollback()
```

직접 호출은 자연어 routing과 `ManagementResponse` 생성이 필요 없을 때 유용하다.
운영 자동화에서는 이 방식이 더 단순할 수 있고, 사람의 자연어 요청을 받아야 할
때만 관리 Agent를 앞에 둔다. 이 직접 호출도 위의 async lifecycle 안에서 수행해야
한다.

### 검색 품질 평가

게시된 active generation은 비공개 평가 질의와 기대 문서의 opaque `source_id`로
검증한다. 리포트에는 질의 원문이나 문서 본문이 들어가지 않고, case별 검색된
source 순서와 source-deduplicated Hit@1/Hit@K, MRR, Recall@K, MAP, hard-negative
오탐률, tag별 slice, 소요 시간과 처리량만 포함된다.

```python
sources = {
    source.relative_path: source
    for source in rag.scan_document_directory(
        document_root,
        kb_id=manager.config.kb_id,
    )
}
cases = (
    rag.RetrievalEvaluationCase(
        "leave-policy",
        "연차 휴가 일수는 며칠인가?",
        (sources["hr/leave-policy.pdf"].source_id,),
    ),
)

async with embedder, milvus:
    quality = await rag.evaluate_retrieval(
        embedder,
        milvus,
        cases,
        top_k=5,
    )

print("Hit@1", quality.hit_rate_at_1)
print("Hit@5", quality.hit_rate)
print("MRR", quality.mean_reciprocal_rank)
```

고정되고 사람이 검토한 평가 case를 CI나 새 generation 게시 직후 실행하는 방식을
권장한다. 소수 smoke case에서 높은 점수가 나왔다는 사실은 연결과 기본 ranking을
확인할 뿐, 운영 검색 품질 전체를 증명하지는 않는다.

예제에는 별도의 생성형 전체 수명주기 검증도 있다. 전용 빈 디렉터리에 TXT,
Markdown, HTML, CSV 정책 문서를 최대 200개 생성하고 문서마다 exact, semantic,
한국어, 식별자 없는 영어 역방향, 식별자 없는 한국어 역방향 질의를 만든다. baseline
게시, unchanged no-op, 10% 수정·5% 삭제·5% 추가, 새 세대 평가, rollback 평가,
변경분 복구와 최종 no-op 및 manifest/Milvus 일치를 한 번에 확인한다.

```bash
# 이 전용 경로는 비어 있거나 이전에 같은 harness가 만든 경로여야 한다.
RAG_DOCUMENT_ROOT=/tmp/rag-validation-documents \
RAG_STATE_DIR=/tmp/rag-validation-state \
RAG_KB_ID=rag-validation \
python3 -m examples.14_rag_index_manager \
  --env-file .env --validate-generated 100 --validation-top-k 5

# 게시된 세대를 변경하지 않고 500개 질의만 다시 평가한다.
RAG_DOCUMENT_ROOT=/tmp/rag-validation-documents \
RAG_STATE_DIR=/tmp/rag-validation-state \
RAG_KB_ID=rag-validation \
python3 -m examples.14_rag_index_manager \
  --env-file .env --evaluate-generated --validation-top-k 5
```

두 번째 명령에만 `--validation-details`를 추가해 opaque case별 순위를 볼 수 있다.
기본 gate는 전체 Hit@1 0.70, Hit@5/Recall@5 0.90, MRR/MAP 0.80 이상,
hard-negative Top-1 0.20 이하이며 모든 형식·주제·질의 slice가 Hit@1 0.60,
Hit@5 0.80 이상이어야 한다. 생성 문서는 의도적으로 template 기반이고 식별자가
들어간 질의는 쉽다. 따라서 anchor-free slice를 더 강한 smoke signal로 보고,
운영에서는 사람이 검토한 실제 질문·정답 문서 세트를 별도로 유지해야 한다.

현재 `hybrid_search`는 최대 500개 dense 후보 안에서 IDF 기반 lexical overlap으로
재정렬한다. 예제와 중간 규모 후보에는 적합하지만 native sparse 검색은 아니다.
dense 후보에 한 번도 들어오지 않은 문서는 lexical match가 있어도 복구할 수 없다.
대규모 운영에서는 Milvus native sparse/BM25 같은 hybrid index로 교체하고 동일한
평가 계약을 유지하는 것이 좋다.

상시 감시 동작 자체를 Notebook에서 확인하려면 같은 manager로 supervisor를 만들고
실행한다. 이 셀은 중단할 때까지 계속 실행되므로 실제 운영에는 앞의 `--watch` CLI를
서비스 관리자로 실행하는 편이 낫다. 앞 셀에서 client lifecycle이 이미 종료됐다면
parser/model/store와 manager를 먼저 새로 구성한다.

```python
def show_supervisor(report):
    print("rag_ingestion_supervisor", report.to_dict())


supervisor = rag.ContinuousIngestionSupervisor(
    manager,
    supervisor_state,
    policy=rag.SupervisorPolicy(
        poll_interval_seconds=5,
        stability_window_seconds=15,
        full_reconcile_interval_seconds=300,
        max_attempts=5,
    ),
    event_sink=show_supervisor,
)

try:
    async with parser, layout_refiner, enricher, embedder, milvus:
        await supervisor.run_forever()  # Notebook interrupt로 종료
finally:
    catalog.close()
```

## 실패 원인과 실행 로그 확인

`PipelineExecutionLog`는 다음과 같은 단계 이벤트를 남긴다.

- `scan`, `plan`, `parse`, `restructure`, `refine_layout`
- `enrich`, `chunk`, `embed`, `index`
- `validate_staging`, `publish_generation`, `commit_manifest`
- `rollback`, `cleanup`

실패 이벤트에는 안정적인 `error_code`, 예외 타입 chain, 허용된 HTTP status나
`errno`, opaque `source_id`, generation ID가 포함된다. 문서 본문, 파일명, 절대
경로, 서비스 URL, credential, Tool argument, provider 응답 body는 기록하지 않는다.

`run_management_request()`에서 `AgentRunError`가 발생하면
`format_management_failure()`로 Agent failure ID와 실제 파이프라인 실패를 연결할
수 있다. 예를 들어 다음과 같은 정보가 표시된다.

```text
management Agent failed
- category: tool_failure
- operation: apply_incremental_sync
pipeline failure
- stage: embed
- error_code: embed_failed
- exception_chain: ModelBackendError -> BackendHTTPStatusError
- http_status: 429
```

이 경우 원인은 관리 Agent의 판단 실패가 아니라 BGE-M3 embedding endpoint가
HTTP 429를 반환한 것이다. 반대로 Tool을 하나도 선택하지 않았거나 여러 개를
선택하면 `category=model_protocol`로 끝나며 파이프라인 Tool은 실행되지 않는다.

## 데이터와 상태의 소유권

- `ManifestCatalog`: 어떤 generation과 source/chunk가 게시되었는지 기록한다.
- `ArtifactStore`: Docling 결과, layout patch, enrichment, chunk를 source revision과
  fingerprint에 묶어 캐시한다.
- `MilvusStore`: 실제 retrieval row와 vector, active/previous alias를 관리한다.
- `PipelineFingerprint`: 캐시를 재사용할 수 있는지 판정하는 처리 계약이다.
- `RAGIndexManager`: 세 저장소를 순서대로 조율하고 불일치 시 fail-closed한다.

SQLite에는 문서 원문이나 vector를 중복 저장하지 않는다. 원문 기반 artifact는 파일
시스템에, serving row와 vector는 Milvus에 둔다.

## 안전 및 운영 경계

- 모델에는 임의 파일 경로, URL, collection 이름, SQL을 받는 Tool이 없다.
- 경로 traversal, symlink, non-regular file, 크기·개수·깊이 제한 위반은 모델 호출
  전에 거부한다.
- Docling JSON과 provenance가 canonical이며 Gemma 출력은 파생 metadata다.
- 페이지 이미지, prompt, JSON 응답, embedding batch와 vector 차원을 모두 제한한다.
- instruction-like 문서 내용은 데이터로 취급하고 시스템 지시로 사용하지 않는다.
- staging 검증 전에는 active alias를 바꾸지 않는다.
- 이 예제의 SQLite catalog와 process-local lock은 단일 프로세스 학습용이다.
  여러 worker를 운영하려면 공유 transaction store, 분산 lease, durable publication
  journal과 reconcile 절차가 필요하다.
- 문서별 ACL은 이 인덱스 관리 예제의 범위 밖이며, 실제 검색/응답 경로의
  guardrail에서 적용한다.

## 확장 포인트

각 backend는 작은 Protocol 경계로 분리되어 있다.

- `DoclingParser`: 다른 문서 변환 서비스 또는 자체 Docling worker로 교체
- `LayoutRefiner`: 다른 VLM이나 규칙 기반 레이아웃 보정기로 교체
- `BlockEnricher`: 사내 taxonomy/NER/classifier를 추가한 분석기로 교체
- `TextEmbedder`: 다른 embedding API로 교체
- `VectorStore`: 다른 Milvus 구성이나 별도 vector store adapter로 교체
- `ArtifactStore`/`ManifestCatalog`: 공유 object storage와 PostgreSQL 기반 구현으로
  확장

교체 구현은 결과 타입, 순서, fingerprint와 제한 계약을 지켜야 한다. 모델명이나
prompt, schema, chunk 정책, index schema가 바뀌면 해당 fingerprint도 반드시
바뀌어야 안전한 단계별 재처리가 가능하다.

## 테스트

오프라인 테스트는 fake Docling/vLLM/Milvus adapter를 사용하며 네트워크를 호출하지
않는다.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python3 -m pytest -q tests/test_rag_index_manager_example_v060.py
```

테스트는 안전한 스캔, 변경 계획, artifact 무결성, PPT 레이아웃 순서 보정,
page-local ID 검증, 청킹, embedding 차원 변경, Milvus 세대 게시/rollback,
Management Agent의 단일 Tool 계약과 실패 진단 상관관계를 포함한다. 실제 Docling,
vLLM, Milvus 연결은 운영자가 명시적으로 실행하는 live 검증으로 분리한다.
