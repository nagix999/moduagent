# RAG index manager Agent

This example builds and maintains the retrieval index for an internal assistant.
It scans one application-approved document directory, converts changed files with
Docling Serve, enriches text and pictures through vLLM, creates deterministic
retrieval chunks, embeds them with BGE-M3, and publishes a validated Milvus
generation.

The bundled defaults are intentionally explicit:

- text and vision analysis: `gemma-4-26B-A4B-it`
- dense embeddings: `BGE-M3`
- parser: Docling Serve's asynchronous file-conversion API
- serving index: Milvus dense-vector retrieval with source text and provenance
- lifecycle source of truth: a local SQLite manifest

## Pipeline

```text
approved directory
  -> bounded scan + SHA-256 revision
  -> fingerprint-aware incremental plan
  -> DoclingDocument JSON
  -> structure-preserving blocks
  -> optional, validated whole-page layout refinement
  -> Gemma text / picture metadata
  -> deterministic chunks and provenance
  -> BGE-M3 embeddings
  -> Milvus staging generation
  -> count/dimension/content validation
  -> validated Milvus alias publication
```

For fixed-layout PDF and PowerPoint sources, Docling Serve is asked to generate
whole-page images and return them as embedded image references
(`generate_page_images=true`, `image_export_mode=embedded`). The configured
image scale is part of the parser fingerprint. Page images are optional in the
DoclingDocument contract, so a document without usable captures follows the
original deterministic Docling reading order without contacting the layout
model. See Docling's [REST API documentation](https://docling-project.github.io/docling/usage/api_server/rest_api/)
and [pipeline options reference](https://docling-project.github.io/docling/reference/pipeline_options/)
for the server and page-image options.

Layout refinement is deliberately narrow. The multimodal model receives a
bounded page image plus page-local block descriptors as untrusted data, and may
only propose an ordering, document role, grouping hierarchy, and section
ancestry using existing opaque block IDs. Human-readable section paths are
derived by application code from the referenced canonical heading text; the
model cannot invent retrieval metadata strings. Decorative items and repeated
headers or footers may be excluded only with an application-enumerated reason;
excluded IDs still have to appear exactly once in the page decision. The
application rejects unknown, missing, duplicate, or cross-page IDs. It never
accepts replacement text or provenance from the model: the raw Docling
artifact, extracted text, source
locations, modality, label, and block identities remain byte-for-byte
application-owned. A changed refinement model, prompt, or policy fingerprint
resumes at that stage while reusing the cached Docling parse.

The CLI and default application policy set `allow_exclusions=false`, so no
block is removed even if the model proposes an exclusion. Enabling exclusion
requires an explicit application-policy opt-in and a refiner configured with
the same policy; that setting is included in the refinement fingerprint.

The live layout adapter defaults to at most 32 blocks per page and 16,384
output tokens, and both limits are fingerprinted. A page above the block budget
is not sent to vLLM; its blocks retain Docling's raw order and structure so one
dense slide cannot fail the rest of the document. Image, page-count, prompt,
and HTTP-response limits are enforced separately, with malformed captures and
responses rejected before a patch is accepted.

The model never chooses source hashes, chunk IDs, deleted rows, collection
names, or alias targets. Those decisions are made by deterministic application
code. An unchanged source and unchanged pipeline fingerprints are skipped. A
changed stage fingerprint resumes at the earliest affected stage; an embedding
or Milvus schema change creates a new generation. The expected embedding
dimension is fingerprinted separately; changing it disables active-row copying
and re-embeds every current document from its cached chunks.

The manifest and parse artifacts are separate from Milvus. Milvus is the
serving index, not the job ledger. A generation is published only after all
planned documents have completed and staging validation succeeds. The previous
alias target remains available for bounded rollback.

The Milvus alias switch and the SQLite manifest commit span two stores, so they
cannot form one database transaction. If a post-publish manifest commit fails,
the manager attempts to restore the previous Milvus alias. A missing or
mismatched generation then fails closed and requires operator reconciliation;
the first-ever publication has no predecessor to restore. This teaching
example does not guess which store should win after such a split. A production
deployment should add a durable transition journal and an operator-approved
reconciler before enabling multiple workers.

## Services

Run Docling Serve, a Gemma vLLM endpoint, a BGE-M3 vLLM pooling endpoint, and
Milvus before using the live CLI. The example imports `pymilvus` lazily, so the
normal ModuAgent installation and offline tests do not require it:

```bash
python3 -m pip install 'pymilvus>=2.5,<3'
```

Configuration is read from environment variables by the CLI. No credentials
are embedded in source or sent to the management model.

```bash
export RAG_DOCUMENT_ROOT=/srv/assistant-documents
export RAG_STATE_DIR=/var/lib/rag-index-manager

export DOCLING_SERVE_URL=http://localhost:5001
export DOCLING_SERVE_REVISION='<pinned image version or digest>'
# export DOCLING_SERVE_API_KEY='...'

export RAG_VLLM_BASE_URL=http://localhost:8000/v1
export RAG_TEXT_MODEL=gemma-4-26B-A4B-it
# The same multimodal Gemma deployment analyzes Docling picture blocks.
# export RAG_VLLM_API_KEY='...'

export RAG_EMBEDDING_BASE_URL=http://localhost:8001/v1
export RAG_EMBEDDING_MODEL=BGE-M3
# export RAG_EMBEDDING_API_KEY='...'

export RAG_MILVUS_URI=http://localhost:19530
# export RAG_MILVUS_TOKEN='...'
```

Run a status check or an incremental dry-run from the repository root. Natural
language selects exactly one bounded management operation:

```bash
python3 -m examples.14_rag_index_manager \
  --request "변경된 문서를 확인하고 인덱싱 계획을 보여줘"
```

Write Tools are absent unless the application explicitly supplies `--apply`.
The same request can then publish a validated generation:

```bash
python3 -m examples.14_rag_index_manager \
  --request "변경된 문서를 반영해서 인덱스를 동기화해줘" \
  --apply
```

Other supported intents are current status, full rebuild, and immediate
previous-generation rollback. `--documents`, `--state-dir`, `--kb-id`, and
`--embedding-dimension` override the corresponding application configuration.
The command returns a closed JSON management result; it never prints document
content, service credentials, or absolute source paths.

The management Agent is bound to the configured directory and service
adapters. It never accepts an arbitrary filesystem path, URL, collection name,
or SQL statement as a model-generated Tool argument. Without the CLI's explicit
apply flag, only status and dry-run planning Tools are exposed.

## Safety and operational boundaries

- Path traversal, symlinks, non-regular files, unsupported extensions, and
  configured file/count/aggregate byte limits fail before any model call.
- The same validated bytes are hashed and uploaded to Docling; source paths are
  never handed to Docling or the model.
- Docling JSON and original provenance remain canonical. Gemma-derived summaries,
  tags, and captions are stored as inferred metadata, never as source text.
- Whole-page captures and layout prompts are size/count bounded. Embedded page
  images increase Docling response size and memory use, so production limits
  should be set to the approved corpus and service capacity. Instructions found
  inside page images or extracted text are source data, not model instructions.
- Layout refinement only changes validated page-local order and hierarchy. If a
  capture is absent, the pipeline keeps Docling's deterministic order and does
  not make a refinement request; malformed or over-limit captures fail closed.
- vLLM output is validated against closed JSON schemas and bounded adapters.
  Picture data is bounded and accepted only from Docling-produced artifacts.
- Embedding batches, vector dimensions, finite values, response indices, HTTP
  timeouts, retries, and response sizes are bounded.
- Destructive changes are isolated in staging until validation succeeds. A
  failure before alias publication leaves the active alias unchanged; the
  cross-store failure boundary described above is deliberately fail-closed.
- SQLite is suitable for this single-process example. Use a transactional,
  shared job store such as PostgreSQL before running multiple workers.

The offline tests use deterministic fake Docling/vLLM/Milvus adapters and make
no network calls. A live run is intentionally opt-in.
