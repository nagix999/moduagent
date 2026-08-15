# RAG index manager Agent

[English](README.md) | [한국어](README.ko.md)

This example builds and maintains the retrieval index for an internal assistant.
It scans one application-approved document directory, converts changed files with
Docling Serve, enriches text and pictures through vLLM, creates deterministic
retrieval chunks, embeds them with BGE-M3, and publishes a validated Milvus
generation.

For continuous operation, `ContinuousIngestionSupervisor` polls that directory,
waits for a stable content snapshot, applies incremental synchronization without
an LLM decision, retries bounded transient failures, and quarantines one
repeatedly failing snapshot until a new revision arrives. The AI management
Agent remains the control plane for status, preview, rebuild, rollback, and
failure explanation.

The bundled defaults are intentionally explicit:

- text and vision analysis: `gemma-4-26B-A4B-it`
- dense embeddings: `BGE-M3`
- parser: Docling Serve's asynchronous file-conversion API
- serving index: Milvus dense retrieval plus bounded lexical reranking, with
  source text and provenance retained for application-owned retrieval
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
(`include_page_images=true`, `image_export_mode=embedded`). The configured
image scale is part of the parser fingerprint. Page images are optional in the
DoclingDocument contract, so a document without usable captures follows the
original deterministic Docling reading order without contacting the layout
model. See Docling's [REST API documentation](https://docling-project.github.io/docling/usage/api_server/rest_api/)
and [pipeline options reference](https://docling-project.github.io/docling/reference/pipeline_options/)
for the server and page-image options.

Some Docling versions return no whole-page image for PPT/PPTX or office
documents even when page-image generation is enabled. With
`RAG_OFFICE_PAGE_CAPTURE=true`, the CLI uses the bounded
`OfficePageCaptureRenderer` fallback: it converts an already validated office
file to PDF with headless LibreOffice and renders pages to PNG with
`pdftocairo`. Temporary files are private, subprocesses and output sizes are
bounded, and the renderer fingerprint invalidates only the layout-refinement
stage. Install `soffice` and `pdftocairo` on the Agent host before enabling it.

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

Office conversion may assign the same bounding box to several lines extracted
from one text box. The image cannot map opaque IDs to individual lines in that
case. The application therefore preserves Docling's relative order within the
shared box and clears model-proposed roles, parents, groups, exclusions, and
ambiguous heading references for those IDs. A page-local section reference
that points at a valid but non-heading block is discarded as optional metadata;
unknown, duplicate, missing, or cross-page IDs still reject the complete patch.

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

Provider-side HTTP rejection is isolated per page: that page retains canonical
Docling order while successful page patches are still applied. A content-free
`rag_layout_fallback` warning records only opaque source ID, page number, and
HTTP status. Malformed model output, forged references, and invalid captures
still fail closed; unsupported optional local relationships are removed rather
than being allowed to manufacture document structure.

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

For the fastest local setup, copy the environment template and provide the
Gemma and BGE-M3 vLLM endpoints and served model names:

```bash
cp examples/14_rag_index_manager/.env.example .env
```

The four required deployment values are `RAG_VLLM_BASE_URL`,
`RAG_TEXT_MODEL`, `RAG_EMBEDDING_BASE_URL`, and `RAG_EMBEDDING_MODEL`. Add the
two API-key variables when the endpoints require authentication, and change
`RAG_EMBEDDING_DIMENSION` if the served dense vector is not 1,024 dimensions.
The bundled Compose file supplies local Docling Serve and Milvus; vLLM remains
application-owned:

```bash
docker compose \
  --env-file .env \
  -f examples/14_rag_index_manager/compose.yaml \
  up -d
```

Download a pinned 100-document NIST SP 800 cybersecurity corpus. The command
stores the PDFs below `.runtime/` with an exact URL selection and SHA-256
manifest and resumes without silently changing that selection:

```bash
python3 -m examples.14_rag_index_manager.sample_data --env-file .env
```

Start with `RAG_SAMPLE_DOCUMENT_COUNT=5` for a live smoke test before processing
all 100 PDFs with page-image refinement. The downloader accepts only the NIST
CSRC and NVL publication hosts, verifies PDF magic and bounded sizes, and keeps
the attribution URLs in `corpus-manifest.json`.

Configuration is read from environment variables by the CLI. No credentials
are embedded in source or sent to the management model.

```bash
export RAG_DOCUMENT_ROOT=/srv/assistant-documents
export RAG_STATE_DIR=/var/lib/rag-index-manager

export DOCLING_SERVE_URL=http://localhost:5001
export DOCLING_SERVE_REVISION='<pinned image version or digest>'
# export DOCLING_SERVE_API_KEY='...'

export RAG_VLLM_BASE_URL=http://localhost:8000/v1
export RAG_TEXT_MODEL=google/gemma-4-26B-A4B-it
# The same multimodal Gemma deployment analyzes Docling picture blocks.
# export RAG_VLLM_API_KEY='...'

export RAG_EMBEDDING_BASE_URL=http://localhost:8001/v1
export RAG_EMBEDDING_MODEL=BAAI/bge-m3
# export RAG_EMBEDDING_API_KEY='...'

export RAG_MILVUS_URI=http://localhost:19530
# export RAG_MILVUS_TOKEN='...'
```

Run a status check or an incremental dry-run from the repository root. Natural
language selects exactly one bounded management operation. The CLI loads
`.env` automatically without shell evaluation; process environment variables
take precedence, and `--env-file` selects another file:

```bash
python3 -m examples.14_rag_index_manager \
  --env-file .env \
  --request "변경된 문서를 확인하고 인덱싱 계획을 보여줘"
```

Write Tools are absent unless the application explicitly supplies `--apply`.
The same request can then publish a validated generation:

```bash
python3 -m examples.14_rag_index_manager \
  --env-file .env \
  --request "변경된 문서를 반영해서 인덱스를 동기화해줘" \
  --apply \
  --verbose
```

`--verbose` streams a pretty, hierarchical Agent and pipeline timeline to
stderr. It nests Docling parsing, VLM layout refinement, chunking, BGE-M3
embedding, and Milvus publication beneath model and Tool activity. Add
`--log-format json` for content-free machine records, or `--log-language en`
to override the default Korean pretty labels. The equivalent `.env` keys are
`RAG_LOG_FORMAT` and `RAG_LOG_LANGUAGE`. Each JSON record identifies the
operation, deterministic pipeline stage, opaque request correlation ID,
`source_id`, generation, progress count, and terminal status. A failed stage
also includes a stable error code, exception type chain, and allowlisted HTTP
status or `errno` when available. It never includes document text, filenames,
absolute paths, service URLs, credentials, Tool arguments, or backend response
bodies. The final management JSON remains the only stdout output.

### Continuous ingestion

Run the deterministic watcher as a long-lived process:

```bash
python3 -m examples.14_rag_index_manager \
  --env-file .env \
  --watch
```

The defaults poll every five seconds, require a 15-second unchanged snapshot,
reconcile every five minutes, and retry a failing snapshot up to five times
with bounded exponential backoff. The checkpoint and single-process lease live
under `RAG_STATE_DIR`; restart the command with systemd, a container restart
policy, or Kubernetes. It emits content-free `rag_ingestion_supervisor` and
`rag_index_progress` JSON records to stderr. Configure the behavior with the
`RAG_WATCH_*` variables in `.env` or the corresponding CLI flags.
Producers should preferably finish a file under an unsupported temporary suffix
on the same filesystem and atomically rename it to its final name. The settle
window and pre-parse identity recheck remain defense in depth.

### Retrieval quality evaluation

After publication, evaluate the active generation with private queries and
opaque expected source IDs. The returned report contains no query or document
content. It reports source-deduplicated Hit@1/Hit@K, MRR, Recall@K, MAP,
forbidden hard-negative rates, tagged slice metrics, duration, and throughput:

```python
sources = {
    source.relative_path: source
    for source in rag.scan_document_directory(document_root, kb_id=config.kb_id)
}
cases = (
    rag.RetrievalEvaluationCase(
        "leave-policy",
        "How many annual leave days are available?",
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

print(quality.hit_rate_at_1, quality.hit_rate, quality.mean_reciprocal_rank)
```

Run evaluation against a fixed, reviewed case set in CI or after each newly
published generation. A high metric on a tiny smoke set is a connectivity
check, not proof of production retrieval quality.

The example also includes a destructive, generated lifecycle harness. It owns
the target directory, creates up to 200 deterministic policy documents in TXT,
Markdown, HTML, and CSV, and produces five queries per document: exact,
semantic, Korean, English anchor-free reverse lookup, and Korean anchor-free
reverse lookup. The harness publishes a baseline, verifies an unchanged no-op,
modifies 10%, deletes 5%, adds 5%, measures the new generation, rolls back and
remeasures, recovers the mutation, and verifies a final no-op and
manifest/Milvus consistency.

```bash
# Use only an empty directory or a directory previously created by this harness.
RAG_DOCUMENT_ROOT=/tmp/rag-validation-documents \
RAG_STATE_DIR=/tmp/rag-validation-state \
RAG_KB_ID=rag-validation \
python3 -m examples.14_rag_index_manager \
  --env-file .env --validate-generated 100 --validation-top-k 5

# Re-evaluate the published generation without rebuilding or mutating it.
RAG_DOCUMENT_ROOT=/tmp/rag-validation-documents \
RAG_STATE_DIR=/tmp/rag-validation-state \
RAG_KB_ID=rag-validation \
python3 -m examples.14_rag_index_manager \
  --env-file .env --evaluate-generated --validation-top-k 5
```

`--validation-details` may be added only to the second command and includes
opaque per-case rankings. The built-in gate requires overall Hit@1 >= 0.70,
Hit@5/Recall@5 >= 0.90, MRR/MAP >= 0.80, forbidden Top-1 <= 0.20, and every
format/category/query slice Hit@1 >= 0.60 and Hit@5 >= 0.80. The generated
documents are deliberately templated, and identifier-bearing queries are
easier than real user questions. Treat the anchor-free slices as the stronger
smoke signal and maintain a separate, human-reviewed production evaluation set.

The current `hybrid_search` implementation reranks at most 500 dense candidates
with bounded IDF-weighted lexical overlap. This is appropriate for the example
and medium candidate sets, but it is not native sparse retrieval: a document
that never enters the dense candidate pool cannot be rescued. At larger scale,
use Milvus native sparse/BM25 or another production hybrid index and preserve
the same evaluation contract.

### Jupyter failure diagnosis

Use a bounded execution log together with ModuAgent's diagnostic sink when
constructing the manager in a notebook. The Agent event sink shows the safe
model/Tool lifecycle, while the pipeline log prints ingestion progress as it
happens:

```python
import importlib
import logging

from moduagent import (
    AgentRunError,
    InMemoryDiagnosticSink,
    LoggingEventSink,
)

rag = importlib.import_module("examples.14_rag_index_manager")

logging.basicConfig(level=logging.INFO, format="%(message)s")
execution_log = rag.PipelineExecutionLog.pretty(
    include_timestamp=True,
    language="en",
)
diagnostics = InMemoryDiagnosticSink(max_records=100)

# Add execution_log to the same application-owned manager construction shown
# in this example. All other parser/model/store arguments stay unchanged.
manager = rag.RAGIndexManager(
    config=config,
    pipeline=pipeline,
    catalog=catalog,
    artifacts=artifacts,
    parser=docling,
    refiner=layout_refiner,
    enricher=enricher,
    embedder=embedder,
    vector_store=milvus,
    execution_log=execution_log,
)

try:
    response = await rag.run_management_request(
        management_model,
        manager,
        "변경된 문서를 반영해서 인덱스를 동기화해줘",
        allow_writes=True,
        event_sink=LoggingEventSink(),
        diagnostic_sink=diagnostics,
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

The formatted diagnosis correlates `AgentRunError.failure_id` with the runtime
diagnostic and reports the precise pipeline stage, stable failure code,
exception type chain, safe HTTP/OS facts, and bounded source-code frames. Raw
provider messages are intentionally excluded because they may contain document
or credential data. If the remaining cause is an external service failure,
use the timestamp, stage, generation, and opaque source ID to inspect the
corresponding Docling, vLLM, or Milvus server log.

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
