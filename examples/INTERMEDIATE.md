# Intermediate examples

These examples target ModuAgent `0.6.2`. Complete the beginner path in
[`README.md`](README.md) first, then follow `10` → `11` → `12` → `13`. Each
example is standalone. Examples `10`–`12` use deterministic application-owned
data; example `13` uses operator-approved local files and a Docling Serve
endpoint. You can replace their bounded Tools with your own integrations
without changing the Agent pattern.

## Install and configure

Install the exact version:

```bash
python -m pip install "moduagent==0.6.2"
```

Or, from this source checkout, use the current working tree:

```bash
python -m pip install -e .
```

Set your OpenAI-compatible vLLM connection through environment variables. Do
not put credentials in an example file or commit them to source control.

```bash
export VLLM_BASE_URL="<your vLLM base URL>"
export VLLM_MODEL="<your tool-capable model>"
export VLLM_API_KEY="<optional token>"
```

Runnable examples attach `ConsoleEventSink` and render content-free Agent,
model, and Tool progress on stderr. Pass a custom sink through each builder's
`event_sink=` argument to select Korean labels, detailed output, JSON, or no
console rendering in a notebook.

## Learning path

| Order | Scenario | Tools | Main lesson | Safety boundary |
| --- | --- | ---: | --- | --- |
| 10 | [Incident investigation](10_incident_investigation.py) | 5 | Correlate parallel evidence and validate a nested report | Read-only investigation; no mitigation, rollback, or deployment |
| 11 | [Customer case resolution](11_customer_case_resolution.py) | 5 | Pass verified values through sequential lookups and calculations | Advisory proposal only; no refund, return, message, or case update |
| 12 | [Release readiness](12_release_readiness.py) | 5 | Apply several evidence gates and enforce a consistent decision | Ship/hold recommendation only; no deployment or system change |
| 13 | [Document Q&A and report](13_document_qa_and_report.py) | 3 | Route the request, then preserve source locations while answering or composing a sectioned Markdown report | Approved local files and read-only evidence Tools only; no arbitrary paths, URLs, or file overwrite |

All four keep model activity bounded, and their Agent Tools remain read-only
or advisory. Examples `10`–`12` use Standard execution, structured Pydantic
output, summary-only Tool traces, and `RunLimits`; example `13` adds bounded
request routing, document conversion, citation-preserving multi-step Markdown
assembly, and an optional application-owned output artifact.

## 10 — Incident investigation

Run:

```bash
python examples/10_incident_investigation.py
```

The Agent first reads the incident, then gathers metrics, deployments, logs,
and dependency health. The last four reads are independent, so the example
enables bounded parallel Tool execution. `IncidentReport` demonstrates nested
models, field validation, and a cross-field rule that prevents a mitigated
incident from being reported as resolved. The final `runbook_actions` object
contains application-owned action codes rather than generated operational
commands; the Agent recommends them but cannot execute them.

Debugging points:

- `run usage` shows duration, model turns, and total Tool calls.
- `tool trace` and `observed calls` should cover all five evidence sources.
- On failure, `error_summary` and the in-memory diagnostic records identify the
  failed component without logging provider bodies or credentials.
- A missing evidence source, incorrect time window, or invalid report field is
  a data-flow problem; inspect the call log before increasing limits.
- All four required `runbook_actions` keys are named explicitly in the
  instruction. This avoids a guided-decoding whitespace loop seen when the
  model tried to close the required object early; a larger `max_tokens` alone
  does not fix that failure mode.

## 11 — Customer case resolution

Run:

```bash
python examples/11_customer_case_resolution.py
```

The Agent follows a dependent chain: case → order → policy → eligibility →
refund quote. Typed arguments make invalid categories and currencies fail at
the Tool boundary. The final schema requires human approval and makes
`write_action_performed=true` impossible, keeping the workflow advisory.

Debugging points:

- The printed `tools` list should show the five calls in dependency order.
- `tool trace` is summary-only; use the local `CALL_LOG` while developing when
  you need to verify selected non-secret arguments.
- If a lookup returns `not_found`, the Agent should choose `manual_review`
  instead of inventing the missing value.
- If calculation validation fails, compare each argument with the immediately
  preceding Tool result rather than adding a broad model retry.

## 12 — Release readiness

Run:

```bash
python examples/12_release_readiness.py
```

The Agent carries the exact commit and change-set identifiers from the release
manifest into CI, security, risk, and capacity checks. `ReleaseDecision` uses a
model validator to require all five evidence types and to keep `ship`/`hold`
decisions consistent with their blocking reasons. The bundled data contains a
blocking security finding, so the evidence-based result should be `hold`.

Debugging points:

- `checks` and `tool trace` should contain all five Tool names.
- A `ship` result for the bundled scenario indicates that evidence was skipped
  or the security result was not applied; inspect `CALL_LOG` and the final
  validation error.
- Verify that the same manifest commit reaches both CI and security Tools and
  that the same change-set ID reaches the risk Tool.
- Keep deploy or approval mutations in a separately authorized application
  workflow; do not add them to this evaluation Agent while debugging.

## 13 — Document Q&A and sourced report

This example accepts file paths, uploads those files to a separately deployed
Docling Serve container, and builds a bounded in-memory evidence corpus. Set an
application-owned root directory before running it:

```bash
export DOCUMENT_ROOT="/srv/approved-documents"
export DOCLING_SERVE_URL="http://localhost:5001"
# export DOCLING_SERVE_API_KEY="<optional Docling Serve key>"
# export DOCLING_SERVE_DO_OCR="true"  # Default: false
```

Ask a question:

```bash
python examples/13_document_qa_and_report.py \
  --file /srv/approved-documents/policy.pdf \
  --file /srv/approved-documents/appendix.docx \
  --prompt "What changed, and what evidence supports the answer?"
```

Create a Markdown report atomically as a new output file:

```bash
python examples/13_document_qa_and_report.py \
  --file /srv/approved-documents/policy.pdf \
  --file /srv/approved-documents/appendix.docx \
  --prompt "Analyze the operational impact and propose next steps." \
  --output ./policy-impact.md
```

`--mode` is optional and defaults to `auto`. The bounded, tool-free intent
Agent sees only the validated, untrusted request text—not file paths, parsed
documents, corpus evidence, or retrieval Tools—and returns a structured
`RequestIntent`. A standalone report, proposal, or review—and a compound
current-state analysis plus improvements or recommendations—routes to
`report`. Direct questions, explanations, summaries, extraction, comparison,
and simple single-fact analysis route to `question`. Ambiguous intent safely
defaults to `question`.

Use `--mode question` or `--mode report` as a hard override when the host
application already knows the workflow. This skips the classifier model call;
`--mode auto` explicitly selects the default behavior. `--output` controls only
artifact storage and does not select report mode.

Automatic routing adds one bounded model call. An invalid structured
`RequestIntent` fails instead of silently guessing a route, so use an explicit
override when route certainty or the lowest latency matters.

Application code can use `run_document_request()` for the same automatic or
overridden routing. The lower-level `classify_intent()`, `run_question()`, and
`run_report()` functions remain available when the host owns those stages.

`DOCLING_SERVE_URL` defaults to `http://localhost:5001`, and `DOCUMENT_ROOT`
defaults to the current working directory. Treat both as trusted deployment
configuration rather than end-user input. `DOCLING_SERVE_TIMEOUT` bounds each
HTTP request. `DOCLING_SERVE_MAX_WAIT_SECONDS` is one overall deadline covering
task submission, polling, retries, and result retrieval. Only a final Docling
result with `status=success` is accepted; `partial_success` and failures are
rejected. OCR is disabled by default; enable `DOCLING_SERVE_DO_OCR` only when
the source needs it and the Docling deployment has suitable OCR resources.

The client submits each approved file as multipart data to Docling Serve's
asynchronous file-conversion API, polls the bounded task, then fetches Markdown
and lossless JSON results. In other words, the Python HTTP I/O and the Docling
job are asynchronous; this is not a path or URL for Docling to fetch on the
caller's behalf. An optional key is sent in Docling Serve's `X-Api-Key`
header. See the
[Docling Serve REST API](https://docling-project.github.io/docling/usage/api_server/rest_api/)
for the server contract.

The Agent receives three read-only Tools: `list_documents`, `search_evidence`,
and `read_evidence`. Question mode returns a cited Markdown answer. Report mode
first creates a bounded detailed outline, writes each section against the
evidence corpus, validates its citations, and then assembles the sections with
deterministic application code. It never asks one model response to invent and
merge the entire report.

When no relevant evidence exists, `QuestionAnswer` may explicitly abstain with
`status=insufficient_evidence`, empty citations, and concrete limitations
instead of fabricating an answer or citation.

Every accepted citation gets a Markdown source-location footnote and an
adjacent, application-owned exact source excerpt. Page and bounding-box values
come from Docling provenance. For
UTF-8 text-like sources, the example reports a matched original Line No; for
other formats it reports a matched Docling-Markdown line when available and
labels that basis explicitly. Without those coordinates it writes
`페이지 확인 불가` and retains structural location such as a Docling item
reference and heading path instead of inventing a position. The model may
select only opaque citation IDs; quotation text and locations always come from
the immutable application corpus. Unknown IDs fail before rendering instead of
silently becoming evidence.

Table evidence is labeled `Docling 표 셀 직렬화(연속 원문 아님)` so serialized
cell values are not misrepresented as one contiguous passage from the source.

Safety and debugging boundaries:

- The application resolves canonical, unique regular files under
  `DOCUMENT_ROOT` before any upload. Symlinks, non-regular files, unsupported
  extensions, and paths outside that root are rejected.
- Limits are checked before network I/O: at most 10 files, 50 MiB per file, and
  200 MiB in total. Docling request and result sizes, polling time, corpus size,
  Tool results, outline size, section count, model turns, and output are also
  bounded in the example.
- Files and converted text are untrusted evidence, not Agent instructions.
  The model cannot choose a path or conversion URL, and no document URL is
  accepted. Keep the configured Docling endpoint behind an application network
  allowlist to preserve this SSRF boundary.
- Source contents and selected evidence are sent to the configured model. Apply
  access control and redaction before the run; citation validation is not a
  confidentiality control.
- Markdown is always printed to stdout. With `--output`, the CLI additionally
  writes the same Markdown and reports the created path on stderr; the atomic
  writer rejects an existing or symlink destination and never overwrites it.
- The run-local default is bounded deterministic lexical retrieval. For a
  large or repeatedly used corpus, inject a Vector Store implementation through
  the `EvidenceRetriever` adapter. Keep opaque evidence IDs and revalidate every
  returned ID against the immutable run corpus before exposing evidence.

## Live assertions

The normal test suite imports and tests all four examples without network
access. The current opt-in vLLM suite covers examples `10`–`12`, not example
`13`. To run those three against a configured vLLM endpoint:

```bash
MODUAGENT_RUN_LIVE_INTERMEDIATE=1 \
python -m pytest -q tests/integration/test_live_intermediate_scenarios_v051a.py
```

The live suite verifies structured values, exact Tool sets and dependency
order, duplicate-call prevention, successful Tool traces, model-turn budgets,
and Tool-call budgets. It never contains or prints connection credentials.

## Adapting the examples

Replace the in-memory datasets with your own read-only adapters first. Preserve
typed Tool arguments, result-size and timeout bounds, structured output
validation, and the explicit no-write boundary. Add write-capable Tools only
after introducing application authorization, idempotency, audit storage, and a
human approval step.
