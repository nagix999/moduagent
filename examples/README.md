# ModuAgent examples

[한국어 번역: 중급 예제](INTERMEDIATE.ko.md) ·
[한국어 번역: 프로덕션 제어](PRODUCTION.ko.md)

These examples grow one concept at a time. Start with `01` and run them in
order; each file is standalone.

## Before you start

These examples target ModuAgent `0.6.0`. Install the exact version:

```bash
python -m pip install "moduagent==0.6.0"
```

Alternatively, when running from a source checkout, install that working tree
in editable mode so `python examples/...` imports the local package:

```bash
python -m pip install -e .
```

Then set the connection details for your OpenAI-compatible vLLM server:

```bash
export VLLM_BASE_URL="<your OpenAI-compatible vLLM base URL>"
export VLLM_MODEL="<your tool-capable model name>"
export VLLM_API_KEY="<your optional token>"  # Omit when no token is required.
```

The examples read these values with `VLLMClient.from_env()`. They do not store
credentials in source code.

## Learning path

1. [`01_hello_agent.py`](01_hello_agent.py) — ask a model one question.
2. [`02_use_a_tool.py`](02_use_a_tool.py) — let the model look up an order.
3. [`03_structured_output.py`](03_structured_output.py) — receive a validated
   Pydantic object instead of free-form text.
4. [`04_report_automation.py`](04_report_automation.py) — query small in-memory
   sales data, create a real SVG chart, and return a structured report.
5. [`05_debug_a_run.py`](05_debug_a_run.py) — inspect safe event logs, token
   usage, Tool calls, and failure diagnostics.

The examples cap model output and close their HTTP client with an async context
manager. They use Standard execution because it is the smaller and faster
default for direct Tool workflows.

Run any example from the repository root:

```bash
python examples/01_hello_agent.py
```

## Domain capstone

After examples `01`–`05`, [`06_waf_log_analysis.py`](06_waf_log_analysis.py)
combines six read-only evidence Tools, scoped authorization, bounded decoding,
and structured output to classify one WAF event. It is a domain capstone, not
the next one-concept beginner step.

The WAF example is intentionally advisory. It scopes a run to one event and
uses these six read-only Tools: `analyze_payload_encoding`,
`get_waf_rule_context`, `get_route_context`,
`get_correlated_app_outcome`, `summarize_related_events`, and
`lookup_threat_intel`. The bundled providers, including threat intelligence,
use deterministic synthetic fixtures and make no network calls. No API key or
live reputation service is required; `no_data` or `unknown` must not be treated
as proof that an address is benign. Synthetic Tool results contain `:fixture:`
in `evidence_ref` so the model can distinguish them from derived log evidence.
Tool calls are sequential by default (`parallel_tool_calls=False`). Enable
parallel calls only after validating the live provider and the model server's
Tool parser with the same six-Tool scenario.

Run it from the repository root:

```bash
python examples/06_waf_log_analysis.py
```

`event_id` is an application-owned opaque scope key, separate from the WAF log
schema. The example binds it into zero-argument Tool closures and checks it
against the trusted `authorized_event_id` run context. It is not authentication
or tenant isolation by itself: a production application must authorize the
event and derive tenant/user scope outside the model before starting the run.
When `event_id` is omitted, the standalone helper derives a deterministic local
scope ID from the validated log; production integrations should supply their
own authorized event ID instead.

Replace `SAMPLE_WAF_LOG`, `SAMPLE_EVENT_ID`, and the synthetic providers with
bounded async adapters when integrating the example. `date` must be an ISO 8601
timestamp with a timezone, and `payload` is limited to 8,192 characters.
Schema validation is not redaction. Remove or mask credentials, cookies,
tokens, personal data, and other sensitive payload values before calling
`analyze_waf_log()`; the remaining fields and Tool results are sent to the
configured model endpoint and must be treated as untrusted evidence.
IP/country and threat-intelligence data are supporting signals, not verdicts.
This example cannot change WAF policy, disable rules, allowlist/block addresses,
replay payloads, or make arbitrary network requests.

`analyze_payload_encoding` performs only bounded local URL-percent and Base64
transforms (at most two layers); it never fetches, decompresses, or executes
content. Decoded preview text is hidden by default
(`include_decoded_previews=False`) while hashes and feature codes remain. Set
the option to `True` only for synthetic data or after decode-aware redaction and
an explicit trust-boundary review, because encoded secrets can appear in the
preview. Hiding previews is not redaction: the raw encoded payload is still
sent to the model, so sensitive encoded values must be removed before the run.

Generated files from example `04` are written to `examples/artifacts/` by
default. Set `MODUAGENT_ARTIFACT_DIR` to use another directory.

To run the opt-in live assertions for Tool order, arguments, call budgets, and
structured output:

```bash
MODUAGENT_RUN_LIVE_SCENARIOS=1 \
python -m pytest -q tests/integration/test_live_agent_scenarios_v051.py
```

See [`SCENARIO_REVIEW.md`](SCENARIO_REVIEW.md) for the scenario checklist,
one live vLLM validation run, the issues found, and the 0.5.1a1 changes made from
that review.

## Intermediate learning path

After the beginner path, continue in this order:

1. [`10_incident_investigation.py`](10_incident_investigation.py) — correlate
   five read-only evidence sources, including bounded parallel Tool calls, and
   diagnose failures without exposing provider data.
2. [`11_customer_case_resolution.py`](11_customer_case_resolution.py) — carry
   verified values through five dependent lookup/calculation Tools while a
   structured contract prevents claims that a refund was executed.
3. [`12_release_readiness.py`](12_release_readiness.py) — combine five release
   gates into a validated ship/hold recommendation without deploying anything.
4. [`13_document_qa_and_report.py`](13_document_qa_and_report.py) — parse
   application-approved files with Docling Serve, answer questions from cited
   evidence, or assemble a sourced Markdown report section by section.

Run them from the repository root, for example:

```bash
python examples/10_incident_investigation.py
```

Example `13` additionally needs an application-approved document root and a
running Docling Serve endpoint. It uploads bytes with the asynchronous
multipart API; Docling is never given an arbitrary local path or source URL.

```bash
export DOCUMENT_ROOT="/srv/approved-documents"
export DOCLING_SERVE_URL="http://localhost:5001"
# export DOCLING_SERVE_API_KEY="<optional key>"
# export DOCLING_SERVE_DO_OCR="true"  # Default: false

python examples/13_document_qa_and_report.py \
  --file /srv/approved-documents/policy.pdf \
  --prompt "Summarize the approval policy and cite the source."
```

The default `--mode auto` uses a bounded, tool-free structured classifier over
the request text only. It routes direct questions, summaries, extraction,
comparison, and simple single-fact analysis to Q&A. An explicit standalone
report, proposal, or review—and a compound current-state analysis plus
improvements or recommendations—goes to the outline-and-sections report
workflow. Ambiguous requests default to Q&A. Use `--mode question` or
`--mode report` as an optional hard override when the application already knows
the desired workflow; an override skips the classifier.

Report mode plans a detailed outline, drafts its sections, validates exact
source quotations and locations, and deterministically assembles a Markdown
report. Markdown is always printed to stdout; `--output` additionally creates
the same content atomically without overwriting an existing or symlink
destination, and the save status goes to stderr. See the
[`13` guide](INTERMEDIATE.md) for the report command, page/bounding-box/line
and structural citation fallback, limits, and trust boundaries.

All four keep model activity bounded and use read-only or advisory safety
boundaries. Examples `10`–`12` use Standard execution, structured output, and
summary Tool traces; example `13` also demonstrates a separately bounded
request router, document-conversion boundary, and citation-preserving report
assembly. See
[`INTERMEDIATE.md`](INTERMEDIATE.md) for each example's Tool count, teaching
goal, safety boundary, execution command, and debugging checklist.

Example `13` is covered by the offline test suite, but it is not part of the
current vLLM live scenario suite. To run the opt-in live assertions for
examples `10`–`12`:

```bash
MODUAGENT_RUN_LIVE_INTERMEDIATE=1 \
python -m pytest -q tests/integration/test_live_intermediate_scenarios_v051a.py
```

## Advanced examples

Once the five small examples feel familiar, continue with:

- [`report_automation_agent.py`](report_automation_agent.py) for a larger
  report workflow with persistence, recovery, and SQLite/PostgreSQL support.
- [`gemma4_postgresql_report_agent.py`](gemma4_postgresql_report_agent.py) for
  a PostgreSQL-focused Gemma deployment example.

Those examples intentionally include production concerns that are omitted from
the beginner path.

## Production controls

After the read-only and advisory examples, use
[`20_production_controls.py`](20_production_controls.py) to learn how one
write-capable Tool is constrained by trusted `user_context`, deny-by-default
RBAC, an application-owned idempotency key/store, checkpoints, bounded
conversation storage, and protected diagnostics. The code remains a generic
framework example; it does not add a change-management Recipe to ModuAgent.

See [`PRODUCTION.md`](PRODUCTION.md) for long-conversation compaction, durable
resume, clean streaming cancellation, concurrent sessions, and the boundary
between in-memory demonstrations and production infrastructure.
