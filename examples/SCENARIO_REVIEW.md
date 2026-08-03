# 0.5.1a1 scenario review

This review starts with small programs and adds one framework concept at a
time. It is both a test checklist and a record of the issues that shaped
ModuAgent 0.5.1a1.

## Scenarios

| Example | Real use case | What it verifies |
|---|---|---|
| `01_hello_agent.py` | Small-team assistant | Minimal setup, text output, clean client shutdown |
| `02_use_a_tool.py` | Order status lookup | Typed Tool schema, exactly one lookup, model-visible result |
| `03_structured_output.py` | Support-ticket triage | Pydantic schema generation and validated return type |
| `04_report_automation.py` | Sales report and chart | Two dependent Tools, exact data transfer, artifact creation, structured final output |
| `05_debug_a_run.py` | Service-status investigation | Safe event timeline, model/Tool counts, Tool trace, terminal error summary, protected diagnostics |

The deterministic tests in `tests/test_beginner_examples_v051.py` import every
example without network access and verify the Tools and artifacts directly.
The opt-in tests in
`tests/integration/test_live_agent_scenarios_v051.py` assert actual Tool
arguments, Tool order, bounded call counts, and structured output against a
live OpenAI-compatible vLLM endpoint.

The intermediate path adds `10_incident_investigation.py`,
`11_customer_case_resolution.py`, and `12_release_readiness.py`. Their
deterministic tests cover fifteen application-owned Tools, while
`tests/integration/test_live_intermediate_scenarios_v051a.py` runs the three
complete workflows behind a separate opt-in environment gate.

## Live validation snapshot

The following is one serial validation run on 2026-07-30 using a
RunPod-hosted OpenAI-compatible vLLM endpoint and
`google/gemma-4-26B-A4B-it`. Times are end-to-end wall-clock observations from
one run, not a benchmark or service-level objective.

| Scenario | Result | Wall time | Expected model turns | Tool calls |
|---|---:|---:|---:|---:|
| Basic assistant | Passed | 4.14 s | 1 | 0 |
| Order lookup | Passed | 4.40 s | 2 | 1 |
| Structured ticket triage | Passed | 5.02 s | 1 | 0 |
| Report automation | Passed | 12.44 s | 4 | 2 |
| Debug timeline | Passed | 6.26 s | 2 | 1 |

The separate live assertions for the Tool and report scenarios both passed in
10.54 seconds total in the final run. The report produced the verified total
`8200.0` and a
real SVG artifact.

The report needs four model turns: one for each of its two Tool calls, one to
finish the ACT loop after the chart result, and one Tool-free structured
FINALIZE call. This staging avoids sending Tool schemas and the final output
schema together to vLLM, but its extra turn should be included in latency and
capacity estimates.

The repository microbenchmark was also run on Python 3.10.12 with a local
provider stub. Its median end-to-end runtime was 4.93 ms with no checkpoint and
a no-op sink, 8.28 ms with the composite observability path, 16.01 ms with
checkpointing and a no-op sink, and 21.36 ms with both checkpointing and
composite observability. These figures are machine-specific, but the
millisecond framework measurements versus the multi-second live calls show
that provider latency and model-turn count dominated these small scenarios.

The same report shape was also probed with strict Plan-and-Execute. In that
single model-specific run it used more calls and tokens, then failed because
the model translated labels instead of copying the Tool result exactly.
Standard execution completed the equivalent workflow. This is why the
beginner report uses Standard execution; strict Plan remains an advanced
choice for applications that need independently validated and resumable
steps, and it should be evaluated against the selected model.

## Intermediate live validation snapshot

One serial run on 2026-08-03 used the same model family on a newly assigned
RunPod endpoint. These are `result.run_usage` observations, not benchmarks or
service-level objectives.

| Scenario | Result | Duration | Model turns | Tool calls |
|---|---:|---:|---:|---:|
| Parallel incident investigation | Passed | 12.84 s | 4 | 5 |
| Customer case resolution | Passed | 12.32 s | 7 | 5 |
| Release readiness | Passed | 12.33 s | 7 | 5 |

The final opt-in integration suite passed all three scenarios in 35.99 seconds.

The incident Agent first called `get_incident`, then issued four independent
evidence reads together. The customer and release Agents intentionally made
five dependent calls in order, so each required five Tool turns, one ACT
completion turn, and one structured FINALIZE turn.

The first incident result exposed two semantic-quality problems: it upgraded a
`mitigated` state to `resolved` in free text and emitted an identifier-like
placeholder as an action. Fixed, named evidence/timeline fields, a cross-field
state validator, and application-owned runbook action codes removed both.

During that correction, the model also tried to close a required nested object
after its first value. vLLM guided decoding rejected the close but allowed
whitespace, so the model emitted whitespace until `length`; increasing
`max_tokens` only prolonged the call. Naming all four required keys and asking
for compact JSON eliminated the loop. The final configuration keeps an
8,192-token ceiling but normally stopped after 357 finalization tokens. This
distinguishes a generation-progress failure inside one HTTP request from the
cross-request repetition handled by `max_model_turns` and the no-progress
circuit breaker.

## Problems found and 0.5.1a1 response

| Area | Finding | 0.5.1a1 response |
|---|---|---|
| First run | A source checkout cannot run `python examples/...` until the local package is installed | The example guide now shows `python -m pip install -e .` |
| Usability | Common stores and observability sinks required the verbose constructor | `Agent.create()` now accepts conversation, event, diagnostic, and Tool-trace options |
| Inspection | Developers had to navigate raw metadata for common run facts | `AgentResult` now exposes immutable `run_usage`, `tool_trace`, and `error_summary` projections |
| Error diagnosis | Output-validation classification changed when diagnostics were disabled | Classification is now independent of diagnostic-sink configuration |
| Error location | A missing or failed diagnostic sink could remove the primary component and operation | A safe primary failure summary is recorded before best-effort diagnostic delivery |
| Model failures | A started model request had no paired failure event or duration | `MODEL_FAILED` now records the safe attempt, turn, phase, duration, code, and retry decision |
| Retry logs | Retry records omitted useful correlation and timing fields | Safe code, retryability, model turn, and duration are retained |
| Log volume | Streaming could enqueue one log record per delta | Built-in logging skips delta events by default; content-free counts remain opt-in |
| Secret keys | Acronym spellings such as `APIKey` and `ACCESS_TOKEN` could evade separator-based normalization | Built-in masking now canonicalizes case and separators before matching |
| Speed and resources | More Tool stages directly increased model turns; examples did not visibly bound generation or close client-owned connections | Examples cap output, clients support `async with`/`aclose()`, and results expose elapsed duration and call counts |
| Dependency injection | Valid custom objects that evaluate to `False` could be replaced by defaults | Component resolution now checks `is None` |
| Intermediate orchestration | Realistic examples stopped at two Tools | Three scenarios now coordinate five Tools with parallel or dependent data flow and explicit no-write boundaries |
| Semantic output | A syntactically valid report upgraded incident state and emitted a placeholder action | Named evidence/timeline fields, state validation, and application-owned runbook codes constrain the result |
| Structured generation | Guided decoding emitted whitespace after trying to close a required object early | The prompt names every required key and requests compact JSON; live tests guard the completed shape and call budgets |
| Truncation diagnosis | A failed FINALIZE exposed only generic `model_protocol_error` even though the provider returned `length` | `model_output_incomplete` now preserves only the allowlisted `provider_finish_reason` in public and protected diagnostics |

## Logging safety

The debug scenario showed the complete model/Tool lifecycle without including
the prompt, message content, Tool arguments, Tool result, API token, or raw
provider body. Enabling `tool_trace_mode="arguments"` can expose ordinary
business inputs even though known secret-key names are masked, so it should
only be used in an access-controlled development environment.

For latency work, begin with `dict(result.run_usage)`. A Tool workflow normally
needs another model turn after each Tool result, so model-server latency and
the number of turns dominate these small examples. Use strict Plan only when
its validation and recovery guarantees justify those additional calls.

## Next scenarios

The next useful additions are conversation-memory compaction, durable
checkpoint recovery, safe write authorization/idempotency, streaming
cancellation, and concurrent-session load. They are intentionally not placed
in the first learning path so a new user can understand the core loop before
adding production components.
