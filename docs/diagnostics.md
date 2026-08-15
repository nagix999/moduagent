# Diagnostics

ModuAgent 0.5 separates execution events from failure diagnostics.

- An `EventSink` receives safe lifecycle events such as model attempts, Tool
  calls, Plan steps, retries, and terminal results.
- A `DiagnosticSink` receives a protected, sanitized `FailureDiagnostic` when
  the runtime captures an exception.

Use an `EventSink` for normal operational timelines. Use a `DiagnosticSink`
only in an access-controlled diagnostic path. Diagnostic records are never
sent to the model or added to conversation history.

## Pretty console progress

For a terminal or Jupyter notebook, progress logging is one Agent option:

```python
from moduagent import ConsoleEventSink

agent = Agent.create(
    name="report-agent",
    model=model,
    instructions="Use the available Tools and answer the request.",
    tools=tools,
    event_sink=ConsoleEventSink(
        language="ko",       # "en" is the default
        detail="detailed",  # or "summary"
    ),
)
```

`ConsoleEventSink` renders run, model, Tool, retry, delegation, planning, and
finalization events as an indented timeline. It uses a sealed, content-free
projection: prompts, model delta text, Tool arguments/results, and private model
reasoning are never printed. ANSI color is enabled automatically only for a TTY.
Use `output_format="json"` to write the same safe event projection as JSON.

## In-memory diagnostics

Pass a diagnostic sink when constructing the Agent:

```python
from moduagent import Agent, InMemoryDiagnosticSink, LoggingEventSink

diagnostics = InMemoryDiagnosticSink(max_records=1_000)

agent = Agent(
    config=config,
    model=model,
    tools=tools,
    event_sink=LoggingEventSink(),
    diagnostic_sink=diagnostics,
    diagnostic_timeout_seconds=0.25,
    diagnostic_max_pending_deliveries=1_024,
)

result = await agent.run(
    "Prepare the report.",
    session_id="report-session",
)

if result.failure_id is not None:
    failure = diagnostics.get(result.failure_id)
    if failure is not None:
        print(failure.to_dict())

for failure in diagnostics.for_run(result.run_id):
    print(failure.failure_id, failure.component, failure.operation)
```

`result.failure_id` is the root failure correlation ID for a failed run when
diagnostics were enabled and the runtime captured a corresponding exception.
Recovered Tool failures can also appear in `for_run()` even when the run
succeeds.

The correlated root record can have `terminal=False`. `terminal` describes
recoverability at capture time, not the final run outcome. For example, the
Tool runtime captures a failed invocation before the Plan policy decides
whether to repair, replan, or stop. If the policy later stops, the terminal
`AgentResult` reuses that Tool record's `failure_id` instead of duplicating the
exception.

`InMemoryDiagnosticSink` is bounded and evicts its oldest record when
`max_records` is reached. It is intended for development and single-process
tests; all records are lost when the process exits.

## Logging and fan-out

`LoggingDiagnosticSink` writes one structured JSON record per failure:

```python
import logging

from moduagent import (
    CompositeDiagnosticSink,
    InMemoryDiagnosticSink,
    LoggingDiagnosticSink,
)

recent = InMemoryDiagnosticSink(max_records=500)
diagnostic_sink = CompositeDiagnosticSink(
    [
        LoggingDiagnosticSink(logging.getLogger("company.agent.diagnostics")),
        recent,
    ]
)

agent = Agent(
    config=config,
    model=model,
    tools=tools,
    diagnostic_sink=diagnostic_sink,
)
```

`CompositeDiagnosticSink` invokes every child independently. Inspect its
`last_errors` when one child may have failed while another still accepted the
record.

Standard-library logging is run in a bounded daemon worker pool so a blocking
handler does not block the Agent event loop or interpreter shutdown. Python
cannot force-cancel a synchronous handler that is already running. Timeout
cancels its awaiter and abandons the result; the in-flight handler may continue
in a daemon worker until it returns.

## Collected details

`FailureDiagnostic` schema version 1 contains correlation fields, stable error
classification, exception and cause type names, and bounded stack frames. A
frame contains only the filename basename, function name, and line number. If
a deep traceback is truncated, the reporter retains the most recent,
innermost frames.

The reporter allowlists a small set of structured exception attributes:

```text
PostgreSQL error: {"sqlstate": "42601"}
HTTP error:       {"http_status": 503}
OS error:         {"errno": 111}
Pydantic error:   {"validation_errors": [{"type": "missing", "loc": ["name"]}]}
```

The reporter reads only eager, allowlisted SQLSTATE and HTTP status attributes,
plus `errno` through the trusted built-in `OSError` descriptor. It does not
invoke unknown exception properties. Pydantic input values, messages, and
contexts are omitted. Locations expose declared output-schema field names, but
replace unrecognized string keys with `[DYNAMIC_KEY]` and integer locations
with `[INDEX_OR_KEY]`.

By default ModuAgent does not collect raw exception messages, SQL text,
prompts, model request or response bodies, Tool arguments or results, source
lines, or traceback locals. Diagnostic records are bounded and sanitized, but
they can still reveal internal component, operation, field, and function
names. Apply access control, retention, and encryption appropriate to that
metadata.

## Built-in event and audit logs

`LoggingEventSink` and `AuditEventSink` project each event through an explicit
allowlist. They do not serialize arbitrary event data. Model content and
deltas, Tool arguments and results, provider payloads, unknown fields, and
free-form failure or validation reasons are omitted. Where a framework-owned,
code-like reason is supported, only its bounded stable code is retained.

Framework-managed model attempts expose a content-free timeline:

- `MODEL_STARTED` records `attempt`, `model_turn`, `phase`, `message_count`,
  `tool_count`, `has_output_schema`, and whether the actual request streams.
- `MODEL_COMPLETED` records duration, usage, finish reason, and Tool-call count.
- `MODEL_FAILED` records duration, exception type, stable code, effective
  retryability, and whether the attempt is terminal.
- `RETRY` keeps the failed attempt's code, retryability, model turn, and
  duration so it can be correlated without a provider body.

These events never contain a prompt, message content, Tool arguments, or raw
provider request/response body. `LoggingEventSink` skips `MODEL_DELTA`,
`STEP_MODEL_DELTA`, and `FINAL_DELTA` by default to avoid a log entry for every
stream chunk. Use `LoggingEventSink(include_deltas=True)` only when chunk-size
evidence is needed; even then the built-in projection logs character and byte
counts, not delta content. Built-in key masking treats acronym, camel-case,
underscore, hyphen, and case variants of known secret names equivalently.

Step and Tool correlation values such as `step_id`, `call_id`,
`failed_call_id`, and `result_ref` are written as SHA-256 hashes. This keeps
events joinable without disclosing their original identifiers. A terminal
event can include the already-public `AgentResult.error`; do not put secrets in
that field.

## Plan validation codes

Strict Plan-and-Execute exposes finite framework-owned validation fields on
internal `STEP_RETRY` and `STEP_FAILED` events:

- `validation_code`: the immediate failure code.
- `validation_location`: `act`, `step_result`, or `step_validator`.
- `validation_cause_code`: the original code when the public code reports
  exhausted attempts.

A terminal Plan validation failure also projects this classification into
`result.metadata["validation_failure"]` and
`result.metadata["error_summary"]`. Prefer these codes over parsing a
free-form reason. Built-in event logs retain the codes and location but omit
the raw reason.

## Default behavior and delivery

Diagnostics are disabled when `diagnostic_sink` is omitted. Explicitly using
`NoopDiagnosticSink` also produces no stored record or failure ID. Safe
`error_summary` fields do not depend on diagnostic delivery: when
`RuntimeServices` captures a failure, it keeps the primary component,
operation, phase, attempt, category, code, and retryability even when a sink is
absent, returns no ID, or fails. Output codec exceptions are likewise reported
consistently as `output_validation` / `output_validation_failed`; enabling
diagnostics only adds a correlation ID and protected diagnostic record.
An incomplete provider response uses `model_output_incomplete` and exposes only
the allowlisted `provider_finish_reason` (`timeout`, `length`, or `max_tokens`)
in both `error_summary` and diagnostic `safe_details`. Partial response content
and arbitrary provider metadata are never retained.

Two `Agent` options bound delivery:

- `diagnostic_timeout_seconds` defaults to `0.25` and limits how long the
  runtime waits for a record during delivery and terminal flushing.
- `diagnostic_max_pending_deliveries` defaults to `1024` and caps records
  queued or in flight for that Agent.

Diagnostic delivery is best effort. `failure_id` is a correlation identifier,
not proof that a sink durably committed the record. A timed-out, failed, or
over-capacity delivery increments `drop_count`, but cannot fail the Agent run:

```python
reporter = agent.diagnostic_reporter

if reporter is not None:
    print("dropped:", reporter.drop_count)
    print("last sink error:", reporter.last_error)
```

`drop_count` increases when delivery to the configured sink fails or times
out. `last_error` contains the latest sink-delivery error and is cleared after
a later successful delivery. For partial `CompositeDiagnosticSink` failures,
also inspect the composite sink's `last_errors`.

Custom asynchronous sinks must honor cancellation: propagate
`asyncio.CancelledError` and cancel or bound downstream I/O. Suppressing
cancellation can leave the sink's own work running after ModuAgent has
abandoned that delivery.

## Custom Engine boundary

For a custom `ExecutionEngine`, `EngineOutcome.error` is public, trusted text.
It becomes `AgentResult.error`, crosses the terminal event boundary, and may be
written by `LoggingEventSink`. Return only a bounded, user-safe summary and
send private causes through `DiagnosticSink`.

`AgentResult.metadata["error_summary"]` is runtime-owned and reserved. A custom
Engine cannot override it through `EngineOutcome.metadata`; ModuAgent computes
the terminal classification and correlation fields. Use the documented Engine
failure contracts instead of placing private data in runtime-owned metadata.

Neither `InMemoryDiagnosticSink` nor `LoggingDiagnosticSink` provides durable
multi-process storage. Production deployments with multiple workers should
implement a custom sink backed by an access-controlled durable system:

```python
from moduagent import FailureDiagnostic


class DurableDiagnosticSink:
    async def capture(self, record: FailureDiagnostic) -> None:
        await protected_store.put(
            key=record.failure_id,
            value=record.to_dict(),
        )
```

The durable implementation should define idempotency, encryption, tenant
isolation, retention, deletion, backpressure, and monitoring. It must accept
only `FailureDiagnostic`; the framework never passes a `BaseException` or raw
traceback to a diagnostic sink.
