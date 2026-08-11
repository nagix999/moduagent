# ModuAgent

[English](https://github.com/nagix999/moduagent/blob/main/README.md) |
[한국어](https://github.com/nagix999/moduagent/blob/main/README.ko.md)

ModuAgent is a composable Python runtime for building AI agents around your
own model endpoints and Python functions.

Start with a normal model or Tool-calling loop. Add bounded Context Memory,
validated Pydantic output, strict Plan-and-Execute, checkpoint recovery,
Skills, and observability only when your application needs them.

> Current version: **0.6.0** · Status: **Alpha** · Python **3.10+** · **MIT License**

New to ModuAgent? Follow the five short steps below. They use the stable Quick
API; the explicit component API remains available for advanced composition.
For runnable files that add only one concept at a time, start with the
[beginner examples](https://github.com/nagix999/moduagent/blob/main/examples/README.md).
When you are ready for multi-Tool workflows, continue with the
[intermediate examples](https://github.com/nagix999/moduagent/blob/main/examples/INTERMEDIATE.md).

## What ModuAgent provides

An Agent is a composition of small, explicit parts:

```text
User input
    │
    ▼
Agent ──► Execution profile ──► Model
  │              │                │
  │              └──────────────► Tools
  │
  ├── AgentDefinition + RuntimeBindings/Profile
  ├── Conversation store + Context Memory
  ├── Delegation coordinator + shared budget
  ├── Output codec
  ├── Checkpoint store
  ├── Skills and authorization
  └── Events, diagnostics, and metrics
```

- `Agent.create()` resolves the common configuration for you.
- `AgentConfig` exposes instructions, retry behavior, and run limits when you
  need explicit composition.
- A model client connects to vLLM, Ollama, or another supported endpoint.
- Tools are typed Python functions the model may call.
- An execution profile controls how work proceeds.
- An output codec returns text or a validated Pydantic object.
- A conversation store saves history; a memory policy selects the model view.
- A checkpoint store saves interrupted runs for safe recovery.
- An `AgentDefinition` pins deployable semantics while `RuntimeBindings` holds
  replaceable endpoints, stores, credentials, and telemetry.
- A delegation coordinator applies lineage, topology, deadline, aggregate
  budget, session namespace, and receipt guards before a child model runs.

Most applications should start with a model and a small set of Tools. Add the
other components as requirements appear.

### Choose an execution mode

| | Standard execution | Strict Plan-and-Execute |
|---|---|---|
| Selection | Default | Explicit opt-in |
| Best for | Chat, direct Tool use, short loops | Dependent, auditable multi-step work |
| Flow | Model → optional Tools → answer | Plan → act → validate/commit → answer |
| Cost | Lower latency and fewer calls | More calls for stronger control |
| Intermediate state | Lightweight | Versioned, validated step state |

Use Standard execution unless intermediate steps must be independently
validated or safely resumed.

## Installation

ModuAgent requires Python 3.10 or later. You also need a reachable model
server; ModuAgent does not host a model itself.

Install the package:

```bash
python -m pip install "moduagent==0.6.0"
```

If your package index does not contain `0.6.0` yet and you already have a 0.6
source checkout, install it from the repository root:

```bash
cd /path/to/moduagent
python -m pip install -e .
```

Contributors can include the development tools:

```bash
python -m pip install -e '.[dev]'
```

Optional integrations are installed separately:

```bash
python -m pip install redis       # Redis conversation/checkpoint stores
python -m pip install matplotlib  # report automation example
python -m pip install "psycopg[binary]>=3.2,<4"  # PostgreSQL report example
```

## Step 1: run your first Agent

The example below connects to a vLLM OpenAI-compatible endpoint and runs a
model-only Agent.

```python
import asyncio

from moduagent import Agent, VLLMClient


async def main() -> None:
    async with VLLMClient.from_env(
        default_options={"temperature": 0, "max_tokens": 256},
    ) as model:
        agent = Agent.create(
            model=model,
            instructions="Answer accurately and concisely.",
        )

        answer = await agent.ask(
            "Explain what an AI agent is in one paragraph.",
            session_id="getting-started",
        )
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())
```

Set the endpoint before running the file:

```bash
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL="your-model-name"
# export VLLM_API_KEY="your-token-if-required"
export VLLM_TIMEOUT="60"
python getting_started.py
```

`VLLMClient.from_env()` reads only these documented variables:

| Variable | Required | Meaning |
|---|---|---|
| `VLLM_MODEL` | Yes | Model name served by vLLM |
| `VLLM_BASE_URL` | No | OpenAI-compatible base URL; defaults to `http://localhost:8000/v1` |
| `VLLM_API_KEY` | No | Bearer token |
| `VLLM_TIMEOUT` | No | Positive request timeout in seconds; defaults to `60` |

Use the regular `VLLMClient(...)` constructor when configuration comes from
another source. An explicit `timeout=` passed to `from_env()` takes precedence
over `VLLM_TIMEOUT`.

The endpoint and selected model must support the capabilities your Agent uses,
such as Tool Calling or JSON Schema output.
For Tool examples, configure vLLM's chat template and Tool parser for the
selected model.

`ask()` is the shortest path: it returns the decoded output and raises the
secret-safe `AgentRunError` when the run does not complete. Use `run()` in
operational code that needs the full result; Step 5 shows both forms.

Ollama uses the same Agent API:

```python
from moduagent import OllamaClient

model = OllamaClient(
    base_url="http://localhost:11434",
    model="qwen3:14b",
)
```

The OpenAI-compatible and Ollama clients also expose a validated embedding
boundary. Configure the client for the embedding endpoint and call `embed()`
directly; an Agent is not required for deterministic vector generation.

```python
from moduagent import ModelCapabilities, VLLMClient

async with VLLMClient(
    base_url="http://localhost:8001/v1",
    model="BAAI/bge-m3",
    capabilities=ModelCapabilities(
        chat=False,
        streaming=False,
        tool_calling=False,
        parallel_tool_calling=False,
        structured_output=False,
        embeddings=True,
        tool_calling_with_structured_output=False,
    ),
) as embedding_model:
    vectors = await embedding_model.embed(["first document", "second document"])
```

The returned batch must match the input count. Vectors must be non-empty,
finite, and dimensionally consistent; OpenAI-compatible response indices must
be the exact unique range `0..N-1`. Malformed responses raise
`ModelProtocolError` without including document or provider response content.

## Step 2: add a Tool

Use `@tool` to expose a typed Python function.

```python
import asyncio

from moduagent import Agent, VLLMClient, tool


@tool(timeout_seconds=5, max_result_bytes=4096)
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def main() -> None:
    async with VLLMClient.from_env(
        default_options={"temperature": 0, "max_tokens": 256},
    ) as model:
        calculator = Agent.create(
            model=model,
            instructions=(
                "Use the add Tool whenever addition is required. "
                "Do not invent a calculated result."
            ),
            tools=[add],
        )
        answer = await calculator.ask(
            "What is 12 plus 30?",
            session_id="calculator-demo",
        )
        print(answer)


asyncio.run(main())
```

The function's type hints become its input schema, and its docstring becomes
the description shown to the model. Only Tools passed through `tools=[...]`
can be called.

`@tool` is the short name for the existing `@function_tool` adapter. Neither
form guesses whether a Tool is safe to retry or repair. For example,
`idempotent=True` declares that repeating the same validated call is safe; it
does not create a transaction or exactly-once guarantee. Write Tools still
need an application idempotency key and duplicate protection.

Blocking functions such as `pandas.read_sql()` run outside the event loop.
For production, share a bounded scheduler across synchronous Tools so timed-out
calls cannot create an unlimited number of background threads:

```python
from moduagent import SyncToolScheduler, tool

blocking_tools = SyncToolScheduler(max_workers=8, max_queue=32)

@tool(sync_scheduler=blocking_tools, timeout_seconds=10)
def query_db(sql: str) -> list[dict]:
    return run_read_only_query(sql)
```

Raw assistant Tool calls and raw Tool results are internal protocol messages.
They are available to the model during the run but are not added to
`ConversationStore` or `AgentResult.messages`. The default public Tool trace is
a bounded, secret-safe summary.

`AgentTool(child_agent)` is the legacy, in-process way to expose one Agent as a
Tool. A non-successful child run is a Tool failure; it is never returned as a
successful `None` value. Child terminal failures are non-retryable at this
legacy boundary: generic Tool retry counts, changed-argument repair, timeout
retry, and `idempotent=True` alone do not rerun them. A custom Agent-like object
may explicitly raise a pre-classified `ToolFailure`; that declared safe contract
is preserved. This adapter has no root budget, cycle/depth guard, receipt, or
parent/child session namespace. Because it forwards the parent's `session_id`,
composing parent and child with the same
`ConversationStore` object emits a warning: use separate stores for legacy
delegation and do not treat it as a production isolation boundary.

## Step 3: return validated structured output

Pass a Pydantic model class through `output=`. The Quick API creates the
existing `PydanticOutputCodec` internally, and `ask()` returns the validated
model object.

```python
import asyncio

from pydantic import BaseModel, Field

from moduagent import Agent, VLLMClient, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)


async def main() -> None:
    async with VLLMClient.from_env(
        default_options={"temperature": 0, "max_tokens": 256},
    ) as model:
        structured_agent = Agent.create(
            model=model,
            instructions=(
                "Use the add Tool for every arithmetic operation, then return "
                "the answer in the requested format."
            ),
            tools=[add],
            output=Answer,
        )
        answer: Answer = await structured_agent.ask(
            "What is 20 plus 22?",
            session_id="structured-demo",
        )
        print(answer.answer, answer.confidence)


if __name__ == "__main__":
    asyncio.run(main())
```

Tools and structured output can be used together. ModuAgent separates the
requests:

```text
ACT:      model receives Tool schemas, without the final output schema
FINALIZE: model receives the Pydantic schema, without Tools
```

This avoids the common vLLM conflict caused by putting Tool Calling and
structured output in the same request. `VLLMClient` declares that combination
unsupported by default, so the runtime uses this separated mode.

## Step 4: use strict Plan-and-Execute

Use Plan-and-Execute when a task has dependent steps and each intermediate
result must be validated before it can affect the final answer.

```python
import asyncio

from pydantic import BaseModel, Field

from moduagent import Agent, RunLimits, VLLMClient, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)

async def main() -> None:
    async with VLLMClient.from_env(
        default_options={"temperature": 0, "max_tokens": 512},
    ) as model:
        planning_agent = Agent.create(
            model=model,
            instructions=(
                "Use the add Tool for every arithmetic operation. "
                "Complete multi-step requests using only validated and committed "
                "step results."
            ),
            tools=[add],
            output=Answer,
            execution="plan",
            limits=RunLimits(
                max_steps=4,
                max_step_attempts=2,
                max_replans=1,
                max_tool_calls=8,
                timeout_seconds=120,
            ),
        )
        answer: Answer = await planning_agent.ask(
            "Calculate 10 + 20, then add 5 to that verified result.",
            session_id="plan-demo",
        )
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())
```

The strict flow is:

```text
PLAN → ACT_TOOL → STEP_RESULT → VALIDATE/COMMIT → VERIFY → FINALIZE
```

- `max_steps` limits generated Plan steps, not model calls.
- `max_step_attempts` limits validation retries for one step.
- `max_replans` limits revisions of unfinished work.
- `max_tool_calls` limits business Tool calls for the whole run.
- `timeout_seconds` is one deadline shared by planning, model calls, Tools,
  finalization, and persistence.

For `execution="plan"`, the Quick API creates an `LLMPlanGenerator` with the
same `model` and synchronizes its `max_steps` with `limits.max_steps`. Pass an
explicit `PlanExecutionProfile` to `execution=` when you need a custom planner,
validator, or recovery policy.

Standard execution remains the better default for chat, direct Tool use, and
short workflows.

## Step 5: inspect results and bound model calls

`ask()` is equivalent to `run()` followed by `unwrap()`. Use it when the only
successful value you need is the decoded output. The snippets in this step
assume an already configured `planning_agent`; keep its model client's async
context open for the lifetime of these calls:

```python
import asyncio


async def main() -> None:
    answer = await planning_agent.ask("Complete the task.")
    print(answer)


asyncio.run(main())
```

Use `run()` when you also need usage, traces, finish reasons, or recovery
metadata:

```python
import asyncio


async def main() -> None:
    result = await planning_agent.run(
        "Complete the task.",
        session_id="operations-demo",
    )
    print(result.explain())  # concise, sanitized terminal summary
    answer = result.unwrap()  # raises AgentRunError unless completed
    print(answer)


asyncio.run(main())
```

Alternatively, replace the `unwrap()` line with the equivalent explicit form:

```python
result.raise_for_error()
answer = result.output
```

The main `AgentResult` fields are:

| Field | Meaning |
|---|---|
| `output` | Final text or validated object |
| `error` | Safe public error message, or `None` |
| `finish_reason` | Stable terminal reason |
| `usage` | Accumulated model token usage |
| `run_id` | Failure-correlation and checkpoint identifier |
| `messages` | Public conversation messages |
| `metadata` | Bounded Tool trace, Plan summary, and safe error category |
| `run_usage` | Immutable model-turn, Tool-call, and elapsed-time summary |
| `tool_trace` | Immutable bounded projection of executed Tools |
| `error_summary` | Immutable safe terminal failure classification |

Terminal finish reasons are:

| Finish reason | Meaning |
|---|---|
| `completed` | Output completed successfully |
| `max_steps` | Plan or execution step budget was exhausted |
| `max_tool_calls` | Tool-call budget was exhausted |
| `max_model_turns` | Whole-run model-attempt budget was exhausted |
| `no_progress` | Repeated semantic state and response tripped the circuit breaker |
| `timeout` | The overall run deadline expired |
| `cancelled` | The caller cancelled the run |
| `error` | Another terminal failure occurred |

`ask()` and `unwrap()` raise a secret-safe `AgentRunError` for every reason
except `completed`:

```python
import asyncio

from moduagent import AgentRunError


async def main() -> None:
    try:
        answer = await planning_agent.ask("Complete the task.")
        print(answer)
    except AgentRunError as exc:
        print(exc.run_id, exc.finish_reason, exc.code)
        print(exc.retryable, exc.resumable, exc.failure_id)


asyncio.run(main())
```

The exception does not retain prompts, output, Tool arguments, raw provider
bodies, or arbitrary result metadata.

### Strict model retry contract

`RetryConfig.max_attempts` includes the first call and defaults to `1`, so
retries are opt-in:

```python
from moduagent import RetryConfig

retrying_agent = Agent.create(
    model=model,
    instructions="Answer accurately.",
    retry=RetryConfig(max_attempts=2),
)
```

Model calls are retried only for this allowlist:

- timeout failures;
- connection or network failures;
- HTTP `408`;
- HTTP `5xx`.

They are not retried for:

- HTTP `429` or any other HTTP `4xx`;
- malformed JSON, invalid Tool arguments, or any provider protocol/parsing
  failure;
- provider output ending with `timeout`, `length`, or `max_tokens`;
- structured-output validation failures;
- invalid requests, capability mismatches, `TypeError`, or programming errors.

A streaming model call is not retried after a public delta has been emitted.
Tool retry and repair are separate contracts: they additionally require the
Tool's declared safety profile and never become safe merely because model
retry is enabled.

An incomplete provider response fails with code `model_output_incomplete`.
Inspect `result.error_summary["provider_finish_reason"]` or
`AgentRunError.provider_finish_reason` to distinguish `timeout`, `length`, and
`max_tokens`; partial output and provider metadata are not retained.

### Whole-run model guards

Every run has two independent model guards:

```python
limits = RunLimits(
    max_model_turns=32,
    no_progress_model_turn_threshold=3,
)
```

- `max_model_turns=32` bounds framework-managed model attempts across planning,
  acting, memory summarization, Skill selection, repairing, and finalization.
  Transport retries consume this budget too.
- `no_progress_model_turn_threshold=3` stops on the third consecutive
  identical semantic-state/normalized-response observation. A successful Tool
  outcome resets the streak only when its run-salted fingerprint is new;
  repeating the same successful Tool outcome does not bypass the guard.
  Each successfully consumed memory-summary batch and each committed Plan step
  also counts as progress. None of these resets the total turn count.

The resulting finish reasons are `max_model_turns` and `no_progress`; neither
is automatically retried or safely resumable. `error_summary` and
`AgentRunError` therefore report `retryable=False` and `resumable=False` with
only bounded counters and safe classification fields.

Built-in components route auxiliary calls through the run's `ModelGateway`.
Custom memory policies, selectors, planners, and model clients must preserve
that boundary: a custom component that calls a provider directly, or a client
that performs hidden internal retries, cannot be counted separately by the
framework.

With a checkpoint store, every framework-managed model attempt, including a
provider retry, is durably reserved immediately before provider I/O. A hard
crash cannot make that consumed turn available again on resume. The model
guard checkpoint stores only numeric counters, a per-run random salt, and an
HMAC-SHA-256 observation digest. Successful Tool progress is likewise
represented by a run-salted fingerprint. Raw prompts, model output, Tool
arguments, Tool results, provider metadata, and provider-generated call IDs
are not stored by the guard.

## Framework boundary

ModuAgent 0.6 does not include domain Recipes, a Workflow DSL, database
abstractions, SQL generation, or report-specific behavior. The framework
composes and runs the components; the application remains responsible for:

- instructions and business rules;
- Tool implementations and input/result schemas;
- authoritative database or service schemas;
- Tool idempotency, repair, and timeout-safety declarations;
- database roles, transactions, query limits, and other real security
  boundaries.

The Quick API only removes repetitive framework wiring. It does not infer
domain semantics or Tool safety.

## Advanced composition: Context Memory

Use the same `session_id` to continue a conversation. `Agent.create()` accepts
the common stores, memory policy, authorization, checkpoints, Skills, and
observability components. Use the explicit constructor for lower-level
composition such as a separate planning model, custom planner or policy,
detailed Tool recovery, or a custom Engine.

```python
from moduagent import Agent, InMemoryConversationStore, RecentTurnsConversationMemoryPolicy

conversations = InMemoryConversationStore(
    ttl_seconds=3600,
    max_sessions=1_000,
    max_total_bytes=16_000_000,
)

memory_agent = Agent.create(
    name="memory-assistant",
    instructions="Use relevant conversation context when answering.",
    model=model,
    conversation_store=conversations,
    memory=RecentTurnsConversationMemoryPolicy(max_turns=6),
)

async def demonstrate_memory() -> None:
    first = await memory_agent.run(
        "Remember that my deployment region is Seoul.",
        session_id="user-42",
    )
    first.raise_for_error()

    result = await memory_agent.run(
        "Which deployment region did I choose?",
        session_id="user-42",
    )
    print(result.unwrap())
```

The store and the memory policy have different jobs:

| Component | Responsibility |
|---|---|
| `ConversationStore` | Saves the complete public conversation |
| `ConversationMemoryPolicy` | Selects the view sent to the model |

These components provide **Context Memory** for the current session and bound
what is sent in each model request. They do not provide **Long-Term Memory**
for retrieving facts, preferences, or episodes across sessions.

`RecentTurnsConversationMemoryPolicy` does not delete stored messages. It sends
only the latest complete turns to the model. In-memory stores are intended for
single-process development and tests. Because this policy does not count
tokens, `MEMORY_COMPACTED` reports `original_tokens=0` and
`selected_tokens=0`; those values mean “not measured,” not a zero-token
request.

The compatibility default, `FullConversationMemoryPolicy`, is unbounded and
can exceed a production endpoint's context window as a session grows. For
production, prefer `TokenBudgetConversationMemoryPolicy` with an exact counter
for the deployed model, such as `VLLMTokenCounter`. Add a summarizer only when
older context must be retained. See the
[Context Memory guide](https://github.com/nagix999/moduagent/blob/main/docs/conversation-memory-policy.md).
If exact vLLM tokenization is used repeatedly, wrap the counter with
`CachingTokenCounter`; it stores only a bounded keyed digest and successful
token count.

For long durable sessions, 0.6 also provides
`DurableSummarizingConversationMemoryPolicy`. It reads only the paginated tail
after a monotonic summary cursor and writes a tenant/Agent/session/policy-bound
summary v2 snapshot through compare-and-swap. It requires a conversation store
with a native bounded `load_tail()` implementation; a full-blob compatibility
fallback is rejected. See the Context Memory guide for Redis list mode,
database repository, reset, and legacy-summary migration requirements.

Wrap the raw history backend in `ScopedConversationStore`; its default
`key_mode="shared"` isolates equal public session IDs across tenant/Agent
namespaces. For a 0.5 `MemorySnapshot`, bind the dedicated legacy state
namespace with `ScopedLegacyMemoryStateStore` and pass it as
`legacy_state_store=`. On the first v2 miss, the loader scans the canonical
prefix twice with bounded pagination and writes v2 only when the count, digest,
store-issued IDs, and both scans agree. `ContextAssembler` then budgets system
and Skill policy, current task/run, Tool protocol, Tool/output schemas, optional
summary, and recent complete turns together. A newly generated summary is
committed only if selected; a persisted summary or CAS winner that does not fit
is omitted while the required/recent-only request continues.

## Advanced composition: delegate to a versioned Agent

Use `Agent.as_tool()` when one Agent should call another. Both Agents have an
exact-version `AgentDefinition`; the registry resolves only that version, and
the coordinator checks authorization, topology, cycles, depth, deadline,
aggregate model/Tool budgets, child session isolation, and receipts before the
child model is called.

```python
from moduagent import (
    Agent,
    AgentEndpoint,
    DefinitionStatus,
    InMemoryAgentRegistry,
    RuntimeBindings,
)
from moduagent.delegation import DelegationCoordinator, DelegationPolicy

registry = InMemoryAgentRegistry()
registry.register(
    specialist_definition,
    AgentEndpoint(handler=specialist, approved=True),
    status=DefinitionStatus.ACTIVE,
)
coordinator = DelegationCoordinator(
    registry=registry,
    policy=DelegationPolicy(
        allowed_edges={"supervisor": {"specialist"}},
        allowed_tenants={"tenant-a"},
        allowed_principals={"analyst-1"},
    ),
)
ask_specialist = specialist.as_tool(
    coordinator=coordinator,
    caller=supervisor_definition.ref,
    input_model=ResearchRequest,
    output_model=ResearchAnswer,
    name="ask_specialist",
)
supervisor = Agent.create(
    name="supervisor",
    model=model,
    instructions="Delegate specialist work, then answer.",
    tools=(ask_specialist,),
    definition=supervisor_definition,
    runtime_bindings=RuntimeBindings(
        tenant_context_provider=lambda: "tenant-a",
        principal_context_provider=lambda: "analyst-1",
    ),
)
```

The complete [offline runnable example and operations guide](https://github.com/nagix999/moduagent/blob/main/docs/delegation.md)
defines the Pydantic contracts and both definitions. In-memory registry,
budget, and receipt implementations are for one-process development. Production
delegation requires durable atomic-CAS store implementations, the same stable
HMAC secret/namespace on every worker, isolated child sessions, and
application-level idempotency for external side effects. Legacy `AgentTool`
remains available for compatibility but does not provide these execution-group
guarantees and is rejected by the Production profile.

### Application example: report automation

The repository includes a complete Plan-and-Execute Agent with only two Tools:

- `query_db`: runs a bounded, read-only query against SQLite (default) or
  PostgreSQL.
- `plot_graph`: reads the run-scoped query artifact and creates a PNG chart.

See
[examples/report_automation_agent.py](https://github.com/nagix999/moduagent/blob/main/examples/report_automation_agent.py).

```bash
python -m pip install matplotlib
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL="your-tool-capable-model"
python examples/report_automation_agent.py
```

To run the same example against PostgreSQL, keep the model variables above and
add:

```bash
python -m pip install "psycopg[binary]>=3.2,<4"
export REPORT_DB_BACKEND="postgresql"
export REPORT_DATABASE_URL="postgresql://report_reader@localhost:5432/reporting"
python examples/report_automation_agent.py
```

Use a dedicated database role with only `CONNECT`, schema `USAGE`, and `SELECT`
privileges. The example also starts a read-only transaction and applies
statement and lock timeouts.

This is application code demonstrating how a user can define prompts, schemas,
Tools, and safety controls. It is not a built-in Recipe or database
abstraction.

## After the Quick API

At this point you can build most Agents. The following features are optional;
add them when you need streaming, recovery, reusable domain procedures, or
deployment controls.

## Streaming results

Use `stream()` for user-facing output:

```python
from moduagent import EventType

async def stream_result() -> None:
    result = None

    async for event in planning_agent.stream(
        "Run the task and stream the final answer.",
        session_id="stream-demo",
    ):
        if event.type in (EventType.MODEL_DELTA, EventType.FINAL_DELTA):
            print(event.data["delta"], end="", flush=True)
        elif event.type in (EventType.RUN_COMPLETED, EventType.RUN_FAILED):
            result = event.data["result"]

    if result is not None:
        result.raise_for_error()
```

Handle both public delta types: direct Standard responses use `MODEL_DELTA`;
staged finalization, including Plan-and-Execute, uses `FINAL_DELTA`.

`stream_all()` also exposes diagnostic internal events and intermediate model
deltas. Use it only in an access-controlled diagnostic path, not as a direct
user-facing stream.

## Inspecting steps and failures

Use an `EventSink` or `stream_all()` for the execution timeline. Add a
`DiagnosticSink` when developers also need the sanitized cause of an exception:

```python
import asyncio
import logging

from moduagent import Agent, InMemoryDiagnosticSink, LoggingEventSink

logging.basicConfig(level=logging.INFO)
diagnostics = InMemoryDiagnosticSink(max_records=1_000)

observable_agent = Agent.create(
    name="observable-agent",
    instructions="Complete the request using the available Tools.",
    model=model,
    tools=[add],
    event_sink=LoggingEventSink(),
    diagnostic_sink=diagnostics,
)


async def main() -> None:
    result = await observable_agent.run("Use add for 20 + 22.")
    print(dict(result.run_usage))
    for trace in result.tool_trace:
        print(dict(trace))
    if result.error_summary:
        print(dict(result.error_summary))

    if result.failure_id is not None:
        failure = diagnostics.get(result.failure_id)
        if failure is not None:
            print(failure.to_dict())

    for failure in diagnostics.for_run(result.run_id):
        print(failure.failure_id, failure.component, failure.operation)


asyncio.run(main())
```

`result.tool_trace` shows executed Tools and their correlation IDs.
`result.failure_id` identifies the root failure of a terminal run. Its Tool
record can have `terminal=False` because that flag means “recoverable when
captured”; the Plan policy may decide to stop afterward. Recovered Tool
failures may appear only in the Tool trace and `diagnostics.for_run()`.

Diagnostics are off by default; omitting `diagnostic_sink` or using
`NoopDiagnosticSink` preserves the default behavior. Delivery is best effort and
bounded by `diagnostic_timeout_seconds` and
`diagnostic_max_pending_deliveries`. Standard-library logging uses a bounded
daemon worker pool; a synchronous handler already in flight cannot be
force-cancelled. Custom async sinks must honor cancellation.

Diagnostic fields are bounded and omit raw exception messages, SQL, prompts,
Tool arguments or results, provider bodies, source lines, and local variables.
Real `OSError.errno` and eager allowlisted attributes can be retained;
Pydantic dynamic keys are hidden and truncated tracebacks keep their innermost
frames. Built-in event logs omit payloads and free-form reasons and hash
step/Tool correlation IDs.

Strict Plan validation exposes framework-owned `validation_code`,
`validation_location`, and optional `validation_cause_code`; inspect them
instead of parsing a reason string. Custom Engine authors must treat
`EngineOutcome.error` as public, trusted text.
`AgentResult.metadata["error_summary"]` is runtime-owned and cannot be
overridden through Engine metadata. See the
[diagnostics guide](https://github.com/nagix999/moduagent/blob/main/docs/diagnostics.md)
for logging, custom durable sinks, and security guidance.

## Performance metrics

`MetricsEventSink` records `model.calls` and phase-aware model duration, plus
memory preparation, Tool, checkpoint, run, and same-session queue timings.
Noop observability skips its queue and copy path completely. Event handoff
queues are bounded, so a slow sink applies backpressure instead of retaining
unlimited payloads.

Run the source-tree microbenchmark after changing execution or persistence
code:

```bash
python benchmarks/performance_v042.py --pretty
```

## Checkpoints and safe resume

Add a checkpoint store when interrupted work must continue:

```python
from moduagent import InMemoryCheckpointStore

checkpoints = InMemoryCheckpointStore()

resumable_agent = Agent.create(
    name="resumable-agent",
    instructions="Complete the request safely.",
    model=model,
    tools=[add],
    conversation_store=InMemoryConversationStore(),
    checkpoint_store=checkpoints,
)

async def resume_if_safe() -> None:
    failed = await resumable_agent.run(
        "Run the task.",
        session_id="resume-demo",
    )

    error_summary = failed.metadata.get("error_summary", {})

    if failed.error and error_summary.get("resumable") is True:
        resumed = await resumable_agent.resume(
            failed.run_id,
            session_id="resume-demo",
        )
        print(resumed.output)
```

Resume with the original `run_id`, the same `session_id`, and a compatible
Agent configuration. Resume only when `error_summary["resumable"]` is true.
`InMemoryCheckpointStore` demonstrates same-process recovery only; it loses
all checkpoints when the process exits.

ModuAgent 0.6 writes checkpoint envelope v5 and event envelope v2; built-in
Engine state remains v1. Checkpoint v1-v4 can be read through
`migrate_checkpoint_payload()`, but 0.5.x cannot read v5. Durable Context Memory
uses its independent summary schema v2. Read the
[0.6 migration guide](https://github.com/nagix999/moduagent/blob/main/docs/migration-0.6.md)
before sharing a storage namespace between mixed-version workers or planning a
rollback.

- `retryable` means a new run may be attempted.
- `resumable` means the saved run can continue without replaying an unsafe
  side effect.

A checkpoint can exist while `resumable` is false. In particular, a Tool may
have started without a durably committed outcome. ModuAgent fails closed and
requires manual review instead of automatically replaying that Tool.
`max_model_turns` and `no_progress` are terminal guard decisions and are also
always `resumable=false`; resume cannot enlarge a consumed turn budget or
reopen a tripped circuit.

Checkpointed Agents require a `ConversationStore` with atomic
`append_once()`. Built-in in-memory and supported Redis stores implement this
contract. Use Redis or a custom durable adapter in production.

## Add domain knowledge with Skills

Skills provide reusable instructions and bounded text resources. They do not
grant Tool permission.

```text
skills/
└── invoice-review/
    ├── SKILL.md
    ├── references/
    │   └── policy.md
    └── assets/
        └── report-template.md
```

```python
from moduagent import SkillRegistry, tool


@tool(idempotent=True)
def lookup_invoice(invoice_id: str) -> dict[str, object]:
    """Look up an invoice by ID."""
    return {
        "invoice_id": invoice_id,
        "amount": 125_000,
        "evidence_attached": True,
        "approved": False,
    }


skills = SkillRegistry.from_paths("./examples/skills")

agent = Agent.create(
    name="invoice-agent",
    instructions="Use verified evidence only.",
    model=model,
    tools=[lookup_invoice],
    skill_registry=skills,
)

async def review_invoice() -> None:
    result = await agent.run(
        "Review invoice INV-100.",
        session_id="invoice-42",
        skills=["invoice-review"],
    )
    print(result.unwrap())
```

The sample Tool returns fixed data for demonstration. Replace its body with
your own authorized data-access code.

The effective Tool scope is the intersection of registered Tools, the Skill's
`allowed-tools`, and the configured `ToolAuthorizer`. Skill `scripts/` are
never executed automatically.

See the
[Agent Skills guide](https://github.com/nagix999/moduagent/blob/main/docs/skills.md)
for authoring, lockfiles, resource limits, and automatic selection.

## Inspect the resolved Agent

`Agent.inspect()` returns an immutable, credential-redacted `AgentSpec` without
making an external request:

```python
spec = planning_agent.inspect()

print(spec.execution_profile.kind)  # plan
print(spec.agent_fingerprint)
print(spec.to_dict(include_instructions=False))
```

The specification includes resolved model capabilities, Tool schema
fingerprints and safety profiles, output behavior, persistence policy, and
compatibility metadata. API keys and tokens are redacted. The original
instructions remain available on the object; do not put secrets in
instructions, and use `include_instructions=False` before logging or exporting
the specification.

## Before production

- Replace in-memory conversation, checkpoint, and summary stores with durable
  stores.
- For delegation, replace the in-memory registry, budget ledger, and receipt
  store with exact-version and atomic-CAS durable implementations; keep one
  stable HMAC secret and namespace across workers.
- Set model, Tool, database, and overall run timeouts independently.
- A timeout around a synchronous Python Tool cannot forcibly stop the
  underlying thread. Configure a driver or server-side statement timeout too.
- Limit database rows, Tool result bytes, context tokens, output tokens, Plan
  steps, and Tool calls.
- Declare retry or changed-argument repair safety only after reviewing Tool
  side effects.
- Classify every Production Tool as `none`, `read`, `advisory`, or `write`.
  The 0.6 Production profile accepts only the first three and freezes the Tool
  registry after validation; write Tools require a future enforced approval
  plane and remain fail-closed.
- Treat model, Tool, adapter, and composition code loaded in the Python process
  as trusted deployment code. Registry freezing blocks supported post-validation
  replacement APIs; it is not a sandbox against code that mutates objects or
  monkey-patches the process.
- Give write Tools application-level idempotency keys and duplicate handling.
- Use `ToolAuthorizer` or RBAC; Skill `allowed-tools` only narrows scope.
- Keep the default summary Tool trace unless argument logging has a clear,
  reviewed purpose.
- Never place raw exceptions, SQL, credentials, customer data, or internal
  paths in model-visible error messages.
- Configure encryption, tenant isolation, access control, retention, and TTLs
  for conversations, checkpoints, events, and generated artifacts.
- Record `agent.inspect()` with the deployment and test resume behavior before
  upgrading a live Agent.
- Treat `RuntimeAttestation` as application-owned Test-profile metadata. Its
  canonical digest detects accidental fact substitution, but it is not a
  signature or a security boundary; derive the facts in trusted deployment
  code and do not accept them from prompts or request payloads.
- Send public streams to users and internal events to protected `EventSink`
  implementations. Store failure diagnostics in a separately access-controlled
  `DiagnosticSink`.

ModuAgent does not provide a distributed lock, worker queue, scheduler,
durable outbox, or end-to-end exactly-once Tool execution. Add these through
your application infrastructure when required.

## Common questions

### Why did a Tool not run when I used Pydantic output?

In ModuAgent, Tool selection and final structured output are separate model
phases.
If no Tool was called, check the model's Tool Calling support, its chat
template/parser configuration, the Tool description, and the Agent
instructions.

### Why did Plan-and-Execute finish with `max_steps`?

For strict Plan-and-Execute, `max_steps` is the maximum number of generated
Plan steps. It is not the total number of model requests. Increase it only
when the task genuinely requires more independently verifiable steps.

### Where can I see which Tool was actually called?

Use `result.metadata["tool_trace"]`. A Plan step's `allowed_tools` lists what
was permitted, not what was executed.

### Is `InMemoryConversationStore` production storage?

No. It is process-local and intended for examples, tests, and development.
Use Redis or a durable custom store for multi-process or restart-safe systems.

## Public API map

| Need | Main API |
|---|---|
| Quick build and output | `Agent.create()`, `Agent.ask()`, `AgentRunError` |
| Operate a run | `Agent.run()`, `AgentResult`, `RunLimits`, `RetryConfig` |
| Compose explicitly | `Agent`, `AgentConfig` |
| Connect models | `VLLMClient`, `OllamaClient` |
| Add Tools | `tool`, `function_tool`, `ToolSafetyProfile`, `ToolAuthorizer` |
| Choose execution | `StandardExecutionProfile`, `PlanExecutionProfile` |
| Validate output | `PydanticOutputCodec`, `TextOutputCodec` |
| Keep bounded session context | `ConversationStore`, `RecentTurnsConversationMemoryPolicy` |
| Delegate typed child work | `AgentDefinition`, `Agent.as_tool()`, `moduagent.delegation.DelegationCoordinator` |
| Resume work | `CheckpointStore`, `Agent.resume()` |
| Add domain procedures | `SkillRegistry`, `SkillSelector` |
| Observe runs | `Agent.stream_all()`, `EventSink`, `DiagnosticSink`, `failure_id` |
| Inspect configuration | `Agent.inspect()`, `AgentSpec` |

## Examples

- [Report automation Agent](https://github.com/nagix999/moduagent/blob/main/examples/report_automation_agent.py):
  strict Plan-and-Execute using only `query_db` and `plot_graph`, with SQLite
  and PostgreSQL query backends.
- [Invoice review Skill](https://github.com/nagix999/moduagent/tree/main/examples/skills/invoice-review):
  Skill instructions, references, and assets.
- [Production controls](https://github.com/nagix999/moduagent/blob/main/examples/PRODUCTION.md):
  authorized idempotent writes, bounded memory, durable resume, cancellation,
  and concurrent-session guidance.

## Documentation

The detailed guides are currently written in Korean.

- [Core API](https://github.com/nagix999/moduagent/blob/main/docs/core-api.md):
  Agent construction and common APIs.
- [Advanced API](https://github.com/nagix999/moduagent/blob/main/docs/advanced-api.md):
  custom Engines, Tool failure contracts, and extension points.
- [Plan-and-Execute](https://github.com/nagix999/moduagent/blob/main/docs/plan-and-execute.md):
  strict state machine and recovery details.
- [Context Memory](https://github.com/nagix999/moduagent/blob/main/docs/conversation-memory-policy.md):
  bounded session context, token budgets, and summarization.
- [Agent delegation](https://github.com/nagix999/moduagent/blob/main/docs/delegation.md):
  versioned child Agents, aggregate budgets, receipts, and operations.
- [Agent Skills](https://github.com/nagix999/moduagent/blob/main/docs/skills.md):
  reusable procedures and resource access.
- [Operations](https://github.com/nagix999/moduagent/blob/main/docs/operations.md):
  security, timeouts, stores, events, and deployment.
- [Diagnostics](https://github.com/nagix999/moduagent/blob/main/docs/diagnostics.md):
  step timelines, failure correlation, sanitized details, and custom sinks.
- [0.4 migration](https://github.com/nagix999/moduagent/blob/main/docs/migration-0.4.md):
  source compatibility and checkpoint migration.
- [0.5 migration](https://github.com/nagix999/moduagent/blob/main/docs/migration-0.5.md):
  Quick API, safety changes, and the migration-free 0.5.3 PATCH.
- [0.6 migration](https://github.com/nagix999/moduagent/blob/main/docs/migration-0.6.md):
  definitions, delegation, checkpoint/event/summary schemas, and rollout limits.
- [Changelog](https://github.com/nagix999/moduagent/blob/main/CHANGELOG.md)

## Development

Run the offline test suite:

```bash
python -m pytest -q tests --ignore=tests/integration
```

Live vLLM, Ollama, and Redis tests are under `tests/integration` and skip when
their environment variables are not configured.

```bash
ruff check .
ruff format --check .
```

## License

ModuAgent is available under the
[MIT License](https://github.com/nagix999/moduagent/blob/main/LICENSE).
