# ModuAgent

[English](https://github.com/nagix999/moduagent/blob/main/README.md) |
[한국어](https://github.com/nagix999/moduagent/blob/main/README.ko.md)

ModuAgent is a composable Python runtime for building AI agents around your
own model endpoints and Python functions.

Start with a normal model or Tool-calling loop. Add bounded conversation
memory, validated Pydantic output, strict Plan-and-Execute, checkpoint recovery,
Skills, and observability only when your application needs them.

> Current version: **0.4.2 (Alpha)** · Python **3.10+** · **MIT License**

New to ModuAgent? Complete Steps 1 and 2 first. Then add memory, structured
output, or Plan-and-Execute only when your use case needs them.

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
  ├── Conversation store + memory policy
  ├── Output codec
  ├── Checkpoint store
  ├── Skills and authorization
  └── Events, diagnostics, and metrics
```

- `AgentConfig` defines instructions, retry behavior, and run limits.
- A model client connects to vLLM, Ollama, or another supported endpoint.
- Tools are typed Python functions the model may call.
- An execution profile controls how work proceeds.
- An output codec returns text or a validated Pydantic object.
- A conversation store saves history; a memory policy selects the model view.
- A checkpoint store saves interrupted runs for safe recovery.

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
python -m pip install "moduagent==0.4.2"
```

If your package index does not contain `0.4.2` yet and you already have a 0.4
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
import os

from moduagent import Agent, AgentConfig, VLLMClient


def create_model() -> VLLMClient:
    return VLLMClient(
        base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        model=os.environ["VLLM_MODEL"],
        api_key=os.getenv("VLLM_API_KEY"),
        timeout=60,
    )


async def main() -> None:
    agent = Agent(
        config=AgentConfig(
            name="assistant",
            instructions="Answer accurately and concisely.",
        ),
        model=create_model(),
    )

    result = await agent.run(
        "Explain what an AI agent is in one paragraph.",
        session_id="getting-started",
    )

    if result.error:
        raise RuntimeError(result.error)

    print(result.output)
    print(f"Tokens used: {result.usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
```

Set the endpoint before running the file:

```bash
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL="your-model-name"
export VLLM_API_KEY="optional-token"
python getting_started.py
```

The endpoint and selected model must support the capabilities your Agent uses,
such as Tool Calling or JSON Schema output.
For Tool examples, configure vLLM's chat template and Tool parser for the
selected model.

`Agent.run()` returns an `AgentResult`. Its most useful fields are:

| Field | Meaning |
|---|---|
| `output` | Final text or validated object |
| `error` | Safe public error message, or `None` |
| `finish_reason` | `completed`, `timeout`, `max_steps`, and other terminal reasons |
| `usage` | Accumulated model token usage |
| `run_id` | Identifier used for checkpoint recovery |
| `messages` | Public conversation messages |
| `metadata` | Safe Tool trace, Plan summary, and error category |

Ollama uses the same Agent API:

```python
from moduagent import OllamaClient

model = OllamaClient(
    base_url="http://localhost:11434",
    model="qwen3:14b",
)
```

## Step 2: add a Tool

Use `@function_tool` to expose a typed Python function. This example reuses
`create_model()` from Step 1.

```python
import asyncio

from moduagent import Agent, AgentConfig, RetryConfig, function_tool


@function_tool(
    idempotent=True,
    timeout_seconds=5,
    max_result_bytes=4096,
)
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


async def main() -> None:
    calculator = Agent(
        config=AgentConfig(
            name="calculator",
            instructions=(
                "Use the add Tool whenever addition is required. "
                "Do not invent a calculated result."
            ),
            retry=RetryConfig(max_attempts=2),
        ),
        model=create_model(),
        tools=[add],
    )

    result = await calculator.run(
        "What is 12 plus 30?",
        session_id="calculator-demo",
    )

    if result.error:
        raise RuntimeError(result.error)

    print(result.output)
    print(result.metadata.get("tool_trace", []))


asyncio.run(main())
```

The function's type hints become its input schema, and its docstring becomes
the description shown to the model. Only Tools passed through `tools=[...]`
can be called.

`idempotent=True` declares that repeating the same validated call is safe. It
does not create a transaction or exactly-once guarantee. For write Tools, your
application still needs an idempotency key and duplicate protection.

Blocking functions such as `pandas.read_sql()` run outside the event loop.
For production, share a bounded scheduler across synchronous Tools so timed-out
calls cannot create an unlimited number of background threads:

```python
from moduagent import SyncToolScheduler, function_tool

blocking_tools = SyncToolScheduler(max_workers=8, max_queue=32)

@function_tool(sync_scheduler=blocking_tools, timeout_seconds=10)
def query_db(sql: str) -> list[dict]:
    return run_read_only_query(sql)
```

Raw assistant Tool calls and raw Tool results are internal protocol messages.
They are available to the model during the run but are not added to
`ConversationStore` or `AgentResult.messages`. The default public Tool trace is
a bounded, secret-safe summary.

## Step 3: add conversation memory

Use the same `session_id` to continue a conversation.

The remaining snippets focus on the new configuration. They reuse
`create_model()` and `add` from the earlier steps. Each asynchronous example
is wrapped in a function that you can call from your application's entry point.

```python
from moduagent import (
    Agent,
    AgentConfig,
    InMemoryConversationStore,
    RecentTurnsConversationMemoryPolicy,
)

conversations = InMemoryConversationStore(ttl_seconds=3600)

memory_agent = Agent(
    config=AgentConfig(
        name="memory-assistant",
        instructions="Use relevant conversation context when answering.",
    ),
    model=create_model(),
    conversation_store=conversations,
    conversation_memory_policy=RecentTurnsConversationMemoryPolicy(max_turns=6),
)

async def demonstrate_memory() -> None:
    remembered = await memory_agent.run(
        "Remember that my deployment region is Seoul.",
        session_id="user-42",
    )
    if remembered.error:
        raise RuntimeError(remembered.error)

    result = await memory_agent.run(
        "Which deployment region did I choose?",
        session_id="user-42",
    )
    if result.error:
        raise RuntimeError(result.error)
    print(result.output)
```

The store and the memory policy have different jobs:

| Component | Responsibility |
|---|---|
| `ConversationStore` | Saves the complete public conversation |
| `ConversationMemoryPolicy` | Selects the view sent to the model |

`RecentTurnsConversationMemoryPolicy` does not delete stored messages. It sends
only the latest complete turns to the model. In-memory stores are intended for
single-process development and tests.

For strict token limits and automatic summarization, see the
[Conversation Memory guide](https://github.com/nagix999/moduagent/blob/main/docs/conversation-memory-policy.md).
If exact vLLM tokenization is used repeatedly, wrap `VLLMTokenCounter` with
`CachingTokenCounter`; it stores only a bounded keyed digest and successful
token count.

## Step 4: return validated structured output

`PydanticOutputCodec` validates the final model response and returns the model
object through `result.output`.

```python
from pydantic import BaseModel, Field

from moduagent import Agent, AgentConfig, PydanticOutputCodec


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)


structured_agent = Agent(
    config=AgentConfig(
        name="structured-assistant",
        instructions=(
            "Use the add Tool for every arithmetic operation, then return "
            "the answer in the requested format."
        ),
    ),
    model=create_model(),
    tools=[add],
    output_codec=PydanticOutputCodec(model=Answer),
)

async def demonstrate_structured_output() -> None:
    result = await structured_agent.run(
        "What is 20 plus 22?",
        session_id="structured-demo",
    )

    if result.error:
        raise RuntimeError(result.error)

    answer: Answer = result.output
    print(answer.answer, answer.confidence)
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

## Step 5: use strict Plan-and-Execute

Use Plan-and-Execute when a task has dependent steps and each intermediate
result must be validated before it can affect the final answer.

```python
from moduagent import (
    Agent,
    AgentConfig,
    InMemoryCheckpointStore,
    InMemoryConversationStore,
    LLMPlanGenerator,
    PlanExecutionProfile,
    RunLimits,
)

model = create_model()

planning_agent = Agent(
    config=AgentConfig(
        name="planning-agent",
        instructions=(
            "Use the add Tool for every arithmetic operation. "
            "Complete multi-step requests using only validated and committed "
            "step results."
        ),
        limits=RunLimits(
            max_steps=4,
            max_step_attempts=2,
            max_replans=1,
            max_tool_calls=8,
            timeout_seconds=120,
        ),
    ),
    model=model,
    tools=[add],
    execution_profile=PlanExecutionProfile(
        plan_generator=LLMPlanGenerator(
            model=model,
            max_steps=4,
        ),
        revise_on_tool_failure=True,
    ),
    conversation_store=InMemoryConversationStore(),
    checkpoint_store=InMemoryCheckpointStore(),
)

async def demonstrate_plan_execution() -> None:
    result = await planning_agent.run(
        "Calculate 10 + 20, then add 5 to that verified result.",
        session_id="plan-demo",
    )
    if result.error:
        raise RuntimeError(result.error)
    print(result.output)
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

Standard execution remains the better default for chat, direct Tool use, and
short workflows.

### Complete report automation example

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

## After the five steps

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

    if result is not None and result.error:
        raise RuntimeError(result.error)
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
from moduagent import InMemoryDiagnosticSink, LoggingEventSink

diagnostics = InMemoryDiagnosticSink(max_records=1_000)

observable_agent = Agent(
    config=AgentConfig(
        name="observable-agent",
        instructions="Complete the request using the available Tools.",
    ),
    model=create_model(),
    tools=[add],
    event_sink=LoggingEventSink(),
    diagnostic_sink=diagnostics,
    diagnostic_timeout_seconds=0.25,
    diagnostic_max_pending_deliveries=1_024,
)

result = await observable_agent.run("Use add for 20 + 22.")

if result.failure_id is not None:
    failure = diagnostics.get(result.failure_id)
    if failure is not None:
        print(failure.to_dict())

for failure in diagnostics.for_run(result.run_id):
    print(failure.failure_id, failure.component, failure.operation)
```

`result.metadata["tool_trace"]` shows executed Tools and their correlation IDs.
`result.failure_id` identifies the root failure of a terminal run. Its Tool
record can have `terminal=False` because that flag means “recoverable when
captured”; the Plan policy may decide to stop afterward. Recovered Tool
failures may appear only in the Tool trace and `diagnostics.for_run()`.

Diagnostics are off by default; omitting `diagnostic_sink` or using
`NoopDiagnosticSink` preserves the 0.4 behavior. Delivery is best effort and
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

resumable_agent = Agent(
    config=AgentConfig(
        name="resumable-agent",
        instructions="Complete the request safely.",
    ),
    model=create_model(),
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

- `retryable` means a new run may be attempted.
- `resumable` means the saved run can continue without replaying an unsafe
  side effect.

A checkpoint can exist while `resumable` is false. In particular, a Tool may
have started without a durably committed outcome. ModuAgent fails closed and
requires manual review instead of automatically replaying that Tool.

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
from moduagent import SkillRegistry, function_tool


@function_tool(idempotent=True)
def lookup_invoice(invoice_id: str) -> dict[str, object]:
    """Look up an invoice by ID."""
    return {
        "invoice_id": invoice_id,
        "amount": 125_000,
        "evidence_attached": True,
        "approved": False,
    }


skills = SkillRegistry.from_paths("./examples/skills")

agent = Agent(
    config=AgentConfig(
        name="invoice-agent",
        instructions="Use verified evidence only.",
    ),
    model=create_model(),
    tools=[lookup_invoice],
    skill_registry=skills,
)

async def review_invoice() -> None:
    result = await agent.run(
        "Review invoice INV-100.",
        session_id="invoice-42",
        skills=["invoice-review"],
    )
    if result.error:
        raise RuntimeError(result.error)
    print(result.output)
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

`Agent.inspect()` returns an immutable, secret-safe `AgentSpec` without making
an external request:

```python
spec = planning_agent.inspect()

print(spec.execution_profile.kind)  # plan
print(spec.agent_fingerprint)
print(spec.to_dict(include_instructions=False))
```

The specification includes resolved model capabilities, Tool schema
fingerprints and safety profiles, output behavior, persistence policy, and
compatibility metadata. API keys and tokens are redacted.

## Before production

- Replace in-memory conversation, checkpoint, and summary stores with durable
  stores.
- Set model, Tool, database, and overall run timeouts independently.
- A timeout around a synchronous Python Tool cannot forcibly stop the
  underlying thread. Configure a driver or server-side statement timeout too.
- Limit database rows, Tool result bytes, context tokens, output tokens, Plan
  steps, and Tool calls.
- Declare retry or changed-argument repair safety only after reviewing Tool
  side effects.
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
- Send public streams to users and internal events to protected `EventSink`
  implementations. Store failure diagnostics in a separately access-controlled
  `DiagnosticSink`.

ModuAgent does not provide a distributed lock, worker queue, scheduler,
durable outbox, or end-to-end exactly-once Tool execution. Add these through
your application infrastructure when required.

## Common questions

### Why did a Tool not run when I used Pydantic output?

In 0.4, Tool selection and final structured output are separate model phases.
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
| Build and run | `Agent`, `AgentConfig`, `RunLimits`, `RetryConfig` |
| Connect models | `VLLMClient`, `OllamaClient` |
| Add Tools | `function_tool`, `ToolSafetyProfile`, `ToolAuthorizer` |
| Choose execution | `StandardExecutionProfile`, `PlanExecutionProfile` |
| Validate output | `PydanticOutputCodec`, `TextOutputCodec` |
| Keep conversations | `ConversationStore`, `RecentTurnsConversationMemoryPolicy` |
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

## Documentation

The detailed guides are currently written in Korean.

- [Core API](https://github.com/nagix999/moduagent/blob/main/docs/core-api.md):
  Agent construction and common APIs.
- [Advanced API](https://github.com/nagix999/moduagent/blob/main/docs/advanced-api.md):
  custom Engines, Tool failure contracts, and extension points.
- [Plan-and-Execute](https://github.com/nagix999/moduagent/blob/main/docs/plan-and-execute.md):
  strict state machine and recovery details.
- [Conversation Memory](https://github.com/nagix999/moduagent/blob/main/docs/conversation-memory-policy.md):
  recent turns, token budgets, and summarization.
- [Agent Skills](https://github.com/nagix999/moduagent/blob/main/docs/skills.md):
  reusable procedures and resource access.
- [Operations](https://github.com/nagix999/moduagent/blob/main/docs/operations.md):
  security, timeouts, stores, events, and deployment.
- [Diagnostics](https://github.com/nagix999/moduagent/blob/main/docs/diagnostics.md):
  step timelines, failure correlation, sanitized details, and custom sinks.
- [0.4 migration](https://github.com/nagix999/moduagent/blob/main/docs/migration-0.4.md):
  source compatibility and checkpoint migration.
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
