# Production controls example

[`20_production_controls.py`](20_production_controls.py) is a focused bridge
from the learning examples to an operated Agent. It reads one change request
and conditionally performs one approval write. The application, rather than
the model, owns identity, authorization, the idempotency key, and storage.

Run it after configuring an OpenAI-compatible vLLM endpoint:

```bash
export VLLM_BASE_URL="http://localhost:8000/v1"
export VLLM_MODEL="your-tool-capable-model"
export VLLM_API_KEY="..."  # Omit when the endpoint needs no token.
python examples/20_production_controls.py
```

Importing the module does not read credentials or create a network client.
`VLLMClient.from_env()` is called only by `main()`.

## What the example enforces

- `get_change_request` and `approve_change` share one authoritative repository.
  Both enforce the controller-issued target/tenant scope; the write accepts the
  exact version read from the record.
- `ScopedChangeAuthorizer` first applies deny-by-default RBAC, then requires the
  requested change and tenant to match the controller-issued scope. The write
  Tool repeats that scope check at the application boundary.
- Roles, `user_id`, target scope, and the idempotency key enter through trusted
  `user_context`; prompt text must never grant authority. Do not place JWTs,
  credentials, or personal data there: context can appear in protected events
  and checkpoints.
- The controller creates one idempotency key before the run. The write Tool
  injects it from trusted context, outside the model-visible schema and prompt,
  and passes it to an application-owned store that atomically binds key and
  payload. An identical replay returns the original receipt; key reuse with a
  different payload fails.
- Eligibility, expected version, the state transition, uniqueness checks, and
  receipt creation occur inside one repository critical section. A production
  database implementation must use one transaction and conditional update
  (for example, `UPDATE ... WHERE version=? AND status='pending' RETURNING ...`)
  rather than a read-then-write check.
- Structured output validates shape, not business success. The controller
  reconciles it with the application receipt, which is the source of truth.
  Model turns, Tool calls, wall time, output size, and recent conversation turns
  are bounded.
- Checkpoint, conversation, and diagnostic components are passed directly to
  `Agent.create()`. Protected diagnostics remain separate from public output.

The example intentionally uses `InMemoryApprovalStore`,
`InMemoryConversationStore`, `InMemoryCheckpointStore`, and
`InMemoryDiagnosticSink`. They make the file runnable, but they are neither
durable nor shared across processes. Conversation TTL, session count, and
serialized-byte limits bound retained message payload accounting, not Python
object overhead or process RSS; eviction is not archival storage. For a real
approval, use a database transaction with unique constraints on both the
idempotency key and the tenant/business operation, and store a payload digest
with the receipt. Make version and eligibility part of the same conditional
write that records the receipt.

## Compact long conversations

The runnable example keeps six recent complete turns. When older context must
be retained semantically, use a token budget and a summarizer:

```python
from moduagent import (
    InMemoryMemoryStateStore,
    ModelConversationSummarizer,
    SummarizingConversationMemoryPolicy,
    TokenBudget,
    VLLMTokenCounter,
)

memory = SummarizingConversationMemoryPolicy(
    budget=TokenBudget(
        context_window_tokens=32_768,
        reserved_output_tokens=1_024,
        safety_margin_tokens=1_024,
    ),
    token_counter=VLLMTokenCounter(model),
    summarizer=ModelConversationSummarizer(
        model=model,
        max_input_tokens=8_192,
        max_output_tokens=512,
    ),
    state_store=InMemoryMemoryStateStore(),
    max_history_turns=40,
)
agent = build_agent(
    model,
    conversation_store=conversations,
    memory=memory,
)
```

Summaries are lossy model output, consume model turns, and reduce only the
model-facing view; they do not delete raw conversation records. Replace the
in-memory summary state with a durable, tenant-isolated implementation when
summaries must survive restarts.

## Durable checkpoint and resume

The example's checkpoint store demonstrates the API, not crash durability. A
single Redis client can back restart-safe conversation and checkpoint stores:

```python
from redis.asyncio import Redis
from moduagent import RedisCheckpointStore, RedisConversationStore

redis = Redis.from_url("redis://localhost:6379/0", decode_responses=True)
conversations = RedisConversationStore(
    redis,
    key_prefix="moduagent:conversation:",
    ttl_seconds=7 * 24 * 60 * 60,
)
checkpoints = RedisCheckpointStore(
    redis,
    key_prefix="moduagent:checkpoint:",
    ttl_seconds=24 * 60 * 60,
)
agent = build_agent(
    model,
    conversation_store=conversations,
    checkpoint_store=checkpoints,
)

result = await agent.resume(saved_run_id, session_id=saved_session_id)
```

Resume requires the same session and a compatible Agent fingerprint. A
checkpoint marked `manual_required` must not be replayed automatically. Keep
the application approval record and idempotency key durable as well; a durable
Agent checkpoint cannot make a process-local write store exactly-once.
Revalidate current identity, role, tenant, and target permission through an
external authorization backend before every resumed write; do not treat roles
stored in an older checkpoint as current authority.

## Cancel a stream cleanly

Keep the `run_id` from `RUN_STARTED`, and always close the async generator when
the consumer disconnects. Closing the generator or cancelling and awaiting its
consumer task lets ModuAgent run cancellation cleanup and preserve a safe
checkpoint when possible:

```python
from moduagent import EventType

stream = agent.stream_all(command, session_id=session_id, user_context=identity)
run_id = None
try:
    async for event in stream:
        if event.type is EventType.RUN_STARTED:
            run_id = event.run_id
        elif event.type is EventType.MODEL_STARTED:
            break  # Example: the downstream client disconnected in flight.
finally:
    await stream.aclose()

checkpoint = await checkpoints.load(run_id) if run_id else None
if checkpoint is not None and checkpoint.resume_safety == "resumable":
    result = await agent.resume(run_id, session_id=session_id)
```

`stream_all()` is for trusted server-side diagnostics. Do not forward its
internal events directly to an end user; project and redact an explicit public
event schema first.

Do not assume cancellation stops a synchronous driver or remote side effect.
Use driver-level timeouts and application idempotency, and inspect
`resume_safety` before resuming.

## Run concurrent sessions

Different session IDs can run concurrently on one Agent instance:

```python
first, second = await asyncio.gather(
    agent.run(first_command, session_id="ticket-4815", user_context=first_user),
    agent.run(second_command, session_id="ticket-4816", user_context=second_user),
)
```

Calls with the same session ID are serialized inside one runtime so their
conversation order is stable. That lock is process-local; multi-worker
deployments still need application-level distributed coordination. Bound
provider concurrency and Tool worker queues, and rely on the durable
idempotency constraint for write races.

## Deployment checklist

1. Authenticate outside the model and build minimal trusted `user_context`
   without credentials, JWTs, or personal data.
2. Persist the command and its idempotency key before starting the Agent.
3. Use durable, tenant-isolated conversation, checkpoint, summary, diagnostic,
   and approval stores with explicit retention.
4. Reauthorize write scope at execution/resume time, and apply database
   statement timeouts, conditional version/eligibility updates, transactions,
   business-operation and idempotency-key uniqueness constraints, and
   least-privilege credentials to write Tools.
5. Test denial, duplicate delivery, process crash, cancellation, resume, and
   concurrent writes before enabling a production approval path.
