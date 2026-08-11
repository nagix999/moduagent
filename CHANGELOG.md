# Changelog

## 0.6.0

- Added immutable, exact-version `AgentDefinition` identities, a lifecycle-aware
  `AgentRegistry`, replaceable `RuntimeBindings`, and Development/Test/Production
  profiles. Production composition now fails closed when required semantic
  digests, identity providers, durable stores, authorization, or telemetry are
  missing.
- Added typed Agent-to-Agent delegation through `Agent.as_tool()` and
  `DelegationCoordinator`. The coordinator validates the caller/callee edge,
  tenant, principal, data classification, cycle, depth, deadline, and aggregate
  execution-group budgets before invoking a child Agent.
- Added isolated HMAC-derived child sessions, deterministic delegation and child
  run IDs, durable-store protocols for budget state and receipts, atomic
  ownership/fencing, cancellation propagation, bounded resume/reconciliation,
  and content-free delegation lifecycle events. The legacy `AgentTool` remains
  available for compatibility but is rejected by the Production profile.
- Added durable Context Memory with paginated conversation-tail reads,
  tenant/Agent/session/policy composite keys, monotonic cursors, bounded source
  provenance, structured summary schema v2, and compare-and-swap Redis/database
  state adapters. `ScopedConversationStore` isolates raw session keys, while
  `ScopedLegacyMemoryStateStore` enables a bounded, automatic two-pass migration
  of verified 0.5 snapshots. ContextAssembler v1 applies one token budget to
  system/Skill, task/run, Tool protocol, request schemas, summary, and recent
  turns. Optional summaries are committed only when selected and omitted when
  an existing snapshot or CAS winner cannot fit. The existing memory policies
  remain available.
- Upgraded checkpoint envelopes to schema v5 and event envelopes to schema v2.
  `migrate_checkpoint_payload()` accepts checkpoint v1-v4 and projects older
  runs as roots; v5 is not downgrade-readable by 0.5.x. Built-in Engine state
  remains v1. Legacy summaries become usable only after the loader has derived
  authoritative source message IDs from two matching canonical-prefix scans.
- Added the offline WAF log analysis capstone with six read-only, event-scoped
  evidence Tools, synthetic no-network threat intelligence, bounded decoding,
  redacted previews by default, and structured provenance checks.
- Added the document Q&A and cited-report capstone with Docling Serve parsing,
  automatic request routing, replaceable retrieval, application-verified source
  excerpts, page/bounding-box/line provenance, and deterministic Markdown report
  assembly.

## 0.5.3

- Made legacy `AgentTool` treat every non-successful child `AgentResult` as a
  stable Tool failure instead of returning a failed child's `output=None` as a
  successful Tool value. Child timeout, cancellation, model-guard, output-
  validation, limit, and generic terminal failures now retain bounded,
  payload-free classifications.
- Kept canonical child terminal results and direct framework failures
  non-retryable at the legacy delegation boundary: generic Tool retry counts,
  changed-argument repair, timeout retry, and `idempotent=True` alone do not
  rerun them. A custom Agent-like object's explicitly pre-classified
  `ToolFailure` retains its declared safe contract.
- Added a composition-time warning when a parent and its legacy `AgentTool`
  child use the same `ConversationStore` object, because the legacy adapter
  forwards the parent's `session_id` and can mix their Context Memory histories.
- Hardened Context Memory regressions across PLAN, ACT, and FINALIZE, including
  consistent content-free compaction counters and duration, and terminal
  propagation of summary-model guard and protocol failures. Incomplete summary
  responses (`timeout`, `length`, `max_tokens`) are never used or cached.
- Clarified that `ConversationMemoryPolicy` provides request-scoped Context
  Memory, not cross-session Long-Term Memory. Documented the production risk of
  unbounded `FullConversationMemoryPolicy` and recommended an exact-token
  `TokenBudgetConversationMemoryPolicy` for production endpoints.
- This PATCH requires no data migration. Checkpoint envelope schema v4, event
  schema v1, built-in Engine state v1, Context Memory snapshot structure, and
  ConversationStore persistence formats are unchanged.

## 0.5.2

- Extended `Agent.create()` with checkpoint, Tool authorization, Skill,
  diagnostic-delivery, model-option, metadata, finalization, and stream
  visibility components while preserving explicitly injected falsey objects.
- Added bounded `InMemoryConversationStore` capacity with session-level LRU
  eviction, deterministic serialized-byte accounting, lazy TTL sweeping,
  explicit cleanup, defensive canonical rows, and content-free usage
  statistics. Capacity pressure removes expired sessions before live LRU
  victims.
- Routed built-in PLAN and replan requests through the same conversation-memory
  and phase-scoped Skill preparation boundary used by execution requests while
  keeping current planning protocol messages protected. Agent model options now
  apply to same-model planner calls; separately configured planning models keep
  isolated `LLMPlanGenerator` options.
- Hardened OpenAI-compatible and Ollama embedding responses: batch counts,
  exact OpenAI indices, non-empty vectors, consistent dimensions, finite
  numeric values, and empty-input behavior are now validated before data
  reaches a vector store.
- Added production-control examples and regression scenarios for conversation
  compaction, durable checkpoint recovery, tenant-scoped idempotent writes,
  atomic version/eligibility transitions, application-receipt reconciliation,
  streaming cancellation, and concurrent sessions without adding a domain
  Recipe or Workflow layer.
- Prevented raw model options from injecting a Tool schema outside the
  framework-owned `ModelRequest.tools` boundary.
- Kept checkpoint schema v4 and verified that 0.5.1a1 snapshot envelopes remain
  readable after the runtime version update.

## 0.5.1a1

- Added a beginner-first example path covering a minimal assistant, typed Tool
  use, Pydantic output, report automation, and safe run debugging, plus
  deterministic and opt-in live vLLM scenario tests.
- Added an intermediate path for parallel incident investigation, sequential
  customer-case resolution, and release-readiness gating. Each scenario uses
  five typed, application-owned Tools, a structured decision contract, explicit
  read-only or advisory boundaries, deterministic tests, and opt-in live vLLM
  assertions.
- Documented and regression-tested a guided-decoding whitespace-loop failure:
  explicit required output keys and compact JSON instructions let the incident
  scenario finish in 357 finalization tokens even with an 8,192-token cap.
- Added the terminal `model_output_incomplete` classification for provider
  `timeout`, `length`, and `max_tokens` finish reasons. The allowlisted
  `provider_finish_reason` is available in `error_summary`, `AgentRunError`,
  event logs, and protected diagnostics without retaining response content.
- Extended `Agent.create()` with common conversation and observability
  components and added immutable, secret-safe `AgentResult.error_summary`,
  `tool_trace`, and `run_usage` projections.
- Added per-run model-turn, Tool-call, and wall-clock counters on both
  successful and failed results.
- Added content-free model request shape evidence, a paired `MODEL_FAILED`
  event, retry timing and classification fields, and failed-call metrics.
- Made output-validation classification and primary failure summaries
  independent of diagnostic-sink availability.
- Disabled per-delta built-in logging by default while retaining an explicit
  content-free opt-in.
- Hardened secret-key masking across acronym, camel-case, underscore, and
  hyphen spellings in built-in observability and public Tool traces.
- Added `aclose()` and async context-manager support to built-in HTTP model
  clients without taking ownership of caller-injected transports.
- Preserved explicitly injected falsey components instead of replacing them
  with defaults.
- Documented the live scenario findings, bounded example generation, and
  Standard-versus-Plan latency tradeoff.

## 0.5.0

- Added the additive `Agent.create()` Quick API for Standard and strict Plan
  execution while retaining the existing `Agent(...)` and `compose_agent()`
  composition paths.
- Added `Agent.ask()`, `AgentResult.unwrap()`, `raise_for_error()`, and
  `explain()` with a secret-safe, structured `AgentRunError`.
- Added `@tool` as an alias of the conservative `@function_tool` adapter and
  `VLLMClient.from_env()` for the documented `VLLM_*` settings.
- Added `ModelProtocolError` and a strict retry allowlist: only timeouts,
  connection failures, HTTP 408, and HTTP 5xx responses are retryable; JSON,
  protocol, HTTP 4xx (including 429), validation, and programming failures
  terminate immediately.
- Changed malformed Plan and StepResult protocol responses to fail immediately
  instead of consuming step or provider retry budgets.
- Added the per-run `RunLimits.max_model_turns` budget and the
  `no_progress_model_turn_threshold` circuit breaker across PLAN, ACT, repair,
  memory, Skill, and FINALIZE model calls.
- Counted a successful Tool outcome as progress only when its run-salted
  fingerprint is new, so a loop of identical successful calls cannot
  continually reset the no-progress breaker. Successful memory-summary batches
  and committed Plan steps remain explicit progress boundaries.
- Persisted only model-guard counters, a per-run salt, HMAC-SHA-256 observation
  digests, and run-salted successful-outcome fingerprints in checkpoints; raw
  prompts, output text, Tool arguments/results, usage, provider metadata, and
  Tool call IDs are not retained by the guard.
- Durably reserved each model attempt immediately before provider I/O when
  checkpointing is enabled, including retry attempts, and made both terminal
  guard outcomes non-retryable and non-resumable.
- Kept domain prompts, schemas, Tool implementations, safety declarations,
  storage, authorization, and database protections application-owned; 0.5.0
  adds no domain-specific Recipe or Workflow layer.

## 0.4.2

- Removed Engine-state encoding and compatibility copies when no checkpoint store is configured.
- Added a direct checkpoint-v4 snapshot path for built-in Standard and Plan Engines while preserving legacy stores and custom Engines.
- Reduced large Plan legacy-adapter round trips by validating committed-result hashes at the persistence boundary instead of every policy transition.
- Removed the duplicate Plan finalization checkpoint without changing the durable finalization barrier.
- Added Noop event-sink fast paths, single-owner Composite isolation, lightweight published-event stamps, and bounded service/sink handoff queues.
- Added phase-aware model, memory, Tool, checkpoint, run, and session-queue timing metrics.
- Added request-local token-count reuse and the bounded, TTL/LRU, singleflight `CachingTokenCounter`.
- Added the opt-in bounded `SyncToolScheduler` and moved synchronous persistence adapters off the event loop through a shared bounded scheduler.
- Reclaimed completed session and Redis fallback locks instead of retaining one lock per identifier.
- Added structural performance regression tests and `benchmarks/performance_v042.py`.

## 0.4.1

- Added an opt-in protected diagnostic pipeline with structured `FailureDiagnostic` records, injectable `DiagnosticSink` implementations, opaque failure IDs, and sanitized cause and innermost-frame metadata.
- Correlated model attempts, Tool execution, output decoding, policy transitions, and terminal failures; a terminal result can reuse a Tool failure captured as recoverable instead of duplicating it.
- Added bounded best-effort diagnostic delivery with configurable timeout and pending-record limits, cancellation-aware async sink guidance, and a bounded daemon worker pool for standard-library logging.
- Added finite Plan validation codes and locations to retry, failure, terminal-result, and safe logging projections so operators do not need to parse free-form validation text.
- Improved built-in terminal, model, Tool, and Plan event projections, hashed step and Tool correlation IDs, omitted arbitrary payloads and raw reasons, and assigned warning or error severity to retries and failures.
- Kept diagnostic attributes bounded and allowlisted, preserved real `OSError.errno`, hid dynamic Pydantic locations, and excluded model output, Tool results, provider responses, secrets, and raw tracebacks.
- Documented `EngineOutcome.error` as a public custom-Engine boundary and reserved runtime-generated `error_summary` metadata.
- Preserved existing public stream behavior, constructor compatibility, and the 0.4.0 checkpoint runtime schema while keeping diagnostics disabled by default.

## 0.4.0

- Split the common run lifecycle from Standard and strict Plan execution through explicit execution profiles and versioned Engine contracts.
- Added immutable, secret-safe `AgentSpec` inspection with resolved model capabilities, Tool safety profiles, output contract, and persistence policy.
- Added `ToolSafetyProfile`, `ToolFailureClassification`, `FailureProjector`, `ToolBatchOutcome`, and guarded repair contracts while retaining the 0.3.2 Tool flags and error mapping APIs.
- Separated same-call retry in `ToolRuntime` from Plan-owned repair, replan, and terminal failure decisions.
- Reorganized strict Plan state into plan progress, step execution, Tool recovery, and finalization state with a compatibility facade for the 0.3 API.
- Added checkpoint schema v4 with an opaque, independently versioned Engine state envelope and validated v1-v3 read migration; v4 snapshots are not downgraded.
- Extended the event envelope with stable identity, session, Engine, schema-version, and monotonically increasing run sequence metadata while preserving existing event types.
- Added `StandardExecutionProfile` and `PlanExecutionProfile`; the existing `decision_policy` constructor argument remains supported through the compatibility resolver.
- Added an explicit Tool-plus-structured-output model capability and safe staged finalization for endpoints, including vLLM, that cannot satisfy both contracts in one request.
- Kept Standard and Plan Tool protocol transcripts internal so raw Tool arguments and results do not enter public conversation history or `AgentResult.messages`.
- Made ambiguous v3 partial-success Tool batches fail closed during v4 checkpoint migration.
- Split documentation into Core, Advanced, Operations, and 0.4 migration guides.
- Kept the core dependency set unchanged and deliberately excluded Graph execution, distributed queues, peer multi-agent protocols, and general Human-in-the-loop workflows.

## 0.3.2

- Added opt-in, generalized Tool failure recovery that can ask ACT to produce corrected arguments before falling back to bounded plan revision.
- Added `ToolRecoveryAction`, `ToolFailure`, and `ToolFailureRecoveryConfig`, including `error_mapper` support for bounded model feedback.
- Added `repair_safe` Tool declarations and the independent `RunLimits.max_tool_repair_attempts` budget so corrected-argument repair does not change identical-call retry or step-validation limits.
- Added internal recovery lifecycle events and checkpointed recovery state without exposing raw Tool failures in public result metadata.
- Enforced same-Tool, single-call, new-ID, changed effective-arguments repair; partial-success Tool batches now fail closed instead of replaying side effects through replan.
- Kept strict repair transcripts type-only by default and suppressed overlapping automatic retries after uncancellable synchronous Tool timeouts.
- Preserved third-party strict Policy compatibility by configuring the additive Tool-repair budget through an optional capability hook.

## 0.3.1

- Updated strict Plan-and-Execute terminal policy transitions so exhausted validation, Tool recovery, or replanning marks the current step `failed` instead of leaving it `in_progress`.
- Added configurable, bounded, sanitized `AgentResult.metadata["tool_trace"]` audit summaries with validated invocation arguments, reserved-key protection, and strict checkpoint persistence so operators can identify the Tool actually used without treating `allowed_tools` as execution history.
- Normalized pandas `DataFrame` and other tabular Tool results into JSON-safe records before they are passed to the model or retained by runtime diagnostics.

## 0.3.0

- Made `PlanAndExecutePolicy` a strict PLAN → ACT_TOOL → STEP_RESULT → STEP_VALIDATE → VERIFY → FINALIZE state machine.
- Added stable plan-step IDs, dependencies, completion criteria, allowed Tool scopes, attempt counters, and content-addressed result references.
- Added strict `StepResult`, `StepValidator`, retry, partial replan, and explicit step commit contracts.
- Separated Tool-only ACT requests from schema-only `StepResult` requests for provider and vLLM compatibility.
- Applied the same one-time, Tool-free FINALIZE boundary to text and Pydantic outputs.
- Kept ACT responses out of public conversation history and persisted only the validated FINALIZE response.
- Added public/internal event visibility, `FINAL_DELTA`, strict step lifecycle events, and diagnostic `Agent.stream_all()`.
- Added phase-scoped Skill instructions through the optional `applies-to` frontmatter extension.
- Upgraded checkpoints to schema v3 with strict execution and finalization state; v1 and v2 payloads remain readable.
- Added `LegacyPlanAndExecutePolicy` with a deprecation warning for temporary 0.2 behavior migration.
- Added independent `max_step_attempts` and `max_replans` limits; strict `max_steps` now constrains plan length rather than model-turn count.
- Added `AgentConfig.finalization_mode`; its `structured_only` default preserves 0.2 Standard-policy call counts, while `always` opts text and structured Standard runs into a separate finalizer. Strict Plan-and-Execute always finalizes and rejects `disabled`.
- Persisted the generic finalizer's raw public assistant response to conversation history.
- Documented the runtime's finalization duplicate-suppression boundary and the need for a durable outbox and Tool idempotency for end-to-end exactly-once behavior.

## 0.2.0

- Added official `SKILL.md` compatible Agent Skills packages.
- Added filesystem and in-memory Skill sources with immutable package and catalog digests.
- Added explicit, model-based, and hybrid Skill selection.
- Applied active Skill instructions consistently to PLAN, ACT, and FINALIZE phases.
- Added bounded `references/` and text `assets/` read/search tools.
- Pinned filesystem resource paths and SHA-256 digests, with POSIX `openat`/`O_NOFOLLOW` traversal protection against path races.
- Added Skill-specific catalog, instruction, resource-read, byte, and token limits.
- Added mount-independent filesystem source IDs and enforceable catalog lock files.
- Restricted active runs to the intersection of registered and Skill-declared tools; `ToolAuthorizer` is still enforced at execution.
- Added Skill lifecycle events without logging instruction or resource contents.
- Upgraded checkpoints to schema v2 with v1 read compatibility and pinned Skill state.
- Added `moduagent skills init|validate|inspect|lock`.
- Added PyYAML as a runtime dependency.
- Added Python 3.10-3.13 CI coverage.
- Bundled scripts remain non-executable in this release.

## 0.1.1

- Initial public release of the ModuAgent runtime.
