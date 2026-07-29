# Changelog

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
