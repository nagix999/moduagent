# Intermediate examples

These examples target ModuAgent `0.5.2`. Complete the beginner path in
[`README.md`](README.md) first, then follow `10` → `11` → `12`. Each example is
standalone and uses deterministic application-owned data, so you can replace
the sample Tools with your own integrations without changing the Agent pattern.

## Install and configure

Install the exact version:

```bash
python -m pip install "moduagent==0.5.2"
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

## Learning path

| Order | Scenario | Tools | Main lesson | Safety boundary |
| --- | --- | ---: | --- | --- |
| 10 | [Incident investigation](10_incident_investigation.py) | 5 | Correlate parallel evidence and validate a nested report | Read-only investigation; no mitigation, rollback, or deployment |
| 11 | [Customer case resolution](11_customer_case_resolution.py) | 5 | Pass verified values through sequential lookups and calculations | Advisory proposal only; no refund, return, message, or case update |
| 12 | [Release readiness](12_release_readiness.py) | 5 | Apply several evidence gates and enforce a consistent decision | Ship/hold recommendation only; no deployment or system change |

All three use Standard execution, structured Pydantic output, bounded
`RunLimits`, summary-only Tool traces, capped model output, and an async model
client context manager.

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

## Live assertions

The normal test suite imports and tests these examples without network access.
To run all three against a configured vLLM endpoint:

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
