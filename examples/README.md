# ModuAgent examples

[한국어 번역: 중급 예제](INTERMEDIATE.ko.md) ·
[한국어 번역: 프로덕션 제어](PRODUCTION.ko.md)

These examples grow one concept at a time. Start with `01` and run them in
order; each file is standalone.

## Before you start

These examples target ModuAgent `0.5.3`. Install the exact version:

```bash
python -m pip install "moduagent==0.5.3"
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

Run them from the repository root, for example:

```bash
python examples/10_incident_investigation.py
```

All three use Standard execution, structured output, bounded run/model/Tool
budgets, summary Tool traces, and read-only or advisory safety boundaries. See
[`INTERMEDIATE.md`](INTERMEDIATE.md) for each example's Tool count, teaching
goal, safety boundary, execution command, and debugging checklist.

To run the opt-in live assertions for all three intermediate scenarios:

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
