from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from moduagent import InMemoryDiagnosticSink


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
BEGINNER_EXAMPLES = tuple(
    EXAMPLES_DIR / name
    for name in (
        "01_hello_agent.py",
        "02_use_a_tool.py",
        "03_structured_output.py",
        "04_report_automation.py",
        "05_debug_a_run.py",
        "06_waf_log_analysis.py",
    )
)


class NoCallModel:
    async def complete(self, request: Any) -> Any:
        del request
        raise AssertionError("the example must not call a model during import or setup")


def _load_example(path: Path) -> ModuleType:
    module_name = f"_moduagent_example_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load example: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_beginner_examples_compile_import_and_use_environment_client() -> None:
    for path in BEGINNER_EXAMPLES:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        module = _load_example(path)

        assert callable(module.build_agent)
        assert "VLLMClient.from_env(" in source
        assert "async with VLLMClient.from_env(" in source
        assert '"max_tokens"' in source
        assert "t62y46bwfim0hq" not in source
        assert "runpod-vllm-token" not in source


def test_beginner_builders_need_no_network_access() -> None:
    modules = [_load_example(path) for path in BEGINNER_EXAMPLES]
    model = NoCallModel()

    agents = [module.build_agent(model) for module in modules[:4]]
    diagnostics = InMemoryDiagnosticSink()
    debug_agent = modules[4].build_agent(model, diagnostics)
    waf_agent = modules[5].build_agent(model)

    assert agents[0].inspect().execution_profile.kind == "standard"
    assert [tool.name for tool in agents[1].tool_registry] == ["lookup_order"]
    assert agents[2].inspect().output_contract["structured"] is True
    assert [tool.name for tool in agents[3].tool_registry] == [
        "query_sales",
        "plot_graph",
    ]
    assert debug_agent.config.tool_trace_mode == "summary"
    assert debug_agent.diagnostic_reporter.sink is diagnostics
    assert waf_agent.inspect().name == "waf-log-analyzer-v01"
    assert waf_agent.inspect().output_contract["structured"] is True
    assert waf_agent.inspect().output_contract["staged_finalization"] is True
    assert waf_agent.config.model_options["parallel_tool_calls"] is False
    assert waf_agent.config.limits.parallel_tool_calls is False
    assert [tool.name for tool in waf_agent.tool_registry] == [
        "analyze_payload_encoding",
        "get_waf_rule_context",
        "get_route_context",
        "get_correlated_app_outcome",
        "summarize_related_events",
        "lookup_threat_intel",
    ]


def test_order_lookup_tool_returns_only_known_shipping_data() -> None:
    module = _load_example(BEGINNER_EXAMPLES[1])

    found = asyncio.run(module.lookup_order.invoke({"order_id": " ord-1001 "}))
    missing = asyncio.run(module.lookup_order.invoke({"order_id": "ORD-9999"}))

    assert found == {
        "order_id": "ORD-1001",
        "status": "shipped",
        "estimated_delivery": "2026-08-01",
    }
    assert missing == {"order_id": "ORD-9999", "status": "not_found"}


def test_report_tools_filter_data_and_write_a_real_svg(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_example(BEGINNER_EXAMPLES[3])
    monkeypatch.setattr(module, "ARTIFACT_DIR", tmp_path)

    query_result = asyncio.run(
        module.query_sales.invoke(
            {
                "start_date": "2025-01-01",
                "end_date": "2025-03-31",
                "region": "north",
            }
        )
    )

    assert query_result == {
        "period": "2025-01-01 to 2025-03-31",
        "region": "north",
        "rows": [
            {"month": "2025-01", "sales": 1200.0},
            {"month": "2025-02", "sales": 1500.0},
            {"month": "2025-03", "sales": 1700.0},
        ],
        "total_sales": 4400.0,
    }

    plot_result = asyncio.run(
        module.plot_graph.invoke(
            {
                "labels": ["2025-01", "2025-02", "2025-03"],
                "values": [1200.0, 1500.0, 1700.0],
                "title": "North <sales> & growth",
            }
        )
    )
    chart_path = Path(plot_result["chart_path"])
    svg = chart_path.read_text(encoding="utf-8")

    assert plot_result == {
        "chart_path": str(tmp_path / "sales_report.svg"),
        "points": 3,
    }
    assert chart_path.parent == tmp_path
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert svg.endswith("</svg>")
    assert "North &lt;sales&gt; &amp; growth" in svg
    assert "<rect" in svg


def test_debug_tool_is_deterministic_and_has_no_import_time_logs(
    caplog,
) -> None:
    module = _load_example(BEGINNER_EXAMPLES[4])

    result = asyncio.run(module.get_service_status.invoke({"service": "orders"}))

    assert result == {
        "service": "orders",
        "status": "degraded",
        "updated_at": "2026-07-30T09:05:00Z",
    }
    assert not caplog.records
