from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from io import StringIO
from pathlib import Path
from types import ModuleType

import pytest

from moduagent import ConsoleEventSink, FinishReason, InMemoryDiagnosticSink, VLLMClient


ROOT = Path(__file__).resolve().parents[2]


def _live_environment() -> tuple[str, str, str | None]:
    if os.getenv("MODUAGENT_RUN_LIVE_CONSOLE_EXAMPLES", "").strip() != "1":
        pytest.skip("set MODUAGENT_RUN_LIVE_CONSOLE_EXAMPLES=1 to run console examples")
    base_url = os.getenv("VLLM_BASE_URL", "").strip()
    model = os.getenv("VLLM_MODEL", "").strip()
    if not base_url or not model:
        pytest.skip("VLLM_BASE_URL and VLLM_MODEL are required")
    return base_url, model, os.getenv("VLLM_API_KEY", "").strip() or None


def _client(*, max_tokens: int = 2048) -> VLLMClient:
    base_url, model, api_key = _live_environment()
    return VLLMClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=90,
        default_options={"temperature": 0, "max_tokens": max_tokens},
    )


def _load(relative_path: str) -> ModuleType:
    path = ROOT / relative_path
    name = f"_moduagent_live_console_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load example: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sink() -> tuple[ConsoleEventSink, StringIO]:
    stream = StringIO()
    return (
        ConsoleEventSink(
            stream=stream,
            detail="detailed",
            color=False,
        ),
        stream,
    )


def _assert_console(stream: StringIO, *, tools: tuple[str, ...] = ()) -> None:
    rendered = stream.getvalue()
    assert "Agent run started" in rendered
    assert "Agent run completed" in rendered
    for tool_name in tools:
        assert f"Running tool · {tool_name}" in rendered
    assert "\x1b[" not in rendered


def test_beginner_examples_render_live_console_progress(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _client(max_tokens=1024) as model:
            hello = _load("examples/01_hello_agent.py")
            sink, console = _sink()
            result = await hello.build_agent(model, event_sink=sink).run(
                "Explain one benefit of an AI agent in one sentence."
            )
            assert result.finish_reason is FinishReason.COMPLETED, result.explain()
            _assert_console(console)

            orders = _load("examples/02_use_a_tool.py")
            sink, console = _sink()
            result = await orders.build_agent(model, event_sink=sink).run(
                "Where is order ORD-1001?"
            )
            assert result.finish_reason is FinishReason.COMPLETED, result.explain()
            assert result.tool_trace[0]["tool_name"] == "lookup_order"
            _assert_console(console, tools=("lookup_order",))
            assert "ORD-1001" not in console.getvalue()

            structured = _load("examples/03_structured_output.py")
            sink, console = _sink()
            result = await structured.build_agent(model, event_sink=sink).run(
                "I was charged twice, but the service remains available."
            )
            assert result.finish_reason is FinishReason.COMPLETED, result.explain()
            assert isinstance(result.output, structured.TicketTriage)
            _assert_console(console)

            report = _load("examples/04_report_automation.py")
            report.ARTIFACT_DIR = tmp_path / "artifacts"
            sink, console = _sink()
            result = await report.build_agent(model, event_sink=sink).run(
                "Report all-region sales from 2025-01-01 through 2025-03-31."
            )
            assert result.finish_reason is FinishReason.COMPLETED, result.explain()
            assert isinstance(result.output, report.SalesReport)
            _assert_console(console, tools=("query_sales", "plot_graph"))

            debug = _load("examples/05_debug_a_run.py")
            diagnostics = InMemoryDiagnosticSink(max_records=20)
            sink, console = _sink()
            result = await debug.build_agent(
                model,
                diagnostics,
                event_sink=sink,
            ).run("Is the orders service healthy?")
            assert result.finish_reason is FinishReason.COMPLETED, result.explain()
            _assert_console(console, tools=("get_service_status",))

    asyncio.run(scenario())


def test_specialized_examples_render_live_console_progress() -> None:
    async def scenario() -> None:
        async with _client(max_tokens=4096) as model:
            waf = _load("examples/06_waf_log_analysis.py")
            sink, console = _sink()
            result = await waf.analyze_waf_log(
                model,
                waf.SAMPLE_WAF_LOG,
                event_id=waf.SAMPLE_EVENT_ID,
                event_sink=sink,
            )
            assert isinstance(result, waf.WAFAnalysis)
            _assert_console(console, tools=tuple(sorted(waf.WAF_TOOL_NAMES)))

            documents = _load("examples/13_document_qa_and_report.py")
            sink, console = _sink()
            intent = await documents.classify_intent(
                model,
                "이 문서의 핵심 내용을 한 문단으로 요약해줘.",
                event_sink=sink,
            )
            assert intent.mode in {"question", "report"}
            _assert_console(console)

            document_id = "doc_0123456789abcdef0123"
            evidence_id = "ev_0123456789abcdef01234567"
            source_digest = "a" * 64
            corpus = documents.DocumentCorpus(
                documents=(
                    documents.DocumentRecord(
                        document_id=document_id,
                        filename="policy.pdf",
                        source_sha256=source_digest,
                        evidence_count=1,
                    ),
                ),
                evidence=(
                    documents.EvidenceRecord(
                        evidence_id=evidence_id,
                        document_id=document_id,
                        filename="policy.pdf",
                        source_sha256=source_digest,
                        quote=(
                            "운영 변경은 배포 전에 담당 관리자 한 명의 승인을 "
                            "받아야 한다."
                        ),
                        quote_basis="docling_text",
                        quote_truncated=False,
                        self_ref="#/texts/0",
                        section=("변경 승인",),
                        page_no=3,
                        bbox=None,
                        charspan=(0, 38),
                        locations=(
                            documents.EvidenceLocation(
                                page_no=3,
                                bbox=None,
                                charspan=(0, 38),
                            ),
                        ),
                        line_start=12,
                        line_end=12,
                        line_basis="docling_markdown",
                    ),
                ),
            )
            sink, console = _sink()
            markdown = await documents.run_question(
                model,
                corpus,
                "운영 변경에는 어떤 승인이 필요한가?",
                event_sink=sink,
            )
            assert evidence_id in markdown
            assert "3페이지" in markdown
            _assert_console(
                console,
                tools=("list_documents", "search_evidence", "read_evidence"),
            )

            controls = _load("examples/20_production_controls.py")
            store = controls.InMemoryApprovalStore()
            sink, console = _sink()
            agent = controls.build_agent(
                model,
                approval_store=store,
                event_sink=sink,
            )
            result = await agent.run(
                "Approve CHG-2048 if every verified control passes.",
                session_id="live-console-change",
                user_context={
                    "user_id": "operator-17",
                    "roles": ["change_approver"],
                    "authorized_change_id": "CHG-2048",
                    "authorized_tenant_id": "tenant-acme",
                    "approval_idempotency_key": "approval:CHG-2048:live-console",
                },
            )
            assert result.finish_reason is FinishReason.COMPLETED, result.explain()
            _assert_console(
                console,
                tools=("get_change_request", "approve_change"),
            )

    asyncio.run(scenario())
