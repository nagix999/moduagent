"""CLI composition for the Docling/Gemma/BGE-M3/Milvus example."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Sequence

from moduagent import (
    AgentRunError,
    ConsoleEventSink,
    InMemoryDiagnosticSink,
    VLLMClient,
)

from .agent import format_management_failure, run_management_request
from .artifacts import ArtifactStore
from .backends import (
    DEFAULT_GEMMA_MODEL,
    DoclingServeClient,
    OfficePageCaptureRenderer,
    VLLMEmbeddingClient,
    VLLMEnrichmentClient,
    VLLMLayoutRefinementClient,
    build_pipeline_fingerprint,
)
from .catalog import ManifestCatalog
from .chunking import ChunkingConfig
from .diagnostics import PipelineExecutionLog
from .environment import environment_secret, load_environment_file
from .models import RAGIndexError
from .pipeline import ManagerConfig, RAGIndexManager
from .restructure import RESTRUCTURING_FINGERPRINT
from .stores import MilvusStore, VectorStoreError
from .supervisor import (
    ContinuousIngestionSupervisor,
    SupervisorPolicy,
    SupervisorReport,
    SupervisorStateStore,
)
from .validation import (
    evaluate_validation_corpus,
    generate_validation_corpus,
    load_validation_corpus,
    run_validation_lifecycle,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    values = list(argv) if argv is not None else sys.argv[1:]
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--env-file",
        default=os.getenv("RAG_ENV_FILE", ".env"),
    )
    environment_args, _ = bootstrap.parse_known_args(values)
    explicitly_selected = (
        any(
            value == "--env-file" or value.startswith("--env-file=") for value in values
        )
        or "RAG_ENV_FILE" in os.environ
    )
    load_environment_file(
        environment_args.env_file,
        required=explicitly_selected,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        default=environment_args.env_file,
        help="UTF-8 environment file loaded before CLI defaults (default: .env)",
    )
    parser.add_argument(
        "--documents",
        default=os.getenv("RAG_DOCUMENT_ROOT"),
        help="application-approved document directory (or RAG_DOCUMENT_ROOT)",
    )
    parser.add_argument(
        "--state-dir",
        default=os.getenv("RAG_STATE_DIR", ".rag-index-manager"),
        help="manifest and resumable artifact directory",
    )
    parser.add_argument(
        "--kb-id",
        default=os.getenv("RAG_KB_ID", "corporate-assistant"),
        help="stable knowledge-base identifier",
    )
    parser.add_argument(
        "--request",
        default=None,
        help="natural-language status, plan, sync, rebuild, or rollback request",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="continuously detect stable file changes and apply incremental sync",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="expose write Tools; without it all change requests are dry-run only",
    )
    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=int(os.getenv("RAG_EMBEDDING_DIMENSION", "1024")),
        help="expected BGE-M3 dense dimension (default: 1024)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="stream content-free pipeline stages and print safe failure diagnostics",
    )
    parser.add_argument(
        "--log-format",
        choices=("pretty", "json"),
        default=os.getenv("RAG_LOG_FORMAT", "pretty"),
        help="verbose progress rendering format (default: pretty)",
    )
    parser.add_argument(
        "--log-language",
        choices=("ko", "en"),
        default=os.getenv("RAG_LOG_LANGUAGE", "ko"),
        help="pretty progress label language (default: ko)",
    )
    parser.add_argument(
        "--validate-generated",
        type=int,
        metavar="DOCUMENTS",
        default=None,
        help=(
            "run a destructive synthetic-corpus lifecycle validation with this many "
            "documents; --documents must be an empty or owned validation directory"
        ),
    )
    parser.add_argument(
        "--validation-top-k",
        type=int,
        default=5,
        help="source-level retrieval cutoff for generated validation (default: 5)",
    )
    parser.add_argument(
        "--evaluate-generated",
        action="store_true",
        help=(
            "evaluate an already published generated corpus without rebuilding or "
            "mutating it"
        ),
    )
    parser.add_argument(
        "--validation-details",
        action="store_true",
        help=("include opaque per-case rankings; valid only with --evaluate-generated"),
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=float(os.getenv("RAG_WATCH_POLL_SECONDS", "5")),
        help="watch polling interval (default: 5)",
    )
    parser.add_argument(
        "--stability-seconds",
        type=float,
        default=float(os.getenv("RAG_WATCH_STABILITY_SECONDS", "15")),
        help="unchanged time required before automatic sync (default: 15)",
    )
    parser.add_argument(
        "--reconcile-seconds",
        type=float,
        default=float(os.getenv("RAG_WATCH_RECONCILE_SECONDS", "300")),
        help="periodic full reconciliation interval (default: 300)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=int(os.getenv("RAG_WATCH_MAX_ATTEMPTS", "5")),
        help="automatic attempts before quarantining one unchanged snapshot",
    )
    parser.add_argument(
        "--retry-initial-seconds",
        type=float,
        default=float(os.getenv("RAG_WATCH_RETRY_INITIAL_SECONDS", "5")),
        help="initial automatic retry delay (default: 5)",
    )
    parser.add_argument(
        "--retry-max-seconds",
        type=float,
        default=float(os.getenv("RAG_WATCH_RETRY_MAX_SECONDS", "300")),
        help="maximum automatic retry delay (default: 300)",
    )
    return parser.parse_args(values)


async def run_cli(args: argparse.Namespace) -> str:
    if not args.documents:
        raise ValueError("--documents or RAG_DOCUMENT_ROOT is required")
    validating = args.validate_generated is not None
    evaluating = args.evaluate_generated
    if validating and evaluating:
        raise ValueError(
            "--validate-generated and --evaluate-generated are mutually exclusive"
        )
    if args.validation_details and not evaluating:
        raise ValueError("--validation-details requires --evaluate-generated")
    if (validating or evaluating) and (args.watch or args.request):
        raise ValueError(
            "generated validation cannot be combined with --watch or --request"
        )
    if args.watch and args.request:
        raise ValueError("--watch and --request are separate operating modes")
    if not args.watch and not args.request and not validating and not evaluating:
        raise ValueError(
            "--request is required unless --watch or --validate-generated is selected"
        )
    document_root = Path(args.documents).expanduser()
    state_root = Path(args.state_dir).expanduser()
    safe_kb = _milvus_identifier(args.kb_id)
    do_ocr = _environment_bool("DOCLING_SERVE_DO_OCR", default=False)
    log_format = getattr(args, "log_format", "pretty")
    log_language = getattr(args, "log_language", "ko")
    execution_log = None
    if args.verbose or args.watch:
        execution_log = PipelineExecutionLog.console(
            stream=sys.stderr,
            include_timestamp=True,
            output_format=log_format,
            language=log_language,
        )
    agent_event_sink = (
        ConsoleEventSink(
            stream=sys.stderr,
            output_format=log_format,
            detail="summary",
            language=log_language,
            include_timestamp=True,
        )
        if args.verbose
        else None
    )
    diagnostic_sink = InMemoryDiagnosticSink(max_records=100) if args.verbose else None
    supervisor_state_store = SupervisorStateStore(state_root / "supervisor-state.json")
    validation_corpus = (
        generate_validation_corpus(
            document_root,
            kb_id=args.kb_id,
            document_count=args.validate_generated,
        )
        if validating
        else load_validation_corpus(document_root)
        if evaluating
        else None
    )
    gemma_api_key = environment_secret("RAG_VLLM_API_KEY")
    embedding_api_key = environment_secret("RAG_EMBEDDING_API_KEY")
    docling_api_key = environment_secret("DOCLING_SERVE_API_KEY")
    milvus_token = environment_secret("RAG_MILVUS_TOKEN")

    parser = DoclingServeClient(
        api_key=docling_api_key,
        do_ocr=do_ocr,
        generate_page_images=True,
    )
    office_renderer = (
        OfficePageCaptureRenderer()
        if _environment_bool("RAG_OFFICE_PAGE_CAPTURE", default=False)
        else None
    )
    refiner = VLLMLayoutRefinementClient(
        api_key=gemma_api_key,
        allow_exclusions=False,
        page_capture_renderer=office_renderer,
    )
    enricher = VLLMEnrichmentClient(api_key=gemma_api_key)
    embedder = VLLMEmbeddingClient(api_key=embedding_api_key)
    vector_store = MilvusStore(
        token=milvus_token,
        alias=f"{safe_kb}_active",
        collection_prefix=f"{safe_kb}_chunks",
    )
    chunking = ChunkingConfig()
    pipeline = build_pipeline_fingerprint(
        parser=parser,
        enricher=enricher,
        embedder=embedder,
        restructuring_revision=RESTRUCTURING_FINGERPRINT,
        chunking_revision=chunking.fingerprint,
        indexing_revision=vector_store.fingerprint,
        embedding_dimension=args.embedding_dimension,
        refiner=refiner,
    )
    catalog = ManifestCatalog(state_root / "manifest.sqlite3")
    try:
        artifacts = ArtifactStore(state_root / "artifacts")
        manager = RAGIndexManager(
            config=ManagerConfig(
                document_root=document_root,
                kb_id=args.kb_id,
                embedding_dimension=args.embedding_dimension,
                chunking=chunking,
            ),
            pipeline=pipeline,
            catalog=catalog,
            artifacts=artifacts,
            parser=parser,
            refiner=refiner,
            enricher=enricher,
            embedder=embedder,
            vector_store=vector_store,
            execution_log=execution_log,
            write_lease=supervisor_state_store.operation_lease,
        )
        async with parser, refiner, enricher, embedder, vector_store:
            if validation_corpus is not None:
                report = (
                    await run_validation_lifecycle(
                        manager,
                        validation_corpus,
                        top_k=args.validation_top_k,
                    )
                    if validating
                    else await evaluate_validation_corpus(
                        manager,
                        validation_corpus,
                        top_k=args.validation_top_k,
                    )
                )
                payload = _validation_report_payload(
                    report,
                    validating=validating,
                    include_results=args.validation_details,
                )
                return json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
            if args.watch:
                supervisor = ContinuousIngestionSupervisor(
                    manager,
                    supervisor_state_store,
                    policy=SupervisorPolicy(
                        poll_interval_seconds=args.poll_seconds,
                        stability_window_seconds=args.stability_seconds,
                        full_reconcile_interval_seconds=args.reconcile_seconds,
                        max_attempts=args.max_attempts,
                        initial_retry_seconds=args.retry_initial_seconds,
                        max_retry_seconds=args.retry_max_seconds,
                    ),
                    event_sink=_console_supervisor_report,
                )
                await supervisor.run_forever()
                return json.dumps(
                    {"status": "stopped", "kb_id": args.kb_id},
                    ensure_ascii=False,
                )

            management_model = VLLMClient(
                base_url=os.getenv(
                    "RAG_VLLM_BASE_URL",
                    os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
                ),
                model=os.getenv("RAG_TEXT_MODEL", DEFAULT_GEMMA_MODEL),
                api_key=gemma_api_key,
                timeout=90,
                default_options={"temperature": 0, "max_tokens": 1_024},
            )
            async with management_model:
                assert args.request is not None
                try:
                    response = await run_management_request(
                        management_model,
                        manager,
                        args.request,
                        allow_writes=args.apply,
                        event_sink=agent_event_sink,
                        diagnostic_sink=diagnostic_sink,
                        supervisor_state_provider=lambda: supervisor_state_store.load(
                            kb_id=manager.config.kb_id,
                            pipeline_digest=manager.pipeline.digest,
                        ),
                    )
                except AgentRunError as exc:
                    if execution_log is not None:
                        print(
                            format_management_failure(
                                exc,
                                execution_log=execution_log,
                                diagnostic_sink=diagnostic_sink,
                            ),
                            file=sys.stderr,
                        )
                    raise
                return response.model_dump_json(indent=2)
    finally:
        catalog.close()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        print(asyncio.run(run_cli(args)))
    except (
        AgentRunError,
        RAGIndexError,
        VectorStoreError,
        ValueError,
        TypeError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("continuous ingestion stopped", file=sys.stderr)
        return 130
    return 0


def _milvus_identifier(value: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError("kb-id must be non-empty and bounded")
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not normalized or not normalized[0].isalpha():
        normalized = "kb_" + normalized
    normalized = normalized[:96]
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", normalized):
        raise ValueError("kb-id cannot be mapped to a Milvus-safe name")
    return normalized


def _environment_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _console_supervisor_report(report: SupervisorReport) -> None:
    payload = json.dumps(
        report.to_dict(),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    print(f"rag_ingestion_supervisor {payload}", file=sys.stderr, flush=True)


def _validation_report_payload(
    report: object,
    *,
    validating: bool,
    include_results: bool,
) -> dict[str, object]:
    """Keep lifecycle and quality-only report signatures deliberately separate."""

    if type(validating) is not bool or type(include_results) is not bool:
        raise TypeError("validation serialization flags must be bool values")
    serializer = getattr(report, "as_dict", None)
    if not callable(serializer):
        raise TypeError("validation report must provide as_dict")
    payload = (
        serializer() if validating else serializer(include_results=include_results)
    )
    if not isinstance(payload, dict):
        raise TypeError("validation report serialization must return a dict")
    return payload


__all__ = ["main", "parse_args", "run_cli"]
