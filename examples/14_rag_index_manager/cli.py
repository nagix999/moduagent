"""CLI composition for the Docling/Gemma/BGE-M3/Milvus example."""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Sequence

from moduagent import AgentRunError, InMemoryDiagnosticSink, VLLMClient

from .agent import format_management_failure, run_management_request
from .artifacts import ArtifactStore
from .backends import (
    DEFAULT_GEMMA_MODEL,
    DoclingServeClient,
    VLLMEmbeddingClient,
    VLLMEnrichmentClient,
    VLLMLayoutRefinementClient,
    build_pipeline_fingerprint,
)
from .catalog import ManifestCatalog
from .chunking import ChunkingConfig
from .diagnostics import PipelineExecutionLog
from .models import RAGIndexError
from .pipeline import ManagerConfig, RAGIndexManager
from .restructure import RESTRUCTURING_FINGERPRINT
from .stores import MilvusStore, VectorStoreError


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        required=True,
        help="natural-language status, plan, sync, rebuild, or rollback request",
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
    return parser.parse_args(argv)


async def run_cli(args: argparse.Namespace) -> str:
    if not args.documents:
        raise ValueError("--documents or RAG_DOCUMENT_ROOT is required")
    document_root = Path(args.documents).expanduser()
    state_root = Path(args.state_dir).expanduser()
    safe_kb = _milvus_identifier(args.kb_id)
    do_ocr = _environment_bool("DOCLING_SERVE_DO_OCR", default=False)
    execution_log = (
        PipelineExecutionLog.console(stream=sys.stderr, include_timestamp=True)
        if args.verbose
        else None
    )
    diagnostic_sink = InMemoryDiagnosticSink(max_records=100) if args.verbose else None

    parser = DoclingServeClient(do_ocr=do_ocr, generate_page_images=True)
    refiner = VLLMLayoutRefinementClient(allow_exclusions=False)
    enricher = VLLMEnrichmentClient()
    embedder = VLLMEmbeddingClient()
    vector_store = MilvusStore(
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
        )
        management_model = VLLMClient(
            base_url=os.getenv(
                "RAG_VLLM_BASE_URL",
                os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            ),
            model=os.getenv("RAG_TEXT_MODEL", DEFAULT_GEMMA_MODEL),
            api_key=os.getenv("RAG_VLLM_API_KEY", os.getenv("VLLM_API_KEY")),
            timeout=90,
            default_options={"temperature": 0, "max_tokens": 1_024},
        )
        async with parser, refiner, enricher, embedder, vector_store, management_model:
            try:
                response = await run_management_request(
                    management_model,
                    manager,
                    args.request,
                    allow_writes=args.apply,
                    diagnostic_sink=diagnostic_sink,
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


__all__ = ["main", "parse_args", "run_cli"]
