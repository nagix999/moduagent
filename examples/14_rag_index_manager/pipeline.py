"""Deterministic incremental orchestration around replaceable AI backends."""

from __future__ import annotations

import asyncio
import os
import secrets
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from .artifacts import ArtifactError, ArtifactStore
from .backends import (
    BlockEnricher,
    DoclingParser,
    EmbeddingBatch,
    TextEmbedder,
    extract_page_captures,
)
from .catalog import ManifestCatalog
from .chunking import ChunkingConfig, chunk_blocks
from .diagnostics import PipelineExecutionLog
from .models import (
    BlockEnrichment,
    ChangeKind,
    Chunk,
    LayoutPatch,
    LayoutRefinementPolicy,
    LayoutRefiner,
    ManifestDocument,
    PageCapture,
    PipelineFingerprint,
    ProcessingStage,
    RAGIndexError,
    SourceDocument,
    StructuredBlock,
    SyncAction,
    SyncPlan,
    apply_layout_patches,
)
from .planner import plan_incremental_sync
from .restructure import restructure_docling_document
from .scanner import ScanPolicy, scan_document_directory
from .stores import CopyResult, IndexValidation, StagingGeneration, VectorStore


class PipelineError(RAGIndexError):
    """An index run failed before a consistent generation could be published."""


@dataclass(frozen=True, slots=True)
class ManagerConfig:
    """Application-owned scope and limits; no field is model-controlled."""

    document_root: Path
    kb_id: str = "corporate-assistant"
    embedding_dimension: int = 1024
    scan_policy: ScanPolicy = field(default_factory=ScanPolicy)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    enrichment_batch_size: int = 128
    embedding_request_size: int = 512
    max_action_details: int = 100
    layout_refinement_policy: LayoutRefinementPolicy = field(
        default_factory=LayoutRefinementPolicy
    )

    def __post_init__(self) -> None:
        if not isinstance(self.document_root, Path):
            raise TypeError("document_root must be a pathlib.Path")
        if not isinstance(self.kb_id, str) or not self.kb_id or len(self.kb_id) > 128:
            raise ValueError("kb_id must be non-empty and bounded")
        if (
            type(self.embedding_dimension) is not int
            or not 1 <= self.embedding_dimension <= 65_536
        ):
            raise ValueError("embedding_dimension must be between one and 65536")
        if not isinstance(self.scan_policy, ScanPolicy):
            raise TypeError("scan_policy must be a ScanPolicy")
        if not isinstance(self.chunking, ChunkingConfig):
            raise TypeError("chunking must be a ChunkingConfig")
        if not isinstance(self.layout_refinement_policy, LayoutRefinementPolicy):
            raise TypeError("layout_refinement_policy must be a LayoutRefinementPolicy")
        if (
            type(self.enrichment_batch_size) is not int
            or not 1 <= self.enrichment_batch_size <= 512
        ):
            raise ValueError("enrichment_batch_size must be between one and 512")
        if (
            type(self.embedding_request_size) is not int
            or not 1 <= self.embedding_request_size <= 10_000
        ):
            raise ValueError("embedding_request_size must be between one and 10000")
        if (
            type(self.max_action_details) is not int
            or not 0 <= self.max_action_details <= 1_000
        ):
            raise ValueError("max_action_details must be between zero and 1000")


@dataclass(frozen=True, slots=True)
class ActionDetail:
    source_id: str
    relative_path: str
    change: str
    start_stage: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "relative_path": self.relative_path,
            "change": self.change,
            "start_stage": self.start_stage,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SyncReport:
    operation: Literal["preview", "sync", "rebuild", "rollback"]
    status: Literal["dry_run", "noop", "published", "rolled_back"]
    kb_id: str
    generation_id: str | None
    previous_generation_id: str | None
    document_count: int
    chunk_count: int
    new_count: int = 0
    modified_count: int = 0
    pipeline_changed_count: int = 0
    deleted_count: int = 0
    unchanged_count: int = 0
    actions: tuple[ActionDetail, ...] = ()
    details_truncated: bool = False
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "status": self.status,
            "kb_id": self.kb_id,
            "generation_id": self.generation_id,
            "previous_generation_id": self.previous_generation_id,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "new_count": self.new_count,
            "modified_count": self.modified_count,
            "pipeline_changed_count": self.pipeline_changed_count,
            "deleted_count": self.deleted_count,
            "unchanged_count": self.unchanged_count,
            "actions": [item.as_dict() for item in self.actions],
            "details_truncated": self.details_truncated,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class IndexStatus:
    kb_id: str
    manifest_generation_id: str | None
    vector_generation_id: str | None
    consistent: bool
    document_count: int
    chunk_count: int
    rollback_candidates: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kb_id": self.kb_id,
            "manifest_generation_id": self.manifest_generation_id,
            "vector_generation_id": self.vector_generation_id,
            "consistent": self.consistent,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "rollback_candidates": list(self.rollback_candidates),
        }


class RAGIndexManager:
    """Own one bounded document root and one Milvus alias lifecycle."""

    def __init__(
        self,
        *,
        config: ManagerConfig,
        pipeline: PipelineFingerprint,
        catalog: ManifestCatalog,
        artifacts: ArtifactStore,
        parser: DoclingParser,
        enricher: BlockEnricher,
        embedder: TextEmbedder,
        vector_store: VectorStore,
        refiner: LayoutRefiner | None = None,
        execution_log: PipelineExecutionLog | None = None,
        write_lease: Callable[[], Any] | None = None,
    ) -> None:
        if not isinstance(config, ManagerConfig):
            raise TypeError("config must be a ManagerConfig")
        if not isinstance(pipeline, PipelineFingerprint):
            raise TypeError("pipeline must be a PipelineFingerprint")
        if not isinstance(catalog, ManifestCatalog):
            raise TypeError("catalog must be a ManifestCatalog")
        if not isinstance(artifacts, ArtifactStore):
            raise TypeError("artifacts must be an ArtifactStore")
        for value, method, name in (
            (parser, "convert", "parser"),
            (enricher, "enrich", "enricher"),
            (embedder, "embed", "embedder"),
            (vector_store, "create_staging", "vector_store"),
        ):
            if not callable(getattr(value, method, None)):
                raise TypeError(f"{name} does not implement its required contract")
        if parser.fingerprint != pipeline.parser:
            raise ValueError("parser fingerprint does not match the pipeline")
        if enricher.fingerprint != pipeline.enrichment:
            raise ValueError("enricher fingerprint does not match the pipeline")
        if embedder.fingerprint != pipeline.embedding:
            raise ValueError("embedder fingerprint does not match the pipeline")
        if vector_store.fingerprint != pipeline.indexing:
            raise ValueError("vector store fingerprint does not match the pipeline")
        if config.chunking.fingerprint != pipeline.chunking:
            raise ValueError("chunking configuration does not match the pipeline")
        if pipeline.layout_refinement is not None:
            if refiner is None or not callable(getattr(refiner, "refine", None)):
                raise TypeError("refiner is required by the layout refinement pipeline")
            if refiner.fingerprint != pipeline.layout_refinement:
                raise ValueError(
                    "layout refiner fingerprint does not match the pipeline"
                )
        elif refiner is not None:
            raise ValueError("refiner requires a layout_refinement fingerprint")
        if execution_log is not None and not isinstance(
            execution_log,
            PipelineExecutionLog,
        ):
            raise TypeError("execution_log must be a PipelineExecutionLog")
        if write_lease is not None and not callable(write_lease):
            raise TypeError("write_lease must be callable or None")
        if (
            pipeline.embedding_dimension is not None
            and pipeline.embedding_dimension != config.embedding_dimension
        ):
            raise ValueError("embedding dimension does not match the pipeline")
        self.config = config
        self.pipeline = pipeline
        self.catalog = catalog
        self.artifacts = artifacts
        self.parser = parser
        self.refiner = refiner
        self.enricher = enricher
        self.embedder = embedder
        self.vector_store = vector_store
        self.execution_log = execution_log
        self.write_lease = write_lease
        self._lock = asyncio.Lock()
        self._write_entry_lock = asyncio.Lock()

    def _stage(
        self,
        operation: str,
        stage: str,
        *,
        source_id: str | None = None,
        generation_id: str | None = None,
        item_index: int | None = None,
        item_count: int | None = None,
        counts: dict[str, int] | None = None,
    ) -> Any:
        if self.execution_log is None:
            return nullcontext()
        return self.execution_log.stage(
            operation,
            stage,
            source_id=source_id,
            generation_id=generation_id,
            item_index=item_index,
            item_count=item_count,
            counts=counts,
        )

    async def status(self) -> IndexStatus:
        with self._stage("status", "run"):
            async with self._lock:
                with self._stage("status", "load_status"):
                    return await self._status_unlocked()

    async def preview(self, *, force_rebuild: bool = False) -> SyncReport:
        if type(force_rebuild) is not bool:
            raise TypeError("force_rebuild must be a bool")
        operation = "rebuild" if force_rebuild else "preview"
        with self._stage(operation, "run"):
            async with self._lock:
                with self._stage(operation, "consistency_check"):
                    await self._require_consistent_active()
                sources, plan = self._scan_and_plan(
                    force_rebuild=force_rebuild,
                    operation=operation,
                )
                with self._stage(operation, "build_report"):
                    return self._plan_report(
                        plan,
                        operation=operation,
                        status="dry_run",
                        document_count=len(sources),
                        chunk_count=sum(
                            item.chunk_count
                            for item in self.catalog.list_documents(self.config.kb_id)
                        ),
                    )

    async def sync(self, *, force_rebuild: bool = False) -> SyncReport:
        if type(force_rebuild) is not bool:
            raise TypeError("force_rebuild must be a bool")
        operation = "rebuild" if force_rebuild else "sync"
        async with self._write_entry_lock:
            lease = (
                self.write_lease() if self.write_lease is not None else nullcontext()
            )
            with lease:
                with self._stage(operation, "run"):
                    return await self._sync_unlogged(force_rebuild=force_rebuild)

    async def sync_snapshot(
        self,
        sources: Sequence[SourceDocument],
    ) -> SyncReport:
        """Sync one application-verified stable subset from a continuous watcher."""

        source_values = tuple(sources)
        if any(not isinstance(value, SourceDocument) for value in source_values):
            raise TypeError("sources must contain SourceDocument values")
        if any(value.kb_id != self.config.kb_id for value in source_values):
            raise PipelineError(
                "stable source snapshot belongs to another knowledge base"
            )
        expected_root = Path(os.path.abspath(os.fspath(self.config.document_root)))
        if any(value.root != expected_root for value in source_values):
            raise PipelineError(
                "stable source snapshot has an unexpected document root"
            )
        async with self._write_entry_lock:
            lease = (
                self.write_lease() if self.write_lease is not None else nullcontext()
            )
            with lease:
                with self._stage("sync", "run"):
                    return await self._sync_unlogged(
                        force_rebuild=False,
                        source_snapshot=source_values,
                    )

    async def _sync_unlogged(
        self,
        *,
        force_rebuild: bool,
        source_snapshot: tuple[SourceDocument, ...] | None = None,
    ) -> SyncReport:
        if type(force_rebuild) is not bool:
            raise TypeError("force_rebuild must be a bool")
        async with self._lock:
            operation: Literal["sync", "rebuild"] = (
                "rebuild" if force_rebuild else "sync"
            )
            with self._stage(operation, "consistency_check"):
                previous_generation = await self._require_consistent_active()
            sources, plan = self._scan_and_plan(
                force_rebuild=force_rebuild,
                operation=operation,
                source_snapshot=source_snapshot,
            )
            dimension_migration = self._requires_dimension_migration(
                plan,
                has_active_generation=previous_generation is not None,
            )
            if dimension_migration:
                plan = self._force_current_sources_to_embedding(plan)
            if not plan.requiring_work():
                documents = self.catalog.list_documents(self.config.kb_id)
                with self._stage(operation, "build_report"):
                    return self._plan_report(
                        plan,
                        operation=operation,
                        status="noop",
                        generation_id=previous_generation,
                        previous_generation_id=previous_generation,
                        document_count=len(documents),
                        chunk_count=sum(item.chunk_count for item in documents),
                    )
            with self._stage(operation, "validate_plan"):
                if not sources:
                    raise PipelineError(
                        "refusing to publish an empty knowledge base; retain or add a document"
                    )
            generation_id = f"gen_{secrets.token_hex(16)}"
            with self._stage(
                operation,
                "begin_generation",
                generation_id=generation_id,
                counts={"actions": len(plan.actions), "documents": len(sources)},
            ):
                run_id = self.catalog.begin_run(
                    self.config.kb_id,
                    self.pipeline,
                    generation_id=generation_id,
                )
            staging: StagingGeneration | None = None
            index_was_published = False
            try:
                with self._stage(
                    operation,
                    "create_staging",
                    generation_id=generation_id,
                ):
                    staging = await self.vector_store.create_staging(
                        generation_id,
                        self.config.embedding_dimension,
                        self.pipeline,
                        # Copying the complete active collection and then
                        # deleting changed sources is not safe on real Milvus:
                        # a delete issued immediately after a generation copy
                        # can remain temporarily invisible.  Start empty and
                        # copy only sources whose vectors are being retained.
                        copy_from_active=False,
                    )
                action_count = len(plan.actions)
                for item_index, action in enumerate(plan.actions, start=1):
                    action_fields = {
                        "source_id": action.source_id,
                        "generation_id": generation_id,
                        "item_index": item_index,
                        "item_count": action_count,
                    }
                    if action.change is ChangeKind.UNCHANGED:
                        assert action.previous is not None
                        with self._stage(operation, "carry_forward", **action_fields):
                            copied = await self.vector_store.copy_sources_to_staging(
                                staging,
                                (action.source_id,),
                            )
                            self._require_copied_source(copied, action.previous)
                            self.catalog.carry_forward_document(
                                run_id,
                                action.previous,
                            )
                        continue
                    if action.change is ChangeKind.DELETED:
                        with self._stage(operation, "delete_source", **action_fields):
                            self.catalog.mark_deleted(run_id, action.source_id)
                            if previous_generation is not None and not force_rebuild:
                                await self.vector_store.delete_sources(
                                    staging,
                                    (action.source_id,),
                                )
                        continue
                    if (
                        action.start_stage is ProcessingStage.INDEX
                        and action.previous is not None
                        and not force_rebuild
                    ):
                        with self._stage(operation, "reuse_index", **action_fields):
                            copied = await self.vector_store.copy_sources_to_staging(
                                staging,
                                (action.source_id,),
                            )
                            self._require_copied_source(copied, action.previous)
                            self.catalog.carry_forward_document(
                                run_id,
                                action.previous,
                                pipeline=self.pipeline,
                            )
                        continue
                    if action.source is None:
                        with self._stage(
                            operation,
                            "validate_action",
                            **action_fields,
                        ):
                            raise PipelineError("work item has no scanned source")
                    if action.previous is not None and not force_rebuild:
                        with self._stage(
                            operation,
                            "replace_source",
                            **action_fields,
                        ):
                            await self.vector_store.delete_sources(
                                staging,
                                (action.source_id,),
                            )
                    chunks = await self._materialize_chunks(
                        action,
                        operation=operation,
                        generation_id=generation_id,
                        item_index=item_index,
                        item_count=action_count,
                    )
                    with self._stage(
                        operation,
                        "record_manifest",
                        counts={"chunks": len(chunks)},
                        **action_fields,
                    ):
                        self.catalog.record_document(
                            run_id,
                            action.source,
                            self.pipeline,
                            chunk_count=len(chunks),
                        )
                        self.catalog.record_chunks(run_id, chunks)
                    await self._embed_and_upsert(
                        staging,
                        chunks,
                        operation=operation,
                        source_id=action.source_id,
                        generation_id=generation_id,
                    )

                with self._stage(
                    operation,
                    "validate_staging",
                    generation_id=generation_id,
                ):
                    expected_ids = self.catalog.list_chunk_ids(
                        self.config.kb_id,
                        generation_id=generation_id,
                    )
                    if not expected_ids:
                        raise PipelineError(
                            "staging manifest contains no retrieval chunks"
                        )
                    validation = await self.vector_store.validate(
                        staging,
                        expected_chunk_ids=expected_ids,
                        expected_count=len(expected_ids),
                    )
                    self._require_valid_staging(validation)
                    self.catalog.mark_staged(run_id)
                with self._stage(
                    operation,
                    "publish_generation",
                    generation_id=generation_id,
                    counts={"chunks": len(expected_ids)},
                ):
                    published = await self.vector_store.publish(staging)
                index_was_published = True
                with self._stage(
                    operation,
                    "verify_publication",
                    generation_id=generation_id,
                ):
                    active = await self.vector_store.active_generation_id()
                    if active != generation_id:
                        raise PipelineError(
                            "Milvus did not publish the expected generation"
                        )
                with self._stage(
                    operation,
                    "commit_manifest",
                    generation_id=generation_id,
                ):
                    try:
                        prior_manifest = self.catalog.commit_published(run_id)
                    except BaseException:
                        await self._rollback_index_after_catalog_failure(
                            expected_previous=previous_generation
                        )
                        raise
                with self._stage(
                    operation,
                    "verify_commit",
                    generation_id=generation_id,
                ):
                    if prior_manifest != previous_generation:
                        raise PipelineError(
                            "manifest publication returned an unexpected predecessor"
                        )
                    if (
                        published.previous_collection is None
                        and previous_generation is not None
                    ):
                        raise PipelineError(
                            "Milvus publication lost the previous alias target"
                        )
                documents = self.catalog.list_documents(self.config.kb_id)
                with self._stage(
                    operation,
                    "build_report",
                    generation_id=generation_id,
                ):
                    return self._plan_report(
                        plan,
                        operation=operation,
                        status="published",
                        generation_id=generation_id,
                        previous_generation_id=previous_generation,
                        document_count=len(documents),
                        chunk_count=len(expected_ids),
                    )
            except BaseException as exc:
                await self._cleanup_failed_run(
                    run_id,
                    staging,
                    index_was_published=index_was_published,
                    error=exc,
                )
                raise

    async def rollback(self) -> SyncReport:
        async with self._write_entry_lock:
            lease = (
                self.write_lease() if self.write_lease is not None else nullcontext()
            )
            with lease:
                with self._stage("rollback", "run"):
                    return await self._rollback_unlogged()

    async def _rollback_unlogged(self) -> SyncReport:
        async with self._lock:
            with self._stage("rollback", "consistency_check"):
                current = await self._require_consistent_active()
                target = self.catalog.previous_generation(self.config.kb_id)
            if current is None or target is None:
                raise PipelineError("no prior successful generation is available")
            with self._stage(
                "rollback",
                "switch_generation",
                generation_id=target,
            ):
                published = await self.vector_store.rollback()
                active = await self.vector_store.active_generation_id()
            if active != target:
                try:
                    await self.vector_store.rollback()
                finally:
                    raise PipelineError(
                        "Milvus rollback target does not match the manifest candidate"
                    )
            with self._stage(
                "rollback",
                "commit_manifest",
                generation_id=target,
            ):
                try:
                    previous = self.catalog.rollback_to_generation(
                        self.config.kb_id,
                        target,
                    )
                except BaseException:
                    await self.vector_store.rollback()
                    raise
            with self._stage(
                "rollback",
                "verify_commit",
                generation_id=target,
            ):
                if previous != current or published.previous_collection is None:
                    raise PipelineError(
                        "rollback predecessor does not match active state"
                    )
            with self._stage(
                "rollback",
                "build_report",
                generation_id=target,
            ):
                documents = self.catalog.list_documents(self.config.kb_id)
                return SyncReport(
                    operation="rollback",
                    status="rolled_back",
                    kb_id=self.config.kb_id,
                    generation_id=target,
                    previous_generation_id=current,
                    document_count=len(documents),
                    chunk_count=sum(item.chunk_count for item in documents),
                )

    async def _status_unlocked(self) -> IndexStatus:
        manifest_generation = self.catalog.current_generation(self.config.kb_id)
        vector_generation = await self.vector_store.active_generation_id()
        documents = self.catalog.list_documents(self.config.kb_id)
        return IndexStatus(
            kb_id=self.config.kb_id,
            manifest_generation_id=manifest_generation,
            vector_generation_id=vector_generation,
            consistent=manifest_generation == vector_generation,
            document_count=len(documents),
            chunk_count=sum(item.chunk_count for item in documents),
            rollback_candidates=self.catalog.rollback_candidates(self.config.kb_id),
        )

    async def _require_consistent_active(self) -> str | None:
        status = await self._status_unlocked()
        if not status.consistent:
            raise PipelineError(
                "manifest and Milvus active generations differ; reconcile before running"
            )
        return status.manifest_generation_id

    def _scan_and_plan(
        self,
        *,
        force_rebuild: bool,
        operation: str,
        source_snapshot: tuple[SourceDocument, ...] | None = None,
    ) -> tuple[tuple[SourceDocument, ...], SyncPlan]:
        with self._stage(operation, "scan"):
            sources = (
                scan_document_directory(
                    self.config.document_root,
                    kb_id=self.config.kb_id,
                    policy=self.config.scan_policy,
                )
                if source_snapshot is None
                else source_snapshot
            )
        with self._stage(
            operation,
            "plan",
            counts={"documents": len(sources)},
        ):
            manifest = self.catalog.list_documents(self.config.kb_id)
            plan = plan_incremental_sync(
                sources,
                manifest,
                self.pipeline,
                kb_id=self.config.kb_id,
            )
            if force_rebuild:
                previous = {item.source_id: item for item in manifest}
                current_ids = {item.source_id for item in sources}
                actions = [
                    SyncAction(
                        source_id=source.source_id,
                        relative_path=source.relative_path,
                        change=ChangeKind.PIPELINE_CHANGED,
                        start_stage=ProcessingStage.PARSE,
                        reason="application requested a full rebuild",
                        source=source,
                        previous=previous.get(source.source_id),
                    )
                    for source in sources
                ]
                actions.extend(
                    SyncAction(
                        source_id=item.source_id,
                        relative_path=item.relative_path,
                        change=ChangeKind.DELETED,
                        start_stage=ProcessingStage.DELETE,
                        reason="published source is absent from the rebuild input",
                        previous=item,
                    )
                    for item in manifest
                    if item.source_id not in current_ids
                )
                plan = SyncPlan(
                    kb_id=self.config.kb_id,
                    pipeline=self.pipeline,
                    actions=tuple(sorted(actions, key=lambda item: item.relative_path)),
                )
        return sources, plan

    def _requires_dimension_migration(
        self,
        plan: SyncPlan,
        *,
        has_active_generation: bool,
    ) -> bool:
        """Detect an incompatible vector schema before creating staging."""

        target = self.pipeline.embedding_dimension
        if not has_active_generation or target is None:
            return False
        return any(
            action.previous is not None
            and action.previous.pipeline.embedding_dimension != target
            for action in plan.actions
        )

    @staticmethod
    def _force_current_sources_to_embedding(plan: SyncPlan) -> SyncPlan:
        """Make a no-copy dimension migration populate every current source."""

        actions: list[SyncAction] = []
        for action in plan.actions:
            if action.source is None or action.start_stage in {
                ProcessingStage.PARSE,
                ProcessingStage.RESTRUCTURE,
                ProcessingStage.REFINE_LAYOUT,
                ProcessingStage.ENRICH,
                ProcessingStage.CHUNK,
                ProcessingStage.EMBED,
            }:
                actions.append(action)
                continue
            actions.append(
                SyncAction(
                    source_id=action.source_id,
                    relative_path=action.relative_path,
                    change=ChangeKind.PIPELINE_CHANGED,
                    start_stage=ProcessingStage.EMBED,
                    reason="embedding dimension changed; active vectors cannot be copied",
                    source=action.source,
                    previous=action.previous,
                )
            )
        return SyncPlan(
            kb_id=plan.kb_id,
            pipeline=plan.pipeline,
            actions=tuple(actions),
        )

    async def _materialize_chunks(
        self,
        action: SyncAction,
        *,
        operation: str,
        generation_id: str,
        item_index: int,
        item_count: int,
    ) -> tuple[Chunk, ...]:
        source = action.source
        if source is None:
            raise PipelineError("processing action has no source")
        source_fields = {
            "source_id": source.source_id,
            "generation_id": generation_id,
            "item_index": item_index,
            "item_count": item_count,
        }
        stage = action.start_stage
        if stage is ProcessingStage.EMBED:
            try:
                return self.artifacts.load_chunks(source, pipeline=self.pipeline)
            except ArtifactError:
                stage = ProcessingStage.CHUNK

        document: Any | None = None
        if stage in {
            ProcessingStage.RESTRUCTURE,
            ProcessingStage.REFINE_LAYOUT,
            ProcessingStage.ENRICH,
            ProcessingStage.CHUNK,
        }:
            try:
                document = self.artifacts.load_docling(
                    source, parser_fingerprint=self.pipeline.parser
                )
            except ArtifactError:
                stage = ProcessingStage.PARSE
        if stage is ProcessingStage.PARSE:
            with self._stage(operation, "parse", **source_fields):
                parsed = await self.parser.convert(source)
                if parsed.parser_fingerprint != self.pipeline.parser:
                    raise PipelineError(
                        "Docling parser revision changed during the run"
                    )
                self.artifacts.save_docling(source, parsed)
                document = parsed.document_json
        if document is None:
            with self._stage(operation, "load_parse_artifact", **source_fields):
                raise PipelineError(
                    "no Docling artifact is available for restructuring"
                )
        with self._stage(operation, "restructure", **source_fields):
            raw_blocks = restructure_docling_document(source, document)
        blocks: tuple[StructuredBlock, ...] = raw_blocks

        if self.pipeline.layout_refinement is not None:
            with self._stage(operation, "extract_page_captures", **source_fields):
                capture_pages = getattr(self.refiner, "capture_pages", None)
                captures = (
                    await capture_pages(source, document)
                    if callable(capture_pages)
                    else extract_page_captures(document)
                )
                if not isinstance(captures, tuple) or any(
                    not isinstance(item, PageCapture) for item in captures
                ):
                    raise PipelineError(
                        "layout page capture extractor returned invalid data"
                    )
            refinement_loaded = False
            if stage in {
                ProcessingStage.REFINE_LAYOUT,
                ProcessingStage.ENRICH,
                ProcessingStage.CHUNK,
            }:
                try:
                    _, blocks = self.artifacts.load_layout_refinement(
                        source,
                        raw_blocks,
                        captures=captures,
                        layout_refinement_fingerprint=(self.pipeline.layout_refinement),
                        policy=self.config.layout_refinement_policy,
                    )
                    refinement_loaded = True
                except ArtifactError:
                    stage = ProcessingStage.REFINE_LAYOUT
            if not refinement_loaded and stage in {
                ProcessingStage.PARSE,
                ProcessingStage.RESTRUCTURE,
                ProcessingStage.REFINE_LAYOUT,
            }:
                with self._stage(operation, "refine_layout", **source_fields):
                    if self.refiner is None:
                        raise PipelineError("layout refiner is unavailable")
                    patches = (
                        await self.refiner.refine(source, raw_blocks, captures)
                        if captures
                        else ()
                    )
                    if not isinstance(patches, tuple):
                        raise PipelineError(
                            "layout refiner must return an ordered tuple"
                        )
                    if any(not isinstance(item, LayoutPatch) for item in patches):
                        raise PipelineError("layout refiner returned an invalid patch")
                    if any(
                        item.model_fingerprint != self.pipeline.layout_refinement
                        for item in patches
                    ):
                        raise PipelineError(
                            "layout refinement model revision changed during the run"
                        )
                    capture_pages = {item.page_no for item in captures}
                    if any(item.page_no not in capture_pages for item in patches):
                        raise PipelineError(
                            "layout refiner returned a patch without a page capture"
                        )
                    blocks = apply_layout_patches(
                        raw_blocks,
                        patches,
                        captured_page_nos=tuple(item.page_no for item in patches),
                        policy=self.config.layout_refinement_policy,
                    )
                    self.artifacts.save_layout_refinement(
                        source,
                        raw_blocks,
                        patches,
                        blocks,
                        captures=captures,
                        layout_refinement_fingerprint=(self.pipeline.layout_refinement),
                        policy=self.config.layout_refinement_policy,
                    )

        enrichments: tuple[BlockEnrichment, ...]
        if stage is ProcessingStage.CHUNK:
            try:
                enrichments = self.artifacts.load_enrichments(
                    source,
                    enrichment_fingerprint=self.pipeline.enrichment,
                    block_ids=tuple(block.block_id for block in blocks),
                )
            except ArtifactError:
                stage = ProcessingStage.ENRICH
        if stage in {
            ProcessingStage.PARSE,
            ProcessingStage.RESTRUCTURE,
            ProcessingStage.REFINE_LAYOUT,
            ProcessingStage.ENRICH,
        }:
            with self._stage(
                operation,
                "enrich",
                counts={"blocks": len(blocks)},
                **source_fields,
            ):
                enrichment_values: list[BlockEnrichment] = []
                for start in range(0, len(blocks), self.config.enrichment_batch_size):
                    batch = await self.enricher.enrich(
                        blocks[start : start + self.config.enrichment_batch_size]
                    )
                    enrichment_values.extend(batch)
                enrichments = tuple(enrichment_values)
                if tuple(value.block_id for value in enrichments) != tuple(
                    block.block_id for block in blocks
                ):
                    raise PipelineError(
                        "Gemma enrichment order does not match source blocks"
                    )
                if any(
                    value.model_fingerprint != self.pipeline.enrichment
                    for value in enrichments
                ):
                    raise PipelineError("Gemma model revision changed during the run")
                self.artifacts.save_enrichments(
                    source,
                    enrichments,
                    enrichment_fingerprint=self.pipeline.enrichment,
                )
        with self._stage(
            operation,
            "chunk",
            counts={"blocks": len(blocks)},
            **source_fields,
        ):
            chunks = chunk_blocks(
                source,
                list(blocks),
                enrichments=list(enrichments),
                config=self.config.chunking,
                pipeline=self.pipeline,
            )
            self.artifacts.save_chunks(source, chunks, pipeline=self.pipeline)
        return chunks

    async def _embed_and_upsert(
        self,
        staging: StagingGeneration,
        chunks: tuple[Chunk, ...],
        *,
        operation: str,
        source_id: str,
        generation_id: str,
    ) -> None:
        batch_count = max(
            1,
            (len(chunks) + self.config.embedding_request_size - 1)
            // self.config.embedding_request_size,
        )
        for batch_index, start in enumerate(
            range(0, len(chunks), self.config.embedding_request_size),
            start=1,
        ):
            batch_chunks = chunks[start : start + self.config.embedding_request_size]
            with self._stage(
                operation,
                "embed",
                source_id=source_id,
                generation_id=generation_id,
                item_index=batch_index,
                item_count=batch_count,
                counts={"chunks": len(batch_chunks)},
            ):
                embedded = await self.embedder.embed(
                    tuple(item.embedding_text for item in batch_chunks)
                )
                self._require_embeddings(embedded, len(batch_chunks))
            with self._stage(
                operation,
                "index",
                source_id=source_id,
                generation_id=generation_id,
                item_index=batch_index,
                item_count=batch_count,
                counts={"chunks": len(batch_chunks)},
            ):
                inserted = await self.vector_store.upsert(
                    staging,
                    batch_chunks,
                    embedded,
                )
                if inserted != len(batch_chunks):
                    raise PipelineError("Milvus did not accept every embedded chunk")

    def _require_embeddings(self, value: EmbeddingBatch, expected: int) -> None:
        if not isinstance(value, EmbeddingBatch):
            raise PipelineError("embedding backend returned the wrong result type")
        if (
            len(value.vectors) != expected
            or value.dimension != self.config.embedding_dimension
            or value.model_fingerprint != self.pipeline.embedding
        ):
            raise PipelineError("BGE-M3 embedding response does not match the run")

    @staticmethod
    def _require_valid_staging(value: IndexValidation) -> None:
        if not isinstance(value, IndexValidation) or not value.valid:
            raise PipelineError("Milvus staging validation failed")

    @staticmethod
    def _require_copied_source(value: CopyResult, document: ManifestDocument) -> None:
        if (
            not isinstance(value, CopyResult)
            or value.source_ids != (document.source_id,)
            or value.copied_rows != document.chunk_count
        ):
            raise PipelineError("Milvus did not copy the retained source exactly")

    async def _rollback_index_after_catalog_failure(
        self, *, expected_previous: str | None
    ) -> None:
        try:
            if expected_previous is None:
                raise PipelineError(
                    "first publication reached Milvus but the manifest commit failed"
                )
            await self.vector_store.rollback()
            active = await self.vector_store.active_generation_id()
            if active != expected_previous:
                raise PipelineError("Milvus rollback did not restore the predecessor")
        except BaseException as exc:
            raise PipelineError(
                "manifest publication failed and Milvus could not be reconciled"
            ) from exc

    async def _cleanup_failed_run(
        self,
        run_id: str,
        staging: StagingGeneration | None,
        *,
        index_was_published: bool,
        error: BaseException,
    ) -> None:
        try:
            self.catalog.fail_run(run_id, type(error).__name__)
        except Exception:
            pass
        if staging is not None and not index_was_published:
            try:
                await self.vector_store.drop_staging(staging)
            except Exception:
                pass

    def _plan_report(
        self,
        plan: SyncPlan,
        *,
        operation: Literal["preview", "sync", "rebuild"],
        status: Literal["dry_run", "noop", "published"],
        document_count: int,
        chunk_count: int,
        generation_id: str | None = None,
        previous_generation_id: str | None = None,
    ) -> SyncReport:
        counts = {kind: 0 for kind in ChangeKind}
        work = plan.requiring_work()
        for action in plan.actions:
            counts[action.change] += 1
        details = tuple(
            ActionDetail(
                source_id=item.source_id,
                relative_path=item.relative_path,
                change=item.change.value,
                start_stage=item.start_stage.value,
                reason=item.reason,
            )
            for item in work[: self.config.max_action_details]
        )
        warnings = (
            ("action details were truncated",)
            if len(work) > self.config.max_action_details
            else ()
        )
        return SyncReport(
            operation=operation,
            status=status,
            kb_id=self.config.kb_id,
            generation_id=generation_id,
            previous_generation_id=previous_generation_id,
            document_count=document_count,
            chunk_count=chunk_count,
            new_count=counts[ChangeKind.NEW],
            modified_count=counts[ChangeKind.MODIFIED],
            pipeline_changed_count=counts[ChangeKind.PIPELINE_CHANGED],
            deleted_count=counts[ChangeKind.DELETED],
            unchanged_count=counts[ChangeKind.UNCHANGED],
            actions=details,
            details_truncated=bool(warnings),
            warnings=warnings,
        )


__all__ = [
    "ActionDetail",
    "IndexStatus",
    "ManagerConfig",
    "PipelineError",
    "RAGIndexManager",
    "SyncReport",
]
