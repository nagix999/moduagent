"""Pure incremental-change planning over scanned and published manifests."""

from __future__ import annotations

from collections.abc import Sequence

from .models import (
    ChangeKind,
    ManifestDocument,
    PipelineFingerprint,
    PlanningError,
    ProcessingStage,
    SourceDocument,
    SyncAction,
    SyncPlan,
)


def plan_incremental_sync(
    sources: Sequence[SourceDocument],
    manifest: Sequence[ManifestDocument],
    pipeline: PipelineFingerprint,
    *,
    kb_id: str | None = None,
) -> SyncPlan:
    """Find the earliest invalid processing stage for every source.

    The function is side-effect free and deliberately includes unchanged
    documents in its output.  The orchestrator can copy their existing rows
    into a new blue/green index generation without accidentally omitting them.
    """

    if not isinstance(pipeline, PipelineFingerprint):
        raise TypeError("pipeline must be a PipelineFingerprint")
    source_values = tuple(sources)
    manifest_values = tuple(manifest)
    if any(not isinstance(item, SourceDocument) for item in source_values):
        raise TypeError("sources must contain SourceDocument values")
    if any(not isinstance(item, ManifestDocument) for item in manifest_values):
        raise TypeError("manifest must contain ManifestDocument values")

    knowledge_bases = {item.kb_id for item in (*source_values, *manifest_values)}
    if kb_id is not None:
        knowledge_bases.add(kb_id)
    if len(knowledge_bases) > 1:
        raise PlanningError("sync inputs must belong to one knowledge base")
    resolved_kb_id = next(iter(knowledge_bases), kb_id)
    if not resolved_kb_id:
        raise PlanningError("kb_id is required when both inputs are empty")

    current: dict[str, SourceDocument] = {}
    current_paths: set[str] = set()
    for source in source_values:
        if source.source_id in current or source.relative_path in current_paths:
            raise PlanningError("scanned sources contain duplicate identities")
        current[source.source_id] = source
        current_paths.add(source.relative_path)

    previous: dict[str, ManifestDocument] = {}
    previous_paths: set[str] = set()
    for document in manifest_values:
        if document.source_id in previous or document.relative_path in previous_paths:
            raise PlanningError("manifest contains duplicate source identities")
        previous[document.source_id] = document
        previous_paths.add(document.relative_path)

    actions: list[SyncAction] = []
    for source in current.values():
        old = previous.get(source.source_id)
        if old is None:
            actions.append(
                SyncAction(
                    source_id=source.source_id,
                    relative_path=source.relative_path,
                    change=ChangeKind.NEW,
                    start_stage=ProcessingStage.PARSE,
                    reason="source is not present in the published manifest",
                    source=source,
                )
            )
            continue
        if old.relative_path != source.relative_path:
            # A correctly generated source_id already binds the path.  Treat a
            # mismatch as corrupt input instead of silently reassigning data.
            raise PlanningError("source ID and relative path do not agree")
        if old.content_sha256 != source.sha256:
            actions.append(
                SyncAction(
                    source_id=source.source_id,
                    relative_path=source.relative_path,
                    change=ChangeKind.MODIFIED,
                    start_stage=ProcessingStage.PARSE,
                    reason="source content hash changed",
                    source=source,
                    previous=old,
                )
            )
            continue
        stage, reason = _pipeline_restart(old.pipeline, pipeline)
        if stage is not ProcessingStage.NONE:
            actions.append(
                SyncAction(
                    source_id=source.source_id,
                    relative_path=source.relative_path,
                    change=ChangeKind.PIPELINE_CHANGED,
                    start_stage=stage,
                    reason=reason,
                    source=source,
                    previous=old,
                )
            )
            continue
        if old.chunk_count < 1:
            actions.append(
                SyncAction(
                    source_id=source.source_id,
                    relative_path=source.relative_path,
                    change=ChangeKind.PIPELINE_CHANGED,
                    start_stage=ProcessingStage.PARSE,
                    reason="published manifest has no chunks",
                    source=source,
                    previous=old,
                )
            )
            continue
        actions.append(
            SyncAction(
                source_id=source.source_id,
                relative_path=source.relative_path,
                change=ChangeKind.UNCHANGED,
                start_stage=ProcessingStage.NONE,
                reason="source and all pipeline fingerprints are unchanged",
                source=source,
                previous=old,
            )
        )

    for source_id, old in previous.items():
        if source_id not in current:
            actions.append(
                SyncAction(
                    source_id=source_id,
                    relative_path=old.relative_path,
                    change=ChangeKind.DELETED,
                    start_stage=ProcessingStage.DELETE,
                    reason="published source is absent from the document directory",
                    previous=old,
                )
            )

    actions.sort(key=lambda item: (item.relative_path, item.source_id))
    return SyncPlan(
        kb_id=resolved_kb_id,
        pipeline=pipeline,
        actions=tuple(actions),
    )


def _pipeline_restart(
    old: PipelineFingerprint,
    new: PipelineFingerprint,
) -> tuple[ProcessingStage, str]:
    stages = (
        ("parser", ProcessingStage.PARSE),
        ("restructuring", ProcessingStage.RESTRUCTURE),
        ("layout_refinement", ProcessingStage.REFINE_LAYOUT),
        ("enrichment", ProcessingStage.ENRICH),
        ("chunking", ProcessingStage.CHUNK),
        ("embedding", ProcessingStage.EMBED),
        ("embedding_dimension", ProcessingStage.EMBED),
        ("indexing", ProcessingStage.INDEX),
    )
    for field_name, stage in stages:
        if getattr(old, field_name) != getattr(new, field_name):
            return stage, f"{field_name} fingerprint changed"
    return ProcessingStage.NONE, "pipeline fingerprints are unchanged"
