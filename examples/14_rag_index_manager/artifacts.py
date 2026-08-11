"""Bounded run-local artifact cache for resumable ingestion stages."""

from __future__ import annotations

import json
import hashlib
import os
import secrets
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .backends import DoclingResult
from .models import (
    BlockEnrichment,
    BlockModality,
    Chunk,
    LayoutPatch,
    LayoutRefinementPolicy,
    PageCapture,
    PipelineFingerprint,
    Provenance,
    RAGIndexError,
    SourceDocument,
    StructuredBlock,
    apply_layout_patches,
)


ARTIFACT_SCHEMA_VERSION = 1
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024


class ArtifactError(RAGIndexError):
    """A cached stage artifact is missing, stale, malformed, or unsafe."""


class ArtifactStore:
    """Persist parser/enrichment/chunk outputs without duplicating vectors.

    Artifacts are addressed only by application-generated source IDs and content
    revisions. They are an optimization: the manifest remains the lifecycle
    source of truth and Milvus remains the serving index.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        raw = Path(root).expanduser()
        try:
            raw.mkdir(parents=True, exist_ok=True)
            if raw.is_symlink():
                raise ArtifactError("artifact root cannot be a symbolic link")
            resolved = raw.resolve(strict=True)
            if not resolved.is_dir():
                raise ArtifactError("artifact root must be a directory")
        except ArtifactError:
            raise
        except OSError as exc:
            raise ArtifactError("artifact root is not accessible") from exc
        self.root = resolved

    def save_docling(self, source: SourceDocument, result: DoclingResult) -> None:
        if not isinstance(source, SourceDocument):
            raise TypeError("source must be a SourceDocument")
        if not isinstance(result, DoclingResult):
            raise TypeError("result must be a DoclingResult")
        self._write(
            source,
            "docling.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_id": source.source_id,
                "source_revision": source.source_revision,
                "parser_fingerprint": result.parser_fingerprint,
                "document": dict(result.document_json),
            },
        )

    def load_docling(
        self,
        source: SourceDocument,
        *,
        parser_fingerprint: str,
    ) -> Mapping[str, Any]:
        value = self._read(source, "docling.json")
        self._verify_header(value, source)
        if value.get("parser_fingerprint") != parser_fingerprint:
            raise ArtifactError(
                "cached Docling artifact has a different parser revision"
            )
        document = value.get("document")
        if not isinstance(document, Mapping):
            raise ArtifactError("cached Docling artifact is malformed")
        return dict(document)

    def save_layout_refinement(
        self,
        source: SourceDocument,
        raw_blocks: Sequence[StructuredBlock],
        patches: Sequence[LayoutPatch],
        refined_blocks: Sequence[StructuredBlock],
        *,
        captures: Sequence[PageCapture],
        layout_refinement_fingerprint: str,
        policy: LayoutRefinementPolicy | None = None,
    ) -> None:
        """Persist source-faithful blocks and a verified layout-only patch."""

        raw = tuple(raw_blocks)
        proposals = tuple(patches)
        refined = tuple(refined_blocks)
        page_captures = tuple(captures)
        resolved_policy = policy or LayoutRefinementPolicy()
        if (
            not isinstance(layout_refinement_fingerprint, str)
            or not layout_refinement_fingerprint
            or len(layout_refinement_fingerprint) > 512
        ):
            raise ValueError("layout_refinement_fingerprint is invalid")
        if any(not isinstance(item, StructuredBlock) for item in (*raw, *refined)):
            raise TypeError("layout blocks must contain StructuredBlock values")
        if any(not isinstance(item, LayoutPatch) for item in proposals):
            raise TypeError("patches must contain LayoutPatch values")
        if any(
            item.model_fingerprint != layout_refinement_fingerprint
            for item in proposals
        ):
            raise ArtifactError("layout patch model revision does not match the run")
        selected_captures = _selected_captures(
            page_captures, tuple(item.page_no for item in proposals)
        )
        expected = apply_layout_patches(
            raw,
            proposals,
            captured_page_nos=tuple(item.page_no for item in selected_captures),
            policy=resolved_policy,
        )
        if refined != expected:
            raise ArtifactError("refined blocks are not the verified patch result")
        capture_manifest = _capture_manifest(selected_captures)
        self._write(
            source,
            "layout_refinement.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_id": source.source_id,
                "source_revision": source.source_revision,
                "layout_refinement_fingerprint": layout_refinement_fingerprint,
                "raw_blocks_digest": _blocks_digest(raw),
                "capture_digest": _canonical_digest(capture_manifest),
                "capture_pages": [item["page_no"] for item in capture_manifest],
                "allow_exclusions": resolved_policy.allow_exclusions,
                "raw_blocks": [_block_to_dict(item) for item in raw],
                "patches": [_patch_to_dict(item) for item in proposals],
                "refined_blocks": [_block_to_dict(item) for item in refined],
            },
        )

    def load_layout_refinement(
        self,
        source: SourceDocument,
        raw_blocks: Sequence[StructuredBlock],
        *,
        captures: Sequence[PageCapture],
        layout_refinement_fingerprint: str,
        policy: LayoutRefinementPolicy | None = None,
    ) -> tuple[tuple[LayoutPatch, ...], tuple[StructuredBlock, ...]]:
        """Load a refinement only after replaying and verifying its patch."""

        raw = tuple(raw_blocks)
        page_captures = tuple(captures)
        resolved_policy = policy or LayoutRefinementPolicy()
        value = self._read(source, "layout_refinement.json")
        self._verify_header(value, source)
        raw_capture_pages = value.get("capture_pages")
        if not isinstance(raw_capture_pages, list) or any(
            type(item) is not int or item < 1 for item in raw_capture_pages
        ):
            raise ArtifactError("cached layout capture page set is malformed")
        selected_captures = _selected_captures(page_captures, tuple(raw_capture_pages))
        capture_manifest = _capture_manifest(selected_captures)
        if (
            value.get("layout_refinement_fingerprint") != layout_refinement_fingerprint
            or value.get("raw_blocks_digest") != _blocks_digest(raw)
            or value.get("capture_digest") != _canonical_digest(capture_manifest)
            or value.get("capture_pages")
            != [item["page_no"] for item in capture_manifest]
            or value.get("allow_exclusions") is not resolved_policy.allow_exclusions
        ):
            raise ArtifactError("cached layout refinement has stale inputs or policy")
        try:
            stored_raw = tuple(
                _block_from_dict(item) for item in _required_array(value, "raw_blocks")
            )
            patches = tuple(
                _patch_from_dict(item) for item in _required_array(value, "patches")
            )
            refined = tuple(
                _block_from_dict(item)
                for item in _required_array(value, "refined_blocks")
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError("cached layout refinement is malformed") from exc
        if stored_raw != raw or any(
            item.model_fingerprint != layout_refinement_fingerprint for item in patches
        ):
            raise ArtifactError("cached layout refinement does not match raw blocks")
        try:
            expected = apply_layout_patches(
                raw,
                patches,
                captured_page_nos=tuple(item.page_no for item in selected_captures),
                policy=resolved_policy,
            )
        except RAGIndexError as exc:
            raise ArtifactError("cached layout patch failed verification") from exc
        if refined != expected:
            raise ArtifactError("cached refined blocks failed deterministic replay")
        return patches, refined

    def save_enrichments(
        self,
        source: SourceDocument,
        values: Sequence[BlockEnrichment],
        *,
        enrichment_fingerprint: str,
    ) -> None:
        items = tuple(values)
        if len(items) > 100_000 or any(
            not isinstance(item, BlockEnrichment) for item in items
        ):
            raise TypeError("values must be a bounded BlockEnrichment sequence")
        self._write(
            source,
            "enrichments.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_id": source.source_id,
                "source_revision": source.source_revision,
                "enrichment_fingerprint": enrichment_fingerprint,
                "items": [
                    {
                        "block_id": item.block_id,
                        "summary": item.summary,
                        "keywords": list(item.keywords),
                        "image_description": item.image_description,
                        "embedding_text": item.embedding_text,
                        "model_fingerprint": item.model_fingerprint,
                    }
                    for item in items
                ],
            },
        )

    def load_enrichments(
        self,
        source: SourceDocument,
        *,
        enrichment_fingerprint: str,
        block_ids: Sequence[str],
    ) -> tuple[BlockEnrichment, ...]:
        value = self._read(source, "enrichments.json")
        self._verify_header(value, source)
        if value.get("enrichment_fingerprint") != enrichment_fingerprint:
            raise ArtifactError("cached enrichment has a different model revision")
        raw_items = value.get("items")
        if not isinstance(raw_items, list) or len(raw_items) > 100_000:
            raise ArtifactError("cached enrichment is malformed")
        try:
            items = tuple(
                BlockEnrichment(
                    block_id=item["block_id"],
                    summary=item.get("summary", ""),
                    keywords=tuple(item.get("keywords", ())),
                    image_description=item.get("image_description", ""),
                    embedding_text=item.get("embedding_text", ""),
                    model_fingerprint=item.get("model_fingerprint", ""),
                )
                for item in raw_items
                if isinstance(item, Mapping)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError("cached enrichment is malformed") from exc
        if tuple(item.block_id for item in items) != tuple(block_ids):
            raise ArtifactError("cached enrichment does not match restructured blocks")
        return items

    def save_chunks(
        self,
        source: SourceDocument,
        chunks: Sequence[Chunk],
        *,
        pipeline: PipelineFingerprint,
    ) -> None:
        values = tuple(chunks)
        if (
            not values
            or len(values) > 1_000_000
            or any(not isinstance(item, Chunk) for item in values)
        ):
            raise TypeError("chunks must be a non-empty bounded Chunk sequence")
        if any(
            item.source_id != source.source_id
            or item.source_revision != source.source_revision
            for item in values
        ):
            raise ArtifactError("chunks do not belong to the source revision")
        self._write(
            source,
            "chunks.json",
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_id": source.source_id,
                "source_revision": source.source_revision,
                "enrichment_fingerprint": pipeline.enrichment,
                "layout_refinement_fingerprint": pipeline.layout_refinement,
                "chunking_fingerprint": pipeline.chunking,
                "items": [_chunk_to_dict(item) for item in values],
            },
        )

    def load_chunks(
        self,
        source: SourceDocument,
        *,
        pipeline: PipelineFingerprint,
    ) -> tuple[Chunk, ...]:
        value = self._read(source, "chunks.json")
        self._verify_header(value, source)
        if (
            value.get("enrichment_fingerprint") != pipeline.enrichment
            or value.get("layout_refinement_fingerprint") != pipeline.layout_refinement
            or value.get("chunking_fingerprint") != pipeline.chunking
        ):
            raise ArtifactError("cached chunks have different pipeline revisions")
        raw_items = value.get("items")
        if (
            not isinstance(raw_items, list)
            or not raw_items
            or len(raw_items) > 1_000_000
        ):
            raise ArtifactError("cached chunks are malformed")
        try:
            chunks = tuple(_chunk_from_dict(item) for item in raw_items)
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactError("cached chunks are malformed") from exc
        if any(
            item.source_id != source.source_id
            or item.source_revision != source.source_revision
            for item in chunks
        ):
            raise ArtifactError("cached chunks do not match the source revision")
        return chunks

    def _directory(self, source: SourceDocument) -> Path:
        if not isinstance(source, SourceDocument):
            raise TypeError("source must be a SourceDocument")
        directory = self.root / source.source_id / source.source_revision
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if directory.is_symlink() or not directory.is_dir():
                raise ArtifactError("artifact directory is unsafe")
            resolved = directory.resolve(strict=True)
        except ArtifactError:
            raise
        except OSError as exc:
            raise ArtifactError("artifact directory is not accessible") from exc
        if self.root not in resolved.parents:
            raise ArtifactError("artifact directory escaped its configured root")
        return resolved

    def _write(
        self, source: SourceDocument, name: str, value: Mapping[str, Any]
    ) -> None:
        directory = self._directory(source)
        destination = directory / name
        if destination.is_symlink():
            raise ArtifactError("artifact destination cannot be a symbolic link")
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ArtifactError("artifact is not JSON serializable") from exc
        if len(encoded) > MAX_ARTIFACT_BYTES:
            raise ArtifactError("artifact exceeds its byte limit")
        temporary = directory / f".{name}.{secrets.token_hex(8)}.tmp"
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise ArtifactError("artifact could not be written atomically") from exc

    def _read(self, source: SourceDocument, name: str) -> Mapping[str, Any]:
        path = self._directory(source) / name
        try:
            if path.is_symlink() or not path.is_file():
                raise ArtifactError("required stage artifact is unavailable")
            with path.open("rb") as stream:
                content = stream.read(MAX_ARTIFACT_BYTES + 1)
        except ArtifactError:
            raise
        except OSError as exc:
            raise ArtifactError("stage artifact could not be read") from exc
        if len(content) > MAX_ARTIFACT_BYTES:
            raise ArtifactError("stage artifact exceeds its byte limit")
        try:
            value = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ArtifactError("stage artifact is invalid JSON") from exc
        if not isinstance(value, Mapping):
            raise ArtifactError("stage artifact must be a JSON object")
        return value

    @staticmethod
    def _verify_header(value: Mapping[str, Any], source: SourceDocument) -> None:
        if (
            value.get("schema_version") != ARTIFACT_SCHEMA_VERSION
            or value.get("source_id") != source.source_id
            or value.get("source_revision") != source.source_revision
        ):
            raise ArtifactError("stage artifact identity does not match the source")


def _chunk_to_dict(value: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": value.chunk_id,
        "kb_id": value.kb_id,
        "source_id": value.source_id,
        "source_revision": value.source_revision,
        "ordinal": value.ordinal,
        "content": value.content,
        "embedding_text": value.embedding_text,
        "section_path": list(value.section_path),
        "block_ids": list(value.block_ids),
        "provenance": [
            {
                "self_ref": item.self_ref,
                "page_no": item.page_no,
                "bbox": item.bbox,
                "charspan": item.charspan,
                "coord_origin": item.coord_origin,
            }
            for item in value.provenance
        ],
        "metadata": dict(value.metadata),
    }


def _chunk_from_dict(value: Any) -> Chunk:
    if not isinstance(value, Mapping):
        raise TypeError("chunk artifact item must be an object")
    raw_provenance = value["provenance"]
    if not isinstance(raw_provenance, list):
        raise TypeError("chunk provenance must be an array")
    provenance = tuple(
        Provenance(
            self_ref=item["self_ref"],
            page_no=item.get("page_no"),
            bbox=None if item.get("bbox") is None else tuple(item["bbox"]),
            charspan=(
                None if item.get("charspan") is None else tuple(item["charspan"])
            ),
            coord_origin=item.get("coord_origin"),
        )
        for item in raw_provenance
        if isinstance(item, Mapping)
    )
    return Chunk(
        chunk_id=value["chunk_id"],
        kb_id=value["kb_id"],
        source_id=value["source_id"],
        source_revision=value["source_revision"],
        ordinal=value["ordinal"],
        content=value["content"],
        embedding_text=value["embedding_text"],
        section_path=tuple(value["section_path"]),
        block_ids=tuple(value["block_ids"]),
        provenance=provenance,
        metadata=value.get("metadata", {}),
    )


def _block_to_dict(value: StructuredBlock) -> dict[str, Any]:
    return {
        "block_id": value.block_id,
        "source_id": value.source_id,
        "source_revision": value.source_revision,
        "ordinal": value.ordinal,
        "modality": value.modality.value,
        "label": value.label,
        "text": value.text,
        "section_path": list(value.section_path),
        "provenance": [
            {
                "self_ref": item.self_ref,
                "page_no": item.page_no,
                "bbox": item.bbox,
                "charspan": item.charspan,
                "coord_origin": item.coord_origin,
            }
            for item in value.provenance
        ],
        "metadata": dict(value.metadata),
        "image_data_uri": value.image_data_uri,
    }


def _block_from_dict(value: Any) -> StructuredBlock:
    if not isinstance(value, Mapping):
        raise TypeError("layout block artifact item must be an object")
    raw_provenance = value["provenance"]
    if not isinstance(raw_provenance, list):
        raise TypeError("layout block provenance must be an array")
    provenance = tuple(
        Provenance(
            self_ref=item["self_ref"],
            page_no=item.get("page_no"),
            bbox=None if item.get("bbox") is None else tuple(item["bbox"]),
            charspan=(
                None if item.get("charspan") is None else tuple(item["charspan"])
            ),
            coord_origin=item.get("coord_origin"),
        )
        for item in raw_provenance
        if isinstance(item, Mapping)
    )
    return StructuredBlock(
        block_id=value["block_id"],
        source_id=value["source_id"],
        source_revision=value["source_revision"],
        ordinal=value["ordinal"],
        modality=BlockModality(value["modality"]),
        label=value["label"],
        text=value["text"],
        section_path=tuple(value["section_path"]),
        provenance=provenance,
        metadata=value.get("metadata", {}),
        image_data_uri=value.get("image_data_uri"),
    )


def _patch_to_dict(value: LayoutPatch) -> dict[str, Any]:
    return {
        "page_no": value.page_no,
        "ordered_block_ids": list(value.ordered_block_ids),
        "parent_by_block": dict(value.parent_by_block),
        "section_heading_ids_by_block": {
            key: list(path) for key, path in value.section_heading_ids_by_block.items()
        },
        "role_by_block": dict(value.role_by_block),
        "group_by_block": dict(value.group_by_block),
        "excluded_reason_by_block": dict(value.excluded_reason_by_block),
        "model_fingerprint": value.model_fingerprint,
    }


def _patch_from_dict(value: Any) -> LayoutPatch:
    if not isinstance(value, Mapping):
        raise TypeError("layout patch artifact item must be an object")
    raw_sections = value["section_heading_ids_by_block"]
    if not isinstance(raw_sections, Mapping):
        raise TypeError("layout patch section heading IDs must be an object")
    return LayoutPatch(
        page_no=value["page_no"],
        ordered_block_ids=tuple(value["ordered_block_ids"]),
        parent_by_block=value["parent_by_block"],
        section_heading_ids_by_block={
            str(key): tuple(path) for key, path in raw_sections.items()
        },
        role_by_block=value["role_by_block"],
        group_by_block=value["group_by_block"],
        excluded_reason_by_block=value["excluded_reason_by_block"],
        model_fingerprint=value["model_fingerprint"],
    )


def _capture_manifest(values: Sequence[PageCapture]) -> list[dict[str, Any]]:
    captures = tuple(values)
    if len(captures) > 10_000 or any(
        not isinstance(item, PageCapture) for item in captures
    ):
        raise ArtifactError("page captures are invalid or exceed their limit")
    if len({item.page_no for item in captures}) != len(captures):
        raise ArtifactError("page captures contain duplicate page numbers")
    return [
        {
            "page_no": item.page_no,
            "image_sha256": hashlib.sha256(
                item.image_data_uri.encode("utf-8")
            ).hexdigest(),
            "width": item.width,
            "height": item.height,
        }
        for item in sorted(captures, key=lambda item: item.page_no)
    ]


def _selected_captures(
    captures: Sequence[PageCapture], page_nos: Sequence[int]
) -> tuple[PageCapture, ...]:
    if any(not isinstance(item, PageCapture) for item in captures):
        raise TypeError("captures must contain PageCapture values")
    by_page = {item.page_no: item for item in captures}
    if len(by_page) != len(captures):
        raise ArtifactError("page captures contain duplicate page numbers")
    selected_pages = tuple(page_nos)
    if len(set(selected_pages)) != len(selected_pages) or any(
        page_no not in by_page for page_no in selected_pages
    ):
        raise ArtifactError("layout patches reference unavailable page captures")
    return tuple(by_page[page_no] for page_no in sorted(selected_pages))


def _blocks_digest(values: Sequence[StructuredBlock]) -> str:
    return _canonical_digest([_block_to_dict(item) for item in values])


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_array(value: Mapping[str, Any], key: str) -> list[Any]:
    result = value.get(key)
    if not isinstance(result, list) or len(result) > 100_000:
        raise TypeError(f"{key} must be a bounded array")
    return result
