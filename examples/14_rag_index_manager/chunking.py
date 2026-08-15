"""Deterministic structure-aware chunking for BGE-M3 retrieval text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from .models import (
    BlockEnrichment,
    BlockModality,
    Chunk,
    ChunkingError,
    PipelineFingerprint,
    Provenance,
    SourceDocument,
    StructuredBlock,
    component_fingerprint,
    stable_digest,
    unique_provenance,
)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Character-bounded teaching default independent of a tokenizer package."""

    max_chars: int = 2_400
    overlap_chars: int = 240
    max_embedding_chars: int = 32_000
    max_document_context_chars: int = 1_000
    separate_modalities: bool = True
    algorithm: str = "docling-structure-v1"

    def __post_init__(self) -> None:
        if type(self.max_chars) is not int or not 128 <= self.max_chars <= 32_000:
            raise ValueError("max_chars must be between 128 and 32000")
        if (
            type(self.overlap_chars) is not int
            or self.overlap_chars < 0
            or self.overlap_chars >= self.max_chars
        ):
            raise ValueError("overlap_chars must be non-negative and below max_chars")
        if (
            type(self.max_embedding_chars) is not int
            or not self.max_chars <= self.max_embedding_chars <= 32_000
        ):
            raise ValueError("max_embedding_chars must be between max_chars and 32000")
        if (
            type(self.max_document_context_chars) is not int
            or not 0 <= self.max_document_context_chars <= 8_000
        ):
            raise ValueError("max_document_context_chars must be between zero and 8000")
        if type(self.separate_modalities) is not bool:
            raise TypeError("separate_modalities must be a bool")
        if not isinstance(self.algorithm, str) or not self.algorithm.strip():
            raise ValueError("algorithm must be non-empty")

    @property
    def fingerprint(self) -> str:
        return component_fingerprint(
            self.algorithm,
            max_chars=self.max_chars,
            overlap_chars=self.overlap_chars,
            max_embedding_chars=self.max_embedding_chars,
            max_document_context_chars=self.max_document_context_chars,
            separate_modalities=self.separate_modalities,
        )


class _Piece(NamedTuple):
    block: StructuredBlock
    part_index: int
    start: int
    end: int
    text: str
    provenance: tuple[Provenance, ...]


def chunk_blocks(
    source: SourceDocument,
    blocks: tuple[StructuredBlock, ...] | list[StructuredBlock],
    *,
    enrichments: tuple[BlockEnrichment, ...] | list[BlockEnrichment] = (),
    config: ChunkingConfig | None = None,
    pipeline: PipelineFingerprint | None = None,
) -> tuple[Chunk, ...]:
    """Chunk one document while keeping exact content and model hints separate."""

    if not isinstance(source, SourceDocument):
        raise TypeError("source must be a SourceDocument")
    if not isinstance(blocks, (tuple, list)) or not blocks:
        raise ChunkingError("at least one structured block is required")
    if len(blocks) > 100_000:
        raise ChunkingError("structured block count exceeds its limit")
    resolved_config = config or ChunkingConfig()
    if not isinstance(resolved_config, ChunkingConfig):
        raise TypeError("config must be a ChunkingConfig")
    if pipeline is not None and not isinstance(pipeline, PipelineFingerprint):
        raise TypeError("pipeline must be a PipelineFingerprint")

    seen_blocks: set[str] = set()
    prior_ordinal = -1
    for block in blocks:
        if not isinstance(block, StructuredBlock):
            raise TypeError("blocks must contain StructuredBlock values")
        if (
            block.source_id != source.source_id
            or block.source_revision != source.sha256
        ):
            raise ChunkingError("block does not belong to the supplied source revision")
        if block.block_id in seen_blocks or block.ordinal <= prior_ordinal:
            raise ChunkingError(
                "blocks must have unique IDs in increasing reading order"
            )
        seen_blocks.add(block.block_id)
        prior_ordinal = block.ordinal

    enrichment_by_block: dict[str, BlockEnrichment] = {}
    model_fingerprints: set[str] = set()
    for enrichment in enrichments:
        if not isinstance(enrichment, BlockEnrichment):
            raise TypeError("enrichments must contain BlockEnrichment values")
        if enrichment.block_id not in seen_blocks:
            raise ChunkingError("enrichment references an unknown block")
        if enrichment.block_id in enrichment_by_block:
            raise ChunkingError("duplicate block enrichment")
        enrichment_by_block[enrichment.block_id] = enrichment
        if enrichment.model_fingerprint:
            model_fingerprints.add(enrichment.model_fingerprint)
    if len(model_fingerprints) > 1:
        raise ChunkingError("block enrichments use inconsistent model fingerprints")

    pieces: list[_Piece] = []
    for block in blocks:
        for part_index, (start, end, text) in enumerate(
            _split_text(block.text, resolved_config.max_chars)
        ):
            pieces.append(
                _Piece(
                    block=block,
                    part_index=part_index,
                    start=start,
                    end=end,
                    text=text,
                    provenance=_piece_provenance(block, start, end),
                )
            )
    if not pieces:
        raise ChunkingError("structured blocks produced no chunk content")

    groups: list[tuple[_Piece, ...]] = []
    current: list[_Piece] = []
    current_length = 0
    for piece in pieces:
        if not current:
            current = [piece]
            current_length = len(piece.text)
            continue
        delimiter = 2
        same_section = current[-1].block.section_path == piece.block.section_path
        compatible = _compatible(current[-1], piece, resolved_config)
        fits = current_length + delimiter + len(piece.text) <= resolved_config.max_chars
        if same_section and compatible and fits:
            current.append(piece)
            current_length += delimiter + len(piece.text)
            continue
        groups.append(tuple(current))
        carry = (
            _overlap_tail(current, resolved_config.overlap_chars)
            if same_section and compatible
            else []
        )
        carry_length = _joined_length(carry)
        if (
            carry
            and carry_length + delimiter + len(piece.text) <= resolved_config.max_chars
        ):
            current = [*carry, piece]
            current_length = carry_length + delimiter + len(piece.text)
        else:
            current = [piece]
            current_length = len(piece.text)
    if current:
        groups.append(tuple(current))

    chunks: list[Chunk] = []
    seen_chunk_ids: set[str] = set()
    chunking_revision = resolved_config.fingerprint
    pipeline_chunking = pipeline.chunking if pipeline is not None else chunking_revision
    pipeline_enrichment = (
        pipeline.enrichment
        if pipeline is not None
        else next(iter(model_fingerprints), "none")
    )
    document_context = _document_context(
        tuple(blocks), maximum=resolved_config.max_document_context_chars
    )
    for ordinal, group in enumerate(groups):
        content = "\n\n".join(piece.text for piece in group)
        if not content.strip() or len(content) > resolved_config.max_chars:
            raise ChunkingError("chunk content violates configured bounds")
        section_path = group[0].block.section_path
        block_ids = tuple(dict.fromkeys(piece.block.block_id for piece in group))
        locations = unique_provenance(
            tuple(location for piece in group for location in piece.provenance)
        )
        hints = _enrichment_hints(group, enrichment_by_block)
        if document_context:
            hints = (f"Document context: {document_context}", *hints)
        embedding_text = _embedding_text(
            section_path,
            content,
            hints,
            maximum=resolved_config.max_embedding_chars,
        )
        identity_ranges = ";".join(
            f"{piece.block.block_id}:{piece.part_index}:{piece.start}:{piece.end}"
            for piece in group
        )
        chunk_id = (
            "chk_"
            + stable_digest(
                source.kb_id,
                source.source_id,
                source.sha256,
                identity_ranges,
                chunking_revision,
                pipeline_chunking,
                pipeline_enrichment,
                content,
            )[:32]
        )
        if chunk_id in seen_chunk_ids:
            raise ChunkingError("chunk ID collision")
        seen_chunk_ids.add(chunk_id)
        pages = sorted(
            {location.page_no for location in locations if location.page_no is not None}
        )
        metadata = {
            "filename": source.filename,
            "relative_path": source.relative_path,
            "modalities": ",".join(
                dict.fromkeys(piece.block.modality.value for piece in group)
            ),
            "labels": ",".join(dict.fromkeys(piece.block.label for piece in group)),
            "chunking_fingerprint": chunking_revision,
            "pipeline_chunking_fingerprint": pipeline_chunking,
            "enrichment_fingerprint": pipeline_enrichment,
        }
        if pages:
            metadata["pages"] = ",".join(str(page) for page in pages)
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                kb_id=source.kb_id,
                source_id=source.source_id,
                source_revision=source.sha256,
                ordinal=ordinal,
                content=content,
                embedding_text=embedding_text,
                section_path=section_path,
                block_ids=block_ids,
                provenance=locations,
                metadata=metadata,
            )
        )
    return tuple(chunks)


def _split_text(text: str, maximum: int) -> tuple[tuple[int, int, str], ...]:
    result: list[tuple[int, int, str]] = []
    cursor = 0
    length = len(text)
    while cursor < length:
        while cursor < length and text[cursor].isspace():
            cursor += 1
        if cursor >= length:
            break
        hard_end = min(cursor + maximum, length)
        end = hard_end
        if hard_end < length:
            minimum = cursor + maximum // 2
            for separator in ("\n\n", "\n", ". ", "。", " "):
                candidate = text.rfind(separator, minimum, hard_end)
                if candidate >= minimum:
                    end = candidate + (1 if separator == ". " else len(separator))
                    break
        raw_piece = text[cursor:end]
        stripped = raw_piece.rstrip()
        actual_end = cursor + len(stripped)
        if stripped:
            result.append((cursor, actual_end, stripped))
        cursor = max(end, cursor + 1)
    return tuple(result)


def _piece_provenance(
    block: StructuredBlock,
    start: int,
    end: int,
) -> tuple[Provenance, ...]:
    overlapping = tuple(
        location
        for location in block.provenance
        if location.charspan is None
        or (location.charspan[0] < end and start < location.charspan[1])
    )
    return overlapping or block.provenance


def _compatible(
    previous: _Piece,
    current: _Piece,
    config: ChunkingConfig,
) -> bool:
    if not config.separate_modalities:
        return True
    isolated = {BlockModality.TABLE, BlockModality.IMAGE}
    if previous.block.modality in isolated or current.block.modality in isolated:
        return previous.block.block_id == current.block.block_id
    return True


def _overlap_tail(values: list[_Piece], maximum: int) -> list[_Piece]:
    if maximum <= 0:
        return []
    selected: list[_Piece] = []
    total = 0
    for piece in reversed(values):
        addition = len(piece.text) + (2 if selected else 0)
        if total + addition > maximum:
            break
        selected.append(piece)
        total += addition
    selected.reverse()
    return selected


def _joined_length(values: list[_Piece]) -> int:
    return sum(len(piece.text) for piece in values) + max(0, len(values) - 1) * 2


def _enrichment_hints(
    group: tuple[_Piece, ...],
    enrichment_by_block: dict[str, BlockEnrichment],
) -> tuple[str, ...]:
    hints: list[str] = []
    seen: set[str] = set()
    for block_id in dict.fromkeys(piece.block.block_id for piece in group):
        enrichment = enrichment_by_block.get(block_id)
        if enrichment is None:
            continue
        candidates = (
            f"Summary: {enrichment.summary.strip()}"
            if enrichment.summary.strip()
            else "",
            (
                f"Keywords: {', '.join(enrichment.keywords)}"
                if enrichment.keywords
                else ""
            ),
            (
                f"Image: {enrichment.image_description.strip()}"
                if enrichment.image_description.strip()
                else ""
            ),
        )
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                hints.append(candidate)
    return tuple(hints)


def _embedding_text(
    section_path: tuple[str, ...],
    content: str,
    hints: tuple[str, ...],
    *,
    maximum: int,
) -> str:
    parts: list[str] = []
    if section_path:
        parts.append("Section: " + " > ".join(section_path))
    parts.append(content)
    base = "\n".join(parts)
    if len(base) > maximum:
        # ``maximum >= max_chars`` normally leaves room for the heading.  If a
        # very long heading consumes it, retain source content preferentially.
        base = content[:maximum]
    for hint in hints:
        available = maximum - len(base) - 1
        if available <= 0:
            break
        base += "\n" + hint[:available]
    return base.strip()


def _document_context(
    blocks: tuple[StructuredBlock, ...],
    *,
    maximum: int,
) -> str:
    """Return bounded leading context for retrieval only, never citation content.

    Structured formats often split a document identifier, business unit, and
    body into separate blocks.  Repeating a small canonical prefix in the
    embedding input keeps those chunks associated without changing ``content``
    or provenance used for grounded answers.
    """

    if maximum == 0:
        return ""
    parts: list[str] = []
    length = 0
    for block in blocks:
        text = " ".join(block.text.split())
        if not text:
            continue
        delimiter = 2 if parts else 0
        available = maximum - length - delimiter
        if available <= 0:
            break
        parts.append(text[:available])
        length += delimiter + min(len(text), available)
        if len(text) > available:
            break
    return "\n\n".join(parts)
