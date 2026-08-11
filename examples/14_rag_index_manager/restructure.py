"""Convert lossless DoclingDocument JSON into retrieval-friendly blocks."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .models import (
    BlockModality,
    Provenance,
    RestructureError,
    SourceDocument,
    StructuredBlock,
    component_fingerprint,
    stable_digest,
    unique_provenance,
)


MAX_BLOCKS = 20_000
MAX_TOTAL_TEXT_CHARS = 50_000_000
MAX_TABLE_CELLS = 10_000
MAX_IMAGE_DATA_URI_CHARS = 28_000_000
RESTRUCTURING_FINGERPRINT = component_fingerprint(
    "docling-reading-order",
    algorithm="structure-preserving-v1",
    table_format="markdown-grid-v1",
    image_mode="embedded-data-uri",
)
_DATA_IMAGE = re.compile(
    r"^data:image/(?:png|jpeg|webp|gif);base64,[A-Za-z0-9+/]+={0,2}$"
)


def restructure_docling_document(
    source: SourceDocument,
    document: Mapping[str, Any],
    *,
    max_blocks: int = MAX_BLOCKS,
    max_total_chars: int = MAX_TOTAL_TEXT_CHARS,
) -> tuple[StructuredBlock, ...]:
    """Preserve reading order, section hierarchy, tables, and provenance."""

    if not isinstance(source, SourceDocument):
        raise TypeError("source must be a SourceDocument")
    if not isinstance(document, Mapping):
        raise TypeError("document must be a mapping")
    if document.get("schema_name") != "DoclingDocument":
        raise RestructureError("Docling JSON has an unexpected schema")
    if type(max_blocks) is not int or not 1 <= max_blocks <= 100_000:
        raise ValueError("max_blocks must be between 1 and 100000")
    if type(max_total_chars) is not int or max_total_chars < 1:
        raise ValueError("max_total_chars must be positive")

    indexed, serialized = _index_items(document)
    ordered = _reading_order(document, indexed, serialized, max_blocks=max_blocks)
    blocks: list[StructuredBlock] = []
    heading_stack: list[str] = []
    total_chars = 0

    for item, collection_name, serialized_index in ordered:
        label = _label(item, collection_name)
        modality = _modality(collection_name, label)
        text, basis = _item_text(item, modality, indexed)
        text = _normalize_text(text)
        image_data_uri = (
            _image_data_uri(item) if modality is BlockModality.IMAGE else None
        )
        if not text and image_data_uri:
            text = "[Image]"
            basis = "image_placeholder"
        if not text:
            continue
        if len(text) > 2_000_000:
            raise RestructureError("one Docling block exceeds its text limit")

        if label in {"title", "section_header"}:
            level = _heading_level(item, label)
            heading_stack = heading_stack[: max(0, level - 1)]
            heading_stack.append(_single_line(text, 500))
        section_path = tuple(heading_stack[-16:])
        self_ref = _self_ref(item, collection_name, serialized_index)
        provenance = _provenance(
            item, self_ref, allow_charspan=modality is not BlockModality.TABLE
        )
        total_chars += len(text)
        if total_chars > max_total_chars:
            raise RestructureError("restructured document exceeds its text limit")
        ordinal = len(blocks)
        block_id = (
            "blk_"
            + stable_digest(
                source.source_id,
                source.sha256,
                self_ref,
                ordinal,
                modality.value,
                text,
            )[:32]
        )
        pages = sorted(
            {
                location.page_no
                for location in provenance
                if location.page_no is not None
            }
        )
        metadata = {
            "filename": source.filename,
            "relative_path": source.relative_path,
            "label": label,
            "text_basis": basis,
            "self_ref": self_ref,
        }
        if pages:
            metadata["pages"] = ",".join(str(page) for page in pages)
        blocks.append(
            StructuredBlock(
                block_id=block_id,
                source_id=source.source_id,
                source_revision=source.sha256,
                ordinal=ordinal,
                modality=modality,
                label=label,
                text=text,
                section_path=section_path,
                provenance=provenance,
                metadata=metadata,
                image_data_uri=image_data_uri,
            )
        )
        if len(blocks) > max_blocks:
            raise RestructureError("restructured document exceeds its block limit")

    if not blocks:
        raise RestructureError("Docling produced no retrieval content")
    return tuple(blocks)


def _index_items(
    document: Mapping[str, Any],
) -> tuple[
    dict[str, tuple[Mapping[str, Any], str, int]],
    tuple[tuple[Mapping[str, Any], str, int], ...],
]:
    indexed: dict[str, tuple[Mapping[str, Any], str, int]] = {}
    serialized: list[tuple[Mapping[str, Any], str, int]] = []
    for collection_name in (
        "texts",
        "tables",
        "pictures",
        "key_value_items",
        "groups",
    ):
        collection = document.get(collection_name, [])
        if collection is None:
            continue
        if not isinstance(collection, list):
            raise RestructureError(f"Docling {collection_name} must be an array")
        for index, value in enumerate(collection):
            if not isinstance(value, Mapping):
                continue
            entry = (value, collection_name, index)
            serialized.append(entry)
            self_ref = value.get("self_ref")
            if isinstance(self_ref, str) and self_ref.startswith("#/"):
                if len(self_ref) > 512:
                    raise RestructureError("Docling self_ref exceeds its length limit")
                if self_ref in indexed:
                    raise RestructureError(
                        "Docling document contains duplicate self_ref"
                    )
                indexed[self_ref] = entry
    if len(serialized) > MAX_BLOCKS * 4:
        raise RestructureError("Docling document contains too many serialized items")
    return indexed, tuple(serialized)


def _reading_order(
    document: Mapping[str, Any],
    indexed: Mapping[str, tuple[Mapping[str, Any], str, int]],
    serialized: Sequence[tuple[Mapping[str, Any], str, int]],
    *,
    max_blocks: int,
) -> tuple[tuple[Mapping[str, Any], str, int], ...]:
    ordered: list[tuple[Mapping[str, Any], str, int]] = []
    visited: set[str] = set()
    traversed = 0

    def visit(value: Any, depth: int) -> None:
        nonlocal traversed
        traversed += 1
        if traversed > max_blocks * 8:
            raise RestructureError("Docling reading-order graph exceeds its limit")
        if depth > 64:
            raise RestructureError("Docling reading-order graph is too deep")
        if not isinstance(value, Mapping):
            return
        reference = value.get("$ref")
        entry = indexed.get(reference) if isinstance(reference, str) else None
        if entry is None:
            self_ref = value.get("self_ref")
            entry = indexed.get(self_ref) if isinstance(self_ref, str) else None
        if entry is not None:
            item, collection_name, serialized_index = entry
            visit_key = _self_ref(item, collection_name, serialized_index)
            if visit_key in visited:
                return
            visited.add(visit_key)
            if collection_name != "groups":
                ordered.append(entry)
            children = item.get("children")
            if isinstance(children, list):
                for child in children:
                    visit(child, depth + 1)
            return
        children = value.get("children")
        if isinstance(children, list):
            for child in children:
                visit(child, depth + 1)

    body = document.get("body")
    if isinstance(body, Mapping):
        visit(body, 0)
    for entry in serialized:
        item, collection_name, serialized_index = entry
        if collection_name == "groups":
            continue
        key = _self_ref(item, collection_name, serialized_index)
        if key not in visited:
            visited.add(key)
            ordered.append(entry)
    return tuple(ordered)


def _item_text(
    item: Mapping[str, Any],
    modality: BlockModality,
    indexed: Mapping[str, tuple[Mapping[str, Any], str, int]],
) -> tuple[str, str]:
    if modality is BlockModality.TABLE:
        table = _table_markdown(item)
        if table:
            return table, "docling_table_cells"
    for key, basis in (("orig", "docling_original"), ("text", "docling_text")):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value, basis
    if modality is BlockModality.IMAGE:
        captions: list[str] = []
        raw_captions = item.get("captions", [])
        if isinstance(raw_captions, list):
            for caption in raw_captions[:16]:
                if isinstance(caption, Mapping):
                    reference = caption.get("$ref")
                    entry = (
                        indexed.get(reference) if isinstance(reference, str) else None
                    )
                    if entry is not None:
                        caption_text = entry[0].get("orig") or entry[0].get("text")
                        if isinstance(caption_text, str) and caption_text.strip():
                            captions.append(caption_text.strip())
        annotations = item.get("annotations", [])
        if isinstance(annotations, list):
            for annotation in annotations[:16]:
                if not isinstance(annotation, Mapping):
                    continue
                for key in ("text", "description", "caption"):
                    value = annotation.get(key)
                    if isinstance(value, str) and value.strip():
                        captions.append(value.strip())
                        break
        if captions:
            return "\n".join(dict.fromkeys(captions)), "docling_image_caption"
    if modality is BlockModality.KEY_VALUE:
        data = item.get("data")
        if isinstance(data, Mapping):
            pairs = data.get("pairs") or data.get("items")
            if isinstance(pairs, list):
                lines: list[str] = []
                for pair in pairs[:1_000]:
                    if not isinstance(pair, Mapping):
                        continue
                    key = pair.get("key")
                    value = pair.get("value")
                    if isinstance(key, str) and isinstance(value, str):
                        lines.append(f"{key.strip()}: {value.strip()}")
                if lines:
                    return "\n".join(lines), "docling_key_value_pairs"
    return "", "none"


def _table_markdown(item: Mapping[str, Any]) -> str:
    data = item.get("data")
    if not isinstance(data, Mapping):
        return ""
    cells = data.get("table_cells")
    if not isinstance(cells, list) or not cells:
        return ""
    if len(cells) > MAX_TABLE_CELLS:
        raise RestructureError("Docling table exceeds its cell limit")
    coordinates: list[tuple[int, int, str, bool]] = []
    max_row = -1
    max_column = -1
    for fallback_index, cell in enumerate(cells):
        if not isinstance(cell, Mapping):
            continue
        text = cell.get("text")
        if not isinstance(text, str):
            continue
        row = cell.get("start_row_offset_idx", cell.get("row", fallback_index))
        column = cell.get("start_col_offset_idx", cell.get("col", 0))
        if type(row) is not int or type(column) is not int or row < 0 or column < 0:
            continue
        if row >= 1_000 or column >= 256:
            raise RestructureError("Docling table dimensions exceed their limit")
        coordinates.append(
            (row, column, _table_cell(text), cell.get("column_header") is True)
        )
        max_row = max(max_row, row)
        max_column = max(max_column, column)
    if not coordinates:
        return ""
    grid = [["" for _ in range(max_column + 1)] for _ in range(max_row + 1)]
    header_rows: set[int] = set()
    for row, column, text, is_header in coordinates:
        if grid[row][column]:
            grid[row][column] += " / " + text
        else:
            grid[row][column] = text
        if is_header:
            header_rows.add(row)
    header_index = min(header_rows) if header_rows else 0
    header = grid[header_index]
    body = [row for index, row in enumerate(grid) if index != header_index]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _provenance(
    item: Mapping[str, Any],
    self_ref: str,
    *,
    allow_charspan: bool,
) -> tuple[Provenance, ...]:
    result: list[Provenance] = []
    values = item.get("prov", [])
    if isinstance(values, list):
        for value in values[:1_000]:
            if not isinstance(value, Mapping):
                continue
            page_no = value.get("page_no")
            if type(page_no) is not int or page_no < 1:
                page_no = None
            bbox, origin = _bbox(value.get("bbox"))
            charspan = _charspan(value.get("charspan")) if allow_charspan else None
            result.append(
                Provenance(
                    self_ref=self_ref,
                    page_no=page_no,
                    bbox=bbox,
                    charspan=charspan,
                    coord_origin=origin,
                )
            )
    if not result:
        result.append(Provenance(self_ref=self_ref))
    return unique_provenance(result)


def _bbox(value: Any) -> tuple[tuple[float, float, float, float] | None, str | None]:
    if not isinstance(value, Mapping):
        return None, None
    coordinates: list[float] = []
    for key in ("l", "t", "r", "b"):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None, None
        number = float(raw)
        if not float("-inf") < number < float("inf"):
            return None, None
        coordinates.append(number)
    origin = value.get("coord_origin")
    return (
        (coordinates[0], coordinates[1], coordinates[2], coordinates[3]),
        origin if isinstance(origin, str) and len(origin) <= 64 else None,
    )


def _charspan(value: Any) -> tuple[int, int] | None:
    if isinstance(value, Mapping):
        start, end = value.get("start"), value.get("end")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            return None
        start, end = value
    else:
        return None
    if type(start) is int and type(end) is int and 0 <= start < end:
        return start, end
    return None


def _image_data_uri(item: Mapping[str, Any]) -> str | None:
    image = item.get("image")
    candidates: list[Any] = []
    if isinstance(image, str):
        candidates.append(image)
    elif isinstance(image, Mapping):
        candidates.extend((image.get("uri"), image.get("data_uri"), image.get("data")))
    for value in candidates:
        if not isinstance(value, str) or len(value) > MAX_IMAGE_DATA_URI_CHARS:
            continue
        if _DATA_IMAGE.fullmatch(value) is not None:
            return value
    return None


def _modality(collection_name: str, label: str) -> BlockModality:
    if collection_name == "tables" or label == "table":
        return BlockModality.TABLE
    if collection_name == "pictures" or label in {"picture", "figure"}:
        return BlockModality.IMAGE
    if collection_name == "key_value_items":
        return BlockModality.KEY_VALUE
    return BlockModality.TEXT


def _label(item: Mapping[str, Any], collection_name: str) -> str:
    value = item.get("label")
    if isinstance(value, str) and value.strip():
        return _single_line(value.strip(), 80)
    return {
        "tables": "table",
        "pictures": "picture",
        "key_value_items": "key_value",
    }.get(collection_name, "text")


def _heading_level(item: Mapping[str, Any], label: str) -> int:
    raw = item.get("level")
    if type(raw) is int and 1 <= raw <= 16:
        return raw
    return 1 if label == "title" else 2


def _self_ref(item: Mapping[str, Any], collection_name: str, index: int) -> str:
    value = item.get("self_ref")
    if isinstance(value, str) and value.startswith("#/") and len(value) <= 512:
        return value
    return f"#/{collection_name}/{index}"


def _normalize_text(value: str) -> str:
    return value.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _single_line(value: str, limit: int) -> str:
    return " ".join(value.split())[:limit]


def _table_cell(value: str) -> str:
    return " ".join(value.replace("|", "\\|").replace("\x00", "").split())[:16_000]
