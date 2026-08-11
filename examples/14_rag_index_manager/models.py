"""Immutable contracts shared by the RAG ingestion example.

The orchestration agent is intentionally separated from these deterministic
records.  External clients may enrich or embed the records, but source text,
provenance, identifiers, and pipeline revisions remain application-owned.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class RAGIndexError(RuntimeError):
    """Base class for bounded, user-facing ingestion failures."""


class ScanError(RAGIndexError):
    """A document directory violates the application-owned scan policy."""


class CatalogError(RAGIndexError):
    """The manifest catalog cannot complete a consistent operation."""


class RestructureError(RAGIndexError):
    """A DoclingDocument cannot be converted into retrieval blocks."""


class LayoutRefinementError(RAGIndexError):
    """A visual layout patch is incomplete, unsafe, or changes source content."""


class ChunkingError(RAGIndexError):
    """Structured blocks cannot be converted into bounded chunks."""


class PlanningError(RAGIndexError):
    """Scanned and manifest inputs cannot form an unambiguous sync plan."""


class ChangeKind(str, Enum):
    """Why a source participates in an incremental synchronization."""

    NEW = "new"
    MODIFIED = "modified"
    PIPELINE_CHANGED = "pipeline_changed"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


class ProcessingStage(str, Enum):
    """Earliest deterministic stage that must run for one source."""

    PARSE = "parse"
    RESTRUCTURE = "restructure"
    REFINE_LAYOUT = "refine_layout"
    ENRICH = "enrich"
    CHUNK = "chunk"
    EMBED = "embed"
    INDEX = "index"
    DELETE = "delete"
    NONE = "none"


class GenerationState(str, Enum):
    """Catalog state of one immutable Milvus/index generation."""

    BUILDING = "building"
    STAGED = "staged"
    PUBLISHED = "published"
    AVAILABLE = "available"
    FAILED = "failed"


class RunState(str, Enum):
    """Durable ingestion run state."""

    RUNNING = "running"
    STAGED = "staged"
    PUBLISHED = "published"
    FAILED = "failed"


class BlockModality(str, Enum):
    """Retrieval modality emitted by the Docling restructuring layer."""

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    KEY_VALUE = "key_value"


def _require_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a safe identifier")


def stable_digest(*values: object) -> str:
    """Hash length-delimited values without delimiter ambiguity."""

    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def component_fingerprint(name: str, **configuration: object) -> str:
    """Return a stable fingerprint for a named model, parser, or policy."""

    if not isinstance(name, str) or not name.strip():
        raise ValueError("fingerprint name must not be empty")
    payload = json.dumps(
        configuration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return f"sha256:{stable_digest(name.strip(), payload)}"


@dataclass(frozen=True, slots=True)
class PipelineFingerprint:
    """Per-stage revisions used to find the earliest invalid cached stage."""

    parser: str
    restructuring: str
    enrichment: str
    chunking: str
    embedding: str
    indexing: str
    # ``None`` reads pre-v2 manifests and preserves construction compatibility.
    # New application composition should always persist the expected dimension.
    embedding_dimension: int | None = None
    # ``None`` preserves the pre-visual-refinement pipeline contract and digest.
    layout_refinement: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "parser",
            "restructuring",
            "enrichment",
            "chunking",
            "embedding",
            "indexing",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 512:
                raise ValueError(f"{name} fingerprint must be non-empty and bounded")
        if self.embedding_dimension is not None and (
            type(self.embedding_dimension) is not int
            or not 1 <= self.embedding_dimension <= 65_536
        ):
            raise ValueError("embedding_dimension must be between one and 65536")
        if self.layout_refinement is not None and (
            not isinstance(self.layout_refinement, str)
            or not self.layout_refinement.strip()
            or len(self.layout_refinement) > 512
        ):
            raise ValueError(
                "layout_refinement fingerprint must be non-empty and bounded"
            )

    @property
    def digest(self) -> str:
        legacy_components = (
            self.parser,
            self.restructuring,
            self.enrichment,
            self.chunking,
            self.embedding,
            self.indexing,
        )
        if self.embedding_dimension is None and self.layout_refinement is None:
            # Preserve v1 generation handles for callers that have not opted
            # into the dimension-aware contract yet.
            return stable_digest(*legacy_components)
        if self.layout_refinement is None:
            return stable_digest(*legacy_components, self.embedding_dimension)
        return stable_digest(
            *legacy_components,
            self.embedding_dimension,
            self.layout_refinement,
        )

    def as_dict(self) -> dict[str, str | int | None]:
        return {
            "parser": self.parser,
            "restructuring": self.restructuring,
            "enrichment": self.enrichment,
            "chunking": self.chunking,
            "embedding": self.embedding,
            "indexing": self.indexing,
            "embedding_dimension": self.embedding_dimension,
            "layout_refinement": self.layout_refinement,
        }


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One regular source file captured by a bounded directory scan."""

    kb_id: str
    source_id: str
    root: Path = field(repr=False)
    path: Path = field(repr=False)
    relative_path: str
    media_type: str
    size_bytes: int
    mtime_ns: int
    sha256: str
    device: int = field(repr=False)
    inode: int = field(repr=False)

    def __post_init__(self) -> None:
        _require_identifier(self.kb_id, "kb_id")
        if not self.source_id.startswith("src_") or len(self.source_id) != 36:
            raise ValueError("source_id must be a deterministic opaque ID")
        if not self.path.is_absolute() or not self.root.is_absolute():
            raise ValueError("source paths must be absolute")
        if not self.relative_path or self.relative_path.startswith("/"):
            raise ValueError("relative_path must be a non-empty relative path")
        if "\\" in self.relative_path or any(
            part in {"", ".", ".."} for part in self.relative_path.split("/")
        ):
            raise ValueError("relative_path must be normalized POSIX text")
        if self.size_bytes < 1 or self.mtime_ns < 0:
            raise ValueError("source size and mtime must be valid")
        if len(self.sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.sha256
        ):
            raise ValueError("source sha256 must be lowercase hexadecimal")

    @property
    def filename(self) -> str:
        return Path(self.relative_path).name

    @property
    def source_revision(self) -> str:
        return self.sha256


@dataclass(frozen=True, slots=True)
class Provenance:
    """One exact Docling location retained through restructuring/chunking."""

    self_ref: str
    page_no: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    charspan: tuple[int, int] | None = None
    coord_origin: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.self_ref, str)
            or not self.self_ref
            or len(self.self_ref) > 512
        ):
            raise ValueError("provenance self_ref must be non-empty and bounded")
        if self.page_no is not None and self.page_no < 1:
            raise ValueError("page_no must be positive")
        if self.charspan is not None:
            start, end = self.charspan
            if type(start) is not int or type(end) is not int or not 0 <= start < end:
                raise ValueError("charspan must be a valid half-open interval")
        if self.bbox is not None and (
            len(self.bbox) != 4
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not float("-inf") < float(value) < float("inf")
                for value in self.bbox
            )
        ):
            raise ValueError("bbox must contain four finite coordinates")
        if self.coord_origin is not None and (
            not isinstance(self.coord_origin, str) or len(self.coord_origin) > 64
        ):
            raise ValueError("coord_origin must be bounded text")


@dataclass(frozen=True, slots=True)
class StructuredBlock:
    """A retrieval-friendly, source-faithful Docling block."""

    block_id: str
    source_id: str
    source_revision: str
    ordinal: int
    modality: BlockModality
    label: str
    text: str
    section_path: tuple[str, ...]
    provenance: tuple[Provenance, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)
    image_data_uri: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.block_id.startswith("blk_") or len(self.block_id) != 36:
            raise ValueError("block_id must be a deterministic opaque ID")
        if not self.source_id.startswith("src_"):
            raise ValueError("source_id is invalid")
        if self.ordinal < 0:
            raise ValueError("block ordinal cannot be negative")
        if not isinstance(self.modality, BlockModality):
            raise TypeError("block modality must be a BlockModality")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("block text must not be empty")
        if len(self.text) > 2_000_000:
            raise ValueError("block text exceeds its safety limit")
        if len(self.section_path) > 16 or any(
            not value or len(value) > 500 for value in self.section_path
        ):
            raise ValueError("section_path is invalid")
        if not self.provenance or any(
            not isinstance(value, Provenance) for value in self.provenance
        ):
            raise ValueError("block provenance must be non-empty and valid")
        normalized = {
            str(key): str(value)
            for key, value in sorted(self.metadata.items())
            if str(key) and len(str(key)) <= 128 and len(str(value)) <= 4_096
        }
        if len(normalized) != len(self.metadata) or len(normalized) > 64:
            raise ValueError("block metadata is invalid or exceeds its limit")
        object.__setattr__(self, "metadata", MappingProxyType(normalized))


LAYOUT_ROLES = frozenset(
    {
        "title",
        "section_header",
        "body",
        "caption",
        "note",
        "figure",
        "table",
        "list",
        "list_item",
        "code",
        "formula",
        "key_value",
        "header",
        "footer",
        "other",
    }
)
LAYOUT_EXCLUSION_REASONS = frozenset(
    {"decorative", "repeated_header", "repeated_footer"}
)


@dataclass(frozen=True, slots=True)
class PageCapture:
    """One bounded Docling whole-page image supplied to the layout VLM."""

    page_no: int
    image_data_uri: str = field(repr=False)
    width: float | None = None
    height: float | None = None

    def __post_init__(self) -> None:
        if type(self.page_no) is not int or self.page_no < 1:
            raise ValueError("page capture number must be positive")
        if (
            not isinstance(self.image_data_uri, str)
            or not self.image_data_uri.startswith("data:image/")
            or ";base64," not in self.image_data_uri[:128]
            or len(self.image_data_uri) > 28_000_000
        ):
            raise ValueError("page capture must be a bounded image data URI")
        for name in ("width", "height"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < float(value) < float("inf")
            ):
                raise ValueError(f"page capture {name} must be finite and positive")
            if value is not None:
                object.__setattr__(self, name, float(value))


@dataclass(frozen=True, slots=True)
class LayoutPatch:
    """A layout-only VLM proposal over every block assigned to one page."""

    page_no: int
    ordered_block_ids: tuple[str, ...]
    parent_by_block: Mapping[str, str | None]
    section_heading_ids_by_block: Mapping[str, tuple[str, ...]]
    role_by_block: Mapping[str, str | None]
    group_by_block: Mapping[str, str | None]
    excluded_reason_by_block: Mapping[str, str | None]
    model_fingerprint: str

    def __post_init__(self) -> None:
        if type(self.page_no) is not int or self.page_no < 1:
            raise ValueError("layout patch page number must be positive")
        ids = tuple(self.ordered_block_ids)
        if (
            not ids
            or len(ids) > 20_000
            or len(set(ids)) != len(ids)
            or any(
                not isinstance(value, str) or not value.startswith("blk_")
                for value in ids
            )
        ):
            raise ValueError("layout patch block order is invalid")
        expected = set(ids)
        parent = _layout_mapping(self.parent_by_block, expected, "parent")
        groups = _layout_mapping(self.group_by_block, expected, "group")
        roles = _layout_mapping(self.role_by_block, expected, "role")
        exclusions = _layout_mapping(
            self.excluded_reason_by_block, expected, "excluded reason"
        )
        sections = _layout_section_id_mapping(
            self.section_heading_ids_by_block, expected
        )
        if any(
            value is not None and value not in expected for value in parent.values()
        ):
            raise ValueError("layout parents must reference a same-page block")
        if any(
            value is not None and value not in expected for value in groups.values()
        ):
            raise ValueError("layout groups must reference a same-page block")
        if any(key == value for key, value in parent.items() if value is not None):
            raise ValueError("a layout block cannot be its own parent")
        if any(
            value is not None and value not in LAYOUT_ROLES for value in roles.values()
        ):
            raise ValueError("layout patch contains an unsupported role")
        if any(
            value is not None and value not in LAYOUT_EXCLUSION_REASONS
            for value in exclusions.values()
        ):
            raise ValueError("layout patch contains an unsupported exclusion reason")
        if (
            not isinstance(self.model_fingerprint, str)
            or not self.model_fingerprint.strip()
            or len(self.model_fingerprint) > 512
        ):
            raise ValueError("layout model fingerprint must be non-empty and bounded")
        object.__setattr__(self, "ordered_block_ids", ids)
        object.__setattr__(self, "parent_by_block", parent)
        object.__setattr__(self, "section_heading_ids_by_block", sections)
        object.__setattr__(self, "role_by_block", roles)
        object.__setattr__(self, "group_by_block", groups)
        object.__setattr__(self, "excluded_reason_by_block", exclusions)


@dataclass(frozen=True, slots=True)
class LayoutRefinementPolicy:
    """Application-owned safety policy for model-proposed content exclusion."""

    allow_exclusions: bool = False

    def __post_init__(self) -> None:
        if type(self.allow_exclusions) is not bool:
            raise TypeError("allow_exclusions must be a bool")


class LayoutRefiner(Protocol):
    """Replaceable vision-model contract that can only propose layout patches."""

    @property
    def fingerprint(self) -> str: ...

    async def refine(
        self,
        source: SourceDocument,
        blocks: Sequence[StructuredBlock],
        captures: Sequence[PageCapture],
    ) -> tuple[LayoutPatch, ...]: ...


@dataclass(frozen=True, slots=True)
class BlockEnrichment:
    """Model-produced retrieval hints; the original block remains immutable."""

    block_id: str
    summary: str = ""
    keywords: tuple[str, ...] = ()
    image_description: str = ""
    embedding_text: str = ""
    model_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.block_id.startswith("blk_"):
            raise ValueError("enrichment block_id is invalid")
        if len(self.summary) > 4_000 or len(self.image_description) > 8_000:
            raise ValueError("enrichment text exceeds its limit")
        if len(self.embedding_text) > 24_000:
            raise ValueError("embedding_text exceeds its limit")
        if len(self.keywords) > 64 or any(
            not item.strip() or len(item) > 200 for item in self.keywords
        ):
            raise ValueError("enrichment keywords are invalid")


@dataclass(frozen=True, slots=True)
class Chunk:
    """One deterministic unit sent to BGE-M3 and Milvus."""

    chunk_id: str
    kb_id: str
    source_id: str
    source_revision: str
    ordinal: int
    content: str
    embedding_text: str
    section_path: tuple[str, ...]
    block_ids: tuple[str, ...]
    provenance: tuple[Provenance, ...]
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.chunk_id.startswith("chk_") or len(self.chunk_id) != 36:
            raise ValueError("chunk_id must be a deterministic opaque ID")
        _require_identifier(self.kb_id, "kb_id")
        if (
            self.ordinal < 0
            or not self.content.strip()
            or not self.embedding_text.strip()
        ):
            raise ValueError("chunk content and ordinal are invalid")
        if not self.block_ids or len(set(self.block_ids)) != len(self.block_ids):
            raise ValueError("chunk block IDs must be non-empty and unique")
        if len(self.section_path) > 16 or any(
            not value or len(value) > 500 for value in self.section_path
        ):
            raise ValueError("chunk section_path is invalid")
        if not self.provenance or any(
            not isinstance(value, Provenance) for value in self.provenance
        ):
            raise ValueError("chunk provenance must be non-empty and valid")
        normalized = {
            str(key): str(value)
            for key, value in sorted(self.metadata.items())
            if str(key) and len(str(key)) <= 128 and len(str(value)) <= 4_096
        }
        if len(normalized) != len(self.metadata) or len(normalized) > 64:
            raise ValueError("chunk metadata is invalid or exceeds its limit")
        object.__setattr__(self, "metadata", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class ManifestDocument:
    """One source snapshot stored in an immutable catalog generation."""

    generation_id: str
    generation_state: GenerationState
    kb_id: str
    source_id: str
    relative_path: str
    media_type: str
    size_bytes: int
    mtime_ns: int
    content_sha256: str
    pipeline: PipelineFingerprint
    chunk_count: int

    def __post_init__(self) -> None:
        if (
            not self.generation_id.startswith("gen_")
            or _IDENTIFIER.fullmatch(self.generation_id) is None
        ):
            raise ValueError("manifest generation_id is invalid")
        if not isinstance(self.generation_state, GenerationState):
            raise TypeError("manifest generation_state must be a GenerationState")
        _require_identifier(self.kb_id, "kb_id")
        if not self.source_id.startswith("src_") or len(self.source_id) != 36:
            raise ValueError("manifest source_id is invalid")
        if not self.relative_path or self.relative_path.startswith("/"):
            raise ValueError("manifest relative_path is invalid")
        if self.size_bytes < 1 or self.mtime_ns < 0 or self.chunk_count < 0:
            raise ValueError("manifest numeric fields are invalid")
        if len(self.content_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.content_sha256
        ):
            raise ValueError("manifest content_sha256 is invalid")
        if not isinstance(self.pipeline, PipelineFingerprint):
            raise TypeError("manifest pipeline must be a PipelineFingerprint")


@dataclass(frozen=True, slots=True)
class SyncAction:
    """One deterministic item in an incremental sync plan."""

    source_id: str
    relative_path: str
    change: ChangeKind
    start_stage: ProcessingStage
    reason: str
    source: SourceDocument | None = field(default=None, repr=False)
    previous: ManifestDocument | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class SyncPlan:
    """Ordered, immutable plan over one knowledge base."""

    kb_id: str
    pipeline: PipelineFingerprint
    actions: tuple[SyncAction, ...]

    @property
    def is_noop(self) -> bool:
        return all(action.change is ChangeKind.UNCHANGED for action in self.actions)

    def requiring_work(self) -> tuple[SyncAction, ...]:
        return tuple(
            action
            for action in self.actions
            if action.change is not ChangeKind.UNCHANGED
        )


def unique_provenance(values: Sequence[Provenance]) -> tuple[Provenance, ...]:
    """Deduplicate provenance while preserving Docling reading order."""

    result: list[Provenance] = []
    seen: set[Provenance] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def apply_layout_patches(
    blocks: Sequence[StructuredBlock],
    patches: Sequence[LayoutPatch],
    *,
    captured_page_nos: Sequence[int] | None = None,
    policy: LayoutRefinementPolicy | None = None,
) -> tuple[StructuredBlock, ...]:
    """Verify and apply layout-only patches without changing source content.

    A multi-page block belongs to its lowest provenance page. Pages without a
    supplied capture are deliberately left untouched. Excluded blocks remain in
    the raw artifact and in each patch's exact permutation, but are omitted from
    the returned retrieval blocks only when the application policy opts in.
    """

    raw = tuple(blocks)
    proposals = tuple(patches)
    resolved_policy = policy or LayoutRefinementPolicy()
    if (
        not raw
        or len(raw) > 100_000
        or any(not isinstance(item, StructuredBlock) for item in raw)
    ):
        raise LayoutRefinementError("raw layout blocks are empty or invalid")
    if len(proposals) > 10_000 or any(
        not isinstance(item, LayoutPatch) for item in proposals
    ):
        raise LayoutRefinementError("layout patches are invalid or exceed their limit")
    if not isinstance(resolved_policy, LayoutRefinementPolicy):
        raise TypeError("policy must be a LayoutRefinementPolicy")

    by_id = {item.block_id: item for item in raw}
    if len(by_id) != len(raw):
        raise LayoutRefinementError("raw layout blocks contain duplicate IDs")
    source_keys = {(item.source_id, item.source_revision) for item in raw}
    if len(source_keys) != 1:
        raise LayoutRefinementError("raw layout blocks span multiple source revisions")
    primary_by_id = {item.block_id: _primary_page(item) for item in raw}
    ids_by_page: dict[int, list[str]] = {}
    for item in raw:
        page_no = primary_by_id[item.block_id]
        if page_no is not None:
            ids_by_page.setdefault(page_no, []).append(item.block_id)

    patch_by_page: dict[int, LayoutPatch] = {}
    for patch in proposals:
        if patch.page_no in patch_by_page:
            raise LayoutRefinementError("layout patches contain duplicate pages")
        expected_ids = set(ids_by_page.get(patch.page_no, ()))
        if not expected_ids or set(patch.ordered_block_ids) != expected_ids:
            raise LayoutRefinementError(
                "layout patch is not an exact permutation of its primary-page blocks"
            )
        patch_by_page[patch.page_no] = patch

    if captured_page_nos is not None:
        capture_pages = tuple(captured_page_nos)
        if any(type(value) is not int or value < 1 for value in capture_pages):
            raise LayoutRefinementError("captured page numbers are invalid")
        if len(set(capture_pages)) != len(capture_pages):
            raise LayoutRefinementError("captured page numbers contain duplicates")
        expected_patch_pages = {
            page_no for page_no in capture_pages if page_no in ids_by_page
        }
        if set(patch_by_page) != expected_patch_pages:
            raise LayoutRefinementError(
                "layout patches do not exactly cover captured pages with blocks"
            )

    fingerprints = {item.model_fingerprint for item in proposals}
    if len(fingerprints) > 1:
        raise LayoutRefinementError("layout patches use multiple model revisions")
    excluded_ids = {
        block_id
        for patch in proposals
        for block_id, reason in patch.excluded_reason_by_block.items()
        if reason is not None
    }
    if excluded_ids and not resolved_policy.allow_exclusions:
        raise LayoutRefinementError(
            "layout exclusion is disabled by application policy"
        )
    for patch in proposals:
        _verify_layout_parent_graph(patch)
        for block_id, heading_ids in patch.section_heading_ids_by_block.items():
            for heading_id in heading_ids:
                heading = by_id[heading_id]
                raw_role = heading.label.strip().lower().replace(" ", "_")
                refined_role = patch.role_by_block[heading_id]
                if raw_role not in {"title", "section_header"} and refined_role not in {
                    "title",
                    "section_header",
                }:
                    raise LayoutRefinementError(
                        "layout section path references a non-heading block"
                    )
        for mapping in (patch.parent_by_block, patch.group_by_block):
            if any(
                value in excluded_ids for value in mapping.values() if value is not None
            ):
                raise LayoutRefinementError(
                    "layout relation cannot target an excluded block"
                )

    ordered: list[StructuredBlock] = []
    emitted_pages: set[int] = set()
    for block in raw:
        page_no = primary_by_id[block.block_id]
        patch = None if page_no is None else patch_by_page.get(page_no)
        if patch is None:
            ordered.append(block)
            continue
        if page_no in emitted_pages:
            continue
        emitted_pages.add(page_no)
        for block_id in patch.ordered_block_ids:
            if block_id in excluded_ids:
                continue
            original = by_id[block_id]
            metadata = dict(original.metadata)
            for key in (
                "layout_parent_block_id",
                "layout_group_block_id",
                "layout_role",
            ):
                metadata.pop(key, None)
            parent = patch.parent_by_block[block_id]
            group = patch.group_by_block[block_id]
            role = patch.role_by_block[block_id]
            if parent is not None:
                metadata["layout_parent_block_id"] = parent
            if group is not None:
                metadata["layout_group_block_id"] = group
            if role is not None:
                metadata["layout_role"] = role
            heading_ids = patch.section_heading_ids_by_block[block_id]
            section_path = (
                tuple(_layout_heading_text(by_id[value].text) for value in heading_ids)
                if heading_ids
                else original.section_path
            )
            ordered.append(
                replace(
                    original,
                    section_path=section_path,
                    metadata=metadata,
                )
            )
    if not ordered:
        raise LayoutRefinementError("layout refinement removed every retrieval block")
    return tuple(replace(item, ordinal=index) for index, item in enumerate(ordered))


def _layout_mapping(
    value: Mapping[str, str | None], expected: set[str], name: str
) -> Mapping[str, str | None]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"layout {name} keys must exactly match ordered blocks")
    normalized: dict[str, str | None] = {}
    for key in sorted(expected):
        item = value[key]
        if item is not None and (
            not isinstance(item, str) or not item or len(item) > 512
        ):
            raise ValueError(f"layout {name} values must be bounded strings or null")
        normalized[key] = item
    return MappingProxyType(normalized)


def _layout_section_id_mapping(
    value: Mapping[str, tuple[str, ...]], expected: set[str]
) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(
            "layout section heading keys must exactly match ordered blocks"
        )
    normalized: dict[str, tuple[str, ...]] = {}
    for key in sorted(expected):
        raw_ids = value[key]
        if isinstance(raw_ids, (str, bytes)) or not isinstance(raw_ids, Sequence):
            raise ValueError("layout section heading IDs must be arrays")
        heading_ids = tuple(raw_ids)
        if (
            len(heading_ids) > 16
            or len(set(heading_ids)) != len(heading_ids)
            or any(
                not isinstance(item, str)
                or not item.startswith("blk_")
                or item not in expected
                for item in heading_ids
            )
        ):
            raise ValueError("layout section heading IDs are invalid")
        normalized[key] = heading_ids
    return MappingProxyType(normalized)


def _layout_heading_text(value: str) -> str:
    return " ".join(value.split())[:500]


def _primary_page(block: StructuredBlock) -> int | None:
    pages = {item.page_no for item in block.provenance if item.page_no is not None}
    return min(pages) if pages else None


def _verify_layout_parent_graph(patch: LayoutPatch) -> None:
    for start in patch.ordered_block_ids:
        seen: set[str] = set()
        current: str | None = start
        while current is not None:
            if current in seen:
                raise LayoutRefinementError(
                    "layout parent relationships contain a cycle"
                )
            seen.add(current)
            current = patch.parent_by_block[current]


def safe_metadata(value: Mapping[str, Any] | None) -> Mapping[str, str]:
    """Normalize bounded scalar metadata for storage and model boundaries."""

    if value is None:
        return MappingProxyType({})
    normalized: dict[str, str] = {}
    for raw_key, raw_value in sorted(value.items(), key=lambda item: str(item[0])):
        key = str(raw_key)
        if not key or len(key) > 128 or len(normalized) >= 64:
            continue
        if isinstance(raw_value, (str, int, float, bool)) or raw_value is None:
            text = str(raw_value)
            if len(text) <= 4_096:
                normalized[key] = text
    return MappingProxyType(normalized)
