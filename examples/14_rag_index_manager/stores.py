"""Milvus blue/green generation stores for the RAG index manager example.

``InMemoryMilvusStore`` is the deterministic offline seam. ``MilvusStore``
imports ``pymilvus`` only when its first operation needs a real client.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol
from urllib.parse import urlsplit

from .backends import EmbeddingBatch
from .models import Chunk, PipelineFingerprint, component_fingerprint, stable_digest


MAX_INDEX_ROWS = 500_000
MAX_COPY_ROWS = 500_000
MAX_DELETE_IDS = 100_000
MAX_MANAGED_COLLECTIONS = 1_000
MAX_FILTER_IDS = 256
MAX_TEXT_FIELD_CHARS = 65_000
MAX_HYBRID_QUERY_CHARS = 4_096
MAX_HYBRID_TOKENS = 2_048
MAX_HYBRID_CANDIDATES = 500
_SAFE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_SAFE_GENERATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class VectorStoreError(RuntimeError):
    """A vector-store operation failed without exposing backend details."""


@dataclass(frozen=True, slots=True)
class StagingGeneration:
    generation_id: str
    collection_name: str
    dimension: int
    pipeline_fingerprint: str


@dataclass(frozen=True, slots=True)
class CopyResult:
    source_ids: tuple[str, ...]
    copied_rows: int


@dataclass(frozen=True, slots=True)
class IndexValidation:
    collection_name: str
    valid: bool
    row_count: int
    dimension: int
    missing_chunk_ids: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PublishedGeneration:
    alias: str
    collection_name: str
    previous_collection: str | None


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Content-free retrieval result used by index quality evaluation."""

    chunk_id: str
    source_id: str
    score: float

    def __post_init__(self) -> None:
        _identifiers((self.chunk_id,), "chunk_id", 1)
        _identifiers((self.source_id,), "source_id", 1)
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise TypeError("score must be a number")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")


class VectorStore(Protocol):
    """Replaceable blue/green vector index contract."""

    @property
    def fingerprint(self) -> str: ...

    async def create_staging(
        self,
        generation_id: str,
        dimension: int,
        pipeline_fingerprint: PipelineFingerprint | str,
        *,
        copy_from_active: bool = False,
    ) -> StagingGeneration: ...

    async def copy_sources_to_staging(
        self, staging: StagingGeneration, source_ids: Sequence[str]
    ) -> CopyResult: ...

    async def upsert(
        self,
        staging: StagingGeneration,
        chunks: Sequence[Chunk],
        embeddings: EmbeddingBatch | Sequence[Sequence[float]],
    ) -> int: ...

    async def delete(
        self, staging: StagingGeneration, chunk_ids: Sequence[str]
    ) -> int: ...

    async def delete_sources(
        self, staging: StagingGeneration, source_ids: Sequence[str]
    ) -> int: ...

    async def validate(
        self,
        staging: StagingGeneration,
        *,
        expected_chunk_ids: Sequence[str] | None = None,
        expected_count: int | None = None,
    ) -> IndexValidation: ...

    async def publish(self, staging: StagingGeneration) -> PublishedGeneration: ...

    async def rollback(self) -> PublishedGeneration: ...

    async def current_alias(self) -> str | None: ...

    async def active_generation_id(self) -> str | None: ...

    async def search(
        self, query_vector: Sequence[float], *, limit: int = 5
    ) -> tuple[SearchHit, ...]: ...

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
    ) -> tuple[SearchHit, ...]: ...

    async def drop_staging(self, staging: StagingGeneration) -> None: ...

    async def aclose(self) -> None: ...


@dataclass(slots=True)
class _MemoryCollection:
    staging: StagingGeneration
    rows: dict[str, dict[str, Any]] = field(default_factory=dict)


class InMemoryMilvusStore:
    """Concurrency-safe offline implementation of the Milvus store contract."""

    def __init__(self, *, alias: str = "assistant_kb_active") -> None:
        _require_name(alias, "alias")
        self.alias = alias
        self._previous_alias = f"{alias}_previous"
        _require_name(self._previous_alias, "previous alias")
        self._collections: dict[str, _MemoryCollection] = {}
        self._aliases: dict[str, str] = {}
        self._validated: set[str] = set()
        self._lock = asyncio.Lock()

    @property
    def fingerprint(self) -> str:
        return component_fingerprint(
            "milvus-index",
            implementation="memory-compatible-v1",
            metric="COSINE",
            index="AUTOINDEX",
            schema_revision="rag-chunk-v1",
        )

    async def __aenter__(self) -> InMemoryMilvusStore:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        return None

    async def create_staging(
        self,
        generation_id: str,
        dimension: int,
        pipeline_fingerprint: PipelineFingerprint | str,
        *,
        copy_from_active: bool = False,
    ) -> StagingGeneration:
        staging = staging_handle("rag", generation_id, dimension, pipeline_fingerprint)
        async with self._lock:
            existing = self._collections.get(staging.collection_name)
            if existing is not None:
                if existing.staging != staging:
                    raise VectorStoreError(
                        "staging collection conflicts with its schema"
                    )
                return staging
            collection = _MemoryCollection(staging)
            if copy_from_active:
                active = self._aliases.get(self.alias)
                if active is not None:
                    active_collection = self._collections[active]
                    if active_collection.staging.dimension != staging.dimension:
                        raise VectorStoreError(
                            "active vectors cannot be copied across dimensions"
                        )
                    collection.rows = {
                        key: {
                            **value,
                            "generation_id": staging.generation_id,
                            "pipeline_fingerprint": _stored_fingerprint(
                                staging.pipeline_fingerprint
                            ),
                        }
                        for key, value in active_collection.rows.items()
                    }
            self._collections[staging.collection_name] = collection
            return staging

    async def copy_sources_to_staging(
        self, staging: StagingGeneration, source_ids: Sequence[str]
    ) -> CopyResult:
        identifiers = _identifiers(source_ids, "source_ids", MAX_DELETE_IDS)
        async with self._lock:
            target = self._collection(staging)
            active_name = self._aliases.get(self.alias)
            if active_name is None:
                return CopyResult(identifiers, 0)
            active = self._collections[active_name]
            if active.staging.dimension != staging.dimension:
                raise VectorStoreError(
                    "active vectors cannot be copied across dimensions"
                )
            selected = set(identifiers)
            additions = {
                chunk_id: {
                    **row,
                    "generation_id": staging.generation_id,
                    "pipeline_fingerprint": _stored_fingerprint(
                        staging.pipeline_fingerprint
                    ),
                }
                for chunk_id, row in active.rows.items()
                if row["source_id"] in selected
            }
            if len(additions) > MAX_COPY_ROWS:
                raise VectorStoreError("source copy exceeds its row limit")
            if len(set(target.rows) | set(additions)) > MAX_INDEX_ROWS:
                raise VectorStoreError("staging collection exceeds its row limit")
            target.rows.update(additions)
            self._validated.discard(staging.collection_name)
            return CopyResult(identifiers, len(additions))

    async def upsert(
        self,
        staging: StagingGeneration,
        chunks: Sequence[Chunk],
        embeddings: EmbeddingBatch | Sequence[Sequence[float]],
    ) -> int:
        rows = _rows(staging, chunks, embeddings)
        async with self._lock:
            collection = self._collection(staging)
            additions = {row["chunk_id"]: row for row in rows}
            if len(set(collection.rows) | set(additions)) > MAX_INDEX_ROWS:
                raise VectorStoreError("staging collection exceeds its row limit")
            collection.rows.update(additions)
            self._validated.discard(staging.collection_name)
            return len(rows)

    async def delete(self, staging: StagingGeneration, chunk_ids: Sequence[str]) -> int:
        identifiers = _identifiers(chunk_ids, "chunk_ids", MAX_DELETE_IDS)
        async with self._lock:
            collection = self._collection(staging)
            removed = sum(
                collection.rows.pop(value, None) is not None for value in identifiers
            )
            self._validated.discard(staging.collection_name)
            return removed

    async def delete_sources(
        self, staging: StagingGeneration, source_ids: Sequence[str]
    ) -> int:
        identifiers = set(_identifiers(source_ids, "source_ids", MAX_DELETE_IDS))
        async with self._lock:
            collection = self._collection(staging)
            targets = [
                chunk_id
                for chunk_id, row in collection.rows.items()
                if row["source_id"] in identifiers
            ]
            for chunk_id in targets:
                del collection.rows[chunk_id]
            self._validated.discard(staging.collection_name)
            return len(targets)

    async def validate(
        self,
        staging: StagingGeneration,
        *,
        expected_chunk_ids: Sequence[str] | None = None,
        expected_count: int | None = None,
    ) -> IndexValidation:
        expected = (
            None
            if expected_chunk_ids is None
            else _identifiers(expected_chunk_ids, "expected_chunk_ids", MAX_INDEX_ROWS)
        )
        _expected_count(expected_count)
        async with self._lock:
            collection = self._collection(staging)
            row_count = len(collection.rows)
            missing = (
                ()
                if expected is None
                else tuple(value for value in expected if value not in collection.rows)
            )
            errors: list[str] = []
            if row_count == 0:
                errors.append("collection is empty")
            if expected_count is not None and row_count != expected_count:
                errors.append("row count does not match the manifest")
            if expected is not None and row_count != len(expected):
                errors.append("collection contains missing or unexpected chunk IDs")
            if missing:
                errors.append("one or more expected chunk IDs are missing")
            valid = not errors
            if valid:
                self._validated.add(staging.collection_name)
            else:
                self._validated.discard(staging.collection_name)
            return IndexValidation(
                staging.collection_name,
                valid,
                row_count,
                staging.dimension,
                missing[:1000],
                tuple(errors),
            )

    async def publish(self, staging: StagingGeneration) -> PublishedGeneration:
        async with self._lock:
            self._collection(staging)
            if staging.collection_name not in self._validated:
                raise VectorStoreError(
                    "staging collection must pass validation before publish"
                )
            previous = self._aliases.get(self.alias)
            if previous is not None and previous != staging.collection_name:
                self._aliases[self._previous_alias] = previous
            self._aliases[self.alias] = staging.collection_name
            return PublishedGeneration(self.alias, staging.collection_name, previous)

    async def rollback(self) -> PublishedGeneration:
        async with self._lock:
            current = self._aliases.get(self.alias)
            previous = self._aliases.get(self._previous_alias)
            if current is None or previous is None or previous not in self._collections:
                raise VectorStoreError("no previous generation is available")
            self._aliases[self.alias] = previous
            self._aliases[self._previous_alias] = current
            return PublishedGeneration(self.alias, previous, current)

    async def current_alias(self) -> str | None:
        async with self._lock:
            return self._aliases.get(self.alias)

    async def active_generation_id(self) -> str | None:
        async with self._lock:
            collection_name = self._aliases.get(self.alias)
            if collection_name is None:
                return None
            return self._collections[collection_name].staging.generation_id

    async def search(
        self, query_vector: Sequence[float], *, limit: int = 5
    ) -> tuple[SearchHit, ...]:
        _search_limit(limit)
        async with self._lock:
            collection_name = self._aliases.get(self.alias)
            if collection_name is None:
                return ()
            collection = self._collections[collection_name]
            query = _vector(query_vector, collection.staging.dimension)
            query_norm = math.sqrt(sum(value * value for value in query))
            if query_norm == 0:
                raise VectorStoreError("query vector must not be all zero")
            hits: list[SearchHit] = []
            for row in collection.rows.values():
                vector = _vector(
                    row.get("dense_vector", ()), collection.staging.dimension
                )
                vector_norm = math.sqrt(sum(value * value for value in vector))
                if vector_norm == 0:
                    raise VectorStoreError("stored vector must not be all zero")
                score = sum(
                    query_value * vector_value
                    for query_value, vector_value in zip(query, vector)
                ) / (query_norm * vector_norm)
                hits.append(
                    SearchHit(
                        chunk_id=str(row.get("chunk_id", "")),
                        source_id=str(row.get("source_id", "")),
                        score=score,
                    )
                )
            hits.sort(key=lambda hit: (-hit.score, hit.chunk_id))
            return tuple(hits[:limit])

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
    ) -> tuple[SearchHit, ...]:
        """Rerank bounded dense candidates with private lexical overlap."""

        _hybrid_query(query_text)
        _search_limit(limit)
        candidate_limit = _hybrid_candidate_limit(limit)
        async with self._lock:
            collection_name = self._aliases.get(self.alias)
            if collection_name is None:
                return ()
            collection = self._collections[collection_name]
            query = _vector(query_vector, collection.staging.dimension)
            query_norm = math.sqrt(sum(value * value for value in query))
            if query_norm == 0:
                raise VectorStoreError("query vector must not be all zero")
            candidates: list[tuple[SearchHit, str]] = []
            for row in collection.rows.values():
                vector = _vector(
                    row.get("dense_vector", ()), collection.staging.dimension
                )
                vector_norm = math.sqrt(sum(value * value for value in vector))
                if vector_norm == 0:
                    raise VectorStoreError("stored vector must not be all zero")
                score = sum(
                    query_value * vector_value
                    for query_value, vector_value in zip(query, vector)
                ) / (query_norm * vector_norm)
                candidates.append(
                    (
                        SearchHit(
                            chunk_id=str(row.get("chunk_id", "")),
                            source_id=str(row.get("source_id", "")),
                            score=score,
                        ),
                        _stored_search_text(row.get("embedding_text")),
                    )
                )
            candidates.sort(key=lambda item: (-item[0].score, item[0].chunk_id))
            return _hybrid_rerank(
                query_text,
                tuple(candidates[:candidate_limit]),
                limit=limit,
            )

    async def drop_staging(self, staging: StagingGeneration) -> None:
        async with self._lock:
            if staging.collection_name in self._aliases.values():
                raise VectorStoreError("an aliased collection cannot be dropped")
            self._collections.pop(staging.collection_name, None)
            self._validated.discard(staging.collection_name)

    def _collection(self, staging: StagingGeneration) -> _MemoryCollection:
        value = self._collections.get(staging.collection_name)
        if value is None or value.staging != staging:
            raise VectorStoreError(
                "staging collection does not exist or does not match"
            )
        return value


class MilvusStore:
    """Lazy ``pymilvus.MilvusClient`` adapter with alias-based publication."""

    def __init__(
        self,
        *,
        uri: str | None = None,
        token: str | None = None,
        database: str = "default",
        alias: str = "assistant_kb_active",
        collection_prefix: str = "assistant_kb",
        index_type: str = "AUTOINDEX",
        metric_type: str = "COSINE",
        operation_timeout: float = 60.0,
        client: Any | None = None,
    ) -> None:
        raw_uri = (
            uri
            if uri is not None
            else os.getenv(
                "RAG_MILVUS_URI",
                os.getenv("MILVUS_URI", "http://localhost:19530"),
            )
        )
        if not isinstance(raw_uri, str):
            raise TypeError("MILVUS_URI must be a string")
        configured_uri = raw_uri.rstrip("/")
        if any(ord(character) < 32 for character in configured_uri):
            raise ValueError("MILVUS_URI must not contain control characters")
        parsed = urlsplit(configured_uri)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("MILVUS_URI must be a credential-free HTTP(S) URI")
        _require_name(alias, "alias")
        _require_name(collection_prefix, "collection_prefix")
        _require_name(database, "database")
        if index_type not in {"AUTOINDEX", "HNSW"}:
            raise ValueError("index_type must be AUTOINDEX or HNSW")
        if metric_type != "COSINE":
            raise ValueError("this BGE-M3 example requires COSINE distance")
        if isinstance(operation_timeout, bool) or not isinstance(
            operation_timeout, (int, float)
        ):
            raise TypeError("operation_timeout must be a number")
        if not math.isfinite(float(operation_timeout)) or operation_timeout <= 0:
            raise ValueError("operation_timeout must be finite and positive")
        self.uri = configured_uri
        self.token = (
            token
            if token is not None
            else os.getenv("RAG_MILVUS_TOKEN", os.getenv("MILVUS_TOKEN"))
        )
        if self.token is not None and not isinstance(self.token, str):
            raise TypeError("Milvus token must be a string")
        if self.token is not None and (
            len(self.token) > 8_192
            or any(
                ord(character) < 32 or ord(character) == 127 for character in self.token
            )
        ):
            raise ValueError("Milvus token is invalid")
        self.database = database
        self.alias = alias
        self.previous_alias = f"{alias}_previous"
        _require_name(self.previous_alias, "previous alias")
        self.collection_prefix = collection_prefix
        self.index_type = index_type
        self.metric_type = metric_type
        self.operation_timeout = float(operation_timeout)
        self._client = client
        self._owns_client = client is None
        self._data_type: Any | None = None
        self._staging_handles: dict[str, StagingGeneration] = {}
        self._validated: set[str] = set()
        self._lock = asyncio.Lock()

    @property
    def fingerprint(self) -> str:
        return component_fingerprint(
            "milvus-index",
            index=self.index_type,
            metric=self.metric_type,
            schema_revision="rag-chunk-v1",
        )

    async def __aenter__(self) -> MilvusStore:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if (
            self._owns_client
            and self._client is not None
            and callable(getattr(self._client, "close", None))
        ):
            await self._run(self._client.close)
            self._client = None
            self._data_type = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                pymilvus = importlib.import_module("pymilvus")
            except ImportError as exc:
                raise VectorStoreError(
                    "pymilvus is required only when the real MilvusStore is used"
                ) from exc
            try:
                self._client = pymilvus.MilvusClient(
                    uri=self.uri,
                    token=self.token,
                    db_name=self.database,
                )
            except Exception as exc:
                raise VectorStoreError("Milvus client could not be created") from exc
            self._data_type = pymilvus.DataType
        return self._client

    async def _run(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(function, *args, **kwargs),
                timeout=self.operation_timeout,
            )
        except asyncio.TimeoutError as exc:
            raise VectorStoreError("Milvus operation exceeded its deadline") from exc
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError("Milvus operation failed") from exc

    async def create_staging(
        self,
        generation_id: str,
        dimension: int,
        pipeline_fingerprint: PipelineFingerprint | str,
        *,
        copy_from_active: bool = False,
    ) -> StagingGeneration:
        staging = staging_handle(
            self.collection_prefix, generation_id, dimension, pipeline_fingerprint
        )
        async with self._lock:
            client = self._get_client()
            exists = await self._run(
                client.has_collection,
                staging.collection_name,
                timeout=self.operation_timeout,
            )
            if not exists:
                await self._run(self._create_collection_sync, staging)
            else:
                await self._run(self._validate_collection_sync, staging)
            previous_handle = self._staging_handles.get(staging.collection_name)
            if previous_handle is not None and previous_handle != staging:
                raise VectorStoreError("staging handle conflicts with an existing run")
            self._staging_handles[staging.collection_name] = staging
        # Copy outside the creation lock because the copy operation owns the
        # same lock while it reads the active alias and writes staging.
        if copy_from_active:
            await self._copy_filter(staging, "chunk_id != ''")
        return staging

    def _create_collection_sync(self, staging: StagingGeneration) -> None:
        client = self._get_client()
        data_type = self._data_type
        if data_type is None:
            # Injected fake clients can provide their DataType equivalent.
            data_type = getattr(client, "DataType", None)
        if data_type is None:
            try:
                data_type = importlib.import_module("pymilvus").DataType
            except (ImportError, AttributeError) as exc:
                raise VectorStoreError("Milvus DataType is unavailable") from exc
            self._data_type = data_type
        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="chunk_id",
            datatype=data_type.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        for field_name, max_length in (
            ("kb_id", 128),
            ("source_id", 128),
            ("source_revision", 128),
            ("generation_id", 128),
            ("content", 65_535),
            ("embedding_text", 65_535),
            ("section_path_json", 8_192),
            ("block_ids_json", 8_192),
            ("provenance_json", 32_768),
            ("metadata_json", 32_768),
            ("pipeline_fingerprint", 128),
        ):
            schema.add_field(
                field_name=field_name,
                datatype=data_type.VARCHAR,
                max_length=max_length,
            )
        schema.add_field(
            field_name="dense_vector",
            datatype=data_type.FLOAT_VECTOR,
            dim=staging.dimension,
        )
        indexes = client.prepare_index_params()
        parameters = (
            {} if self.index_type == "AUTOINDEX" else {"M": 16, "efConstruction": 200}
        )
        indexes.add_index(
            field_name="dense_vector",
            index_type=self.index_type,
            metric_type=self.metric_type,
            params=parameters,
        )
        client.create_collection(
            collection_name=staging.collection_name,
            schema=schema,
            index_params=indexes,
            consistency_level="Bounded",
            timeout=self.operation_timeout,
        )

    def _validate_collection_sync(self, staging: StagingGeneration) -> None:
        description = self._get_client().describe_collection(
            collection_name=staging.collection_name,
            timeout=self.operation_timeout,
        )
        if not isinstance(description, Mapping):
            raise VectorStoreError("Milvus returned an invalid collection schema")
        fields = description.get("fields")
        if not isinstance(fields, Sequence):
            raise VectorStoreError("Milvus returned an invalid collection schema")
        by_name = {
            field.get("name"): field
            for field in fields
            if isinstance(field, Mapping) and isinstance(field.get("name"), str)
        }
        required = {
            "chunk_id",
            "kb_id",
            "source_id",
            "source_revision",
            "generation_id",
            "content",
            "embedding_text",
            "section_path_json",
            "block_ids_json",
            "provenance_json",
            "metadata_json",
            "pipeline_fingerprint",
            "dense_vector",
        }
        vector = by_name.get("dense_vector")
        params = vector.get("params") if isinstance(vector, Mapping) else None
        try:
            dimension = int(params.get("dim")) if isinstance(params, Mapping) else -1
        except (TypeError, ValueError):
            dimension = -1
        if not required.issubset(by_name) or dimension != staging.dimension:
            raise VectorStoreError("existing staging collection schema does not match")

    async def copy_sources_to_staging(
        self, staging: StagingGeneration, source_ids: Sequence[str]
    ) -> CopyResult:
        identifiers = _identifiers(source_ids, "source_ids", MAX_DELETE_IDS)
        if not identifiers:
            return CopyResult((), 0)
        expressions = [
            f"source_id in {json.dumps(list(identifiers[start : start + MAX_FILTER_IDS]))}"
            for start in range(0, len(identifiers), MAX_FILTER_IDS)
        ]
        copied = await self._copy_filters(staging, expressions)
        return CopyResult(identifiers, copied)

    async def _copy_filter(self, staging: StagingGeneration, expression: str) -> int:
        return await self._copy_filters(staging, (expression,))

    async def _copy_filters(
        self, staging: StagingGeneration, expressions: Sequence[str]
    ) -> int:
        async with self._lock:
            self._require_staging(staging)
            active = await self.current_alias(_already_locked=True)
            if active is None:
                return 0
            if active == staging.collection_name:
                raise VectorStoreError("cannot copy a staging collection onto itself")
            copied = await self._run(
                self._copy_filters_sync,
                active,
                staging,
                tuple(expressions),
            )
            self._validated.discard(staging.collection_name)
            return copied

    def _copy_filters_sync(
        self,
        active: str,
        staging: StagingGeneration,
        expressions: Sequence[str],
    ) -> int:
        copied = 0
        for expression in expressions:
            copied += self._copy_filter_sync(
                active,
                staging,
                expression,
                maximum=MAX_COPY_ROWS - copied,
            )
        return copied

    def _copy_filter_sync(
        self,
        source: str,
        staging: StagingGeneration,
        expression: str,
        *,
        maximum: int = MAX_COPY_ROWS,
    ) -> int:
        client = self._get_client()

        def upsert_batch(batch: Sequence[Any], copied: int) -> int:
            if copied + len(batch) > maximum:
                raise VectorStoreError("source copy exceeds its row limit")
            copied_rows = [
                {
                    **_entity_mapping(row),
                    "generation_id": staging.generation_id,
                    "pipeline_fingerprint": _stored_fingerprint(
                        staging.pipeline_fingerprint
                    ),
                }
                for row in batch
            ]
            if copied_rows:
                result = client.upsert(
                    collection_name=staging.collection_name,
                    data=copied_rows,
                    timeout=self.operation_timeout,
                )
                if _mutation_count(result, "upsert_count") != len(copied_rows):
                    raise VectorStoreError("Milvus did not copy every selected row")
            return copied + len(copied_rows)

        copied = 0
        if hasattr(client, "query_iterator"):
            iterator = client.query_iterator(
                collection_name=source,
                batch_size=1_000,
                filter=expression,
                output_fields=["*"],
                timeout=self.operation_timeout,
            )
            try:
                while True:
                    batch = iterator.next()
                    if not batch:
                        break
                    copied = upsert_batch(batch, copied)
            finally:
                iterator.close()
        else:
            # PyMilvus 2.5+ has query_iterator. The fallback is deliberately
            # fail-closed at the scalar-query pagination ceiling.
            rows = client.query(
                collection_name=source,
                filter=expression,
                output_fields=["*"],
                limit=16_384,
                timeout=self.operation_timeout,
            )
            if len(rows) >= 16_384:
                raise VectorStoreError(
                    "Milvus query_iterator is required to copy this generation"
                )
            copied = upsert_batch(rows, copied)
        return copied

    async def upsert(
        self,
        staging: StagingGeneration,
        chunks: Sequence[Chunk],
        embeddings: EmbeddingBatch | Sequence[Sequence[float]],
    ) -> int:
        rows = _rows(staging, chunks, embeddings)
        async with self._lock:
            self._require_staging(staging)
            client = self._get_client()
            if not await self._run(
                client.has_collection,
                staging.collection_name,
                timeout=self.operation_timeout,
            ):
                raise VectorStoreError("staging collection does not exist")
            for start in range(0, len(rows), 1_000):
                batch = rows[start : start + 1_000]
                result = await self._run(
                    client.upsert,
                    collection_name=staging.collection_name,
                    data=batch,
                    timeout=self.operation_timeout,
                )
                if _mutation_count(result, "upsert_count") != len(batch):
                    raise VectorStoreError("Milvus did not upsert every chunk")
            self._validated.discard(staging.collection_name)
            return len(rows)

    async def delete(self, staging: StagingGeneration, chunk_ids: Sequence[str]) -> int:
        identifiers = _identifiers(chunk_ids, "chunk_ids", MAX_DELETE_IDS)
        async with self._lock:
            self._require_staging(staging)
            client = self._get_client()
            deleted = 0
            for start in range(0, len(identifiers), 1_000):
                expression = f"chunk_id in {json.dumps(list(identifiers[start : start + 1_000]))}"
                result = await self._run(
                    client.delete,
                    collection_name=staging.collection_name,
                    filter=expression,
                    timeout=self.operation_timeout,
                )
                deleted += _mutation_count(result, "delete_count")
            self._validated.discard(staging.collection_name)
            return deleted

    async def delete_sources(
        self, staging: StagingGeneration, source_ids: Sequence[str]
    ) -> int:
        identifiers = _identifiers(source_ids, "source_ids", MAX_DELETE_IDS)
        async with self._lock:
            self._require_staging(staging)
            client = self._get_client()
            deleted = 0
            for start in range(0, len(identifiers), 1_000):
                expression = f"source_id in {json.dumps(list(identifiers[start : start + 1_000]))}"
                result = await self._run(
                    client.delete,
                    collection_name=staging.collection_name,
                    filter=expression,
                    timeout=self.operation_timeout,
                )
                deleted += _mutation_count(result, "delete_count")
            self._validated.discard(staging.collection_name)
            return deleted

    async def validate(
        self,
        staging: StagingGeneration,
        *,
        expected_chunk_ids: Sequence[str] | None = None,
        expected_count: int | None = None,
    ) -> IndexValidation:
        expected = (
            None
            if expected_chunk_ids is None
            else _identifiers(expected_chunk_ids, "expected_chunk_ids", MAX_INDEX_ROWS)
        )
        _expected_count(expected_count)
        async with self._lock:
            self._require_staging(staging)
            client = self._get_client()
            if not await self._run(
                client.has_collection,
                staging.collection_name,
                timeout=self.operation_timeout,
            ):
                return IndexValidation(
                    staging.collection_name,
                    False,
                    0,
                    staging.dimension,
                    (),
                    ("collection does not exist",),
                )
            await self._run(
                client.flush,
                staging.collection_name,
                timeout=self.operation_timeout,
            )
            stats = await self._run(
                client.get_collection_stats,
                staging.collection_name,
                timeout=self.operation_timeout,
            )
            try:
                row_count = int(stats.get("row_count", 0))
            except (AttributeError, TypeError, ValueError) as exc:
                raise VectorStoreError(
                    "Milvus returned invalid collection statistics"
                ) from exc
            missing: list[str] = []
            if expected is not None:
                for start in range(0, len(expected), 1_000):
                    batch = expected[start : start + 1_000]
                    expression = f"chunk_id in {json.dumps(list(batch))}"
                    rows = await self._run(
                        client.query,
                        collection_name=staging.collection_name,
                        filter=expression,
                        output_fields=["chunk_id"],
                        limit=len(batch),
                        timeout=self.operation_timeout,
                    )
                    found = {
                        row.get("chunk_id") for row in rows if isinstance(row, Mapping)
                    }
                    missing.extend(value for value in batch if value not in found)
            errors: list[str] = []
            if row_count == 0:
                errors.append("collection is empty")
            if expected_count is not None and row_count != expected_count:
                errors.append("row count does not match the manifest")
            if expected is not None and row_count != len(expected):
                errors.append("collection contains missing or unexpected chunk IDs")
            if missing:
                errors.append("one or more expected chunk IDs are missing")
            valid = not errors
            if valid:
                self._validated.add(staging.collection_name)
            else:
                self._validated.discard(staging.collection_name)
            return IndexValidation(
                staging.collection_name,
                valid,
                row_count,
                staging.dimension,
                tuple(missing[:1000]),
                tuple(errors),
            )

    async def publish(self, staging: StagingGeneration) -> PublishedGeneration:
        async with self._lock:
            self._require_staging(staging)
            if staging.collection_name not in self._validated:
                raise VectorStoreError(
                    "staging collection must pass validation before publish"
                )
            client = self._get_client()
            previous = await self.current_alias(_already_locked=True)
            if previous == staging.collection_name:
                return PublishedGeneration(
                    self.alias, staging.collection_name, previous
                )
            if previous is not None:
                await self._set_alias(client, self.previous_alias, previous)
            await self._set_alias(client, self.alias, staging.collection_name)
            return PublishedGeneration(self.alias, staging.collection_name, previous)

    async def rollback(self) -> PublishedGeneration:
        async with self._lock:
            client = self._get_client()
            current = await self.current_alias(_already_locked=True)
            previous = await self._describe_alias(self.previous_alias)
            if current is None or previous is None:
                raise VectorStoreError("no previous generation is available")
            await self._set_alias(client, self.alias, previous)
            await self._set_alias(client, self.previous_alias, current)
            return PublishedGeneration(self.alias, previous, current)

    async def current_alias(self, *, _already_locked: bool = False) -> str | None:
        if _already_locked:
            return await self._describe_alias(self.alias)
        async with self._lock:
            return await self._describe_alias(self.alias)

    async def active_generation_id(self) -> str | None:
        async with self._lock:
            collection = await self.current_alias(_already_locked=True)
            if collection is None:
                return None
            client = self._get_client()
            rows = await self._run(
                client.query,
                collection_name=collection,
                filter="generation_id != ''",
                output_fields=["generation_id"],
                limit=1,
                timeout=self.operation_timeout,
            )
            if not isinstance(rows, Sequence) or len(rows) != 1:
                raise VectorStoreError("active Milvus generation has no identity row")
            row = rows[0]
            generation_id = (
                row.get("generation_id") if isinstance(row, Mapping) else None
            )
            if (
                not isinstance(generation_id, str)
                or _SAFE_GENERATION.fullmatch(generation_id) is None
            ):
                raise VectorStoreError("active Milvus generation identity is invalid")
            return generation_id

    async def search(
        self, query_vector: Sequence[float], *, limit: int = 5
    ) -> tuple[SearchHit, ...]:
        _search_limit(limit)
        async with self._lock:
            collection = await self.current_alias(_already_locked=True)
            if collection is None:
                return ()
            client = self._get_client()
            description = await self._run(
                client.describe_collection,
                collection_name=collection,
                timeout=self.operation_timeout,
            )
            dimension = _collection_dimension(description)
            query = _vector(query_vector, dimension)
            raw = await self._run(
                client.search,
                collection_name=collection,
                data=[list(query)],
                anns_field="dense_vector",
                limit=limit,
                output_fields=["chunk_id", "source_id"],
                search_params={"metric_type": self.metric_type, "params": {}},
                timeout=self.operation_timeout,
            )
            return _search_hits(raw, limit=limit)

    async def hybrid_search(
        self,
        query_text: str,
        query_vector: Sequence[float],
        *,
        limit: int = 5,
    ) -> tuple[SearchHit, ...]:
        """Rerank at most 100 dense candidates without returning stored text."""

        _hybrid_query(query_text)
        _search_limit(limit)
        candidate_limit = _hybrid_candidate_limit(limit)
        async with self._lock:
            collection = await self.current_alias(_already_locked=True)
            if collection is None:
                return ()
            client = self._get_client()
            description = await self._run(
                client.describe_collection,
                collection_name=collection,
                timeout=self.operation_timeout,
            )
            dimension = _collection_dimension(description)
            query = _vector(query_vector, dimension)
            raw = await self._run(
                client.search,
                collection_name=collection,
                data=[list(query)],
                anns_field="dense_vector",
                limit=candidate_limit,
                output_fields=["chunk_id", "source_id", "embedding_text"],
                search_params={"metric_type": self.metric_type, "params": {}},
                timeout=self.operation_timeout,
            )
            candidates = _search_candidates(raw, limit=candidate_limit)
            return _hybrid_rerank(query_text, candidates, limit=limit)

    async def _describe_alias(self, alias: str) -> str | None:
        client = self._get_client()
        # Milvus 2.6 does not reliably mirror aliases in
        # ``describe_collection()["aliases"]`` even though ``describe_alias``
        # resolves them.  Prefer the server's alias catalog when available so
        # an absent alias can still be distinguished from an unavailable
        # server without relying on exception text.
        if hasattr(client, "list_aliases") and hasattr(client, "describe_alias"):
            catalog = await self._run(
                client.list_aliases,
                timeout=self.operation_timeout,
            )
            aliases = catalog.get("aliases") if isinstance(catalog, Mapping) else None
            if not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes)):
                raise VectorStoreError("Milvus returned an invalid alias catalog")
            if len(aliases) > MAX_MANAGED_COLLECTIONS:
                raise VectorStoreError("Milvus alias scan exceeds its safety limit")
            if any(
                not isinstance(value, str) or _SAFE_NAME.fullmatch(value) is None
                for value in aliases
            ):
                raise VectorStoreError("Milvus returned an invalid alias name")
            if alias not in aliases:
                return None
            value = await self._run(
                client.describe_alias,
                alias,
                timeout=self.operation_timeout,
            )
            return _alias_collection(value)
        # ``describe_alias`` raises the same public exception class for an
        # absent alias and an unavailable server. Enumerating collection
        # descriptions avoids misclassifying an outage as "no active index".
        if hasattr(client, "list_collections") and hasattr(
            client, "describe_collection"
        ):
            return await self._run(self._find_alias_sync, alias)
        try:
            value = await self._run(
                client.describe_alias, alias, timeout=self.operation_timeout
            )
        except VectorStoreError as exc:
            if isinstance(exc.__cause__, (KeyError, LookupError)):
                return None
            raise
        return _alias_collection(value)

    def _find_alias_sync(self, alias: str) -> str | None:
        client = self._get_client()
        collections = client.list_collections(timeout=self.operation_timeout)
        if not isinstance(collections, Sequence) or isinstance(
            collections, (str, bytes)
        ):
            raise VectorStoreError("Milvus returned an invalid collection list")
        if len(collections) > MAX_MANAGED_COLLECTIONS:
            raise VectorStoreError("Milvus collection scan exceeds its safety limit")
        matches: list[str] = []
        for collection in collections:
            if (
                not isinstance(collection, str)
                or not collection
                or len(collection) > 255
                or any(ord(character) < 32 for character in collection)
            ):
                raise VectorStoreError("Milvus returned an invalid collection name")
            description = client.describe_collection(
                collection_name=collection,
                timeout=self.operation_timeout,
            )
            aliases = (
                description.get("aliases") if isinstance(description, Mapping) else None
            )
            if not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes)):
                raise VectorStoreError("Milvus returned invalid collection aliases")
            if alias in aliases:
                matches.append(collection)
        if len(matches) > 1:
            raise VectorStoreError("Milvus alias is bound to multiple collections")
        return matches[0] if matches else None

    async def _set_alias(self, client: Any, alias: str, collection: str) -> None:
        existing = await self._describe_alias(alias)
        if existing is None:
            await self._run(
                client.create_alias,
                collection,
                alias,
                timeout=self.operation_timeout,
            )
        elif existing != collection:
            await self._run(
                client.alter_alias,
                collection,
                alias,
                timeout=self.operation_timeout,
            )

    async def drop_staging(self, staging: StagingGeneration) -> None:
        async with self._lock:
            self._require_staging(staging)
            active = await self.current_alias(_already_locked=True)
            previous = await self._describe_alias(self.previous_alias)
            if staging.collection_name in {active, previous}:
                raise VectorStoreError("an aliased collection cannot be dropped")
            client = self._get_client()
            if await self._run(
                client.has_collection,
                staging.collection_name,
                timeout=self.operation_timeout,
            ):
                await self._run(
                    client.drop_collection,
                    staging.collection_name,
                    timeout=self.operation_timeout,
                )
            self._staging_handles.pop(staging.collection_name, None)
            self._validated.discard(staging.collection_name)

    def _require_staging(self, staging: StagingGeneration) -> None:
        if not isinstance(staging, StagingGeneration):
            raise TypeError("staging must be a StagingGeneration")
        if self._staging_handles.get(staging.collection_name) != staging:
            raise VectorStoreError("staging collection is not owned by this run")


def _alias_collection(value: Any) -> str:
    if not isinstance(value, Mapping):
        raise VectorStoreError("Milvus returned an invalid alias description")
    collection = value.get("collection_name") or value.get("collection")
    if not isinstance(collection, str) or _SAFE_NAME.fullmatch(collection) is None:
        raise VectorStoreError("Milvus alias has no valid collection")
    return collection


def _collection_dimension(value: Any) -> int:
    if not isinstance(value, Mapping):
        raise VectorStoreError("Milvus returned an invalid collection schema")
    fields = value.get("fields")
    if not isinstance(fields, Sequence) or isinstance(fields, (str, bytes)):
        raise VectorStoreError("Milvus returned an invalid collection schema")
    vector = next(
        (
            field
            for field in fields
            if isinstance(field, Mapping) and field.get("name") == "dense_vector"
        ),
        None,
    )
    params = vector.get("params") if isinstance(vector, Mapping) else None
    try:
        dimension = int(params.get("dim")) if isinstance(params, Mapping) else -1
    except (TypeError, ValueError) as exc:
        raise VectorStoreError("Milvus returned an invalid vector dimension") from exc
    if not 1 <= dimension <= 65_536:
        raise VectorStoreError("Milvus returned an invalid vector dimension")
    return dimension


def _search_limit(value: int) -> None:
    if type(value) is not int or not 1 <= value <= 100:
        raise ValueError("search limit must be between one and 100")


def _search_hits(value: Any, *, limit: int) -> tuple[SearchHit, ...]:
    return tuple(hit for hit, _text in _search_candidates(value, limit=limit))


def _search_candidates(
    value: Any,
    *,
    limit: int,
) -> tuple[tuple[SearchHit, str], ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 1
    ):
        raise VectorStoreError("Milvus returned an invalid search result")
    raw_hits = value[0]
    if not isinstance(raw_hits, Sequence) or isinstance(raw_hits, (str, bytes)):
        raise VectorStoreError("Milvus returned an invalid search result")
    if len(raw_hits) > limit:
        raise VectorStoreError("Milvus search result exceeds the requested limit")
    hits: list[tuple[SearchHit, str]] = []
    seen: set[str] = set()
    for raw_hit in raw_hits:
        if not isinstance(raw_hit, Mapping):
            raise VectorStoreError("Milvus returned an invalid search hit")
        raw_entity = raw_hit.get("entity")
        entity = _entity_mapping(raw_entity) if raw_entity is not None else raw_hit
        chunk_id = entity.get("chunk_id", raw_hit.get("id"))
        source_id = entity.get("source_id")
        score = raw_hit.get("distance", raw_hit.get("score"))
        hit = SearchHit(chunk_id=chunk_id, source_id=source_id, score=score)
        if hit.chunk_id in seen:
            raise VectorStoreError("Milvus returned duplicate search hits")
        seen.add(hit.chunk_id)
        hits.append((hit, _stored_search_text(entity.get("embedding_text", ""))))
    return tuple(hits)


def _hybrid_query(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > MAX_HYBRID_QUERY_CHARS
        or any(ord(character) < 32 and character not in "\t\n\r" for character in value)
    ):
        raise ValueError("hybrid query text must be non-empty and bounded")
    return value.strip()


def _hybrid_candidate_limit(limit: int) -> int:
    return min(MAX_HYBRID_CANDIDATES, max(limit, limit * 10))


def _stored_search_text(value: Any) -> str:
    if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_TEXT_FIELD_CHARS:
        raise VectorStoreError("Milvus returned invalid retrieval text")
    return value


def _lexical_tokens(value: str) -> frozenset[str]:
    tokens = re.findall(r"[A-Za-z0-9]+|[가-힣]+", value.casefold())
    if len(tokens) > MAX_HYBRID_TOKENS:
        tokens = tokens[:MAX_HYBRID_TOKENS]
    return frozenset(tokens)


def _identifier_anchors(value: str) -> frozenset[str]:
    """Keep digit-bearing compound identifiers intact for exact retrieval."""

    return frozenset(
        re.findall(
            r"\b(?=[a-z0-9-]*\d)[a-z0-9]+(?:-[a-z0-9]+)+\b",
            value.casefold(),
        )[:64]
    )


def _hybrid_rerank(
    query_text: str,
    candidates: tuple[tuple[SearchHit, str], ...],
    *,
    limit: int,
) -> tuple[SearchHit, ...]:
    if not candidates:
        return ()
    query_tokens = _lexical_tokens(_hybrid_query(query_text))
    query_anchors = _identifier_anchors(query_text)
    if not query_tokens:
        return tuple(hit for hit, _text in candidates[:limit])
    document_tokens = tuple(_lexical_tokens(text) for _hit, text in candidates)
    document_count = len(document_tokens)
    document_frequency = {
        token: sum(token in tokens for tokens in document_tokens)
        for token in query_tokens
    }
    weights = {
        token: math.log((document_count + 1) / (frequency + 1)) + 1.0
        for token, frequency in document_frequency.items()
    }
    maximum_coverage = sum(weights.values())
    reranked: list[SearchHit] = []
    for (hit, text), tokens in zip(candidates, document_tokens):
        lexical = (
            sum(weights[token] for token in query_tokens & tokens) / maximum_coverage
        )
        # COSINE is in [-1, 1].  Preserve its absolute distance instead of
        # min-max stretching tiny candidate differences into a full 0..1 gap.
        dense = min(1.0, max(0.0, (hit.score + 1.0) / 2.0))
        anchors = (
            len(query_anchors & _identifier_anchors(text)) / len(query_anchors)
            if query_anchors
            else 0.0
        )
        # Exact compound identifiers need a separate signal: tokenizing
        # ``CORP-INCIDENT-001`` loses the relation between its common and
        # unique parts. Dense similarity remains dominant when no identifier
        # is present.
        if query_anchors:
            combined = dense * 0.50 + lexical * 0.20 + anchors * 0.30
        else:
            combined = dense * 0.65 + lexical * 0.35
        reranked.append(SearchHit(hit.chunk_id, hit.source_id, combined))
    reranked.sort(key=lambda hit: (-hit.score, hit.chunk_id))
    return tuple(reranked[:limit])


def staging_handle(
    prefix: str,
    generation_id: str,
    dimension: int,
    pipeline_fingerprint: PipelineFingerprint | str,
) -> StagingGeneration:
    _require_name(prefix, "collection prefix")
    if (
        not isinstance(generation_id, str)
        or _SAFE_GENERATION.fullmatch(generation_id) is None
    ):
        raise ValueError("generation_id must be a safe bounded identifier")
    if type(dimension) is not int or not 1 <= dimension <= 65_536:
        raise ValueError("dimension must be between one and 65,536")
    fingerprint = (
        pipeline_fingerprint.digest
        if isinstance(pipeline_fingerprint, PipelineFingerprint)
        else pipeline_fingerprint
    )
    if (
        not isinstance(fingerprint, str)
        or not fingerprint.strip()
        or _utf8_length(fingerprint, "pipeline fingerprint") > 512
    ):
        raise ValueError("pipeline_fingerprint must be non-empty and bounded")
    suffix = stable_digest(generation_id, dimension, fingerprint)[:24]
    name = f"{prefix}_g_{suffix}"
    _require_name(name, "collection name")
    return StagingGeneration(generation_id, name, dimension, fingerprint)


def _rows(
    staging: StagingGeneration,
    chunks: Sequence[Chunk],
    embeddings: EmbeddingBatch | Sequence[Sequence[float]],
) -> list[dict[str, Any]]:
    if isinstance(chunks, (str, bytes)) or not chunks or len(chunks) > MAX_INDEX_ROWS:
        raise ValueError("chunks must be a non-empty bounded sequence")
    if isinstance(embeddings, EmbeddingBatch):
        vectors: Sequence[Sequence[float]] = embeddings.vectors
        if embeddings.dimension != staging.dimension:
            raise VectorStoreError("embedding dimension does not match staging")
    else:
        vectors = embeddings
    if isinstance(vectors, (str, bytes)) or len(vectors) != len(chunks):
        raise ValueError("chunks and vectors must have the same length")
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for chunk, raw_vector in zip(chunks, vectors):
        if not isinstance(chunk, Chunk):
            raise TypeError("chunks must contain Chunk values")
        if chunk.chunk_id in seen:
            raise ValueError("chunk IDs must be unique in one upsert")
        seen.add(chunk.chunk_id)
        vector = _vector(raw_vector, staging.dimension)
        if (
            _utf8_length(chunk.content, "chunk content") > MAX_TEXT_FIELD_CHARS
            or _utf8_length(chunk.embedding_text, "embedding text")
            > MAX_TEXT_FIELD_CHARS
        ):
            raise VectorStoreError("chunk text exceeds the Milvus field limit")
        section_json = _bounded_json(chunk.section_path, 8_192, "section path")
        block_json = _bounded_json(chunk.block_ids, 8_192, "block IDs")
        provenance_json = _bounded_json(
            [asdict(value) for value in chunk.provenance], 32_768, "provenance"
        )
        metadata_json = _bounded_json(dict(chunk.metadata), 32_768, "metadata")
        rows.append(
            {
                "chunk_id": chunk.chunk_id,
                "kb_id": chunk.kb_id,
                "source_id": chunk.source_id,
                "source_revision": chunk.source_revision,
                "generation_id": staging.generation_id,
                "content": chunk.content,
                "embedding_text": chunk.embedding_text,
                "section_path_json": section_json,
                "block_ids_json": block_json,
                "provenance_json": provenance_json,
                "metadata_json": metadata_json,
                "pipeline_fingerprint": _stored_fingerprint(
                    staging.pipeline_fingerprint
                ),
                "dense_vector": list(vector),
            }
        )
    return rows


def _vector(values: Sequence[float], dimension: int) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != dimension:
        raise VectorStoreError("vector dimension does not match staging")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VectorStoreError("vector values must be numeric")
        converted = float(value)
        if not math.isfinite(converted):
            raise VectorStoreError("vector values must be finite")
        result.append(converted)
    return tuple(result)


def _bounded_json(value: Any, maximum: int, name: str) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    if _utf8_length(encoded, name) > maximum:
        raise VectorStoreError(f"{name} exceeds its Milvus field limit")
    return encoded


def _entity_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    raise VectorStoreError("Milvus returned an invalid entity row")


def _mutation_count(value: Any, field: str) -> int:
    if isinstance(value, Mapping):
        raw = value.get(field)
    else:
        raw = getattr(value, field, None)
    if type(raw) is not int or raw < 0:
        raise VectorStoreError("Milvus returned an invalid mutation result")
    return raw


def _utf8_length(value: str, name: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise VectorStoreError(f"{name} is not valid UTF-8 text") from exc


def _stored_fingerprint(value: str) -> str:
    return (
        value
        if _utf8_length(value, "pipeline fingerprint") <= 128
        else stable_digest(value)
    )


def _identifiers(values: Sequence[str], name: str, maximum: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or len(values) > maximum:
        raise ValueError(f"{name} must be a bounded sequence")
    result = tuple(values)
    if any(
        not isinstance(value, str)
        or not value
        or len(value) > 128
        or _SAFE_GENERATION.fullmatch(value) is None
        for value in result
    ):
        raise ValueError(f"{name} contains an invalid identifier")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _expected_count(value: int | None) -> None:
    if value is not None and (
        type(value) is not int or not 0 <= value <= MAX_INDEX_ROWS
    ):
        raise ValueError("expected_count is invalid")


def _require_name(value: str, name: str) -> None:
    if not isinstance(value, str) or _SAFE_NAME.fullmatch(value) is None:
        raise ValueError(f"{name} must be a Milvus-safe identifier")


__all__ = [
    "CopyResult",
    "InMemoryMilvusStore",
    "IndexValidation",
    "MAX_HYBRID_CANDIDATES",
    "MAX_HYBRID_QUERY_CHARS",
    "MAX_HYBRID_TOKENS",
    "MilvusStore",
    "PublishedGeneration",
    "SearchHit",
    "StagingGeneration",
    "VectorStore",
    "VectorStoreError",
    "staging_handle",
]
