"""SQLite manifest catalog with staged, published, and rollback generations."""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import secrets
import sqlite3
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    CatalogError,
    Chunk,
    GenerationState,
    ManifestDocument,
    PipelineFingerprint,
    RunState,
    SourceDocument,
)


_SCHEMA_VERSION = 3
_KB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_GENERATION_ID = re.compile(r"^gen_[A-Za-z0-9][A-Za-z0-9_.-]{0,123}$")


class ManifestCatalog:
    """Generation-based manifest storage for incremental RAG ingestion.

    A run writes only into its new generation.  ``commit_published`` performs
    the small atomic pointer switch; therefore a failed parse, model call, or
    Milvus build cannot corrupt the previously published manifest snapshot.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        raw_path = os.fspath(path)
        if raw_path != ":memory:":
            database_path = Path(raw_path).expanduser()
            try:
                database_path.parent.mkdir(parents=True, exist_ok=True)
                if database_path.is_symlink():
                    raise CatalogError("manifest database cannot be a symbolic link")
                if database_path.exists() and not database_path.is_file():
                    raise CatalogError("manifest database path must be a regular file")
            except CatalogError:
                raise
            except OSError as exc:
                raise CatalogError("manifest database path is not accessible") from exc
            raw_path = os.fspath(database_path)
        try:
            self._connection = sqlite3.connect(
                raw_path,
                isolation_level=None,
                timeout=10.0,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 10000")
            if raw_path != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._migrate()
        except CatalogError:
            raise
        except sqlite3.Error as exc:
            raise CatalogError("manifest database could not be opened") from exc
        self._closed = False

    def __enter__(self) -> ManifestCatalog:
        self._ensure_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._connection.close()
            self._closed = True

    def begin_run(
        self,
        kb_id: str,
        pipeline: PipelineFingerprint,
        *,
        generation_id: str | None = None,
    ) -> str:
        """Create an isolated building generation and return its run ID."""

        self._ensure_open()
        _validate_kb_id(kb_id)
        if not isinstance(pipeline, PipelineFingerprint):
            raise TypeError("pipeline must be a PipelineFingerprint")
        resolved_generation = generation_id or f"gen_{secrets.token_hex(16)}"
        _validate_generation_id(resolved_generation)
        run_id = f"run_{secrets.token_hex(16)}"
        now = _utc_now()
        try:
            with self._transaction():
                self._connection.execute(
                    """
                    INSERT INTO generations (
                        generation_id, kb_id, state,
                        parser_fp, restructuring_fp, layout_refinement_fp,
                        enrichment_fp,
                        chunking_fp, embedding_fp, embedding_dimension, indexing_fp,
                        pipeline_digest, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resolved_generation,
                        kb_id,
                        GenerationState.BUILDING.value,
                        pipeline.parser,
                        pipeline.restructuring,
                        pipeline.layout_refinement,
                        pipeline.enrichment,
                        pipeline.chunking,
                        pipeline.embedding,
                        pipeline.embedding_dimension,
                        pipeline.indexing,
                        pipeline.digest,
                        now,
                    ),
                )
                self._connection.execute(
                    """
                    INSERT INTO runs (
                        run_id, generation_id, kb_id, state,
                        pipeline_digest, started_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        resolved_generation,
                        kb_id,
                        RunState.RUNNING.value,
                        pipeline.digest,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise CatalogError("generation ID already exists") from exc
        except sqlite3.Error as exc:
            raise CatalogError("manifest run could not be started") from exc
        return run_id

    def record_document(
        self,
        run_id: str,
        source: SourceDocument,
        pipeline: PipelineFingerprint,
        *,
        chunk_count: int = 0,
    ) -> None:
        """Record a processed source in the run's isolated generation."""

        if not isinstance(source, SourceDocument):
            raise TypeError("source must be a SourceDocument")
        if not isinstance(pipeline, PipelineFingerprint):
            raise TypeError("pipeline must be a PipelineFingerprint")
        if type(chunk_count) is not int or chunk_count < 0:
            raise ValueError("chunk_count cannot be negative")
        run = self._writable_run(run_id)
        if source.kb_id != run["kb_id"] or pipeline.digest != run["pipeline_digest"]:
            raise CatalogError("source or pipeline does not belong to this run")
        try:
            self._connection.execute(
                """
                INSERT INTO document_versions (
                    generation_id, source_id, kb_id, relative_path, media_type,
                    size_bytes, mtime_ns, content_sha256,
                    parser_fp, restructuring_fp, layout_refinement_fp,
                    enrichment_fp,
                    chunking_fp, embedding_fp, embedding_dimension,
                    indexing_fp, chunk_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(generation_id, source_id) DO UPDATE SET
                    relative_path=excluded.relative_path,
                    media_type=excluded.media_type,
                    size_bytes=excluded.size_bytes,
                    mtime_ns=excluded.mtime_ns,
                    content_sha256=excluded.content_sha256,
                    parser_fp=excluded.parser_fp,
                    restructuring_fp=excluded.restructuring_fp,
                    layout_refinement_fp=excluded.layout_refinement_fp,
                    enrichment_fp=excluded.enrichment_fp,
                    chunking_fp=excluded.chunking_fp,
                    embedding_fp=excluded.embedding_fp,
                    embedding_dimension=excluded.embedding_dimension,
                    indexing_fp=excluded.indexing_fp,
                    chunk_count=excluded.chunk_count
                """,
                (
                    run["generation_id"],
                    source.source_id,
                    source.kb_id,
                    source.relative_path,
                    source.media_type,
                    source.size_bytes,
                    source.mtime_ns,
                    source.sha256,
                    pipeline.parser,
                    pipeline.restructuring,
                    pipeline.layout_refinement,
                    pipeline.enrichment,
                    pipeline.chunking,
                    pipeline.embedding,
                    pipeline.embedding_dimension,
                    pipeline.indexing,
                    chunk_count,
                ),
            )
        except sqlite3.Error as exc:
            raise CatalogError("document manifest could not be recorded") from exc

    stage_document = record_document

    def record_chunks(self, run_id: str, chunks: Sequence[Chunk]) -> None:
        """Record deterministic chunk manifests, not vectors or source bodies."""

        run = self._writable_run(run_id)
        if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
            raise TypeError("chunks must be a sequence")
        if len(chunks) > 1_000_000:
            raise CatalogError("chunk manifest exceeds its safety limit")
        rows: list[tuple[object, ...]] = []
        counts: dict[str, int] = {}
        seen: set[str] = set()
        for chunk in chunks:
            if not isinstance(chunk, Chunk):
                raise TypeError("chunks must contain Chunk values")
            if chunk.kb_id != run["kb_id"]:
                raise CatalogError("chunk does not belong to this run's knowledge base")
            if chunk.chunk_id in seen:
                raise CatalogError("duplicate chunk ID")
            seen.add(chunk.chunk_id)
            counts[chunk.source_id] = counts.get(chunk.source_id, 0) + 1
            rows.append(
                (
                    run["generation_id"],
                    chunk.chunk_id,
                    chunk.source_id,
                    chunk.source_revision,
                    chunk.ordinal,
                    hashlib.sha256(chunk.content.encode("utf-8")).hexdigest(),
                    hashlib.sha256(chunk.embedding_text.encode("utf-8")).hexdigest(),
                )
            )
        try:
            with self._transaction():
                for source_id in counts:
                    # One call is the complete chunk manifest for each source.
                    # Removing old rows makes retries and changed chunk counts
                    # idempotent instead of leaving stale IDs behind.
                    self._connection.execute(
                        """
                        DELETE FROM chunk_versions
                        WHERE generation_id=? AND source_id=?
                        """,
                        (run["generation_id"], source_id),
                    )
                self._connection.executemany(
                    """
                    INSERT INTO chunk_versions (
                        generation_id, chunk_id, source_id, source_revision,
                        ordinal, content_sha256, embedding_text_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(generation_id, chunk_id) DO UPDATE SET
                        source_id=excluded.source_id,
                        source_revision=excluded.source_revision,
                        ordinal=excluded.ordinal,
                        content_sha256=excluded.content_sha256,
                        embedding_text_sha256=excluded.embedding_text_sha256
                    """,
                    rows,
                )
                for source_id, count in counts.items():
                    cursor = self._connection.execute(
                        """
                        UPDATE document_versions SET chunk_count=?
                        WHERE generation_id=? AND source_id=?
                        """,
                        (count, run["generation_id"], source_id),
                    )
                    if cursor.rowcount != 1:
                        raise CatalogError(
                            "chunks require a document manifest in the same run"
                        )
        except CatalogError:
            raise
        except sqlite3.Error as exc:
            raise CatalogError("chunk manifests could not be recorded") from exc

    def carry_forward_document(
        self,
        run_id: str,
        document: ManifestDocument,
        *,
        pipeline: PipelineFingerprint | None = None,
    ) -> None:
        """Copy an unchanged document/chunk manifest into a building snapshot.

        ``pipeline`` may override only the stored stage fingerprints.  This is
        useful for an indexing-only revision: existing chunk payloads remain
        byte-identical while the new generation records its current policy.
        """

        if not isinstance(document, ManifestDocument):
            raise TypeError("document must be a ManifestDocument")
        run = self._writable_run(run_id)
        if document.kb_id != run["kb_id"]:
            raise CatalogError("document does not belong to this run")
        resolved_pipeline = pipeline or document.pipeline
        if not isinstance(resolved_pipeline, PipelineFingerprint):
            raise TypeError("pipeline must be a PipelineFingerprint")
        if resolved_pipeline.digest != run["pipeline_digest"]:
            raise CatalogError("carried document pipeline does not belong to this run")
        try:
            with self._transaction():
                source_row = self._connection.execute(
                    """
                    SELECT 1 FROM document_versions
                    WHERE generation_id=? AND source_id=?
                    """,
                    (document.generation_id, document.source_id),
                ).fetchone()
                if source_row is None:
                    raise CatalogError("source document generation is unavailable")
                self._connection.execute(
                    """
                    INSERT INTO document_versions (
                        generation_id, source_id, kb_id, relative_path, media_type,
                        size_bytes, mtime_ns, content_sha256,
                        parser_fp, restructuring_fp, layout_refinement_fp,
                        enrichment_fp,
                        chunking_fp, embedding_fp, embedding_dimension,
                        indexing_fp, chunk_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run["generation_id"],
                        document.source_id,
                        document.kb_id,
                        document.relative_path,
                        document.media_type,
                        document.size_bytes,
                        document.mtime_ns,
                        document.content_sha256,
                        resolved_pipeline.parser,
                        resolved_pipeline.restructuring,
                        resolved_pipeline.layout_refinement,
                        resolved_pipeline.enrichment,
                        resolved_pipeline.chunking,
                        resolved_pipeline.embedding,
                        resolved_pipeline.embedding_dimension,
                        resolved_pipeline.indexing,
                        document.chunk_count,
                    ),
                )
                self._connection.execute(
                    """
                    INSERT INTO chunk_versions (
                        generation_id, chunk_id, source_id, source_revision,
                        ordinal, content_sha256, embedding_text_sha256
                    ) SELECT ?, chunk_id, source_id, source_revision,
                        ordinal, content_sha256, embedding_text_sha256
                    FROM chunk_versions
                    WHERE generation_id=? AND source_id=?
                    """,
                    (run["generation_id"], document.generation_id, document.source_id),
                )
        except sqlite3.IntegrityError as exc:
            raise CatalogError("document was already recorded in this run") from exc
        except sqlite3.Error as exc:
            raise CatalogError(
                "document manifest could not be carried forward"
            ) from exc

    def mark_deleted(self, run_id: str, source_id: str) -> None:
        """Record a tombstone; generation snapshots omit deleted documents."""

        run = self._writable_run(run_id)
        if not isinstance(source_id, str) or not source_id.startswith("src_"):
            raise ValueError("source_id is invalid")
        try:
            self._connection.execute(
                """
                INSERT INTO run_deletions (run_id, generation_id, source_id)
                VALUES (?, ?, ?)
                ON CONFLICT(run_id, source_id) DO NOTHING
                """,
                (run_id, run["generation_id"], source_id),
            )
        except sqlite3.Error as exc:
            raise CatalogError("deletion tombstone could not be recorded") from exc

    def mark_staged(self, run_id: str) -> str:
        """Seal a building snapshot after external index validation succeeds."""

        run = self._writable_run(run_id)
        now = _utc_now()
        try:
            with self._transaction():
                self._connection.execute(
                    "UPDATE generations SET state=?, staged_at=? WHERE generation_id=?",
                    (GenerationState.STAGED.value, now, run["generation_id"]),
                )
                self._connection.execute(
                    "UPDATE runs SET state=?, staged_at=? WHERE run_id=?",
                    (RunState.STAGED.value, now, run_id),
                )
        except sqlite3.Error as exc:
            raise CatalogError("manifest generation could not be staged") from exc
        return str(run["generation_id"])

    def commit_published(self, run_id: str) -> str | None:
        """Atomically publish a sealed manifest and return the prior generation."""

        self._ensure_open()
        run = self._run(run_id)
        if run["state"] != RunState.STAGED.value:
            raise CatalogError("only a staged run can be published")
        generation_id = str(run["generation_id"])
        kb_id = str(run["kb_id"])
        now = _utc_now()
        try:
            with self._transaction():
                state = self._connection.execute(
                    "SELECT current_generation_id FROM kb_state WHERE kb_id=?",
                    (kb_id,),
                ).fetchone()
                previous = None if state is None else state["current_generation_id"]
                if previous and previous != generation_id:
                    self._connection.execute(
                        "UPDATE generations SET state=? WHERE generation_id=?",
                        (GenerationState.AVAILABLE.value, previous),
                    )
                self._connection.execute(
                    "UPDATE generations SET state=?, published_at=? WHERE generation_id=?",
                    (GenerationState.PUBLISHED.value, now, generation_id),
                )
                self._connection.execute(
                    """
                    INSERT INTO kb_state (kb_id, current_generation_id, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(kb_id) DO UPDATE SET
                        current_generation_id=excluded.current_generation_id,
                        updated_at=excluded.updated_at
                    """,
                    (kb_id, generation_id, now),
                )
                self._connection.execute(
                    "UPDATE runs SET state=?, finished_at=? WHERE run_id=?",
                    (RunState.PUBLISHED.value, now, run_id),
                )
        except sqlite3.Error as exc:
            raise CatalogError("manifest generation could not be published") from exc
        return None if previous is None else str(previous)

    def fail_run(self, run_id: str, error: str) -> None:
        """Mark a non-published run failed without changing the active snapshot."""

        self._ensure_open()
        run = self._run(run_id)
        if run["state"] == RunState.PUBLISHED.value:
            raise CatalogError("a published run cannot be marked failed")
        bounded_error = str(error).replace("\x00", "")[:2_000]
        now = _utc_now()
        try:
            with self._transaction():
                self._connection.execute(
                    "UPDATE generations SET state=? WHERE generation_id=?",
                    (GenerationState.FAILED.value, run["generation_id"]),
                )
                self._connection.execute(
                    "UPDATE runs SET state=?, error=?, finished_at=? WHERE run_id=?",
                    (RunState.FAILED.value, bounded_error, now, run_id),
                )
        except sqlite3.Error as exc:
            raise CatalogError("manifest run could not be marked failed") from exc

    def rollback_to_generation(self, kb_id: str, generation_id: str) -> str | None:
        """Move the manifest pointer to a retained successful generation."""

        self._ensure_open()
        _validate_kb_id(kb_id)
        _validate_generation_id(generation_id)
        target = self._connection.execute(
            "SELECT state, kb_id FROM generations WHERE generation_id=?",
            (generation_id,),
        ).fetchone()
        if target is None or target["kb_id"] != kb_id:
            raise CatalogError("rollback generation does not exist")
        if target["state"] not in {
            GenerationState.PUBLISHED.value,
            GenerationState.AVAILABLE.value,
        }:
            raise CatalogError("rollback target is not a successful generation")
        current = self.current_generation(kb_id)
        if current == generation_id:
            return current
        now = _utc_now()
        try:
            with self._transaction():
                if current:
                    self._connection.execute(
                        "UPDATE generations SET state=? WHERE generation_id=?",
                        (GenerationState.AVAILABLE.value, current),
                    )
                self._connection.execute(
                    "UPDATE generations SET state=?, published_at=? WHERE generation_id=?",
                    (GenerationState.PUBLISHED.value, now, generation_id),
                )
                self._connection.execute(
                    """
                    INSERT INTO kb_state (kb_id, current_generation_id, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(kb_id) DO UPDATE SET
                        current_generation_id=excluded.current_generation_id,
                        updated_at=excluded.updated_at
                    """,
                    (kb_id, generation_id, now),
                )
        except sqlite3.Error as exc:
            raise CatalogError("manifest rollback failed") from exc
        return current

    def rollback_candidates(self, kb_id: str) -> tuple[str, ...]:
        """List retained successful generations, newest first, excluding active."""

        self._ensure_open()
        _validate_kb_id(kb_id)
        rows = self._connection.execute(
            """
            SELECT generation_id FROM generations
            WHERE kb_id=? AND state=?
            ORDER BY COALESCE(published_at, created_at) DESC, generation_id DESC
            """,
            (kb_id, GenerationState.AVAILABLE.value),
        ).fetchall()
        return tuple(str(row["generation_id"]) for row in rows)

    def previous_generation(self, kb_id: str) -> str | None:
        """Return the immediate rollback target used by the two-alias store."""

        candidates = self.rollback_candidates(kb_id)
        return candidates[0] if candidates else None

    def current_generation(self, kb_id: str) -> str | None:
        self._ensure_open()
        row = self._connection.execute(
            "SELECT current_generation_id FROM kb_state WHERE kb_id=?",
            (kb_id,),
        ).fetchone()
        return None if row is None else str(row["current_generation_id"])

    def run_state(self, run_id: str) -> RunState:
        return RunState(self._run(run_id)["state"])

    def generation_state(self, generation_id: str) -> GenerationState:
        self._ensure_open()
        row = self._connection.execute(
            "SELECT state FROM generations WHERE generation_id=?",
            (generation_id,),
        ).fetchone()
        if row is None:
            raise CatalogError("generation does not exist")
        return GenerationState(row["state"])

    def list_documents(
        self,
        kb_id: str,
        *,
        generation_id: str | None = None,
    ) -> tuple[ManifestDocument, ...]:
        """List one immutable generation, defaulting to the published one."""

        self._ensure_open()
        resolved_generation = generation_id or self.current_generation(kb_id)
        if resolved_generation is None:
            return ()
        rows = self._connection.execute(
            """
            SELECT d.*, g.state AS generation_state
            FROM document_versions AS d
            JOIN generations AS g USING (generation_id)
            WHERE d.kb_id=? AND d.generation_id=?
            ORDER BY d.relative_path, d.source_id
            """,
            (kb_id, resolved_generation),
        ).fetchall()
        return tuple(_manifest_document(row) for row in rows)

    def get_document(
        self,
        kb_id: str,
        source_id: str,
        *,
        generation_id: str | None = None,
    ) -> ManifestDocument | None:
        return next(
            (
                item
                for item in self.list_documents(kb_id, generation_id=generation_id)
                if item.source_id == source_id
            ),
            None,
        )

    def list_chunk_ids(
        self,
        kb_id: str,
        *,
        generation_id: str | None = None,
        source_id: str | None = None,
    ) -> tuple[str, ...]:
        resolved_generation = generation_id or self.current_generation(kb_id)
        if resolved_generation is None:
            return ()
        if source_id is None:
            rows = self._connection.execute(
                """
                SELECT c.chunk_id FROM chunk_versions AS c
                JOIN document_versions AS d
                  ON d.generation_id=c.generation_id AND d.source_id=c.source_id
                WHERE d.kb_id=? AND c.generation_id=?
                ORDER BY c.source_id, c.ordinal, c.chunk_id
                """,
                (kb_id, resolved_generation),
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT chunk_id FROM chunk_versions
                WHERE generation_id=? AND source_id=?
                ORDER BY ordinal, chunk_id
                """,
                (resolved_generation, source_id),
            ).fetchall()
        return tuple(str(row["chunk_id"]) for row in rows)

    def _migrate(self) -> None:
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version > _SCHEMA_VERSION:
            raise CatalogError("manifest schema is newer than this example")
        if version < 1:
            try:
                # ``executescript`` manages its own transaction boundary, so
                # include BEGIN/COMMIT in the migration script itself.
                self._connection.executescript(
                    """
                        BEGIN IMMEDIATE;
                        CREATE TABLE generations (
                            generation_id TEXT PRIMARY KEY,
                            kb_id TEXT NOT NULL,
                            state TEXT NOT NULL,
                            parser_fp TEXT NOT NULL,
                            restructuring_fp TEXT NOT NULL,
                            enrichment_fp TEXT NOT NULL,
                            chunking_fp TEXT NOT NULL,
                            embedding_fp TEXT NOT NULL,
                            indexing_fp TEXT NOT NULL,
                            pipeline_digest TEXT NOT NULL,
                            created_at TEXT NOT NULL,
                            staged_at TEXT,
                            published_at TEXT
                        );
                        CREATE INDEX generations_kb_state
                            ON generations(kb_id, state, created_at);

                        CREATE TABLE kb_state (
                            kb_id TEXT PRIMARY KEY,
                            current_generation_id TEXT NOT NULL
                                REFERENCES generations(generation_id),
                            updated_at TEXT NOT NULL
                        );

                        CREATE TABLE runs (
                            run_id TEXT PRIMARY KEY,
                            generation_id TEXT NOT NULL UNIQUE
                                REFERENCES generations(generation_id),
                            kb_id TEXT NOT NULL,
                            state TEXT NOT NULL,
                            pipeline_digest TEXT NOT NULL,
                            started_at TEXT NOT NULL,
                            staged_at TEXT,
                            finished_at TEXT,
                            error TEXT
                        );

                        CREATE TABLE document_versions (
                            generation_id TEXT NOT NULL
                                REFERENCES generations(generation_id) ON DELETE CASCADE,
                            source_id TEXT NOT NULL,
                            kb_id TEXT NOT NULL,
                            relative_path TEXT NOT NULL,
                            media_type TEXT NOT NULL,
                            size_bytes INTEGER NOT NULL,
                            mtime_ns INTEGER NOT NULL,
                            content_sha256 TEXT NOT NULL,
                            parser_fp TEXT NOT NULL,
                            restructuring_fp TEXT NOT NULL,
                            enrichment_fp TEXT NOT NULL,
                            chunking_fp TEXT NOT NULL,
                            embedding_fp TEXT NOT NULL,
                            indexing_fp TEXT NOT NULL,
                            chunk_count INTEGER NOT NULL,
                            PRIMARY KEY (generation_id, source_id),
                            UNIQUE (generation_id, relative_path)
                        );
                        CREATE INDEX document_versions_source
                            ON document_versions(source_id, generation_id);

                        CREATE TABLE chunk_versions (
                            generation_id TEXT NOT NULL,
                            chunk_id TEXT NOT NULL,
                            source_id TEXT NOT NULL,
                            source_revision TEXT NOT NULL,
                            ordinal INTEGER NOT NULL,
                            content_sha256 TEXT NOT NULL,
                            embedding_text_sha256 TEXT NOT NULL,
                            PRIMARY KEY (generation_id, chunk_id),
                            FOREIGN KEY (generation_id, source_id)
                                REFERENCES document_versions(generation_id, source_id)
                                ON DELETE CASCADE
                        );
                        CREATE INDEX chunk_versions_source
                            ON chunk_versions(generation_id, source_id, ordinal);

                        CREATE TABLE run_deletions (
                            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                            generation_id TEXT NOT NULL
                                REFERENCES generations(generation_id) ON DELETE CASCADE,
                            source_id TEXT NOT NULL,
                            PRIMARY KEY (run_id, source_id)
                        );
                        PRAGMA user_version = 1;
                        COMMIT;
                        """
                )
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise CatalogError("manifest schema migration failed") from exc
            version = 1
        if version < 2:
            try:
                self._connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    ALTER TABLE generations
                        ADD COLUMN embedding_dimension INTEGER;
                    ALTER TABLE document_versions
                        ADD COLUMN embedding_dimension INTEGER;
                    PRAGMA user_version = 2;
                    COMMIT;
                    """
                )
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise CatalogError("manifest schema migration failed") from exc
            version = 2
        if version < 3:
            try:
                self._connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    ALTER TABLE generations
                        ADD COLUMN layout_refinement_fp TEXT;
                    ALTER TABLE document_versions
                        ADD COLUMN layout_refinement_fp TEXT;
                    PRAGMA user_version = 3;
                    COMMIT;
                    """
                )
            except sqlite3.Error as exc:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                raise CatalogError("manifest schema migration failed") from exc

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[None]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def _run(self, run_id: str) -> sqlite3.Row:
        self._ensure_open()
        if not isinstance(run_id, str) or not run_id.startswith("run_"):
            raise ValueError("run_id is invalid")
        row = self._connection.execute(
            "SELECT * FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise CatalogError("manifest run does not exist")
        return row

    def _writable_run(self, run_id: str) -> sqlite3.Row:
        run = self._run(run_id)
        if run["state"] != RunState.RUNNING.value:
            raise CatalogError("manifest run is no longer writable")
        return run

    def _ensure_open(self) -> None:
        if getattr(self, "_closed", True):
            raise CatalogError("manifest catalog is closed")


def _manifest_document(row: sqlite3.Row) -> ManifestDocument:
    return ManifestDocument(
        generation_id=str(row["generation_id"]),
        generation_state=GenerationState(row["generation_state"]),
        kb_id=str(row["kb_id"]),
        source_id=str(row["source_id"]),
        relative_path=str(row["relative_path"]),
        media_type=str(row["media_type"]),
        size_bytes=int(row["size_bytes"]),
        mtime_ns=int(row["mtime_ns"]),
        content_sha256=str(row["content_sha256"]),
        pipeline=PipelineFingerprint(
            parser=str(row["parser_fp"]),
            restructuring=str(row["restructuring_fp"]),
            layout_refinement=(
                None
                if row["layout_refinement_fp"] is None
                else str(row["layout_refinement_fp"])
            ),
            enrichment=str(row["enrichment_fp"]),
            chunking=str(row["chunking_fp"]),
            embedding=str(row["embedding_fp"]),
            indexing=str(row["indexing_fp"]),
            embedding_dimension=(
                None
                if row["embedding_dimension"] is None
                else int(row["embedding_dimension"])
            ),
        ),
        chunk_count=int(row["chunk_count"]),
    )


def _validate_kb_id(value: str) -> None:
    if not isinstance(value, str) or _KB_ID.fullmatch(value) is None:
        raise ValueError("kb_id must be a safe bounded identifier")


def _validate_generation_id(value: str) -> None:
    if not isinstance(value, str) or _GENERATION_ID.fullmatch(value) is None:
        raise ValueError("generation_id must be a bounded gen_ identifier")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")
