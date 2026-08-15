"""Durable, deterministic continuous ingestion above :class:`RAGIndexManager`.

Routine filesystem reconciliation does not need an LLM decision.  This module
polls the application-owned document root, waits for a stable content snapshot,
and invokes the incremental manager.  The management Agent remains the control
plane for status, rebuild, rollback, and failure explanation.
"""

from __future__ import annotations

import asyncio
import errno
import fcntl
import json
import os
import re
import secrets
import stat
import time
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .diagnostics import PipelineExecutionLog, PipelineLogEvent
from .models import RAGIndexError, SourceDocument, stable_digest
from .pipeline import RAGIndexManager
from .scanner import scan_document_directory


_STATE_SCHEMA_VERSION = 1
_MAX_STATE_BYTES = 512 * 1024
_SAFE_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
SupervisorOutcome = Literal[
    "stabilizing",
    "idle",
    "empty",
    "synced",
    "retry_wait",
    "retry_scheduled",
    "quarantined",
    "busy",
]


class SupervisorError(RAGIndexError):
    """Continuous ingestion configuration or durable state is invalid."""


class SupervisorAlreadyRunningError(SupervisorError):
    """Another process already owns the supervisor lease for this state root."""


@dataclass(frozen=True, slots=True)
class SupervisorPolicy:
    """Bounded application policy for polling, settling, and retries."""

    poll_interval_seconds: float = 5.0
    stability_window_seconds: float = 15.0
    full_reconcile_interval_seconds: float = 300.0
    max_attempts: int = 5
    initial_retry_seconds: float = 5.0
    max_retry_seconds: float = 300.0

    def __post_init__(self) -> None:
        for name, minimum, maximum in (
            ("poll_interval_seconds", 0.05, 3_600.0),
            ("stability_window_seconds", 0.0, 86_400.0),
            ("full_reconcile_interval_seconds", 1.0, 604_800.0),
            ("initial_retry_seconds", 0.05, 86_400.0),
            ("max_retry_seconds", 0.05, 604_800.0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not minimum <= float(value) <= maximum:
                raise ValueError(f"{name} is outside its bounded range")
            object.__setattr__(self, name, float(value))
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 100:
            raise ValueError("max_attempts must be between one and 100")
        if self.max_retry_seconds < self.initial_retry_seconds:
            raise ValueError("max_retry_seconds cannot be below initial_retry_seconds")


@dataclass(frozen=True, slots=True)
class SupervisorObservation:
    """Opaque per-source revision timer; no filename or document content is stored."""

    source_id: str
    revision: str
    observed_at: float

    def __post_init__(self) -> None:
        if not _safe_label(self.source_id) or not self.source_id.startswith("src_"):
            raise SupervisorError("supervisor observation has an invalid source_id")
        if self.revision != "deleted" and not _is_digest(self.revision):
            raise SupervisorError("supervisor observation has an invalid revision")
        if (
            isinstance(self.observed_at, bool)
            or not isinstance(self.observed_at, (int, float))
            or not 0 <= float(self.observed_at) <= 32_503_680_000
        ):
            raise SupervisorError("supervisor observation has an invalid timestamp")
        object.__setattr__(self, "observed_at", float(self.observed_at))


@dataclass(frozen=True, slots=True)
class SupervisorState:
    """Small content-free checkpoint used to resume polling after a restart."""

    schema_version: int
    kb_id: str
    pipeline_digest: str
    observed_digest: str | None = None
    observed_at: float | None = None
    last_success_digest: str | None = None
    last_success_at: float | None = None
    last_reconcile_at: float | None = None
    last_generation_id: str | None = None
    retry_attempts: int = 0
    retry_digest: str | None = None
    next_retry_at: float | None = None
    quarantined_digest: str | None = None
    last_failure_code: str | None = None
    last_failure_stage: str | None = None
    last_failure_types: tuple[str, ...] = ()
    observations: tuple[SupervisorObservation, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != _STATE_SCHEMA_VERSION:
            raise SupervisorError("unsupported supervisor state schema")
        if not _safe_label(self.kb_id):
            raise SupervisorError("supervisor state has an invalid kb_id")
        for name in (
            "pipeline_digest",
            "observed_digest",
            "last_success_digest",
            "retry_digest",
            "quarantined_digest",
        ):
            value = getattr(self, name)
            if value is not None and not _is_digest(value):
                raise SupervisorError(f"supervisor state has an invalid {name}")
        for name in (
            "observed_at",
            "last_success_at",
            "last_reconcile_at",
            "next_retry_at",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= float(value) <= 32_503_680_000
            ):
                raise SupervisorError(f"supervisor state has an invalid {name}")
            if value is not None:
                object.__setattr__(self, name, float(value))
        if type(self.retry_attempts) is not int or not 0 <= self.retry_attempts <= 100:
            raise SupervisorError("supervisor retry count is invalid")
        for name in ("last_generation_id", "last_failure_code", "last_failure_stage"):
            value = getattr(self, name)
            if value is not None and not _safe_label(value):
                raise SupervisorError(f"supervisor state has an invalid {name}")
        failure_types = tuple(self.last_failure_types)
        if len(failure_types) > 6 or any(
            not _safe_label(value) for value in failure_types
        ):
            raise SupervisorError("supervisor failure types are invalid")
        object.__setattr__(self, "last_failure_types", failure_types)
        observations = tuple(self.observations)
        if len(observations) > 10_000 or any(
            not isinstance(value, SupervisorObservation) for value in observations
        ):
            raise SupervisorError("supervisor observations are invalid")
        if tuple(sorted(value.source_id for value in observations)) != tuple(
            value.source_id for value in observations
        ) or len({value.source_id for value in observations}) != len(observations):
            raise SupervisorError("supervisor observations must be unique and sorted")
        object.__setattr__(self, "observations", observations)


@dataclass(frozen=True, slots=True)
class SupervisorReport:
    """One bounded observation suitable for a console or monitoring sink."""

    outcome: SupervisorOutcome
    document_count: int
    observed_digest: str | None
    retry_attempts: int = 0
    next_retry_at: float | None = None
    generation_id: str | None = None
    error_code: str | None = None
    failure_stage: str | None = None
    failure_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.outcome not in {
            "stabilizing",
            "idle",
            "empty",
            "synced",
            "retry_wait",
            "retry_scheduled",
            "quarantined",
            "busy",
        }:
            raise ValueError("invalid supervisor outcome")
        if type(self.document_count) is not int or self.document_count < 0:
            raise ValueError("document_count must be non-negative")
        if self.observed_digest is not None and not _is_digest(self.observed_digest):
            raise ValueError("observed_digest must be a SHA-256 digest or None")
        if type(self.retry_attempts) is not int or not 0 <= self.retry_attempts <= 100:
            raise ValueError("retry_attempts must be between zero and 100")
        if self.next_retry_at is not None and (
            isinstance(self.next_retry_at, bool)
            or not isinstance(self.next_retry_at, (int, float))
            or not 0 <= float(self.next_retry_at) <= 32_503_680_000
        ):
            raise ValueError("next_retry_at must be a bounded timestamp or None")
        for name in ("generation_id", "error_code", "failure_stage"):
            value = getattr(self, name)
            if value is not None and not _safe_label(value):
                raise ValueError(f"{name} must be a stable label or None")
        failure_types = tuple(self.failure_types)
        if len(failure_types) > 6 or any(
            not _safe_label(value) for value in failure_types
        ):
            raise ValueError("failure_types must be bounded stable labels")
        object.__setattr__(self, "failure_types", failure_types)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["failure_types"] = list(self.failure_types)
        return {key: item for key, item in value.items() if item is not None}


class SupervisorStateStore:
    """Atomic JSON state plus a single-writer process lease."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser().absolute()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def load(self, *, kb_id: str, pipeline_digest: str) -> SupervisorState:
        default = SupervisorState(
            schema_version=_STATE_SCHEMA_VERSION,
            kb_id=kb_id,
            pipeline_digest=pipeline_digest,
        )
        try:
            descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return default
        except OSError as exc:
            raise SupervisorError("supervisor state cannot be opened safely") from exc
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_STATE_BYTES:
                raise SupervisorError("supervisor state must be a bounded regular file")
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                raw = stream.read(_MAX_STATE_BYTES + 1)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SupervisorError("supervisor state is not valid UTF-8 JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "kb_id",
            "pipeline_digest",
            "observed_digest",
            "observed_at",
            "last_success_digest",
            "last_success_at",
            "last_reconcile_at",
            "last_generation_id",
            "retry_attempts",
            "retry_digest",
            "next_retry_at",
            "quarantined_digest",
            "last_failure_code",
            "last_failure_stage",
            "last_failure_types",
            "observations",
        }:
            raise SupervisorError("supervisor state has an unexpected schema")
        try:
            payload["last_failure_types"] = tuple(payload["last_failure_types"])
            payload["observations"] = tuple(
                SupervisorObservation(**value) for value in payload["observations"]
            )
            state = SupervisorState(**payload)
        except (TypeError, ValueError, SupervisorError) as exc:
            raise SupervisorError("supervisor state values are invalid") from exc
        if state.kb_id != kb_id or state.pipeline_digest != pipeline_digest:
            return default
        return state

    def save(self, state: SupervisorState) -> None:
        if not isinstance(state, SupervisorState):
            raise TypeError("state must be a SupervisorState")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {
                **asdict(state),
                "last_failure_types": list(state.last_failure_types),
                "observations": [asdict(value) for value in state.observations],
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(payload) > _MAX_STATE_BYTES:
            raise SupervisorError("supervisor state exceeds its size limit")
        temporary = self.path.with_name(f".{self.path.name}.{secrets.token_hex(8)}.tmp")
        descriptor = -1
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            raise SupervisorError(
                "supervisor state could not be saved atomically"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def lease(self) -> _SupervisorLease:
        return _SupervisorLease(self.lock_path)

    def operation_lease(self) -> _SupervisorLease:
        """Return the shared write lease used by watcher and manual management."""

        return _SupervisorLease(
            self.path.with_suffix(self.path.suffix + ".operations.lock"),
            busy_message="another RAG write operation is already running",
        )


class ContinuousIngestionSupervisor:
    """Continuously reconcile one directory into one bound RAG index."""

    def __init__(
        self,
        manager: RAGIndexManager,
        state_store: SupervisorStateStore,
        *,
        policy: SupervisorPolicy | None = None,
        event_sink: Callable[[SupervisorReport], None] | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        if not isinstance(manager, RAGIndexManager):
            raise TypeError("manager must be a RAGIndexManager")
        if not isinstance(state_store, SupervisorStateStore):
            raise TypeError("state_store must be a SupervisorStateStore")
        if event_sink is not None and not callable(event_sink):
            raise TypeError("event_sink must be callable or None")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable or None")
        if sleep is not None and not callable(sleep):
            raise TypeError("sleep must be callable or None")
        self.manager = manager
        self.state_store = state_store
        self.policy = policy or SupervisorPolicy()
        self.event_sink = event_sink
        self._clock = clock or time.time
        self._sleep = sleep or asyncio.sleep
        self._tick_lock = asyncio.Lock()
        self.sink_error_count = 0

    @property
    def state(self) -> SupervisorState:
        return self.state_store.load(
            kb_id=self.manager.config.kb_id,
            pipeline_digest=self.manager.pipeline.digest,
        )

    async def tick(self) -> SupervisorReport:
        """Perform one poll/reconcile decision; useful for tests and notebooks."""

        async with self._tick_lock:
            now = float(self._clock())
            state = self.state
            if (
                state.last_failure_stage == "scan"
                and state.next_retry_at is not None
                and now < state.next_retry_at
            ):
                return self._deliver(
                    SupervisorReport(
                        outcome="retry_wait",
                        document_count=0,
                        observed_digest=state.observed_digest,
                        retry_attempts=state.retry_attempts,
                        next_retry_at=state.next_retry_at,
                        generation_id=state.last_generation_id,
                        error_code=state.last_failure_code,
                        failure_stage=state.last_failure_stage,
                        failure_types=state.last_failure_types,
                    )
                )
            try:
                sources = scan_document_directory(
                    self.manager.config.document_root,
                    kb_id=self.manager.config.kb_id,
                    policy=self.manager.config.scan_policy,
                )
            except Exception as exc:
                return self._record_scan_failure(state, now, exc)

            (
                ready_sources,
                observations,
                current_digest,
                ready_digest,
                has_unstable_sources,
                pipeline_change_blocked,
            ) = _stable_reconciliation_snapshot(
                self.manager,
                sources,
                state.observations,
                now=now,
                stability_window_seconds=self.policy.stability_window_seconds,
            )
            current_changed = current_digest != state.observed_digest
            ready_changed = ready_digest != state.retry_digest
            updated_state = replace(
                state,
                observed_digest=current_digest,
                observed_at=now if current_changed else state.observed_at,
                observations=observations,
                retry_attempts=0 if ready_changed else state.retry_attempts,
                retry_digest=None if ready_changed else state.retry_digest,
                next_retry_at=None if ready_changed else state.next_retry_at,
                quarantined_digest=(
                    None
                    if state.quarantined_digest != ready_digest
                    else state.quarantined_digest
                ),
                last_failure_code=(None if ready_changed else state.last_failure_code),
                last_failure_stage=(
                    None if ready_changed else state.last_failure_stage
                ),
                last_failure_types=() if ready_changed else state.last_failure_types,
            )
            if updated_state != state:
                self.state_store.save(updated_state)
            state = updated_state

            if not ready_sources:
                outcome: SupervisorOutcome = "stabilizing" if sources else "empty"
                return self._deliver(_report(outcome, sources, state))
            if pipeline_change_blocked:
                return self._deliver(_report("stabilizing", sources, state))
            if state.quarantined_digest == ready_digest:
                return self._deliver(_report("quarantined", sources, state))
            if state.next_retry_at is not None and now < state.next_retry_at:
                return self._deliver(_report("retry_wait", sources, state))

            reconcile_due = (
                state.last_success_digest != ready_digest
                or state.last_reconcile_at is None
                or now - state.last_reconcile_at
                >= self.policy.full_reconcile_interval_seconds
            )
            if not reconcile_due:
                outcome = "stabilizing" if has_unstable_sources else "idle"
                return self._deliver(_report(outcome, sources, state))

            log = self.manager.execution_log
            correlation_id = f"watch_{secrets.token_hex(16)}"
            correlation = log.bind(correlation_id) if log is not None else nullcontext()
            try:
                with correlation:
                    result = await self.manager.sync_snapshot(ready_sources)
            except SupervisorAlreadyRunningError:
                return self._deliver(_report("busy", sources, state))
            except Exception as exc:
                return self._record_sync_failure(
                    state,
                    sources,
                    now,
                    ready_digest,
                    exc,
                    log=log,
                    correlation_id=correlation_id,
                )

            state = replace(
                state,
                last_success_digest=ready_digest,
                last_success_at=now,
                last_reconcile_at=now,
                last_generation_id=result.generation_id,
                retry_attempts=0,
                retry_digest=None,
                next_retry_at=None,
                quarantined_digest=None,
                last_failure_code=None,
                last_failure_stage=None,
                last_failure_types=(),
            )
            self.state_store.save(state)
            return self._deliver(_report("synced", sources, state))

    async def run_forever(
        self,
        *,
        stop_event: asyncio.Event | None = None,
        max_cycles: int | None = None,
    ) -> None:
        """Hold the process lease and reconcile until cancelled or stopped."""

        if stop_event is not None and not isinstance(stop_event, asyncio.Event):
            raise TypeError("stop_event must be an asyncio.Event or None")
        if max_cycles is not None and (type(max_cycles) is not int or max_cycles < 1):
            raise ValueError("max_cycles must be a positive integer or None")
        cycles = 0
        with self.state_store.lease():
            while stop_event is None or not stop_event.is_set():
                await self.tick()
                cycles += 1
                if max_cycles is not None and cycles >= max_cycles:
                    return
                if stop_event is None:
                    await self._sleep(self.policy.poll_interval_seconds)
                else:
                    try:
                        await asyncio.wait_for(
                            stop_event.wait(),
                            timeout=self.policy.poll_interval_seconds,
                        )
                    except TimeoutError:
                        pass

    def _record_scan_failure(
        self,
        state: SupervisorState,
        now: float,
        error: BaseException,
    ) -> SupervisorReport:
        attempts = min(state.retry_attempts + 1, self.policy.max_attempts)
        retry_at = now + self._retry_delay(attempts)
        failure_types = _safe_exception_types(error)
        state = replace(
            state,
            retry_attempts=attempts,
            retry_digest=None,
            next_retry_at=retry_at,
            last_failure_code="scan_failed",
            last_failure_stage="scan",
            last_failure_types=failure_types,
        )
        self.state_store.save(state)
        report = SupervisorReport(
            outcome="retry_scheduled",
            document_count=0,
            observed_digest=state.observed_digest,
            retry_attempts=attempts,
            next_retry_at=retry_at,
            generation_id=state.last_generation_id,
            error_code="scan_failed",
            failure_stage="scan",
            failure_types=failure_types,
        )
        return self._deliver(report)

    def _record_sync_failure(
        self,
        state: SupervisorState,
        sources: tuple[SourceDocument, ...],
        now: float,
        digest: str,
        error: BaseException,
        *,
        log: PipelineExecutionLog | None,
        correlation_id: str,
    ) -> SupervisorReport:
        attempts = min(state.retry_attempts + 1, self.policy.max_attempts)
        event = _correlated_failure(log, correlation_id)
        code = event.error_code if event is not None else "sync_failed"
        stage = event.stage if event is not None else "sync"
        failure_types = (
            tuple(
                value for value in (event.exception_type, *event.cause_types) if value
            )
            if event is not None
            else _safe_exception_types(error)
        )
        quarantined = attempts >= self.policy.max_attempts
        retry_at = None if quarantined else now + self._retry_delay(attempts)
        state = replace(
            state,
            retry_attempts=attempts,
            retry_digest=digest,
            next_retry_at=retry_at,
            quarantined_digest=digest if quarantined else None,
            last_failure_code=code,
            last_failure_stage=stage,
            last_failure_types=failure_types,
        )
        self.state_store.save(state)
        return self._deliver(
            SupervisorReport(
                outcome="quarantined" if quarantined else "retry_scheduled",
                document_count=len(sources),
                observed_digest=state.observed_digest,
                retry_attempts=attempts,
                next_retry_at=retry_at,
                generation_id=state.last_generation_id,
                error_code=code,
                failure_stage=stage,
                failure_types=failure_types,
            )
        )

    def _retry_delay(self, attempts: int) -> float:
        return min(
            self.policy.max_retry_seconds,
            self.policy.initial_retry_seconds * (2 ** max(0, attempts - 1)),
        )

    def _deliver(self, report: SupervisorReport) -> SupervisorReport:
        if self.event_sink is not None:
            try:
                self.event_sink(report)
            except BaseException:
                self.sink_error_count += 1
        return report


class _SupervisorLease(AbstractContextManager["_SupervisorLease"]):
    def __init__(
        self,
        path: Path,
        *,
        busy_message: str = "another continuous ingestion supervisor is already running",
    ) -> None:
        self.path = path
        self.busy_message = busy_message
        self._descriptor: int | None = None

    def __enter__(self) -> _SupervisorLease:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.path,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
                0o600,
            )
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if "descriptor" in locals():
                os.close(descriptor)
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise SupervisorAlreadyRunningError(self.busy_message) from exc
            raise SupervisorError("supervisor lease could not be acquired") from exc
        self._descriptor = descriptor
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del exc_type, exc, traceback
        if self._descriptor is not None:
            try:
                fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self._descriptor)
                self._descriptor = None
        return False


def _snapshot_digest(sources: tuple[SourceDocument, ...]) -> str:
    values: list[object] = ["rag-directory-snapshot-v1", len(sources)]
    for source in sources:
        values.extend(
            (
                source.source_id,
                source.source_revision,
                source.size_bytes,
                source.media_type,
            )
        )
    return stable_digest(*values)


def _stable_reconciliation_snapshot(
    manager: RAGIndexManager,
    sources: tuple[SourceDocument, ...],
    previous_observations: tuple[SupervisorObservation, ...],
    *,
    now: float,
    stability_window_seconds: float,
) -> tuple[
    tuple[SourceDocument, ...],
    tuple[SupervisorObservation, ...],
    str,
    str,
    bool,
    bool,
]:
    """Mask unstable published sources while allowing stable files to progress."""

    current = {value.source_id: value for value in sources}
    manifest_values = manager.catalog.list_documents(manager.config.kb_id)
    published = {value.source_id: value for value in manifest_values}
    previous = {value.source_id: value for value in previous_observations}
    observations: list[SupervisorObservation] = []
    stable_ids: set[str] = set()
    for source_id in sorted(set(current) | set(published)):
        revision = (
            current[source_id].source_revision if source_id in current else "deleted"
        )
        old = previous.get(source_id)
        observed_at = (
            old.observed_at if old is not None and old.revision == revision else now
        )
        observation = SupervisorObservation(source_id, revision, observed_at)
        observations.append(observation)
        if now - observed_at >= stability_window_seconds:
            stable_ids.add(source_id)

    effective: list[SourceDocument] = []
    pipeline_change_blocked = False
    for source in sources:
        old = published.get(source.source_id)
        if source.source_id in stable_ids or old is None:
            if source.source_id in stable_ids:
                effective.append(source)
            continue
        if old.pipeline.digest != manager.pipeline.digest or old.chunk_count < 1:
            pipeline_change_blocked = True
        effective.append(
            replace(
                source,
                media_type=old.media_type,
                size_bytes=old.size_bytes,
                mtime_ns=old.mtime_ns,
                sha256=old.content_sha256,
            )
        )

    root = Path(os.path.abspath(os.fspath(manager.config.document_root)))
    for source_id, old in published.items():
        if source_id in current or source_id in stable_ids:
            continue
        effective.append(
            SourceDocument(
                kb_id=old.kb_id,
                source_id=old.source_id,
                root=root,
                path=root.joinpath(*old.relative_path.split("/")),
                relative_path=old.relative_path,
                media_type=old.media_type,
                size_bytes=old.size_bytes,
                mtime_ns=old.mtime_ns,
                sha256=old.content_sha256,
                device=0,
                inode=0,
            )
        )

    effective_values = tuple(sorted(effective, key=lambda value: value.relative_path))
    observation_values = tuple(observations)
    has_unstable = len(stable_ids) < len(observation_values)
    return (
        effective_values,
        observation_values,
        _snapshot_digest(sources),
        _snapshot_digest(effective_values),
        has_unstable,
        pipeline_change_blocked,
    )


def _report(
    outcome: SupervisorOutcome,
    sources: tuple[SourceDocument, ...],
    state: SupervisorState,
) -> SupervisorReport:
    return SupervisorReport(
        outcome=outcome,
        document_count=len(sources),
        observed_digest=state.observed_digest,
        retry_attempts=state.retry_attempts,
        next_retry_at=state.next_retry_at,
        generation_id=state.last_generation_id,
        error_code=state.last_failure_code,
        failure_stage=state.last_failure_stage,
        failure_types=state.last_failure_types,
    )


def _correlated_failure(
    log: PipelineExecutionLog | None,
    correlation_id: str,
) -> PipelineLogEvent | None:
    if log is None:
        return None
    return next(
        (
            event
            for event in reversed(log.events)
            if event.correlation_id == correlation_id and event.status == "failed"
        ),
        None,
    )


def _safe_exception_types(error: BaseException) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(result) < 6:
        seen.add(id(current))
        name = type(current).__name__
        result.append(name if _safe_label(name) else "Exception")
        current = current.__cause__ or current.__context__
    return tuple(result)


def _safe_label(value: object) -> bool:
    return isinstance(value, str) and _SAFE_CODE.fullmatch(value) is not None


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "ContinuousIngestionSupervisor",
    "SupervisorAlreadyRunningError",
    "SupervisorError",
    "SupervisorObservation",
    "SupervisorPolicy",
    "SupervisorReport",
    "SupervisorState",
    "SupervisorStateStore",
]
