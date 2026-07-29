from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from moduagent.observability.diagnostics import (
    CompositeDiagnosticSink,
    DiagnosticFrame,
    DiagnosticReporter,
    FailureDiagnostic,
    InMemoryDiagnosticSink,
    LoggingDiagnosticSink,
    NoopDiagnosticSink,
)


def run(awaitable: Any) -> Any:
    return asyncio.run(awaitable)


def test_failure_diagnostic_is_immutable_and_json_safe() -> None:
    record = FailureDiagnostic(
        failure_id="diag-1",
        run_id="run-1",
        component="model",
        operation="complete",
        category="model_invocation",
        code="model_invocation_failed",
        exception_type="ValueError",
        cause_types=("OSError",),
        safe_details={
            "status": 503,
            "api_key": "MUST-NOT-LEAK",
            "nested": {"values": [1, 2]},
        },
        frames=(DiagnosticFrame("/private/service/provider.py", "complete", 42),),
        phase="finalize",
        attempt=2,
        terminal=True,
        retryable=True,
        occurred_at=datetime(2026, 7, 29),
    )

    with pytest.raises(FrozenInstanceError):
        record.code = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        record.safe_details["status"] = 200  # type: ignore[index]
    nested = record.safe_details["nested"]
    assert isinstance(nested, dict)
    with pytest.raises(TypeError):
        nested["other"] = True

    payload = record.to_dict()
    assert payload["schema_version"] == 1
    assert payload["occurred_at"].endswith("+00:00")
    assert payload["frames"][0]["filename"] == "provider.py"
    assert payload["safe_details"]["api_key"] == "[REDACTED]"
    assert payload["safe_details"]["nested"]["values"] == [1, 2]
    assert "MUST-NOT-LEAK" not in json.dumps(payload)


def test_reporter_captures_bounded_cause_types_frames_and_sqlstate() -> None:
    class DatabaseSyntaxError(Exception):
        sqlstate = "42601"
        errno = 22

    def database_call() -> None:
        local_secret = "DATABASE-PASSWORD"
        del local_secret
        raise DatabaseSyntaxError("SELECT secret FROM private_table")

    def wrapped_call() -> None:
        try:
            database_call()
        except DatabaseSyntaxError as exc:
            raise RuntimeError("provider bearer SECRET") from exc

    try:
        wrapped_call()
    except RuntimeError as captured:
        exception = captured

    sink = InMemoryDiagnosticSink()
    reporter = DiagnosticReporter(
        sink,
        failure_id_factory=lambda: "diag-fixed",
        clock=lambda: datetime(2026, 7, 29, tzinfo=timezone.utc),
    )
    failure_id = run(
        reporter.capture_exception(
            exception=exception,
            run_id="run-db",
            component="tool",
            operation="invoke",
            category="tool_invocation",
            code="invalid_read_query",
            phase="act",
            step_id="step-1",
            call_id="call-1",
            tool_name="query_db",
            attempt=1,
            terminal=False,
            retryable=False,
        )
    )

    assert failure_id == "diag-fixed"
    record = sink.get(failure_id)
    assert record is not None
    assert record.exception_type == "RuntimeError"
    assert record.cause_types == ("DatabaseSyntaxError",)
    assert record.safe_details == {"sqlstate": "42601", "errno": 22}
    assert record.step_id == "step-1"
    assert record.call_id == "call-1"
    assert record.tool_name == "query_db"
    assert {frame.function for frame in record.frames} >= {
        "wrapped_call",
        "database_call",
    }
    assert all("/" not in frame.filename for frame in record.frames)

    serialized = json.dumps(record.to_dict(), ensure_ascii=False)
    assert "DATABASE-PASSWORD" not in serialized
    assert "private_table" not in serialized
    assert "provider bearer SECRET" not in serialized
    assert "message" not in serialized
    assert "source" not in serialized
    assert "locals" not in serialized


def test_reporter_extracts_http_status_and_pydantic_locations_without_inputs() -> None:
    class Response:
        status_code = 422

    class ProviderError(Exception):
        response = Response()

    class Payload(BaseModel):
        count: int

    try:
        Payload.model_validate({"count": "PRIVATE-CUSTOMER-VALUE"})
    except ValidationError as validation_error:
        try:
            raise ProviderError("PRIVATE-HTTP-BODY") from validation_error
        except ProviderError as captured:
            exception = captured

    sink = InMemoryDiagnosticSink()
    reporter = DiagnosticReporter(sink)
    failure_id = run(
        reporter.capture_exception(
            exception=exception,
            run_id="run-validation",
            component="output",
            operation="decode",
            category="output_validation",
            code="output_validation_failed",
            validation_fields={"count"},
        )
    )
    record = sink.get(failure_id)

    assert record is not None
    assert record.safe_details["http_status"] == 422
    errors = record.safe_details["validation_errors"]
    assert errors[0]["loc"] == ("count",)
    assert isinstance(errors[0]["type"], str)
    serialized = json.dumps(record.to_dict(), ensure_ascii=False)
    assert "PRIVATE-CUSTOMER-VALUE" not in serialized
    assert "PRIVATE-HTTP-BODY" not in serialized


def test_reporter_hides_dynamic_pydantic_mapping_keys() -> None:
    class Payload(BaseModel):
        values: dict[str, int]

    secret_key = "PRIVATE-CUSTOMER-DICTIONARY-KEY"
    try:
        Payload.model_validate({"values": {secret_key: "not-an-integer"}})
    except ValidationError as captured:
        exception = captured

    sink = InMemoryDiagnosticSink()
    reporter = DiagnosticReporter(sink)
    failure_id = run(
        reporter.capture_exception(
            exception=exception,
            run_id="run-dynamic-key",
            component="output",
            operation="decode",
            category="output_validation",
            code="output_validation_failed",
            validation_fields={"values"},
        )
    )
    record = sink.get(failure_id)

    assert record is not None
    location = record.safe_details["validation_errors"][0]["loc"]
    assert location == ("values", "[DYNAMIC_KEY]")
    assert secret_key not in json.dumps(record.to_dict())


def test_reporter_hides_integer_pydantic_mapping_keys() -> None:
    class Payload(BaseModel):
        values: dict[int, int]

    secret_key = 123456789
    try:
        Payload.model_validate({"values": {secret_key: "not-an-integer"}})
    except ValidationError as captured:
        exception = captured

    sink = InMemoryDiagnosticSink()
    reporter = DiagnosticReporter(sink)
    failure_id = run(
        reporter.capture_exception(
            exception=exception,
            run_id="run-integer-key",
            component="output",
            operation="decode",
            category="output_validation",
            code="output_validation_failed",
            validation_fields={"values"},
        )
    )
    record = sink.get(failure_id)

    assert record is not None
    location = record.safe_details["validation_errors"][0]["loc"]
    assert location == ("values", "[INDEX_OR_KEY]")
    assert str(secret_key) not in json.dumps(record.to_dict())


def test_reporter_does_not_invoke_exception_attribute_descriptors() -> None:
    class HostileError(RuntimeError):
        attribute_was_read = False

        @property
        def sqlstate(self) -> str:
            type(self).attribute_was_read = True
            time.sleep(0.08)
            return "42601"

    sink = InMemoryDiagnosticSink()
    reporter = DiagnosticReporter(sink, timeout_seconds=0.01)

    async def scenario() -> float:
        loop = asyncio.get_running_loop()
        started = loop.time()
        await reporter.capture_exception(
            exception=HostileError("private"),
            run_id="run-hostile-property",
            component="tool",
            operation="invoke",
            category="tool_invocation",
            code="execution_error",
        )
        return loop.time() - started

    elapsed = run(scenario())

    assert elapsed < 0.05
    assert HostileError.attribute_was_read is False


def test_reporter_extracts_errno_from_real_oserror_without_dynamic_lookup() -> None:
    class HostileOSError(OSError):
        attribute_was_read = False

        @property
        def errno(self) -> int:
            type(self).attribute_was_read = True
            raise AssertionError("subclass property must not be invoked")

    error = HostileOSError(13, "PRIVATE-PATH")
    sink = InMemoryDiagnosticSink()
    reporter = DiagnosticReporter(sink)

    failure_id = run(
        reporter.capture_exception(
            exception=error,
            run_id="run-oserror",
            component="tool",
            operation="invoke",
            category="tool_invocation",
            code="execution_error",
        )
    )
    record = sink.get(failure_id)

    assert record is not None
    assert record.safe_details["errno"] == 13
    assert HostileOSError.attribute_was_read is False
    assert "PRIVATE-PATH" not in json.dumps(record.to_dict())


def test_reporter_does_not_invoke_exception_chain_attribute_hooks() -> None:
    class HostileError(RuntimeError):
        attribute_reads = 0

        def __getattribute__(self, name: str) -> Any:
            if name in {
                "__cause__",
                "__context__",
                "__suppress_context__",
                "__traceback__",
            }:
                type(self).attribute_reads += 1
                time.sleep(0.05)
            return super().__getattribute__(name)

    try:
        raise HostileError("PRIVATE")
    except HostileError as error:
        exception = error

    sink = InMemoryDiagnosticSink()
    reporter = DiagnosticReporter(sink, timeout_seconds=0.001)

    async def scenario() -> float:
        loop = asyncio.get_running_loop()
        started = loop.time()
        await reporter.capture_exception(
            exception=exception,
            run_id="run-hostile-chain",
            component="runtime",
            operation="execute",
            category="execution",
            code="run_failed",
        )
        return loop.time() - started

    elapsed = run(scenario())

    assert elapsed < 0.05
    assert HostileError.attribute_reads == 0


def test_reporter_retains_innermost_frames_when_traceback_exceeds_bound() -> None:
    def leaf() -> None:
        raise RuntimeError("PRIVATE")

    def recurse(depth: int) -> None:
        if depth == 0:
            leaf()
            return
        recurse(depth - 1)

    try:
        recurse(20)
    except RuntimeError as error:
        exception = error

    sink = InMemoryDiagnosticSink()
    reporter = DiagnosticReporter(sink, max_frames=4)
    failure_id = run(
        reporter.capture_exception(
            exception=exception,
            run_id="run-deep-traceback",
            component="runtime",
            operation="execute",
            category="execution",
            code="run_failed",
        )
    )
    record = sink.get(failure_id)

    assert record is not None
    assert len(record.frames) == 4
    assert record.frames[-1].function == "leaf"


def test_reporter_records_reused_exception_objects_per_failure_boundary() -> None:
    ids = iter(("diag-1", "diag-2"))
    sink = InMemoryDiagnosticSink()
    reporter = DiagnosticReporter(sink, failure_id_factory=lambda: next(ids))
    exception = RuntimeError("private")

    async def scenario() -> tuple[str | None, str | None]:
        first = await reporter.capture_exception(
            exception=exception,
            run_id="run-1",
            component="tool",
            operation="invoke",
            category="tool_invocation",
            code="execution_error",
            call_id="call-1",
        )
        second = await reporter.capture_exception(
            exception=exception,
            run_id="run-1",
            component="tool",
            operation="invoke",
            category="tool_invocation",
            code="execution_error",
            call_id="call-2",
        )
        await reporter.flush_run("run-1")
        return first, second

    first, second = run(scenario())

    assert (first, second) == ("diag-1", "diag-2")
    assert [record.failure_id for record in sink.records] == ["diag-1", "diag-2"]
    assert [record.call_id for record in sink.records] == ["call-1", "call-2"]


def test_reporter_replaces_colliding_custom_failure_ids() -> None:
    sink = InMemoryDiagnosticSink()
    reporter = DiagnosticReporter(
        sink,
        failure_id_factory=lambda: "diag-collision",
    )

    async def scenario() -> tuple[str | None, str | None]:
        first = await reporter.capture_exception(
            exception=RuntimeError("first"),
            run_id="run-collision",
            component="runtime",
            operation="execute",
            category="execution",
            code="run_failed",
        )
        second = await reporter.capture_exception(
            exception=RuntimeError("second"),
            run_id="run-collision",
            component="runtime",
            operation="execute",
            category="execution",
            code="run_failed",
        )
        await reporter.flush_run("run-collision")
        return first, second

    first, second = run(scenario())

    assert first == "diag-collision"
    assert isinstance(second, str)
    assert second.startswith("diag_")
    assert second != first
    assert sink.get(first) is not None
    assert sink.get(second) is not None


def test_noop_reporter_returns_no_failure_id() -> None:
    reporter = DiagnosticReporter(NoopDiagnosticSink())

    failure_id = run(
        reporter.capture_exception(
            exception=RuntimeError("private"),
            run_id="run-noop",
            component="runtime",
            operation="execute",
            category="execution",
            code="run_failed",
        )
    )

    assert failure_id is None
    assert reporter.drop_count == 0


def test_in_memory_sink_evicts_oldest_record_at_its_bound() -> None:
    ids = iter(("diag-1", "diag-2"))
    sink = InMemoryDiagnosticSink(max_records=1)
    reporter = DiagnosticReporter(sink, failure_id_factory=lambda: next(ids))

    async def scenario() -> None:
        for run_id in ("run-1", "run-2"):
            await reporter.capture_exception(
                exception=RuntimeError(run_id),
                run_id=run_id,
                component="runtime",
                operation="execute",
                category="execution",
                code="run_failed",
            )

    run(scenario())

    assert sink.get("diag-1") is None
    assert sink.get("diag-2") is not None
    assert len(sink.records) == 1


def test_reporter_bounds_cause_cycles_frames_and_safe_detail_size() -> None:
    first = RuntimeError("first")
    second = ValueError("second")
    first.__cause__ = second
    second.__cause__ = first

    sink = InMemoryDiagnosticSink()
    reporter = DiagnosticReporter(
        sink,
        max_cause_depth=1,
        max_frames=0,
    )
    failure_id = run(
        reporter.capture_exception(
            exception=first,
            run_id="run-cycle",
            component="runtime",
            operation="execute",
            category="execution",
            code="run_failed",
            safe_details={"many": "x" * 10_000},
        )
    )
    record = sink.get(failure_id)

    assert record is not None
    assert record.cause_types == ("ValueError",)
    assert record.frames == ()
    assert len(json.dumps(record.to_dict())) < 5000


def test_reporter_isolates_sink_failure_and_timeout() -> None:
    class BrokenSink:
        async def capture(self, record: FailureDiagnostic) -> None:
            assert isinstance(record, FailureDiagnostic)
            raise RuntimeError("sink failed")

    class HangingSink:
        async def capture(self, record: FailureDiagnostic) -> None:
            assert isinstance(record, FailureDiagnostic)
            await asyncio.Future()

    async def scenario() -> tuple[DiagnosticReporter, DiagnosticReporter]:
        broken = DiagnosticReporter(BrokenSink())
        await broken.capture_exception(
            exception=ValueError("private"),
            run_id="broken",
            component="runtime",
            operation="execute",
            category="execution",
            code="run_failed",
        )
        await broken.flush_run("broken")
        hanging = DiagnosticReporter(HangingSink(), timeout_seconds=0.001)
        await hanging.capture_exception(
            exception=ValueError("private"),
            run_id="hanging",
            component="runtime",
            operation="execute",
            category="execution",
            code="run_failed",
        )
        await hanging.flush_run("hanging")
        return broken, hanging

    broken, hanging = run(scenario())

    assert broken.drop_count == 1
    assert isinstance(broken.last_error, RuntimeError)
    assert hanging.drop_count == 1
    assert isinstance(hanging.last_error, asyncio.TimeoutError)


def test_composite_sink_isolates_one_child_and_preserves_healthy_capture() -> None:
    class BrokenSink:
        async def capture(self, record: FailureDiagnostic) -> None:
            del record
            raise RuntimeError("broken")

    memory = InMemoryDiagnosticSink()
    composite = CompositeDiagnosticSink([BrokenSink(), memory])
    reporter = DiagnosticReporter(composite)

    async def scenario() -> str | None:
        failure_id = await reporter.capture_exception(
            exception=LookupError("private"),
            run_id="run-composite",
            component="persistence",
            operation="load",
            category="persistence",
            code="persistence_failed",
            retryable=True,
        )
        await reporter.flush_run("run-composite")
        return failure_id

    failure_id = run(scenario())

    assert memory.get(failure_id) is not None
    assert len(composite.last_errors) == 1
    assert reporter.drop_count == 0


def test_composite_sink_reports_empty_or_total_delivery_failure() -> None:
    class BrokenSink:
        async def capture(self, record: FailureDiagnostic) -> None:
            del record
            raise RuntimeError("broken")

    async def scenario() -> None:
        for index, composite in enumerate(
            (
                CompositeDiagnosticSink(),
                CompositeDiagnosticSink([BrokenSink()]),
            )
        ):
            reporter = DiagnosticReporter(composite)
            run_id = f"run-composite-failed-{index}"
            await reporter.capture_exception(
                exception=RuntimeError("private"),
                run_id=run_id,
                component="runtime",
                operation="execute",
                category="execution",
                code="run_failed",
            )
            await reporter.flush_run(run_id)
            assert reporter.drop_count == 1
            assert isinstance(reporter.last_error, RuntimeError)

    run(scenario())


def test_logging_and_noop_sinks_receive_records_without_exception_messages(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test.moduagent.diagnostics")
    logging_sink = LoggingDiagnosticSink(logger)
    composite = CompositeDiagnosticSink([logging_sink, NoopDiagnosticSink()])
    reporter = DiagnosticReporter(
        composite,
        failure_id_factory=lambda: "diag-log",
    )

    async def scenario() -> str | None:
        failure_id = await reporter.capture_exception(
            exception=RuntimeError("Authorization bearer TOPSECRET"),
            run_id="run-log",
            component="model",
            operation="complete",
            category="model_invocation",
            code="model_invocation_failed",
            safe_details={
                "authorization": "Bearer TOPSECRET",
                "http_status": 500,
            },
        )
        await reporter.flush_run("run-log")
        return failure_id

    with caplog.at_level(logging.ERROR, logger=logger.name):
        failure_id = run(scenario())

    assert failure_id == "diag-log"
    assert "agent_failure" in caplog.text
    assert "diag-log" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "TOPSECRET" not in caplog.text
    assert "[REDACTED]" in caplog.text


def test_logging_diagnostic_sink_does_not_block_the_event_loop() -> None:
    class SlowHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            del record
            time.sleep(0.08)

    logger = logging.Logger("test.moduagent.slow-diagnostic")
    logger.addHandler(SlowHandler())
    reporter = DiagnosticReporter(
        LoggingDiagnosticSink(logger),
        timeout_seconds=0.01,
    )

    async def scenario() -> float:
        loop = asyncio.get_running_loop()
        started = loop.time()
        await reporter.capture_exception(
            exception=RuntimeError("private"),
            run_id="run-slow-log",
            component="runtime",
            operation="execute",
            category="execution",
            code="run_failed",
        )
        elapsed = loop.time() - started
        await reporter.flush_run("run-slow-log")
        return elapsed

    elapsed = run(scenario())

    assert elapsed < 0.05
    assert reporter.drop_count == 1
    assert isinstance(reporter.last_error, asyncio.TimeoutError)


def test_stalled_sync_observability_is_globally_thread_bounded() -> None:
    release = threading.Event()

    class StalledHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            del record
            release.wait(timeout=1)

    logger = logging.Logger("test.moduagent.bounded-diagnostic")
    logger.addHandler(StalledHandler())
    reporter = DiagnosticReporter(
        LoggingDiagnosticSink(logger),
        timeout_seconds=0.002,
        max_pending_deliveries=1,
    )

    async def scenario() -> tuple[int, int]:
        for index in range(12):
            run_id = f"run-bounded-log-{index}"
            await reporter.capture_exception(
                exception=RuntimeError("private"),
                run_id=run_id,
                component="runtime",
                operation="execute",
                category="execution",
                code="run_failed",
            )
            await reporter.flush_run(run_id)
            reporter.clear_run(run_id)
        worker_count = sum(
            thread.name.startswith("moduagent-observability-")
            for thread in threading.enumerate()
        )
        poller_count = sum(
            task.get_name() == "moduagent-observability-completions"
            for task in asyncio.all_tasks()
        )
        release.set()
        await asyncio.sleep(0.05)
        return worker_count, poller_count

    worker_count, poller_count = run(scenario())

    assert worker_count <= 4
    assert poller_count == 1
    assert reporter.drop_count == 12


def test_capture_returns_before_a_hanging_sink_timeout_and_flush_waits() -> None:
    class HangingSink:
        async def capture(self, record: FailureDiagnostic) -> None:
            assert isinstance(record, FailureDiagnostic)
            await asyncio.Future()

    reporter = DiagnosticReporter(HangingSink(), timeout_seconds=0.5)

    async def scenario() -> str | None:
        failure_id = await asyncio.wait_for(
            reporter.capture_exception(
                exception=RuntimeError("Authorization bearer TOPSECRET"),
                run_id="run-non-blocking",
                component="model",
                operation="complete",
                category="model_invocation",
                code="model_invocation_failed",
            ),
            timeout=0.1,
        )
        assert reporter.drop_count == 0
        await reporter.flush_run("run-non-blocking")
        return failure_id

    failure_id = run(scenario())

    assert isinstance(failure_id, str)
    assert reporter.drop_count == 1
    assert isinstance(reporter.last_error, asyncio.TimeoutError)


def test_flush_is_bounded_when_a_sink_temporarily_suppresses_cancellation() -> None:
    release = asyncio.Event()

    class StubbornSink:
        async def capture(self, record: FailureDiagnostic) -> None:
            assert isinstance(record, FailureDiagnostic)
            try:
                await release.wait()
            except asyncio.CancelledError:
                await release.wait()

    reporter = DiagnosticReporter(StubbornSink(), timeout_seconds=0.01)

    async def scenario() -> float:
        loop = asyncio.get_running_loop()
        await reporter.capture_exception(
            exception=RuntimeError("private"),
            run_id="run-stubborn",
            component="runtime",
            operation="execute",
            category="execution",
            code="run_failed",
        )
        started = loop.time()
        await reporter.flush_run("run-stubborn")
        elapsed = loop.time() - started
        release.set()
        await asyncio.sleep(0)
        reporter.clear_run("run-stubborn")
        return elapsed

    elapsed = run(scenario())

    assert elapsed < 0.1
    assert reporter.drop_count == 1
    assert isinstance(reporter.last_error, asyncio.TimeoutError)


def test_clear_run_cancels_pending_delivery() -> None:
    cancelled = asyncio.Event()

    class HangingSink:
        async def capture(self, record: FailureDiagnostic) -> None:
            assert isinstance(record, FailureDiagnostic)
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

    reporter = DiagnosticReporter(HangingSink(), timeout_seconds=60)

    async def scenario() -> None:
        await reporter.capture_exception(
            exception=RuntimeError("private"),
            run_id="run-clear",
            component="runtime",
            operation="execute",
            category="execution",
            code="run_failed",
        )
        reporter.clear_run("run-clear")
        await asyncio.wait_for(cancelled.wait(), timeout=0.1)
        await reporter.flush_run("run-clear")

    run(scenario())

    assert reporter.drop_count == 1
    assert isinstance(reporter.last_error, asyncio.CancelledError)


def test_invalid_reporter_and_record_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        DiagnosticReporter(timeout_seconds=0)
    with pytest.raises(ValueError, match="max_frames"):
        DiagnosticReporter(max_frames=-1)
    with pytest.raises(ValueError, match="max_pending_deliveries"):
        DiagnosticReporter(max_pending_deliveries=0)
    with pytest.raises(ValueError, match="schema version"):
        FailureDiagnostic(
            failure_id="diag",
            run_id="run",
            component="runtime",
            operation="execute",
            category="execution",
            code="failed",
            exception_type="RuntimeError",
            schema_version=2,
        )

    reporter = DiagnosticReporter()
    with pytest.raises(TypeError, match="BaseException"):
        run(
            reporter.capture_exception(
                exception="not-an-exception",  # type: ignore[arg-type]
                run_id="run",
                component="runtime",
                operation="execute",
                category="execution",
                code="failed",
            )
        )
