from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from typing import Any

import pytest

import moduagent.observability.sinks as observability_sinks
from moduagent.messages import FinishReason, Message, Usage
from moduagent.observability import (
    AuditEventSink,
    CompositeEventSink,
    LoggingEventSink,
    MetricsEventSink,
)
from moduagent.runtime import AgentEvent, AgentResult, EventType


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _logging_sink(
    *,
    level: int = logging.INFO,
) -> tuple[LoggingEventSink, _CaptureHandler]:
    logger = logging.Logger("test.moduagent.logging-v041", level=logging.DEBUG)
    handler = _CaptureHandler()
    logger.addHandler(handler)
    return LoggingEventSink(logger, level=level), handler


def _publish(sink: LoggingEventSink, event: AgentEvent) -> None:
    asyncio.run(sink.publish(event))


def _payload(record: logging.LogRecord) -> dict[str, Any]:
    message = record.getMessage()
    assert message.startswith("agent_event ")
    payload = json.loads(message.removeprefix("agent_event "))
    assert isinstance(payload, dict)
    return payload


def _correlation_hash(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def test_terminal_log_projects_bounded_error_and_safe_summary() -> None:
    sink, handler = _logging_sink(level=logging.DEBUG)
    raw_error = "  model\ninvocation\x00failed  " + ("x" * 600)
    result = AgentResult(
        run_id="run-failed",
        output="private model output",
        messages=(Message.assistant("private transcript"),),
        usage=Usage(7, 3, 10),
        finish_reason=FinishReason.ERROR,
        error=raw_error,
        metadata={
            "error_summary": {
                "category": "model_invocation",
                "code": "provider_error",
                "retryable": True,
                "resumable": False,
                "failure_id": "failure-123",
                "operation": "model",
                "phase": "finalize",
                "provider_finish_reason": "timeout",
                "step_id": "S2",
                "attempt": 3,
                "traceback": "raw traceback must not be logged",
                "api_key": "must-not-leak",
            },
            "provider_response": "private response",
        },
    )

    _publish(
        sink,
        AgentEvent(EventType.RUN_FAILED, "run-failed", {"result": result}),
    )

    assert len(handler.records) == 1
    assert handler.records[0].levelno == logging.ERROR
    data = _payload(handler.records[0])["data"]
    assert data["finish_reason"] == "error"
    assert data["has_output"] is True
    assert data["has_error"] is True
    assert data["message_count"] == 1
    assert data["usage"] == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }
    assert data["error"].startswith("model invocation failed")
    assert len(data["error"]) == 512
    assert all(character.isprintable() for character in data["error"])
    assert data["error_summary"] == {
        "category": "model_invocation",
        "code": "provider_error",
        "failure_id": "failure-123",
        "operation": "model",
        "phase": "finalize",
        "provider_finish_reason": "timeout",
        "step_id": _correlation_hash("S2"),
        "retryable": True,
        "resumable": False,
        "attempt": 3,
    }
    serialized = handler.records[0].getMessage()
    assert "private model output" not in serialized
    assert "private transcript" not in serialized
    assert "raw traceback must not be logged" not in serialized
    assert "must-not-leak" not in serialized
    assert "private response" not in serialized


def test_model_completed_uses_top_level_diagnostic_fields() -> None:
    sink, handler = _logging_sink()

    _publish(
        sink,
        AgentEvent(
            EventType.MODEL_COMPLETED,
            "run-model",
            {
                "step": 4,
                "attempt": 2,
                "phase": "step_result",
                "finish_reason": "tool_calls",
                "tool_call_count": 2,
                "usage": Usage(11, 5, 16),
                "response": {"api_key": "ignored-response-secret"},
            },
        ),
    )
    _publish(
        sink,
        AgentEvent(
            EventType.MODEL_COMPLETED,
            "run-model",
            {
                "step": 5,
                "attempt": 1,
                "phase": "finalize",
                "finish_reason": FinishReason.COMPLETED,
                "usage": Usage(3, 2, 5),
            },
        ),
    )

    first = _payload(handler.records[0])["data"]
    assert first == {
        "step": 4,
        "attempt": 2,
        "phase": "step_result",
        "finish_reason": "tool_calls",
        "usage": {
            "input_tokens": 11,
            "output_tokens": 5,
            "total_tokens": 16,
        },
        "tool_call_count": 2,
    }
    second = _payload(handler.records[1])["data"]
    assert second["finish_reason"] == "completed"
    assert "tool_call_count" not in second
    assert "ignored-response-secret" not in handler.records[0].getMessage()


def test_model_attempt_logs_keep_only_safe_evidence() -> None:
    sink, handler = _logging_sink()
    secret = "PRIVATE prompt, tool arguments, and provider body"
    events = (
        AgentEvent(
            EventType.MODEL_STARTED,
            "run-model-evidence",
            {
                "step": 2,
                "attempt": 1,
                "model_turn": 4,
                "phase": "act",
                "message_count": 5,
                "tool_count": 2,
                "has_output_schema": True,
                "streaming": False,
                "messages": [{"content": secret}],
                "tools": [{"arguments": secret}],
            },
        ),
        AgentEvent(
            EventType.MODEL_FAILED,
            "run-model-evidence",
            {
                "step": 2,
                "attempt": 1,
                "model_turn": 4,
                "phase": "act",
                "duration_seconds": 0.25,
                "error_type": "ReadTimeout",
                "code": "model_timeout",
                "retryable": True,
                "terminal": False,
                "provider_body": secret,
            },
        ),
        AgentEvent(
            EventType.RETRY,
            "run-model-evidence",
            {
                "operation": "model",
                "attempt": 1,
                "model_turn": 4,
                "phase": "act",
                "duration_seconds": 0.25,
                "error_type": "ReadTimeout",
                "code": "model_timeout",
                "retryable": True,
                "error": secret,
            },
        ),
    )

    for event in events:
        _publish(sink, event)

    assert _payload(handler.records[0])["data"] == {
        "step": 2,
        "attempt": 1,
        "model_turn": 4,
        "phase": "act",
        "message_count": 5,
        "tool_count": 2,
        "has_output_schema": True,
        "streaming": False,
    }
    assert _payload(handler.records[1])["data"] == {
        "step": 2,
        "attempt": 1,
        "model_turn": 4,
        "phase": "act",
        "duration_seconds": 0.25,
        "error_type": "ReadTimeout",
        "code": "model_timeout",
        "retryable": True,
        "terminal": False,
    }
    assert _payload(handler.records[2])["data"] == {
        "operation": "model",
        "attempt": 1,
        "model_turn": 4,
        "phase": "act",
        "duration_seconds": 0.25,
        "error_type": "ReadTimeout",
        "code": "model_timeout",
        "retryable": True,
    }
    assert [record.levelno for record in handler.records] == [
        logging.INFO,
        logging.WARNING,
        logging.WARNING,
    ]
    assert secret not in "\n".join(record.getMessage() for record in handler.records)


def test_logging_sink_skips_deltas_by_default_and_can_opt_in() -> None:
    sink, handler = _logging_sink()
    delta_events = tuple(
        AgentEvent(event_type, "run-deltas", {"delta": "private output"})
        for event_type in (
            EventType.MODEL_DELTA,
            EventType.STEP_MODEL_DELTA,
            EventType.FINAL_DELTA,
        )
    )

    for event in delta_events:
        _publish(sink, event)
    assert handler.records == []

    logger = logging.Logger("test.moduagent.logging-deltas", level=logging.DEBUG)
    enabled_handler = _CaptureHandler()
    logger.addHandler(enabled_handler)
    enabled = LoggingEventSink(logger, include_deltas=True)
    for event in delta_events:
        _publish(enabled, event)

    assert len(enabled_handler.records) == 3
    assert all(
        _payload(record)["data"] == {"delta_chars": 14, "delta_bytes": 14}
        for record in enabled_handler.records
    )
    assert "private output" not in "\n".join(
        record.getMessage() for record in enabled_handler.records
    )


def test_logging_mask_handles_acronym_and_separator_secret_keys() -> None:
    masked = observability_sinks.mask_sensitive(
        {
            "APIKey": "PRIVATE-API-KEY",
            "ACCESS_TOKEN": "PRIVATE-ACCESS-TOKEN",
            "client-secret": "PRIVATE-CLIENT-SECRET",
            "safe": "visible",
        }
    )

    assert masked == {
        "APIKey": "[REDACTED]",
        "ACCESS_TOKEN": "[REDACTED]",
        "client-secret": "[REDACTED]",
        "safe": "visible",
    }


def test_tool_log_preserves_safe_correlation_and_failure_fields() -> None:
    sink, handler = _logging_sink(level=logging.DEBUG)

    _publish(
        sink,
        AgentEvent(
            EventType.TOOL_COMPLETED,
            "run-tool",
            {
                "tool_name": "query_db",
                "call_id": "call-7",
                "step": 6,
                "step_id": "S3",
                "attempt": 2,
                "duration_seconds": 0.125,
                "success": False,
                "arguments_fingerprint": "sha256:arguments",
                "arguments": {"password": "must-not-leak"},
                "result": "private database row",
                "failure": {
                    "type": "execution_error",
                    "reason": "invalid_query",
                    "recovery": "repair_call",
                    "retryable": False,
                    "failure_id": "failure-tool-7",
                    "message": "raw database error",
                },
            },
        ),
    )

    assert handler.records[0].levelno == logging.ERROR
    data = _payload(handler.records[0])["data"]
    assert data == {
        "tool_name": "query_db",
        "call_id": _correlation_hash("call-7"),
        "step": 6,
        "step_id": _correlation_hash("S3"),
        "attempt": 2,
        "duration_seconds": 0.125,
        "success": False,
        "arguments_fingerprint": "sha256:arguments",
        "failure": {
            "type": "execution_error",
            "reason": "invalid_query",
            "recovery": "repair_call",
            "retryable": False,
            "failure_id": "failure-tool-7",
        },
    }
    serialized = handler.records[0].getMessage()
    assert "must-not-leak" not in serialized
    assert "private database row" not in serialized
    assert "raw database error" not in serialized


def test_logging_sink_uses_event_aware_levels() -> None:
    sink, handler = _logging_sink(level=logging.DEBUG)
    completed = AgentResult(
        run_id="run-level",
        output="done",
        messages=(),
        usage=Usage(),
        finish_reason=FinishReason.COMPLETED,
    )
    failed = AgentResult(
        run_id="run-level",
        output=None,
        messages=(),
        usage=Usage(),
        finish_reason=FinishReason.ERROR,
        error="run failed",
    )
    events = (
        AgentEvent(EventType.RETRY, "run-level"),
        AgentEvent(EventType.STEP_RETRY, "run-level"),
        AgentEvent(EventType.RUN_FAILED, "run-level", {"result": failed}),
        AgentEvent(EventType.STEP_FAILED, "run-level"),
        AgentEvent(
            EventType.TOOL_COMPLETED,
            "run-level",
            {"tool_name": "lookup", "success": False},
        ),
        AgentEvent(
            EventType.TOOL_COMPLETED,
            "run-level",
            {"tool_name": "lookup", "success": True},
        ),
        AgentEvent(EventType.RUN_COMPLETED, "run-level", {"result": completed}),
    )

    for event in events:
        _publish(sink, event)

    assert [record.levelno for record in handler.records] == [
        logging.WARNING,
        logging.WARNING,
        logging.ERROR,
        logging.ERROR,
        logging.ERROR,
        logging.DEBUG,
        logging.DEBUG,
    ]
    completed_data = _payload(handler.records[-1])["data"]
    assert completed_data["error"] is None
    assert completed_data["error_summary"] is None


def test_logging_and_audit_use_exhaustive_minimal_event_projections() -> None:
    sink, handler = _logging_sink()
    audit = AuditEventSink(event_types=set(EventType))
    secret = "PRIVATE prompt, model facts, and SQL"
    events = (
        AgentEvent(
            EventType.STEP_STARTED,
            "run-safe-events",
            {
                "step_id": secret,
                "attempt": 1,
                "plan_version": 2,
                "objective": secret,
            },
        ),
        AgentEvent(
            EventType.POLICY_DECISION,
            "run-safe-events",
            {
                "kind": "retry_step",
                "source": "model",
                "metadata": {
                    "execution_state": {"facts": [secret]},
                    "raw_output": secret,
                },
            },
        ),
        AgentEvent(
            EventType.STEP_RETRY,
            "run-safe-events",
            {
                "step_id": "S1",
                "attempt": 2,
                "count_attempt": True,
                "reason": secret,
            },
        ),
        AgentEvent(
            EventType.PLAN_REVISED,
            "run-safe-events",
            {
                "plan_version": 3,
                "replan_count": 1,
                "reason": secret,
            },
        ),
        AgentEvent(
            EventType.SKILL_DENIED,
            "run-safe-events",
            {
                "skill_name": "reporting",
                "error_type": "authorization_error",
                "reason": secret,
                "unknown_payload": secret,
            },
        ),
    )

    for event in events:
        _publish(sink, event)
        _publish(audit, event)

    serialized_logs = "\n".join(record.getMessage() for record in handler.records)
    serialized_audit = json.dumps(audit.records)
    assert secret not in serialized_logs
    assert secret not in serialized_audit
    assert _payload(handler.records[0])["data"] == {
        "step_id": _correlation_hash(secret),
        "attempt": 1,
        "plan_version": 2,
    }
    assert _payload(handler.records[1])["data"] == {
        "kind": "retry_step",
        "source": "model",
    }
    assert "reason" not in _payload(handler.records[2])["data"]
    assert "reason" not in _payload(handler.records[3])["data"]
    assert "reason" not in _payload(handler.records[4])["data"]


def test_observability_projection_table_covers_every_event_type() -> None:
    assert set(observability_sinks._OBSERVABILITY_EVENT_FIELDS) == set(EventType)


def test_sync_observability_adapters_cannot_block_the_event_loop() -> None:
    release = threading.Event()

    class BlockingWriter:
        def write(self, record: Any) -> None:
            del record
            release.wait(timeout=1)

    class BlockingRecorder:
        def increment(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            release.wait(timeout=1)

        def observe(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            release.wait(timeout=1)

    class BlockingEventSink:
        def publish(self, event: AgentEvent) -> None:
            del event
            release.wait(timeout=1)

    event = AgentEvent(EventType.RUN_STARTED, "run-blocking-observability")
    sinks = (
        AuditEventSink(BlockingWriter()),
        MetricsEventSink(BlockingRecorder()),
        CompositeEventSink([BlockingEventSink()]),
    )

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        for sink in sinks:
            started = loop.time()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(sink.publish(event), timeout=0.01)
            assert loop.time() - started < 0.05
        release.set()
        await asyncio.sleep(0.02)

    asyncio.run(scenario())
