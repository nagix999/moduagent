from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from moduagent import (
    Agent,
    AgentConfig,
    AuditEventSink,
    CompositeEventSink,
    EventType,
    FinishReason,
    LoggingEventSink,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    Plan,
    PlanAndExecutePolicy,
    PlanStep,
    RunLimits,
    StepResult,
    StepValidation,
    StepValidator,
    ValidationKind,
)
from moduagent.runtime import AgentEvent


class _StaticPlanGenerator:
    async def create(self, context: Any) -> Plan:
        return Plan(
            [
                PlanStep(
                    step_id="S1",
                    objective="validate the result",
                    completion_criteria=["the result is valid"],
                )
            ]
        )

    async def revise(self, context: Any, plan: Plan, feedback: str) -> Plan:
        return plan


class _ScriptedModel:
    capabilities = ModelCapabilities(streaming=False)

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return self.responses.pop(0)

    async def stream(self, request: ModelRequest):
        raise AssertionError("stream should not be called")
        yield


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _RejectingValidator(StepValidator):
    def validate(self, step: PlanStep, result: StepResult) -> StepValidation:
        return StepValidation(
            ValidationKind.FAIL,
            reason="PRIVATE VALIDATOR REASON",
        )


def _log_payloads(handler: _CaptureHandler) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for record in handler.records:
        message = record.getMessage()
        if message.startswith("agent_event "):
            payloads.append(json.loads(message.removeprefix("agent_event ")))
    return payloads


def test_schema_invalid_retry_and_max_attempt_are_structurally_observable() -> None:
    async def scenario() -> None:
        secret = "PRIVATE-MODEL-VALUE"
        malformed = f'{{"step_id":"S1","facts":["{secret}"]'
        logger = logging.Logger("test.semantic-validation", logging.DEBUG)
        handler = _CaptureHandler()
        logger.addHandler(handler)
        audit_records: list[dict[str, Any]] = []
        audit = AuditEventSink(
            audit_records.append,
            event_types={EventType.STEP_RETRY, EventType.STEP_FAILED},
        )
        agent = Agent(
            config=AgentConfig(
                "semantic-validation",
                "Validate each step.",
                limits=RunLimits(max_step_attempts=2),
            ),
            model=_ScriptedModel(
                [
                    ModelResponse(Message.assistant(malformed)),
                    ModelResponse(Message.assistant(malformed)),
                ]
            ),
            decision_policy=PlanAndExecutePolicy(_StaticPlanGenerator()),
            event_sink=CompositeEventSink((LoggingEventSink(logger), audit)),
        )

        events = [
            event
            async for event in agent.stream(
                "validate",
                include_internal=True,
            )
        ]
        result = events[-1].data["result"]

        retry = next(event for event in events if event.type is EventType.STEP_RETRY)
        failed = next(event for event in events if event.type is EventType.STEP_FAILED)
        assert retry.data["validation_code"] == "step_result_schema_invalid"
        assert retry.data["validation_location"] == "step_result"
        assert failed.data["validation_code"] == ("step_result_max_attempts_exceeded")
        assert failed.data["validation_cause_code"] == ("step_result_schema_invalid")

        assert result.finish_reason is FinishReason.ERROR
        assert result.metadata["validation_failure"] == {
            "code": "step_result_max_attempts_exceeded",
            "location": "step_result",
            "phase": "failed",
            "step_id": "S1",
            "attempt": 2,
            "cause_code": "step_result_schema_invalid",
        }
        assert result.metadata["error_summary"] == {
            "category": "step_validation",
            "code": "step_result_max_attempts_exceeded",
            "component": "policy",
            "operation": "step_result",
            "retryable": False,
            "resumable": False,
            "phase": "failed",
            "step_id": "S1",
            "attempt": 2,
        }

        projected = [
            payload
            for payload in _log_payloads(handler)
            if payload["event_type"]
            in {EventType.STEP_RETRY.value, EventType.STEP_FAILED.value}
        ]
        assert [payload["data"]["validation_code"] for payload in projected] == [
            "step_result_schema_invalid",
            "step_result_max_attempts_exceeded",
        ]
        assert projected[-1]["data"]["validation_cause_code"] == (
            "step_result_schema_invalid"
        )
        assert [record["data"]["validation_code"] for record in audit_records] == [
            "step_result_schema_invalid",
            "step_result_max_attempts_exceeded",
        ]
        serialized = json.dumps(
            {
                "result_metadata": result.metadata,
                "logs": projected,
                "audit": audit_records,
            },
            ensure_ascii=False,
        )
        assert secret not in serialized

    asyncio.run(scenario())


def test_built_in_sinks_drop_unrecognized_validation_codes() -> None:
    async def scenario() -> None:
        logger = logging.Logger("test.semantic-validation-untrusted", logging.DEBUG)
        handler = _CaptureHandler()
        logger.addHandler(handler)
        audit_records: list[dict[str, Any]] = []
        event = AgentEvent(
            EventType.STEP_FAILED,
            "untrusted-validation",
            {
                "step_id": "S1",
                "attempt": 1,
                "validation_code": "PRIVATE_STABLE_SECRET",
                "validation_cause_code": "PRIVATE_STABLE_CAUSE",
                "validation_location": "PRIVATE_STABLE_LOCATION",
                "reason": "PRIVATE RAW REASON",
            },
        )

        await LoggingEventSink(logger).publish(event)
        await AuditEventSink(
            audit_records.append,
            event_types={EventType.STEP_FAILED},
        ).publish(event)

        logged_data = _log_payloads(handler)[0]["data"]
        audited_data = audit_records[0]["data"]
        for data in (logged_data, audited_data):
            assert "validation_code" not in data
            assert "validation_cause_code" not in data
            assert "validation_location" not in data
            assert "reason" not in data

    asyncio.run(scenario())


def test_non_exception_validator_rejection_has_structured_terminal_summary() -> None:
    async def scenario() -> None:
        payload = json.dumps(
            {
                "step_id": "S1",
                "status": "completed",
                "facts": ["done"],
                "completion_evidence": ["the result is valid"],
            }
        )
        logger = logging.Logger("test.semantic-validator-reject", logging.DEBUG)
        handler = _CaptureHandler()
        logger.addHandler(handler)
        agent = Agent(
            config=AgentConfig(
                "semantic-validator-reject",
                "Validate each step.",
            ),
            model=_ScriptedModel([ModelResponse(Message.assistant(payload))]),
            decision_policy=PlanAndExecutePolicy(
                _StaticPlanGenerator(),
                step_validator=_RejectingValidator(),
            ),
            event_sink=LoggingEventSink(logger),
        )

        result = await agent.run("validate")

        assert result.metadata["error_summary"] == {
            "category": "step_validation",
            "code": "step_validation_rejected",
            "component": "policy",
            "operation": "step_validator",
            "retryable": False,
            "resumable": False,
            "phase": "failed",
            "step_id": "S1",
            "attempt": 1,
        }
        failed_log = next(
            payload
            for payload in _log_payloads(handler)
            if payload["event_type"] == EventType.STEP_FAILED.value
        )
        assert failed_log["data"]["validation_code"] == ("step_validation_rejected")
        assert "PRIVATE VALIDATOR REASON" not in json.dumps(
            {
                "metadata": result.metadata,
                "log": failed_log,
            }
        )

    asyncio.run(scenario())
