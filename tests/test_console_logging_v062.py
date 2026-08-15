from __future__ import annotations

import asyncio
import json
from io import StringIO

import pytest

from moduagent import ConsoleEventSink
from moduagent.messages import FinishReason, Usage
from moduagent.runtime import AgentEvent, AgentResult, EventType


def _publish(sink: ConsoleEventSink, *events: AgentEvent) -> None:
    async def run() -> None:
        for event in events:
            await sink.publish(event)

    asyncio.run(run())


def test_pretty_console_sink_renders_korean_agent_and_tool_progress() -> None:
    stream = StringIO()
    sink = ConsoleEventSink(stream=stream, language="ko", color=False)
    result = AgentResult(
        run_id="run-1",
        output={"private": "MODEL-OUTPUT-MUST-NOT-LEAK"},
        messages=(),
        usage=Usage(30, 12, 42),
        finish_reason=FinishReason.COMPLETED,
    )

    _publish(
        sink,
        AgentEvent(
            EventType.RUN_STARTED,
            "run-1",
            {"agent": "assistant", "user_context": "PRIVATE-CONTEXT"},
        ),
        AgentEvent(
            EventType.MODEL_STARTED,
            "run-1",
            {
                "phase": "act",
                "model_turn": 1,
                "prompt": "PRIVATE-PROMPT",
            },
        ),
        AgentEvent(
            EventType.MODEL_COMPLETED,
            "run-1",
            {
                "phase": "act",
                "model_turn": 1,
                "duration_seconds": 0.125,
                "tool_call_count": 1,
                "response": "PRIVATE-RESPONSE",
            },
        ),
        AgentEvent(
            EventType.TOOL_STARTED,
            "run-1",
            {
                "tool_name": "search_documents",
                "arguments": {"query": "PRIVATE-QUERY"},
            },
        ),
        AgentEvent(
            EventType.TOOL_COMPLETED,
            "run-1",
            {
                "tool_name": "search_documents",
                "success": True,
                "duration_seconds": 0.04,
                "result": "PRIVATE-TOOL-RESULT",
            },
        ),
        AgentEvent(EventType.FINALIZATION_STARTED, "run-1", {"phase": "final"}),
        AgentEvent(EventType.RUN_COMPLETED, "run-1", {"result": result}),
    )

    output = stream.getvalue()
    assert "Agent 실행 시작" in output
    assert "모델 응답 생성 중 · phase=act · turn=1" in output
    assert "모델 응답 수신 완료 · phase=act · turn=1 · 125ms · 1 개 Tool 호출" in output
    assert "    ● Tool 실행 중 · search_documents" in output
    assert "    ✓ Tool 실행 완료 · search_documents · 40ms" in output
    assert "최종 답변 구성 중" in output
    assert "Agent 실행 완료 · 42 토큰" in output
    for private_value in (
        "PRIVATE-CONTEXT",
        "PRIVATE-PROMPT",
        "PRIVATE-RESPONSE",
        "PRIVATE-QUERY",
        "PRIVATE-TOOL-RESULT",
        "MODEL-OUTPUT-MUST-NOT-LEAK",
    ):
        assert private_value not in output


def test_console_sink_json_uses_sealed_projection_and_skips_deltas() -> None:
    stream = StringIO()
    sink = ConsoleEventSink(
        stream=stream,
        output_format="json",
        detail="detailed",
    )

    _publish(
        sink,
        AgentEvent(
            EventType.MODEL_DELTA,
            "run-json",
            {"delta": "PRIVATE-DELTA"},
        ),
        AgentEvent(
            EventType.TOOL_STARTED,
            "run-json",
            {"tool_name": "lookup", "arguments": {"secret": "PRIVATE"}},
        ),
    )

    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event_type"] == "tool_started"
    assert record["data"] == {"tool_name": "lookup"}
    assert "PRIVATE" not in lines[0]


def test_console_sink_summary_filters_low_level_events_but_detailed_shows_them() -> (
    None
):
    event = AgentEvent(
        EventType.CHECKPOINT_SAVED,
        "run-detail",
        {"boundary": "model_completed", "private": "PRIVATE"},
    )
    summary_stream = StringIO()
    detailed_stream = StringIO()

    _publish(ConsoleEventSink(stream=summary_stream), event)
    _publish(
        ConsoleEventSink(stream=detailed_stream, detail="detailed", color=False),
        event,
    )

    assert summary_stream.getvalue() == ""
    assert "Checkpoint saved" in detailed_stream.getvalue()
    assert "PRIVATE" not in detailed_stream.getvalue()


@pytest.mark.parametrize(
    ("keyword", "value", "exception"),
    [
        ("output_format", "yaml", ValueError),
        ("detail", "all", ValueError),
        ("language", "ja", ValueError),
        ("color", "yes", TypeError),
        ("include_timestamp", 1, TypeError),
    ],
)
def test_console_sink_rejects_invalid_configuration(
    keyword: str,
    value: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        ConsoleEventSink(**{keyword: value})
