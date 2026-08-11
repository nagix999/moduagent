from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

from moduagent import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolExecutionContext,
)
from moduagent.messages import Message


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "06_waf_log_analysis.py"


class NoCallModel:
    async def complete(self, request: Any) -> Any:
        del request
        raise AssertionError("building the example must not call the model")


class ScriptedModel:
    capabilities = ModelCapabilities(
        streaming=False,
        parallel_tool_calling=True,
        tool_calling_with_structured_output=False,
    )

    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("the WAF example made an unexpected model call")
        return self.responses.pop(0)


def _load_example() -> ModuleType:
    module_name = "_moduagent_example_06_waf_log_analysis"
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load example: {EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _valid_analysis_payload() -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "verdict": "true_positive",
        "confidence": 0.91,
        "summary": "관찰된 SQL 구문이 탐지 규칙과 일치합니다.",
        "evidence": {
            "true_positive": ["payload에 boolean tautology가 관찰됩니다."],
            "false_positive": [],
            "uncertainties": ["단일 로그로 공격 성공 여부는 알 수 없습니다."],
        },
        "risk": {
            "score": 72,
            "rationale": "성공하면 데이터 접근에 영향을 줄 수 있습니다.",
        },
        "attack_syntax": {
            "status": "observed",
            "category": "sql_injection",
            "location": "payload",
            "observed_fragment": "%27%20OR%201%3D1--",
            "explanation": "조건을 항상 참으로 만들려는 형태입니다.",
        },
        "recommendations": ["원본 요청과 애플리케이션 로그를 연계 확인합니다."],
        "human_review_required": True,
    }


def _conflicted_analysis_payload() -> dict[str, object]:
    payload = _valid_analysis_payload()
    payload.update(
        {
            "verdict": "inconclusive",
            "confidence": 0.64,
            "summary": "공격 문법과 정상 검색 문맥이 함께 관찰되어 추가 확인이 필요합니다.",
            "risk": {
                "score": 55,
                "rationale": "공격 형태는 있으나 애플리케이션 도달 여부는 관찰되지 않았습니다.",
            },
        }
    )
    payload["evidence"] = {
        "true_positive": ["디코딩 특징에 SQL boolean tautology 형태가 있습니다."],
        "false_positive": ["검색 경로는 자유 입력과 매개변수화 조회를 사용합니다."],
        "uncertainties": [
            "WAF 차단 이후 애플리케이션 결과와 외부 평판 자료가 없습니다."
        ],
    }
    return payload


def _tool_call_response(names: list[str]) -> ModelResponse:
    calls = tuple(
        ToolCall(f"waf-tool-{index}", name, {})
        for index, name in enumerate(names, start=1)
    )
    return ModelResponse(
        Message.assistant(None, calls),
        calls,
        finish_reason="tool_calls",
    )


def _complete_waf_model(tool_names: list[str]) -> ScriptedModel:
    return ScriptedModel(
        [
            *(_tool_call_response([tool_name]) for tool_name in tool_names),
            ModelResponse(Message.assistant("bounded evidence collected")),
            ModelResponse(
                Message.assistant(
                    json.dumps(
                        _conflicted_analysis_payload(),
                        ensure_ascii=False,
                    )
                )
            ),
        ]
    )


def _scope_and_tools(module: ModuleType) -> tuple[Any, tuple[Any, ...]]:
    log = module.WAFLog.model_validate(module.SAMPLE_WAF_LOG)
    scope = module.WAFRunScope(event_id=module.SAMPLE_EVENT_ID, log=log)
    tools = module.make_waf_evidence_tools(
        scope,
        module.DeterministicWAFEvidenceProvider(),
    )
    return scope, tools


def _invoke_scoped_tool(module: ModuleType, tool: Any) -> Any:
    return asyncio.run(
        tool.invoke(
            {},
            ToolExecutionContext(
                user_context={"authorized_event_id": module.SAMPLE_EVENT_ID}
            ),
        )
    )


def test_waf_example_imports_without_network_or_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from moduagent import VLLMClient

    def fail_from_env(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("VLLMClient.from_env must only run in main")

    monkeypatch.setattr(VLLMClient, "from_env", fail_from_env)
    source = EXAMPLE.read_text(encoding="utf-8")
    compile(source, str(EXAMPLE), "exec")
    module = _load_example()

    assert callable(module.build_agent)
    assert callable(module.analyze_waf_log)
    assert "async with VLLMClient.from_env(" in source
    assert '"max_tokens": 1536' in source
    for forbidden in (
        "api_key=",
        "runpod-vllm-token",
        "t62y46bwfim0hq",
        "07x6ogvl5iyw85",
    ):
        assert forbidden not in source


def test_waf_agent_is_bounded_structured_and_has_six_read_tools() -> None:
    module = _load_example()
    agent = module.build_agent(NoCallModel())
    spec = agent.inspect()

    assert spec.name == "waf-log-analyzer-v01"
    assert spec.execution_profile.kind == "standard"
    assert spec.output_contract["structured"] is True
    assert spec.output_contract["staged_finalization"] is True
    assert [tool.name for tool in agent.tool_registry] == [
        "analyze_payload_encoding",
        "get_waf_rule_context",
        "get_route_context",
        "get_correlated_app_outcome",
        "summarize_related_events",
        "lookup_threat_intel",
    ]
    assert agent.config.limits.max_steps == 8
    assert agent.config.limits.max_tool_calls == 6
    assert agent.config.limits.max_model_turns == 10
    assert agent.config.limits.no_progress_model_turn_threshold == 3
    assert agent.config.limits.timeout_seconds == 120
    assert agent.config.model_options["parallel_tool_calls"] is False
    assert agent.config.limits.parallel_tool_calls is False
    assert agent.config.limits.max_parallel_tools == 1
    assert agent.config.tool_trace_mode == "summary"

    parallel_agent = module.build_agent(
        NoCallModel(),
        parallel_tool_calls=True,
    )
    assert parallel_agent.config.model_options["parallel_tool_calls"] is True
    assert parallel_agent.config.limits.parallel_tool_calls is True
    assert parallel_agent.config.limits.max_parallel_tools == 3


def test_waf_input_accepts_exact_fields_and_builds_an_untrusted_data_request() -> None:
    module = _load_example()
    log = module.WAFLog.model_validate(module.SAMPLE_WAF_LOG)
    request = json.loads(
        module.build_analysis_request(log, event_id=module.SAMPLE_EVENT_ID)
    )

    assert request["event_id"] == module.SAMPLE_EVENT_ID
    assert set(request["waf_log"]) == {
        "action",
        "attack_pattern",
        "dest_country",
        "dest_ip",
        "dest_port",
        "date",
        "payload",
        "signature",
        "src_country",
        "src_ip",
        "vendor",
    }
    assert request["waf_log"]["action"] == "D"
    assert request["waf_log"]["vendor"] == "F5"
    assert request["waf_log"]["src_ip"] == "198.51.100.24"
    assert "event_id" not in request["waf_log"]
    assert "untrusted evidence" in module.INSTRUCTIONS
    assert "not proof" in module.INSTRUCTIONS


def test_six_waf_tools_are_argument_free_scoped_and_deterministic() -> None:
    module = _load_example()
    scope, tools = _scope_and_tools(module)
    expected_names = [
        "analyze_payload_encoding",
        "get_waf_rule_context",
        "get_route_context",
        "get_correlated_app_outcome",
        "summarize_related_events",
        "lookup_threat_intel",
    ]
    original_log = scope.log.model_dump(mode="json")

    assert [tool.name for tool in tools] == expected_names
    assert all(tool.schema.parameters["properties"] == {} for tool in tools)

    first = [_invoke_scoped_tool(module, tool) for tool in tools]
    second = [_invoke_scoped_tool(module, tool) for tool in tools]
    first_json = [item.model_dump_json() for item in first]
    second_json = [item.model_dump_json() for item in second]

    assert first_json == second_json
    assert scope.log.model_dump(mode="json") == original_log
    assert len(original_log) == 11
    assert "event_id" not in original_log
    assert all(
        item.evidence_ref.startswith(f"{module.SAMPLE_EVENT_ID}:") for item in first
    )

    encoding, rule, route, outcome, related, threat_intel = first
    assert encoding.feature_codes == [
        "percent_encoding_observed",
        "base64_encoding_observed",
        "nested_encoding_observed",
        "sql_boolean_tautology_shape",
        "sql_comment_marker_shape",
    ]
    assert [preview.transform_path for preview in encoding.decoded_previews] == [
        "url_percent",
        "url_percent>base64",
    ]
    assert all(preview.preview is None for preview in encoding.decoded_previews)
    assert all(preview.redacted is True for preview in encoding.decoded_previews)
    assert all(preview.untrusted is True for preview in encoding.decoded_previews)
    assert all(preview.decoded_bytes <= 4096 for preview in encoding.decoded_previews)
    assert rule.known_false_positive_codes == ["free_text_keyword_collision"]
    assert route.accepts_free_text is True
    assert route.parameterized_backend_query is True
    assert outcome.blocked_before_application is True
    assert outcome.trace_coverage == "none"
    assert related.status == "no_data"
    assert threat_intel.reputation == "unknown"


def test_payload_decoder_exposes_bounded_text_only_with_explicit_opt_in() -> None:
    module = _load_example()
    scope = module.WAFRunScope(
        event_id=module.SAMPLE_EVENT_ID,
        log=module.WAFLog.model_validate(module.SAMPLE_WAF_LOG),
        include_decoded_previews=True,
    )
    decoder = module.make_waf_evidence_tools(
        scope,
        module.DeterministicWAFEvidenceProvider(),
    )[0]

    result = _invoke_scoped_tool(module, decoder)

    assert result.feature_codes == [
        "percent_encoding_observed",
        "base64_encoding_observed",
        "nested_encoding_observed",
        "sql_boolean_tautology_shape",
        "sql_comment_marker_shape",
    ]
    assert result.decoded_previews[1].preview == "SELECT * FROM users"
    assert all(preview.redacted is False for preview in result.decoded_previews)
    assert all(preview.untrusted is True for preview in result.decoded_previews)
    assert all(preview.decoded_bytes <= 4096 for preview in result.decoded_previews)
    assert all(
        preview.preview is not None and len(preview.preview) <= 160
        for preview in result.decoded_previews
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"include_decoded_previews": 1},
        {"parallel_tool_calls": 1},
    ],
    ids=["decoded-preview-flag", "parallel-tool-flag"],
)
def test_waf_analysis_rejects_non_boolean_flags_before_model_call(
    kwargs: dict[str, object],
) -> None:
    module = _load_example()
    model = ScriptedModel([])

    with pytest.raises(TypeError, match="must be a bool"):
        asyncio.run(
            module.analyze_waf_log(
                model,
                module.SAMPLE_WAF_LOG,
                event_id=module.SAMPLE_EVENT_ID,
                **kwargs,
            )
        )

    assert model.requests == []


@pytest.mark.parametrize("signature", ["", "S" * 512])
def test_rule_tool_accepts_the_full_valid_signature_domain(signature: str) -> None:
    module = _load_example()
    raw = dict(module.SAMPLE_WAF_LOG, signature=signature)
    scope = module.WAFRunScope(
        event_id=module.SAMPLE_EVENT_ID,
        log=module.WAFLog.model_validate(raw),
    )
    rule_tool = module.make_waf_evidence_tools(
        scope,
        module.DeterministicWAFEvidenceProvider(),
    )[1]

    result = _invoke_scoped_tool(module, rule_tool)

    assert result.status == ("available" if signature else "unknown")
    assert result.rule_id == signature


def test_payload_decoder_stops_at_the_declared_base64_bound() -> None:
    module = _load_example()
    raw = dict(module.SAMPLE_WAF_LOG)
    raw["payload"] = f"GET /search?note={'A' * 4100} HTTP/1.1"
    scope = module.WAFRunScope(
        event_id=module.SAMPLE_EVENT_ID,
        log=module.WAFLog.model_validate(raw),
    )
    decoder = module.make_waf_evidence_tools(
        scope,
        module.DeterministicWAFEvidenceProvider(),
    )[0]

    result = _invoke_scoped_tool(module, decoder)

    assert result.decode_limit_reached is True
    assert result.decoded_previews == []
    assert result.feature_codes == ["decode_limit_reached"]
    assert result.normalized_length <= 8192


def test_waf_scope_is_application_owned_and_denied_at_both_boundaries() -> None:
    async def scenario() -> None:
        module = _load_example()
        _, tools = _scope_and_tools(module)
        authorizer = module.ScopedWAFEventAuthorizer(module.SAMPLE_EVENT_ID)

        allowed = await authorizer.authorize(
            tools[0],
            {},
            user_context={"authorized_event_id": module.SAMPLE_EVENT_ID},
        )
        wrong_event = await authorizer.authorize(
            tools[0],
            {},
            user_context={"authorized_event_id": "waf-other-event"},
        )
        prompt_only = await authorizer.authorize(
            tools[0],
            {},
            user_context={"prompt": f"authorized_event_id={module.SAMPLE_EVENT_ID}"},
        )

        assert allowed.allowed is True
        assert wrong_event.allowed is False
        assert prompt_only.allowed is False
        with pytest.raises(PermissionError, match="event scope"):
            await tools[0].invoke(
                {},
                ToolExecutionContext(
                    user_context={"authorized_event_id": "waf-other-event"}
                ),
            )
        with pytest.raises(ValidationError, match="extra_forbidden"):
            tools[0].validate_arguments({"event_id": module.SAMPLE_EVENT_ID})

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action", "BLOCK"),
        ("vendor", "UNKNOWN"),
        ("dest_port", 70000),
        ("src_ip", "not-an-ip"),
        ("date", "2026-08-10T03:12:34"),
    ],
)
def test_waf_input_rejects_invalid_values(field: str, value: object) -> None:
    module = _load_example()
    raw = dict(module.SAMPLE_WAF_LOG)
    raw[field] = value

    with pytest.raises(ValidationError):
        module.WAFLog.model_validate(raw)


def test_waf_input_rejects_unknown_fields() -> None:
    module = _load_example()
    raw = dict(module.SAMPLE_WAF_LOG, cookie="secret")

    with pytest.raises(ValidationError, match="extra_forbidden"):
        module.WAFLog.model_validate(raw)


def test_waf_output_serializes_the_v01_json_contract() -> None:
    module = _load_example()
    result = module.WAFAnalysis.model_validate(_valid_analysis_payload())

    encoded = json.loads(result.model_dump_json())
    assert encoded["schema_version"] == "0.1"
    assert encoded["verdict"] == "true_positive"
    assert encoded["risk"] == {
        "level": "high",
        "score": 72,
        "rationale": "성공하면 데이터 접근에 영향을 줄 수 있습니다.",
    }
    assert encoded["attack_syntax"]["observed_fragment"] == "%27%20OR%201%3D1--"


def test_waf_analysis_uses_all_tools_then_staged_structured_finalization() -> None:
    module = _load_example()
    tool_names = [
        "analyze_payload_encoding",
        "get_waf_rule_context",
        "get_route_context",
        "get_correlated_app_outcome",
        "summarize_related_events",
        "lookup_threat_intel",
    ]
    model = _complete_waf_model(tool_names)

    result = asyncio.run(
        module.analyze_waf_log(
            model,
            module.SAMPLE_WAF_LOG,
            event_id=module.SAMPLE_EVENT_ID,
        )
    )

    assert isinstance(result, module.WAFAnalysis)
    assert result.verdict == "inconclusive"
    assert result.evidence.true_positive
    assert result.evidence.false_positive
    assert result.evidence.uncertainties
    assert len(model.requests) == 8
    tool_requests = model.requests[:6]
    draft_request, final_request = model.requests[-2:]
    assert all(
        [tool.name for tool in request.tools] == tool_names for request in tool_requests
    )
    assert all(request.output_schema is None for request in tool_requests)
    assert [tool.name for tool in draft_request.tools] == tool_names
    assert draft_request.output_schema is None
    assert final_request.tools == ()
    assert final_request.output_schema["title"] == "WAFAnalysis"
    assert all(
        request.options["parallel_tool_calls"] is False
        for request in model.requests[:-1]
    )
    assert "parallel_tool_calls" not in final_request.options

    messages = tool_requests[0].messages
    assert messages[0].role == "system"
    assert messages[-1].role == "user"
    assert "untrusted evidence" in messages[0].content
    assert str(module.SAMPLE_WAF_LOG["payload"]) not in messages[0].content
    assert str(module.SAMPLE_WAF_LOG["payload"]) in messages[-1].content
    request = json.loads(messages[-1].content)
    assert request["event_id"] == module.SAMPLE_EVENT_ID
    assert request["waf_log"]["vendor"] == "F5"

    tool_messages = [
        message for message in draft_request.messages if message.role == "tool"
    ]
    assert len(tool_messages) == 6
    tool_evidence = "".join(message.content or "" for message in tool_messages)
    assert "SELECT * FROM users" not in tool_evidence
    assert '"preview": null' in tool_evidence
    assert '"redacted": true' in tool_evidence
    assert "sql_boolean_tautology_shape" in tool_evidence
    for evidence_ref in (
        f"{module.SAMPLE_EVENT_ID}:payload-encoding",
        f"{module.SAMPLE_EVENT_ID}:fixture:waf-rule",
        f"{module.SAMPLE_EVENT_ID}:fixture:route",
        f"{module.SAMPLE_EVENT_ID}:fixture:app-outcome",
        f"{module.SAMPLE_EVENT_ID}:fixture:related-events",
        f"{module.SAMPLE_EVENT_ID}:fixture:threat-intel",
    ):
        assert evidence_ref in tool_evidence


def test_analyze_waf_log_propagates_decoded_preview_opt_in() -> None:
    module = _load_example()
    tool_names = [
        "analyze_payload_encoding",
        "get_waf_rule_context",
        "get_route_context",
        "get_correlated_app_outcome",
        "summarize_related_events",
        "lookup_threat_intel",
    ]
    model = _complete_waf_model(tool_names)

    result = asyncio.run(
        module.analyze_waf_log(
            model,
            module.SAMPLE_WAF_LOG,
            event_id=module.SAMPLE_EVENT_ID,
            include_decoded_previews=True,
        )
    )

    assert result.verdict == "inconclusive"
    tool_evidence = "".join(
        message.content or ""
        for message in model.requests[-2].messages
        if message.role == "tool"
    )
    assert "SELECT * FROM users" in tool_evidence
    assert '"redacted": false' in tool_evidence
    assert '"preview": null' not in tool_evidence


def test_parallel_tool_call_opt_in_reaches_the_model_request() -> None:
    module = _load_example()
    tool_names = [
        "analyze_payload_encoding",
        "get_waf_rule_context",
        "get_route_context",
        "get_correlated_app_outcome",
        "summarize_related_events",
        "lookup_threat_intel",
    ]
    model = ScriptedModel(
        [
            _tool_call_response(tool_names),
            ModelResponse(Message.assistant("bounded evidence collected")),
            ModelResponse(
                Message.assistant(
                    json.dumps(
                        _conflicted_analysis_payload(),
                        ensure_ascii=False,
                    )
                )
            ),
        ]
    )

    result = asyncio.run(
        module.analyze_waf_log(
            model,
            module.SAMPLE_WAF_LOG,
            event_id=module.SAMPLE_EVENT_ID,
            parallel_tool_calls=True,
        )
    )

    assert result.verdict == "inconclusive"
    assert len(model.requests) == 3
    assert model.requests[0].options["parallel_tool_calls"] is True
    assert model.requests[1].options["parallel_tool_calls"] is True
    assert "parallel_tool_calls" not in model.requests[2].options


def test_invalid_input_is_rejected_before_the_model_is_called() -> None:
    module = _load_example()
    model = ScriptedModel([])
    invalid = dict(module.SAMPLE_WAF_LOG, vendor="UNKNOWN")

    with pytest.raises(ValidationError):
        asyncio.run(module.analyze_waf_log(model, invalid))

    assert model.requests == []


def test_explicit_empty_event_id_is_rejected_before_the_model_is_called() -> None:
    module = _load_example()
    model = ScriptedModel([])

    with pytest.raises(ValueError, match="event_id"):
        asyncio.run(
            module.analyze_waf_log(
                model,
                module.SAMPLE_WAF_LOG,
                event_id="",
            )
        )

    assert model.requests == []


@pytest.mark.parametrize(
    "called_tools",
    [
        [],
        ["analyze_payload_encoding", "get_waf_rule_context"],
    ],
    ids=["no-tools", "partial-tools"],
)
def test_waf_analysis_rejects_incomplete_evidence_after_finalization(
    called_tools: list[str],
) -> None:
    module = _load_example()
    responses = []
    responses.extend(_tool_call_response([tool_name]) for tool_name in called_tools)
    responses.extend(
        [
            ModelResponse(Message.assistant("bounded evidence collected")),
            ModelResponse(
                Message.assistant(
                    json.dumps(
                        _conflicted_analysis_payload(),
                        ensure_ascii=False,
                    )
                )
            ),
        ]
    )
    model = ScriptedModel(responses)

    with pytest.raises(RuntimeError, match="exactly one successful call"):
        asyncio.run(
            module.analyze_waf_log(
                model,
                module.SAMPLE_WAF_LOG,
                event_id=module.SAMPLE_EVENT_ID,
            )
        )

    assert model.responses == []
    assert model.requests[-1].tools == ()
    assert model.requests[-1].output_schema["title"] == "WAFAnalysis"


def test_model_cannot_invent_an_observed_attack_fragment() -> None:
    module = _load_example()
    payload = _valid_analysis_payload()
    payload["attack_syntax"] = {
        "status": "observed",
        "category": "sql_injection",
        "location": "payload",
        "observed_fragment": "INVENTED ATTACK TEXT",
        "explanation": "원문에 없는 문자열입니다.",
    }

    log = module.WAFLog.model_validate(module.SAMPLE_WAF_LOG)
    analysis = module.WAFAnalysis.model_validate(payload)
    with pytest.raises(ValueError, match="must be copied exactly"):
        module.validate_attack_syntax_provenance(log, analysis)


def test_waf_output_rejects_unsupported_or_inconsistent_decisions() -> None:
    module = _load_example()
    base = copy.deepcopy(_valid_analysis_payload())
    base.update(
        {
            "verdict": "inconclusive",
            "confidence": 0.4,
            "summary": "판단 근거가 부족합니다.",
            "human_review_required": False,
        }
    )
    base["evidence"] = {
        "true_positive": [],
        "false_positive": [],
        "uncertainties": ["엔드포인트 문맥이 없습니다."],
    }
    base["attack_syntax"] = {
        "status": "not_observed",
        "category": "unknown",
        "location": "unknown",
        "observed_fragment": None,
        "explanation": "명확한 공격 문법을 확인할 수 없습니다.",
    }

    with pytest.raises(ValidationError, match="literal_error"):
        module.WAFAnalysis.model_validate(base)

    base["human_review_required"] = True
    base["risk"] = {
        "score": 75,
        "rationale": "점수로부터 high가 계산됩니다.",
    }
    result = module.WAFAnalysis.model_validate(base)
    assert result.risk.level == "high"
