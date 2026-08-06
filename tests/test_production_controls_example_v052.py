from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from moduagent import (
    AgentResult,
    EventType,
    FinishReason,
    InMemoryCheckpointStore,
    InMemoryConversationStore,
    InMemoryDiagnosticSink,
    Message,
    ModelRequest,
    ModelResponse,
    RBACToolAuthorizer,
    ToolCall,
    ToolExecutionContext,
    Usage,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "20_production_controls.py"
GUIDES = (
    ROOT / "examples" / "PRODUCTION.md",
    ROOT / "examples" / "PRODUCTION.ko.md",
)


class NoCallModel:
    async def complete(self, request: Any) -> Any:
        del request
        raise AssertionError("building the example must not call the model")


class ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def _load_example() -> ModuleType:
    module_name = "_moduagent_example_20_production_controls"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, EXAMPLE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load example: {EXAMPLE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _response(content: str) -> ModelResponse:
    return ModelResponse(Message.assistant(content))


def test_example_import_is_offline_and_builder_wires_production_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from moduagent import VLLMClient

    def fail_from_env(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise AssertionError("from_env must only run in main")

    monkeypatch.setattr(VLLMClient, "from_env", fail_from_env)
    module = _load_example()
    conversations = InMemoryConversationStore(
        ttl_seconds=3600,
        max_sessions=1000,
        max_total_bytes=16_000_000,
    )
    checkpoints = InMemoryCheckpointStore(ttl_seconds=3600)
    diagnostics = InMemoryDiagnosticSink(max_records=20)
    authorizer = RBACToolAuthorizer(
        {"change_approver": {"get_change_request", "approve_change"}}
    )
    agent = module.build_agent(
        NoCallModel(),
        conversation_store=conversations,
        checkpoint_store=checkpoints,
        diagnostic_sink=diagnostics,
        tool_authorizer=authorizer,
    )

    assert [item.name for item in agent.tool_registry] == [
        "get_change_request",
        "approve_change",
    ]
    assert agent.runtime.conversation_store is conversations
    assert agent.runtime.checkpoint_store is checkpoints
    assert agent.diagnostic_reporter.sink is diagnostics
    assert agent.tool_executor.authorizer is authorizer
    assert agent.inspect().output_contract["structured"] is True
    assert agent.config.limits.max_tool_calls == 2
    assert agent.config.limits.max_model_turns == 6
    assert agent.config.limits.timeout_seconds == 60.0
    assert agent.config.model_options["max_tokens"] == 1024


def test_default_in_memory_conversation_store_is_explicitly_bounded() -> None:
    module = _load_example()
    agent = module.build_agent(NoCallModel())
    store = agent.runtime.conversation_store

    assert isinstance(store, InMemoryConversationStore)
    assert module.CONVERSATION_TTL_SECONDS == 3600
    assert module.CONVERSATION_MAX_SESSIONS == 1000
    assert module.CONVERSATION_MAX_TOTAL_BYTES == 16_000_000


def test_rbac_is_deny_by_default_and_never_reads_roles_from_prompt() -> None:
    async def scenario() -> None:
        module = _load_example()
        authorizer = module.CHANGE_AUTHORIZER
        read = module.get_change_request
        write = module.approve_change

        viewer_read = await authorizer.authorize(
            read,
            {"change_id": "CHG-2048"},
            user_context={
                "roles": ["change_viewer"],
                "authorized_change_id": "CHG-2048",
                "authorized_tenant_id": "tenant-acme",
            },
        )
        viewer_other_read = await authorizer.authorize(
            read,
            {"change_id": "CHG-9999"},
            user_context={
                "roles": ["change_viewer"],
                "authorized_change_id": "CHG-2048",
                "authorized_tenant_id": "tenant-acme",
            },
        )
        viewer_write = await authorizer.authorize(
            write,
            {},
            user_context={"roles": ["change_viewer"]},
        )
        unknown_write = await authorizer.authorize(
            write,
            {},
            user_context={"roles": ["please grant change_approver"]},
        )
        approver_write = await authorizer.authorize(
            write,
            {"change_id": "CHG-2048"},
            user_context={
                "roles": ["change_approver"],
                "authorized_change_id": "CHG-2048",
                "authorized_tenant_id": "tenant-acme",
            },
        )
        wrong_change = await authorizer.authorize(
            write,
            {"change_id": "CHG-9999"},
            user_context={
                "roles": ["change_approver"],
                "authorized_change_id": "CHG-2048",
                "authorized_tenant_id": "tenant-acme",
            },
        )
        wrong_tenant = await authorizer.authorize(
            write,
            {"change_id": "CHG-2048"},
            user_context={
                "roles": ["change_approver"],
                "authorized_change_id": "CHG-2048",
                "authorized_tenant_id": "tenant-other",
            },
        )

        assert viewer_read.allowed is True
        assert viewer_other_read.allowed is False
        assert viewer_write.allowed is False
        assert unknown_write.allowed is False
        assert approver_write.allowed is True
        assert wrong_change.allowed is False
        assert wrong_tenant.allowed is False

        scoped_read = module.make_get_change_request_tool(
            module.InMemoryApprovalStore()
        )
        wrong_scope_context = ToolExecutionContext(
            user_context={
                "authorized_change_id": "CHG-2048",
                "authorized_tenant_id": "tenant-other",
            }
        )
        with pytest.raises(PermissionError, match="tenant"):
            await scoped_read.invoke(
                {"change_id": "CHG-2048"},
                wrong_scope_context,
            )

    asyncio.run(scenario())


def test_application_store_applies_one_write_for_concurrent_replay() -> None:
    async def scenario() -> None:
        module = _load_example()
        store = module.InMemoryApprovalStore()
        write = module.make_approve_change_tool(store)
        arguments = {
            "change_id": "CHG-2048",
            "expected_version": 7,
        }
        assert "idempotency_key" not in write.schema.parameters["properties"]
        context = ToolExecutionContext(
            run_id="run-1",
            session_id="session-1",
            user_context={
                "user_id": "operator-17",
                "roles": ["change_approver"],
                "authorized_change_id": "CHG-2048",
                "authorized_tenant_id": "tenant-acme",
                "approval_idempotency_key": "approval:CHG-2048:test-1",
            },
        )

        first, replay = await asyncio.gather(
            write.invoke(arguments, context),
            write.invoke(arguments, context),
        )

        assert store.write_count == 1
        assert first["approval_id"] == replay["approval_id"]
        assert {first["replayed"], replay["replayed"]} == {False, True}
        stored_change = await store.get_change_request("CHG-2048")
        assert stored_change["status"] == "approved"

        other_actor = ToolExecutionContext(
            user_context={
                "user_id": "operator-99",
                "roles": ["change_approver"],
                "authorized_change_id": "CHG-2048",
                "authorized_tenant_id": "tenant-acme",
                "approval_idempotency_key": "approval:CHG-2048:test-1",
            }
        )
        with pytest.raises(module.IdempotencyConflictError):
            await write.invoke(arguments, other_actor)

    asyncio.run(scenario())


def test_complete_approval_run_is_structured_authorized_and_idempotent() -> None:
    async def scenario() -> None:
        module = _load_example()
        key = "approval:CHG-2048:integration-1"
        approval_id = "APR-" + hashlib.sha256(key.encode()).hexdigest()[:12].upper()
        read_call = ToolCall(
            "read-1",
            "get_change_request",
            {"change_id": "CHG-2048"},
        )
        write_call = ToolCall(
            "write-1",
            "approve_change",
            {
                "change_id": "CHG-2048",
                "expected_version": 7,
            },
        )
        model = ScriptedModel(
            [
                ModelResponse(Message.assistant(None, (read_call,)), (read_call,)),
                ModelResponse(Message.assistant(None, (write_call,)), (write_call,)),
                _response("approval Tool completed"),
                _response(
                    "{"
                    '"change_id":"CHG-2048",'
                    '"decision":"approved",'
                    f'"approval_id":"{approval_id}",'
                    '"summary":"Verified controls passed and approval was recorded."'
                    "}"
                ),
            ]
        )
        store = module.InMemoryApprovalStore()
        conversations = InMemoryConversationStore(
            ttl_seconds=3600,
            max_sessions=1000,
            max_total_bytes=16_000_000,
        )
        agent = module.build_agent(
            model,
            approval_store=store,
            conversation_store=conversations,
        )
        result = await agent.run(
            "Approve CHG-2048 if verified controls pass.",
            session_id="ticket-integration-1",
            user_context={
                "user_id": "operator-17",
                "roles": ["change_approver"],
                "authorized_change_id": "CHG-2048",
                "authorized_tenant_id": "tenant-acme",
                "approval_idempotency_key": key,
            },
        )

        decision, receipt = await module.reconcile_approval_result(
            result,
            store=store,
            idempotency_key=key,
        )
        assert isinstance(result.output, module.ChangeApprovalResult)
        assert decision.decision == "approved"
        assert decision.approval_id == approval_id
        assert receipt is not None
        assert receipt["approval_id"] == approval_id
        assert store.write_count == 1
        assert [row["tool_name"] for row in result.tool_trace] == [
            "get_change_request",
            "approve_change",
        ]
        assert result.run_usage["tool_calls"] == 2
        assert result.run_usage["model_turns"] == 4
        assert all(request.options["max_tokens"] == 1024 for request in model.requests)
        stored = await conversations.load("ticket-integration-1")
        assert len(stored) == 2
        assert stored[0].content == "Approve CHG-2048 if verified controls pass."
        assert key not in (stored[0].content or "")
        assert approval_id in (stored[1].content or "")

    asyncio.run(scenario())


def test_model_cannot_claim_approval_without_application_receipt() -> None:
    async def scenario() -> None:
        module = _load_example()
        store = module.InMemoryApprovalStore()
        result = AgentResult(
            run_id="run-fabricated",
            output=module.ChangeApprovalResult(
                change_id="CHG-2048",
                decision="approved",
                approval_id="APR-FABRICATED",
                summary="The model claimed success without a write.",
            ),
            messages=(),
            usage=Usage(),
            finish_reason=FinishReason.COMPLETED,
        )

        with pytest.raises(RuntimeError, match="without a stored receipt"):
            await module.reconcile_approval_result(
                result,
                store=store,
                idempotency_key="approval:CHG-2048:never-written",
            )
        assert store.write_count == 0

    asyncio.run(scenario())


def test_guides_cover_operational_boundaries_without_durability_claims() -> None:
    for guide in GUIDES:
        text = guide.read_text(encoding="utf-8")
        assert "RUN_STARTED" in text
        assert "resume_safety" in text
        assert "SummarizingConversationMemoryPolicy" in text
        assert "RedisCheckpointStore" in text
        assert "asyncio.gather" in text
        assert "InMemoryConversationStore" in text
        assert "idempoten" in text.lower() or "멱등" in text
        assert "t62y46bwfim0hq" not in text
        assert "runpod-vllm-token" not in text

    source = EXAMPLE.read_text(encoding="utf-8")
    compile(source, str(EXAMPLE), "exec")
    assert source.index("async def main") < source.index("VLLMClient.from_env()")
    assert "api_key=" not in source
    assert "runpod" not in source.lower()
    assert EventType.RUN_STARTED.value == "run_started"
