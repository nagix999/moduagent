from __future__ import annotations

import asyncio
import copy
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import pytest

from moduagent.definitions import AgentEndpoint
from moduagent.delegation import (
    BudgetStateStore,
    DelegationEndpointError,
    DelegationOutcome,
    DelegationReceipt,
    DelegationReceiptStore,
    ExecutionGroupBudgetState,
    InMemoryBudgetStateStore,
    InMemoryDelegationReceiptStore,
    LocalAgentInvoker,
    ReceiptClaim,
)
from moduagent.memory.context import (
    ContextMemoryCursorRegressionError,
    ConversationSummarySnapshot,
    InMemoryContextMemoryStateStore,
    MemoryStateKey,
)
from moduagent.messages import Message
from moduagent.models import (
    ModelCapabilities,
    ModelChunk,
    ModelProtocolError,
    ModelRequest,
    ModelResponse,
    OpenAICompatibleClient,
)
from moduagent.persistence import InMemoryCheckpointStore, RunCheckpoint
from moduagent.persistence.conversation import InMemoryConversationStore

from .contracts import (
    assert_agent_invoker_contract,
    assert_budget_state_store_contract,
    assert_checkpoint_store_contract,
    assert_context_memory_store_contract,
    assert_conversation_store_contract,
    assert_model_provider_contract,
    assert_model_provider_error_contract,
    assert_receipt_store_contract,
)


class _ModelTransport:
    def __init__(
        self,
        *,
        response: Mapping[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = dict(response or {})
        self.error = error
        self.requests: list[dict[str, Any]] = []

    async def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        json: Mapping[str, Any],
        timeout: float | None = None,
    ) -> Mapping[str, Any]:
        self.requests.append(
            {
                "url": url,
                "headers": dict(headers or {}),
                "json": dict(json),
                "timeout": timeout,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response

    async def stream_lines(self, *_: Any, **__: Any) -> AsyncIterator[str]:
        if False:
            yield ""


class _CustomModelClient:
    capabilities = ModelCapabilities(streaming=True, embeddings=True)

    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return ModelResponse(
            Message.assistant("contract-ok"),
            usage={"prompt_tokens": 2, "completion_tokens": 3},
            finish_reason="stop",
            provider_metadata={"provider": "custom-fake"},
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelChunk]:
        yield ModelChunk(response=await self.complete(request))

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        del options
        rows = (inputs,) if isinstance(inputs, str) else tuple(inputs)
        return tuple((float(index),) for index, _ in enumerate(rows))


class _CustomConversationStore:
    durable = True
    supports_idempotent_append = True
    supports_bounded_load_tail = False

    def __init__(self) -> None:
        self.rows: dict[str, list[dict[str, Any]]] = {}
        self.idempotency: dict[tuple[str, str], str] = {}
        self.lock = asyncio.Lock()

    async def load(self, session_id: str) -> list[Message]:
        async with self.lock:
            rows = copy.deepcopy(self.rows.get(session_id, []))
        return [Message.from_dict(row) for row in rows]

    async def append(
        self,
        session_id: str,
        messages: Sequence[Message],
    ) -> None:
        rows = [copy.deepcopy(message.to_dict()) for message in messages]
        async with self.lock:
            self.rows.setdefault(session_id, []).extend(rows)

    async def append_once(
        self,
        session_id: str,
        idempotency_key: str,
        messages: Sequence[Message],
    ) -> bool:
        rows = [copy.deepcopy(message.to_dict()) for message in messages]
        digest = json.dumps(
            rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        key = (session_id, idempotency_key)
        async with self.lock:
            existing = self.idempotency.get(key)
            if existing is not None:
                if existing != digest:
                    raise ValueError("idempotency key was reused")
                return False
            self.idempotency[key] = digest
            self.rows.setdefault(session_id, []).extend(rows)
            return True

    async def clear(self, session_id: str) -> None:
        async with self.lock:
            self.rows.pop(session_id, None)
            for key in tuple(self.idempotency):
                if key[0] == session_id:
                    self.idempotency.pop(key, None)


class _CustomCheckpointStore:
    durable = True

    def __init__(self) -> None:
        self.rows: dict[str, str] = {}
        self.lock = asyncio.Lock()

    async def load(self, run_id: str) -> RunCheckpoint | None:
        async with self.lock:
            payload = self.rows.get(run_id)
        return None if payload is None else RunCheckpoint.from_json(payload)

    async def save(self, run_id: str, context: RunCheckpoint) -> None:
        if not isinstance(context, RunCheckpoint):
            raise TypeError("custom checkpoint fake expects RunCheckpoint")
        if context.run_id != run_id:
            raise ValueError("run_id does not match context.run_id")
        payload = context.to_json()
        async with self.lock:
            self.rows[run_id] = payload

    async def delete(self, run_id: str) -> None:
        async with self.lock:
            self.rows.pop(run_id, None)


class _CustomReceiptStore:
    durable = True

    def __init__(self) -> None:
        self.rows: dict[str, DelegationReceipt] = {}
        self.lock = asyncio.Lock()

    async def get(self, delegation_id: str) -> DelegationReceipt | None:
        async with self.lock:
            return copy.deepcopy(self.rows.get(delegation_id))

    async def claim(self, receipt: DelegationReceipt) -> ReceiptClaim:
        async with self.lock:
            current = self.rows.get(receipt.delegation_id)
            if current is not None:
                return ReceiptClaim(copy.deepcopy(current), False)
            self.rows[receipt.delegation_id] = copy.deepcopy(receipt)
            return ReceiptClaim(copy.deepcopy(receipt), True)

    async def compare_and_swap(
        self,
        delegation_id: str,
        expected_revision: int,
        receipt: DelegationReceipt,
    ) -> bool:
        async with self.lock:
            current = self.rows.get(delegation_id)
            if current is None or current.revision != expected_revision:
                return False
            if receipt.delegation_id != delegation_id:
                raise ValueError("replacement receipt changed identity")
            if receipt.revision != expected_revision + 1:
                raise ValueError("replacement receipt skipped a revision")
            self.rows[delegation_id] = copy.deepcopy(receipt)
            return True


class _CustomBudgetStateStore:
    durable = True

    def __init__(self) -> None:
        self.rows: dict[str, ExecutionGroupBudgetState] = {}
        self.lock = asyncio.Lock()

    async def load(
        self,
        execution_group_id: str,
    ) -> ExecutionGroupBudgetState | None:
        async with self.lock:
            return copy.deepcopy(self.rows.get(execution_group_id))

    async def create(self, state: ExecutionGroupBudgetState) -> bool:
        async with self.lock:
            if state.execution_group_id in self.rows:
                return False
            self.rows[state.execution_group_id] = copy.deepcopy(state)
            return True

    async def compare_and_swap(
        self,
        execution_group_id: str,
        expected_revision: int,
        state: ExecutionGroupBudgetState,
    ) -> bool:
        async with self.lock:
            current = self.rows.get(execution_group_id)
            if current is None or current.revision != expected_revision:
                return False
            if state.execution_group_id != execution_group_id:
                raise ValueError("replacement budget state changed identity")
            if state.revision != expected_revision + 1:
                raise ValueError("replacement budget state skipped a revision")
            self.rows[execution_group_id] = copy.deepcopy(state)
            return True


class _CustomContextMemoryStateStore:
    durable = True

    def __init__(self) -> None:
        self.rows: dict[str, ConversationSummarySnapshot] = {}
        self.lock = asyncio.Lock()

    async def load(
        self,
        key: MemoryStateKey,
    ) -> ConversationSummarySnapshot | None:
        async with self.lock:
            return copy.deepcopy(self.rows.get(key.to_storage_key()))

    async def save_if_version(
        self,
        expected_version: int,
        next_snapshot: ConversationSummarySnapshot,
    ) -> bool:
        if next_snapshot.version != expected_version + 1:
            raise ValueError("next version must increment once")
        key = next_snapshot.key.to_storage_key()
        async with self.lock:
            current = self.rows.get(key)
            current_version = 0 if current is None else current.version
            if current_version != expected_version:
                return False
            if (
                current is not None
                and next_snapshot.covered_through_sequence
                < current.covered_through_sequence
            ):
                raise ContextMemoryCursorRegressionError("cursor regressed")
            self.rows[key] = copy.deepcopy(next_snapshot)
            return True

    async def clear(self, key: MemoryStateKey) -> None:
        async with self.lock:
            self.rows.pop(key.to_storage_key(), None)


class _CustomAgentInvoker:
    async def invoke(
        self,
        endpoint: AgentEndpoint,
        request: Any,
        context: Any,
        budget: Any,
        budget_ledger: Any,
        *,
        child_run_id: str,
    ) -> DelegationOutcome:
        handler = endpoint.handler
        run = getattr(handler, "_run_delegated", None)
        if not callable(run):
            raise DelegationEndpointError("delegation_endpoint_protocol_error")
        outcome = await run(
            request,
            context=context,
            budget=budget,
            budget_ledger=budget_ledger,
            child_run_id=child_run_id,
        )
        if not isinstance(outcome, DelegationOutcome):
            raise DelegationEndpointError("delegation_endpoint_protocol_error")
        if outcome.child_run_id != child_run_id:
            raise DelegationEndpointError("delegation_child_run_id_mismatch")
        return outcome


def _openai_success_client() -> tuple[OpenAICompatibleClient, _ModelTransport]:
    transport = _ModelTransport(
        response={
            "id": "contract-response",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "contract-ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            },
        }
    )
    return (
        OpenAICompatibleClient(
            base_url="http://model.invalid/v1",
            model="contract-model",
            transport=transport,
        ),
        transport,
    )


@pytest.mark.parametrize("kind", ["builtin", "custom"])
def test_model_provider_request_and_result_conformance(kind: str) -> None:
    async def scenario() -> None:
        if kind == "builtin":
            client, transport = _openai_success_client()
        else:
            client = _CustomModelClient()
            transport = None

        await assert_model_provider_contract(client)

        if transport is not None:
            sent = transport.requests[0]["json"]
            assert sent["messages"][0]["role"] == "system"
            assert sent["messages"][1]["role"] == "user"
            assert sent["temperature"] == 0
            assert sent["seed"] == 17
        else:
            assert len(client.requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("kind", "failure", "category", "code", "retryable"),
    [
        (
            "builtin-protocol",
            None,
            "model_protocol",
            "model_protocol_error",
            False,
        ),
        (
            "custom-protocol",
            ModelProtocolError("malformed provider payload"),
            "model_protocol",
            "model_protocol_error",
            False,
        ),
        ("builtin-timeout", TimeoutError(), "timeout", "model_timeout", True),
        ("custom-timeout", TimeoutError(), "timeout", "model_timeout", True),
    ],
)
def test_model_provider_error_conformance(
    kind: str,
    failure: Exception | None,
    category: str,
    code: str,
    retryable: bool,
) -> None:
    async def scenario() -> None:
        if kind == "builtin-protocol":
            transport = _ModelTransport(
                response={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-malformed",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": "{not-json",
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            )
            client = OpenAICompatibleClient(
                base_url="http://model.invalid/v1",
                model="contract-model",
                transport=transport,
            )
        elif kind.startswith("builtin"):
            assert failure is not None
            client = OpenAICompatibleClient(
                base_url="http://model.invalid/v1",
                model="contract-model",
                transport=_ModelTransport(error=failure),
            )
        else:
            assert failure is not None
            client = _CustomModelClient(error=failure)
        await assert_model_provider_error_contract(
            client,
            category=category,
            code=code,
            retryable=retryable,
        )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "store",
    [InMemoryConversationStore, _CustomConversationStore],
)
def test_conversation_store_conformance(store: type[Any]) -> None:
    asyncio.run(assert_conversation_store_contract(store()))


@pytest.mark.parametrize(
    "store",
    [InMemoryCheckpointStore, _CustomCheckpointStore],
)
def test_checkpoint_store_conformance(store: type[Any]) -> None:
    asyncio.run(assert_checkpoint_store_contract(store()))


@pytest.mark.parametrize(
    "store",
    [InMemoryDelegationReceiptStore, _CustomReceiptStore],
)
def test_delegation_receipt_store_conformance(store: type[Any]) -> None:
    instance = store()
    assert isinstance(instance, DelegationReceiptStore)
    asyncio.run(assert_receipt_store_contract(instance))


@pytest.mark.parametrize(
    "store",
    [InMemoryBudgetStateStore, _CustomBudgetStateStore],
)
def test_budget_state_store_conformance(store: type[Any]) -> None:
    instance = store()
    assert isinstance(instance, BudgetStateStore)
    asyncio.run(assert_budget_state_store_contract(instance))


@pytest.mark.parametrize(
    "store",
    [InMemoryContextMemoryStateStore, _CustomContextMemoryStateStore],
)
def test_context_memory_state_store_conformance(store: type[Any]) -> None:
    asyncio.run(assert_context_memory_store_contract(store()))


@pytest.mark.parametrize("invoker", [LocalAgentInvoker, _CustomAgentInvoker])
def test_agent_invoker_conformance(invoker: type[Any]) -> None:
    asyncio.run(
        assert_agent_invoker_contract(
            invoker(),
            lambda handler: AgentEndpoint(handler=handler, approved=True),
        )
    )
