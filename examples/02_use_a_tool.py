"""Give an Agent one safe, typed Tool."""

import asyncio

from moduagent import Agent, ConsoleEventSink, VLLMClient, tool


ORDERS = {
    "ORD-1001": {
        "status": "shipped",
        "estimated_delivery": "2026-08-01",
    },
    "ORD-1002": {
        "status": "processing",
        "estimated_delivery": "2026-08-04",
    },
}


@tool
def lookup_order(order_id: str) -> dict[str, str]:
    """Look up shipping status for an order ID such as ORD-1001."""

    normalized_id = order_id.strip().upper()
    order = ORDERS.get(normalized_id)
    if order is None:
        return {"order_id": normalized_id, "status": "not_found"}
    return {"order_id": normalized_id, **order}


def build_agent(model, *, event_sink=None):
    return Agent.create(
        model=model,
        instructions=(
            "Help users check orders. Use lookup_order when an order ID is "
            "provided, and never invent an order status."
        ),
        tools=[lookup_order],
        event_sink=event_sink,
    )


async def main() -> None:
    async with VLLMClient.from_env(
        default_options={"temperature": 0, "max_tokens": 256},
    ) as model:
        agent = build_agent(model, event_sink=ConsoleEventSink())
        answer = await agent.ask("Where is order ORD-1001?")
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())
