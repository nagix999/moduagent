"""Return a validated Pydantic object from an Agent."""

import asyncio
from typing import Literal

from pydantic import BaseModel, Field

from moduagent import Agent, VLLMClient


class TicketTriage(BaseModel):
    priority: Literal["low", "medium", "high", "urgent"]
    category: Literal["billing", "account", "bug", "feature", "other"]
    summary: str = Field(description="A one-sentence description of the issue")
    next_action: str = Field(
        description="The first action the support team should take"
    )


def build_agent(model):
    return Agent.create(
        model=model,
        instructions=(
            "Triage support tickets. Mark an issue urgent only when it blocks "
            "critical work or creates an immediate security or safety risk."
        ),
        output=TicketTriage,
    )


async def main() -> None:
    async with VLLMClient.from_env(
        default_options={"temperature": 0, "max_tokens": 256},
    ) as model:
        agent = build_agent(model)
        triage = await agent.ask(
            "I was charged twice for invoice INV-204, but I can still use the service."
        )
        print(triage.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
