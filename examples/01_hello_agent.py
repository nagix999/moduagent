"""The smallest useful ModuAgent program."""

import asyncio

from moduagent import Agent, ConsoleEventSink, VLLMClient


def build_agent(model, *, event_sink=None):
    return Agent.create(
        model=model,
        instructions="You are a helpful assistant. Answer in two sentences or fewer.",
        event_sink=event_sink,
    )


async def main() -> None:
    async with VLLMClient.from_env(
        default_options={"temperature": 0, "max_tokens": 160},
    ) as model:
        agent = build_agent(model, event_sink=ConsoleEventSink())
        answer = await agent.ask("What can an AI agent do for a small team?")
        print(answer)


if __name__ == "__main__":
    asyncio.run(main())
