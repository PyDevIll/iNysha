import asyncio
from app import get_agent

async def process_command_prompt(text: str) -> None:
    async def reasoning_callback(thought):
        print(" > ...", thought)

    agent = get_agent()
    response_text = await agent.run_with_crash_recovery(
        initial_user_request="[Command prompt]: \n" + text,
        reasoning_callback=reasoning_callback
    )
    print(" >", response_text)
