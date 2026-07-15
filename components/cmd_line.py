import asyncio
from .tts_stt import tts_stt_enable, tts_stt_disable


async def process_command_prompt(text: str) -> None:
    async def reasoning_callback(thought):
        print(" > ...", thought)

    response_text = await agent.run_with_crash_recovery(
        initial_user_request="[Command prompt]: " + text,
        reasoning_callback=reasoning_callback
    )
    print(" >", response_text)
