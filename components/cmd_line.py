import asyncio
from .tts_stt import tts_stt_enable, tts_stt_disable

async def get_command():
    while True:
        user_request = await asyncio.to_thread(input, "User command: ")
        parsed = user_request.split(" ", 1)
        command = parsed[0]
        parameters = parsed[1] if len(parsed) > 1 else None

        print("Got user command:", command, ", parameters:", parameters)
        if command == "/v":     # voice
            if tts_stt_enabled:
                tts_stt_disable(tts_context)
            else:
                tts_context = tts_stt_enable()
            tts_stt_enabled = not tts_stt_enabled
        if command == "/p":     # prompt
            if parameters:
                await request_queue.put({"type": "user", "prompt": f"[Command prompt]: {parameters}"})
