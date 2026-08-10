import os
from pathlib import Path
from loguru import logger

from agent import Agent
from builtin_tools import register_all as register_builtin_tools
from tool_registry import get_registry
import sys
import asyncio

# def global_exception_handler(exc_type, exc_value, exc_traceback):
#     logger.exception("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
#
# sys.excepthook = global_exception_handler

# auto-download incoming files to data/downloads/
DOWNLOADS_DIR = Path(__file__).resolve().parent / "data" / "downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).resolve().parent / "data"

request_queue = None
agent = None

def get_request_queue():
    return request_queue

def get_agent():
    return agent


# ---- Worker coroutine (processes requests sequentially) ----
async def worker() -> None:
    from components.max import process_max_message
    from components.cmd_line import process_command_prompt
    from components.tts_stt import process_voice_input
    from components.live import process_live_scene

    while True:
        # log queue content
        _log_msg = f"Worker queue count: {request_queue.qsize()}"
        for num, item in enumerate(list(request_queue._queue)):
            _log_msg += f"\n {num}: {item['type']}"
        logger.info(_log_msg)

        # process request_queue item
        req = await request_queue.get()
        try:
            if req["type"] == "max":
                await process_max_message(req["update"])
            elif req["type"] == "user":
                await process_command_prompt(req["prompt"])
            elif req["type"] == "tts":
                await process_voice_input(req["text"])
            elif req["type"] == "live":
                await process_live_scene(req["scene"])
            # elif req["type"] == "scheduled":
            #     await process_scheduled_task(req["task"])
            else:
                logger.warning(f"Unknown request type: {req.get('type')}")
        except Exception as e:
            logger.exception(f"Worker failed processing {req.get('type')}: {e}")
        finally:
            request_queue.task_done()


async def get_command():
    from components.tts_stt import tts_stt_toggle
    from components.live import live_toggle

    while True:
        user_request = await asyncio.to_thread(input, "User command: ")
        parsed = user_request.split(" ", 1)
        command = parsed[0]
        parameters = parsed[1] if len(parsed) > 1 else None

        print("Got user command:", command, ", parameters:", parameters)
        if command == "/v":     # voice
            _on_off = await tts_stt_toggle()
            print("[Command]: TTS-STT Dialogue mode switched", "on" if _on_off else "off")

        if command == "/p":     # prompt
            if parameters:
                await request_queue.put({"type": "user", "prompt": f"[Command prompt]: {parameters}"})

        if command == "/l":     # live
            _on_off = await tts_stt_toggle(False)
            print("[Command]: TTS-STT Dialogue mode switched", "on" if _on_off else "off")
            _on_off = await live_toggle()
            print("[Command]: Live mode switched", "on" if _on_off else "off")


# --------- APP ENTRY POINT -----------
async def start_app():
    from components.max import run_http_server, max_deferred_reply

    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")  # os.environ.get("LOG_LEVEL", "INFO"))
    logger.add(
        Path(__file__).resolve().parent / "data" / "agent.log",
        rotation="1 MB",
        retention="7 days",
        level="DEBUG",
    )

    logger.info("iНюша starting...")

    # Global queue for incoming requests (user messages + scheduled tasks)
    global request_queue
    request_queue = asyncio.Queue()


    # Initialize tool registry with builtin tools
    registry = get_registry()
    register_builtin_tools(registry)
    logger.info(f"Registered {len(registry.tool_names)} tools: {registry.tool_names}")


    # Create helper agent for compression
    helper_agent = Agent(
        name="HELPER",
        system_prompt="""## Values:
    - Meaning: Retain core semantic content. Highest priority.
    - Relevance: Extract only information relevant to the given task.
    - Fidelity: Accurately reproduce key elements.
    - Concise: Return only processed content. No questions.""",
        use_tools=False,
        save_history=False,
    )

    # Create main agent
    global agent
    agent = Agent(
        name="INYSHA",
        base_prompts=[
            ("## **IDENTITY**\n", "system_prompts/core.md"),
            ("\n## **APPLICATION ARCHITECTURE**\n", "system_prompts/extended.md"),
            ("\n## **Tools Guidelines & Best Practices**\n", "system_prompts/tools_guidelines.md"),
        ],
        last_memory=[
            ("# **PREVIOUS MEMORY SUMMARY**\n", "data/last_compression.txt"),
        ],
        use_tools=True,
        save_history=True,
    )
    agent.add_helper_agent(helper_agent)

    # Start all components concurrently
    logger.info("Starting worker, scheduler checker and http listening...")
    worker_task = asyncio.create_task(worker())
    # checker_task = asyncio.create_task(scheduler_checker())
    deferred_reply_task = asyncio.create_task(max_deferred_reply())
    http_task = asyncio.create_task(run_http_server())
    command_task = asyncio.create_task(get_command())


    logger.info("All components started. Awaiting tasks...")
    await asyncio.gather(worker_task, deferred_reply_task, http_task, command_task)
