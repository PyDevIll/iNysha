import os
from pathlib import Path
from loguru import logger

from agent import Agent
from builtin_tools import register_all as register_builtin_tools
from scheduler import get_scheduler
from time import time
import re


# auto-download incoming files to data/downloads/
DOWNLOADS_DIR = Path(__file__).resolve().parent / "data" / "downloads"
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

request_queue = None  # will be set in main
agent = None

async def start_app():
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

