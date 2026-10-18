"""Process-wide runtime: one agent, one queue, one worker, one MAX bot."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from contextlib import suppress

from loguru import logger

from agent import Agent
from lib.max_bot import MAXBot
from server import config
from server.context import RequestContext, current_request
from server.queue import RequestQueue
from tool_registry import get_registry

HELPER_PROMPT = """## Values:
    - Meaning: Retain core semantic content. Highest priority.
    - Relevance: Extract only information relevant to the given task.
    - Fidelity: Accurately reproduce key elements.
    - Concise: Return only processed content. No questions."""


def configure_logging() -> None:
    config.ensure_dirs()
    logger.remove()
    logger.add(sys.stderr, level=config.LOG_LEVEL)
    logger.add(
        config.LOG_FILE,
        rotation="5 MB",
        retention="7 days",
        level="DEBUG",
        enqueue=True,
    )


async def worker_loop(rt: "Runtime") -> None:
    """Process requests strictly one at a time (shared context)."""
    while True:
        req = await rt.queue.get()
        ctx = RequestContext(
            request_id=req.id,
            source=req.source,
            chat_id=req.chat_id,
            client_id=req.client_id,
            trusted=req.trusted,
        )
        token = current_request.set(ctx)
        started = time.time()

        try:
            logger.info(
                f"[{req.id}] {req.source} request (queue left={rt.queue.qsize()}): "
                f"{req.text[:160]!r}"
            )

            reasoning_callback = None
            if req.send_reasoning and rt.bot is not None:
                async def reasoning_callback(thought: str) -> None:  # noqa: F811
                    await rt.bot.send_reasoning(thought)

            response = await rt.agent.run_with_crash_recovery(
                initial_user_request=req.text,
                reasoning_callback=reasoning_callback,
            )
            elapsed = time.time() - started
            logger.info(
                f"[{req.id}] done in {elapsed:.2f}s, "
                f"{len(response or '')} chars"
            )
            await req.reply.send_text(response or "")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(f"[{req.id}] request failed")
            with suppress(Exception):
                await req.reply.send_error(f"Ошибка обработки: {exc}")
        finally:
            current_request.reset(token)
            rt.queue.task_done()


class Runtime:
    def __init__(self) -> None:
        self.agent: Agent | None = None
        self.helper: Agent | None = None
        self.bot: MAXBot | None = None
        self.queue = RequestQueue()
        self.worker_task: asyncio.Task | None = None
        self.poll_task: asyncio.Task | None = None
        self.started = False
        self.started_at = 0.0
        self.max_username: str | None = None
        self.max_user_id: int | None = None

    async def startup(self) -> None:
        if self.started:
            return

        config.ensure_dirs()
        configure_logging()
        logger.info("iНюша cloud starting...")

        # get_registry() auto-registers builtins exactly once.  Do not call
        # builtin_tools.register_all() again here (it would duplicate logs and
        # work); only layer the client-facing overrides on top.
        registry = get_registry()
        from server.remote_tools import register_all as register_remote_tools

        register_remote_tools(registry)
        logger.info(f"Registered {len(registry.tool_names)} tools")

        self.helper = Agent(
            name="HELPER",
            system_prompt=HELPER_PROMPT,
            use_tools=False,
            save_history=False,
            api_key=os.getenv("DEEPSEEK_API_KEY_HELPER") or None,
        )

        memory_file = config.DATA_DIR / "last_compression.txt"
        self.agent = Agent(
            name="DELIRIA",
            base_prompts=[
                ("## **IDENTITY**\n", "system_prompts/core.md"),
                ("\n## **APPLICATION ARCHITECTURE**\n", "system_prompts/extended.md"),
                ("\n## **Tools Guidelines & Best Practices**\n", "system_prompts/tools_guidelines.md"),
            ],
            last_memory=(
                [("# **PREVIOUS MEMORY SUMMARY**\n", str(memory_file))]
                if memory_file.exists()
                else []
            ),
            use_tools=True,
            save_history=True,
        )
        self.agent.add_helper_agent(self.helper)

        self.bot = MAXBot(token=config.MAX_BOT_TOKEN)
        registry.set_bot(self.bot)

        try:
            info = await self.bot._api.get_my_info()
            if info.get("user_id"):
                self.max_username = info.get("username") or info.get("name")
                self.max_user_id = info.get("user_id")
                logger.info(f"MAX bot online: @{self.max_username} ({self.max_user_id})")
            else:
                logger.warning(f"MAX bot identity check failed: {info}")
        except Exception:
            logger.exception("MAX bot identity check failed")

        self.worker_task = asyncio.create_task(worker_loop(self), name="inysha-worker")

        if config.MAX_TRANSPORT == "poll":
            from server.sources.max_poll import poll_max_updates

            self.poll_task = asyncio.create_task(
                poll_max_updates(self), name="max-poller"
            )
            logger.info("MAX transport: long polling")
        else:
            logger.info("MAX transport: webhook")

        self.started = True
        self.started_at = time.time()
        logger.info("iНюша cloud ready")

    async def shutdown(self) -> None:
        if not self.started:
            return
        self.started = False

        if self.worker_task is not None:
            self.worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.worker_task
            self.worker_task = None

        if self.poll_task is not None:
            self.poll_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.poll_task
            self.poll_task = None

        if self.bot is not None:
            with suppress(Exception):
                await self.bot.close()

        for agent in (self.agent, self.helper):
            if agent is not None:
                with suppress(Exception):
                    await agent._client.close()

        logger.info("iНюша cloud stopped")


runtime = Runtime()
