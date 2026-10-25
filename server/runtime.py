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
from server.queue import Request, RequestQueue
from server.routing import MaxReply, NullReply
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


async def scheduler_loop(rt: "Runtime", tick: float = 15.0) -> None:
    """Autonomously fire due scheduled tasks into the queue.

    Without this loop ``schedule_task`` only stores a task and waits for the
    next incoming message; here the scheduler itself pushes a ``Request`` so
    the agent wakes up on time with no external trigger.
    """
    from lib.scheduler import get_scheduler

    while True:
        try:
            await asyncio.sleep(tick)
            sched = get_scheduler()
            for task in sched.get_due_tasks():
                chat_id = task.get("chat_id")
                reply = (
                    MaxReply(rt.bot, chat_id)
                    if (chat_id and rt.bot is not None)
                    else NullReply()
                )
                req = Request(
                    text=task["prompt"],
                    source="scheduled",
                    chat_id=chat_id,
                    trusted=True,
                    reply=reply,
                )
                sched.mark_done(task["id"])
                await rt.queue.put(req)
                logger.info(
                    f"[{req.id}] scheduled task {task['id']} fired "
                    f"(queue left={rt.queue.qsize()})"
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("scheduler loop iteration failed")


async def worker_loop(rt: "Runtime") -> None:
    """Process requests strictly one at a time (shared context)."""
    while True:
        req = await rt.queue.get()
        ctx = RequestContext(
            request_id=req.id,
            source=req.source,
            chat_id=req.chat_id,
            client_id=req.client_id,
            voice_session_id=req.voice_session_id,
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

            agent_input = req.text
            if req.tool_result is not None:
                voice_turn = req.source in ("voice", "tool_result") and bool(
                    req.client_id
                )
                if voice_turn:
                    instruction = (
                        "Скажи пользователю голосом 1-3 коротких предложения: что "
                        "нашлось. Без списков, без цифр через запятую, без чтения "
                        "сырого текста. Если по делу сказать нечего — ответь "
                        "пустой строкой, и ты промолчишь."
                    )
                else:
                    instruction = (
                        "Используй это как контекст и при необходимости скажи о "
                        "находке в одной-двух фразах. Не зачитывай сырой текст "
                        "целиком."
                    )
                agent_input = (
                    f"[ФОНОВЫЙ РЕЗУЛЬТАТ] Инструмент «{req.tool_name}» завершился, "
                    f"пока ты говорила с пользователем.\n"
                    f"Это данные, которые ты запрашивала. {instruction}\n\n"
                    f"{req.tool_result}"
                )
                logger.info(
                    f"[{req.id}] tool_result turn for '{req.tool_name}' "
                    f"({len(str(req.tool_result))} chars)"
                )

            response = await rt.agent.run_with_crash_recovery(
                initial_user_request=agent_input,
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
        self.scheduler_task: asyncio.Task | None = None
        self.started = False
        self.started_at = 0.0
        self.max_username: str | None = None
        self.max_user_id: int | None = None

    async def startup(self) -> None:
        if self.started:
            return

        config.ensure_dirs()
        configure_logging()
        logger.info(f"{config.INSTANCE_NAME} cloud starting...")

        # get_registry() auto-registers builtins exactly once.  Do not call
        # builtin_tools.register_all() again here (it would duplicate logs and
        # work); only install the client tool layer on top.
        registry = get_registry()
        from server.remote_tools import register_all as register_remote_tools

        register_remote_tools(registry)
        logger.info(
            f"Registered {len(registry.tool_names)} tools "
            f"({len(registry.client_tool_names)} client-side)"
        )

        self.helper = Agent(
            name=config.HELPER_NAME,
            system_prompt=HELPER_PROMPT,
            use_tools=False,
            save_history=False,
            api_key=config.DEEPSEEK_API_KEY_HELPER or None,
        )

        self.agent = Agent(
            name=config.AGENT_NAME,
            base_prompts=[
                ("## **IDENTITY**\n", str(config.CORE_PROMPT_PATH)),
                (
                    "\n## **APPLICATION ARCHITECTURE**\n",
                    str(config.EXTENDED_PROMPT_PATH),
                ),
                (
                    "\n## **Tools Guidelines & Best Practices**\n",
                    str(config.TOOLS_GUIDELINES_PROMPT_PATH),
                ),
            ],
            # last_compression.txt загружает ровно один раз сам ContextPool
            # (_seed_from_last_compression -> _compressed_dicts). Через
            # last_memory его передавать нельзя: construct_history() превращает
            # текст в system/base-промпт, который оседал в all_entries[0] и
            # навсегда сохранялся в emergency_save.json (дубль + лишний токен-
            # расход). Базовые промпты собираются отдельно в Agent каждый запрос.
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

        self.worker_task = asyncio.create_task(
            worker_loop(self), name=f"{config.INSTANCE_SLUG}-worker"
        )
        self.scheduler_task = asyncio.create_task(
            scheduler_loop(self), name=f"{config.INSTANCE_SLUG}-scheduler"
        )

        if config.MAX_TRANSPORT == "poll":
            from server.sources.max_poll import poll_max_updates

            self.poll_task = asyncio.create_task(
                poll_max_updates(self), name=f"{config.INSTANCE_SLUG}-max-poller"
            )
            logger.info("MAX transport: long polling")
        else:
            logger.info("MAX transport: webhook")

        self.started = True
        self.started_at = time.time()
        logger.info(f"{config.INSTANCE_NAME} cloud ready")

    async def shutdown(self) -> None:
        if not self.started:
            return
        self.started = False

        if self.worker_task is not None:
            self.worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.worker_task
            self.worker_task = None

        if self.scheduler_task is not None:
            self.scheduler_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.scheduler_task
            self.scheduler_task = None

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

        logger.info(f"{config.INSTANCE_NAME} cloud stopped")


runtime = Runtime()
