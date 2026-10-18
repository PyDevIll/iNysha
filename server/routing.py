"""Reply targets for the different request sources."""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

from loguru import logger

from server.context import get_request_context


class ReplyTarget(Protocol):
    async def send_text(self, text: str) -> None: ...

    async def send_error(self, text: str) -> None: ...


class NullReply:
    """Used for fire-and-forget requests (e.g. scheduled tasks)."""

    async def send_text(self, text: str) -> None:
        logger.info(f"[NullReply] {text[:200]}")

    async def send_error(self, text: str) -> None:
        logger.error(f"[NullReply] {text}")


class FutureReply:
    """Resolves an ``asyncio.Future`` — used by ``POST /api/prompt``."""

    def __init__(self, future: asyncio.Future[str]) -> None:
        self.future = future

    async def send_text(self, text: str) -> None:
        if not self.future.done():
            self.future.set_result(text)

    async def send_error(self, text: str) -> None:
        if not self.future.done():
            self.future.set_exception(RuntimeError(text))


class MaxReply:
    """Sends the final answer to the originating MAX chat."""

    def __init__(self, bot: Any, chat_id: str | int) -> None:
        self.bot = bot
        self.chat_id = str(chat_id)

    async def send_text(self, text: str) -> None:
        if not text:
            return
        ctx = get_request_context()
        if ctx is not None and self.chat_id in ctx.sent_chats:
            logger.info(
                f"MAX chat {self.chat_id} already answered by a tool; skipping final send"
            )
            return
        await self.bot.send_message(int(self.chat_id), text)

    async def send_error(self, text: str) -> None:
        await self.send_text(f"⚠️ {text}")
