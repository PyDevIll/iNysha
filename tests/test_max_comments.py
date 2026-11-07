"""Тесты ответов в комментариях канала MAX.

Комментарий приходит как ``comment_created`` с ``recipient.post_id``. Ответ
обязан уйти в ``POST /messages/{post_id}/comments``; ``POST /messages`` создал
бы новый пост в канале. Тесты не ходят в сеть и не пишут в data/.
Запуск: .venv/bin/python -m unittest tests.test_max_comments -v
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.max_bot import MAXBot, MAXBotApi, sent_message
from server import config
from server.queue import Request
from server.routing import MaxReply
from server.sources import max as max_source


class _QueueSpy:
    """Подмена ``runtime.queue.put``: копит запросы, воркер не запускает."""

    def __init__(self) -> None:
        self.requests: list[Request] = []

    async def put(self, request: Request) -> None:
        self.requests.append(request)


def _comment_update(
    chat_id: str,
    post_id: str,
    text: str,
    comment_mid: str,
    user_id: str = "42",
) -> dict[str, Any]:
    return {
        "update_type": "comment_created",
        "message": {
            "body": {"mid": comment_mid, "text": text},
            "recipient": {
                "chat_id": chat_id,
                "chat_type": "channel",
                "post_id": post_id,
            },
            "sender": {"user_id": user_id, "name": "Читатель"},
        },
    }


class _FakeBot:
    """Бот, который запоминает, куда ушёл ответ."""

    def __init__(self) -> None:
        self.comments: list[tuple[str, str]] = []
        self.messages: list[tuple[int, str]] = []

    async def send_comment(self, post_id, text):
        self.comments.append((str(post_id), text))
        return {"ok": True, "result": {"body": {"mid": "comment-out"}}}

    async def send_message(self, chat_id, text):
        self.messages.append((chat_id, text))
        return {"ok": True, "result": {"body": {"mid": "message-out"}}}


class MaxCommentRoutingTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._spy = _QueueSpy()
        self._patchers = [
            mock.patch.object(
                max_source.runtime.queue, "put", new=self._spy.put
            ),
            mock.patch.object(config, "MAX_DEFER_SECONDS", 600.0),
            mock.patch.object(config, "MAX_BATCH_MESSAGES", 10),
            mock.patch.object(config, "MAX_BATCH_CHARS", 4000),
        ]
        for patcher in self._patchers:
            patcher.start()
        max_source._reset_state()

    async def asyncTearDown(self) -> None:
        for task in list(max_source._background):
            task.cancel()
        await asyncio.sleep(0)
        max_source._reset_state()
        for patcher in reversed(self._patchers):
            patcher.stop()

    async def test_extract_reads_post_id_from_comment(self) -> None:
        data = max_source._extract(
            _comment_update("chan-1", "mid.post-1", "привет", "mid.c-1")
        )
        self.assertEqual(data["chat_id"], "chan-1")
        self.assertEqual(data["post_id"], "mid.post-1")
        self.assertEqual(data["chat_type"], "channel")
        self.assertEqual(data["message_id"], "mid.c-1")

    async def test_comment_reply_goes_to_comments_endpoint(self) -> None:
        chat, post = "chan-1", "mid.post-1"
        await max_source._process_update(
            _comment_update(chat, post, "вопрос", "mid.c-1")
        )

        key = max_source._thread_key(chat, post)
        request = await max_source._flush_batch(key)
        self.assertIsNotNone(request)
        self.assertEqual(request.chat_id, chat)
        self.assertIn(f"post_id={post}", request.text)
        self.assertIn("channel_comment=true", request.text)
        self.assertIn("вопрос", request.text)

        bot = _FakeBot()
        request.reply._inner.bot = bot
        await request.reply.send_text("ответ")

        self.assertEqual(bot.comments, [(post, "ответ")])
        self.assertEqual(bot.messages, [], "в канал не должен уйти новый пост")

    async def test_two_posts_are_not_batched_together(self) -> None:
        chat = "chan-1"
        await max_source._process_update(
            _comment_update(chat, "mid.post-1", "под первым", "mid.c-1")
        )
        await max_source._process_update(
            _comment_update(chat, "mid.post-2", "под вторым", "mid.c-2")
        )

        self.assertEqual(len(self._spy.requests), 0)
        first = await max_source._flush_batch(
            max_source._thread_key(chat, "mid.post-1")
        )
        second = await max_source._flush_batch(
            max_source._thread_key(chat, "mid.post-2")
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIn("под первым", first.text)
        self.assertNotIn("под вторым", first.text)
        self.assertIn("под вторым", second.text)
        self.assertNotIn("под первым", second.text)
        self.assertEqual(first.reply._inner.post_id, "mid.post-1")
        self.assertEqual(second.reply._inner.post_id, "mid.post-2")

    async def test_comments_on_same_post_stay_one_batch(self) -> None:
        chat, post = "chan-1", "mid.post-1"
        await max_source._process_update(
            _comment_update(chat, post, "раз", "mid.c-1")
        )
        await max_source._process_update(
            _comment_update(chat, post, "два", "mid.c-2")
        )

        key = max_source._thread_key(chat, post)
        self.assertEqual(len(max_source._batches[key].messages), 2)
        request = await max_source._flush_batch(key)
        self.assertIsNotNone(request)
        self.assertIn("раз", request.text)
        self.assertIn("два", request.text)


class MaxReplyTest(unittest.IsolatedAsyncioTestCase):
    async def test_plain_reply_still_uses_send_message(self) -> None:
        bot = _FakeBot()
        await MaxReply(bot, "123").send_text("привет")
        self.assertEqual(bot.messages, [(123, "привет")])
        self.assertEqual(bot.comments, [])

    async def test_reply_with_post_id_uses_comment(self) -> None:
        bot = _FakeBot()
        await MaxReply(bot, "123", post_id="mid.post").send_text("привет")
        self.assertEqual(bot.comments, [("mid.post", "привет")])
        self.assertEqual(bot.messages, [])


class MaxCommentToolTest(unittest.IsolatedAsyncioTestCase):
    class _Api:
        def __init__(self) -> None:
            self.comments: list[tuple] = []
            self.messages: list[tuple] = []

        async def send_comment_to_post(self, post_id, text, extra=None):
            self.comments.append((str(post_id), text, extra))
            return {"message": {"body": {"mid": "comment-out"}}}

        async def send_message_to_chat(self, chat_id, text, extra=None):
            self.messages.append((chat_id, text, extra))
            return {"message": {"body": {"mid": "message-out"}}}

    class _Bot:
        def __init__(self) -> None:
            self._api = MaxCommentToolTest._Api()

    def setUp(self) -> None:
        from builtin_tools import max_tools

        self._max_tools = max_tools
        self._bot = self._Bot()

    async def _call(self, ctx, tool, **kwargs):
        from server.context import current_request

        token = current_request.set(ctx)
        try:
            with mock.patch.object(
                self._max_tools,
                "_get_bot_from_registry",
                return_value=self._bot,
            ):
                return await tool(**kwargs)
        finally:
            current_request.reset(token)

    async def test_tool_is_registered(self) -> None:
        names = {item[0] for item in self._max_tools.TOOL_DEFINITIONS}
        self.assertIn("max_send_comment_reply", names)

    async def test_send_comment_tool_posts_to_comments(self) -> None:
        from server.context import RequestContext

        ctx = RequestContext(
            request_id="r1",
            source="max",
            chat_id="chan-1",
            post_id="mid.post-1",
        )
        result = await self._call(
            ctx,
            self._max_tools.max_send_comment_reply,
            post_id="mid.post-1",
            text="ответ",
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["delivered_as"], "comment")
        self.assertEqual(
            self._bot._api.comments,
            [("mid.post-1", "ответ", {"format": "markdown"})],
        )
        self.assertEqual(self._bot._api.messages, [])
        self.assertIn("chan-1", ctx.sent_chats)

    async def test_comment_under_other_post_does_not_suppress_final_reply(
        self,
    ) -> None:
        from server.context import RequestContext

        ctx = RequestContext(
            request_id="r2",
            source="max",
            chat_id="chan-1",
            post_id="mid.post-1",
        )
        result = await self._call(
            ctx,
            self._max_tools.max_send_comment_reply,
            post_id="mid.post-2",
            text="ответ",
        )

        self.assertTrue(result["ok"])
        self.assertEqual(self._bot._api.comments[0][0], "mid.post-2")
        self.assertEqual(ctx.sent_chats, set())

    async def test_max_send_message_redirects_into_current_comment_thread(
        self,
    ) -> None:
        from server.context import RequestContext

        ctx = RequestContext(
            request_id="r3",
            source="max",
            chat_id="chan-1",
            post_id="mid.post-1",
        )
        result = await self._call(
            ctx,
            self._max_tools.max_send_message,
            chat_id="chan-1",
            text="ответ",
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["delivered_as"], "comment")
        self.assertEqual(
            self._bot._api.comments,
            [("mid.post-1", "ответ", {"format": "markdown"})],
        )
        self.assertEqual(self._bot._api.messages, [])

    async def test_max_send_message_in_plain_chat_still_posts(self) -> None:
        from server.context import RequestContext

        ctx = RequestContext(request_id="r4", source="max", chat_id="123")
        result = await self._call(
            ctx,
            self._max_tools.max_send_message,
            chat_id="123",
            text="привет",
        )

        self.assertTrue(result["ok"])
        self.assertEqual(self._bot._api.comments, [])
        self.assertEqual(self._bot._api.messages, [(123, "привет", {"format": "markdown"})])


class CommentApiTest(unittest.IsolatedAsyncioTestCase):
    async def test_send_comment_to_post_drops_attachments(self) -> None:
        api = MAXBotApi("dummy-token")
        captured: dict[str, Any] = {}

        async def fake_request(method, path, query=None, body=None):
            captured.update(method=method, path=path, query=query, body=body)
            return {"message": {"body": {"mid": "comment-out"}}}

        with mock.patch.object(api, "_request", fake_request):
            result = await api.send_comment_to_post(
                "mid.post-1",
                "ответ",
                {
                    "format": "markdown",
                    "attachments": [{"type": "file", "payload": {"token": "x"}}],
                },
            )

        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["path"], "/messages/mid.post-1/comments")
        self.assertEqual(
            captured["body"], {"text": "ответ", "format": "markdown"}
        )
        self.assertEqual(sent_message(result), {"body": {"mid": "comment-out"}})

    async def test_bot_send_comment_truncates_and_returns_ok(self) -> None:
        bot = MAXBot(token="dummy-token")
        api = mock.Mock()
        api.send_comment_to_post = mock.AsyncMock(
            return_value={"message": {"body": {"mid": "comment-out"}}}
        )
        bot._api = api

        result = await bot.send_comment("mid.post-1", "я" * 5000)

        self.assertTrue(result["ok"])
        _post_id, sent_text, extra = api.send_comment_to_post.call_args.args
        self.assertLessEqual(len(sent_text), 4000 + len("\n\n... (truncated)"))
        self.assertEqual(extra, {"format": "markdown"})


if __name__ == "__main__":
    unittest.main()
