"""Тесты per-chat батчинга входящих обновлений MAX.

Проверяем, что сообщения одного чата в пределах окна тишины уходят одним
``Request``, разные чаты не склеиваются, переполнение пачки сбрасывается
сразу, а сообщения, пришедшие во время обработки, образуют следующую пачку.

Тесты не отправляют реальных сообщений в MAX и не пишут в data/:
``runtime.queue.put``, скачивание и транскрипция подменяются.
Запуск: .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import config
from server.queue import Request
from server.sources import max as max_source


class _QueueSpy:
    """Подмена ``runtime.queue.put``: копит запросы, воркер не запускает."""

    def __init__(self) -> None:
        self.requests: list[Request] = []
        self.event = asyncio.Event()

    async def put(self, request: Request) -> None:
        self.requests.append(request)
        self.event.set()


class _StubHttpRequest:
    """Минимальная замена fastapi.Request для вызова webhook."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.headers: dict[str, str] = {}

    async def json(self) -> dict[str, Any]:
        return self._payload


class MaxBatchingTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._spy = _QueueSpy()
        self._patchers = [
            mock.patch.object(
                max_source.runtime.queue, "put", new=self._spy.put
            ),
            # По умолчанию таймер тишины не должен срабатывать сам по себе.
            mock.patch.object(config, "MAX_DEFER_SECONDS", 600.0),
            mock.patch.object(config, "MAX_BATCH_MESSAGES", 10),
            mock.patch.object(config, "MAX_BATCH_CHARS", 4000),
        ]
        for patcher in self._patchers:
            patcher.start()
        max_source._reset_state()
        self._tmp = tempfile.TemporaryDirectory()

    async def asyncTearDown(self) -> None:
        for task in list(max_source._background):
            task.cancel()
        await asyncio.sleep(0)
        max_source._reset_state()
        for patcher in reversed(self._patchers):
            patcher.stop()
        self._tmp.cleanup()

    # ── helpers ────────────────────────────────────────────────────────
    def _update(
        self,
        chat_id: str,
        text: str,
        mid: int,
        user_id: str = "1",
    ) -> dict[str, Any]:
        return {
            "update_type": "message_created",
            "chat_id": chat_id,
            "message": {
                "mid": mid,
                "body": {"text": text},
                "recipient": {"chat_id": chat_id},
                "sender": {"user_id": user_id, "name": "Тест"},
            },
        }

    async def _wait_for_requests(self, count: int, timeout: float = 2.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while len(self._spy.requests) < count:
            remaining = deadline - loop.time()
            if remaining <= 0:
                self.fail(
                    f"ожидали {count} запросов, получили {len(self._spy.requests)}"
                )
            try:
                await asyncio.wait_for(self._spy.event.wait(), remaining)
            except asyncio.TimeoutError:
                self.fail(
                    f"ожидали {count} запросов, получили {len(self._spy.requests)}"
                )
            self._spy.event.clear()

    # ── tests ──────────────────────────────────────────────────────────
    async def test_messages_within_debounce_window_become_one_request(
        self,
    ) -> None:
        """Два сообщения одного чата в окне тишины — один Request."""
        chat = "chat-a"
        with mock.patch.object(config, "MAX_DEFER_SECONDS", 0.05):
            await max_source._process_update(self._update(chat, "раз", 1))
            await max_source._process_update(self._update(chat, "два", 2))

        await self._wait_for_requests(1)
        await asyncio.sleep(0.1)

        self.assertEqual(len(self._spy.requests), 1)
        request = self._spy.requests[0]
        self.assertEqual(request.chat_id, chat)
        self.assertEqual(request.source, "max")
        self.assertIsInstance(request.reply, max_source._BatchReply)

        # Заголовок сохранён у каждого сообщения, между ними пустая строка.
        self.assertEqual(request.text.count("[MAX messenger | chat_id=chat-a"), 2)
        self.assertIn("trusted=false]\nраз", request.text)
        self.assertIn("\n\n[MAX messenger | chat_id=chat-a", request.text)
        self.assertIn("| user_id=1 | sender=Тест |", request.text)
        self.assertIn("| attachments=none |", request.text)
        self.assertLess(request.text.index("раз"), request.text.index("два"))

    async def test_two_chats_produce_two_independent_requests(self) -> None:
        """Разные чаты никогда не склеиваются в один запрос."""
        with mock.patch.object(config, "MAX_DEFER_SECONDS", 0.05):
            await max_source._process_update(
                self._update("chat-a", "a1", 1)
            )
            await max_source._process_update(
                self._update("chat-b", "b1", 2)
            )
            await max_source._process_update(
                self._update("chat-a", "a2", 3)
            )

        await self._wait_for_requests(2)

        self.assertEqual(len(self._spy.requests), 2)
        by_chat = {req.chat_id: req for req in self._spy.requests}
        self.assertEqual(set(by_chat), {"chat-a", "chat-b"})

        self.assertIn("a1", by_chat["chat-a"].text)
        self.assertIn("a2", by_chat["chat-a"].text)
        self.assertNotIn("b1", by_chat["chat-a"].text)

        self.assertIn("b1", by_chat["chat-b"].text)
        self.assertNotIn("a1", by_chat["chat-b"].text)
        self.assertNotIn("a2", by_chat["chat-b"].text)

    async def test_overflow_by_message_count_keeps_remainder(self) -> None:
        """10 сообщений сбрасывают пачку, остаток ждёт следующей."""
        chat = "chat-a"
        for i in range(1, 11):
            await max_source._process_update(
                self._update(chat, f"m{i}", i)
            )

        self.assertEqual(len(self._spy.requests), 1)
        first = self._spy.requests[0]
        self.assertEqual(first.text.count("[MAX messenger"), 10)
        self.assertIn("m10", first.text)
        self.assertNotIn("m11", first.text)

        # Ещё два сообщения копятся в новом буфере.
        await max_source._process_update(self._update(chat, "m11", 11))
        await max_source._process_update(self._update(chat, "m12", 12))
        self.assertEqual(len(max_source._batches[chat].messages), 2)
        self.assertEqual(len(self._spy.requests), 1)

        max_source._finish_batch(chat)
        await max_source._flush_batch(chat)

        self.assertEqual(len(self._spy.requests), 2)
        second = self._spy.requests[1]
        self.assertEqual(second.text.count("[MAX messenger"), 2)
        self.assertIn("m11", second.text)
        self.assertIn("m12", second.text)

    async def test_overflow_while_processing_keeps_batches_bounded(
        self,
    ) -> None:
        """Переполнение во время обработки не теряет сообщения и держит лимит."""
        chat = "chat-a"
        await max_source._process_update(self._update(chat, "старт", 1))
        await max_source._flush_batch(chat)
        self.assertEqual(len(self._spy.requests), 1)

        # 13 сообщений, пока идёт обработка: 10 уходят в backlog, 3 копятся.
        for i in range(2, 15):
            await max_source._process_update(
                self._update(chat, f"m{i}", i)
            )

        backlog = max_source._backlog.get(chat, [])
        self.assertEqual(len(backlog), 1)
        self.assertEqual(len(backlog[0].messages), 10)
        self.assertEqual(len(max_source._batches[chat].messages), 3)
        self.assertEqual(len(self._spy.requests), 1)

        max_source._finish_batch(chat)
        await max_source._flush_batch(chat)
        self.assertEqual(len(self._spy.requests), 2)
        self.assertEqual(self._spy.requests[1].text.count("[MAX messenger"), 10)

        max_source._finish_batch(chat)
        await max_source._flush_batch(chat)
        self.assertEqual(len(self._spy.requests), 3)
        self.assertEqual(self._spy.requests[2].text.count("[MAX messenger"), 3)
        self.assertIn("m14", self._spy.requests[2].text)

    async def test_overflow_by_characters_keeps_remainder(self) -> None:
        """Переполнение по символам сбрасывает пачку, остаток копится дальше."""
        chat = "chat-a"
        with mock.patch.object(config, "MAX_BATCH_CHARS", 300):
            await max_source._process_update(
                self._update(chat, "x" * 150, 1)
            )
            self.assertEqual(len(self._spy.requests), 0)

            await max_source._process_update(
                self._update(chat, "y" * 150, 2)
            )

        self.assertEqual(len(self._spy.requests), 1)
        first = self._spy.requests[0]
        self.assertIn("x" * 150, first.text)
        self.assertNotIn("y" * 150, first.text)

        self.assertEqual(len(max_source._batches[chat].messages), 1)
        max_source._finish_batch(chat)
        await max_source._flush_batch(chat)

        self.assertEqual(len(self._spy.requests), 2)
        second = self._spy.requests[1]
        self.assertIn("y" * 150, second.text)
        self.assertNotIn("x" * 150, second.text)

    async def test_message_during_processing_goes_to_next_batch(self) -> None:
        """Сообщение во время обработки не попадает в текущий запрос."""
        chat = "chat-a"
        await max_source._process_update(self._update(chat, "первое", 1))
        first = await max_source._flush_batch(chat)
        self.assertIsNotNone(first)
        self.assertEqual(len(self._spy.requests), 1)

        # Первый запрос ещё не обработан: чат занят.
        await max_source._process_update(self._update(chat, "второе", 2))
        self.assertIsNone(await max_source._flush_batch(chat))
        self.assertEqual(len(self._spy.requests), 1)
        self.assertIn("второе", max_source._batches[chat].messages[0])

        # Ответ отправлен — буфер уходит отдельной пачкой.
        max_source._finish_batch(chat)
        await max_source._flush_batch(chat)

        self.assertEqual(len(self._spy.requests), 2)
        second = self._spy.requests[1]
        self.assertIn("второе", second.text)
        self.assertNotIn("первое", second.text)

    async def test_duplicate_updates_are_ignored(self) -> None:
        """Повторный апдейт отбрасывается до попадания в буфер."""
        chat = "chat-a"
        update = self._update(chat, "раз", 1)

        self.assertTrue(await max_source.handle_update(update))
        self.assertFalse(await max_source.handle_update(update))

        self.assertEqual(len(max_source._batches[chat].messages), 1)
        self.assertEqual(len(self._spy.requests), 0)

    async def test_batch_is_untrusted_if_any_message_is_untrusted(self) -> None:
        """Пачка недоверенна, если хотя бы одно сообщение пришло не от доверенного.

        Доверие считается по каждому сообщению: в группе чужое сообщение не
        должно получить привилегии из-за соседа в пачке.
        """
        chat = "chat-a"
        await max_source._process_update(
            self._update(chat, "раз", 1, user_id="1")
        )
        with mock.patch.object(
            config,
            "is_trusted",
            side_effect=lambda chat_id, user_id: user_id == "42",
        ):
            await max_source._process_update(
                self._update(chat, "два", 2, user_id="42")
            )

        request = await max_source._flush_batch(chat)

        self.assertIsNotNone(request)
        self.assertFalse(request.trusted)

    async def test_batch_is_trusted_when_all_messages_trusted(self) -> None:
        """Пачка доверенна, только если доверенны ВСЕ её сообщения."""
        chat = "chat-a"
        with mock.patch.object(
            config, "is_trusted", side_effect=lambda chat_id, user_id: True
        ):
            await max_source._process_update(self._update(chat, "раз", 1))
            await max_source._process_update(self._update(chat, "два", 2))

        request = await max_source._flush_batch(chat)

        self.assertIsNotNone(request)
        self.assertTrue(request.trusted)

    async def test_voice_transcript_block_stays_intact(self) -> None:
        """Блок расшифровки вложения сохраняется внутри сообщения."""
        chat = "chat-a"
        audio_path = Path(self._tmp.name) / "voice.wav"
        audio_path.write_bytes(b"RIFF" + b"\x00" * 64)

        async def fake_download(token, url, *, max_bytes=None):
            return {"ok": True, "path": str(audio_path)}

        async def fake_transcribe(audio, *, filename=""):
            return "привет из голоса"

        update = self._update(chat, "", 1)
        update["message"]["body"]["attachments"] = [
            {
                "type": "audio",
                "payload": {"token": "t", "url": "https://example.invalid/a"},
            }
        ]

        with mock.patch.object(
            max_source, "_download_attachment", new=fake_download
        ), mock.patch.object(
            max_source, "transcribe_bytes", new=fake_transcribe
        ):
            await max_source._process_update(update)

        request = await max_source._flush_batch(chat)

        self.assertIsNotNone(request)
        self.assertIn("[MAX messenger | chat_id=chat-a", request.text)
        self.assertIn("| attachments=audio |", request.text)
        self.assertIn(
            "[Voice Message Transcript]\nпривет из голоса", request.text
        )

    async def test_webhook_returns_ok_before_processing(self) -> None:
        """Webhook отвечает {"ok": True} сразу, разбор идёт в фоне."""
        chat = "chat-a"
        # Дедупликация общая на модуль — чистим, чтобы апдейт был новым.
        max_source._seen.clear()

        with mock.patch.object(config, "MAX_WEBHOOK_SECRET", ""):
            result = await max_source.max_webhook(
                _StubHttpRequest(self._update(chat, "раз", 1))
            )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(len(self._spy.requests), 0)

        await asyncio.sleep(0)

        self.assertEqual(len(max_source._batches[chat].messages), 1)


if __name__ == "__main__":
    unittest.main()
