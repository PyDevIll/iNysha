"""Тесты доставки MAX: реальная причина ошибки должна доходить до агента.

Раньше ``send_message_to_chat`` глотал ответ API и возвращал ``{}``, поэтому
инструмент сообщал модели нейтральное «send failed». Агент не мог отличить
сломанный транспорт (таймаут до platform-api2.max.ru) от неудачного текста и
бесконечно повторял отправку, блокируя очередь.

Тесты не ходят в сеть и не пишут в data/: ``_request``/``_get_bot_from_registry``
подменяются.
Запуск: .venv/bin/python -m unittest tests.test_max_send_errors -v
"""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from builtin_tools import max_tools
from lib.max_bot import MAXBotApi, delivery_error, sent_message


class _FakeApi:
    """Подмена ``MAXBotApi`` внутри бота: управляемый ответ /messages."""

    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[tuple] = []

    async def send_message_to_chat(self, chat_id, text, extra=None):
        self.calls.append((chat_id, text, extra))
        return self.response


class _FakeBot:
    def __init__(self, response: dict) -> None:
        self._api = _FakeApi(response)

    async def send_voice(self, chat_id, voice_path, caption=None):
        return {"ok": True, "result": {"message": {"body": {"mid": "1"}}}}


class MaxSendErrorsTest(unittest.TestCase):
    def test_sent_message_success(self) -> None:
        response = {"message": {"body": {"text": "ok"}}}
        self.assertEqual(sent_message(response), {"body": {"text": "ok"}})
        self.assertEqual(delivery_error(response), "send failed")

    def test_sent_message_failure(self) -> None:
        response = {"ok": False, "error": "timeout"}
        self.assertEqual(sent_message(response), {})
        self.assertEqual(delivery_error(response), "timeout")

    def test_api_client_propagates_transport_error(self) -> None:
        """``send_message_to_chat`` не должен терять причину сбоя."""
        api = MAXBotApi("dummy-token")

        async def fake_request(method, path, query=None, body=None):
            return {"ok": False, "error": "timeout"}

        async def run() -> dict:
            with mock.patch.object(api, "_request", fake_request):
                return await api.send_message_to_chat(123, "привет")

        result = asyncio.run(run())
        self.assertEqual(result.get("error"), "timeout")
        self.assertEqual(sent_message(result), {})

    def test_max_send_message_reports_reason_and_no_retry_hint(self) -> None:
        bot = _FakeBot({"ok": False, "error": "timeout"})

        async def run() -> dict:
            with mock.patch.object(max_tools, "_get_bot_from_registry", return_value=bot):
                return await max_tools.max_send_message(chat_id="123", text="привет")

        result = asyncio.run(run())
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "timeout")
        self.assertIs(result["retry_same_turn"], False)
        self.assertIn("Не повторяй вызов", result["hint"])
        self.assertEqual(len(bot._api.calls), 1, "инструмент не должен повторять отправку сам")

    def test_max_send_message_success_marks_chat(self) -> None:
        bot = _FakeBot({"message": {"body": {"text": "привет"}}})

        async def run() -> dict:
            with mock.patch.object(max_tools, "_get_bot_from_registry", return_value=bot):
                return await max_tools.max_send_message(chat_id="123", text="привет")

        result = asyncio.run(run())
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"], {"body": {"text": "привет"}})


if __name__ == "__main__":
    unittest.main()
