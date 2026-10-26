"""Фоновый экстрактор: только люди, только кандидаты, без сюрпризов при старте.

Проверяем то, что дороже всего ошибить:
  * история не переразбирается после первого запуска (курсор = последний эпизод);
  * голос и реплики агента не попадают в разбор (сарказм не становится фактом);
  * ответ модели парсится даже в markdown-заборе, а мусор не ломает батч;
  * секретоподобное отбрасывается до записи;
  * кандидаты идут в `pending_facts`, а не в `nodes`;
  * падение API не двигает курсор (повторим), кривой JSON — двигает (без вечного цикла).

Запуск: .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import memory_extract as mx
from server.memory_store import MemoryStore, close_store

GOOD_REPLY = json.dumps({
    "facts": [
        {"kind": "fact", "name": "лимит ElevenLabs",
         "text": "free-tier 10000 символов в месяц", "confidence": 0.8},
        {"kind": "person", "name": "оператор Андрей",
         "text": "просил не менять тон персонажа", "confidence": 0.9},
    ]
}, ensure_ascii=False)


class FakeCompletions:
    def __init__(self, reply: str = GOOD_REPLY, error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error

        class Message:
            content = self.reply

        class Choice:
            message = Message()

        class Usage:
            prompt_tokens = 100
            completion_tokens = 50

        class Response:
            choices = [Choice()]
            usage = Usage()

        return Response()


class FakeClient:
    def __init__(self, **kwargs):
        self.chat = type("Chat", (), {"completions": FakeCompletions(**kwargs)})()


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmp.name) / "memory.db")

    def tearDown(self) -> None:
        self.store.close()
        close_store()
        self._tmp.cleanup()

    def add(self, text: str, *, speaker: str = "human", kind: str = "message",
            chat_id: str = "1") -> int:
        return int(self.store.add_episode(
            chat_id=chat_id, speaker=speaker, kind=kind,
            ts=1_790_000_000 + (self.store.stats()["episodes"] or 0),
            text=text,
        ))

    def pending(self) -> list:
        return self.store.connection.execute(
            "SELECT * FROM pending_facts ORDER BY id"
        ).fetchall()


class CursorTest(Base):
    def test_first_run_starts_at_newest_episode(self) -> None:
        for index in range(5):
            self.add(f"длинная реплика номер {index} про сервер и лимиты")
        cursor = mx.ensure_cursor(self.store)
        self.assertEqual(cursor, 5)
        batch = mx.select_batch(self.store, cursor, 10)
        self.assertEqual(batch.episodes, [])

    def test_cursor_persists(self) -> None:
        mx.set_cursor(self.store, 42)
        self.store.close()
        reopened = MemoryStore(Path(self._tmp.name) / "memory.db")
        try:
            self.assertEqual(mx.ensure_cursor(reopened), 42)
        finally:
            reopened.close()

    def test_batch_is_chronological_and_capped(self) -> None:
        ids = [self.add(f"реплика {i} " + "текст " * 10) for i in range(5)]
        batch = mx.select_batch(self.store, 0, 3)
        self.assertEqual([int(e["id"]) for e in batch.episodes], ids[:3])
        self.assertEqual(batch.last_id, ids[2])


class SelectionTest(Base):
    def test_only_human_text_messages(self) -> None:
        self.add("Человек сказал важный факт про сервер prolight и его диски",
                 speaker="human", kind="message")
        self.add("Агент ответил шуткой про сервер", speaker="agent")
        self.add("Голосовая реплика человека про сервер", speaker="human",
                 kind="voice")
        batch = mx.select_batch(self.store, 0, 10)
        self.assertEqual(len(batch.episodes), 1)
        self.assertEqual(batch.episodes[0]["speaker"], "human")

    def test_short_replies_are_skipped(self) -> None:
        self.add("ок")
        self.add("длинная содержательная реплика про настройку сервера и память")
        batch = mx.select_batch(self.store, 0, 10, min_chars=40)
        self.assertEqual(len(batch.episodes), 1)

    def test_batch_never_mixes_chats(self) -> None:
        for index in range(3):
            self.add(f"длинная реплика из первого чата {index} " + "текст " * 4,
                     chat_id="111")
        self.add("длинная реплика из второго чата " + "текст " * 4, chat_id="222")
        batch = mx.select_batch(self.store, 0, 10)
        self.assertEqual({str(e["chat_id"]) for e in batch.episodes}, {"111"})

    def test_batch_messages_include_dates(self) -> None:
        self.add("факт про лимит и подписку " + "текст " * 5)
        batch = mx.select_batch(self.store, 0, 10)
        messages = mx.build_messages(batch)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Реплики человека", messages[1]["content"])
        self.assertRegex(messages[1]["content"], r"\[\d\d\.\d\d\.\d{4}\]")


class ParseTest(unittest.TestCase):
    def test_plain_json(self) -> None:
        facts = mx.parse_facts(GOOD_REPLY)
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0]["kind"], "fact")

    def test_markdown_fence(self) -> None:
        facts = mx.parse_facts(f"```json\n{GOOD_REPLY}\n```")
        self.assertEqual(len(facts), 2)

    def test_prose_around_json(self) -> None:
        facts = mx.parse_facts(f"Вот результат:\n{GOOD_REPLY}\nГотово.")
        self.assertEqual(len(facts), 2)

    def test_unknown_kind_falls_back_to_fact(self) -> None:
        facts = mx.parse_facts(json.dumps(
            {"facts": [{"kind": "weird", "name": "x", "text": "y"}]}
        ))
        self.assertEqual(facts[0]["kind"], "fact")

    def test_confidence_is_clamped(self) -> None:
        facts = mx.parse_facts(json.dumps(
            {"facts": [{"kind": "fact", "name": "x", "text": "y",
                        "confidence": 7}]}
        ))
        self.assertEqual(facts[0]["confidence"], 1.0)
        facts = mx.parse_facts(json.dumps(
            {"facts": [{"kind": "fact", "name": "x", "text": "y",
                        "confidence": "bad"}]}
        ))
        self.assertEqual(facts[0]["confidence"], 0.5)

    def test_empty_and_broken_replies(self) -> None:
        self.assertEqual(mx.parse_facts('{"facts": []}'), [])
        with self.assertRaises(ValueError):
            mx.parse_facts("совсем не json")
        with self.assertRaises(ValueError):
            mx.parse_facts('{"other": 1}')

    def test_secret_like_candidate_is_dropped(self) -> None:
        facts = mx.parse_facts(json.dumps({"facts": [
            {"kind": "fact", "name": "ключ", "text": "sk-abcdefghijklmnopqrstuvwxyz"},
            {"kind": "fact", "name": "ok", "text": "ключи лежат в data/downloads"},
        ]}))
        self.assertEqual([fact["name"] for fact in facts], ["ok"])

    def test_missing_name_or_text_is_dropped(self) -> None:
        facts = mx.parse_facts(json.dumps({"facts": [
            {"kind": "fact", "name": "", "text": "текст"},
            {"kind": "fact", "name": "имя", "text": ""},
        ]}))
        self.assertEqual(facts, [])

    def test_batch_of_facts_is_capped(self) -> None:
        many = {"facts": [
            {"kind": "fact", "name": f"f{i}", "text": "t"} for i in range(50)
        ]}
        self.assertEqual(len(mx.parse_facts(json.dumps(many))),
                         mx.MAX_FACTS_PER_BATCH)


class TickTest(Base):
    def setUp(self) -> None:
        super().setUp()
        mx.set_cursor(self.store, 0)

    def add_utterance(self, text: str = "Андрей просил не трогать его ключи, они в data/downloads") -> int:
        return self.add(text)

    def test_facts_go_to_pending_not_to_nodes(self) -> None:
        self.add_utterance()
        client = FakeClient()
        result = asyncio.run(mx.run_tick(self.store, client, batch_size=5))
        self.assertEqual(result.facts, 2)
        self.assertEqual(result.inserted, 2)
        self.assertEqual(self.store.stats()["nodes"], 0)
        self.assertEqual(len(self.pending()), 2)
        payload = json.loads(self.pending()[0]["payload"])
        self.assertIn("name", payload)
        self.assertEqual(self.pending()[0]["status"], "pending")

    def test_cursor_advances_after_success(self) -> None:
        self.add_utterance()
        asyncio.run(mx.run_tick(self.store, FakeClient(), batch_size=5))
        self.assertGreater(mx.get_cursor(self.store), 0)
        self.assertEqual(mx.select_batch(self.store, mx.get_cursor(self.store), 5).episodes, [])

    def test_zero_facts_still_advances(self) -> None:
        self.add_utterance()
        client = FakeClient(reply='{"facts": []}')
        asyncio.run(mx.run_tick(self.store, client, batch_size=5))
        self.assertGreater(mx.get_cursor(self.store), 0)
        self.assertEqual(self.pending(), [])

    def test_api_error_keeps_cursor_for_retry(self) -> None:
        self.add_utterance()
        client = FakeClient(error=RuntimeError("502 bad gateway"))
        result = asyncio.run(mx.run_tick(self.store, client, batch_size=5))
        self.assertIn("502", result.error)
        self.assertEqual(mx.get_cursor(self.store), 0)
        self.assertEqual(self.pending(), [])
        row = self.store.connection.execute(
            "SELECT value FROM meta WHERE key=?", (mx.LAST_ERROR_KEY,)
        ).fetchone()
        self.assertIn("502", row["value"])

    def test_broken_json_does_not_loop_forever(self) -> None:
        self.add_utterance()
        client = FakeClient(reply="я не умею в json")
        result = asyncio.run(mx.run_tick(self.store, client, batch_size=5))
        self.assertIn("parse:", result.error)
        # Курсор сдвинут: ядовитый батч не будет переразбираться вечно.
        self.assertGreater(mx.get_cursor(self.store), 0)

    def test_repeated_identical_candidate_is_ignored(self) -> None:
        self.add_utterance()
        client = FakeClient()
        asyncio.run(mx.run_tick(self.store, client, batch_size=5))
        # Повторная вставка того же батча вручную.
        batch = mx.select_batch(self.store, 0, 5)
        inserted = mx.store_pending(self.store, batch, mx.parse_facts(GOOD_REPLY))
        self.assertEqual(inserted, 0)
        self.assertEqual(len(self.pending()), 2)

    def test_known_fact_is_not_offered_again(self) -> None:
        self.add_utterance()
        self.store.upsert_node(
            kind="fact", name="лимит ElevenLabs",
            name_norm="лимит elevenlabs", node_text="уже знаю",
        )
        asyncio.run(mx.run_tick(self.store, FakeClient(), batch_size=5))
        names = [json.loads(row["payload"])["name"] for row in self.pending()]
        self.assertNotIn("лимит ElevenLabs", names)
        self.assertIn("оператор Андрей", names)

    def test_no_new_utterances_is_a_noop(self) -> None:
        client = FakeClient()
        result = asyncio.run(mx.run_tick(self.store, client, batch_size=5))
        self.assertEqual(result.skipped_reason, "no new utterances")
        self.assertEqual(client.chat.completions.calls, [])

    def test_prompt_comes_from_human_only(self) -> None:
        self.add("Человек говорит: сервер prolight занят на 82%", speaker="human")
        self.add("А я шучу про сервер", speaker="agent")
        client = FakeClient()
        asyncio.run(mx.run_tick(self.store, client, batch_size=5))
        call = client.chat.completions.calls[0]
        joined = "\n".join(m["content"] for m in call["messages"])
        self.assertIn("prolight", joined)
        self.assertNotIn("шучу", joined)


class LoopTest(Base):
    def test_loop_runs_a_tick_and_stops_on_cancel(self) -> None:
        mx.set_cursor(self.store, 0)
        self.add("Человек подтвердил: ключи лежат в data/downloads и в репозиторий не идут")
        client = FakeClient()

        async def scenario() -> None:
            task = asyncio.create_task(
                mx.extractor_loop(self.store, 0.01, batch_size=5,
                                  client_factory=lambda: client)
            )
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(scenario())
        self.assertEqual(len(self.pending()), 2)

    def test_client_factory_failure_does_not_kill_loop(self) -> None:
        async def scenario() -> None:
            def boom():
                raise RuntimeError("no api key")

            task = asyncio.create_task(
                mx.extractor_loop(self.store, 0.01, batch_size=5,
                                  client_factory=boom)
            )
            await asyncio.sleep(0.05)
            self.assertFalse(task.done(), "цикл не должен умирать")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
