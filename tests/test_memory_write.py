"""Явная запись в память: правила, отказ от секретов, soft delete.

Запись — единственная операция, которая меняет поведение навсегда (факт
возвращается в промпт каждый ход). Поэтому проверяем не только «сохранилось»,
но и все отказы: тип, пустое имя, слишком длинный текст, секреты, забывание.

Запуск: .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.context import RequestContext, current_request
from server.memory_search import format_all, search_nodes
from server.memory_store import MemoryStore, close_store
from server.memory_write import (
    MemoryWriteError,
    forget,
    looks_like_secret,
    normalize_name,
    remember,
)


class Base(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmp.name) / "memory.db")

    def tearDown(self) -> None:
        self.store.close()
        close_store()
        current_request.set(None)
        self._tmp.cleanup()

    def remember(self, **kwargs):
        defaults = dict(kind="fact", name="лимит ElevenLabs",
                        text="free-tier 10000 символов в месяц")
        defaults.update(kwargs)
        return remember(self.store, **defaults)

    def node(self, node_id: int):
        return self.store.get_node(node_id)


class RememberTest(Base):
    def test_creates_node_with_agent_origin(self) -> None:
        result = self.remember()
        row = self.node(result.node_id)
        self.assertTrue(result.created)
        self.assertEqual(row["kind"], "fact")
        self.assertEqual(row["origin"], "agent")
        self.assertAlmostEqual(row["confidence"], 0.6)
        self.assertEqual(row["status"], "active")
        self.assertEqual(row["scope"], "shared")

    def test_trusted_operator_writes_stronger_fact(self) -> None:
        result = self.remember(trusted=True)
        row = self.node(result.node_id)
        self.assertEqual(row["origin"], "operator")
        self.assertAlmostEqual(row["confidence"], 0.9)

    def test_second_remember_updates_instead_of_duplicating(self) -> None:
        first = self.remember()
        second = self.remember(text="обновлено: лимит 10000, сброс 07.11")
        self.assertFalse(second.created)
        self.assertEqual(first.node_id, second.node_id)
        self.assertEqual(self.store.stats()["nodes"], 1)
        self.assertIn("обновлено", self.node(first.node_id)["node_text"])

    def test_confidence_is_never_downgraded(self) -> None:
        self.remember(trusted=True)
        self.remember(trusted=False)
        self.assertAlmostEqual(self.node(1)["confidence"], 0.9)

    def test_retired_node_is_revived(self) -> None:
        result = self.remember()
        forget(self.store, node_id=result.node_id)
        self.assertEqual(self.node(result.node_id)["status"], "retired")
        again = self.remember(text="снова актуально")
        self.assertEqual(again.node_id, result.node_id)
        self.assertEqual(self.node(result.node_id)["status"], "active")

    def test_scope_chat_uses_request_chat(self) -> None:
        current_request.set(RequestContext(request_id="r", source="max", chat_id="42"))
        result = self.remember(scope="chat")
        self.assertEqual(result.scope, "chat:42")

    def test_scope_chat_without_request_falls_back_to_shared(self) -> None:
        self.assertEqual(self.remember(scope="chat").scope, "shared")

    def test_same_name_in_different_scopes_are_different_facts(self) -> None:
        shared = self.remember()
        scoped = self.remember(scope="chat", chat_id="42")
        self.assertNotEqual(shared.node_id, scoped.node_id)

    def test_name_and_text_are_normalised(self) -> None:
        result = self.remember(name="  лимит    ElevenLabs ", text="  факт\n\nтекст ")
        self.assertEqual(result.name, "лимит ElevenLabs")
        self.assertEqual(self.node(result.node_id)["node_text"], "факт текст")

    def test_node_is_searchable_by_inflected_form(self) -> None:
        self.remember(name="Сервер prolight", text="торговый робот и VPN")
        hits = search_nodes(self.store, "сервером")
        self.assertEqual(len(hits), 1)
        self.assertIn("торговый робот", hits[0].text)

    def test_node_name_is_indexed_too(self) -> None:
        self.remember(name="аикартинки")
        self.assertEqual(len(search_nodes(self.store, "аикартинок")), 1)


class RejectionTest(Base):
    def test_unknown_kind_is_rejected(self) -> None:
        with self.assertRaises(MemoryWriteError) as ctx:
            self.remember(kind="thought")
        self.assertIn("kind", str(ctx.exception))

    def test_empty_name_is_rejected(self) -> None:
        with self.assertRaises(MemoryWriteError):
            self.remember(name="   ")

    def test_empty_text_is_rejected(self) -> None:
        with self.assertRaises(MemoryWriteError):
            self.remember(text="  ")

    def test_too_long_text_is_rejected(self) -> None:
        with self.assertRaises(MemoryWriteError) as ctx:
            self.remember(text="я" * 2500)
        self.assertIn("lessons.md", str(ctx.exception))

    def test_bad_scope_is_rejected(self) -> None:
        with self.assertRaises(MemoryWriteError):
            self.remember(scope="global")


class SecretTest(Base):
    CASES = (
        "ключ: sk-abcdefghijklmnopqrstuvwxyz123",
        "github token ghp_abcdefghijklmnopqrstuvwxyz012345",
        "AIzaSyA1234567890abcdefghijklmnopqrstu",
        "xoxb-1234567890-abcdefghij",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signature",
        "-----BEGIN RSA PRIVATE KEY-----",
        "пароль: Sup3rS3cretPass",
        "api_key=abcdef123456789",
    )

    def test_known_credentials_are_blocked(self) -> None:
        for case in self.CASES:
            with self.subTest(case=case[:30]):
                self.assertIsNotNone(looks_like_secret(case))
                with self.assertRaises(MemoryWriteError) as ctx:
                    self.remember(text=case)
                # Значение не должно попасть ни в ошибку, ни в лог-поле.
                self.assertNotIn("Sup3rS3cret", str(ctx.exception))

    def test_harmless_text_is_not_blocked(self) -> None:
        for case in (
            "пароль лежит в личных заметках Андрея, не в репозитории",
            "ключи приходят txt-файлами в data/downloads",
            "пароль: нет",
            "коммит 786c680 закрыл баг",
            "лимит free-tier 10000 символов",
        ):
            with self.subTest(case=case[:30]):
                self.assertIsNone(looks_like_secret(case))

    def test_secret_is_never_stored(self) -> None:
        with self.assertRaises(MemoryWriteError):
            self.remember(text="token: abcdef1234567890")
        self.assertEqual(self.store.stats()["nodes"], 0)


class ForgetTest(Base):
    def test_retire_keeps_row_but_hides_from_search(self) -> None:
        result = self.remember(name="старый лимит", text="устарело")
        retired = forget(self.store, node_id=result.node_id)
        self.assertEqual(retired[0][0], result.node_id)
        self.assertEqual(self.store.stats()["nodes"], 1)
        self.assertEqual(search_nodes(self.store, "лимит"), [])

    def test_forget_by_name(self) -> None:
        self.remember(name="оператор Андрей", kind="person")
        retired = forget(self.store, name="оператор Андрей")
        self.assertEqual(retired[0][1:], ("person", "оператор Андрей"))

    def test_forget_closes_live_edges(self) -> None:
        first = self.remember(name="Андрей", kind="person", text="оператор")
        second = self.remember(name="prolight", kind="server", text="сервер")
        self.store.connection.execute(
            "INSERT INTO edges(src, dst, rel, valid_from) VALUES(?,?,?,?)",
            (second.node_id, first.node_id, "about", 100),
        )
        self.store.connection.commit()
        forget(self.store, node_id=second.node_id)
        row = self.store.connection.execute(
            "SELECT valid_to FROM edges").fetchone()
        self.assertIsNotNone(row["valid_to"])

    def test_forget_twice_reports_nothing_found(self) -> None:
        result = self.remember()
        forget(self.store, node_id=result.node_id)
        with self.assertRaises(MemoryWriteError):
            forget(self.store, name="лимит ElevenLabs")

    def test_forget_without_arguments_is_rejected(self) -> None:
        with self.assertRaises(MemoryWriteError):
            forget(self.store)

    def test_forget_unknown_id(self) -> None:
        with self.assertRaises(MemoryWriteError):
            forget(self.store, node_id=9999)


class FormatTest(Base):
    def test_facts_and_history_are_separated(self) -> None:
        self.remember(name="лимит ElevenLabs", text="10000 символов")
        from server.memory_search import search_episodes

        self.store.add_episode(chat_id="1", speaker="human", kind="message",
                               ts=1_790_000_000,
                               text="Помни, лимит ElevenLabs кончается")
        text = format_all(
            "лимит ElevenLabs",
            search_nodes(self.store, "лимит ElevenLabs"),
            search_episodes(self.store, "лимит ElevenLabs"),
        )
        self.assertIn("Из памяти (факты):", text)
        self.assertIn("Из истории:", text)
        self.assertIn("[fact]", text)

    def test_empty_result_message(self) -> None:
        self.assertIn("ничего не найдено",
                      format_all("что-то", [], []).lower())


if __name__ == "__main__":
    unittest.main()
