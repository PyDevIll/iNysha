"""Поиск по памяти: леммы, синонимы, префиксы, бюджеты и изоляция.

Этот слой скоро начнёт подмешивать фрагменты в промпт, поэтому проверяем не
только «находит», но и границы: пустой запрос, лимит k, фильтр по чату,
обрезка сниппета и то, что сырой текст пользователя не роняет FTS5.

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

from server import config as app_config
from server.memory_search import (
    SNIPPET_CHARS,
    EpisodeHit,
    format_hits,
    search_episodes,
)
from server.memory_store import MemoryStore, close_store

CHAT = "326963375"


class SearchTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self._tmp.name) / "memory.db")

    def tearDown(self) -> None:
        self.store.close()
        close_store()
        self._tmp.cleanup()

    def add(self, text: str, *, ts: int = 1_000_000, speaker: str = "human",
            chat_id: str = CHAT) -> int:
        row = self.store.add_episode(
            chat_id=chat_id, speaker=speaker, kind="message", ts=ts, text=text
        )
        self.assertIsNotNone(row, f"эпизод не записался: {text!r}")
        return int(row)

    def texts(self, query: str, **kwargs) -> list[str]:
        return [hit.text for hit in search_episodes(self.store, query, **kwargs)]


class MorphologyTest(SearchTestBase):
    def test_inflected_query_finds_base_form(self) -> None:
        self.add("Сервер перезапустили дважды")
        self.assertTrue(self.texts("сервером"))

    def test_base_query_finds_inflected_text(self) -> None:
        self.add("Говорили про серверы и их диски")
        self.assertTrue(self.texts("сервер"))

    def test_synonym_bridges_language_gap(self) -> None:
        self.add("MAX API вернул timeout на отправке")
        self.assertTrue(self.texts("таймаут"))

    def test_synonym_for_restart(self) -> None:
        self.add("Сделала systemctl restart inysha.service")
        self.assertTrue(self.texts("перезагрузить"))

    def test_prefix_fallback_for_unknown_stem(self) -> None:
        # «картинк» — не лемма; без отката на префикс ничего не найдётся.
        self.add("Сгенерировала картинку про кота")
        self.assertTrue(self.texts("картинк"))


class RankingTest(SearchTestBase):
    def test_best_match_is_first(self) -> None:
        self.add("Ничего важного про погоду", ts=1)
        self.add("ElevenLabs лимит символов закончился", ts=2)
        self.add("Снова про ElevenLabs лимит", ts=3)
        hits = search_episodes(self.store, "ElevenLabs лимит", k=2)
        self.assertEqual(len(hits), 2)
        self.assertIn("ElevenLabs", hits[0].text)

    def test_k_limits_results(self) -> None:
        for index in range(6):
            self.add(f"рестарт сервиса номер {index}", ts=100 + index)
        self.assertEqual(len(search_episodes(self.store, "рестарт", k=3)), 3)

    def test_k_is_capped(self) -> None:
        for index in range(70):
            self.add(f"уникальноеслово{index} рестарт", ts=100 + index)
        self.assertLessEqual(len(search_episodes(self.store, "уникальноеслово", k=500)), 50)

    def test_chat_filter(self) -> None:
        self.add("рестарт в первом чате", ts=1, chat_id="111")
        self.add("рестарт во втором чате", ts=2, chat_id="222")
        hits = search_episodes(self.store, "рестарт", chat_id="222")
        self.assertEqual([hit.chat_id for hit in hits], ["222"])

    def test_empty_query_returns_nothing(self) -> None:
        self.add("что-то")
        self.assertEqual(search_episodes(self.store, ""), [])
        self.assertEqual(search_episodes(self.store, "?!.,"), [])

    def test_no_match_returns_nothing(self) -> None:
        self.add("совершенно другой текст")
        self.assertEqual(search_episodes(self.store, "астрофизика"), [])

    def test_min_score_filters_weak_hits(self) -> None:
        self.add("рестарт сервиса")
        strong = search_episodes(self.store, "рестарт", min_score=None)
        self.assertTrue(strong)
        self.assertGreater(strong[0].score, 0, "relevance должна быть > 0")
        self.assertEqual(
            search_episodes(self.store, "рестарт", min_score=1e9), []
        )

    def test_wild_user_text_does_not_crash(self) -> None:
        self.add("рестарт сервиса")
        for raw in ('рестарт" OR "1', "NEAR(рестарт)", "рестарт* AND -сервис",
                    '("рестарт"', "рестарт^"):
            search_episodes(self.store, raw)  # не должно бросить


class FormattingTest(SearchTestBase):
    def test_empty_result_message(self) -> None:
        text = format_hits("что-то", [])
        self.assertIn("ничего не найдено", text.lower())

    def test_hits_are_dated_and_attributed(self) -> None:
        row_id = self.add("Рестарт сервиса прошёл", ts=1_790_000_000)
        hits = search_episodes(self.store, "рестарт")
        text = format_hits("рестарт", hits)
        self.assertIn("Найдено 1", text)
        self.assertIn("человек:", text)
        self.assertRegex(text, r"\[\d\d\.\d\d\.\d{4} \d\d:\d\d\]")
        self.assertEqual(hits[0].id, row_id)

    def test_snippet_is_shortened(self) -> None:
        long_text = "рестарт " + ("подробность " * 200)
        self.add(long_text)
        hit = search_episodes(self.store, "рестарт")[0]
        line = hit.as_line(1)
        self.assertLess(len(line), SNIPPET_CHARS + 80)
        self.assertIn("chars", line)

    def test_output_is_capped_by_token_budget(self) -> None:
        for index in range(10):
            self.add(f"рестарт {index} " + ("слово " * 120), ts=100 + index)
        hits = search_episodes(self.store, "рестарт", k=10)
        text = format_hits("рестарт", hits)
        self.assertIn("лимит", text)
        self.assertLess(len(text), app_config.MEMORY_BLOCK_TOKENS * 4)


class HitShapeTest(unittest.TestCase):
    def test_when_and_speaker_rendering(self) -> None:
        hit = EpisodeHit(id=1, ts=1_790_000_000, speaker="agent",
                         chat_id=CHAT, text="привет", score=1.0)
        self.assertRegex(hit.when, r"\d\d\.\d\d\.\d{4} \d\d:\d\d")
        self.assertIn("я:", hit.as_line(1))


class ToolTest(SearchTestBase):
    def setUp(self) -> None:
        super().setUp()
        from builtin_tools import memory_tools

        self.tools = memory_tools
        self._orig = memory_tools.get_store
        memory_tools.get_store = lambda: self.store

    def tearDown(self) -> None:
        self.tools.get_store = self._orig
        super().tearDown()

    def test_tool_returns_hits(self) -> None:
        import asyncio

        self.add("Рестарт сервиса прошёл успешно")
        text = asyncio.run(self.tools.memory_search("рестарт"))
        self.assertIn("Найдено 1", text)

    def test_tool_rejects_empty_query(self) -> None:
        import asyncio

        text = asyncio.run(self.tools.memory_search("  "))
        self.assertIn("Пустой запрос", text)

    def test_tool_is_registered_for_voice(self) -> None:
        from tool_registry import VOICE_DIRECT_TOOLS

        self.assertIn("memory_search", VOICE_DIRECT_TOOLS)
        names = [definition[0] for definition in self.tools.TOOL_DEFINITIONS]
        self.assertIn("memory_search", names)


if __name__ == "__main__":
    unittest.main()
