"""Нормализация и лемматизация для поиска по памяти.

Проверяем то, на чём держится FTS5 в русском: ё/регистр/пунктуация,
леммы вместо словоформ и безопасная сборка MATCH-запроса (сырой текст
пользователя роняет FTS5 в OperationalError).

Тесты не трогают data/ и не требуют сети.
Запуск: .venv/bin/python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.memory_text import (
    DEFAULT_PREFIX_LEN,
    fts_query,
    has_morphology,
    lemma,
    lemmatize,
    normalize_name,
    normalize_text,
    tokenize,
)


class NormalizeTest(unittest.TestCase):
    def test_normalize_text_lowercases_and_replaces_diacritics(self) -> None:
        self.assertEqual(normalize_text("Ёлка, Привет!"), "елка привет")

    def test_normalize_text_drops_punctuation_between_words(self) -> None:
        self.assertEqual(normalize_text("сервер: упал? — да."), "сервер упал да")

    def test_normalize_text_keeps_digits(self) -> None:
        self.assertEqual(normalize_text("порт 8010"), "порт 8010")

    def test_normalize_name_keeps_dotted_token_intact(self) -> None:
        # Домен — отдельная сущность, его нельзя разваливать на два слова.
        self.assertEqual(normalize_name("aikartinki.ru"), "aikartinki.ru")

    def test_normalize_name_strips_surrounding_punctuation(self) -> None:
        self.assertEqual(normalize_name("  Нюша! "), "нюша")

    def test_normalize_name_does_not_transliterate_latin(self) -> None:
        # «iНюша» — смешанный алфавит: латинскую i нельзя молча превращать в
        # «и», иначе поиск по одному слову начнёт склеивать разные сущности.
        # Такие варианты живут в node_aliases, а не в нормализации.
        self.assertEqual(normalize_name("iНюша"), "iнюша")

    def test_normalize_name_collapses_spaces(self) -> None:
        self.assertEqual(normalize_name("Делирия   Фора"), "делирия фора")

    def test_tokenize_returns_empty_for_punctuation_only(self) -> None:
        self.assertEqual(tokenize("?!.,—"), [])


class LemmaTest(unittest.TestCase):
    def test_morphology_available(self) -> None:
        # Зависимость объявлена в requirements.txt; без неё поиск деградирует.
        self.assertTrue(has_morphology(), "pymorphy3 не установлен")

    def test_lemma_merges_russian_word_forms(self) -> None:
        self.assertEqual(lemma("сервером"), lemma("сервер"))
        self.assertEqual(lemma("сервер"), "сервер")

    def test_lemma_handles_names(self) -> None:
        self.assertEqual(lemma("андрею"), lemma("андрей"))

    def test_lemma_lowercases_and_normalises_yo(self) -> None:
        self.assertEqual(lemma("ёлка"), lemma("елка"))

    def test_lemmatize_joins_lemmas(self) -> None:
        self.assertEqual(lemmatize("Сервер упал дважды"), "сервер упасть дважды")


class FtsQueryTest(unittest.TestCase):
    def test_empty_query_for_empty_text(self) -> None:
        self.assertEqual(fts_query(""), "")
        self.assertEqual(fts_query("!!! ... ???"), "")

    def test_query_is_lemmatised(self) -> None:
        query = fts_query("сервером")
        self.assertIn('"сервер"', query)

    def test_prefix_mode_uses_star(self) -> None:
        query = fts_query("перезапустил", prefix_len=DEFAULT_PREFIX_LEN)
        self.assertTrue(query.endswith("*"), query)
        self.assertIn('"переза"', query)

    def test_tokens_are_quoted_and_deduplicated(self) -> None:
        query = fts_query("сервер сервер сервер")
        self.assertEqual(query, '"сервер"')

    def test_short_tokens_are_dropped(self) -> None:
        self.assertEqual(fts_query("я в"), "")

    def test_query_is_capped(self) -> None:
        query = fts_query(" ".join(f"слово{i}" for i in range(200)))
        self.assertLessEqual(query.count('"'), 24 * 2)

    def test_query_expands_known_synonyms(self) -> None:
        # Морфология не спасёт «таймаут» → «timeout»; это делает словарь.
        self.assertEqual(fts_query("таймаут"), '("таймаут" OR "timeout")')

    def test_unknown_word_has_no_group(self) -> None:
        # Лемматизация всё равно сработает, но без OR-группы синонимов.
        self.assertEqual(fts_query("астрофизика"), '"астрофизик"')

    def test_query_survives_fts_syntax_in_user_text(self) -> None:
        # Раньше такой текст ронял MATCH; теперь он просто слова.
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(lemma_text)")
        conn.execute("INSERT INTO t VALUES('сервер упасть дважды')")
        for raw in (
            'сервер" OR "1',
            "NEAR(сервер упал)",
            "сервер* AND -упал",
            '("сервер"',
            "сервер^ упал:",
        ):
            query = fts_query(raw)
            if not query:
                continue
            rows = conn.execute(
                "SELECT rowid FROM t WHERE t MATCH ?", (query,)
            ).fetchall()
            self.assertIsInstance(rows, list, raw)

    def test_query_matches_inflected_text(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(lemma_text)")
        conn.execute(
            "INSERT INTO t VALUES(?)", (lemmatize("Сервер перезапустили"),)
        )
        rows = conn.execute(
            "SELECT rowid FROM t WHERE t MATCH ?", (fts_query("сервером"),)
        ).fetchall()
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
