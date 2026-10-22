"""Тесты бесплатного Google Translate TTS.

Проверяем:
  1. нарезку длинного ответа на куски не длиннее лимита без потери слов;
  2. конвертацию реального MP3 в PCM 16000 Гц / моно / s16le;
  3. (опционально) сетевой запрос к Google — только при NET_TESTS=1.

Запуск: .venv/bin/python -m unittest discover -s tests -v
Для сетевого теста: NET_TESTS=1 .venv/bin/python -m unittest tests.test_tts_google -v
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.tts_google import mp3_to_pcm16, split_for_tts, synth_piece

# Реальный MP3 от Google Translate (24 кГц, моно), уже лежит в data/.
GOOGLE_MP3 = ROOT / "data" / "downloads" / "gtts_test.mp3"

_DURATION_RE = re.compile(
    r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)"
)


def _words(text: str) -> list[str]:
    """Слова текста — для проверки, что нарезка ничего не потеряла."""
    return text.split()


def _mp3_duration_seconds(path: Path) -> float:
    """Длительность MP3 через ffmpeg -i (ffprobe в imageio_ffmpeg нет)."""
    import imageio_ffmpeg

    proc = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
    )
    match = _DURATION_RE.search(proc.stderr)
    if match is None:
        raise AssertionError(f"не удалось определить длительность {path}")
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


class SplitForTtsTest(unittest.TestCase):
    def test_long_text_pieces_fit_limit_and_keep_all_words(self) -> None:
        text = (
            "Первое предложение для проверки разбиения на куски. "
            "Второе предложение содержит запятые, чтобы проверить резку "
            "по запятым, и остаётся достаточно длинным для теста. "
            "Третье предложение заканчивается многоточием… "
            "Четвёртое: содержит двоеточие и точку с запятой; проверяем "
            "границы предложений.\n"
            "Пятое предложение без знаков в конце. "
            + "Это очень длинное предложение без точек, которое должно "
            "быть разрезано по запятым или пробелам, потому что иначе "
            "Google Translate TTS просто не примет его целиком, "
            "а нам нужно отдавать звук как можно раньше. " * 4
        )

        pieces = split_for_tts(text)

        self.assertTrue(pieces)
        self.assertTrue(all(len(piece) <= 180 for piece in pieces))
        self.assertEqual(
            _words(" ".join(pieces)),
            _words(text),
            "нарезка потеряла или переставила слова",
        )

    def test_sentence_boundaries(self) -> None:
        pieces = split_for_tts("Привет, мир. Как дела? Всё хорошо!")

        self.assertEqual(
            pieces,
            ["Привет, мир.", "Как дела?", "Всё хорошо!"],
        )

    def test_long_sentence_without_dots_is_split(self) -> None:
        text = "слово " * 80
        pieces = split_for_tts(text.strip())

        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(piece) <= 180 for piece in pieces))
        self.assertEqual(_words(" ".join(pieces)), _words(text))

    def test_very_long_token_is_hard_cut(self) -> None:
        pieces = split_for_tts("а" * 400, limit=180)

        self.assertEqual([len(piece) for piece in pieces], [180, 180, 40])

    def test_empty_text(self) -> None:
        self.assertEqual(split_for_tts("   \n  "), [])


class Mp3ToPcm16Test(unittest.TestCase):
    @unittest.skipUnless(
        GOOGLE_MP3.exists(),
        f"нет реального MP3 для проверки: {GOOGLE_MP3}",
    )
    def test_real_google_mp3_converts_to_16k_mono_pcm(self) -> None:
        mp3 = GOOGLE_MP3.read_bytes()
        pcm = asyncio.run(mp3_to_pcm16(mp3))

        self.assertGreater(len(pcm), 0)
        self.assertEqual(len(pcm) % 2, 0, "PCM должен быть 16-битным")

        estimated = len(pcm) / 2 / 16000
        expected = _mp3_duration_seconds(GOOGLE_MP3)

        # MP3-энкодер добавляет небольшую задержку, поэтому допускаем
        # полсекунды или 5% длительности.
        tolerance = max(0.5, expected * 0.05)
        self.assertLessEqual(
            abs(estimated - expected),
            tolerance,
            f"длительность PCM {estimated:.2f}s не совпала с MP3 "
            f"{expected:.2f}s",
        )

    def test_empty_mp3_returns_empty_pcm(self) -> None:
        self.assertEqual(asyncio.run(mp3_to_pcm16(b"")), b"")


class SynthPieceNetworkTest(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("NET_TESTS"),
        "сетевой тест: задайте NET_TESTS=1",
    )
    def test_synth_piece_returns_mp3(self) -> None:
        mp3 = asyncio.run(synth_piece("Проверка связи, один два три."))

        self.assertGreater(len(mp3), 512)
        self.assertTrue(
            mp3.startswith(b"ID3") or mp3[:2] in (b"\xff\xfb", b"\xff\xf3"),
            "ответ Google не похож на MP3",
        )


if __name__ == "__main__":
    unittest.main()
