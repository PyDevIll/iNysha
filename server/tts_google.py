"""Бесплатный TTS через Google Translate с потоковой отдачей PCM.

Google Translate TTS отдаёт MP3 и принимает не больше ~200 символов за
запрос. Поэтому длинный ответ режется на короткие куски (предложения),
каждый кусок синтезируется отдельным HTTP-запросом, конвертируется в
сырой PCM 16000 Гц / моно / s16le и сразу отдаётся вызывающему коду.

Контракт для браузерного клиента не меняется: он по-прежнему получает
бинарные кадры 16 кГц / 16 бит / моно.

Пайплайн устроен так, чтобы минимизировать задержку первого звука:
пока воспроизводится (отправляется в сокет) кусок N, следующий кусок
N+1 уже качается и конвертируется в фоне. Одновременно в работе не
больше одного префетча, чтобы не злить Google.
"""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Any

import httpx
from loguru import logger

# Официальный лимит Google Translate TTS — около 200 символов.
# 180 оставляем с запасом на urlencode и особенности endpoint'а.
DEFAULT_LIMIT = 180

GOOGLE_TTS_URL = "https://translate.google.com/translate_tts"

# Без этих заголовков Google отвечает 403 Forbidden.
GOOGLE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://translate.google.com/",
    "Accept": "audio/mpeg,audio/*;q=0.9,*/*;q=0.8",
}

# Границы предложений: точки, восклицательный/вопросительный знаки,
# многоточие, точка с запятой, двоеточие и переводы строк.
_SENTENCE_RE = re.compile(r"[^.!?…;:\n]+(?:[.!?…;:]+|$)")

StatusCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


def split_for_tts(text: str, limit: int = DEFAULT_LIMIT) -> list[str]:
    """Разрезать текст на куски не длиннее ``limit`` символов.

    Сначала режем по границам предложений (``. ! ? … ; :`` и переводы
    строк), затем слишком длинные предложения — по запятым, а если их
    нет, то по пробелам. Пустые куски выбрасываются.

    Куски сохраняют исходные слова и знаки препинания: если склеить
    результат через пробел, получится нормализованный исходный текст.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")

    text = (text or "").strip()
    if not text:
        return []

    pieces: list[str] = []
    for raw in _SENTENCE_RE.findall(text):
        sentence = re.sub(r"\s+", " ", raw).strip()
        if not sentence:
            continue
        if len(sentence) <= limit:
            pieces.append(sentence)
        else:
            pieces.extend(_split_long(sentence, limit))
    return [piece for piece in pieces if piece]


def _split_long(text: str, limit: int) -> list[str]:
    """Разрезать длинное предложение, не превышая ``limit``."""
    result: list[str] = []
    rest = text

    while len(rest) > limit:
        window = rest[:limit]
        cut = -1

        # Сначала пробуем резать по запятой, но только если она не в
        # самом начале окна — иначе получим неестественно короткий кусок.
        comma = window.rfind(",")
        if comma >= limit // 2:
            cut = comma + 1

        if cut <= 0:
            space = window.rfind(" ")
            if space > 0:
                cut = space

        # Совсем длинное слово без пробелов/запятых режем жёстко.
        if cut <= 0:
            cut = limit

        piece = rest[:cut].strip()
        rest = rest[cut:].strip()
        if piece:
            result.append(piece)

    if rest:
        result.append(rest)
    return result


async def synth_piece(text: str) -> bytes:
    """Синтезировать один короткий кусок и вернуть MP3-байты.

    Делает один повтор при 4xx/5xx/таймауте. Если и вторая попытка
    неудачна — поднимает исключение, чтобы вызывающий код смог отдать
    клиенту понятную ошибку вместо тишины.
    """
    text = (text or "").strip()
    if not text:
        return b""

    params = {
        "ie": "UTF-8",
        "q": text,
        "tl": "ru",
        "client": "tw-ob",
        "total": "1",
        "idx": "0",
        "textlen": str(len(text)),
    }

    last_error: Exception | None = None
    for attempt in range(1, 3):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(20.0, connect=10.0),
                follow_redirects=True,
            ) as client:
                response = await client.get(
                    GOOGLE_TTS_URL,
                    params=params,
                    headers=GOOGLE_HEADERS,
                )
            if response.status_code == 200:
                audio = response.content
                if not audio:
                    raise RuntimeError("Google TTS вернул пустой ответ")
                return audio
            last_error = RuntimeError(
                f"Google TTS HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
            logger.warning(
                f"Google TTS: попытка {attempt}/2, "
                f"HTTP {response.status_code} для {len(text)} символов"
            )
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            last_error = exc
            logger.warning(
                f"Google TTS: попытка {attempt}/2 не удалась: {exc}"
            )

        if attempt == 1:
            await asyncio.sleep(0.3)

    raise RuntimeError(f"Google TTS не ответил после 2 попыток: {last_error}")


async def mp3_to_pcm16(mp3: bytes) -> bytes:
    """Сконвертировать MP3 в сырой PCM 16000 Гц / моно / s16le через ffmpeg."""
    if not mp3:
        return b""

    try:
        import imageio_ffmpeg
    except Exception as exc:  # pragma: no cover - зависит от окружения
        raise RuntimeError(
            "imageio_ffmpeg недоступен: нечем конвертировать MP3 в PCM"
        ) from exc

    try:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - зависит от окружения
        raise RuntimeError(
            f"imageio_ffmpeg не смог найти ffmpeg: {exc}"
        ) from exc

    proc = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await proc.communicate(mp3)
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise

    if proc.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"ffmpeg завершился с кодом {proc.returncode}: {detail}")

    pcm = stdout
    if len(pcm) % 2:
        # На всякий случай не отдаём клиенту неполный 16-битный сэмпл.
        pcm = pcm[:-1]
    if not pcm:
        raise RuntimeError("ffmpeg вернул пустой PCM")
    return pcm


async def _synth_pcm(text: str) -> tuple[bytes, bytes, float]:
    """Синтез + конвертация одного куска. Возвращает (mp3, pcm, секунды)."""
    started = time.perf_counter()
    mp3 = await synth_piece(text)
    pcm = await mp3_to_pcm16(mp3)
    return mp3, pcm, time.perf_counter() - started


async def stream_google_pcm(
    text: str,
    on_status: StatusCallback | None = None,
) -> AsyncIterator[bytes]:
    """Отдавать PCM-куски по мере готовности Google TTS.

    Конвейер: ждём текущий кусок, запускаем синтез следующего, отдаём
    текущий PCM. Пока клиент проигрывает кусок N, кусок N+1 уже
    скачивается. Максимум один префетч в полёте.
    """
    pieces = split_for_tts(text)
    if not pieces:
        return

    total = len(pieces)
    current: asyncio.Task | None = None

    try:
        current = asyncio.create_task(
            _synth_pcm(pieces[0]),
            name="google-tts-piece-1",
        )

        for index, piece in enumerate(pieces):
            mp3, pcm, elapsed = await current
            current = None

            # Следующий кусок качается, пока текущий уходит в сокет.
            if index + 1 < total:
                current = asyncio.create_task(
                    _synth_pcm(pieces[index + 1]),
                    name=f"google-tts-piece-{index + 2}",
                )

            if on_status is not None:
                status = {
                    "index": index,
                    "total": total,
                    "text": piece,
                    "mp3_bytes": len(mp3),
                    "pcm_bytes": len(pcm),
                    "elapsed_s": round(elapsed, 3),
                }
                maybe_awaitable = on_status(status)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable

            if pcm:
                yield pcm
    finally:
        # При отмене/обрыве сокета не оставляем висящий HTTP-запрос.
        if current is not None and not current.done():
            current.cancel()
        if current is not None:
            with suppress(asyncio.CancelledError, Exception):
                await current
