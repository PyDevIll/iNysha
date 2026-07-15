"""Google Translate TTS + Yandex Transcribe tools for MASTERMIND v2.


Provides:
- tts_generate: generate MP3 audio file from text
- yandex_transcribe
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import urllib.parse
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

DEFAULT_TTS_DIR = Path(__file__).resolve().parent.parent / "data" / "tts_cache"
MAX_TEXT_LENGTH = 200  # Google TTS query limit

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXTENSION_CONTENT_TYPE = {
    "ogg": "audio/ogg",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "wav": "audio/wav",
}


def _content_type_from_extension(filename: str) -> str:
    """Guess audio MIME type based on filename extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _EXTENSION_CONTENT_TYPE.get(ext, "audio/ogg")


# ---------------------------------------------------------------------------
# Yandex SpeechKit STT (free tier)
# ---------------------------------------------------------------------------

_YANDEX_STT_URL = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"


async def _transcribe_yandex_bytes(
    audio_bytes: bytes,
    content_type: str,
    lang: str = "ru-RU",
    topic: str = "general",
) -> dict[str, Any]:
    """Send raw audio bytes to Yandex SpeechKit STT API and return result."""
    api_key = os.environ.get("YANDEX_SPEECH_API_KEY", "")
    if not api_key:
        return {"ok": False, "error": "YANDEX_SPEECH_API_KEY env var not set"}
    params = {"topic": topic, "lang": lang}
    headers = {"Authorization": f"Api-Key {api_key}", "Content-Type": content_type}
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _YANDEX_STT_URL,
                params=params,
                headers=headers,
                content=audio_bytes,
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("result", "")
            if not text:
                return {"ok": False, "error": "Empty recognition result"}
            return {
                "ok": True,
                "text": text,
                "language": lang,
                "model": "yandex-speechkit-stt",
            }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Tool for transcribe
# ---------------------------------------------------------------------------


async def yandex_transcribe(
    audio_path: str | None = None,
    url: str | None = None,
    lang: str = "ru-RU",
    topic: str = "general",
) -> dict[str, Any]:
    """Transcribe audio file to text using Yandex SpeechKit STT (free tier).

    Accepts either a local file path or a public URL to an audio file.
    If both are provided, the URL is used.

    Args:
        audio_path: Path to audio file (OGG/WAV/MP3). Required if url not given.
        url: Public URL of audio file (OGG/WAV/MP3). Required if audio_path not given.
        lang: Recognition language (default "ru-RU").
        topic: Topic domain (default "general").

    Returns:
        dict with ok, text, language, model on success.
    """
    if not audio_path and not url:
        return {"ok": False, "error": "Either audio_path or url must be provided"}

    try:
        if url:
            # Prefer URL over local path
            async with httpx.AsyncClient(verify=False) as client:
                resp = await client.get(url, timeout=30.0)
                resp.raise_for_status()
                audio_bytes = resp.content
                content_type = _content_type_from_extension(url)
        else:
            path = Path(audio_path)
            if not path.exists():
                return {"ok": False, "error": f"File not found: {audio_path}"}
            audio_bytes = await asyncio.to_thread(path.read_bytes)
            content_type = _content_type_from_extension(audio_path)

        return await _transcribe_yandex_bytes(audio_bytes, content_type, lang, topic)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# TTS via Google Translate (unchanged)
# ---------------------------------------------------------------------------


async def _tts_google(
    text: str,
    lang: str = "ru",
    speed: float = 1.5,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Generate TTS audio via Google Translate and save as MP3.

    Args:
        text: Text to speak (max 200 chars)
        lang: Language code (ru, en, ja, etc.)
        speed: Speech speed (1.0 = normal, 1.5 = fast)
        output_dir: Directory to save audio (default: data/tts_cache)

    Returns:
        dict with ok, path, file_size, text, lang
    """
    text = text.strip()
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
        logger.warning(f"TTS text truncated to {MAX_TEXT_LENGTH} chars")

    dest_dir = Path(output_dir) if output_dir else DEFAULT_TTS_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Derive filename from text hash
    text_hash = hashlib.md5(f"{lang}:{text}".encode()).hexdigest()[:12]
    out_path = dest_dir / f"tts_{lang}_{text_hash}.mp3"

    # Build Google Translate TTS URL
    encoded_text = urllib.parse.quote(text)
    tts_url = (
        f"https://translate.google.com/translate_tts"
        f"?ie=UTF-8&client=tw-ob&tl={lang}&ttsspeed={speed}&q={encoded_text}"
    )

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(tts_url, timeout=30.0, follow_redirects=True)
            resp.raise_for_status()

            # Check if we got actual audio (not an error page)
            content_type = resp.headers.get("content-type", "")
            if "audio" not in content_type and len(resp.content) < 512:
                return {"ok": False, "error": f"Google TTS returned non-audio: {resp.content[:200]}"}

            await asyncio.to_thread(out_path.write_bytes, resp.content)

        file_size = out_path.stat().st_size
        logger.info(f"TTS generated: '{text[:50]}...' → {out_path.name} ({file_size} bytes)")
        return {
            "ok": True,
            "path": str(out_path),
            "file_size": file_size,
            "text": text,
            "lang": lang,
        }
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


async def tts_generate(
    text: str,
    lang: str = "ru",
    speed: float = 1.5,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Generate speech audio from text using Google Translate TTS (FREE, no API key).

    Args:
        text: Text to speak (max 200 chars, Russian by default)
        lang: Language code: ru (Russian), en (English), ja (Japanese), etc.
        speed: Speech speed (1.0 = normal, 1.5 = fast)
        output_dir: Optional output directory (default: data/tts_cache)

    Returns:
        dict: {\"ok\": True, \"path\": \"...\", \"file_size\": 1234} on success
    """
    return await _tts_google(text=text, lang=lang, speed=speed, output_dir=output_dir)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_all(registry) -> None:
    """Register TTS/STT tools with the given ToolRegistry."""
    registry.register_function(
        func=tts_generate,
        name="tts_generate",
        description="Generate speech audio (MP3) from text using Google Translate TTS. FREE, no API key needed. Max 200 chars. Supports ru, en, ja, and 50+ languages.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to speak (max 200 chars, Russian by default)",
                },
                "lang": {
                    "type": "string",
                    "description": "Language code: ru, en, ja, etc. (default: ru)",
                },
                "speed": {
                    "type": "number",
                    "description": "Скорость речи (1.0 = нормальная, 1.5 = быстрая)",
                    "default": 1.5,
                },
                "output_dir": {
                    "type": "string",
                    "description": "Optional output directory (default: data/tts_cache)",
                },
            },
            "required": ["text"],
        },
    )
    registry.register_function(
        func=yandex_transcribe,
        name="yandex_transcribe",
        description="Transcribe audio file to text using Yandex SpeechKit STT (free tier). Supports OGG/MP3/WAV from local file or URL. Returns recognized text.",
        parameters={
            "type": "object",
            "properties": {
                "audio_path": {
                    "type": "string",
                    "description": "Path to audio file (OGG/WAV/MP3). Required if url not provided.",
                },
                "url": {
                    "type": "string",
                    "description": "Public URL of audio file (OGG/WAV/MP3). Required if audio_path not provided. If both provided, url is used.",
                },
                "lang": {
                    "type": "string",
                    "description": "Recognition language (default: ru-RU)",
                },
                "topic": {
                    "type": "string",
                    "description": "Topic domain (default: general)",
                },
            },
            "required": [],  # at least one of audio_path or url must be provided (validated in function)
        },
    )
    logger.info("TTS/STT tools registered: tts_generate, yandex_transcribe")
