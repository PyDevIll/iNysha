"""ElevenLabs text-to-speech helper.

Used by the MAX voice reply tool (and by ``tts_generate``).  The default
model is ``eleven_multilingual_v2``, which handles Russian and most other
languages without a separate language parameter.

The API key and voice id are read from the project ``.env`` (or the process
environment).  Only the returned MP3 path is exposed to callers; the caller
owns the file and should delete it when it is no longer needed.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any, AsyncGenerator

import httpx
from dotenv import load_dotenv
from elevenlabs.client import AsyncElevenLabs
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DEFAULT_OUTPUT_DIR = ROOT / "data" / "tts_cache"
DEFAULT_MODEL = "eleven_multilingual_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"
API_BASE = "https://api.elevenlabs.io/v1/text-to-speech"


async def synthesize_to_file(
    text: str,
    *,
    voice_id: str | None = None,
    model_id: str | None = None,
    api_key: str | None = None,
    output_dir: str | Path | None = None,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> dict[str, Any]:
    """Generate an MP3 file with ElevenLabs.

    Args:
        text: Text to speak.
        voice_id: ElevenLabs voice id (defaults to ``ELEVENLABS_VOICE_ID``).
        model_id: ElevenLabs model (defaults to ``ELEVENLABS_MODEL_ID`` or
            ``eleven_multilingual_v2``).
        api_key: API key (defaults to ``ELEVENLABS_API_KEY``).
        output_dir: Directory for the generated MP3.
        output_format: ElevenLabs output format.

    Returns:
        ``{"ok": True, "path": ..., "file_size": ..., "voice_id": ..., "model": ...}``
        or ``{"ok": False, "error": ...}``.
    """
    text = (text or "").strip()
    if not text:
        return {"ok": False, "error": "text is required"}

    api_key = (
        api_key
        or os.getenv("ELEVENLABS_API_KEY")
        or os.getenv("ELEVENLABS_TTS_API_KEY")
        or ""
    )
    voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID") or ""
    model_id = model_id or os.getenv("ELEVENLABS_MODEL_ID") or DEFAULT_MODEL

    if not api_key:
        return {"ok": False, "error": "ELEVENLABS_API_KEY is not set"}
    if not voice_id:
        return {"ok": False, "error": "ELEVENLABS_VOICE_ID is not set"}

    max_chars = int(os.getenv("ELEVENLABS_MAX_CHARS", "4000"))
    if len(text) > max_chars:
        logger.warning(
            f"ElevenLabs text truncated from {len(text)} to {max_chars} chars"
        )
        text = text[:max_chars]

    dest_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / f"elevenlabs_{uuid.uuid4().hex}.mp3"

    headers = {
        "xi-api-key": api_key,
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{API_BASE}/{voice_id}",
                params={"output_format": output_format},
                headers=headers,
                json=payload,
            )
        if response.status_code != 200:
            detail = response.text[:500]
            logger.error(
                f"ElevenLabs TTS failed: HTTP {response.status_code}: {detail}"
            )
            return {
                "ok": False,
                "error": f"ElevenLabs HTTP {response.status_code}: {detail}",
            }
        if not response.content:
            return {"ok": False, "error": "ElevenLabs returned empty audio"}

        await asyncio.to_thread(out_path.write_bytes, response.content)
    except Exception as exc:
        logger.exception("ElevenLabs TTS request failed")
        return {"ok": False, "error": str(exc)}

    logger.info(
        f"ElevenLabs TTS generated: {len(text)} chars -> "
        f"{out_path.name} ({len(response.content)} bytes, voice={voice_id})"
    )
    return {
        "ok": True,
        "path": str(out_path),
        "file_size": len(response.content),
        "voice_id": voice_id,
        "model": model_id,
        "text": text,
    }


def _resolve_tts_settings(
    voice_id: str | None,
    model_id: str | None,
    api_key: str | None,
) -> tuple[str, str, str]:
    api_key = (
        api_key
        or os.getenv("ELEVENLABS_API_KEY")
        or os.getenv("ELEVENLABS_TTS_API_KEY")
        or ""
    )
    voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID") or ""
    model_id = model_id or os.getenv("ELEVENLABS_MODEL_ID") or DEFAULT_MODEL
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")
    if not voice_id:
        raise RuntimeError("ELEVENLABS_VOICE_ID is not set")
    return api_key, voice_id, model_id


async def stream_pcm(
    text: str,
    *,
    voice_id: str | None = None,
    model_id: str | None = None,
    api_key: str | None = None,
    output_format: str = "pcm_16000",
    optimize_streaming_latency: int = 3,
) -> AsyncGenerator[bytes, None]:
    """Yield raw little-endian 16-bit mono PCM from ElevenLabs.

    ``pcm_16000`` is the format used by the desktop voice client, so no
    client-side decoding or resampling is needed.
    """
    text = (text or "").strip()
    if not text:
        return
    api_key, voice_id, model_id = _resolve_tts_settings(
        voice_id, model_id, api_key
    )
    max_chars = int(os.getenv("ELEVENLABS_MAX_CHARS", "4000"))
    if len(text) > max_chars:
        logger.warning(
            f"ElevenLabs text truncated from {len(text)} to {max_chars} chars"
        )
        text = text[:max_chars]

    client = AsyncElevenLabs(api_key=api_key)
    stream = client.text_to_speech.stream(
        voice_id=voice_id,
        text=text,
        model_id=model_id,
        output_format=output_format,
        optimize_streaming_latency=optimize_streaming_latency,
        voice_settings={
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True,
        },
    )
    async for chunk in stream:
        if chunk:
            yield chunk


async def synthesize_pcm_bytes(
    text: str,
    *,
    voice_id: str | None = None,
    model_id: str | None = None,
    api_key: str | None = None,
    output_format: str = "pcm_16000",
    optimize_streaming_latency: int = 3,
) -> bytes:
    """Collect :func:`stream_pcm` into one bytes object."""
    chunks: list[bytes] = []
    async for chunk in stream_pcm(
        text,
        voice_id=voice_id,
        model_id=model_id,
        api_key=api_key,
        output_format=output_format,
        optimize_streaming_latency=optimize_streaming_latency,
    ):
        chunks.append(chunk)
    return b"".join(chunks)
