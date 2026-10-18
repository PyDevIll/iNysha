"""Audio helpers: PCM/WAV conversion, RMS, Groq Whisper transcription."""

from __future__ import annotations

import array
import io
import re
import wave
from pathlib import Path

from groq import AsyncGroq
from loguru import logger

from server import config

_CONTENT_TYPES = {
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
}

# Minimum duration worth transcribing (0.4 s at 16 kHz / 16-bit mono).
MIN_AUDIO_BYTES = int(16000 * 2 * 0.4)
# Below this RMS a PCM clip is treated as silence.
SILENCE_RMS_THRESHOLD = 120.0

_HALLUCINATION_PATTERNS = [
    r"продолжение следует",
    r"спасибо за просмотр",
    r"спасибо за внимание",
    r"подписывайтесь на канал",
    r"не забудьте подписаться",
    r"субтитры сделал",
    r"редактор субтитров",
    r"ставьте лайк",
]


def pcm_to_wav(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Wrap raw little-endian PCM in a RIFF/WAVE container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def pcm_rms(pcm_bytes: bytes, stride: int = 4) -> float:
    """Root-mean-square amplitude of 16-bit mono PCM (sampled for speed)."""
    usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
    if usable == 0:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm_bytes[:usable])
    if not samples:
        return 0.0
    total = 0
    count = 0
    for i in range(0, len(samples), stride):
        value = samples[i]
        total += value * value
        count += 1
    return (total / count) ** 0.5 if count else 0.0


def is_likely_hallucination(text: str) -> bool:
    normalized = re.sub(r"[^\w\s]", "", text.lower()).strip()
    if not normalized:
        return True
    return any(re.search(p, normalized) for p in _HALLUCINATION_PATTERNS)


def content_type_for(filename: str) -> str:
    return _CONTENT_TYPES.get(Path(filename).suffix.lower(), "application/octet-stream")


def sanitize_for_tts(text: str) -> str:
    """Strip markdown/emoji so the text reads well through TTS."""
    allowed = r"[^a-zA-Zа-яА-ЯёЁ0-9\s\.\,\!\?\;\:\'\-\–\—]"
    cleaned = re.sub(allowed, " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


async def transcribe_bytes(
    data: bytes,
    filename: str = "voice.ogg",
    content_type: str | None = None,
    language: str | None = None,
) -> str:
    """Transcribe an audio container (OGG/MP3/WAV/...) with Groq Whisper."""
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set")
    if not data:
        return ""

    client = AsyncGroq(api_key=config.GROQ_API_KEY)
    try:
        result = await client.audio.transcriptions.create(
            file=(filename, data, content_type or content_type_for(filename)),
            model=config.GROQ_STT_MODEL,
            language=language or config.STT_LANGUAGE,
            response_format="text",
        )
    finally:
        await client.close()

    text = result.strip() if isinstance(result, str) else (result.text or "").strip()
    logger.debug(f"STT ({filename}, {len(data)}B): {text[:160]!r}")
    return text


async def transcribe_pcm(pcm_bytes: bytes, sample_rate: int = 16000) -> str:
    """Transcribe raw 16-bit mono PCM (voice sessions)."""
    return await transcribe_bytes(
        pcm_to_wav(pcm_bytes, sample_rate=sample_rate),
        filename="audio.wav",
        content_type="audio/wav",
    )
