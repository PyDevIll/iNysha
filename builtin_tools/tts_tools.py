"""ElevenLabs text-to-speech generation tool."""

from __future__ import annotations

from typing import Any

from loguru import logger

from lib.elevenlabs_tts import synthesize_to_file


async def tts_generate(
    text: str,
    lang: str = "ru-RU",
    speed: float = 1.2,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Generate speech audio with ElevenLabs.

    Args:
        text: Text to speak (plain speech, no code/URLs).
        lang: Language hint (unused; ElevenLabs auto-detects).
        speed: Speech speed hint (unused; kept for backward compatibility).
        output_dir: Optional output directory (default: configured TTS cache)

    Returns:
        dict: {"ok": True, "path": "...", "file_size": 1234} on success
    """
    return await synthesize_to_file(text, output_dir=output_dir)


def register_all(registry) -> None:
    """Register TTS tools with the given ToolRegistry."""
    registry.register_function(
        func=tts_generate,
        name="tts_generate",
        description="Generate speech audio (MP3) from text using ElevenLabs (eleven_multilingual_v2). Plain speech only; no code or URLs. Language is auto-detected.",
        parameters={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to speak (plain speech, no code or URLs)",
                },
                "lang": {
                    "type": "string",
                    "description": "Language hint (unused; auto-detected)",
                },
                "speed": {
                    "type": "number",
                    "description": "Speech speed hint (unused; kept for compatibility)",
                    "default": 1.2,
                },
                "output_dir": {
                    "type": "string",
                    "description": "Optional output directory (default: configured TTS cache)",
                },
            },
            "required": ["text"],
        },
    )
    logger.info("TTS tools registered: tts_generate")
