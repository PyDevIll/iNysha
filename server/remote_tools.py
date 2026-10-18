"""Tools that call the connected desktop client.

These override the local display/audio tools from ``builtin_tools`` on the
cloud server.  The server has no screen, microphone or speaker; the thin
client owns all three.
"""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime
from typing import Any

from loguru import logger

from lib.elevenlabs_tts import synthesize_pcm_bytes
from server import config
from server.audio import sanitize_for_tts, transcribe_bytes
from server.clients import hub
from server.context import get_request_context


def _safe_client_id(client_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in client_id)[:48]


async def list_clients() -> str:
    """List desktop clients currently connected to the server."""
    return json.dumps({"ok": True, "clients": hub.list()}, ensure_ascii=False)


async def get_screenshot(
    monitor: int = 0,
    client_id: str | None = None,
    resize_factor: float = 0.5,
) -> dict[str, Any]:
    """Capture a screenshot on the target desktop client and save it server-side.

    Args:
        monitor: 0 = all monitors combined, 1 = primary, 2 = secondary, ...
        client_id: Optional client id; defaults to the current request's client.
        resize_factor: Scale factor applied on the client (0.5 = half size).
    """
    info = hub.resolve(client_id)
    result = await info.call(
        "screenshot",
        {
            "monitor": int(monitor),
            "resize_factor": float(resize_factor),
            "format": "png",
        },
        timeout=30.0,
    )
    data = base64.b64decode(result["data"])
    config.SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = (
        config.SCREENSHOTS_DIR
        / f"screenshot_{stamp}_{_safe_client_id(info.client_id)}.png"
    )
    await asyncio.to_thread(path.write_bytes, data)
    logger.info(
        f"screenshot from client {info.client_id}: {path} "
        f"({len(data)} bytes, {result.get('width')}x{result.get('height')})"
    )
    return {
        "ok": True,
        "path": str(path),
        "file_size": len(data),
        "client_id": info.client_id,
        "width": result.get("width"),
        "height": result.get("height"),
    }


async def analyze_dynamic_scene(
    monitor: int = 1,
    interval_seconds: float = 0.5,
    num_frames: int = 5,
    query: str = (
        "Опиши динамические изменения в этих последовательных кадрах. "
        "Дай детальный анализ того, что происходит в целом, на русском языке."
    ),
    client_id: str | None = None,
) -> str:
    """Capture a sequence on the client and analyze it with DeepSeek vision."""
    from builtin_tools.vision_tools import _call_deepseek_vision

    info = hub.resolve(client_id)
    frames = max(1, min(int(num_frames), 10))
    interval = max(0.0, min(float(interval_seconds), 5.0))
    result = await info.call(
        "capture_sequence",
        {
            "monitor": int(monitor),
            "interval_seconds": interval,
            "num_frames": frames,
            "resize_factor": 0.5,
            "format": "jpeg",
        },
        timeout=15.0 + frames * (interval + 2.0),
    )
    images = [
        f"data:{item.get('mime') or 'image/png'};base64,{item['data']}"
        for item in result.get("frames", [])
        if item.get("data")
    ]
    if not images:
        return "Не удалось получить кадры с экрана клиента."
    logger.info(
        f"analyze_dynamic_scene: {len(images)} frames from client "
        f"{info.client_id}"
    )
    return await _call_deepseek_vision(images, query)


async def tts_speak_aloud(
    text: str,
    client_id: str | None = None,
) -> dict[str, Any]:
    """Synthesize speech and play it on the target client's speakers."""
    ctx = get_request_context()
    if ctx is not None and ctx.source == "voice":
        return {
            "ok": False,
            "error": (
                "tts_speak_aloud is disabled during voice turns; the voice "
                "pipeline speaks the final answer automatically"
            ),
        }
    text = sanitize_for_tts(text)
    if not text:
        return {"ok": False, "error": "text is required"}
    info = hub.resolve(client_id)
    pcm = await synthesize_pcm_bytes(
        text,
        output_format=config.VOICE_TTS_FORMAT,
        optimize_streaming_latency=config.VOICE_TTS_LATENCY,
    )
    if not pcm:
        return {"ok": False, "error": "TTS returned empty audio"}
    duration = len(pcm) / (config.VOICE_SAMPLE_RATE * 2)
    await info.call(
        "play_pcm",
        {
            "data": base64.b64encode(pcm).decode("ascii"),
            "sample_rate": config.VOICE_SAMPLE_RATE,
            "channels": 1,
        },
        timeout=max(30.0, duration + 20.0),
    )
    return {
        "ok": True,
        "client_id": info.client_id,
        "bytes": len(pcm),
        "duration_seconds": round(duration, 2),
    }


async def client_record_audio(
    duration: float = 5.0,
    transcribe: bool = True,
    client_id: str | None = None,
) -> dict[str, Any]:
    """Record microphone audio on the target client and optionally transcribe it."""
    info = hub.resolve(client_id)
    seconds = max(0.5, min(float(duration), 30.0))
    result = await info.call(
        "record_audio",
        {"duration": seconds},
        timeout=seconds + 20.0,
    )
    wav = base64.b64decode(result["data"])
    config.DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = (
        config.DOWNLOADS_DIR
        / f"client_{_safe_client_id(info.client_id)}_{stamp}.wav"
    )
    await asyncio.to_thread(path.write_bytes, wav)

    text = ""
    if transcribe:
        try:
            text = await transcribe_bytes(
                wav, filename=path.name, content_type="audio/wav"
            )
        except Exception as exc:
            logger.exception("client recording transcription failed")
            return {
                "ok": False,
                "path": str(path),
                "error": f"recording saved but transcription failed: {exc}",
            }
    return {
        "ok": True,
        "path": str(path),
        "file_size": len(wav),
        "duration_seconds": round(seconds, 2),
        "client_id": info.client_id,
        "text": text,
    }


async def client_notify(
    message: str,
    title: str = "iNysha",
    client_id: str | None = None,
) -> dict[str, Any]:
    """Show a desktop notification on the target client."""
    info = hub.resolve(client_id)
    await info.call(
        "notify",
        {"title": title, "message": message},
        timeout=10.0,
    )
    return {"ok": True, "client_id": info.client_id}


def register_all(registry) -> None:
    """Register/override client-facing tools."""
    registry.register_function(
        func=list_clients,
        name="list_clients",
        description=(
            "List desktop clients connected to the server. Returns client_id, "
            "name and capabilities. Use client_id to target a specific client."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
    )
    registry.register_function(
        func=get_screenshot,
        name="get_screenshot",
        description=(
            "Capture a screenshot on a connected desktop client and save it "
            "server-side. Returns the local path for vision_analyze."
        ),
        parameters={
            "type": "object",
            "properties": {
                "monitor": {
                    "type": "integer",
                    "description": "0 = all monitors, 1 = primary, 2 = secondary",
                    "default": 0,
                },
                "client_id": {
                    "type": "string",
                    "description": "Optional target client id; defaults to the origin client",
                },
                "resize_factor": {
                    "type": "number",
                    "description": "Scale factor applied on the client",
                    "default": 0.5,
                },
            },
            "required": [],
        },
    )
    registry.register_function(
        func=analyze_dynamic_scene,
        name="analyze_dynamic_scene",
        description=(
            "Capture several screenshots on a desktop client and analyze the "
            "sequence with DeepSeek vision."
        ),
        parameters={
            "type": "object",
            "properties": {
                "monitor": {
                    "type": "integer",
                    "description": "1 = primary, 2 = secondary, etc.",
                    "default": 1,
                },
                "interval_seconds": {
                    "type": "number",
                    "description": "Time between captures",
                    "default": 0.5,
                },
                "num_frames": {
                    "type": "integer",
                    "description": "Number of frames (max 10)",
                    "default": 5,
                },
                "query": {
                    "type": "string",
                    "description": "Question about the sequence",
                },
                "client_id": {
                    "type": "string",
                    "description": "Optional target client id",
                },
            },
            "required": [],
        },
    )
    registry.register_function(
        func=tts_speak_aloud,
        name="tts_speak_aloud",
        description=(
            "Synthesize speech with ElevenLabs and play it on a connected "
            "desktop client's speakers. Do not use during voice turns; the "
            "voice pipeline speaks the final answer automatically."
        ),
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to speak"},
                "client_id": {
                    "type": "string",
                    "description": "Optional target client id",
                },
            },
            "required": ["text"],
        },
    )
    registry.register_function(
        func=client_record_audio,
        name="client_record_audio",
        description=(
            "Record microphone audio on a desktop client and optionally "
            "transcribe it with Groq Whisper. Returns a local WAV path and text."
        ),
        parameters={
            "type": "object",
            "properties": {
                "duration": {
                    "type": "number",
                    "description": "Recording length in seconds (0.5-30)",
                    "default": 5.0,
                },
                "transcribe": {
                    "type": "boolean",
                    "description": "Transcribe the recording with Groq",
                    "default": True,
                },
                "client_id": {
                    "type": "string",
                    "description": "Optional target client id",
                },
            },
            "required": [],
        },
    )
    registry.register_function(
        func=client_notify,
        name="client_notify",
        description="Show a desktop notification on a connected client.",
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Notification body"},
                "title": {"type": "string", "description": "Notification title"},
                "client_id": {
                    "type": "string",
                    "description": "Optional target client id",
                },
            },
            "required": ["message"],
        },
    )
    logger.info("Remote client tools registered")
