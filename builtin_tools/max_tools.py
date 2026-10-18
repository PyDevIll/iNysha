"""MAX Messenger tools for MASTERMIND v2.

Provides agent‑accessible tools for:
- Sending files from the filesystem to MAX chats
- Downloading files from MAX chats by token or CDN URL
- Generating TTS audio and sending as voice message
- Sending plain text messages
"""

from __future__ import annotations

import json

from contextlib import suppress
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from lib.elevenlabs_tts import synthesize_to_file


# ----------------------------------------------------------------------
# Existing tools: max_send_file, max_download_file, max_send_voice
# (keep them as they are; included here for completeness)
# ----------------------------------------------------------------------

async def max_send_file(
    absolute_path: str,
    chat_id: str,
    caption: str = "",
) -> dict[str, Any]:
    """Send a file from the local filesystem to a MAX chat.

    Args:
        absolute_path: Absolute path to the file on the filesystem.
        chat_id: MAX chat ID (numeric string).
        caption: Optional caption for the document.

    Returns:
        dict: {"ok": True, "result": {...}} on success,
              {"ok": False, "error": "..."} on failure.
    """
    if not absolute_path:
        return {"ok": False, "error": "absolute_path is required"}

    path = Path(absolute_path).resolve()
    if not path.exists():
        return {"ok": False, "error": f"File not found: {absolute_path}"}
    if not path.is_file():
        return {"ok": False, "error": f"Not a regular file: {absolute_path}"}

    bot = _get_bot_from_registry()
    if bot is None:
        return {"ok": False, "error": "MAX bot not available"}

    logger.info(f"MAX tool: sending file {path} to chat_id={chat_id}")
    result = await bot.send_document(int(chat_id), str(path), caption=caption)
    return result


async def max_download_file(
    file_token: str,
    destination_dir: str,
    url: Optional[str] = None,
) -> dict[str, Any]:
    """Download a file from a MAX chat by file token (or CDN URL) to a local directory.

    Args:
        file_token: MAX file token (obtained from message attachment).
        destination_dir: Directory to save the downloaded file.
        url: Optional CDN URL (from attachment payload). If provided, bot.download_file
             will download directly from CDN without API authentication.

    Returns:
        dict: {"ok": True, "path": "...", "file_name": "..."} on success,
              {"ok": False, "error": "..."} on failure.
    """
    if not file_token:
        return {"ok": False, "error": "file_token is required"}
    if not destination_dir:
        return {"ok": False, "error": "destination_dir is required"}

    dest = Path(destination_dir).resolve()
    if not dest.exists():
        return {"ok": False, "error": f"Destination directory not found: {destination_dir}"}
    if not dest.is_dir():
        return {"ok": False, "error": f"Destination is not a directory: {destination_dir}"}

    bot = _get_bot_from_registry()
    if bot is None:
        return {"ok": False, "error": "MAX bot not available"}

    logger.info(f"MAX tool: downloading file_token={file_token} to {dest} (url={'provided' if url else 'none'})")
    result = await bot.download_file(file_token, str(dest), url=url)
    return result


async def max_send_voice(
    text: str,
    chat_id: str,
    lang: str = "ru",
    caption: str = "",
) -> dict[str, Any]:
    """Generate TTS audio with ElevenLabs and send it as a MAX voice message.

    ⚠️ TTS‑READINESS: 'text' should be plain speech — no code, no URLs, no
    excessive markup.  ElevenLabs auto-detects the language; ``lang`` is kept
    for backward compatibility and is not sent to the API.

    Args:
        text: Text to speak.
        chat_id: MAX chat ID (numeric string)
        lang: Language hint (unused; ElevenLabs auto-detects)
        caption: Optional caption for the voice message

    Returns:
        dict: {"ok": True, "result": {...}} on success
    """
    text = (text or "").strip()
    if not text:
        return {"ok": False, "error": "text is required"}

    # Step 1: Generate TTS audio with ElevenLabs.
    tts_result = await synthesize_to_file(text)
    if not tts_result.get("ok"):
        return tts_result

    audio_path = Path(tts_result["path"])

    # Step 2: Get bot instance
    bot = _get_bot_from_registry()
    if bot is None:
        with suppress(OSError):
            audio_path.unlink(missing_ok=True)
        return {"ok": False, "error": "MAX bot not available"}

    # Step 3: Send as voice.  The generated file is temporary; remove it after
    # the upload/send completes.
    try:
        send_result = await bot.send_voice(
            chat_id=int(chat_id),
            voice_path=str(audio_path),
            caption=caption if caption else None,
        )
    except Exception as exc:
        logger.exception("MAX voice send failed")
        return {"ok": False, "error": str(exc)}
    finally:
        with suppress(OSError):
            audio_path.unlink(missing_ok=True)

    if send_result.get("ok"):
        logger.info(
            f"Voice sent to chat {chat_id} via ElevenLabs "
            f"({tts_result.get('voice_id')}): {text[:50]!r}..."
        )
        _mark_chat_replied(chat_id)

    return send_result


# ----------------------------------------------------------------------
# NEW TOOL: max_send_message
# ----------------------------------------------------------------------

async def max_send_message(
    chat_id: str,
    text: str,
    format: str = "markdown",  # "markdown" or "plain"
) -> dict[str, Any]:
    """Send a plain text message to a MAX chat.

    Use this to send a text message to any chat by its numeric ID.

    Args:
        chat_id: MAX chat ID (numeric string, e.g., "123456789").
        text: The message text to send.
        format: "markdown" (default) or "plain". If Markdown, MAX will render bold, italic, etc.

    Returns:
        dict: {"ok": True, "result": {...}} on success,
              {"ok": False, "error": "..."} on failure.
    """
    if not chat_id:
        return {"ok": False, "error": "chat_id is required"}
    if not text:
        return {"ok": False, "error": "text is required"}

    bot = _get_bot_from_registry()
    if bot is None:
        return {"ok": False, "error": "MAX bot not available"}

    # Prepare extra parameters
    extra = {}
    if format == "markdown":
        extra["format"] = "markdown"

    logger.info(f"MAX tool: sending message to chat_id={chat_id}: {text[:50]}...")
    result = await bot._api.send_message_to_chat(int(chat_id), text, extra)
    if result:
        _mark_chat_replied(chat_id)
        return {"ok": True, "result": result}
    else:
        return {"ok": False, "error": "send failed"}


async def max_get_messages(
    chat_id: str,
    count: int = 10,
) -> dict[str, Any]:
    """Retrieve last N messages from a MAX chat by chat ID.

    Args:
        chat_id: MAX chat ID (numeric string, e.g., "123456789").
        count: Number of recent messages to fetch (default 10, max 50).

    Returns:
        dict with:
            ok: bool
            messages: list of messages (each with text, sender, timestamp, etc.)
            error: optional error description.
    """
    if not chat_id:
        return {"ok": False, "error": "chat_id is required"}
    if count < 1 or count > 50:
        count = min(max(count, 1), 50)

    bot = _get_bot_from_registry()
    if bot is None:
        return {"ok": False, "error": "MAX bot not available"}

    try:
        messages = await bot._api.get_messages(int(chat_id), count=count)
        # Format messages into a readable string (the agent gets a string anyway)
        if not messages:
            return {"ok": True, "messages": [], "text": "No messages found."}
        # Build a human-readable summary
        lines = []
        for msg in messages:
            body = msg.get('body', {})
            logger.debug(f"[MAX] Raw message: {json.dumps(msg, ensure_ascii=False, default=str)}")
            if 'attachments' in body:
                for att in body['attachments']:
                    logger.debug(f"[MAX] Attachment type={att.get('type')} payload={att.get('payload')}")
            text = body.get('text', '')
            sender = msg.get('sender', {})
            sender_name = sender.get('name') or sender.get('first_name', 'Unknown')
            timestamp = msg.get('timestamp', 0)
            # Convert timestamp to datetime (assuming milliseconds)
            if timestamp:
                from datetime import datetime
                dt = datetime.fromtimestamp(timestamp / 1000)
                time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                time_str = 'unknown time'
            lines.append(f"[{time_str}] {sender_name}: {text[:200]}")
        text = "\n".join(lines)
        return {"ok": True, "messages": messages, "text": text}
    except Exception as e:
        logger.error(f"get_messages error: {e}")
        return {"ok": False, "error": str(e)}
# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------

def _get_bot_from_registry():
    from tool_registry import get_registry
    registry = get_registry()
    bot = registry.get_bot()
    if bot is None:
        logger.error("MAX bot not found in registry")
    return bot


def _mark_chat_replied(chat_id) -> None:
    """Record that this turn already answered a chat (avoids a duplicate final reply)."""
    try:
        from server.context import get_request_context
    except Exception:
        return
    ctx = get_request_context()
    if ctx is not None:
        ctx.sent_chats.add(str(chat_id))


# ----------------------------------------------------------------------
# Tool definitions
# ----------------------------------------------------------------------

TOOL_DEFINITIONS: list[tuple] = [
    (
        "max_send_file",
        max_send_file,
        "Send a file from the local filesystem to a MAX chat by absolute path. "
        "Use the chat_id from the user's message (e.g., `Current chat_id: 12345`).",
        {
            "type": "object",
            "properties": {
                "absolute_path": {"type": "string", "description": "Absolute path to the file"},
                "chat_id": {"type": "string", "description": "MAX chat ID (numeric string)"},
                "caption": {"type": "string", "description": "Optional caption for the file"},
            },
            "required": ["absolute_path", "chat_id"],
        },
    ),
    (
        "max_download_file",
        max_download_file,
        "Download a file from a MAX chat by file token (or CDN URL) to a local directory",
        {
            "type": "object",
            "properties": {
                "file_token": {"type": "string", "description": "MAX file token to download"},
                "destination_dir": {"type": "string", "description": "Directory to save the downloaded file"},
                "url": {"type": "string", "description": "Optional CDN URL (from attachment payload). If provided, download directly via CDN."},
            },
            "required": ["file_token", "destination_dir"],
        },
    ),
    (
        "max_send_voice",
        max_send_voice,
        "Generate TTS with ElevenLabs and send it as a voice message to a MAX chat. "
        "⚠️ text MUST be TTS‑ready: no code, no URLs, no excessive markup. "
        "ElevenLabs auto-detects the language.",
        {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to speak. Plain speech only.",
                },
                "chat_id": {
                    "type": "string",
                    "description": "MAX chat ID (numeric string)",
                },
                "lang": {
                    "type": "string",
                    "description": "Language hint (unused; ElevenLabs auto-detects)",
                },
                "caption": {
                    "type": "string",
                    "description": "Optional caption for the voice message",
                },
            },
            "required": ["text", "chat_id"],
        },
    ),
    # NEW: max_send_message
    (
        "max_send_message",
        max_send_message,
        "Send a plain text message to a MAX chat by chat ID. Use markdown formatting by default.",
        {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "string",
                    "description": "MAX chat ID (numeric string, e.g., '123456789')",
                },
                "text": {
                    "type": "string",
                    "description": "The message text to send.",
                },
                "format": {
                    "type": "string",
                    "description": "Format: 'markdown' (default) or 'plain'.",
                    "enum": ["markdown", "plain"],
                },
            },
            "required": ["chat_id", "text"],
        },
    ),
    (
        "max_get_messages",
        max_get_messages,
        "Retrieve the last N messages from a MAX chat by chat ID. "
        "Returns a human‑readable list of messages with sender, timestamp, and text. "
        "Useful for catching up on conversation history or analyzing recent activity.",
        {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "string",
                    "description": "MAX chat ID (numeric string, e.g., '123456789')",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of recent messages to fetch (1–50, default 10)",
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["chat_id"],
        },
    ),
]


def register_all(registry) -> None:
    """Register all MAX tools with the given registry."""
    for name, func, desc, params in TOOL_DEFINITIONS:
        registry.register_function(func, name, desc, params)
    logger.info(f"Registered {len(TOOL_DEFINITIONS)} MAX tools")
