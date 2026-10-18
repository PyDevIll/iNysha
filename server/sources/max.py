"""MAX messenger source: webhook -> parse/download/transcribe -> queue."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi import Request as HttpRequest
from loguru import logger

from server import config
from server.audio import is_likely_hallucination, transcribe_bytes
from server.queue import Request
from server.routing import MaxReply
from server.runtime import runtime

router = APIRouter(tags=["max"])

# Webhook retries are common; remember recent updates for a short while.
_seen: OrderedDict[str, float] = OrderedDict()
_SEEN_MAX = 1024
_background: set[asyncio.Task] = set()


def _check_secret(request: HttpRequest) -> None:
    if not config.MAX_WEBHOOK_SECRET:
        return
    provided = request.headers.get("x-inysha-secret")
    if provided != config.MAX_WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="invalid webhook secret")


def _update_key(update: dict[str, Any]) -> str:
    message = update.get("message") or {}
    body = message.get("body") or {}
    mid = body.get("mid") or message.get("mid") or update.get("update_id")
    if mid:
        return f"id:{mid}"
    payload = json.dumps(update, sort_keys=True, ensure_ascii=False, default=str)
    return "sha:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_new(key: str) -> bool:
    if key in _seen:
        return False
    _seen[key] = time.time()
    while len(_seen) > _SEEN_MAX:
        _seen.popitem(last=False)
    return True


@router.post("/inysha/max-webhook")
@router.post("/max-webhook")
async def max_webhook(request: HttpRequest):
    """Receive a MAX update forwarded by the Yandex Cloud Function."""
    _check_secret(request)

    try:
        update = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid JSON: {exc}")

    if not isinstance(update, dict):
        raise HTTPException(status_code=400, detail="update must be an object")

    key = _update_key(update)
    if not _is_new(key):
        logger.info(f"MAX duplicate update ignored ({key[:24]})")
        return {"ok": True, "duplicate": True}

    # Return to MAX immediately; parsing/transcription/downloads happen in the
    # background and the actual agent turn is serialized through the queue.
    task = asyncio.create_task(_handle_update(update), name=f"max-{key[:16]}")
    _background.add(task)
    task.add_done_callback(_background.discard)
    return {"ok": True}


async def handle_update(update: dict[str, Any]) -> bool:
    """Deduplicate and process one update (shared by webhook and poller)."""
    key = _update_key(update)
    if not _is_new(key):
        return False
    await _process_update(update)
    return True


async def _handle_update(update: dict[str, Any]) -> None:
    try:
        await _process_update(update)
    except Exception:
        logger.exception("MAX update processing failed")


def _extract(update: dict[str, Any]) -> dict[str, Any]:
    message = update.get("message") or {}
    body = message.get("body") or {}

    chat_id = update.get("chat_id")
    if not chat_id:
        recipient = message.get("recipient") or {}
        chat_id = recipient.get("chat_id")
    if not chat_id:
        chat_id = update.get("user_id")

    sender = message.get("sender") or {}
    user_id = sender.get("user_id") or update.get("user_id")
    sender_name = sender.get("name") or sender.get("first_name") or ""

    return {
        "chat_id": chat_id,
        "text": body.get("text") or "",
        "attachments": body.get("attachments") or [],
        "user_id": user_id,
        "sender_name": sender_name,
        "update_type": update.get("update_type") or "unknown",
    }


def _attachment_summary(attachment: dict[str, Any]) -> dict[str, Any]:
    """Small, log-safe description of an attachment (no tokens/URLs)."""
    payload = attachment.get("payload") or {}
    return {
        "type": attachment.get("type"),
        "payload_keys": sorted(payload.keys()),
        "size": payload.get("size"),
        "duration": payload.get("duration"),
        "filename": payload.get("filename"),
        "content_type": payload.get("content_type") or payload.get("mime_type"),
        "has_url": bool(payload.get("url")),
        "has_token": bool(payload.get("token")),
    }


async def _process_update(update: dict[str, Any]) -> None:
    data = _extract(update)
    chat_id = data["chat_id"]
    if not chat_id:
        if (
            data["update_type"] == "message_created"
            and not data["text"]
            and not data["attachments"]
        ):
            logger.warning(
                "MAX sent an empty message_created update. Native voice/video "
                "messages are currently lost by MAX long polling; switch "
                "MAX_TRANSPORT=webhook (the webhook delivers the full audio "
                "attachment). "
                + json.dumps(update, ensure_ascii=False)[:300]
            )
        else:
            logger.warning(
                "MAX update without chat_id: "
                + json.dumps(update, ensure_ascii=False)[:400]
            )
        return
    chat_id = str(chat_id)

    text: str = data["text"]
    user_id = data["user_id"]
    trusted = config.is_trusted(chat_id, str(user_id) if user_id else None)

    voice_transcripts: list[str] = []
    downloaded: list[str] = []
    attachment_types: list[str] = []
    notes: list[str] = []

    if data["attachments"]:
        logger.info(
            "MAX attachments: "
            + json.dumps(
                [_attachment_summary(a) for a in data["attachments"]],
                ensure_ascii=False,
            )
        )

    for attachment in data["attachments"]:
        att_type = attachment.get("type") or "unknown"
        attachment_types.append(att_type)
        payload = attachment.get("payload") or {}
        token = payload.get("token") or ""
        url = payload.get("url")

        if att_type == "inline_keyboard":
            # UI metadata, not message content.
            continue

        limit = (
            config.MAX_AUDIO_MAX_BYTES
            if att_type in ("audio", "voice")
            else config.MAX_ATTACHMENT_MAX_BYTES
        )

        declared_size = payload.get("size")
        if declared_size not in (None, ""):
            try:
                declared_size = int(declared_size)
            except (TypeError, ValueError):
                declared_size = None
            if declared_size is not None and declared_size > limit:
                notes.append(
                    f"{att_type}: skipped, size {declared_size} bytes "
                    f"exceeds limit {limit} bytes"
                )
                continue

        if att_type in ("audio", "voice"):
            result = await _download_attachment(token, url, max_bytes=limit)
            if not result.get("ok"):
                notes.append(
                    f"{att_type}: download failed: "
                    f"{result.get('error') or 'unknown error'}"
                )
                continue

            path = result["path"]
            try:
                size = Path(path).stat().st_size
            except OSError:
                size = 0
            if size > config.MAX_AUDIO_MAX_BYTES:
                notes.append(
                    f"{att_type}: file is {size} bytes, exceeds transcription "
                    f"limit {config.MAX_AUDIO_MAX_BYTES} bytes"
                )
                continue

            try:
                audio = Path(path).read_bytes()
                transcript = await transcribe_bytes(
                    audio, filename=Path(path).name
                )
            except Exception as exc:
                logger.exception("MAX voice transcription failed")
                notes.append(f"{att_type}: transcription failed: {exc}")
                continue

            if not transcript:
                logger.warning(
                    f"MAX voice message produced empty transcript: {path}"
                )
                notes.append(
                    f"{att_type}: transcription returned no text"
                )
                continue

            if is_likely_hallucination(transcript):
                logger.warning(
                    "MAX voice transcript looks like silence/hallucination: "
                    f"{transcript[:120]!r}"
                )
                notes.append(
                    f"{att_type}: transcript looked like silence or a "
                    "Whisper hallucination and was ignored"
                )
                continue

            logger.info(
                f"MAX voice transcribed ({len(transcript)} chars): "
                f"{transcript[:120]!r}"
            )
            voice_transcripts.append(transcript)
            continue

        if att_type in ("file", "image", "video"):
            result = await _download_attachment(token, url, max_bytes=limit)
            if result.get("ok"):
                downloaded.append(result["path"])
            else:
                notes.append(
                    f"{att_type}: download failed: "
                    f"{result.get('error') or 'unknown error'}"
                )
            continue

        notes.append(f"{att_type}: attachment type not processed")

    parts: list[str] = []
    if text:
        parts.append(text)
    if voice_transcripts:
        parts.append(
            "[Voice Message Transcript]\n" + "\n\n".join(voice_transcripts)
        )
    if downloaded:
        parts.append(
            "[Downloaded attachments]\n"
            + "\n".join(f"- {path}" for path in downloaded)
        )
    if notes:
        parts.append(
            "[Attachment processing notes]\n"
            + "\n".join(f"- {note}" for note in notes)
        )

    if parts:
        content = "\n\n".join(parts)
    else:
        content = (
            f"Received update of type '{data['update_type']}':\n"
            + json.dumps(update, ensure_ascii=False, indent=2)
        )

    header = f"[MAX messenger | chat_id={chat_id}"
    if user_id:
        header += f" | user_id={user_id}"
    if data["sender_name"]:
        header += f" | sender={data['sender_name']}"
    header += (
        f" | attachments={','.join(attachment_types) if attachment_types else 'none'}"
    )
    header += f" | trusted={'true' if trusted else 'false'}]"

    request = Request(
        text=f"{header}\n{content}",
        source="max",
        chat_id=chat_id,
        trusted=trusted,
        send_reasoning=config.MAX_SEND_REASONING,
        reply=MaxReply(runtime.bot, chat_id),
    )
    await runtime.queue.put(request)
    logger.info(
        f"[{request.id}] queued MAX message from chat {chat_id} "
        f"(trusted={trusted}, {len(content)} chars)"
    )


async def _download_attachment(
    token: str,
    url: str | None,
    *,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    if not token and not url:
        return {"ok": False, "error": "attachment has neither token nor url"}
    try:
        result = await runtime.bot.download_file(
            token,
            str(config.DOWNLOADS_DIR),
            url=url,
            max_bytes=max_bytes,
        )
    except Exception as exc:
        logger.exception("MAX attachment download failed")
        return {"ok": False, "error": str(exc)}
    if result.get("ok"):
        return result
    logger.error(f"MAX attachment download failed: {result.get('error')}")
    return {
        "ok": False,
        "error": result.get("error") or "unknown error",
        "too_large": result.get("too_large", False),
    }
