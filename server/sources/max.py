"""MAX messenger source: webhook -> parse/download/transcribe -> per-chat batch.

Каждое обновление разбирается сразу (скачивание вложений, транскрипция), но
результат не уходит в очередь немедленно: текст копится в буфере своего чата,
и через ``MAX_DEFER_SECONDS`` тишины все накопленные сообщения чата уходят
одним ``Request``. Разные чаты никогда не склеиваются.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
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


@dataclass
class _ChatBatch:
    """Сообщения одного чата, ещё не отданные воркеру."""

    messages: list[str] = field(default_factory=list)
    chars: int = 0
    # Доверие считается по каждому сообщению отдельно: в группе чужое
    # сообщение не должно получить привилегии из-за соседа в пачке.
    trust_flags: list[bool] = field(default_factory=list)
    timer: asyncio.TimerHandle | None = None

    @property
    def trusted(self) -> bool:
        return bool(self.trust_flags) and all(self.trust_flags)


# chat_id -> активный буфер сообщений; чаты независимы друг от друга.
_batches: dict[str, _ChatBatch] = {}
# Rate-limit: chat_id -> список меток времени последних сообщений.
_rate: dict[str, list[float]] = {}
# chat_id -> полные пачки, ждущие, пока освободится занятый чат.
_backlog: dict[str, list[_ChatBatch]] = {}
# Чаты, для которых запрос уже стоит в очереди или обрабатывается воркером.
_inflight: set[str] = set()


def _reset_state() -> None:
    """Полностью очистить буферы и дедупликацию (используется в тестах)."""
    for batch in _batches.values():
        if batch.timer is not None:
            batch.timer.cancel()
    for task in list(_background):
        task.cancel()
    _background.clear()
    _batches.clear()
    _backlog.clear()
    _inflight.clear()
    _seen.clear()
    _rate.clear()
    global _update_semaphore, _semaphore_loop
    _update_semaphore = None
    _semaphore_loop = None


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
    # background, then the per-chat debounce batches them and the actual agent
    # turn is serialized through the queue.
    task = asyncio.create_task(_handle_update(update), name=f"max-{key[:16]}")
    _background.add(task)
    task.add_done_callback(_background.discard)
    return {"ok": True}


async def handle_update(update: dict[str, Any]) -> bool:
    """Deduplicate and process one update (shared by webhook and poller).

    Идёт через те же защиты, что и webhook: rate-limit на чат и семафор.
    Без этого поллер (``MAX_TRANSPORT=poll``) полностью обходил защиту.
    """
    key = _update_key(update)
    if not _is_new(key):
        return False
    chat_id = _extract(update).get("chat_id")
    try:
        if chat_id:
            _check_rate_limit(str(chat_id))
    except _RateLimited:
        await _defer_without_attachments(update)
        return True
    async with _get_semaphore():
        await _process_update(update)
    return True


# ── Защита от лавины: глобальный семафор + rate-limit на чат ────────────────
# Семафор ограничивает число одновременных разборов апдейтов (скачивание
# вложений и распознавание голоса идут ДО батчинга и стоят дорого).
_update_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: asyncio.AbstractEventLoop | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Ленивая инициализация: семафор привязан к текущему event loop.

    Пересоздаётся, только если сменился loop (например, между тестами):
    ``asyncio.Semaphore`` нельзя переиспользовать в другом цикле.
    """
    global _update_semaphore, _semaphore_loop
    loop = asyncio.get_running_loop()
    limit = max(1, int(config.MAX_CONCURRENT_UPDATES))
    if _update_semaphore is None or _semaphore_loop is not loop:
        _update_semaphore = asyncio.Semaphore(limit)
        _semaphore_loop = loop
    return _update_semaphore


def _check_rate_limit(chat_id: str) -> bool:
    """True, если чат не превысил лимит сообщений в окне."""
    limit = int(config.MAX_CHAT_RATE_LIMIT)
    if limit <= 0:
        return True
    window = float(config.MAX_CHAT_RATE_WINDOW)
    now = time.monotonic()
    stamps = _rate.setdefault(chat_id, [])
    while stamps and now - stamps[0] > window:
        stamps.pop(0)
    if len(stamps) >= limit:
        # Не выбрасываем сообщение: оно ещё не разобрано, но является частью
        # разговора. Разбор откладываем (без вложений) — так теряется только
        # текстовая часть, а не весь апдейт. Пишем в лог один раз в окно.
        logger.warning(
            f"MAX rate limit: чат {chat_id} превысил {limit} сообщ. за "
            f"{window:.0f} с — разбор отложен"
        )
        raise _RateLimited()
    stamps.append(now)
    return True


async def _handle_update(update: dict[str, Any]) -> None:
    chat_id = _extract(update).get("chat_id")
    try:
        if chat_id:
            _check_rate_limit(str(chat_id))
    except _RateLimited:
        # Лимит исчерпан: текст всё равно попадает в пачку (без скачивания
        # вложений и без распознавания голоса), чтобы разговор не рвался.
        await _defer_without_attachments(update)
        return
    async with _get_semaphore():
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


def _has_pending(chat_id: str) -> bool:
    """Есть ли у чата что отдавать: активный буфер или готовый backlog."""
    batch = _batches.get(chat_id)
    return bool(_backlog.get(chat_id)) or bool(batch and batch.messages)


def _schedule_flush(chat_id: str, delay: float | None = None) -> None:
    """(Пере)завести таймер тишины для чата."""
    batch = _batches.get(chat_id)
    if batch is None:
        if not _backlog.get(chat_id):
            return
        batch = _batches[chat_id] = _ChatBatch()
    if batch.timer is not None:
        batch.timer.cancel()
        batch.timer = None
    if not _has_pending(chat_id):
        return
    if delay is None:
        delay = config.MAX_DEFER_SECONDS
    loop = asyncio.get_running_loop()
    batch.timer = loop.call_later(max(0.0, delay), _on_flush_timer, chat_id)


def _on_flush_timer(chat_id: str) -> None:
    batch = _batches.get(chat_id)
    if batch is None:
        if not _backlog.get(chat_id):
            return
        batch = _batches[chat_id] = _ChatBatch()
    batch.timer = None
    if not _has_pending(chat_id):
        return
    task = asyncio.create_task(
        _flush_batch_safe(chat_id, reason="тишина"), name=f"max-batch-{chat_id}"
    )
    _background.add(task)
    task.add_done_callback(_background.discard)


async def _flush_batch_safe(chat_id: str, reason: str) -> None:
    try:
        await _flush_batch(chat_id, reason)
    except Exception:
        logger.exception(f"MAX: не удалось отправить пачку чата {chat_id}")


def _append_to_batch(batch: _ChatBatch, text: str, trusted: bool) -> None:
    """Дописать сообщение в пачку, поддерживая счётчик символов в O(1)."""
    if batch.messages:
        # Сообщения склеиваются через пустую строку — два символа-разделителя.
        batch.chars += len(text) + 2
    else:
        batch.chars = len(text)
    batch.messages.append(text)
    batch.trust_flags.append(bool(trusted))


def _batch_would_overflow(batch: _ChatBatch, text: str) -> bool:
    """Не выйдет ли пачка за лимиты, если дописать ещё одно сообщение."""
    return (
        len(batch.messages) + 1 > config.MAX_BATCH_MESSAGES
        or batch.chars + len(text) + 2 > config.MAX_BATCH_CHARS
    )


def _park_batch(chat_id: str, batch: _ChatBatch) -> None:
    """Убрать пачку в backlog чата, не оставив висящий таймер тишины."""
    if batch.timer is not None:
        batch.timer.cancel()
        batch.timer = None
    _backlog.setdefault(chat_id, []).append(batch)


async def _buffer_message(chat_id: str, text: str, trusted: bool) -> None:
    """Добавить готовый текст в пачку чата; переполнение сбрасывается сразу."""
    batch = _batches.get(chat_id)
    if batch is None:
        batch = _batches[chat_id] = _ChatBatch()

    # Пока по этому чату ещё отвечает воркер, отдать пачку нельзя: копим
    # сообщения, но не даём пачке выйти за лимиты — полная уходит в backlog
    # и будет отправлена следующей, как только чат освободится.
    if chat_id in _inflight:
        if batch.messages and _batch_would_overflow(batch, text):
            _park_batch(chat_id, batch)
            batch = _batches[chat_id] = _ChatBatch()
        _append_to_batch(batch, text, trusted)
        logger.debug(
            f"MAX: чат {chat_id} занят, {len(batch.messages)} сообщ. "
            "ждут следующей пачки"
        )
        if (
            len(batch.messages) >= config.MAX_BATCH_MESSAGES
            or batch.chars >= config.MAX_BATCH_CHARS
        ):
            _park_batch(chat_id, batch)
            _batches[chat_id] = _ChatBatch()
        return

    # Не влезает в текущую пачку — сначала отдаём её, остаток копим заново.
    if batch.messages and _batch_would_overflow(batch, text):
        logger.info(
            f"MAX: пачка чата {chat_id} заполнена "
            f"({len(batch.messages)} сообщ., {batch.chars} симв.), "
            "сбрасываю её до лимита"
        )
        await _flush_batch(chat_id, reason="переполнение")
        # После успешного сброса пачки в буфере её уже нет — заводим новую.
        if _batches.get(chat_id) is None:
            _batches[chat_id] = _ChatBatch()
        batch = _batches[chat_id]

    _append_to_batch(batch, text, trusted)

    if (
        len(batch.messages) >= config.MAX_BATCH_MESSAGES
        or batch.chars >= config.MAX_BATCH_CHARS
    ):
        logger.info(
            f"MAX: пачка чата {chat_id} достигла лимита "
            f"({len(batch.messages)} сообщ., {batch.chars} симв.), "
            "сбрасываю сразу"
        )
        await _flush_batch(chat_id, reason="переполнение")
        return

    if chat_id not in _inflight:
        _schedule_flush(chat_id)


def _take_next_batch(chat_id: str) -> _ChatBatch | None:
    """Следующая пачка: сначала backlog (порядок), затем активный буфер."""
    backlog = _backlog.get(chat_id)
    if backlog:
        batch = backlog.pop(0)
        if not backlog:
            _backlog.pop(chat_id, None)
        return batch
    batch = _batches.pop(chat_id, None)
    if batch is None or not batch.messages:
        return None
    return batch


async def _flush_batch(chat_id: str, reason: str = "тишина") -> Request | None:
    """Отдать накопленную пачку чата в очередь одним ``Request``."""
    batch = _batches.get(chat_id)
    if batch is not None and batch.timer is not None:
        batch.timer.cancel()
        batch.timer = None

    if chat_id in _inflight:
        # Предыдущий запрос этого чата ещё в очереди или в работе: новые
        # сообщения остаются в буфере и станут следующими пачками.
        pending = len(batch.messages) if batch is not None else 0
        pending += sum(len(item.messages) for item in _backlog.get(chat_id, []))
        if pending:
            logger.debug(
                f"MAX: чат {chat_id} занят, {pending} сообщ. "
                "ждут следующей пачки"
            )
            _schedule_flush(chat_id)
        return None

    batch = _take_next_batch(chat_id)
    if batch is None:
        _batches.pop(chat_id, None)
        return None

    # Копия списка: пока идёт await ниже, в batch.messages могут дописаться
    # новые сообщения, и они не должны попасть в уже собранный текст.
    messages = list(batch.messages)
    trusted = batch.trusted
    batch.messages.clear()
    batch.trust_flags.clear()
    batch.chars = 0
    _inflight.add(chat_id)

    text = "\n\n".join(messages)
    request = Request(
        text=text,
        source="max",
        chat_id=chat_id,
        trusted=trusted,
        send_reasoning=config.MAX_SEND_REASONING,
        reply=_BatchReply(MaxReply(runtime.bot, chat_id), chat_id),
    )
    try:
        await runtime.queue.put(request)
    except Exception:
        # Порядок важен: неудавшаяся пачка старше всего, что уже в backlog.
        # Сообщения возвращаем обратно в batch, чтобы ничего не потерялось.
        batch.messages[0:0] = messages
        batch.trust_flags[0:0] = [trusted] * len(messages)
        batch.chars = sum(len(m) for m in batch.messages) + 2 * (len(batch.messages) - 1)
        _inflight.discard(chat_id)
        _backlog.setdefault(chat_id, []).insert(0, batch)
        _schedule_flush(chat_id)
        raise

    logger.info(
        f"[{request.id}] пачка MAX из чата {chat_id} поставлена в очередь "
        f"({len(messages)} сообщ., {len(text)} симв., причина={reason}, "
        f"trusted={trusted})"
    )
    return request


def _finish_batch(chat_id: str) -> None:
    """Ответ по пачке отправлен — чат свободен, буфер можно отдавать."""
    _inflight.discard(chat_id)
    if _has_pending(chat_id):
        _schedule_flush(chat_id, delay=0.0)


class _BatchReply:
    """``MaxReply``, который освобождает чат после отправки ответа."""

    def __init__(self, inner: MaxReply, chat_id: str) -> None:
        self._inner = inner
        self._chat_id = chat_id

    async def send_text(self, text: str) -> None:
        try:
            await self._inner.send_text(text)
        finally:
            _finish_batch(self._chat_id)

    async def send_error(self, text: str) -> None:
        try:
            await self._inner.send_error(text)
        finally:
            _finish_batch(self._chat_id)


class _RateLimited(Exception):
    """Чат превысил лимит сообщений в окне — разбор нужно отложить."""


async def _defer_without_attachments(update: dict[str, Any]) -> None:
    """Положить в пачку только текст сообщения, без обработки вложений."""
    data = _extract(update)
    chat_id = data["chat_id"]
    text = (data.get("text") or "").strip()
    if not chat_id or not text:
        return
    header = (
        f"[MAX messenger | chat_id={chat_id} | user_id={data.get('user_id')} "
        f"| sender={data.get('sender_name')} | attachments=none "
        f"| trusted={data.get('trusted')}]"
    )
    await _buffer_message(chat_id, f"{header}\n{text}", data.get("trusted", False))


async def _process_update(update: dict[str, Any]) -> None:
    prepared = await _prepare_message(update)
    if prepared is None:
        return
    chat_id, message_text, trusted = prepared
    await _buffer_message(chat_id, message_text, trusted)


async def _prepare_message(
    update: dict[str, Any],
) -> tuple[str, str, bool] | None:
    """Разобрать обновление в готовый текст сообщения, не отправляя его."""
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
        return None
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

    return chat_id, f"{header}\n{content}", trusted


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
