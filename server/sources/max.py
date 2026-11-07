"""MAX messenger source: webhook -> parse/download/transcribe -> per-chat batch.

Каждое обновление разбирается сразу (скачивание вложений, транскрипция), но
результат не уходит в очередь немедленно: текст копится в буфере своего чата
(а для комментариев канала — в буфере своего поста), и через
``MAX_DEFER_SECONDS`` тишины все накопленные сообщения уходят одним
``Request``. Разные чаты и посты никогда не склеиваются.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
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
    """Сообщения одного треда (чат или пост канала), ещё не отданные воркеру."""

    # Чат, куда уйдёт ответ. Для комментария канала это id канала.
    chat_id: str = ""
    # Пост канала, если сообщения пришли из комментариев под ним.
    post_id: str | None = None
    messages: list[str] = field(default_factory=list)
    chars: int = 0
    # Доверие считается по каждому сообщению отдельно: в группе чужое
    # сообщение не должно получить привилегии из-за соседа в пачке.
    trust_flags: list[bool] = field(default_factory=list)
    timer: asyncio.TimerHandle | None = None

    @property
    def trusted(self) -> bool:
        return bool(self.trust_flags) and all(self.trust_flags)


def _thread_key(chat_id: str, post_id: str | None) -> str:
    """Ключ буфера: обычный чат или обсуждение конкретного поста канала.

    Комментарии разных постов нельзя склеивать в одну пачку: ответ уходит
    ``POST /messages/{post_id}/comments`` и у него может быть только один
    ``post_id``.
    """
    chat_id = str(chat_id)
    return f"{chat_id}#{post_id}" if post_id else chat_id


# thread key (chat_id или "chat_id#post_id") -> активный буфер сообщений.
_batches: dict[str, _ChatBatch] = {}
# Rate-limit: chat_id -> список меток времени последних сообщений.
_rate: dict[str, list[float]] = {}
# thread key -> полные пачки, ждущие, пока освободится занятый тред.
_backlog: dict[str, list[_ChatBatch]] = {}
# Треды, для которых запрос уже стоит в очереди или обрабатывается воркером.
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
    provided = request.headers.get(config.SECRET_HEADER)
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


@router.post(config.MAX_WEBHOOK_PATH)
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


if (
    config.MAX_WEBHOOK_ALIAS_PATH
    and config.MAX_WEBHOOK_ALIAS_PATH != config.MAX_WEBHOOK_PATH
):
    router.add_api_route(
        config.MAX_WEBHOOK_ALIAS_PATH,
        max_webhook,
        methods=["POST"],
        include_in_schema=False,
        name="max_webhook_alias",
    )


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
    # События-отметки (удаление, правка, вход/выход) не тратят ни семафор,
    # ни rate-limit: LLM-запроса по ним всё равно не будет.
    if _collect_event(update):
        return
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
    recipient = message.get("recipient") or {}

    chat_id = update.get("chat_id")
    if not chat_id:
        chat_id = recipient.get("chat_id")
    if not chat_id:
        chat_id = update.get("user_id")

    # Комментарий канала: recipient.post_id — это mid поста, а не комментария.
    # Именно он нужен для POST /messages/{post_id}/comments.
    post_id = recipient.get("post_id") or update.get("post_id")
    message_id = body.get("mid") or message.get("mid") or update.get("message_id")

    sender = message.get("sender") or {}
    user_id = sender.get("user_id") or update.get("user_id")
    sender_name = sender.get("name") or sender.get("first_name") or ""

    return {
        "chat_id": chat_id,
        "post_id": str(post_id) if post_id not in (None, "") else None,
        "message_id": str(message_id) if message_id not in (None, "") else None,
        "chat_type": recipient.get("chat_type"),
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


def _new_batch(chat_id: str, post_id: str | None) -> _ChatBatch:
    return _ChatBatch(chat_id=str(chat_id), post_id=post_id)


def _has_pending(key: str) -> bool:
    """Есть ли у треда что отдавать: активный буфер или готовый backlog."""
    batch = _batches.get(key)
    return bool(_backlog.get(key)) or bool(batch and batch.messages)


def _schedule_flush(key: str, delay: float | None = None) -> None:
    """(Пере)завести таймер тишины для треда."""
    batch = _batches.get(key)
    if batch is None:
        backlog = _backlog.get(key) or []
        if not backlog:
            return
        # Заглушка для таймера: при сбросе первым всё равно возьмётся backlog.
        ref = backlog[0]
        batch = _batches[key] = _new_batch(ref.chat_id, ref.post_id)
    if batch.timer is not None:
        batch.timer.cancel()
        batch.timer = None
    if not _has_pending(key):
        return
    if delay is None:
        delay = config.MAX_DEFER_SECONDS
    loop = asyncio.get_running_loop()
    batch.timer = loop.call_later(max(0.0, delay), _on_flush_timer, key)


def _on_flush_timer(key: str) -> None:
    batch = _batches.get(key)
    if batch is None:
        backlog = _backlog.get(key) or []
        if not backlog:
            return
        ref = backlog[0]
        batch = _batches[key] = _new_batch(ref.chat_id, ref.post_id)
    batch.timer = None
    if not _has_pending(key):
        return
    task = asyncio.create_task(
        _flush_batch_safe(key, reason="тишина"), name=f"max-batch-{key}"
    )
    _background.add(task)
    task.add_done_callback(_background.discard)


async def _flush_batch_safe(key: str, reason: str) -> None:
    try:
        await _flush_batch(key, reason)
    except Exception:
        logger.exception(f"MAX: не удалось отправить пачку {key}")


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


def _park_batch(key: str, batch: _ChatBatch) -> None:
    """Убрать пачку в backlog треда, не оставив висящий таймер тишины."""
    if batch.timer is not None:
        batch.timer.cancel()
        batch.timer = None
    _backlog.setdefault(key, []).append(batch)


async def _buffer_message(
    chat_id: str,
    text: str,
    trusted: bool,
    post_id: str | None = None,
) -> None:
    """Добавить готовый текст в пачку треда; переполнение сбрасывается сразу."""
    chat_id = str(chat_id)
    key = _thread_key(chat_id, post_id)
    batch = _batches.get(key)
    if batch is None:
        batch = _batches[key] = _new_batch(chat_id, post_id)

    # Пока по этому треду ещё отвечает воркер, отдать пачку нельзя: копим
    # сообщения, но не даём пачке выйти за лимиты — полная уходит в backlog
    # и будет отправлена следующей, как только тред освободится.
    if key in _inflight:
        if batch.messages and _batch_would_overflow(batch, text):
            _park_batch(key, batch)
            batch = _batches[key] = _new_batch(chat_id, post_id)
        _append_to_batch(batch, text, trusted)
        logger.debug(
            f"MAX: тред {key} занят, {len(batch.messages)} сообщ. "
            "ждут следующей пачки"
        )
        if (
            len(batch.messages) >= config.MAX_BATCH_MESSAGES
            or batch.chars >= config.MAX_BATCH_CHARS
        ):
            _park_batch(key, batch)
            _batches[key] = _new_batch(chat_id, post_id)
        return

    # Не влезает в текущую пачку — сначала отдаём её, остаток копим заново.
    if batch.messages and _batch_would_overflow(batch, text):
        logger.info(
            f"MAX: пачка треда {key} заполнена "
            f"({len(batch.messages)} сообщ., {batch.chars} симв.), "
            "сбрасываю её до лимита"
        )
        await _flush_batch(key, reason="переполнение")
        # После успешного сброса пачки в буфере её уже нет — заводим новую.
        if _batches.get(key) is None:
            _batches[key] = _new_batch(chat_id, post_id)
        batch = _batches[key]

    _append_to_batch(batch, text, trusted)

    if (
        len(batch.messages) >= config.MAX_BATCH_MESSAGES
        or batch.chars >= config.MAX_BATCH_CHARS
    ):
        logger.info(
            f"MAX: пачка треда {key} достигла лимита "
            f"({len(batch.messages)} сообщ., {batch.chars} симв.), "
            "сбрасываю сразу"
        )
        await _flush_batch(key, reason="переполнение")
        return

    if key not in _inflight:
        _schedule_flush(key)


def _take_next_batch(key: str) -> _ChatBatch | None:
    """Следующая пачка: сначала backlog (порядок), затем активный буфер."""
    backlog = _backlog.get(key)
    if backlog:
        batch = backlog.pop(0)
        if not backlog:
            _backlog.pop(key, None)
        return batch
    batch = _batches.pop(key, None)
    if batch is None or not batch.messages:
        return None
    return batch


async def _flush_batch(key: str, reason: str = "тишина") -> Request | None:
    """Отдать накопленную пачку треда в очередь одним ``Request``."""
    batch = _batches.get(key)
    if batch is not None and batch.timer is not None:
        batch.timer.cancel()
        batch.timer = None

    if key in _inflight:
        # Предыдущий запрос этого треда ещё в очереди или в работе: новые
        # сообщения остаются в буфере и станут следующими пачками.
        pending = len(batch.messages) if batch is not None else 0
        pending += sum(len(item.messages) for item in _backlog.get(key, []))
        if pending:
            logger.debug(
                f"MAX: тред {key} занят, {pending} сообщ. "
                "ждут следующей пачки"
            )
            _schedule_flush(key)
        return None

    batch = _take_next_batch(key)
    if batch is None:
        _batches.pop(key, None)
        return None

    chat_id = batch.chat_id or key.split("#", 1)[0]
    post_id = batch.post_id
    # Копия списка: пока идёт await ниже, в batch.messages могут дописаться
    # новые сообщения, и они не должны попасть в уже собранный текст.
    messages = list(batch.messages)
    trusted = batch.trusted
    batch.messages.clear()
    batch.trust_flags.clear()
    batch.chars = 0
    _inflight.add(key)

    # Отметки о событиях (удаление, правка и т.п.) идут перед сообщениями:
    # агент видит, что успело случиться, но хода по ним не было.
    event_notes = _drain_events(chat_id)
    text = "\n\n".join(event_notes + messages)
    request = Request(
        text=text,
        source="max",
        chat_id=chat_id,
        post_id=post_id,
        trusted=trusted,
        send_reasoning=config.MAX_SEND_REASONING,
        reply=_BatchReply(MaxReply(runtime.bot, chat_id, post_id), key),
    )
    try:
        await runtime.queue.put(request)
    except Exception:
        # Порядок важен: неудавшаяся пачка старше всего, что уже в backlog.
        # Сообщения возвращаем обратно в batch, чтобы ничего не потерялось.
        batch.messages[0:0] = messages
        batch.trust_flags[0:0] = [trusted] * len(messages)
        batch.chars = sum(len(m) for m in batch.messages) + 2 * (len(batch.messages) - 1)
        _inflight.discard(key)
        _backlog.setdefault(key, []).insert(0, batch)
        _schedule_flush(key)
        raise

    logger.info(
        f"[{request.id}] пачка MAX из треда {key} поставлена в очередь "
        f"({len(messages)} сообщ., {len(text)} симв., причина={reason}, "
        f"chat_id={chat_id}, post_id={post_id}, trusted={trusted})"
    )
    return request


def _finish_batch(key: str) -> None:
    """Ответ по пачке отправлен — тред свободен, буфер можно отдавать."""
    _inflight.discard(key)
    if _has_pending(key):
        _schedule_flush(key, delay=0.0)


class _BatchReply:
    """``MaxReply``, который освобождает тред после отправки ответа."""

    def __init__(self, inner: MaxReply, key: str) -> None:
        self._inner = inner
        self._key = key

    async def send_text(self, text: str) -> None:
        try:
            await self._inner.send_text(text)
        finally:
            _finish_batch(self._key)

    async def send_error(self, text: str) -> None:
        try:
            await self._inner.send_error(text)
        finally:
            _finish_batch(self._key)


class _RateLimited(Exception):
    """Чат превысил лимит сообщений в окне — разбор нужно отложить."""


async def _defer_without_attachments(update: dict[str, Any]) -> None:
    """Положить в пачку только текст сообщения, без обработки вложений."""
    data = _extract(update)
    chat_id = data["chat_id"]
    text = (data.get("text") or "").strip()
    if not chat_id or not text:
        return
    user_id = data.get("user_id")
    trusted = config.is_trusted(
        str(chat_id), str(user_id) if user_id else None
    )
    header = f"[MAX messenger | chat_id={chat_id}"
    if data.get("post_id"):
        header += f" | post_id={data['post_id']} | channel_comment=true"
    header += (
        f" | user_id={data.get('user_id')} "
        f"| sender={data.get('sender_name')} | attachments=none "
        f"| trusted={'true' if trusted else 'false'}]"
    )
    await _buffer_message(
        chat_id,
        f"{header}\n{text}",
        trusted,
        post_id=data.get("post_id"),
    )


# События, на которые НЕ нужен ход агента: только отметка в контексте.
# Такие апдейты не батчатся и не порождают Request — они копятся в
# ``_event_notes`` и уходят в контекст как "[MAX event] ..." к следующему
# настоящему сообщению этого чата.
_EVENT_ONLY_TYPES = {
    "message_removed",
    "comment_removed",
    "message_edited",
    "comment_edited",
    "bot_added",
    "bot_removed",
    "user_added",
    "user_removed",
    "chat_title_changed",
    "bot_admin_permissions_changed",
    "dialog_cleared",
    "dialog_removed",
    "dialog_muted",
    "dialog_unmuted",
}

# chat_id -> список строк-отметок, ждущих следующего сообщения.
_event_notes: dict[str, list[str]] = {}
_EVENT_NOTES_MAX = 20


def _describe_event(update: dict[str, Any]) -> str | None:
    """Короткая однострочная отметка о событии, без LLM-запроса."""
    data = _extract(update)
    update_type = data.get("update_type") or "unknown"
    if update_type not in _EVENT_ONLY_TYPES:
        return None
    message = update.get("message") or {}
    body = message.get("body") or {}
    recipient = message.get("recipient") or {}
    user = message.get("sender") or {}

    message_id = data.get("message_id") or body.get("mid")
    user_id = data.get("user_id") or user.get("user_id")
    user_name = data.get("sender_name") or user.get("name") or "?"
    post_id = recipient.get("post_id") or update.get("post_id")
    title = update.get("title")

    bits = [f"[MAX event] {update_type}"]
    if message_id:
        bits.append(f"message_id={message_id}")
    if post_id:
        bits.append(f"post_id={post_id}")
    if user_id:
        bits.append(f"user_id={user_id}")
    if user_name and user_name != "?":
        bits.append(f"sender={user_name}")
    if title:
        bits.append(f"title={title}")
    return " | ".join(bits)


_NOT_A_MESSAGE = re.compile(
    r"^Received update of type '(?P<etype>[a-z_]+)':\s*(?P<body>\{.*\})\s*$",
    re.S,
)


def _collect_event(update: dict[str, Any]) -> bool:
    """Сохранить отметку о событии. True, если апдейт не требует хода агента.

    Yandex-функция пересылает не-сообщенческие апдейты **строкой** вида
    ``Received update of type 'message_removed': {...}`` в теле обычного
    ``message_created``. Поэтому настоящий ``update_type`` ищем и в поле,
    и в тексте.
    """
    data = _extract(update)
    note = _describe_event(update)
    if note is None:
        text = (data.get("text") or "").strip()
        match = _NOT_A_MESSAGE.match(text)
        if match:
            etype = match.group("etype")
            if etype in _EVENT_ONLY_TYPES:
                try:
                    inner = json.loads(match.group("body"))
                except json.JSONDecodeError:
                    inner = {}
                bits = [f"[MAX event] {etype}"]
                for key in ("message_id", "post_id", "user_id", "chat_id"):
                    if inner.get(key) is not None:
                        bits.append(f"{key}={inner[key]}")
                note = " | ".join(bits)
    if note is None:
        return False
    chat_id = data.get("chat_id")
    if not chat_id:
        logger.info(note)
        return True
    notes = _event_notes.setdefault(str(chat_id), [])
    notes.append(note)
    if len(notes) > _EVENT_NOTES_MAX:
        del notes[: len(notes) - _EVENT_NOTES_MAX]
    logger.info(note)
    return True


def _drain_events(chat_id: str) -> list[str]:
    """Забрать накопленные отметки чата (вызывается при сборке пачки)."""
    return _event_notes.pop(str(chat_id), [])


async def _process_update(update: dict[str, Any]) -> None:
    if _collect_event(update):
        return
    prepared = await _prepare_message(update)
    if prepared is None:
        return
    chat_id, message_text, trusted, post_id = prepared
    await _buffer_message(chat_id, message_text, trusted, post_id=post_id)


async def _prepare_message(
    update: dict[str, Any],
) -> tuple[str, str, bool, str | None] | None:
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
    downloaded: list[tuple[str, str, dict[str, Any]]] = []
    quoted_parts: list[str] = []
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
                downloaded.append((result["path"], att_type, payload))
            else:
                notes.append(
                    f"{att_type}: download failed: "
                    f"{result.get('error') or 'unknown error'}"
                )
            continue

        if att_type == "link":
            # Ответ на более раннее сообщение: MAX присылает ссылку вида
            # https://max.ru/<chat>/<message_id>. Полную цитату подтягиваем
            # сами, иначе агент видит только «ответ на что-то».
            quote = await _fetch_quoted_message(payload, chat_id)
            if quote:
                quoted_parts.append(quote)
            else:
                notes.append(
                    "link: quoted message could not be fetched "
                    f"({payload.get('url') or payload.get('message_id') or 'no url'})"
                )
            continue

        # Неизвестный тип вложения: дампим payload целиком в лог, чтобы потом
        # можно было понять структуру (в сообщение не попадает).
        logger.debug(
            "MAX raw attachment: %s", json.dumps(payload, ensure_ascii=False)
        )
        notes.append(f"{att_type}: attachment type not processed")

    parts: list[str] = []
    if text:
        parts.append(text)
    if voice_transcripts:
        parts.append(
            "[Voice Message Transcript]\n" + "\n\n".join(voice_transcripts)
        )
    if quoted_parts:
        parts.append(
            "[Quoted messages]\n" + "\n\n".join(quoted_parts)
        )
    if downloaded:
        parts.append(
            "[Downloaded attachments]\n"
            + "\n".join(_describe_attachment(item) for item in downloaded)
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
    if data.get("post_id"):
        header += f" | post_id={data['post_id']} | channel_comment=true"
    if user_id:
        header += f" | user_id={user_id}"
    if data["sender_name"]:
        header += f" | sender={data['sender_name']}"
    header += (
        f" | attachments={','.join(attachment_types) if attachment_types else 'none'}"
    )
    header += f" | trusted={'true' if trusted else 'false'}]"

    return chat_id, f"{header}\n{content}", trusted, data.get("post_id")


def _describe_attachment(item: tuple[str, str, dict[str, Any]]) -> str:
    """Одна строка про скачанный файл: путь, тип, имя, размер, mime."""
    path, att_type, payload = item
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    name = payload.get("filename") or "?"
    mime = payload.get("mime_type") or payload.get("content_type") or "?"
    return f"- {path} | type={att_type} | name={name} | size={size} | mime={mime}"


def _parse_message_link(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    """Достать (chat_id, message_id) из attachment type=link.

    MAX присылает ссылку вида ``https://max.ru/<chat>/<message_id>`` либо
    поля ``chat_id`` / ``message_id`` прямо в payload.
    """
    chat_id = payload.get("chat_id")
    message_id = payload.get("message_id")
    url = payload.get("url") or payload.get("link") or ""
    if url:
        parts = [p for p in url.split("?")[0].split("/") if p]
        if not message_id and parts:
            message_id = parts[-1]
        if not chat_id and len(parts) >= 2:
            chat_id = parts[-2]
    return (
        str(chat_id) if chat_id is not None else None,
        str(message_id) if message_id is not None else None,
    )


async def _fetch_quoted_message(
    payload: dict[str, Any],
    current_chat_id: str,
) -> str | None:
    """Подтянуть цитируемое сообщение по attachment type=link."""
    chat_id, message_id = _parse_message_link(payload)
    bot = getattr(runtime, "bot", None)
    if bot is None or not message_id:
        return None
    try:
        message = await bot.get_message(message_id)
    except Exception as exc:
        logger.warning(f"MAX quote fetch failed: {exc}")
        return None
    if not message:
        return None
    body = message.get("body") or {}
    sender = (message.get("sender") or {}).get("name") or "?"
    quote_text = (body.get("text") or "").strip()
    if not quote_text:
        quote_text = "(без текста)"
    return (
        f"[in reply to | sender={sender} | message_id={message_id}"
        f" | chat_id={chat_id or current_chat_id}]\n{quote_text}"
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
