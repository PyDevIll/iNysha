import os
from loguru import logger
import asyncio
from fastapi import FastAPI, Request, BackgroundTasks
import uvicorn
from lib.max_bot import MAXBot
from tool_registry import get_registry
from app import get_agent, get_request_queue, DOWNLOADS_DIR
from time import time


fast_api_app = FastAPI()
trusted_user_ids = [115302544, 5156907]
trusted_chat_ids = [326963375, 367837204]
DEFERRED_REPLY_TIME = 6    # seconds

@fast_api_app.post("/max-webhook")
async def max_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receive forwarded MAX updates from the serverless function.
    """
    request_queue = get_request_queue()
    if request_queue is None:
        return {"ok": False, "error": "Queue not initialized"}

    try:
        update = await request.json()
    except Exception as e:
        return {"ok": False, "error": f"Invalid JSON: {e}"}

    # Enqueue the update for processing (non-blocking)
    await request_queue.put({"type": "max", "update": update})
    return {"ok": True}


async def run_http_server():
    """Run FastAPI with uvicorn in a subprocess or thread."""
    config = uvicorn.Config(fast_api_app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


# Initialize MAX bot
bot = MAXBot(token=os.environ.get("MAX_BOT_TOKEN"), reasoning_chat_id=326963375)

registry = get_registry()
registry.set_bot(bot)

max_chat_updates_queue = {}
max_last_update_time = time()

def enqueue_max_chat_update(chat_id, msg):
    if chat_id in max_chat_updates_queue:
        max_chat_updates_queue[chat_id].append(msg)
    else:
        max_chat_updates_queue[chat_id] = [msg]
    global max_last_update_time
    max_last_update_time = time()
    print("Max message append to queue:", msg[:100])


async def max_deferred_reply():
    max_updates_list = []
    while True:
        await asyncio.sleep(1)
        if (time() - max_last_update_time) < DEFERRED_REPLY_TIME:
            continue

            # ----- Single chat reply -----
        if len(max_chat_updates_queue.items()) == 1:
            async def reasoning_callback(thought: str) -> None:
                await bot.send_reasoning(thought)

            chat_id = list(max_chat_updates_queue.keys())[0]
            msg_list = list(max_chat_updates_queue.values())[0]
            single_max_update = "[MAX messenger]:" + '\n'.join(msg_list)
            max_chat_updates_queue.clear()
            print("Send single MAX chat update to LLM", single_max_update)

            agent = get_agent()
            response = await agent.run_with_crash_recovery(
                initial_user_request=single_max_update,
                reasoning_callback=reasoning_callback,
            )

            if response:
                await bot.send_reply(chat_id, response)

        # ----- Multiple chat reply -----
        elif len(max_chat_updates_queue.items()) > 1:
            # ---- Collect chat updates ----
            max_updates_list.append("[MAX messenger][Bundled update]:")
            for chat_id in max_chat_updates_queue:
                max_updates_list.append(f"# chat_id: {chat_id}")
                max_updates_list.append('\n'.join(max_chat_updates_queue[chat_id]))
                max_updates_list.append('---\n')

            cumulative_max_update = '\n'.join(max_updates_list)
            print("Cumulative MAX update is sent to LLM:\n\n", cumulative_max_update)
            max_updates_list.clear()
            max_chat_updates_queue.clear()

            async def reasoning_callback(thought: str) -> None:
                await bot.send_reasoning(thought)

            # ---- Запуск агента и отправка ответа ----
            agent = get_agent()
            response = await agent.run_with_crash_recovery(
                initial_user_request=cumulative_max_update,
                reasoning_callback=reasoning_callback,
            )

            if response:
                await bot.send_reasoning(response)


# ---- Actual processing logic for user messages ----

async def process_max_message(update: dict) -> None:
    import json
    logger.info(f"Received MAX update: {json.dumps(update, indent=2, ensure_ascii=False)}")

    # ---- 1. Извлечение chat_id ----
    chat_id = update.get('chat_id')
    if not chat_id:
        message = update.get('message')
        if message:
            recipient = message.get('recipient')
            if recipient:
                chat_id = recipient.get('chat_id')
    if not chat_id:
        user_id = update.get('user_id')
        if user_id:
            chat_id = user_id
    if not chat_id:
        logger.warning("MAX update missing chat_id, cannot reply.")
        return

    # ---- 2. Извлечение данных ----
    update_type = update.get('update_type', 'unknown')
    message = update.get('message', {})
    body = message.get('body', {})
    text = body.get('text', '') or ''
    attachments = body.get('attachments', [])

    # ---- 3. Обработка вложений ----
    transcribed_text = ""
    downloaded_files = []

    for att in attachments:
        att_type = att.get('type')
        payload = att.get('payload', {})
        token = payload.get('token')
        url = payload.get('url')

        if att_type == 'audio':
            # Для аудио используем CDN URL, если есть, иначе токен (но токен без URL не работает)
            if url:
                logger.info(f"Downloading audio from CDN: {url[:100]}...")
                result = await bot.download_file(token, str(DOWNLOADS_DIR), url=url)
            else:
                logger.warning("No CDN URL for audio, trying token-only (will likely fail)")
                result = await bot.download_file(token, str(DOWNLOADS_DIR))

            if result.get('ok'):
                audio_path = result['path']
                logger.info(f"Audio downloaded: {audio_path}")
                # Транскрибируем через Yandex STT
                from builtin_tools.tts_tools import yandex_transcribe
                transcript_result = await yandex_transcribe(audio_path, lang="ru-RU")
                if transcript_result.get('ok'):
                    transcribed_text = transcript_result['text']
                    logger.info(f"Transcribed: {transcribed_text[:100]}")
                else:
                    logger.error(f"Transcription failed: {transcript_result.get('error')}")
            else:
                logger.error(f"Audio download failed: {result.get('error')}")

        elif att_type in ('file', 'image', 'video'):
            # Скачиваем файл/изображение/видео (используем токен, но можно и URL)
            result = await bot.download_file(token, str(DOWNLOADS_DIR), url=url)
            if result.get('ok'):
                downloaded_files.append(result['path'])
                logger.info(f"File downloaded: {result['path']}")
                await bot.send_message(
                    chat_id,
                    f"📥 Downloaded {att_type}: `{result['file_name']}` ({result['file_size']} bytes)"
                )
            else:
                logger.error(f"Download failed for {att_type}: {result.get('error')}")

    # ---- 4. Формирование итогового текста для агента ----
    # Если есть транскрипция, добавляем её к тексту
    if transcribed_text and not text:
        text = transcribed_text
    elif transcribed_text and text:
        text = f"{text}\n\n[Voice message transcription]: {transcribed_text}"

    # Если нет текста, но есть загруженные файлы, сообщаем о них
    if not text and downloaded_files:
        text = f"User sent files: {', '.join(downloaded_files)}"
    elif not text:
        # Если совсем ничего нет, используем JSON
        text = f"Received update of type '{update_type}':\n{json.dumps(update, indent=2, ensure_ascii=False)}"

    # ---- 5. Извлечение информации об отправителе ----
    sender = message.get('sender', {})
    sender_name = sender.get('name') or sender.get('first_name', '')
    user_id = sender.get('user_id')

    # is_trusted = (chat_id in trusted_chat_ids) or (user_id in trusted_user_ids)

    # ---- 6. Построение контекста ----
    context_parts = [f"Current chat_id: {chat_id}"]
    if user_id:
        sender_str = f"User: {user_id}"
        if sender_name:
            sender_str += f" ({sender_name})"
        context_parts.append(sender_str)
    context_parts.append(f"Update type: {update_type}")
    if downloaded_files:
        context_parts.append(f"Downloaded files: {', '.join(downloaded_files)}")
    context_parts.append(f"Content: {text}")

    enhanced_request = "\n".join(context_parts)
    # enhanced_request = f"{context_str}\n\nUser request: {text}"

    logger.info(f"MAX update in chat {chat_id}: {text[:100]}")

    # Enqueue update for deferred reply
    enqueue_max_chat_update(chat_id, enhanced_request)
