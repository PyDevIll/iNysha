import asyncio
from time import time
import re

from app import get_request_queue, get_agent
from lib.tts_rest_services import (
    tts_speak, stt_transcriber, stt_start, stt_stop,
    audio
)

# |---------- TTS - STT Dialogue code -----------|
# |                                              |

tts_stt_enabled = False
stt_listen_task = None


async def input_from_mic():
    global tts_stt_enabled
    try:
        async for next_phrase in stt_transcriber():
            if next_phrase:
                request_queue = get_request_queue()
                await request_queue.put({"type": "tts", "text": next_phrase})

    except asyncio.CancelledError:
        print("Mic listening task cancelled")
    except Exception as e:
        raise e
    finally:
        stt_stop()
        tts_stt_enabled = False


def tts_stt_toggle():
    global tts_stt_enabled
    tts_stt_enabled = not tts_stt_enabled
    if tts_stt_enabled:
        stt_start()
        stt_listen_task = asyncio.create_task(input_from_mic())
    else:
        sst_listen_task.cancel()
        stt_stop()


def _sanitize_for_tts(text: str) -> str:
    """
    Очищает текст от символов, которые не озвучиваются или ломают синтез речи:
    - удаляет маркдаун-символы (*, _, `, ~, #, >, +, -, =, |, [], {}, () и т.д.)
    - заменяет слэши, обратные слэши, звёздочки, подчёркивания на пробелы
    - удаляет эмодзи и прочие непечатные символы
    - сохраняет буквы, цифры, пробелы и базовую пунктуацию (.,!?;:'-–—)
    - схлопывает множественные пробелы в один
    """
    # Оставляем только буквы, цифры, пробелы, и разрешённые знаки препинания
    allowed = r"[^a-zA-Zа-яА-ЯёЁ0-9\s\.\,\!\?\;\:\'\-\–\—]"
    cleaned = re.sub(allowed, " ", text)
    # Удаляем множественные пробелы
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


async def process_voice_input(text):
    async def reasoning_callback(thought):
        print(" > ...", thought)

    agent = get_agent()
    response_text = await agent.run_with_crash_recovery(
        initial_user_request=text,
        reasoning_callback=reasoning_callback,
    )

    if response_text:
        sanitized_text = _sanitize_for_tts(response_text)
        await tts_speak(sanitized_text)

# |                                                |
# |---------- / TTS - STT Dialogue code -----------|
