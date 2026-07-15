import asyncio
from tts_rest_services import (
    tts_speak, tts_send_for_speaking, tts_request_queue,
    stt_transcriber, stt_start_audio_stream, stt_stop_audio_stream,
    audio
)

# |---------- TTS - STT Dialogue code -----------|
# |                                              |

stt_request_to_llm_queue = []
tts_context = None
tts_stt_enabled = False


async def gather_inputs_from_mic(stream):
    try:
        async for next_phrase in stt_transcriber(stream):
            if next_phrase:
                stt_request_to_llm_queue.append(next_phrase)
    except asyncio.CancelledError:
        print("Mic listening task cancelled")
    except Exception as e:
        raise e
    finally:
        stt_stop_audio_stream(stream)


async def tts_send_to_llm():
    text = "[Local mic audio transcription]:" + ' '.join(stt_request_to_llm_queue)
    print("Text is sent to LLM:", text)
    stt_request_to_llm_queue.clear()
    await request_queue.put({"type": "tts", "text": text})


async def llm_request_starter():
    max_stt_request_count = 5
    first_request_time = time()
    while True:
        if len(stt_request_to_llm_queue) > 0:
            time_passed = time() - first_request_time
            if len(stt_request_to_llm_queue) >= max_stt_request_count or time_passed > 10:
                print("Speaking starts: (queue size =", len(stt_request_to_llm_queue), ")") #, ", time passed =", time_passed, ")")
                await tts_send_to_llm()
        else:
            first_request_time = time()
            # print("No tts requests to speak")

        await asyncio.sleep(5)


def tts_stt_enable():
    stream = stt_start_audio_stream()
    tts_context = {
        "stream": stream,
        "transcriber_task": asyncio.create_task(gather_inputs_from_mic(stream)),
        "llm_request_starter_task": asyncio.create_task(llm_request_starter()),
        # "tts_speaking_starter_task": asyncio.create_task(tts_speaking_starter())
    }

    print("> Voice enabled")
    return tts_context


def tts_stt_disable(tts_context):
    stt_stop_audio_stream(tts_context["stream"])
    tts_context["transcriber_task"].cancel()
    tts_context["llm_request_starter_task"].cancel()
    tts_request_queue.clear()
    stt_request_to_llm_queue.clear()
    print("> Voice disabled")


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


async def process_voice_reply(text):
    async def reasoning_callback(thought):
        print(" > ...", thought)

    response_text = await agent.run_with_crash_recovery(
        initial_user_request=text,
        reasoning_callback=reasoning_callback,
    )

    if response_text:
        sanitized_text = _sanitize_for_tts(response_text)
        tts_send_for_speaking(sanitized_text)
        await tts_speak()

# |                                                |
# |---------- / TTS - STT Dialogue code -----------|
