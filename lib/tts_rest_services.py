import requests

import httpx
import io
import pydub
from pydub.playback import play
import os
import asyncio
from time import time

import atexit


TTS_RATE = 48000
STT_RATE = 8000
STT_LISTEN_TIME = 3     # seconds between spoken phrases
STT_MAX_AUDIO_QUEUE_LEN = 10

stt_audio_queue = []
stt_last_request_time = time()
stt_audio_stream = None
stt_listen_task = None

def stt_collect_for_transcribing(audio_bytes):
    global stt_last_request_time
    stt_last_request_time = time()
    stt_audio_queue.append(b''.join(audio_bytes))
    print("Audio for STT appended to queue.", "Size = ", len(stt_audio_queue))


async def _tts_synthesizer(
        text: str,
        lang: str = 'ru-RU',
        speed: float = 1.2,
        format: str = 'lpcm',

    ):
    api_key = os.environ.get("YANDEX_SPEECH_API_KEY", "")
    url = 'https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize'
    data = {
        'text': text,
        'lang': lang,
        'voice': 'jane', #'marina' 'friendly', 'alena', 'good'
        'emotion': 'evil',
        'format': format,
        'sampleRateHertz': TTS_RATE,
        'speed': speed
    }
    headers = {
        "Authorization": f"Api-Key {api_key}",
    }

    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, headers=headers, data=data) as response:
            if response.status_code != 200:
                # Read the error body (if any) before raising
                error_body = await response.aread()
                raise RuntimeError(
                    f"Invalid response received: code: {response.status_code}, "
                    f"message: {error_body.decode()}"
                )

            # Yield audio chunks as they arrive
            async for chunk in response.aiter_bytes():
                yield chunk
    print("Finished synthesizing")


def play_pydub(chunk, rate):
    if chunk:
        print("Playing...")
        segment = pydub.AudioSegment.from_raw(
            io.BytesIO(chunk),
            sample_width=2,  # 16-bit = 2 bytes
            frame_rate=rate,
            channels=1
        )
        play(segment)
        print("Playing done")


async def tts_speak(text):
    print("TTS started!")

    print("Озвучиваемый текст:", text)
    audio_data = b''
    async for audio_chunk in _tts_synthesizer(text):
        audio_data += audio_chunk
    print("TTS ended")
    print('Speaking!')
    await asyncio.to_thread(play_pydub, audio_data, TTS_RATE)
    print('Speaking done')


# |----------------|
# |-- Listening ---|
# |----------------|

import webrtcvad
import collections
import pyaudio

# Настройки потокового распознавания.
FORMAT = pyaudio.paInt16
CHANNELS = 1
FRAME_MS = 30
FRAME_SIZE = int(STT_RATE * FRAME_MS / 1000)
FRAME_BYTES = FRAME_SIZE * 2
MINIMAL_VOICE_FRAMES = 10

audio = pyaudio.PyAudio()

def _transcribe_yandex_bytes(
    audio_bytes: bytes,
    lang: str = "ru-RU",
):
    """Send raw audio bytes to Yandex SpeechKit STT API and return result."""
    url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
    api_key = os.environ.get("YANDEX_SPEECH_API_KEY", "")
    if not api_key:
        return {"ok": False, "error": "YANDEX_SPEECH_API_KEY env var not set"}
    params = {
        "topic": "general",
        "lang": lang,
        "format": "lpcm",
        "sampleRateHertz": STT_RATE,
        "rawResults": True,
        "literature_text": True
    }
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "audio/x-pcm;bit=16;rate=8000"  # Правильный MIME-тип
    }
    try:
        with httpx.Client() as client:
            resp = client.post(
                url,
                params=params,
                headers=headers,
                content=audio_bytes,
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("result", "")
            if not text:
                return {"ok": False, "error": "Empty recognition result"}
            return {
                "ok": True,
                "text": text,
                "language": lang,
                "model": "yandex-speechkit-stt",
            }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def stt_listen_and_collect(speech_callback):
    vad = webrtcvad.Vad(3)
    last_frames = collections.deque(maxlen=10)
    voice_frames = []
    voice_frames_count = 0
    voice_started = False
    stream = stt_audio_stream
    long_speech = False     # whether the speech is long enough
    speech_cooldown = 0.3   # seconds for quiet endings
    last_voice_frame_time = time()

    while True:
        try:
            # Читаем данные в отдельном потоке
            data = await asyncio.to_thread(stream.read, FRAME_BYTES)
        except (OSError, IOError) as e:
            # Стрим закрыт – выходим (это ожидаемое поведение)
            print("STT listener: stream closed, stopping")
            break
        except Exception as e:
            print(f"STT listener: unexpected error: {e}")
            break

        # Убедимся, что кадр имеет правильный размер
        if len(data) != FRAME_BYTES * 2:
            print(f"STT listener: invalid frame size {len(data)}, skipping")
            continue

        # Обработка VAD
        is_speech = vad.is_speech(data, STT_RATE, FRAME_SIZE)

        if is_speech:
            if not voice_started:
                voice_frames.append(b''.join(last_frames))
            voice_started = True
            voice_frames.append(data)
            voice_frames_count += 1
            # --- long speech reaction ---
            if speech_callback:
                if voice_frames_count >= MINIMAL_VOICE_FRAMES:
                    await speech_callback()
            # ----------------------------
            last_voice_frame_time = time()

            print("", end="\r")
            print("VOICE", voice_frames_count, end='')
        else:
            if voice_started:
                if voice_frames_count >= MINIMAL_VOICE_FRAMES:
                    long_speech = True
                    last_frames.clear()
                else:
                    if long_speech:
                        last_frames.clear()
                        print("Short piece appended to speech")
                    else:
                        print("No record - too short")
            voice_started = False

        if long_speech:
            if (time() - last_voice_frame_time) >= speech_cooldown:
                print("Voice frames captured", voice_frames_count)
                stt_collect_for_transcribing(voice_frames)
                long_speech = False
                voice_frames.clear()
                voice_frames_count = 0
                last_frames.clear()
            else:
                print("Cooldown... speak on")

        last_frames.append(data)


def stt_start_audio_stream():
    global stt_audio_stream
    print("Chunk size:", FRAME_BYTES)
    stt_audio_stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=STT_RATE,
        input=True,
        frames_per_buffer=FRAME_BYTES
    )


def stt_stop_audio_stream():
    global stt_audio_stream
    if stt_audio_stream:
        try:
            stt_audio_stream.stop_stream()
            stt_audio_stream.close()
        except Exception as e:
            print(f"Error closing stream: {e}")


async def stt_transcriber():
    global stt_last_request_time
    while True:
        await asyncio.sleep(1)
        if len(stt_audio_queue) > 0:
            if  ((time() - stt_last_request_time) >= STT_LISTEN_TIME) or (len(stt_audio_queue) > STT_MAX_AUDIO_QUEUE_LEN):
                print('Send collected audio for transcription')
                audio_bytes = b''.join(stt_audio_queue)
                stt_audio_queue.clear()
                # play_pydub(audio_bytes, STT_RATE)
                stt_result = _transcribe_yandex_bytes(audio_bytes)
                print(stt_result)
                yield stt_result.get("text")


def stt_start(speech_callback = None):
    global stt_listen_task
    stt_start_audio_stream()
    stt_listen_task = asyncio.create_task(stt_listen_and_collect(speech_callback))


def stt_stop():
    stt_listen_task.cancel()
    stt_stop_audio_stream()
    stt_audio_queue.clear()


atexit.register(audio.terminate)
