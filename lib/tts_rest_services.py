import requests

import httpx
import io
import pydub
from pydub.playback import play
import os
import asyncio

TTS_RATE = 48000
STT_RATE = 8000

tts_request_queue = []


def tts_send_for_speaking(text):
    tts_request_queue.append(text)
    print("Request for generating text appended to queue.", "Size = ", len(tts_request_queue))


async def _tts_synthesizer(text):
    api_key = os.environ.get("YANDEX_SPEECH_API_KEY", "")
    url = 'https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize'
    data = {
        'text': text,
        'lang': 'ru-RU',
        'voice': 'jane', #'marina' 'friendly', 'alena', 'good'
        'emotion': 'evil',
        'format': 'lpcm',
        'sampleRateHertz': TTS_RATE,
        'speed': 1.2
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


def play_pydub(chunk):
    if chunk:
        print("Playing...")
        segment = pydub.AudioSegment.from_raw(
            io.BytesIO(chunk),
            sample_width=2,  # 16-bit = 2 bytes
            frame_rate=TTS_RATE,
            channels=1
        )
        play(segment)
        print("Playing done")


async def tts_speak():
    print("Speech session started!")

    text = '\n'.join(tts_request_queue)
    tts_request_queue.clear()


    print("Озвучиваемый текст:", text)
    audio_data = b''
    async for audio_chunk in _tts_synthesizer(text):
        audio_data += audio_chunk

    await asyncio.to_thread(play_pydub, audio_data)
    print("End of speech session")


# |----------------|
# |-- Listening ---|
# |----------------|

import webrtcvad
import collections
import pyaudio

# Настройки потокового распознавания.
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000
FRAME_MS = 30
FRAME_SIZE = int(RATE * FRAME_MS / 1000)
FRAME_BYTES = FRAME_SIZE * 2
MINIMAL_VOICE_FRAMES = 12

audio = pyaudio.PyAudio()

def _transcribe_yandex_bytes(
    audio_bytes: bytes,
    lang: str = "ru-RU",
    topic: str = "general",
):
    """Send raw audio bytes to Yandex SpeechKit STT API and return result."""
    url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
    api_key = os.environ.get("YANDEX_SPEECH_API_KEY", "")
    if not api_key:
        return {"ok": False, "error": "YANDEX_SPEECH_API_KEY env var not set"}
    params = {
        "topic": topic,
        "lang": lang,
        "format": "lpcm",
        "sampleRateHertz": RATE,
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


async def _stt_listener(stream):
    vad = webrtcvad.Vad(3)
    last_frames = collections.deque(maxlen=10)
    voice_frames = []
    voice_started = False

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

        # Обработка VAD (как у вас)
        last_frames.append(data)
        is_speech = vad.is_speech(data, RATE, FRAME_SIZE)

        if is_speech:
            if not voice_started:
                voice_frames.append(b''.join(last_frames))
            else:
                voice_frames.append(data)
            print("", end="\r")
            print("VOICE", len(voice_frames), end='')
            voice_started = True
        else:
            if voice_started:
                print("Voice frames captured", len(voice_frames))
                if len(voice_frames) >= MINIMAL_VOICE_FRAMES:
                    yield voice_frames
                else:
                    print("Too short")
                voice_frames.clear()
            voice_started = False


def stt_start_audio_stream():
    print("Chunk size:", FRAME_BYTES)
    return audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=FRAME_BYTES
    )


def stt_stop_audio_stream(stream):
    if stream is not None:
        try:
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Error closing stream: {e}")

async def stt_transcriber(stream):
    async for voice_frames in _stt_listener(stream):
        print('Send audio for transcription')
        audio_bytes = b''.join(voice_frames)
        stt_result = _transcribe_yandex_bytes(audio_bytes)
        print(stt_result)
        yield stt_result.get("text")

