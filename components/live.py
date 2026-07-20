import asyncio
from builtin_tools.vision_tools import analyze_dynamic_scene
from lib.tts_rest_services import stt_transcriber, stt_start, stt_stop
from components.tts_stt import _sanitize_for_tts, tts_speak
from app import get_request_queue, get_agent

from time import time


live_checker_task = None
update_audio_task = None
update_vision_task = None

live_enabled = False
last_scene_update_time = time()
SCENE_MINIMAL_UPDATE_PERIOD = 30    # seconds
SCENE_MONITOR = 3

# scene data for LLM
live_scene_request: dict = None


async def update_live_audio():
    global live_scene_request
    try:
        async for next_phrase in stt_transcriber():
            if next_phrase:
                live_scene_request["audio_transcription"] += f'.\n{next_phrase}'
                print("[Audio]: Speech captured: ", next_phrase)

    except asyncio.CancelledError:
        print("Mic listening task cancelled")
    except Exception as e:
        raise e


async def update_live_vision():
    global live_scene_request
    print("[Vision]: Started scene capturing and analysis")
    live_scene_request["vision_analysis"] = await analyze_dynamic_scene(monitor=SCENE_MONITOR, num_frames=5)
    print("[Vision]: Scene analysis done")


def _clear_live_scene():
    global live_scene_request
    live_scene_request = {
        'audio_transcription': '',
        'vision_analysis': '',
    }


def _live_scene_is_ready():
    _scene_is_ready = live_scene_request["audio_transcription"] and len(live_scene_request["vision_analysis"]) > 10
    _scene_is_ready = (_scene_is_ready and (time() - last_scene_update_time) > SCENE_MINIMAL_UPDATE_PERIOD)
    return _scene_is_ready


def _live_scene_request_to_str():
    _llm_request = [
        "# [Live scene]",
        "## [Local mic]:",
        live_scene_request['audio_transcription'],
        "---\n",
        "## [Vision]:",
        live_scene_request['vision_analysis'],
        "---\n"
    ]
    return '\n'.join(_llm_request)


# --- Live scene checker and sender to LLM ---
async def live_scene_checker():
    global last_scene_update_time
    while True:
        await asyncio.sleep(1)
        if _live_scene_is_ready():
            print("[Live]: Scene is ready")
            last_scene_update_time = time()
            llm_request = _live_scene_request_to_str()
            print("[Live]: ", llm_request)
            _clear_live_scene()
            # send request to LLM
            request_queue = get_request_queue()
            await request_queue.put({"type": "live", "scene": llm_request})
        else:
            print("[Live]: Scene is not ready")


# --- Trigger to start vision analysis ---
# only sync!
def _start_speech_callback(voice_frames_count):
    global update_vision_task
    if not update_vision_task or update_vision_task.done():
        print("[Audio]: Vision analysis triggered with voice", f"({voice_frames_count})")
        update_vision_task = asyncio.create_task(update_live_vision())


def start_live():
    global live_enabled, live_checker_task, update_audio_task
    _clear_live_scene()
    live_checker_task = asyncio.create_task(live_scene_checker())
    update_audio_task = asyncio.create_task(update_live_audio())
    live_enabled = True
    stt_start(speech_callback=_start_speech_callback)


async def stop_live():
    global live_enabled, live_checker_task, update_audio_task, update_vision_task
    if update_vision_task:
        update_vision_task.cancel()
        try:
            await update_vision_task
        except asyncio.CancelledError:
            pass
        update_vision_task = None

    if update_audio_task:
        update_audio_task.cancel()
        try:
            await update_audio_task
        except asyncio.CancelledError:
            pass
        update_audio_task = None

    if live_checker_task:
        live_checker_task.cancel()
        try:
            await live_checker_task
        except asyncio.CancelledError:
            pass
        live_checker_task = None

    _clear_live_scene()
    live_enabled = False
    stt_stop()


async def live_toggle():
    global live_enabled
    live_enabled = not live_enabled
    if live_enabled:
        start_live()
    else:
        await stop_live()
    return live_enabled


async def process_live_scene(scene_text):
    print("[Live]: Scene is sent to LLM: ", repr(scene_text[:100]), "...")
    agent = get_agent()
    response_text = await agent.run_with_crash_recovery(
        initial_user_request=scene_text
        # reasoning_callback=reasoning_callback,
    )

    if response_text:
        sanitized_text = _sanitize_for_tts(response_text)
        await tts_speak(sanitized_text)
