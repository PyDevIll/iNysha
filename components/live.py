import asyncio
from builtin_tools.vision_tools import analyze_dynamic_scene, get_screenshot, vision_analyze
from builtin_tools.tts_tools import yandex_transcribe
from lib.tts_rest_services import stt_transcriber, stt_start, stt_stop
from app import get_request_queue

from time import time


live_updater_task = None
live_enabled = False
last_scene_update_time = time()
SCENE_MINIMAL_UPDATE_PERIOD = 30    # seconds
SCENE_MONITOR = 3

# scene data for LLM
live_scene_request: dict = None


async def input_from_mic():
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
        "# [Live scene]"
        "## [Local mic]:",
        live_scene_request['audio_transcription'],
        "---\n",
        "## [Vision]:",
        live_scene_request['vision_analysis'],
        "---\n"
    ]
    return '\n'.join(_llm_request)


async def live_scene_updater():
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
            request_queue.put({"type": "live", "text": llm_request})
        else:
            print("[Live]: Scene is not ready")


async def _start_speech_callback():
    global live_scene_request
    if not live_scene_request["vision_analysis"]:
        live_scene_request["vision_analysis"] = "..."
        print("[Vision]: Started scene capturing and analysis")
        live_scene_request["vision_analysis"] = await analyze_dynamic_scene(monitor=SCENE_MONITOR, num_frames=5)
        print("[Vision]: Scene analysis done")
        ...


async def start_live():
    global live_enabled, live_updater_task
    live_updater_task = asyncio.create_task(live_scene_updater())
    live_enabled = True
    stt_start(speech_callback=_start_speech_callback())


async def stop_live():
    global live_enabled
    if live_updater_task and not live_updater_task.task_done():
        live_updater_task.cancel()
        live_scene_request.clear()
        live_enabled = False
        stt_stop()

