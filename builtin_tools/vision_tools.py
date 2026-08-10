"""Vision tools using Qwen VL (OpenAI-compatible API)."""

import base64
import mimetypes
import os
from datetime import datetime
import asyncio

import mss
import mss.tools
import io
from PIL import Image

import httpx
from loguru import logger
import atexit

QWEN_MODEL = "qwen-vl-plus"
_mss_instance = None

def _get_env_or_raise(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise ValueError(f"Environment variable {key} is not set")
    return value

# Add the multi-image API caller
async def _call_qwen_vl_multi(image_urls: list[str], query: str) -> str:
    """Call Qwen VL with multiple images (as data URLs)."""
    api_key = _get_env_or_raise("QWEN_API_KEY")
    endpoint = _get_env_or_raise("QWEN_API_ENDPOINT")
    chat_url = f"{endpoint}/chat/completions"

    content = [{"type": "text", "text": query}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    payload = {
        "model": QWEN_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 2048,  # adjust as needed
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(chat_url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("No choices in Qwen VL response")
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise RuntimeError("Empty content in Qwen VL response")
        return content


async def vision_analyze(image_path: str, query: str = "Опиши, что изображено на этой картинке. Подробно, на русском.") -> str:
    """
    Analyze a local image file using Qwen VL vision model.

    Args:
        image_path: Path to the image file on disk.
        query: Text prompt for the model. Defaults to a Russian description request.

    Returns:
        The model's textual description of the image.
    """
    logger.debug(f"vision_analyze called with path={image_path}, query={query}")

    # Read the file
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Determine MIME type
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"
        logger.warning(f"Cannot determine image MIME type for {image_path}, falling back to {mime_type}")

    b64_str = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64_str}"

    result = await _call_qwen_vl_multi([data_url], query)
    logger.debug(f"vision_analyze returned: {result}")
    return result


async def vision_analyze_url(image_url: str, query: str = "Опиши, что изображено на этой картинке. Подробно, на русском.") -> str:
    """
    Analyze an image from a publicly accessible URL using Qwen VL vision model.

    Args:
        image_url: HTTP/HTTPS URL of the image.
        query: Text prompt for the model. Defaults to a Russian description request.

    Returns:
        The model's textual description of the image.
    """
    logger.debug(f"vision_analyze_url called with url={image_url}, query={query}")

    result = await _call_qwen_vl_multi([image_url], query)
    logger.debug(f"vision_analyze_url returned: {result}")
    return result


# ----- Making screenshots -----

def get_mss():
    global _mss_instance
    if not _mss_instance:
        _mss_instance = mss.mss()
    return _mss_instance


def _capture_screenshot_to_bytes(monitor: int = 1, resize_factor: float = 0.5) -> bytes:
    """
    Capture a screenshot and return PNG bytes, optionally resized.
    """
    sct = get_mss()
    if monitor < 0 or monitor >= len(sct.monitors):
        logger.warning(f"Monitor {monitor} out of range (0-{len(sct.monitors)-1}), using primary (1)")
        monitor = 1

    # Grab the screenshot (returns a mss.ScreenShot object)
    raw = sct.grab(sct.monitors[monitor])

    # Convert to PIL Image
    pil_img = Image.frombytes("RGB", raw.size, raw.rgb)

    # Resize if needed
    if resize_factor != 1.0:
        width, height = pil_img.size
        new_size = (int(width * resize_factor), int(height * resize_factor))
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)

    # Convert to PNG bytes
    with io.BytesIO() as output:
        pil_img.save(output, format="PNG")
        return output.getvalue()


async def analyze_dynamic_scene(
    monitor: int = 1,
    interval_seconds: float = 0.5,
    num_frames: int = 5,
    query: str = "Опиши динамические изменения в этих последовательных кадрах. Дай детальный анализ того, что происходит в целом, на русском языке."
) -> str:
    """
    Capture multiple consecutive screenshots, resize them, and analyze together.

    Args:
        monitor: Monitor index (1 = primary, 2 = secondary, etc.).
        interval_seconds: Time between captures.
        num_frames: Number of screenshots to capture.
        resize_factor: Scale factor for image dimensions (0.5 = half size). Default 0.5.
        query: Question to ask about the sequence.

    Returns:
        The model's textual description of the dynamic scene.
    """
    resize_factor: float = 0.5
    logger.debug(f"analyze_dynamic_scene: monitor={monitor}, interval={interval_seconds}, frames={num_frames}, resize={resize_factor}")

    image_urls = []
    for i in range(num_frames):
        # Capture and resize in one step
        png_bytes = _capture_screenshot_to_bytes(monitor, resize_factor)
        b64_str = base64.b64encode(png_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_str}"
        image_urls.append(data_url)

        if i < num_frames - 1:
            await asyncio.sleep(interval_seconds)

    result = await _call_qwen_vl_multi(image_urls, query)
    logger.debug(f"analyze_dynamic_scene result: {result}")
    return result


async def get_screenshot(monitor: int = 0) -> dict:
    """
    Capture a screenshot of the specified monitor and save it to /data.

    Args:
        monitor: Monitor index (0 = all monitors combined, 1 = primary, 2 = secondary, etc.).
                 Default 0 captures the entire virtual screen.

    Returns:
        dict with keys:
            ok (bool): Whether capture succeeded.
            path (str): Path to the saved image (if ok).
            file_size (int): Size of the saved file in bytes (if ok).
            error (str): Error message if not ok.
    """
    from app import DATA_DIR
    logger.debug(f"get_screenshot called with monitor={monitor}")

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    output_path = os.path.join(DATA_DIR / "screenshots", filename)

    try:
        sct = get_mss()
        # mss monitors: index 0 is the combined virtual screen, 1..N are physical monitors
        # If the requested monitor index is out of range, fallback to 1 (primary)
        if monitor < 0 or monitor >= len(sct.monitors):
            logger.warning(f"Monitor {monitor} out of range (0-{len(sct.monitors)-1}), using primary (1)")
            monitor = 1

        # Capture and save directly
        sct.shot(mon=monitor, output=output_path)
        logger.info(f"Screenshot saved to {output_path}")

        file_size = os.path.getsize(output_path)
        return {
            "ok": True,
            "path": output_path,
            "file_size": file_size,
        }
    except Exception as e:
        logger.error(f"Screenshot capture failed: {e}")
        return {"ok": False, "error": str(e)}


def cleanup_mss():
    global _mss_instance
    if _mss_instance is not None:
        _mss_instance.close()
        _mss_instance = None

atexit.register(cleanup_mss)


TOOL_DEFINITIONS = [
    (
        "vision_analyze",
        vision_analyze,
        "Analyze a local image file using Qwen VL vision model",
        {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file on disk",
                },
                "query": {
                    "type": "string",
                    "description": "Text prompt for the model",
                    "default": "Опиши, что изображено на этой картинке. Подробно, на русском.",
                },
            },
            "required": ["image_path"],
        },
    ),
    (
        "vision_analyze_url",
        vision_analyze_url,
        "Analyze an image from URL using Qwen VL vision model",
        {
            "type": "object",
            "properties": {
                "image_url": {
                    "type": "string",
                    "description": "HTTP/HTTPS URL of the image",
                },
                "query": {
                    "type": "string",
                    "description": "Text prompt for the model",
                    "default": "Опиши, что изображено на этой картинке. Подробно, на русском.",
                },
            },
            "required": ["image_url"],
        },
    ),
    (
        "get_screenshot",
        get_screenshot,
        "Capture a screenshot of a monitor and save it to /data",
        {
            "type": "object",
            "properties": {
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index (0 = all monitors combined, 1 = primary, 2 = secondary, etc.)",
                    "default": 0,
                },
            },
            "required": [],
        },
    ),
    (
        "analyze_dynamic_scene",
        analyze_dynamic_scene,
        "Capture multiple screenshots at intervals and analyze the dynamic scene",
        {
            "type": "object",
            "properties": {
                "monitor": {
                    "type": "integer",
                    "description": "Monitor index (1 = primary, 2 = secondary, etc.)",
                    "default": 1,
                },
                "interval_seconds": {
                    "type": "number",
                    "description": "Time between consecutive captures (seconds)",
                    "default": 0.5,
                },
                "num_frames": {
                    "type": "integer",
                    "description": "Number of screenshots to capture",
                    "default": 5,
                },
                "query": {
                    "type": "string",
                    "description": "Question to ask about the sequence",
                    "default": "Опиши динамические изменения в этих последовательных кадрах. Дай детальный анализ того, что происходит в целом на русском языке.",
                },
            },
            "required": [],
        },
    )
]


def register_all(registry):
    """Register all vision tools in the given tool registry."""
    for name, func, description, schema in TOOL_DEFINITIONS:
        registry.register_function(name=name, func=func, description=description, parameters=schema)
    logger.info("Vision tools registered")