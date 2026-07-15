"""Vision tools using Qwen VL (OpenAI-compatible API)."""

import base64
import mimetypes
import os

import httpx
from loguru import logger

QWEN_MODEL = "qwen3-vl-plus"


def _get_env_or_raise(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise ValueError(f"Environment variable {key} is not set")
    return value


async def _call_qwen_vl(image_url: str, query: str) -> str:
    """Call the Qwen VL API with a prepared image URL."""
    api_key = _get_env_or_raise("QWEN_API_KEY")
    endpoint = _get_env_or_raise("QWEN_API_ENDPOINT")
    chat_url = f"{endpoint}/chat/completions"

    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": 1024,
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

    result = await _call_qwen_vl(data_url, query)
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

    result = await _call_qwen_vl(image_url, query)
    logger.debug(f"vision_analyze_url returned: {result}")
    return result


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
]


def register_all(registry):
    """Register all vision tools in the given tool registry."""
    for name, func, description, schema in TOOL_DEFINITIONS:
        registry.register_function(name=name, func=func, description=description, parameters=schema)
    logger.info("Vision tools registered")
