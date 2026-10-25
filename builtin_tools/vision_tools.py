"""Vision tools using DeepSeek (OpenAI-compatible API)."""

import base64
import mimetypes
import os

from openai import AsyncOpenAI
from loguru import logger

DEEPSEEK_MODEL = "deepseek-flash"
# Thinking mode in DeepSeek is ENABLED BY DEFAULT (effort=high). It burns the
# whole token budget on chain-of-thought: finish_reason="length" and an empty
# answer. Verified 22.10.2026: 2.9s / 400 completion tokens / 0 chars returned
# with thinking on, vs 1.4s / 35 tokens / full answer with thinking disabled.
DEEPSEEK_MAX_TOKENS = 1024
DEEPSEEK_RETRY_MAX_TOKENS = 4096
DEEPSEEK_THINKING_DISABLED = {"thinking": {"type": "disabled"}}

def _get_env_or_raise(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        raise ValueError(f"Environment variable {key} is not set")
    return value


def _deepseek_client() -> AsyncOpenAI:
    """Build an OpenAI-compatible client for the DeepSeek vision endpoint."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("DEEPSEEK_API_ENDPOINT")
    if not api_key:
        raise ValueError("Environment variable DEEPSEEK_API_KEY is not set")
    if not base_url:
        raise ValueError(
            "Neither DEEPSEEK_BASE_URL nor DEEPSEEK_API_ENDPOINT is set"
        )
    return AsyncOpenAI(api_key=api_key, base_url=base_url.rstrip("/"))


# Add the multi-image API caller
async def _call_deepseek_vision(image_urls: list[str], query: str) -> str:
    """Call DeepSeek vision with one or more images (data or public URLs).

    Thinking mode is explicitly disabled via extra_body: with it enabled the
    model spends its entire token budget on reasoning and returns an empty
    message with finish_reason="length". If the answer is still truncated, one
    retry is made with a larger budget.
    """
    client = _deepseek_client()

    content = [{"type": "text", "text": query}]
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    messages = [{"role": "user", "content": content}]
    budgets = [DEEPSEEK_MAX_TOKENS, DEEPSEEK_RETRY_MAX_TOKENS]
    last_reason = None

    for attempt, budget in enumerate(budgets, start=1):
        response = await client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            max_completion_tokens=budget,
            extra_body=DEEPSEEK_THINKING_DISABLED,
        )

        if not response.choices:
            raise RuntimeError("No choices in DeepSeek response")

        choice = response.choices[0]
        message = choice.message
        text = (message.content or "").strip()
        finish = getattr(choice, "finish_reason", None)

        if text and finish != "length":
            return text

        last_reason = f"empty/truncated answer (finish_reason={finish}, budget={budget})"
        logger.warning(
            f"DeepSeek vision attempt {attempt}: {last_reason}; "
            f"images={len(image_urls)}"
        )

        if finish != "length" and not text:
            break  # not a budget problem — retrying with more tokens won't help

    raise RuntimeError(f"Empty content in DeepSeek response: {last_reason}")


async def vision_analyze(image_path: str, query: str = "Опиши, что изображено на этой картинке. Подробно, на русском.") -> str:
    """
    Analyze a local image file using DeepSeek vision model.

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

    result = await _call_deepseek_vision([data_url], query)
    logger.debug(f"vision_analyze returned: {result}")
    return result


async def vision_analyze_url(image_url: str, query: str = "Опиши, что изображено на этой картинке. Подробно, на русском.") -> str:
    """
    Analyze an image from a publicly accessible URL using DeepSeek vision model.

    Args:
        image_url: HTTP/HTTPS URL of the image.
        query: Text prompt for the model. Defaults to a Russian description request.

    Returns:
        The model's textual description of the image.
    """
    logger.debug(f"vision_analyze_url called with url={image_url}, query={query}")

    result = await _call_deepseek_vision([image_url], query)
    logger.debug(f"vision_analyze_url returned: {result}")
    return result


TOOL_DEFINITIONS = [
    (
        "vision_analyze",
        vision_analyze,
        "Analyze a local image file using DeepSeek vision model",
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
        "Analyze an image from URL using DeepSeek vision model",
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
