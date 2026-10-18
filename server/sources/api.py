"""HTTP prompt API — useful for smoke tests and the future desktop client."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from fastapi import Request as HttpRequest
from pydantic import BaseModel, Field

from server import config
from server.queue import Request
from server.routing import FutureReply
from server.runtime import runtime

router = APIRouter(tags=["api"])


def _check_secret(request: HttpRequest) -> None:
    if not config.MAX_WEBHOOK_SECRET:
        return
    provided = request.headers.get("x-inysha-secret")
    if provided != config.MAX_WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="invalid secret")


class PromptBody(BaseModel):
    text: str
    source: str = "api"
    trusted: bool = True
    timeout: float = Field(default=180.0, ge=1.0, le=600.0)


@router.post("/api/prompt")
async def prompt(body: PromptBody, request: HttpRequest):
    """Enqueue a text prompt and wait for the agent's final answer."""
    _check_secret(request)
    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()
    item = Request(
        text=body.text,
        source=body.source,
        trusted=body.trusted,
        reply=FutureReply(future),
    )
    await runtime.queue.put(item)

    try:
        response = await asyncio.wait_for(future, timeout=body.timeout)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="agent timeout")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "ok": True,
        "request_id": item.id,
        "response": response,
        "queue": runtime.queue.qsize(),
    }


@router.get("/api/queue")
async def queue_info(request: HttpRequest):
    _check_secret(request)
    return {
        "size": runtime.queue.qsize(),
        "items": runtime.queue.snapshot(),
        "worker": runtime.worker_task is not None and not runtime.worker_task.done(),
    }
