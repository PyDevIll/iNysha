"""FastAPI entry point for one configured iNysha-compatible instance.

Identity, paths and the public MAX webhook route come from ``server.config``
(and the checkout's ``.env``).  Local URL: http://127.0.0.1:<SERVER_PORT>.
Desktop clients: ws://<host>:<port>/ws/client and /ws/voice.
"""
from __future__ import annotations

import truststore
truststore.inject_into_ssl()

import ssl
print("ssl module:", ssl.__file__)
print("create_default_context:", ssl.create_default_context)
print("load_default_certs module:", ssl.SSLContext.load_default_certs.__module__)

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from server import config
from server.clients import hub, router as client_router
from server.runtime import runtime
from server.sources.api import router as api_router
from server.sources.max import router as max_router
from server.voice import router as voice_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    await runtime.startup()
    try:
        yield
    finally:
        await runtime.shutdown()


app = FastAPI(title=config.SERVER_TITLE, version="1.0.0", lifespan=lifespan)
app.include_router(max_router)
app.include_router(api_router)
app.include_router(client_router)
app.include_router(voice_router)

# ── Статика и браузерный голосовой чат ─────────────────────────────────
# Готовая сборка Vite лежит в VOICE_STATIC_DIR (index.html + assets/).
STATIC_DIR = config.STATIC_DIR
VOICE_DIR = config.VOICE_STATIC_DIR

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# index.html сборки ссылается на абсолютные пути /assets/*.
app.mount("/assets", StaticFiles(directory=VOICE_DIR / "assets"), name="assets")


@app.get("/voice", include_in_schema=False)
@app.get("/voice/", include_in_schema=False)
async def voice_page() -> FileResponse:
    """Страница голосового чата (снаружи может быть за префиксом прокси)."""
    return FileResponse(VOICE_DIR / "index.html")


@app.get("/audio-processor.js", include_in_schema=False)
async def audio_processor() -> FileResponse:
    """AudioWorklet: сборка грузит его с корня, а не из /static/voice/."""
    return FileResponse(VOICE_DIR / "audio-processor.js")


@app.get("/api/health")
async def health():
    return {
        "status": "ok" if runtime.started else "starting",
        "instance": config.INSTANCE_SLUG,
        "name": config.INSTANCE_NAME,
        "uptime_s": round(time.time() - runtime.started_at, 1) if runtime.started_at else 0,
        "queue": runtime.queue.qsize(),
        "queue_items": runtime.queue.snapshot(),
        "agent_ready": runtime.agent is not None,
        "max_bot": runtime.max_username,
        "max_user_id": runtime.max_user_id,
        "clients": hub.list(),
    }


@app.get("/api/function-ping")
async def function_ping(request: Request):
    """Diagnostic endpoint used by the Yandex Cloud Function.

    Set ``DEBUG_PING_URL`` in the function environment to this route to prove
    that MAX updates reach the function even when forwarding fails.
    """
    logger.info(f"Yandex function ping: {dict(request.query_params)}")
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server.main:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=False,
    )
