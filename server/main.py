"""FastAPI entry point for the iNysha cloud server.

Local URL:  http://127.0.0.1:8010
Public URL: http://2.24.160.163:8010  (plain HTTP; no nginx/TLS on this path)
Desktop clients: ws://<host>:8010/ws/client and /ws/voice
MAX webhook:    http://<host>:8010/inysha/max-webhook
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


app = FastAPI(title="iNysha Cloud", version="1.0.0", lifespan=lifespan)
app.include_router(max_router)
app.include_router(api_router)
app.include_router(client_router)
app.include_router(voice_router)


@app.get("/api/health")
async def health():
    return {
        "status": "ok" if runtime.started else "starting",
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

    Set ``DEBUG_PING_URL=http://2.24.160.163:8010/api/function-ping`` in the
    function environment to prove that MAX updates reach the function even
    when forwarding fails.
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
