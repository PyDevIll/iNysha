"""Desktop client hub.

A client keeps one persistent control WebSocket open at ``/ws/client``.  The
server can call methods on it (screenshot, record audio, play PCM, ...) with a
simple request/response RPC.  Voice traffic uses a separate ``/ws/voice``
socket, so a long recording or a screenshot never blocks audio.

Only the server initiates RPC calls.  Results are correlated by an opaque
request id and every pending call fails as soon as the client disconnects.

The same socket also carries typed console prompts:

    client -> server: {"type": "prompt", "id": "...", "text": "..."}
    server -> client: {"type": "prompt_result", "id": "...", "ok": true,
                       "text": "..."}

Typed prompts enter the same serialized agent queue as MAX, API and voice
traffic, and inherit the originating ``client_id`` for remote tools.
"""

from __future__ import annotations

import asyncio
import hmac
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from server import config
from server.context import get_request_context

router = APIRouter(tags=["clients"])


class ClientError(RuntimeError):
    """A client RPC failed or the client is not available."""


class ClientDisconnected(ClientError):
    """The client went away while a call was pending."""


def _clean_client_id(raw: str | None) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", (raw or "").strip()).strip("-")
    return value[:64] or f"client-{uuid.uuid4().hex[:8]}"


@dataclass
class ClientInfo:
    client_id: str
    name: str
    capabilities: dict[str, Any]
    websocket: WebSocket
    connected_at: float
    _send_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _pending: dict[str, asyncio.Future[Any]] = field(default_factory=dict, repr=False)

    async def send_json(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            await self.websocket.send_json(payload)

    async def call(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float = 60.0,
    ) -> Any:
        """Call a method on the desktop client and wait for its result."""
        request_id = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        try:
            await self.send_json(
                {
                    "type": "rpc",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                }
            )
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise ClientError(
                f"client '{self.client_id}' did not answer {method!r} "
                f"within {timeout:.0f}s"
            ) from exc
        except ClientError:
            # A real RPC error from the client (or a disconnect reported by
            # fail_pending) must reach the caller unchanged.
            raise
        except (WebSocketDisconnect, RuntimeError) as exc:
            if future.done() and not future.cancelled():
                # A real disconnect is reported by fail_pending; keep that
                # message instead of a generic "cannot call send".
                try:
                    future.result()
                except ClientDisconnected:
                    raise
                except Exception:
                    pass
            raise ClientDisconnected(
                f"client '{self.client_id}' disconnected"
            ) from exc
        finally:
            self._pending.pop(request_id, None)

    def resolve(self, request_id: str, ok: bool, result: Any, error: str | None) -> None:
        future = self._pending.get(request_id)
        if future is None or future.done():
            logger.warning(
                f"client '{self.client_id}': late/unknown RPC result {request_id}"
            )
            return
        if ok:
            future.set_result(result)
        else:
            future.set_exception(ClientError(error or "client RPC failed"))

    def fail_pending(self, reason: str) -> None:
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(ClientDisconnected(reason))
        self._pending.clear()

    def public_info(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "name": self.name,
            "capabilities": self.capabilities,
            "connected_at": self.connected_at,
        }


class ClientHub:
    def __init__(self) -> None:
        self._clients: dict[str, ClientInfo] = {}

    def get(self, client_id: str) -> ClientInfo | None:
        return self._clients.get(client_id)

    def list(self) -> list[dict[str, Any]]:
        return [info.public_info() for info in self._clients.values()]

    def origin(self) -> ClientInfo | None:
        ctx = get_request_context()
        if ctx and ctx.client_id:
            return self._clients.get(ctx.client_id)
        return None

    def resolve(self, client_id: str | None = None) -> ClientInfo:
        """Pick the target client.

        Explicit ``client_id`` wins, then the client that originated the
        request.  If exactly one client is connected, it is a safe fallback
        for MAX/API requests (where there is no originating desktop client).
        """
        if client_id:
            info = self._clients.get(client_id)
            if info is None:
                raise ClientError(f"client '{client_id}' is not connected")
            return info

        origin = self.origin()
        if origin is not None:
            return origin

        if len(self._clients) == 1:
            return next(iter(self._clients.values()))

        if not self._clients:
            raise ClientError("no desktop client is connected")
        raise ClientError(
            "several clients are connected; specify client_id "
            f"({', '.join(sorted(self._clients))})"
        )

    async def register(self, info: ClientInfo) -> None:
        old = self._clients.get(info.client_id)
        if old is not None and old is not info:
            logger.info(
                f"client '{info.client_id}' reconnected; replacing old socket"
            )
            old.fail_pending("client reconnected with a new socket")
            try:
                await old.websocket.close(code=4000, reason="replaced")
            except Exception:
                pass
        self._clients[info.client_id] = info
        logger.info(
            f"client connected: {info.client_id!r} ({info.name}) "
            f"capabilities={sorted(info.capabilities)}"
        )

    def unregister(self, info: ClientInfo) -> None:
        # Do not remove a newer connection that replaced this one.
        if self._clients.get(info.client_id) is info:
            del self._clients[info.client_id]
            logger.info(f"client disconnected: {info.client_id!r} ({info.name})")
        info.fail_pending(f"client '{info.client_id}' disconnected")


hub = ClientHub()


async def _authenticate(websocket: WebSocket) -> dict[str, Any] | None:
    if not config.CLIENT_TOKEN:
        logger.error("CLIENT_TOKEN is not configured; rejecting client")
        await websocket.close(code=1011, reason="CLIENT_TOKEN is not configured")
        return None

    try:
        hello = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
    except Exception:
        await websocket.close(code=4001, reason="Expected hello message")
        return None

    if hello.get("type") != "hello":
        await websocket.close(code=4001, reason="Expected hello message")
        return None

    token = str(hello.get("token") or "")
    if not hmac.compare_digest(token, config.CLIENT_TOKEN):
        logger.warning("client authentication failed")
        await websocket.close(code=4001, reason="Authentication failed")
        return None

    return hello


@router.websocket("/ws/client")
async def client_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    hello = await _authenticate(websocket)
    if hello is None:
        return

    info = ClientInfo(
        client_id=_clean_client_id(hello.get("client_id")),
        name=str(hello.get("name") or hello.get("client_id") or "desktop-client"),
        capabilities=hello.get("capabilities") or {},
        websocket=websocket,
        connected_at=asyncio.get_running_loop().time(),
    )
    await hub.register(info)
    await info.send_json(
        {
            "type": "welcome",
            "client_id": info.client_id,
            "name": info.name,
            "server": "inysha",
        }
    )

    prompt_tasks: set[asyncio.Task] = set()

    async def run_text_prompt(prompt_id: str, text: str) -> None:
        """Enqueue a console prompt and return the agent's answer."""
        from server.queue import Request
        from server.routing import FutureReply
        from server.runtime import runtime

        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        header = (
            f"[Desktop client: {info.name} | client_id={info.client_id} | "
            f"trusted={'true' if config.CLIENT_TRUSTED else 'false'}]"
        )
        request = Request(
            text=f"{header}\n{text}",
            source="client_text",
            client_id=info.client_id,
            trusted=config.CLIENT_TRUSTED,
            reply=FutureReply(future),
        )
        try:
            await runtime.queue.put(request)
            logger.info(
                f"[{request.id}] queued text prompt from client "
                f"{info.client_id} (queue left={runtime.queue.qsize()})"
            )
            answer = await asyncio.wait_for(
                future, timeout=config.VOICE_TURN_TIMEOUT
            )
            payload = {
                "type": "prompt_result",
                "id": prompt_id,
                "ok": True,
                "text": (answer or "").strip(),
                "request_id": request.id,
            }
        except asyncio.TimeoutError:
            payload = {
                "type": "prompt_result",
                "id": prompt_id,
                "ok": False,
                "error": "agent turn timed out",
            }
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(f"text prompt failed for {info.client_id}")
            payload = {
                "type": "prompt_result",
                "id": prompt_id,
                "ok": False,
                "error": str(exc),
            }

        try:
            await info.send_json(payload)
        except Exception:
            # The client may have disconnected while the agent was working.
            pass

    try:
        while True:
            message = await websocket.receive_json()
            mtype = message.get("type")

            if mtype == "rpc_result":
                info.resolve(
                    str(message.get("id") or ""),
                    bool(message.get("ok")),
                    message.get("result"),
                    message.get("error"),
                )
            elif mtype == "prompt":
                prompt_id = str(message.get("id") or uuid.uuid4().hex)
                text = str(message.get("text") or "").strip()
                if len(text) <= 1:
                    await info.send_json(
                        {
                            "type": "prompt_result",
                            "id": prompt_id,
                            "ok": False,
                            "error": "prompt text is too short",
                        }
                    )
                else:
                    # Run in a task: a slow agent turn must not block
                    # screenshots or other control RPC traffic.
                    task = asyncio.create_task(
                        run_text_prompt(prompt_id, text[:8000]),
                        name=f"prompt-{info.client_id}",
                    )
                    prompt_tasks.add(task)
                    task.add_done_callback(prompt_tasks.discard)
            elif mtype == "ping":
                await info.send_json({"type": "pong", "t": message.get("t")})
            elif mtype == "log":
                logger.info(
                    f"[client {info.client_id}] {str(message.get('message'))[:400]}"
                )
            else:
                logger.debug(
                    f"client '{info.client_id}': ignored message type {mtype!r}"
                )
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception(f"client '{info.client_id}' control socket failed")
    finally:
        for task in list(prompt_tasks):
            task.cancel()
        if prompt_tasks:
            await asyncio.gather(*prompt_tasks, return_exceptions=True)
        hub.unregister(info)
