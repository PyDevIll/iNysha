"""Async MAX bot with reasoning-to-separate-chat for MASTERMIND v2."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable, Awaitable, Any, Dict, List, Union

import httpx
from loguru import logger

# ---------- Constants ----------
MAX_API_URL = "https://platform-api2.max.ru"
MAX_API_VERSION = "v1"
LONG_POLL_TIMEOUT = 60
RETRY_INTERVAL = 5  # seconds

import hashlib
import mimetypes

def _safe_filename_from_token(token: str, content_type: str = "") -> str:
    """
    Generate a safe filename from a token, avoiding path separators and invalid chars.
    """
    # Если токен содержит path-разделители — берём последний сегмент
    if "/" in token:
        token = token.split("/")[-1]
    if "\\" in token:
        token = token.split("\\")[-1]

    # Удаляем недопустимые символы для Windows/Linux
    # Разрешены: буквы, цифры, точка, дефис, подчёркивание
    import re
    safe = re.sub(r'[^a-zA-Z0-9_.\-]', '', token)
    if not safe:
        # Если после очистки ничего не осталось — берём хеш
        safe = hashlib.md5(token.encode()).hexdigest()[:8]

    # Добавляем расширение из Content-Type
    ext = ""
    if content_type:
        content_type = content_type.split(';')[0].strip()
        if content_type:
            ext = mimetypes.guess_extension(content_type)
            if not ext:
                # fallback по типу
                if "image/" in content_type:
                    ext = ".jpg" if "jpeg" in content_type or "jpg" in content_type else ".png"
                elif "audio/" in content_type:
                    ext = ".ogg" if "ogg" in content_type else ".mp3" if "mpeg" in content_type else ".wav"
                elif "video/" in content_type:
                    ext = ".mp4"
                else:
                    ext = ".bin"
    return safe + ext


# ---------- Attachment Helpers (matching TypeScript helpers) ----------
class Attachment:
    """Base class for MAX attachments."""
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError("Attachment not implemented.")


class MediaAttachment(Attachment):
    """Attachment with a token (uploaded file)."""
    def __init__(self, token: Optional[str] = None):
        self.token = token

    @property
    def payload(self) -> Dict[str, Any]:
        return {"token": self.token}


class ImageAttachment(MediaAttachment):
    """Image attachment."""
    def __init__(
        self,
        token: Optional[str] = None,
        url: Optional[str] = None,
        photos: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        super().__init__(token)
        self.url = url
        self.photos = photos

    @property
    def payload(self) -> Dict[str, Any]:
        if self.token:
            return {"token": self.token}
        if self.url:
            return {"url": self.url}
        return {"photos": self.photos or {}}

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "image", "payload": self.payload}


class VideoAttachment(MediaAttachment):
    """Video attachment."""
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "video", "payload": self.payload}


class AudioAttachment(MediaAttachment):
    """Audio attachment."""
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "audio", "payload": self.payload}


class FileAttachment(MediaAttachment):
    """File attachment."""
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "file", "payload": self.payload}


class StickerAttachment(Attachment):
    """Sticker attachment."""
    def __init__(self, code: str):
        self.code = code

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "sticker", "payload": {"code": self.code}}


class LocationAttachment(Attachment):
    """Location attachment."""
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "location",
            "latitude": self.latitude,
            "longitude": self.longitude,
        }


class ShareAttachment(Attachment):
    """Share attachment."""
    def __init__(self, url: Optional[str] = None, token: Optional[str] = None):
        self.url = url
        self.token = token

    def to_dict(self) -> Dict[str, Any]:
        payload = {}
        if self.url:
            payload["url"] = self.url
        if self.token:
            payload["token"] = self.token
        return {"type": "share", "payload": payload}


def inline_keyboard(buttons: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Create an inline keyboard attachment."""
    return {
        "type": "inline_keyboard",
        "payload": {"buttons": buttons},
    }


# ---------- Button Helpers ----------
def callback_button(text: str, payload: str, intent: Optional[str] = None) -> Dict[str, Any]:
    """Create a callback button."""
    button: Dict[str, Any] = {"type": "callback", "text": text, "payload": payload}
    if intent:
        button["intent"] = intent
    return button


def link_button(text: str, url: str) -> Dict[str, Any]:
    """Create a link button."""
    return {"type": "link", "text": text, "url": url}


def request_contact_button(text: str) -> Dict[str, Any]:
    """Create a request contact button."""
    return {"type": "request_contact", "text": text}


def request_geo_location_button(text: str, quick: bool = False) -> Dict[str, Any]:
    """Create a request geo location button."""
    return {"type": "request_geo_location", "text": text, "quick": quick}


def chat_button(
    text: str,
    chat_title: str,
    chat_description: Optional[str] = None,
    start_payload: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a chat button."""
    button: Dict[str, Any] = {
        "type": "chat",
        "text": text,
        "chat_title": chat_title,
    }
    if chat_description:
        button["chat_description"] = chat_description
    if start_payload:
        button["start_payload"] = start_payload
    return button


# ---------- Context Class (matching TypeScript Context) ----------
class Context:
    """
    MAX bot context - wraps an update and provides convenient methods.
    Matches the TypeScript Context class interface.
    """

    def __init__(self, update: Dict[str, Any], api: "MAXBotApi", bot_info: Optional[Dict[str, Any]] = None):
        self.update = update
        self.api = api
        self.bot_info = bot_info
        self.match: Optional[re.Match] = None

    @property
    def update_type(self) -> Optional[str]:
        return self.update.get("update_type")

    @property
    def chat_id(self) -> Optional[int]:
        """Get chat ID from the update."""
        return self.update.get("chat_id")

    @property
    def user_id(self) -> Optional[int]:
        """Get user ID from the update (if available)."""
        # For message_created, we need to fetch the message to get sender
        if self.update_type == "message_created":
            # We'll fetch on demand
            return None
        return self.update.get("user_id")

    @property
    def message(self) -> Optional[Dict[str, Any]]:
        """Get the message from the update (for message_created)."""
        if self.update_type == "message_created":
            # The update contains chat_id, we need to fetch the actual message
            return self.update.get("message")
        return None

    @property
    def start_payload(self) -> Optional[str]:
        """Get start payload for bot_started updates."""
        if self.update_type == "bot_started":
            return self.update.get("payload")
        return None

    def has(self, *filters: Union[str, Callable]) -> bool:
        """Check if update matches any filter."""
        for filter_item in filters:
            if isinstance(filter_item, str):
                if filter_item == self.update_type:
                    return True
            elif callable(filter_item):
                if filter_item(self.update):
                    return True
        return False

    async def reply(self, text: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Reply to the current chat."""
        chat_id = self.chat_id
        if chat_id is None:
            raise ValueError("No chat_id available for reply")
        return await self.api.send_message_to_chat(chat_id, text, extra)

    async def send_message_to_user(self, user_id: int, text: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send a message to a specific user."""
        return await self.api.send_message_to_user(user_id, text, extra)

    async def get_messages(self, count: int = 10, from_time: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get messages from the current chat."""
        chat_id = self.chat_id
        if chat_id is None:
            raise ValueError("No chat_id available")
        return await self.api.get_messages(chat_id, count=count, from_time=from_time)

    async def get_chat(self) -> Optional[Dict[str, Any]]:
        """Get chat information."""
        chat_id = self.chat_id
        if chat_id is None:
            raise ValueError("No chat_id available")
        return await self.api.get_chat(chat_id)


# ---------- MAX Bot API Client ----------
class MAXBotApi:
    """Low-level MAX API client."""

    def __init__(self, token: str, base_url: str = MAX_API_URL):
        self.token = token
        self.base_url = base_url
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": self.token,
                    "Content-Type": "application/json",
                },
                timeout=70.0,
                verify=False
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        path: str,
        query: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        client = await self._get_client()
        logger.debug(f"[_request] {method} {path} query={query} body={body}")
        try:
            response = await client.request(
                method=method,
                url=path,
                params=query,
                json=body,
            )
            logger.debug(f"[_request] response status={response.status_code} body={response.text[:500]}")
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            logger.debug(f"MAX API timeout: {method} {path}")
            return {"ok": False, "error": "timeout"}
        except Exception as e:
            logger.error(f"MAX API error: {method} {path} - {e}")
            return {"ok": False, "error": str(e)}

    # ---------- Bot API ----------
    async def get_my_info(self) -> Dict[str, Any]:
        """Get bot information."""
        return await self._request("GET", "/bots/me")

    async def edit_my_info(self, **kwargs) -> Dict[str, Any]:
        """Edit bot information."""
        return await self._request("PUT", "/bots/me", body=kwargs)

    # ---------- Chat API ----------
    async def get_all_chats(self, **kwargs) -> Dict[str, Any]:
        """Get all chats."""
        return await self._request("GET", "/chats", query=kwargs)

    async def get_chat(self, chat_id: int) -> Optional[Dict[str, Any]]:
        """Get chat by ID."""
        result = await self._request("GET", f"/chats/{chat_id}")
        return result.get("chat") if result.get("ok") else None

    async def get_chat_by_link(self, link: str) -> Optional[Dict[str, Any]]:
        """Get chat by link."""
        result = await self._request("GET", "/chats/by-link", query={"chat_link": link})
        return result.get("chat") if result.get("ok") else None

    # ---------- Message API ----------
    async def send_message_to_chat(self, chat_id: int, text: str, extra: Optional[Dict] = None) -> Dict:
        body = {"text": text}
        if extra:
            body.update(extra)
        logger.debug(f"[send_message_to_chat] chat_id={chat_id} text='{text}' extra={extra} final_body={body}")
        result = await self._request(
            "POST",
            "/messages",
            query={"chat_id": chat_id},
            body=body,
        )
        logger.debug(f"[send_message_to_chat] result={result}")
        return result.get("message", {})

    async def send_message_to_user(self, user_id: int, text: str, extra: Optional[Dict] = None) -> Dict:
        body = {"text": text}
        if extra:
            body.update(extra)
        result = await self._request(
            "POST",
            "/messages",
            query={"user_id": user_id},
            body=body,
        )
        return result.get("message", {})


    async def get_messages(
        self,
        chat_id: int,
        message_ids: Optional[List[str]] = None,
        from_time: Optional[int] = None,
        to_time: Optional[int] = None,
        count: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get messages from a chat."""
        query: Dict[str, Any] = {"chat_id": chat_id, "count": count}
        if message_ids:
            query["message_ids"] = ",".join(message_ids)
        if from_time:
            query["from"] = from_time
        if to_time:
            query["to"] = to_time
        result = await self._request("GET", "/messages", query=query)
        return result.get("messages", [])

    async def get_message(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get a single message by ID."""
        result = await self._request("GET", f"/messages/{message_id}")
        return result.get("message") if result.get("ok") else None

    async def edit_message(
        self,
        message_id: str,
        text: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Edit a message."""
        body: Dict[str, Any] = {}
        if text is not None:
            body["text"] = text
        if attachments is not None:
            body["attachments"] = attachments
        result = await self._request(
            "PUT",
            "/messages",
            query={"message_id": message_id},
            body=body,
        )
        return result

    async def delete_message(self, message_id: str) -> Dict[str, Any]:
        """Delete a message."""
        return await self._request(
            "DELETE",
            "/messages",
            query={"message_id": message_id},
        )

    async def answer_on_callback(
        self,
        callback_id: str,
        message: Optional[Dict[str, Any]] = None,
        notification: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Answer a callback."""
        body: Dict[str, Any] = {}
        if message is not None:
            body["message"] = message
        if notification is not None:
            body["notification"] = notification
        return await self._request(
            "POST",
            "/answers",
            query={"callback_id": callback_id},
            body=body,
        )

    # ---------- Subscription API ----------
    async def get_subscriptions(self) -> Dict[str, Any]:
        """Get all subscriptions."""
        return await self._request("GET", "/subscriptions")

    async def create_subscription(
        self,
        url: str,
        update_types: Optional[List[str]] = None,
        secret: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a webhook subscription."""
        body: Dict[str, Any] = {"url": url}
        if update_types:
            body["update_types"] = update_types
        if secret:
            body["secret"] = secret
        return await self._request("POST", "/subscriptions", body=body)

    async def delete_subscription(self, url: str) -> Dict[str, Any]:
        """Delete a webhook subscription."""
        return await self._request("DELETE", "/subscriptions", query={"url": url})

    # ---------- Upload API ----------
    async def upload_file(self, file_path: str, file_type: str = "file") -> Dict:
        path = Path(file_path).resolve()
        if not path.exists():
            return {"ok": False, "error": f"File not found: {file_path}"}

        client = await self._get_client()
        try:
            # Step 1: Get upload URL from MAX API
            response = await client.post(
                "/uploads",
                params={"type": file_type},
            )
            response.raise_for_status()
            upload_info = response.json()
        except Exception as e:
            logger.error(f"getUploadUrl error: {e}")
            return {"ok": False, "error": f"getUploadUrl failed: {e}"}

        upload_url = upload_info.get("url")
        if not upload_url:
            return {"ok": False, "error": "No upload URL in response"}

        token = upload_info.get("token")  # obtained here

        # Step 2: Upload file to CDN
        file_bytes = await asyncio.to_thread(path.read_bytes)
        file_name = path.name

        try:
            async with httpx.AsyncClient(verify=False, timeout=70.0) as upload_client:
                upload_resp = await upload_client.post(
                    upload_url,
                    files={"data": (file_name, file_bytes)},
                )
                upload_resp.raise_for_status()
                cdn_body = upload_resp.text.strip()

            # Try to get token from CDN response body (for file type where step 1 has no token)
            cdn_token = None
            if cdn_body:
                try:
                    cdn_data = json.loads(cdn_body)
                    cdn_token = cdn_data.get("token")
                except (json.JSONDecodeError, TypeError):
                    pass  # XML or other format, ignore

            return {
                "ok": True,
                "result": {
                    "token": token or cdn_token,  # step 1 token first, CDN token fallback
                    "file_name": file_name,
                    "file_size": len(file_bytes),
                }
            }
        except Exception as e:
            logger.error(f"CDN upload error: {e}")
            return {"ok": False, "error": f"CDN upload failed: {e}"}


# ---------- MAX Bot Class (matching TelegramBot interface) ----------
class MAXBot:
    """
    Async MAX bot with long polling and separate reasoning chat.
    Matches the interface of TelegramBot for easy migration.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        reasoning_chat_id: Optional[int] = None,
    ):
        self._token = token or os.environ.get("MAX_BOT_TOKEN", "")
        self._reasoning_chat_id = reasoning_chat_id or int(
            os.environ.get("REASONING_CHAT_ID", "0") or "0"
        )
        self._api = MAXBotApi(self._token)
        self._marker: Optional[int] = None
        self._running: bool = False
        self._message_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._bot_info: Optional[Dict[str, Any]] = None

        if not self._token:
            logger.warning("MAX_BOT_TOKEN not set!")

    async def _api_call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generic API call (for compatibility with TelegramBot pattern)."""
        # Map Telegram-like methods to MAX API
        if method == "sendMessage":
            chat_id = params.get("chat_id")
            text = params.get("text", "")
            if chat_id:
                return await self._api.send_message_to_chat(chat_id, text)
            return {"ok": False, "error": "chat_id required"}

        if method == "getUpdates":
            # For long polling, we use a different approach
            return await self._poll_updates()

        return {"ok": False, "error": f"Method {method} not implemented"}

    async def send_message(self, chat_id: int, text: str) -> Optional[Dict[str, Any]]:
        """Send a message to a chat with Markdown formatting."""
        if not text:
            return None
        # Truncate if too long (MAX limit: 4000)
        if len(text) > 4000:
            text = text[:4000] + "\n\n... (truncated)"
        # Отправляем без экранирования, с явным format=markdown
        result = await self._api.send_message_to_chat(
            chat_id,
            text,
            {"format": "markdown"},
        )
        if not result:
            # Retry without markdown if failed (should not happen)
            result = await self._api.send_message_to_chat(chat_id, text)
        return {"ok": True, "result": result} if result else {"ok": False, "error": "send failed"}

    async def send_reasoning(self, text: str) -> None:
        """Send reasoning thought to the dedicated reasoning chat."""
        if not self._reasoning_chat_id or not text:
            return
        await self.send_message(self._reasoning_chat_id, f"🧠 *Reasoning:*\n{text[:3800]}")

    async def send_reply(self, chat_id: int, text: str) -> None:
        """Reply to a user in their chat."""
        await self.send_message(chat_id, text)

    # ---------- File transfer methods ----------
    async def send_document(self, chat_id: int, file_path: str, caption: Optional[str] = None) -> Dict[str, Any]:
        path = Path(file_path).resolve()
        if not path.exists():
            return {"ok": False, "error": f"File not found: {file_path}"}
        if not path.is_file():
            return {"ok": False, "error": f"Not a regular file: {file_path}"}

        # Upload file first
        upload_result = await self._api.upload_file(str(path), "file")
        logger.debug(f"[send_document] upload_result={upload_result}")
        if not upload_result.get("ok"):
            return {"ok": False, "error": upload_result.get("error", "Upload failed")}

        token = upload_result.get("result", {}).get("token")
        logger.debug(f"[send_document] token='{token}'")
        if not token:
            return {"ok": False, "error": "No token from upload"}

        # Wait for MAX to process the uploaded file on CDN
        await asyncio.sleep(2)

        attachment = FileAttachment(token)
        extra = {"attachments": [attachment.to_dict()]}
        text_param = caption or "document"
        if caption:
            extra["text"] = caption

        logger.debug(f"[send_document] chat_id={chat_id} caption={caption!r} text_param={text_param!r} extra={extra}")
        result = await self._api.send_message_to_chat(chat_id, text_param, extra)
        logger.debug(f"[send_document] send_message_to_chat result={result}")
        return {"ok": True, "result": result} if result else {"ok": False, "error": "send failed"}


    async def send_voice(self, chat_id: int, voice_path: str, caption: Optional[str] = None,
                         duration: Optional[int] = None) -> Dict[str, Any]:

        path = Path(voice_path).resolve()
        if not path.exists():
            return {"ok": False, "error": f"File not found: {voice_path}"}
        if not path.is_file():
            return {"ok": False, "error": f"Not a regular file: {voice_path}"}

        # Upload audio
        upload_result = await self._api.upload_file(str(path), "audio")
        logger.debug(f"[send_voice] upload_result={upload_result}")
        if not upload_result.get("ok"):
            return {"ok": False, "error": upload_result.get("error", "Upload failed")}

        token = upload_result.get("result", {}).get("token")
        logger.debug(f"[send_voice] token='{token}'")
        if not token:
            return {"ok": False, "error": "No token from upload"}

        attachment = AudioAttachment(token)
        extra = {"attachments": [attachment.to_dict()]}
        if caption:
            extra["text"] = caption
        if duration:
            extra["duration"] = duration

        # Wait for MAX to process the uploaded file on CDN
        await asyncio.sleep(2)

        text_param = caption or "voice"
        logger.debug(f"[send_voice] chat_id={chat_id} caption={caption!r} text_param={text_param!r} extra={extra}")
        result = await self._api.send_message_to_chat(chat_id, text_param, extra)
        logger.debug(f"[send_voice] send_message_to_chat result={result}")
        return {"ok": True, "result": result} if result else {"ok": False, "error": "send failed"}


    async def download_file(
        self,
        file_token: str,
        destination: str,
        url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Download a file from MAX using its token or a CDN URL.

        Args:
            file_token: The file token from a message attachment (used as fallback).
            destination: Absolute path or directory to save the file.
            url: Optional CDN URL (e.g., from attachment payload). If provided,
                 download directly without API authentication.

        Returns:
            dict with ok, path, file_size, file_name.
        """
        if url:
            # Direct CDN download – no auth needed
            try:
                async with httpx.AsyncClient(verify=False, timeout=70.0) as client:
                    response = await client.get(url, follow_redirects=True)
                    response.raise_for_status()
            except Exception as e:
                logger.error(f"CDN download error: {e}")
                return {"ok": False, "error": f"CDN download failed: {e}"}

                # Determine destination path
            dest = Path(destination).resolve()
            if dest.is_dir():
                # Try to get filename from Content-Disposition header
                content_disposition = response.headers.get("content-disposition")
                if content_disposition and "filename=" in content_disposition:
                    match = re.search(r'filename["\s]*=[\s]*"?([^"]+)"?', content_disposition)
                    if match:
                        filename = match.group(1)
                    else:
                        content_type = response.headers.get("content-type", "")
                        filename = _safe_filename_from_token(file_token, content_type)
                else:
                    content_type = response.headers.get("content-type", "")
                    filename = _safe_filename_from_token(file_token, content_type)
                dest = dest / filename

            dest.parent.mkdir(parents=True, exist_ok=True)

            # Write content to disk
            try:
                await asyncio.to_thread(dest.write_bytes, response.content)
            except Exception as e:
                return {"ok": False, "error": f"File write error: {e}"}

            return {
                "ok": True,
                "path": str(dest),
                "file_size": len(response.content),
                "file_name": dest.name,
            }

        # Fallback to API-based download (legacy)
        logger.warning("No `url` provided for download_file – using API-based download (may fail)")
        client = await self._api._get_client()
        try:
            response = await client.get(f"/files/{file_token}", follow_redirects=True)
            response.raise_for_status()

            dest = Path(destination).resolve()
            if dest.is_dir():
                content_disposition = response.headers.get("content-disposition")
                if content_disposition and "filename=" in content_disposition:
                    match = re.search(r'filename["\s]*=[\s]*"?([^"]+)"?', content_disposition)
                    if match:
                        filename = match.group(1)
                    else:
                        content_type = response.headers.get("content-type", "")
                        filename = _safe_filename_from_token(file_token, content_type)
                else:
                    content_type = response.headers.get("content-type", "")
                    filename = _safe_filename_from_token(file_token, content_type)
                dest = dest / filename

            dest.parent.mkdir(parents=True, exist_ok=True)

            await asyncio.to_thread(dest.write_bytes, response.content)

            return {
                "ok": True,
                "path": str(dest),
                "file_size": len(response.content),
                "file_name": dest.name,
            }
        except Exception as e:
            logger.error(f"Download error: {e}")
            return {"ok": False, "error": str(e)}


    async def _poll_updates(self) -> Dict[str, Any]:
        """Fetch new updates via long polling."""
        # MAX long polling uses marker-based approach
        params: Dict[str, Any] = {}
        if self._marker is not None:
            params["marker"] = self._marker

        # We need to use raw API for polling
        client = await self._api._get_client()
        try:
            response = await client.get(
                "/updates",
                params=params,
                timeout=LONG_POLL_TIMEOUT + 10,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("ok"):
                self._marker = data.get("marker")
                return {"ok": True, "result": data.get("updates", [])}
            return {"ok": False, "error": data.get("error", "Unknown error")}
        except httpx.TimeoutException:
            # Timeout is normal when no updates arrive
            return {"ok": True, "result": []}
        except Exception as e:
            logger.error(f"Polling error: {e}")
            return {"ok": False, "error": str(e)}

    def stop(self) -> None:
        self._running = False

    def get_message(self) -> Optional[Dict[str, Any]]:
        """Get next message from queue (non-blocking)."""
        try:
            return self._message_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def close(self):
        """Close the API client."""
        await self._api.close()


# ---------- Convenience exports ----------
__all__ = [
    # Main class
    "MAXBot",
    "MAXBotApi",
    "Context",
    # Attachments
    "Attachment",
    "MediaAttachment",
    "ImageAttachment",
    "VideoAttachment",
    "AudioAttachment",
    "FileAttachment",
    "StickerAttachment",
    "LocationAttachment",
    "ShareAttachment",
    "inline_keyboard",
    # Buttons
    "callback_button",
    "link_button",
    "request_contact_button",
    "request_geo_location_button",
    "chat_button",
    # Markdown
    "_escape_markdown",
]
