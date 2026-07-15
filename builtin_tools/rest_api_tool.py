"""REST API call tool for MASTERMIND v2."""

import os
import sys
import json
import asyncio
from typing import Optional, Dict, Any, Union, List, Tuple
from loguru import logger

import httpx
import toons


async def rest_api_call(
    method: str = "GET",
    endpoint: str = "https://www.example.com/",
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Union[Dict[str, Any], List[Any], str, bytes]] = None,
    json_body: Optional[Union[Dict[str, Any], List[Any]]] = None,
    data: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    cookies: Optional[Dict[str, str]] = None,
    auth: Optional[Union[Tuple[str, str], str]] = None,
    timeout: float = 30.0,
    follow_redirects: bool = True,
    verify: bool = True,
    proxy: Optional[str] = None,  # 新增
    **kwargs
) -> Dict[str, Any]:
    """
    Universal REST API caller for LLM agents.

    Supports GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS.
    Handles query parameters, headers, multiple body formats,
    authentication, cookies, timeouts, redirects, and SSL verification.

    Returns structured response with status, headers, body, TOON serialization,
    elapsed time, and final URL.
    """
    # Normalise method to uppercase
    method = method.upper()

    # Build request arguments
    # Build request arguments (excluding client-level params)
    req_kwargs = {
        "params": params or {},
        "headers": headers or {},
        "cookies": cookies or {},
        "timeout": timeout,
        "follow_redirects": follow_redirects,
    }

    # Authentication handling
    if auth is not None:
        if isinstance(auth, tuple) and len(auth) == 2:
            req_kwargs["auth"] = httpx.BasicAuth(*auth)
        elif isinstance(auth, str):
            req_kwargs["headers"]["Authorization"] = f"Bearer {auth}"
        else:
            req_kwargs["auth"] = auth  # fallback, httpx will handle

    # Body handling with priority: body > json_body > data > files
    if body is not None:
        if isinstance(body, (dict, list)):
            req_kwargs["json"] = body
        else:
            req_kwargs["content"] = body
    elif json_body is not None:
        req_kwargs["json"] = json_body
    elif data is not None:
        req_kwargs["data"] = data
    elif files is not None:
        req_kwargs["files"] = files

    # Merge any extra kwargs (proxies, trust_env, etc.)
    req_kwargs.update(kwargs)

    # Execute request
    try:
        async with httpx.AsyncClient(verify=verify, proxy=proxy) as client:
            response = await client.request(method, endpoint, **req_kwargs)
            elapsed = response.elapsed.total_seconds()
            final_url = str(response.url)

            # Parse response body
            content_type = response.headers.get("content-type", "").lower()
            if "application/json" in content_type:
                try:
                    body_data = response.json()
                except Exception:
                    body_data = response.text
            elif "text/" in content_type:
                body_data = response.text
            else:
                body_data = response.content

            # Build result
            result = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": body_data,
                "elapsed": elapsed,
                "url": final_url,
            }

            # Add TOON serialization if possible
            try:
                if isinstance(body_data, (dict, list)):
                    result["toon"] = toons.dumps(body_data)
                # else we skip toon
            except toons.ToonEncodeError as e:
                logger.debug(f"TOON serialization skipped: {e}")

            return result

    except httpx.TimeoutException as e:
        return {"error": f"Request timeout ({timeout}s): {str(e)}"}
    except httpx.RequestError as e:
        return {"error": f"Network error: {str(e)}"}
    except Exception as e:
        logger.exception("Unexpected error in rest_api_call")
        return {"error": f"Unexpected error: {str(e)}"}


# Tool definition for agent registration
TOOL_DEFINITIONS = [
    (
        "rest_api_call",
        rest_api_call,
        "Universal REST API caller. Supports GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS. "
        "Handles query params, headers, body (raw, JSON, form, file uploads), auth (Basic/Bearer), "
        "cookies, timeouts, SSL, and returns structured response with TOON serialization.",
        {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "HTTP method (GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS)",
                    "default": "GET"
                },
                "endpoint": {
                    "type": "string",
                    "description": "Full API endpoint URL"
                },
                "params": {
                    "type": "object",
                    "description": "Query parameters as key-value dict"
                },
                "headers": {
                    "type": "object",
                    "description": "Request headers as key-value dict"
                },
                "body": {
                    # 关键修复：使用 anyOf 来明确多种可能的类型
                    "anyOf": [
                        {"type": "object"},
                        {"type": "array"},
                        {"type": "string"},
                        {"type": "number"},
                        {"type": "boolean"}
                    ],
                    "description": "Raw request body. Takes precedence over json_body, data, files."
                },
                "json_body": {
                    "anyOf": [{"type": "object"}, {"type": "array"}],
                    "description": "JSON-serializable body, sets Content-Type: application/json"
                },
                "data": {
                    "type": "object",
                    "description": "Form-encoded data, sets Content-Type: application/x-www-form-urlencoded"
                },
                "files": {
                    "type": "object",
                    "description": "Multipart file uploads, e.g. {'file': open('data.txt','rb')}"
                },
                "cookies": {
                    "type": "object",
                    "description": "Cookies to send with request"
                },
                "auth": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                        {"type": "string"}
                    ],
                    "description": "Authentication: tuple (username, password) for Basic, or string for Bearer token"
                },
                "timeout": {
                    "type": "number",
                    "description": "Request timeout in seconds",
                    "default": 30.0
                },
                "follow_redirects": {
                    "type": "boolean",
                    "description": "Automatically follow redirects",
                    "default": True
                },
                "verify": {
                    "type": "boolean",
                    "description": "Verify SSL certificates",
                    "default": True
                },
                "proxy": {
                    "type": "string",
                    "description": "Proxy URL (e.g., 'socks5://127.0.0.1:1080')"
                }
            },
            "required": ["endpoint"],
            # 为启用 strict mode 做准备
            "additionalProperties": False
        },
    ),
]

def register_all(registry):
    """Register the REST API tool with the given registry."""
    for name, func, desc, params in TOOL_DEFINITIONS:
        registry.register_function(func, name, desc, params)
    logger.info("Registered rest_api_call tool")

