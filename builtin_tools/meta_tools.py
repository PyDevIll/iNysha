"""Meta-tools for the MASTERMIND v2 Agent — self-management commands.

Provides: reload_tools — hot-reload all builtin tool modules without restart.
"""

from loguru import logger
from tool_registry import get_registry


async def reload_tools() -> str:
    """Hot-reload all builtin tool modules from disk.

    Use this after creating or editing tool files (e.g. new *_tools.py in builtin_tools/).
    Reloads every submodule of builtin_tools via importlib.reload(),
    re-running each module's register_all() to pick up new/changed tool definitions.
    Returns count of reloaded modules and current tool list.
    """
    registry = get_registry()
    count = registry.hot_reload()
    tool_list = registry.list_tools()
    logger.info(f"Hot-reload complete: {count} modules, {len(registry._tools)} tools")
    return f"Reloaded {count} module(s).\n\n{tool_list}"


async def ping() -> str:
    """Simple ping/pong health check. Returns 'pong' with current timestamp."""
    from datetime import datetime
    return json.dumps({
        "result": "pong",
        "timestamp": datetime.now().isoformat(),
    }, ensure_ascii=False)


TOOL_DEFINITIONS = [
    ("ping", ping, "Simple ping/pong health check. Returns pong with current timestamp.", {
        "type": "object",
        "properties": {},
        "required": [],
    }),
    ("reload_tools", reload_tools, "Hot-reload all builtin tool modules without restarting", {
        "type": "object",
        "properties": {},
        "required": [],
    }),
]


def register_all(registry):
    for name, func, desc, params in TOOL_DEFINITIONS:
        registry.register_function(func, name, desc, params)
    logger.info(f"Registered {len(TOOL_DEFINITIONS)} meta tool(s)")
