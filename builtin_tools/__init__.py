"""Builtin tools for MASTERMIND v2. Auto-registered on import."""

from loguru import logger


def register_all(registry):
    from . import fs_tools, git_tools, tavily_tools, meta_tools, edit_tools, max_tools, tts_tools, additional_tools, rest_api_tool, vision_tools
    fs_tools.register_all(registry)
    git_tools.register_all(registry)
    tavily_tools.register_all(registry)
    meta_tools.register_all(registry)
    edit_tools.register_all(registry)
    max_tools.register_all(registry)
    tts_tools.register_all(registry)
    additional_tools.register_all(registry)
    rest_api_tool.register_all(registry)
    vision_tools.register_all(registry)
    logger.info("All builtin tools registered")
