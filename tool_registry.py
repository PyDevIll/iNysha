"""Dynamic tool registry: builtin layer + client layer, builtin hot-reload."""

from __future__ import annotations

import importlib
import sys
import pkgutil
import inspect
import traceback
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class ToolDef:
    """Definition of a registered tool."""
    name: str
    description: str
    func: Callable
    parameters: dict = field(default_factory=dict)
    module_name: str = ""
    group: str = ""


# Tools exposed to the model only inside a voice turn (see get_openai_tools).
VOICE_ONLY_TOOLS = frozenset({"start_tool", "check_tool", "list_tool_tasks"})

# Strict allowlist for direct calls during a voice turn.  Everything else is
# reachable only through start_tool, which returns immediately and delivers the
# result as a separate turn.  Keep this list to quick/local/control tools; a
# missing name is intentionally hidden rather than merely discouraged.
VOICE_DIRECT_TOOLS = frozenset({
    "start_tool",
    "check_tool",
    "list_tool_tasks",
    "fs_read",
    "fs_stat",
    "list_clients",
    "schedule_task",
    "list_scheduled_tasks",
    "cancel_scheduled_task",
    # Локальный SQLite/FTS5, ~1 мс, только чтение — не блокирует разговор.
    "memory_search",
})


class ToolRegistry:
    """Async-first tool registry with separate builtin and client layers."""

    def __init__(self, tools_package: str = "builtin_tools"):
        self._tools_package = tools_package
        self._tools: Dict[str, ToolDef] = {}
        # Server-only tools (desktop client RPC) live in a separate lifecycle
        # layer: hot reload rebuilds builtins but must never clear this layer.
        self._client_tools: Dict[str, ToolDef] = {}
        self._version: int = 0
        self._loaded_modules: set[str] = set()
        self._bot = None

    def set_bot(self, bot) -> None:
        """Store the Telegram bot instance for tools to use."""
        self._bot = bot
        if not self._bot:
            logger.error(f"Failed to set bot for tool {self._tools_package}")

    def get_bot(self):
        """Retrieve the stored bot instance."""
        if not self._bot:
            logger.error(f"Failed to get bot for tool {self._tools_package}")
        return self._bot

    @property
    def version(self) -> int:
        return self._version


    @property
    def tool_names(self) -> list[str]:
        return list(self._visible_tools().keys())

    @property
    def client_tool_names(self) -> list[str]:
        return list(self._client_tools.keys())

    def _visible_tools(self) -> Dict[str, ToolDef]:
        """Merged read view. Names are unique across layers."""
        tools = dict(self._tools)
        tools.update(self._client_tools)
        return tools

    def _assert_builtin_name_available(self, name: str) -> None:
        if name in self._client_tools:
            raise ValueError(
                f"Tool '{name}' is already registered in the client layer; "
                "remove or rename one of the definitions"
            )

    def _assert_new_client_name(self, name: str) -> None:
        if name in self._tools or name in self._client_tools:
            raise ValueError(
                f"Tool '{name}' is already registered; client tools must have "
                "unique names and cannot override builtin tools"
            )

    def register(
        self,
        name: str,
        description: str,
        parameters: Optional[dict] = None,
        group: str = "",
    ) -> Callable:
        """Decorator to register an async tool function."""
        self._assert_builtin_name_available(name)

        def wrapper(func: Callable) -> Callable:
            if not inspect.iscoroutinefunction(func):
                logger.warning(f"Tool '{name}' is not async")
            module_name = func.__module__
            self._tools[name] = ToolDef(
                name=name, description=description,
                func=func, parameters=parameters or {}, module_name=module_name,
                group=group,
            )
            self._version += 1
            logger.debug(f"Registered tool: {name} from {module_name}")
            return func
        return wrapper


    def register_function(
        self, func: Callable, name: str, description: str,
        parameters: Optional[dict] = None, group: str = "",
    ) -> None:
        """Register a builtin (hot-reloadable) tool."""
        self._assert_builtin_name_available(name)
        module_name = func.__module__
        self._tools[name] = ToolDef(
            name=name, description=description,
            func=func, parameters=parameters or {}, module_name=module_name,
            group=group,
        )
        self._version += 1


    def register_client_function(
        self, func: Callable, name: str, description: str,
        parameters: Optional[dict] = None, group: str = "",
    ) -> None:
        """Register a server-side tool that must survive builtin hot reload.

        Client tools are not part of ``builtin_tools``: they depend on the
        desktop client hub and are installed once by ``server.runtime``.
        Duplicate names across layers are rejected instead of silently
        replacing one implementation with another.
        """
        self._assert_new_client_name(name)
        module_name = func.__module__
        self._client_tools[name] = ToolDef(
            name=name, description=description,
            func=func, parameters=parameters or {}, module_name=module_name,
            group=group,
        )
        self._version += 1
        logger.debug(f"Registered client tool: {name} from {module_name}")


    def get_tool(self, name: str) -> Optional[ToolDef]:
        return self._client_tools.get(name) or self._tools.get(name)


    def is_tool_allowed(
        self, name: str, *, voice_mode: bool = False
    ) -> bool:
        """Whether ``name`` may be called directly in the current mode.

        Background execution via ``start_tool`` is a separate path and may
        call tools that are not in the voice allowlist.
        """
        if self.get_tool(name) is None:
            return False
        if voice_mode:
            return name in VOICE_DIRECT_TOOLS
        return name not in VOICE_ONLY_TOOLS

    def get_openai_tools(self, voice_mode: bool | None = None) -> list[dict]:
        """Export tools for the LLM.

        Text sessions get the normal tool set except the fire-and-forget
        helpers.  Voice sessions get only ``VOICE_DIRECT_TOOLS``; all slow or
        otherwise blocking tools are reachable through ``start_tool`` and are
        deliberately absent from the schema.
        """
        tools = []
        for tdef in self._visible_tools().values():
            if not self.is_tool_allowed(tdef.name, voice_mode=bool(voice_mode)):
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": tdef.name,
                    "description": tdef.description,
                    "parameters": tdef.parameters,
                }
            })
        return tools

    async def call_tool(self, tool_name: str, **kwargs: Any) -> str:
        tdef = self.get_tool(tool_name)
        if not tdef:
            logger.error(
                f"call_tool: '{tool_name}' NOT FOUND. "
                f"Available: {sorted(self.tool_names)}"
            )
            return f"Error: tool '{tool_name}' not found"
        try:
            result = await tdef.func(**kwargs)
            return str(result)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' error: {e}")
            return f"Error executing '{tool_name}': {e}"


    def hot_reload(self) -> int:
        """Reload builtin modules and rebuild only the builtin tool layer.

        The client layer (desktop RPC tools) is installed once by
        ``server.runtime`` and intentionally survives this operation. The
        builtin layer is rebuilt transactionally: if registration fails, the
        previous layer is restored instead of being left half-populated.
        """
        reloaded = 0
        package = self._tools_package
        if package not in sys.modules:
            try:
                importlib.import_module(package)
            except ImportError as e:
                logger.error(f"Failed to import '{package}': {e}")
                return 0
        pkg = sys.modules[package]
        if not hasattr(pkg, '__path__'):
            return 0

        # 1) Reload all submodules
        for _, mod_name, _ in pkgutil.iter_modules(pkg.__path__, prefix=package + '.'):
            if mod_name in sys.modules:
                try:
                    importlib.reload(sys.modules[mod_name])
                    self._loaded_modules.add(mod_name)
                    reloaded += 1
                except Exception as e:
                    logger.error(f"Failed to reload '{mod_name}': {e}")
            else:
                try:
                    importlib.import_module(mod_name)
                    self._loaded_modules.add(mod_name)
                    reloaded += 1
                except Exception as e:
                    logger.error(f"Failed to load '{mod_name}': {e}")

        # 2) Also reload the package __init__.py itself so register_all is fresh
        try:
            importlib.reload(sys.modules[package])
            reloaded += 1
        except Exception as e:
            logger.error(f"Failed to reload package '{package}': {e}")

        pkg = sys.modules[package]
        register_all = getattr(pkg, "register_all", None)
        if register_all is None:
            logger.error(f"Package '{package}' has no register_all()")
            return 0

        # 3) Rebuild only the builtin layer. The client layer is untouched.
        old_tools = self._tools
        old_names = set(old_tools)
        self._tools = {}
        try:
            register_all(self)
        except Exception as e:
            self._tools = old_tools
            logger.error(f"register_all failed: {e}")
            logger.error(traceback.format_exc())
            logger.warning("hot_reload: restored the previous builtin tool layer")
            return 0

        new_names = set(self._tools)
        added = new_names - old_names
        removed = old_names - new_names
        if added:
            logger.info(f"hot_reload: +{len(added)} new builtin tools: {sorted(added)}")
        if removed:
            logger.warning(
                f"hot_reload: -{len(removed)} builtin tools: {sorted(removed)}"
            )
        logger.info(
            f"Registry reloaded: {reloaded} modules, "
            f"{len(old_names)}→{len(new_names)} builtin tools, "
            f"{len(self._client_tools)} client tool(s) preserved"
        )

        # 4) Verify all enabled TOOL_DEFINITIONS are actually registered
        self._verify_registration()

        self._version += 1
        return reloaded

    def _verify_registration(self) -> None:
        """Check that every enabled tool in TOOL_DEFINITIONS was registered."""
        package = self._tools_package
        pkg = sys.modules.get(package)
        if not pkg or not hasattr(pkg, '__path__'):
            return
        for _, mod_name, _ in pkgutil.iter_modules(pkg.__path__, prefix=package + '.'):
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            tdefs = getattr(mod, 'TOOL_DEFINITIONS', None)
            if not tdefs:
                continue
            disabled = getattr(mod, "DISABLED_TOOLS", set())
            for name, func, desc, params in tdefs:
                if name in disabled:
                    continue
                if name not in self._tools:
                    logger.error(f"VERIFY: tool '{name}' defined in {mod_name}.TOOL_DEFINITIONS "
                                 f"but NOT registered in registry!")

    def list_tools(self) -> str:
        visible = self._visible_tools()
        lines = [
            f"ToolRegistry v{self._version} — {len(visible)} tools "
            f"({len(self._tools)} builtin, {len(self._client_tools)} client):"
        ]
        for tool_name, tdef in sorted(visible.items()):
            layer = "client" if tool_name in self._client_tools else "builtin"
            lines.append(f"  {tool_name} [{layer}]: {tdef.description[:80]}")
        return '\n'.join(lines)
_registry: Optional[ToolRegistry] = None
_registry_initialized: bool = False


def get_registry() -> ToolRegistry:
    global _registry, _registry_initialized
    if _registry is None:
        _registry = ToolRegistry()
    if not _registry_initialized:
        try:
            from builtin_tools import register_all
            register_all(_registry)
            _registry_initialized = True
        except ImportError:
            logger.error("Cannot auto-register builtin tools (import failed)")
    return _registry
