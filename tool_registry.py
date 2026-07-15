"""Dynamic tool registry with hot-reload for MASTERMIND v2."""

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


class ToolRegistry:
    """Async-first tool registry with hot-reload via importlib."""

    def __init__(self, tools_package: str = "builtin_tools"):
        self._tools_package = tools_package
        self._tools: Dict[str, ToolDef] = {}
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
        return list(self._tools.keys())


    def register(
        self,
        name: str,
        description: str,
        parameters: Optional[dict] = None,
    ) -> Callable:
        """Decorator to register an async tool function."""
        def wrapper(func: Callable) -> Callable:
            if not inspect.iscoroutinefunction(func):
                logger.warning(f"Tool '{name}' is not async")
            module_name = func.__module__
            self._tools[name] = ToolDef(
                name=name, description=description,
                func=func, parameters=parameters or {}, module_name=module_name,
            )
            self._version += 1
            logger.debug(f"Registered tool: {name} from {module_name}")
            return func
        return wrapper


    def register_function(
        self, func: Callable, name: str, description: str,
        parameters: Optional[dict] = None,
    ) -> None:
        module_name = func.__module__
        self._tools[name] = ToolDef(
            name=name, description=description,
            func=func, parameters=parameters or {}, module_name=module_name,
        )


    def get_tool(self, name: str) -> Optional[ToolDef]:
        return self._tools.get(name)


    def get_openai_tools(self) -> list[dict]:
        tools = []
        names = []
        for tdef in self._tools.values():
            names.append(tdef.name)
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
        tdef = self._tools.get(tool_name)
        if not tdef:
            logger.error(f"call_tool: '{tool_name}' NOT FOUND. Available: {sorted(self._tools.keys())}")
            return f"Error: tool '{tool_name}' not found"
        tdef = self._tools.get(tool_name)
        if not tdef:
            return f"Error: tool '{tool_name}' not found"
        try:
            result = await tdef.func(**kwargs)
            return str(result)
        except Exception as e:
            logger.error(f"Tool '{tool_name}' error: {e}")
            return f"Error executing '{tool_name}': {e}"


    def hot_reload(self) -> int:
        """Reload all submodules and re-register their tools.

        Fixes applied:
          1. Log errors from any failed submodule reload as ERROR.
          2. Also reload the package __init__.py so that register_all
             picks up the current submodule versions.
          3. Catch ANY exception from register_all, not just ImportError.
          4. Log which tools were added AND which disappeared.
          5. Compare post-register list against TOOL_DEFINITIONS from each
             submodule to detect missing registrations.
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

        # 2) Also reload the package __init__.py itself
        try:
            importlib.reload(sys.modules[package])
            reloaded += 1
        except Exception as e:
            logger.error(f"Failed to reload package '{package}': {e}")

        # 3) Re-register all tools
        old_names = set(self._tools.keys())
        old_tools = dict(self._tools)          # save a copy before clearing
        # ⚠️ Clear registry BEFORE re-registration so removed tools are purged
        self._tools.clear()
        try:
            from builtin_tools import register_all
            register_all(self)
        except Exception as e:
            logger.error(f"register_all failed: {e}")
            logger.error(traceback.format_exc())
            # On error, restore old tools to avoid data loss
            for k in list(old_names - set(self._tools.keys())):
                if k in old_tools:
                    self._tools[k] = old_tools[k]

        new_count = len(self._tools)
        new_names = set(self._tools.keys())
        added = new_names - old_names
        removed = old_names - new_names
        if added:
            logger.info(f"hot_reload: +{len(added)} new tools: {sorted(added)}")
        if removed:
            logger.warning(f"hot_reload: -{len(removed)} tools DISAPPEARED: {sorted(removed)}")
        logger.info(f"Registry reloaded: {reloaded} modules, {len(old_names)}→{new_count} tools")

        # 4) Verify all TOOL_DEFINITIONS are actually registered
        self._verify_registration()

        self._version += 1
        return reloaded

    def _verify_registration(self) -> None:
        """Check that every tool defined in TOOL_DEFINITIONS is actually
        present in the registry after reload."""
        import importlib
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
            for name, func, desc, params in tdefs:
                if name not in self._tools:
                    logger.error(f"VERIFY: tool '{name}' defined in {mod_name}.TOOL_DEFINITIONS "
                                 f"but NOT registered in registry!")

    def list_tools(self) -> str:
        lines = [f"ToolRegistry v{self._version} — {len(self._tools)} tools:"]
        for tool_name, tdef in sorted(self._tools.items()):
            lines.append(f"  {tool_name}: {tdef.description[:80]}")
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
