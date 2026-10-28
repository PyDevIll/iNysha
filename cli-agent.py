#!/usr/bin/env python3
"""cli-agent.py — одной командой, один прогон, без памяти и без сервера.

Это «выжимка» из ``agent.py`` для пакетных шагов: один системный промпт, один
промпт прогона, несколько файлов контекста, один файл на выходе. Всё состояние
живёт в памяти процесса и умирает вместе с ним.

Чего здесь сознательно НЕТ (в отличие от серверного агента):

* ``ContextPool``: компрессии, скользящего окна, подсчёта токенов, дампов
  ``failed_*.json`` / ``invalid_messages.json``;
* памяти (SQLite, ``persistent_memory.json``, эпизоды, ``read_sources``);
* очереди запросов, ``current_request``, маршрутизации ответов, voice-политики;
* hot-reload реестра инструментов и всех серверных builtin-модулей;

Инструменты берутся из живых модулей ``builtin_tools`` (fs / edit / exec), но
регистрируется только жёстко заданный список имён: остальное просто не попадает
в схему и не может быть вызвано.

Пример (шаг конвейера новостей):

    python cli-agent.py \\
        --prompt_file_for_run data/newsfeed/runs/<дата>-<тема>/02_history.tpl.md \\
        --included_files 'data/newsfeed/runs/<дата>-<тема>/0[01]_*' \\
        --output_file data/newsfeed/runs/<дата>-<тема>/02_history.md
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent

# .env читается напрямую: server.config тянет за собой весь инстанс.
load_dotenv(ROOT / ".env")

DEFAULT_BASE_PROMPT = ROOT / "system_prompts" / "identity.md"

DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MAX_TOKENS = 30000
DEFAULT_TEMPERATURE = 1.3
DEFAULT_MAX_ITERATIONS = 15
TOOL_TIMEOUT = 120.0
TOOL_OUTPUT_LIMIT = 8000

#: Единственные инструменты, которые видит модель. fs_read_docx здесь не нужен
#: (он требует docxpy, которого нет в requirements).
TOOL_NAMES = (
    # fs_tools (включённые в реестре сервера)
    "fs_tree",
    "fs_read",
    "fs_stat",
    "fs_append",
    "fs_repair_encoding",
    # edit_tools
    "fs_write_file",
    "fs_edit",
    "fs_aedit",
    "fs_apply_patch",
    "fs_edit_blocks",
    "fs_edit_diff",
    # additional_tools
    "exec_shell",
    "exec_python",
    # tavily_tools
    "tavily_search",
    "tavily_browse",
    "tavily_map",
    "tavily_research"
)


# ── Инструменты ────────────────────────────────────────────────────────


class ToolRegistry:
    """Плоский реестр поверх TOOL_DEFINITIONS

    Никакого ``tool_registry.get_registry()``: он авторегистрирует весь
    ``builtin_tools``, включая max/vision/memory/tavily, а те тянут ``server.*``.
    """

    def __init__(self, allowed: tuple[str, ...] = TOOL_NAMES) -> None:
        from builtin_tools import additional_tools, edit_tools, fs_tools

        self._tools: dict[str, tuple[Any, str, dict]] = {}
        for module in (fs_tools, edit_tools, additional_tools, tavily_tools):
            for name, func, desc, params in getattr(module, "TOOL_DEFINITIONS", []):
                if name in allowed:
                    self._tools[name] = (func, desc, params)

        missing = sorted(set(allowed) - set(self._tools))
        if missing:
            raise RuntimeError(
                "tools declared but not found in builtin_tools: " + ", ".join(missing)
            )

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": params,
                },
            }
            for name, (_, desc, params) in self._tools.items()
        ]

    async def call(self, name: str, arguments: str, timeout: float) -> tuple[str, Optional[dict]]:
        """Вернуть (результат_для_контекста, разобранные_аргументы|None)."""
        entry = self._tools.get(name)
        if entry is None:
            return f"Error: unknown tool '{name}'", None
        func = entry[0]

        try:
            args = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return f"Error: invalid JSON arguments — {exc}", None
        if not isinstance(args, dict):
            return "Error: arguments must be a JSON object", None

        logger.info(f"tool: {name}({json.dumps(args, ensure_ascii=False)[:180]})")
        try:
            result = await asyncio.wait_for(func(**args), timeout=timeout)
        except asyncio.TimeoutError:
            return f"Error: tool '{name}' timed out (>{timeout:.0f}s)", args
        except Exception as exc:  # инструмент не должен ронять прогон
            logger.error(f"tool '{name}' failed: {exc}")
            return f"Error executing '{name}': {exc}", args
        return ("" if result is None else str(result)), args


def _limit_output(result: Any, limit: int = TOOL_OUTPUT_LIMIT) -> str:
    text = "" if result is None else str(result)
    if len(text) <= limit:
        return text
    return (
        text[:limit]
        + f"\n\n[... truncated: shown first {limit} of {len(text)} chars — "
          f"narrow the request or read the rest]"
    )


# ── Сборка промпта ─────────────────────────────────────────────────────


def _read_text(path: Path) -> str:
    from lib import textio

    text, _info = textio.read_text(path)
    return text


def expand_included(patterns: list[str]) -> list[Path]:
    """Развернуть пути и глобы в список файлов, сохранив порядок аргументов."""
    resolved: list[Path] = []
    for pattern in patterns:
        expanded = str(Path(pattern).expanduser())
        if glob.has_magic(expanded):
            matches = sorted(Path(p) for p in glob.glob(expanded, recursive=True))
            matches = [p for p in matches if p.is_file()]
            if not matches:
                raise FileNotFoundError(f"--included_files: no files match '{pattern}'")
            resolved.extend(matches)
            continue
        raw = Path(expanded)
        if not raw.is_file():
            raise FileNotFoundError(f"--included_files: not a file: {pattern}")
        resolved.append(raw)
    return resolved


def build_messages(
    base_prompt: str,
    run_prompt: str,
    included: list[Path],
    base_prompt_name: str = "",
) -> list[dict]:
    """system (базовый промпт) + user (промпт прогона и приложенные файлы)."""
    messages: list[dict] = []
    if base_prompt.strip():
        messages.append({"role": "system", "content": base_prompt.rstrip() + "\n"})

    parts = [run_prompt.rstrip()]
    for path in included:
        parts.append(f"### FILE: {path}\n{_read_text(path).rstrip()}")
    messages.append({"role": "user", "content": "\n\n".join(p for p in parts if p)})
    return messages


def _warn_placeholders(base_prompt: str, base_prompt_name: str) -> None:
    if "{{" in base_prompt:
        logger.warning(
            f"{base_prompt_name}: contains {{{{...}}}} placeholders — cli-agent.py "
            "does not render them (use system_prompts/identity.md or a rendered file)"
        )


# ── Цикл агента ────────────────────────────────────────────────────────


class CliAgent:
    """Копия формы Agent.run(): столько же итераций, тот же параллелизм."""

    def __init__(
        self,
        client: Any,
        *,
        model: str,
        max_tokens: int,
        temperature: float,
        reasoning: bool,
        max_iterations: int,
        tool_timeout: float,
        tool_output_limit: int,
        registry: Optional[ToolRegistry] = None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._reasoning = reasoning
        self._max_iterations = max_iterations
        self._tool_timeout = tool_timeout
        self._tool_output_limit = tool_output_limit
        self._registry = registry
        self.tool_calls = 0
        self.iterations = 0

    async def llm_request(self, messages: list[dict]) -> dict:
        tools = self._registry.schemas() if self._registry else []
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools or None,
            stream=False,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            extra_body={
                "thinking": {"type": "enabled" if self._reasoning else "disabled"}
            },
        )
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        else:
            data = response
        if not data or not data.get("choices"):
            raise ValueError(f"LLM returned no choices: {data!r}")
        return data

    async def run(self, messages: list[dict]) -> str:
        for _ in range(self._max_iterations):
            self.iterations += 1
            response = await self.llm_request(messages)
            message = response["choices"][0]["message"]

            if not message.get("tool_calls"):
                return message.get("content") or ""

            messages.append(
                {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": message["tool_calls"],
                }
            )
            self.tool_calls += len(message["tool_calls"])
            messages.extend(await self._run_tools(message["tool_calls"]))

        raise RuntimeError(
            f"max iterations ({self._max_iterations}) reached without a final answer"
        )

    async def _run_tools(self, tool_calls: list[dict]) -> list[dict]:
        assert self._registry is not None
        results = await asyncio.gather(
            *(
                self._registry.call(
                    call["function"]["name"],
                    call["function"].get("arguments") or "{}",
                    self._tool_timeout,
                )
                for call in tool_calls
            )
        )
        return [
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["function"]["name"],
                "content": _limit_output(content, self._tool_output_limit),
            }
            for call, (content, _args) in zip(tool_calls, results)
        ]


# ── Ввод-вывод ─────────────────────────────────────────────────────────


def write_output(text: str, target: str) -> Optional[Path]:
    """Записать результат атомарно; '-' означает stdout. Вернуть путь."""
    if target == "-":
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
        sys.stdout.flush()
        return None

    path = Path(target).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".cli-agent-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text if text.endswith("\n") else text + "\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return path


def _env(*names: str) -> str:
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cli-agent.py",
        description="Один прогон iNysha-подобного агента без памяти, очереди и истории.",
    )
    parser.add_argument(
        "--base_prompt_file",
        default=str(DEFAULT_BASE_PROMPT),
        help="системный промпт (по умолчанию system_prompts/identity.md)",
    )
    parser.add_argument(
        "--prompt_file_for_run",
        required=True,
        help="файл с заданием прогона: его текст становится сообщением user",
    )
    parser.add_argument(
        "--included_files",
        action="extend",
        nargs="+",
        default=[],
        metavar="PATH",
        help="файлы контекста; можно несколько раз, поддерживаются глобы",
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="куда писать финальный текст ('-' — в stdout)",
    )
    parser.add_argument("--cwd", default=".", help="рабочая директория для инструментов")
    parser.add_argument(
        "--model",
        default=_env("DEEPSEEK_MODEL") or DEFAULT_MODEL,
        help=f"модель (по умолчанию {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base_url",
        default=_env("DEEPSEEK_BASE_URL", "DEEPSEEK_API_ENDPOINT") or DEFAULT_BASE_URL,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=int(_env("LLM_MAX_OUTPUT_TOKENS") or DEFAULT_MAX_TOKENS),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"температура, 0..2 (по умолчанию {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--no-reasoning",
        dest="reasoning",
        action="store_false",
        help="отключить thinking (по умолчанию включён)",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=DEFAULT_MAX_ITERATIONS
    )
    parser.add_argument("--tool_timeout", type=float, default=TOOL_TIMEOUT)
    parser.add_argument("--tool_output_limit", type=int, default=TOOL_OUTPUT_LIMIT)
    parser.add_argument(
        "--no-tools", dest="tools", action="store_false", help="не давать модель инструменты"
    )
    parser.add_argument("--quiet", action="store_true", help="только ошибки в stderr")
    return parser.parse_args(argv)


def _configure_logging(quiet: bool) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="ERROR" if quiet else "INFO",
        format="<level>{level: <7}</level> {message}",
    )


async def amain(ns: argparse.Namespace, client: Any = None) -> str:
    base_path = Path(ns.base_prompt_file).expanduser()
    if not base_path.is_file():
        raise FileNotFoundError(f"--base_prompt_file: not a file: {ns.base_prompt_file}")
    run_path = Path(ns.prompt_file_for_run).expanduser()
    if not run_path.is_file():
        raise FileNotFoundError(f"--prompt_file_for_run: not a file: {ns.prompt_file_for_run}")

    base_prompt = _read_text(base_path)
    _warn_placeholders(base_prompt, str(base_path))
    included = expand_included(list(ns.included_files))

    messages = build_messages(
        base_prompt,
        _read_text(run_path),
        included,
        base_prompt_name=str(base_path),
    )

    registry = ToolRegistry() if ns.tools else None
    if registry is not None:
        logger.info(f"tools: {len(registry.names)} — {', '.join(registry.names)}")

    if client is None:
        api_key = _env("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY_HELPER")
        if not api_key:
            raise SystemExit(
                "DEEPSEEK_API_KEY (or DEEPSEEK_API_KEY_HELPER) is not set in .env"
            )
        client = AsyncOpenAI(api_key=api_key, base_url=ns.base_url)

    agent = CliAgent(
        client,
        model=ns.model,
        max_tokens=ns.max_tokens,
        temperature=ns.temperature,
        reasoning=ns.reasoning,
        max_iterations=ns.max_iterations,
        tool_timeout=ns.tool_timeout,
        tool_output_limit=ns.tool_output_limit,
        registry=registry,
    )
    answer = await agent.run(messages)

    written = write_output(answer, ns.output_file)
    if not ns.quiet:
        target = str(written) if written is not None else "stdout"
        logger.info(
            f"done: {agent.iterations} iterations, {agent.tool_calls} tool calls, "
            f"{len(answer)} chars -> {target}"
        )
    return answer


def main(argv: Optional[list[str]] = None) -> int:
    ns = parse_args(argv)
    _configure_logging(ns.quiet)

    cwd = Path(ns.cwd).expanduser().resolve()
    if not cwd.is_dir():
        print(f"cli-agent.py: --cwd is not a directory: {ns.cwd}", file=sys.stderr)
        return 2
    os.chdir(cwd)
    from builtin_tools import fs_tools

    fs_tools._current_dir = cwd

    try:
        asyncio.run(amain(ns))
    except SystemExit as exc:  # отсутствующий ключ
        print(f"cli-agent.py: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"cli-agent.py: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logger.exception(f"run failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
