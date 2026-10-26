"""MASTERMIND v2 — Core async agent loop. ContextPool v3 integrated."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI
from loguru import logger

from context_manager import ContextPool, count_tokens, repair_tool_sequence
from server import config as app_config
from server.context import is_voice_context
from tool_registry import get_registry

LLM_MAX_OUTPUT_TOKENS = app_config.LLM_MAX_OUTPUT_TOKENS


def _resolve_base_url() -> str:
    return app_config.DEEPSEEK_BASE_URL


def _resolve_model() -> str:
    # The API uses hyphens ("deepseek-flash"); config accepts underscores too.
    return app_config.DEEPSEEK_MODEL

def _render_prompt_text(text: str) -> str:
    """Expand central-config placeholders in a prompt file."""
    for key, value in app_config.prompt_substitutions().items():
        text = text.replace(key, value)
    return text


def construct_history(prompts_list: list[tuple[str, Optional[str]]]) -> list[dict]:
    """Build base prompt messages from text/file tuples."""
    composed = []
    for prompt_text, prompt_file in prompts_list:
        content = prompt_text
        if prompt_file:
            try:
                fp = Path(prompt_file)
                if not fp.is_absolute():
                    fp = Path(__file__).resolve().parent / fp
                content += fp.read_text(encoding='utf-8')
            except Exception:
                logger.warning(f"File {prompt_file} not loaded")
        content = _render_prompt_text(content)
        content += "\n\n---\n\n"
        composed.append({
            "role": "system",
            "name": app_config.AGENT_DISPLAY_NAME,
            "content": content,
        })
    return composed


def validate_message_sequence(messages: list[dict]) -> None:
    """Validate the assistant ``tool_calls`` ↔ ``tool`` result pairing.

    Providers require each assistant tool-call group to be followed immediately
    by exactly one ``tool`` message per ``tool_call_id``. Both a missing reply
    and an orphaned reply invalidate the whole request, so check both
    directions. Raises ValueError with details on the first violation.
    """
    # tool_call_id -> index of the assistant message that declared it
    pending: dict[str, int] = {}

    for i, msg in enumerate(messages):
        role = msg.get("role")

        if role == "tool":
            tc_id = msg.get("tool_call_id")
            if tc_id not in pending:
                prev = messages[i - 1] if i > 0 else None
                raise ValueError(
                    f"Orphaned tool message at index {i} with tool_call_id={tc_id}. "
                    f"Preceding message: {prev} (if any)."
                )
            del pending[tc_id]
            continue

        if pending:
            raise ValueError(
                f"Assistant message(s) at index(es) "
                f"{sorted(set(pending.values()))} have unanswered tool_calls "
                f"{sorted(pending)} before index {i} (role={role})."
            )

        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                call_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if not call_id:
                    raise ValueError(
                        f"Assistant message at index {i} has a tool_call without id."
                    )
                pending[call_id] = i

    if pending:
        raise ValueError(
            f"Assistant message(s) at index(es) "
            f"{sorted(set(pending.values()))} end the history with unanswered "
            f"tool_calls {sorted(pending)}."
        )


TOOL_OUTPUT_LIMIT = 2000   # chars kept from a single tool result


def _limit_tool_output(result, limit: int = TOOL_OUTPUT_LIMIT) -> str:
    """Bound how much of a tool result enters the context."""
    text = "" if result is None else str(result)
    if len(text) <= limit:
        return text
    return (
        text[:limit]
        + f"\n\n[... truncated: shown first {limit} of {len(text)} chars — "
          f"narrow the request or fetch the rest if needed]"
    )


class Agent:
    """Async reasoning agent with hot-reload tools, crash recovery, and layered context."""

    def __init__(
        self,
        name: str = "MASTERMIND",
        system_prompt: str = "",
        base_prompts: Optional[list] = None,
        last_memory: Optional[list] = None,
        use_tools: bool = True,
        save_history: bool = True,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.name = name
        self._max_tokens = LLM_MAX_OUTPUT_TOKENS
        self._client = AsyncOpenAI(
            api_key=api_key or app_config.DEEPSEEK_API_KEY,
            base_url=base_url or _resolve_base_url(),
        )
        self._model = model or _resolve_model()
        self._system_prompt = {"role": "system", "name": "Creator", "content": system_prompt}
        self._base_prompts = base_prompts or []
        self._last_memory = last_memory or []
        self._use_tools = use_tools
        self._save_history = save_history
        self._helper_agent: Optional[Agent] = None
        self.messages = ContextPool()
        self.messages.set_base_prompt_tokens(self._static_prompt_tokens())

        # Load legacy last_memory (for backward compatibility). ContextPool
        # отфильтрует system/base-промпты, а прошлая выжимка из
        # last_compression.txt уже загружена один раз в _compressed_dicts.
        if self._last_memory:
            self.messages.assign_messages(construct_history(self._last_memory))

        logger.info(f"Agent '{name}' initialized ({self._model}), "
                     f"crash recovery: {self.messages.length > 0}")

    def add_helper_agent(self, helper: Agent) -> None:
        self._helper_agent = helper

    def _static_prompt_tokens(self) -> int:
        """Token cost of the system prompt plus all base prompt files."""
        base_messages = construct_history(self._base_prompts)
        total = count_tokens(self._system_prompt["content"])
        total += sum(
            count_tokens(message.get("content") or "")
            for message in base_messages
        )
        return total

    @property
    def registry(self):
        """Get the current global tool registry (always up‑to‑date)."""
        return get_registry()

    async def _execute_single_tool(self, tool_call, user_request: str = "") -> dict:
        """Execute one tool call and return the result message."""
        func_name = tool_call.function.name
        registry = get_registry()   # always live
        voice_mode = is_voice_context()
        logger.debug(
            f"_execute_single_tool: '{func_name}' | voice={voice_mode} | "
            f"registry v{registry._version} | tools: {len(registry.tool_names)}"
        )
        if registry.get_tool(func_name) is None:
            logger.error(
                f"_execute_single_tool: '{func_name}' NOT registered! "
                f"Available: {sorted(registry.tool_names)}"
            )
        elif not registry.is_tool_allowed(func_name, voice_mode=voice_mode):
            if voice_mode:
                hint = (
                    "This tool is not available for direct calls in a voice "
                    "turn. Use start_tool to run it in the background and keep "
                    "talking."
                )
            else:
                hint = (
                    "This helper is available only during voice turns; call "
                    "the tool directly in a text turn."
                )
            logger.info(f"Tool policy rejected direct '{func_name}' (voice={voice_mode})")
            return {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": func_name,
                "content": f"Error: {hint}",
            }

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error for {func_name}: {e}")
            return {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": func_name,
                "content": f"Error: invalid JSON arguments — {e}",
            }

        logger.info(f"Tool: {func_name}({json.dumps(args, ensure_ascii=False)[:200]})")
        try:
            result = await asyncio.wait_for(
                registry.call_tool(func_name, **args),
                timeout=120.0  # 2-minute timeout per tool
            )
        except asyncio.TimeoutError:
            logger.error(f"Tool '{func_name}' timed out after 120s")
            return {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": func_name,
                "content": f"Error: tool '{func_name}' timed out (>120s). "
                           f"Try narrowing the search scope.",
            }

        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": func_name,
            "content": _limit_tool_output(result),
        }

    async def _execute_tools_parallel(self, tool_calls, user_request: str = "") -> list[dict]:
        """Execute multiple tool calls concurrently."""
        tasks = [self._execute_single_tool(tc, user_request) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                output.append({
                    "tool_call_id": tool_calls[i].id,
                    "role": "tool",
                    "name": tool_calls[i].function.name,
                    "content": f"Error: {result}",
                })
            else:
                output.append(result)
        return output

    def _build_messages_for_llm(self) -> list[dict]:
        """Build full message list with all memory layers.

        Layer assembly lives in ``ContextPool.build_llm_messages()`` so the
        agent, tests and diagnostics all use the same path.  Order is
        DeepSeek cache-optimized:

          system → base prompts → persistent → compressed → masked → sliding
        """
        self.messages.log_state("STATE AT build_messages_for_llm")

        messages = self.messages.build_llm_messages(
            self._system_prompt,
            construct_history(self._base_prompts),
        )

        # ---- NORMALISE: guarantee a valid tool-call / tool-result pairing ----
        # A crash between persisting an assistant tool call and its result (or a
        # window/compression boundary cutting a group in half) must never reach
        # the provider: one unanswered tool_call rejects the entire request 400.
        fixed_messages = repair_tool_sequence(messages)

        # Validate the sequence (now should pass)
        try:
            validate_message_sequence(fixed_messages)
        except ValueError as e:
            logger.error(f"Message sequence validation failed: {e}")
            with open("invalid_messages.json", "w", encoding='utf-8') as f:
                json.dump(fixed_messages, f, indent=2, default=str)
            raise

        return fixed_messages

    @staticmethod
    def _is_voice_turn() -> bool:
        """True for a voice turn, including its deferred tool-result turns."""
        try:
            return is_voice_context()
        except Exception:
            return False

    async def llm_request(self, enable_reasoning) -> dict:
        """Make an async LLM API call with layered context."""
        messages = self._build_messages_for_llm()

        ctx_len = sum(len(m.get("content", "") or "") + len(m.get("reasoning_content", "") or "")
                      for m in messages)
        est_tokens = ctx_len // 3
        logger.info(f"LLM request: {len(messages)} msgs, ~{est_tokens} tokens")

        registry = get_registry()   # always live
        voice_mode = self._is_voice_turn()
        tools = registry.get_openai_tools(voice_mode=voice_mode) if self._use_tools else []
        if tools:
            logger.debug(f"llm_request: passing {len(tools)} tools")
            logger.trace(
                f"llm_request: tools={[t['function']['name'] for t in tools]}"
            )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools or None,
                stream=False,
                max_tokens=self._max_tokens,
                temperature=1.0,
                extra_body={
                    "thinking": {"type": "enabled" if enable_reasoning else "disabled"}
                },
            )
        except Exception as e:
            logger.exception(f"LLM API call failed: {e}")
            import json
            with open("failed_messages.json", "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=2, default=str)
            raise

        if response is None:
            raise ValueError("API returned None response")
        if not hasattr(response, "choices") or not response.choices:
            raise ValueError(f"Response missing 'choices': {response}")

        response_dict = response.model_dump()
        if not response_dict.get("choices"):
            logger.error(f"LLM response has no choices: {response_dict}")
            raise ValueError(f"Invalid LLM response: {response_dict}")
        return response_dict

    async def run(self, initial_user_request: str = "", reasoning_callback=None) -> str:
        """
        Main agent loop.
        Returns final response text.
        reasoning_callback: async callable(thought_text) for live reasoning output.
        """
        max_iterations = 15
        iteration = 0
        enable_reasoning = bool(reasoning_callback)

        # A background compression rewrites _all_entries. Wait for it before
        # appending anything, otherwise a new turn interleaves with the trim.
        await self.messages.await_compression()

        # Add user message to context
        if initial_user_request:
            self.messages.append({
                "role": "user",
                "name": "User",
                "content": initial_user_request,
            }, save=True)

        # Proactive compression: if context overflowed before LLM call, compress now
        if self.messages.overflow and self._helper_agent:
            logger.warning(
                f"Overflow before LLM request ({self.messages.get_context_length()} tokens) — "
                "starting background compression"
            )
            self.messages.start_background_compression(self._helper_agent)

        while iteration < max_iterations:
            iteration += 1

            # Check for new tools (hot-reload) – always use live registry
            registry = get_registry()
            if registry.version > 0:
                logger.debug(f"Registry v{registry.version}, {len(registry.tool_names)} tools ready")

            try:
                response = await self.llm_request(enable_reasoning)
            except Exception as e:
                logger.exception(f"LLM request failed at iteration {iteration}")
                self.messages.log_state("FAILED LLM REQUEST", detailed=True)
                import json
                with open("failed_context.json", "w", encoding='utf-8') as f:
                    json.dump([self.messages._entry_to_dict(e) for e in self.messages._all_entries], f, indent=2,
                              default=str)
                raise

            choice = response["choices"][0]
            message = choice["message"]

            # Extract reasoning if present
            reasoning = message.get("reasoning_content", "")
            if reasoning and reasoning_callback:
                await reasoning_callback(reasoning)

            has_tool_calls = bool(message.get("tool_calls"))
            logger.trace(
                f"run: has_tool_calls={has_tool_calls}, "
                f"message keys={list(message.keys())}"
            )
            self.messages.append_assistant_message(message, self._save_history, has_tool_calls)

            if has_tool_calls:
                tool_calls_raw = message["tool_calls"]
                # Convert dicts to objects for compatibility
                tool_call_objects = []
                for tc in tool_calls_raw:
                    class ToolCallObj:
                        pass
                    obj = ToolCallObj()
                    obj.id = tc["id"]
                    obj.function = type('fn', (), {})()
                    obj.function.name = tc["function"]["name"]
                    obj.function.arguments = tc["function"]["arguments"]
                    tool_call_objects.append(obj)

                # Execute tools in parallel
                tool_results = await self._execute_tools_parallel(
                    tool_call_objects, initial_user_request
                )

                for result in tool_results:
                    self.messages.append(result, save=True)

                # Check if context overflowed — trigger compression
                if self.messages.overflow and self._helper_agent:
                    logger.info("Context overflow after tools – starting background compression")
                    self.messages.start_background_compression(self._helper_agent)
            else:
                # Final response — no tool calls
                content = message.get("content", "")
                logger.info(f"Agent finished in {iteration} iterations, "
                           f"context: {self.messages.length} msgs, "
                           f"~{self.messages.get_context_length()} tokens")
                return content

        logger.warning(f"Max iterations ({max_iterations}) reached")
        return "Max iterations reached without final response."

    async def run_with_crash_recovery(self, initial_user_request: str = "", reasoning_callback=None) -> str:
        """Run agent with crash recovery."""
        try:
            return await self.run(initial_user_request, reasoning_callback)
        except Exception as e:
            logger.exception(f"Agent crashed: {e}")
            self.messages._emergency_save()
            raise


def NOW() -> str:
    return datetime.now().strftime("%d.%m.%Y, %H:%M")
