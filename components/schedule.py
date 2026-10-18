from loguru import logger

from lib.scheduler import get_scheduler


# ---- Processing logic for scheduled tasks ----
async def process_scheduled_task(task: dict) -> None:
    """Run the agent with a scheduled prompt and report the result."""
    from app import get_agent
    from tool_registry import get_registry

    chat_id = task.get("chat_id")
    prompt = task["prompt"]
    task_id = task["id"]
    created_at = task.get("created_at", "unknown")

    logger.info(f"Processing scheduled task {task_id}: {prompt[:100]}")

    bot = get_registry().get_bot()
    agent = get_agent()

    # Prefix with metadata so the LLM can distinguish it from user messages.
    prefixed_prompt = (
        f"[Scheduled task #{task_id}]\n"
        f"Scheduled at: {created_at}\n"
        f"---\n"
        f"{prompt}"
    )

    async def reasoning_callback(thought: str) -> None:
        if bot:
            await bot.send_reasoning(thought)

    response = await agent.run_with_crash_recovery(
        initial_user_request=prefixed_prompt,
        reasoning_callback=reasoning_callback,
    )

    if response and bot and chat_id:
        await bot.send_reply(chat_id, f"🕒 **Scheduled task result:**\n\n{response}")

    get_scheduler().mark_done(task_id)
