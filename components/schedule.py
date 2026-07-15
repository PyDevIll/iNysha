from lib.scheduler import get_scheduler

# ---- Processing logic for scheduled tasks ----
async def process_scheduled_task(task: dict) -> None:
    """Run agent with a scheduled prompt."""
    chat_id = task["chat_id"]
    prompt = task["prompt"]
    task_id = task["id"]
    created_at = task.get("created_at", "unknown")
    logger.info(f"Processing scheduled task {task_id}: {prompt[:100]}")

    # Prefix with metadata so LLM can distinguish from user messages
    prefixed_prompt = (
        f"[Scheduled task #{task_id}]\n"
        f"Scheduled at: {created_at}\n"
        f"---\n"
        f"{prompt}"
    )

    async def reasoning_callback(thought: str) -> None:
        await bot.send_reasoning(thought)

    response = await agent.run_with_crash_recovery(
        initial_user_request=prefixed_prompt,
        reasoning_callback=reasoning_callback,
    )

    if response:
        await bot.send_reply(chat_id, f"🕒 **Scheduled task result:**\n\n{response}")

    scheduler = get_scheduler()
    scheduler.mark_done(task["id"])
