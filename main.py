import asyncio
from dotenv import load_dotenv
import sys


def global_exception_handler(exc_type, exc_value, exc_traceback):
    logger.exception("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = global_exception_handler

async def main() -> None:
    await app.start_app()


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())


