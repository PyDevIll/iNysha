import asyncio
from dotenv import load_dotenv
import app


async def main() -> None:
    await app.start_app()

    while True:
        await asyncio.sleep(1)

async def test():
    from builtin_tools.tts_tools import tts_generate
    await tts_generate("Сгенерирован+о и сохранен+о в отдельн+ый файл")

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())
    # asyncio.run(test())



