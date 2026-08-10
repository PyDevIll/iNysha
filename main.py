import asyncio
from dotenv import load_dotenv
import app

async def test_vision():
    from builtin_tools.vision_tools import vision_analyze_url
    result = await vision_analyze_url(image_url="https://upload.wikimedia.org/wikipedia/commons/8/80/Epica_Simone_Simons.jpg", query="Who is she? She is nice, isn't she?")
    print(result)


async def test_screenshot():
    from builtin_tools.vision_tools import get_screenshot
    await get_screenshot(2)
    input()
    await get_screenshot(1)
    input()
    await get_screenshot(0)


async def test_queue():
    q = asyncio.Queue()
    q.put_nowait("a")
    q.put_nowait("b")
    for num, item in enumerate(list(q._queue)):
        print(f" {num}: {item}")
    print("---")
    while True:
        try:
            print(q.get_nowait())
            await asyncio.sleep(0.5)
        except asyncio.QueueEmpty as e:
            break
    print("---")
    for num, item in enumerate(list(q._queue)):
        print(f" {num}: {item}")


async def test_get_yandex_models():
    import requests
    import os
    import json
    response = requests.get(
        url='https://ai.api.cloud.yandex.net/v1/models',
        headers={"Authorization" : f"Bearer {os.getenv('YANDEXGPT_API_KEY')}" }
    )
    print(json.dumps(response.json(), indent=4))


async def main() -> None:
    await app.start_app()

    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())

    # asyncio.run(test_get_yandex_models())
    # asyncio.run(test_vision())
    # asyncio.run(test_screenshot())
    # asyncio.run(test_queue())
