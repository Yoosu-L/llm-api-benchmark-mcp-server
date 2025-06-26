import time
import asyncio
from asyncio import Queue
from typing import Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from .random_prompt import generate_random_phrase
from .statistics import calculate_ttft_statistics

async def _measure_ttft_worker(
    client: AsyncOpenAI,
    model: str,
    messages: list[ChatCompletionMessageParam],
    ttft_queue: Queue
):
    start = time.time()
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512,
            temperature=1,
            stream=True,
        )
        await anext(stream)
        ttft = time.time() - start
        await ttft_queue.put(ttft)
        async for _ in stream:
            pass
    except Exception as e:
        print(f"TTFT Stream error: {e}")

async def measure_ttft(
    client: AsyncOpenAI,
    model: str,
    concurrency: int,
    prompt: Optional[str] = None,
    num_words: Optional[int] = None
) -> dict:
    ttft_values = []
    ttft_queue = Queue()

    if prompt is not None:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    elif num_words is not None:
        random_prompt = generate_random_phrase(num_words)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": random_prompt},
        ]
    else:
        raise ValueError("Either prompt or num_words must be provided.")

    tasks = []
    for _ in range(concurrency):
        task = asyncio.create_task(_measure_ttft_worker(client, model, messages, ttft_queue))
        tasks.append(task)

    await asyncio.gather(*tasks)

    while not ttft_queue.empty():
        ttft_values.append(ttft_queue.get_nowait())

    return calculate_ttft_statistics(ttft_values)
