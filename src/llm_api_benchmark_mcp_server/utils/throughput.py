import time
import asyncio
from typing import Optional

from openai import AsyncOpenAI

from .openai_client import ask_openai, ask_openai_with_random_input
from .ttft import measure_ttft
from .statistics import calculate_statistics

async def _measure_throughput_worker(
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
    latency: float,
    generation_throughputs: list,
    prompt_throughputs: list,
    prompt: Optional[str] = None,
    num_words: Optional[int] = None
):
    start_time = time.time()
    try:
        if prompt is not None:
            resp = await ask_openai(client, model, prompt, max_tokens)
        elif num_words is not None:
            resp = await ask_openai_with_random_input(client, model, num_words, max_tokens)
        else:
            raise ValueError("Either prompt or num_words must be provided.")

        if resp.usage:
            duration = time.time() - start_time
            adjusted_duration = max(0.001, duration - (latency / 1000))
            generation_throughput = resp.usage.completion_tokens / adjusted_duration
            generation_throughputs.append(generation_throughput)
            prompt_throughput = resp.usage.prompt_tokens / adjusted_duration
            prompt_throughputs.append(prompt_throughput)
        else:
            print("Warning: No usage information in response.")
    except Exception as e:
        print(f"Error in worker: {e}")

async def measure_throughput(
    client: AsyncOpenAI,
    model: str,
    concurrency: int,
    max_tokens: int,
    latency: float,
    prompt: Optional[str] = None,
    num_words: Optional[int] = None
) -> dict:
    if prompt is not None:
        ttft_results = await measure_ttft(client, model, concurrency, prompt=prompt)
    elif num_words is not None:
        ttft_results = await measure_ttft(client, model, concurrency, num_words=num_words)
    else:
        raise ValueError("Either prompt or num_words must be provided.")

    generation_throughputs = []
    prompt_throughputs = []

    tasks = []
    for _ in range(concurrency):
        task = asyncio.create_task(_measure_throughput_worker(
            client, model, max_tokens, latency, generation_throughputs, prompt_throughputs, prompt, num_words
        ))
        tasks.append(task)

    await asyncio.gather(*tasks)

    return {
        "concurrency": concurrency,
        "generation_throughput_tokens_per_s": calculate_statistics(generation_throughputs),
        "prompt_throughput_tokens_per_s": calculate_statistics(prompt_throughputs),
        "ttft_s": ttft_results
    }
