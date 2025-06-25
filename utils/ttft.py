import time
import asyncio
from asyncio import Queue
import numpy as np
import math

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from utils.random_prompt import generate_random_phrase

def calculate_statistics(data):
    if not data:
        return {
            "avg": 0.0,
            "distribution": {
                "min": 0.0,
                "p50": 0.0,
                "p90": 0.0,
                "p99": 0.0,
                "max": 0.0
            }
        }
    data_np = np.array(data)
    sorted_data = np.sort(data_np)
    n = len(sorted_data)

    # Calculate p50 (median) - using 'lower' interpolation equivalent for actual value
    p50_index = math.floor(0.50 * (n - 1))
    p50_actual = float(sorted_data[p50_index])

    # Calculate p90 (actual value from sorted data)
    p90_index = min(n - 1, math.ceil(0.90 * n) - 1) if n > 0 else 0
    p90_actual = float(sorted_data[p90_index]) if n > 0 else 0.0

    # Calculate p99 (actual value from sorted data)
    p99_index = min(n - 1, math.ceil(0.99 * n) - 1) if n > 0 else 0
    p99_actual = float(sorted_data[p99_index]) if n > 0 else 0.0

    return {
        "avg": float(np.mean(data_np)),
        "distribution": {
            "min": float(np.min(data_np)),
            "p50": p50_actual,
            "p90": p90_actual,
            "p99": p99_actual,
            "max": float(np.max(data_np))
        }
    }

async def measure_ttft(client: AsyncOpenAI, model: str, prompt: str, concurrency: int) -> dict:
    """
    Calculates the Time to First Token (TTFT) statistics for API responses.
    """
    ttft_values = []
    ttft_queue = Queue()
    
    async def worker():
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        
        start = time.time()
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=512,
                temperature=1,
                stream=True,
            )
            
            # Listen for the first response
            await anext(stream) # Iterate to get the first chunk
            
            ttft = time.time() - start
            await ttft_queue.put(ttft)
            
            # Consume the rest of the stream to ensure connection closure
            async for _ in stream:
                pass
            
        except Exception as e:
            print(f"TTFT Stream error: {e}")
            # Optionally put a sentinel value or handle error in queue
            
    tasks = []
    for _ in range(concurrency):
        task = asyncio.create_task(worker())
        tasks.append(task)
        
    await asyncio.gather(*tasks)
        
    while not ttft_queue.empty():
        ttft_values.append(ttft_queue.get_nowait())
            
    return calculate_statistics(ttft_values)

async def measure_ttft_with_random_input(client: AsyncOpenAI, model: str, num_words: int, concurrency: int) -> dict:
    """
    Calculates the Time to First Token (TTFT) statistics for API responses with random input.
    """
    ttft_values = []
    ttft_queue = Queue()
    
    async def worker():
        prompt = generate_random_phrase(num_words)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        
        start = time.time()
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=512,
                temperature=1,
                stream=True,
            )
            
            # Listen for the first response
            await anext(stream) # Iterate to get the first chunk
            
            ttft = time.time() - start
            await ttft_queue.put(ttft)
            
            # Consume the rest of the stream to ensure connection closure
            async for _ in stream:
                pass
            
        except Exception as e:
            print(f"TTFT Stream error: {e}")
            # Optionally put a sentinel value or handle error in queue
            
    tasks = []
    for _ in range(concurrency):
        task = asyncio.create_task(worker())
        tasks.append(task)
        
    await asyncio.gather(*tasks)
        
    while not ttft_queue.empty():
        ttft_values.append(ttft_queue.get_nowait())
            
    return calculate_statistics(ttft_values)