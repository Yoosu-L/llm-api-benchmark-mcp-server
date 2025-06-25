import time
import asyncio
from asyncio import Queue
import numpy as np

import math

from openai import AsyncOpenAI

from utils.openai_client import ask_openai, ask_openai_with_random_input
from utils.ttft import measure_ttft, measure_ttft_with_random_input

def calculate_statistics(data):
    if not data:
        return {
            "total": 0.0,
            "avg": 0.0,
            "distribution": {
                "max": 0.0,
                "p50": 0.0,
                "p10": 0.0,
                "p1": 0.0,
                "min": 0.0
            }
        }
    data_np = np.array(data)
    sorted_data = np.sort(data_np)
    n = len(sorted_data)

    # Calculate p1 (actual value from sorted data)
    p1_index = min(n - 1, math.ceil(0.01 * n) - 1) if n > 0 else 0
    p1_actual = float(sorted_data[p1_index]) if n > 0 else 0.0

    # Calculate p10 (actual value from sorted data)
    p10_index = min(n - 1, math.ceil(0.10 * n) - 1) if n > 0 else 0
    p10_actual = float(sorted_data[p10_index]) if n > 0 else 0.0

    # Calculate p50 (median) - using 'lower' interpolation equivalent for actual value
    p50_index = math.floor(0.50 * (n - 1))
    p50_actual = float(sorted_data[p50_index])

    return {
        "total": float(np.sum(data_np)),
        "avg": float(np.mean(data_np)),
        "distribution": {
            "max": float(np.max(data_np)),
            "p50": p50_actual,
            "p10": p10_actual,
            "p1": p1_actual,
            "min": float(np.min(data_np))
        }
    }

async def measure_speed(client: AsyncOpenAI, model: str, prompt: str, concurrency: int, max_tokens: int, latency: float) -> dict:
    """
    Measures API generation throughput and TTFT.
    """
    ttft_results = await measure_ttft(client, model, prompt, concurrency)
    
    generation_throughputs = []
    prompt_throughputs = []

    async def worker():
        start_time = time.time()
        try:
            resp = await ask_openai(client, model, prompt, max_tokens)
            if resp.usage:
                duration = time.time() - start_time
                adjusted_duration = max(0.001, duration - (latency / 1000))
                
                generation_throughput = resp.usage.completion_tokens / adjusted_duration
                generation_throughputs.append(generation_throughput)

                # For prompt throughput, we need the TTFT for each individual request.
                # Since measure_ttft already handles this, we'll use the overall TTFT stats.
                # If individual prompt TTFT was needed, it would require a different approach.
                # For now, we'll calculate prompt throughput based on the total prompt tokens and overall duration.
                # This might not align perfectly with the "per request" prompt throughput,
                # but it's consistent with the current structure.
                prompt_throughput = resp.usage.prompt_tokens / adjusted_duration # Using adjusted_duration for consistency
                prompt_throughputs.append(prompt_throughput)

            else:
                print("Warning: No usage information in response.")
        except Exception as e:
            print(f"Error in worker (ask_openai): {e}")
            
    tasks = []
    for _ in range(concurrency):
        task = asyncio.create_task(worker())
        tasks.append(task)
        
    await asyncio.gather(*tasks)
        
    return {
        "concurrency": concurrency,
        "generation_throughput_tokens_per_s": calculate_statistics(generation_throughputs),
        "prompt_throughput_tokens_per_s": calculate_statistics(prompt_throughputs),
        "ttft_s": ttft_results
    }

async def measure_speed_with_random_input(client: AsyncOpenAI, model: str, num_words: int, concurrency: int, max_tokens: int, latency: float) -> dict:
    """
    Measures API generation throughput and TTFT with random input.
    """
    ttft_results = await measure_ttft_with_random_input(client, model, num_words, concurrency)
    
    generation_throughputs = []
    prompt_throughputs = []
    
    async def worker():
        start_time = time.time()
        try:
            resp = await ask_openai_with_random_input(client, model, num_words, max_tokens)
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
            print(f"Error in worker (ask_openai_with_random_input): {e}")
            
    tasks = []
    for _ in range(concurrency):
        task = asyncio.create_task(worker())
        tasks.append(task)
        
    await asyncio.gather(*tasks)
        
    return {
        "concurrency": concurrency,
        "generation_throughput_tokens_per_s": calculate_statistics(generation_throughputs),
        "prompt_throughput_tokens_per_s": calculate_statistics(prompt_throughputs),
        "ttft_s": ttft_results
    }