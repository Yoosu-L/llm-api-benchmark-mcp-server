from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from typing import Optional

from .random_prompt import generate_random_phrase

async def ask_openai(client: AsyncOpenAI, model: str, prompt: str, max_tokens: int) -> ChatCompletion:
    """
    Sends a prompt to the OpenAI API and retrieves the response.
    """
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=1,
    )
    return resp

async def ask_openai_with_random_input(client: AsyncOpenAI, model: str, num_words: int, max_tokens: int) -> ChatCompletion:
    """
    Sends a prompt with random input to the OpenAI API and retrieves the response.
    """
    prompt = generate_random_phrase(num_words)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=1,
    )
    return resp

async def get_first_available_model(client: AsyncOpenAI) -> str:
    """
    Retrieves the first available model from the OpenAI API.
    """
    model_list = await client.models.list()
    
    if not model_list.data:
        raise ValueError("No models available")
    
    return model_list.data[0].id

async def estimate_input_tokens(client: AsyncOpenAI, model_name: str, prompt: str, num_words: int) -> int:
    """
    Estimates the number of input tokens for a given prompt or random input.
    """
    if num_words > 0:
        resp = await ask_openai_with_random_input(client, model_name, num_words, 1)
        if resp.usage:
            return resp.usage.prompt_tokens
        return 0 # Or raise an error, depending on desired behavior
    
    resp = await ask_openai(client, model_name, prompt, 1)
    if resp.usage:
        return resp.usage.prompt_tokens
    return 0 # Or raise an error