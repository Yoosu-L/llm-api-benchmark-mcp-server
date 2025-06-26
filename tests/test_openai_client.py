import pytest
from unittest.mock import AsyncMock, MagicMock
from llm_api_benchmark_mcp_server.utils.openai_client import (
    ask_openai,
    ask_openai_with_random_input,
    get_first_available_model,
    estimate_input_tokens,
)


@pytest.mark.asyncio
async def test_ask_openai():
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = MagicMock()
    await ask_openai(mock_client, "test-model", "test-prompt", 10)
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_ask_openai_with_random_input():
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = MagicMock()
    await ask_openai_with_random_input(mock_client, "test-model", 10, 10)
    mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_get_first_available_model():
    mock_client = AsyncMock()
    mock_model = MagicMock()
    mock_model.id = "test-model"
    mock_client.models.list.return_value = MagicMock(data=[mock_model])
    model = await get_first_available_model(mock_client)
    assert model == "test-model"


@pytest.mark.asyncio
async def test_get_first_available_model_no_models():
    mock_client = AsyncMock()
    mock_client.models.list.return_value = MagicMock(data=[])
    with pytest.raises(ValueError):
        await get_first_available_model(mock_client)


@pytest.mark.asyncio
async def test_estimate_input_tokens_with_prompt():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_client.chat.completions.create.return_value = mock_response
    tokens = await estimate_input_tokens(mock_client, "test-model", "test-prompt", 0)
    assert tokens == 10


@pytest.mark.asyncio
async def test_estimate_input_tokens_with_num_words():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.usage.prompt_tokens = 20
    mock_client.chat.completions.create.return_value = mock_response
    tokens = await estimate_input_tokens(mock_client, "test-model", "", 10)
    assert tokens == 20
