import pytest
import respx
from httpx import Response
from llm_api_benchmark_mcp_server.utils.latency import measure_latency


@respx.mock
@pytest.mark.asyncio
async def test_measure_latency_success():
    base_url = "http://test.com"
    respx.get(f"{base_url}/").mock(return_value=Response(200))
    latency = await measure_latency(base_url, 1)
    assert isinstance(latency, float)
    assert latency > 0


@respx.mock
@pytest.mark.asyncio
async def test_measure_latency_connection_error():
    base_url = "http://test.com"
    respx.get(f"{base_url}/").mock(side_effect=ConnectionError("Test error"))
    with pytest.raises(ConnectionError):
        await measure_latency(base_url, 1)


@pytest.mark.asyncio
async def test_measure_latency_invalid_url():
    with pytest.raises(ValueError):
        await measure_latency("invalid-url", 1)
