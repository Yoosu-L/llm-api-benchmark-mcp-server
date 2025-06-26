import time
import httpx # Use httpx for async requests
from urllib.parse import urlparse

async def measure_latency(base_url: str, num_requests: int) -> float:
    """
    Tests the speed of the API by making multiple requests and calculating the average latency.
    """
    if not base_url:
        raise ValueError("Empty base URL")
        
    parsed_url = urlparse(base_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid base URL: {base_url}")
        
    total_latency = 0.0
    
    async with httpx.AsyncClient() as client:
        for _ in range(num_requests):
            start_time = time.time()
            try:
                response = await client.get(f"{parsed_url.scheme}://{parsed_url.netloc}", timeout=5)
                response.raise_for_status()  # Raise an exception for HTTP errors
                end_time = time.time()
                total_latency += (end_time - start_time) * 1000  # Convert to milliseconds
            except httpx.RequestError as e:
                raise ConnectionError(f"Failed to connect to {base_url}: {e}")
                
    return total_latency / num_requests