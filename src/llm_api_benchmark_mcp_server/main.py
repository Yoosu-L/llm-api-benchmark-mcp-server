import logging
import sys # Import sys module
import gc # Import garbage collector module
from typing import Optional, Dict, Any

from openai import AsyncOpenAI

from mcp.server.fastmcp import FastMCP
from mcp import types as mcp_types

# Import utilities from the project
from .utils.openai_client import get_first_available_model, estimate_input_tokens
from .utils.concurrency import parse_concurrency_levels
from .utils.latency import measure_latency
from .utils.throughput import measure_throughput

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

mcp = FastMCP("llm-api-benchmark-mcp-server")

@mcp.tool(
    name="run_llm_benchmark",
    description="Runs a throughput benchmark for LLM APIs, measuring generation throughput, prompt throughput, and Time To First Token (TTFT) under various concurrency levels."
)
async def run_llm_benchmark(
    base_url: str,
    api_key: str = "sk-dummy",
    model: Optional[str] = "",
    prompt: str = "Write a long story, no less than 10,000 words, starting from a long, long time ago.",
    num_words: int = 0,
    concurrency: str = "1",
    max_tokens: int = 512
) -> Dict[str, Any]:
    """
    Runs the LLM API throughput benchmark.

    Parameters:
        base_url: str - Base URL of the OpenAI API.
        api_key: str - API key for authentication (default: sk-dummy).
        model: Optional[str] - Model to be used for the requests (optional, will be discovered if not provided).
        prompt: str - Prompt to be used for generating responses (default: a long story prompt).
        num_words: int - Number of words for random input (if prompt is default and num_words > 0).
        concurrency: str - Comma-separated list of concurrency levels (default: "1").
        max_tokens: int - Maximum number of tokens to generate (default: 512).

    Returns:
        dict - Dictionary containing benchmark results and metadata.
    """
    output_messages = []
    original_stdout = sys.stdout # Store original stdout before redirection

    class OutputCapturer:
        def __init__(self, max_lines: int = 1000):
            self.max_lines = max_lines
            self.buffer = []

        def write(self, msg):
            # Only capture non-empty messages (e.g., ignore newline characters if they are sent separately)
            if msg.strip():
                self.buffer.append(msg.strip())
                # Keep only the last max_lines
                if len(self.buffer) > self.max_lines:
                    self.buffer = self.buffer[-self.max_lines:]
            # Always write to the original stdout for immediate feedback
            original_stdout.write(msg)

        def flush(self):
            original_stdout.flush()

    # Temporarily redirect sys.stdout to capture output
    capturer = OutputCapturer(max_lines=1000)
    sys.stdout = capturer

    client = None # Initialize client to None
    try:
        # Parse concurrency levels
        try:
            concurrency_levels = parse_concurrency_levels(concurrency)
        except ValueError as e:
            logging.fatal(f"Invalid concurrency levels: {e}")
            return {"error": f"Invalid concurrency levels: {e}", "output_log": capturer.buffer} # Note: This line was output_messages before, now it's capturer.buffer

        # Initialize OpenAI client and ensure it's properly closed
        async with AsyncOpenAI(base_url=base_url, api_key=api_key) as initialized_client: # Use a temporary name for the client in the 'with' statement
            client = initialized_client # Assign to the outer 'client' variable
            model_name = model
            if not model_name:
                try:
                    discovered_model = await get_first_available_model(client)
                    model_name = discovered_model
                except Exception as e:
                    logging.error(f"Error discovering model: {e}")
                    return {"error": f"Error discovering model: {e}", "output_log": capturer.buffer}

            # Determine input parameters
            use_random_input = False
            if prompt == "Write a long story, no less than 10,000 words, starting from a long, long time ago." and num_words != 0:
                use_random_input = True
            
            input_tokens = 0
            try:
                if use_random_input:
                    input_tokens = await estimate_input_tokens(client, model_name, "", num_words // 4)
                else:
                    input_tokens = await estimate_input_tokens(client, model_name, prompt, 0)
            except Exception as e:
                logging.fatal(f"Error getting prompt tokens: {e}")
                return {"error": f"Error getting prompt tokens: {e}", "output_log": capturer.buffer}

            # Test latency
            latency = 0.0
            try:
                latency = await measure_latency(base_url, 5)
            except Exception as e:
                logging.warning(f"Latency test error: {e}")

            benchmark_results = []
            for conc in concurrency_levels:
                if use_random_input:
                    benchmark_result_item = await measure_throughput(
                        client, model_name, conc, max_tokens, latency, num_words=num_words // 4
                    )
                else:
                    benchmark_result_item = await measure_throughput(
                        client, model_name, conc, max_tokens, latency, prompt=prompt
                    )
                benchmark_results.append(benchmark_result_item)

            # Round all float values in the benchmark_results dictionary
            def round_floats_in_dict(obj):
                if isinstance(obj, dict):
                    return {k: round_floats_in_dict(v) for k, v in obj.items()}
                elif isinstance(obj, float):
                    return round(obj, 2)
                else:
                    return obj

            return {
                "model_name": model_name,
                "input_tokens": input_tokens,
                "max_tokens": max_tokens,
                "latency_ms": round(latency, 2),
                "benchmark_results": [round_floats_in_dict(r) for r in benchmark_results],
                "output_log": capturer.buffer # Use the buffer from the capturer instance
            }
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
        # Explicitly delete client and run garbage collection for debugging memory leak
        if 'client' in locals() and client:
            await client.close() # Ensure the client is explicitly closed
            del client
        gc.collect()


# --- Define MCP Tool Schema ---
LLM_BENCHMARK_TOOL_SCHEMA = mcp_types.Tool(
    name="run_llm_benchmark",
    description="Runs a throughput benchmark for LLM APIs, measuring generation throughput, prompt throughput, and Time To First Token (TTFT) under various concurrency levels.",
    inputSchema={
        "type": "object",
        "properties": {
            "base_url": {
                "type": "string",
                "description": "Base URL of the OpenAI API (e.g., http://localhost:8000/v1)"
            },
            "api_key": {
                "type": "string",
                "description": "API key for authentication (default: sk-dummy)",
                "default": "sk-dummy"
            },
            "model": {
                "type": "string",
                "description": "Model to be used for the requests (optional, will be discovered if not provided)",
                "default": ""
            },
            "prompt": {
                "type": "string",
                "description": "Prompt to be used for generating responses (default: a long story prompt). If num_words is provided and not 0, random input will be used instead.",
                "default": "Write a long story, no less than 10,000 words, starting from a long, long time ago."
            },
            "num_words": {
                "type": "integer",
                "description": "Number of words for random input. If > 0 and default prompt is used, random input will be generated.",
                "default": 0,
                "minimum": 0
            },
            "concurrency": {
                "type": "string",
                "description": "Comma-separated list of concurrency levels (e.g., '1,2,4,8').",
                "default": "1"
            },
            "max_tokens": {
                "type": "integer",
                "description": "Maximum number of tokens to generate.",
                "default": 512,
                "minimum": 1
            }
        },
        "required": [
            "base_url"
        ]
    }
)
# --- End of MCP Tool Schema ---

def main():
    # Patch the tool's schema immediately to include all metadata
    tool = mcp._tool_manager.get_tool("run_llm_benchmark")
    if tool: # Ensure tool is not None before patching
        tool.parameters = LLM_BENCHMARK_TOOL_SCHEMA.inputSchema
        
    # # For sse transport
    # mcp.settings.host = "0.0.0.0"
    # mcp.settings.port = 47564
    # mcp.run(transport="sse")
    
    # For stdio transport
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()