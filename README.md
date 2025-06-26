# LLM API Benchmark MCP Server

This project provides an MCP (Model Context Protocol) server designed to benchmark Large Language Model (LLM) APIs. It allows you to measure various performance metrics such as generation throughput, prompt throughput, and Time To First Token (TTFT) for your LLM endpoints.

## Features

*   **Comprehensive Benchmarking:** Measure key performance indicators like generation throughput, prompt throughput, and Time To First Token (TTFT).
*   **Flexible Deployment:** Supports both remote SSE server for quick trials and local deployment via stdio or SSE transport.
*   **Customizable Benchmarks:** Configure various parameters for your LLM API benchmarks, including concurrency levels, model names, token limits, and more.
*   **Detailed Output:** Provides structured JSON output with aggregated and distributed metrics for in-depth analysis.
*   **Easy Integration:** Designed as an MCP server for seamless integration with MCP clients.

## Table of Contents

- [LLM API Benchmark MCP Server](#llm-api-benchmark-mcp-server)
  - [Features](#features)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
    - [Install uv](#install-uv)
  - [Quick Start](#quick-start)
    - [Demo (Recommended for Simple Trials)](#demo-recommended-for-simple-trials)
    - [Local Deployment (stdio)](#local-deployment-stdio)
    - [Local Deployment (sse)](#local-deployment-sse)
  - [Usage Example](#usage-example)
  - [Benchmark Parameters](#benchmark-parameters)
  - [License](#license)

## Prerequisites

This project uses [`uv`](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) for environment management. Please ensure you have `uv` installed on your system.

### Install uv

Refer to the [official `uv` documentation](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) for installation methods.

## Quick Start

You can experience this MCP server through a remote SSE server for quick trials or deploy it locally using either stdio or SSE transport.

### Demo (Recommended for Simple Trials)

If you wish to quickly try out the MCP server, a public SSE server is available. You can configure your MCP client to connect to it directly.

**Note**: While this project does not collect your API endpoint or API key, please be cautious about potential exposure risks when using remote services.

**MCP Configuration:**

```json
{
  "mcpServers": {
    "llm-benchmark-sse": {
      "url": "https://llm-api-benchmark.pikoo.de/sse"
    }
  }
}
```

### Local Deployment (stdio)

This is the recommended method for local deployment.

**MCP Configuration:**

```json
{
  "mcpServers": {
    "llm-benchmark-stdio":{
      "command": "uvx",
      "args": [
        "--refresh",
        "--quiet",
        "llm-api-benchmark-mcp-server"
      ]
    }
  }
}
```

### Local Deployment (sse)

Alternatively, you can deploy the server locally using SSE transport.

1.  **Clone the repository and navigate into it:**

    ```sh
    git clone https://github.com/Yoosu-L/llm-api-benchmark-mcp-server.git
    cd llm-api-benchmark-mcp-server
    ```

2.  **Modify `src/llm_api_benchmark_mcp_server/main.py`:**
    Comment out the stdio transport section and uncomment the SSE transport section. You can also change the `mcp.settings.port` to your desired port.

3.  **Build and Start the MCP Server (SSE):**

    ```sh
    uv build
    uv tool install dist/llm_api_benchmark_mcp_server-0.1.3-py3-none-any.whl # path may varies
    llm-api-benchmark-mcp-server
    ```

4.  **Configure MCP Client:**
    Update your MCP client configuration to connect to your local SSE server.

    ```json
    {
      "mcpServers": {
        "llm-benchmark-sse": {
          "url": "http://localhost:47564/sse"
        }
      }
    }
    ```

## Usage Example

Once the MCP server is configured and running, you can use it to perform LLM API benchmarks.

**Example Prompt:**

```
Please help me perform a LLM api benchmark on this address with concurrency levels of 1 and 2. https://my-llm-api-service.com/v1, sk-xxx
```

**Example Output from MCP Tools:**

```json
{
  "model_name": "gemini-2.0-flash",
  "input_tokens": 32,
  "max_tokens": 512,
  "latency_ms": 2.46,
  "benchmark_results": [
    {
      "concurrency": 1,
      "generation_throughput_tokens_per_s": {
        "total": 135.74,
        "avg": 135.74,
        "distribution": {
          "max": 135.74,
          "p50": 135.74,
          "p10": 135.74,
          "p1": 135.74,
          "min": 135.74
        }
      },
      "prompt_throughput_tokens_per_s": {
        "total": 8.48,
        "avg": 8.48,
        "distribution": {
          "max": 8.48,
          "p50": 8.48,
          "p10": 8.48,
          "p1": 8.48,
          "min": 8.48
        }
      },
      "ttft_s": {
        "avg": 0.41,
        "distribution": {
          "min": 0.41,
          "p50": 0.41,
          "p90": 0.41,
          "p99": 0.41,
          "max": 0.41
        }
      }
    },
    {
      "concurrency": 2,
      "generation_throughput_tokens_per_s": {
        "total": 247.6,
        "avg": 123.8,
        "distribution": {
          "max": 124.07,
          "p50": 123.53,
          "p10": 123.53,
          "p1": 123.53,
          "min": 123.53
        }
        
      },
      "prompt_throughput_tokens_per_s": {
        "total": 15.52,
        "avg": 7.76,
        "distribution": {
          "max": 7.78,
          "p50": 7.74,
          "p10": 7.74,
          "p1": 7.74,
          "min": 7.74
        }
      },
      "ttft_s": {
        "avg": 0.68,
        "distribution": {
          "min": 0.43,
          "p50": 0.43,
          "p90": 0.94,
          "p99": 0.94,
          "max": 0.94
        }
      }
    }
  ],
  "output_log": []
}
```


## Benchmark Parameters

The `run_llm_benchmark` MCP tool accepts the following parameters:

*   **`base_url`** (string, **required**): Base URL of the OpenAI API endpoint (e.g., `http://localhost:8000/v1`).
*   **`api_key`** (string, optional, default: `sk-dummy`): API key for authentication.
*   **`model`** (string, optional, default: `""`): Model to be used for the requests. If not provided, the server will attempt to discover the first available model.
*   **`prompt`** (string, optional, default: `"Write a long story, no less than 10,000 words, starting from a long, long time ago."`): Prompt to be used for generating responses. If `num_words` is provided and greater than 0, random input will be generated instead of using this prompt.
*   **`num_words`** (integer, optional, default: `0`, minimum: `0`): Number of words for random input. If greater than 0 and the default prompt is used, random input will be generated.
*   **`concurrency`** (string, optional, default: `"1"`): Comma-separated list of concurrency levels (e.g., `"1,2,4,8"`).
*   **`max_tokens`** (integer, optional, default: `512`, minimum: `1`): Maximum number of tokens to generate.


## License

This project is licensed under the [MIT License](LICENSE).