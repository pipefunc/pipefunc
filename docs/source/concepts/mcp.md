---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# MCP Server Integration

The {func}`~pipefunc.mcp.build_mcp_server` function exposes your PipeFunc pipelines as [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers, enabling AI assistants like Claude, ChatGPT, and others to execute your computational workflows directly.
With a single function call, your pipeline becomes an AI-accessible tool with automatic parameter validation, job management, and progress tracking.

**Key capabilities:**

- **Automatic Tool Generation**: Pipeline parameters and documentation become MCP tool schemas
- **Dual Execution Modes**: Synchronous (blocking) and asynchronous (background) execution
- **Job Management**: Track, monitor, and cancel long-running async jobs
- **Type Safety**: Automatic validation using Pydantic models from your type hints
- **Multiple Transports**: HTTP, Server-Sent Events (SSE), and stdio support

```{note}
MCP servers work with JSON-serializable inputs. Complex types (NumPy arrays, etc.) are automatically converted to/from JSON-compatible formats.
```

---

## Quick Start

Here's a minimal example of creating and running an MCP server:

```{code-cell} ipython3
from pipefunc import Pipeline, pipefunc
from pipefunc.mcp import build_mcp_server

@pipefunc(output_name="result")
def add(x: float, y: float) -> float:
    """Add two numbers together."""
    return x + y

pipeline = Pipeline([add])
mcp = build_mcp_server(pipeline)

# For testing, we can inspect the server
print(f"Server name: {mcp.name}")
print(f"Available tools: {[tool.name for tool in mcp.list_tools()]}")
```

To actually run the server (not executed in docs):

```python
if __name__ == "__main__":
    mcp.run(path="/add", port=8000, transport="streamable-http")
```

---

## How It Works

When you call `build_mcp_server(pipeline)`, the following happens automatically:

1. **Schema Generation**: Creates Pydantic models from your pipeline's type annotations and default values
2. **Tool Registration**: Registers execution tools (`execute_pipeline_sync`, `execute_pipeline_async`) with detailed descriptions
3. **Documentation Extraction**: Pulls parameter descriptions from your docstrings for tool help text
4. **Job Management Setup**: Configures async job tracking with status monitoring and cancellation

The MCP server exposes several tools:

- **`execute_pipeline_sync`**: Run pipeline and return results immediately (blocking)
- **`execute_pipeline_async`**: Start background execution and return a job ID
- **`check_job_status`**: Monitor async job progress and retrieve results
- **`list_jobs`**: View all tracked jobs in the current session
- **`cancel_job`**: Stop a running async job
- **`run_info`**: Inspect any pipeline run folder on disk
- **`list_historical_runs`**: Browse previous pipeline executions
- **`load_outputs`**: Load results from completed runs

---

## Execution Modes

### Synchronous Execution

Best for quick computations where you want immediate results:

```python
# AI assistant calls:
execute_pipeline_sync(inputs={"x": 5, "y": 3})
# Returns: {'result': {'output': 8.0, 'shape': None}}
```

### Asynchronous Execution

Ideal for long-running pipelines, parameter sweeps, or when you need progress tracking:

```python
# AI assistant starts job:
execute_pipeline_async(inputs={"x": [1, 2, 3], "y": [4, 5, 6]})
# Returns: {"job_id": "uuid-string", "run_folder": "runs/job_uuid"}

# Check status:
check_job_status(job_id="uuid-string")
# Returns: {"status": "running", "progress": {...}, ...}

# Cancel if needed:
cancel_job(job_id="uuid-string")
```

---

## Client Configuration

Configure your AI assistant to use the server. For example, in Cursor IDE's `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "my-pipeline": {
      "url": "http://127.0.0.1:8000/add"
    }
  }
}
```

Or for Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "my-pipeline": {
      "command": "python",
      "args": ["/path/to/my_mcp_server.py"]
    }
  }
}
```

---

## Transport Options

The server supports multiple transport methods:

**HTTP (recommended for development):**

```python
mcp.run(path="/api", port=8000, transport="streamable-http")
```

**Server-Sent Events:**

```python
mcp.run(transport="sse")
```

**Standard I/O (for subprocess integration):**

```python
mcp.run(transport="stdio")
```

---

## Complete Example

A more realistic example with parameter sweeps and mapspecs:

```{code-cell} ipython3
%%writefile mcp_example.py

from pipefunc import Pipeline, pipefunc
from pipefunc.mcp import build_mcp_server

@pipefunc(output_name="squared", mapspec="x[i] -> squared[i]")
def square(x: float) -> float:
    """
    Square a number.

    Parameters
    ----------
    x : float
        Input value to square.

    Returns
    -------
    float
        The squared value.
    """
    return x ** 2

@pipefunc(output_name="sum_of_squares")
def sum_squares(squared: list[float]) -> float:
    """
    Sum all squared values.

    Parameters
    ----------
    squared : list[float]
        List of squared values.

    Returns
    -------
    float
        The sum of all squares.
    """
    return sum(squared)

pipeline = Pipeline([square, sum_squares])

if __name__ == "__main__":
    mcp = build_mcp_server(pipeline, name="Square Summer")
    mcp.run(path="/squares", port=8000, transport="streamable-http")
```

The AI assistant can now call:

```python
execute_pipeline_async(inputs={"x": [1, 2, 3, 4, 5]})
```

And receive results for the entire parameter sweep with progress tracking.

---

## Dependencies

Install MCP support with:

```bash
pip install "pipefunc[mcp]"
```

This installs:

- **fastmcp** (≥2.12.0) – MCP server framework
- **Rich** – Enhanced terminal output
- **Griffe** – Docstring parsing for parameter descriptions

---

## Best Practices

```{admonition} Production Deployment
:class: tip
- Use environment variables for sensitive configuration
- Enable CORS properly for HTTP servers
- Consider rate limiting for public endpoints
- Use `run_folder` to organize outputs by job/date
```

```{admonition} Parallel Execution
:class: warning
When using parallel execution with `pipeline.map()`, ensure your pipeline module is importable and functions are defined at module level. Use `if __name__ == "__main__":` to wrap the server startup.
```

---

By exposing your pipelines as MCP servers, you make complex computational workflows accessible to AI assistants with minimal setup.
The server handles all the complexity of parameter validation, type coercion, job tracking, and result formatting, so you can focus on building great pipelines.
