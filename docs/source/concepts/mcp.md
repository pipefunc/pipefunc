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

The {func}`~pipefunc.mcp.build_mcp_server` function exposes PipeFunc pipelines as [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) servers.
This allows AI agents and assistants to use your ``pipefunc.Pipeline``s as tools.

A complete Series Analyzer example later in this guide walks through building, wiring, and consuming an MCP server end-to-end.

The server automatically extracts your pipeline's type annotations and docstrings to generate:
- **Pydantic models** for input validation (from type hints)
- **Tool descriptions** for AI assistants (from docstrings)
- **Parameter schemas** with types, defaults, and documentation

Additionally provides:
- Synchronous and asynchronous execution modes
- Job management for long-running computations
- Multiple transport protocols (HTTP, SSE, stdio)

```{note}
MCP servers work with JSON-serializable inputs. Complex types (NumPy arrays, etc.) are automatically converted to/from JSON-compatible formats.
```

---
## How It Works

When you call `build_mcp_server(pipeline)`:

1. **Pydantic Model Generation**: Uses {meth}`pipeline.pydantic_model() <pipefunc.Pipeline.pydantic_model>` to create validation schemas from type annotations
2. **Docstring Extraction**: Parses function docstrings (NumPy/Google/Sphinx style) to extract parameter descriptions
3. **Tool Registration**: Creates MCP tools with the generated schemas and descriptions
4. **Job Registry**: Sets up a global registry for tracking async jobs

The server exposes these tools:

- **`execute_pipeline_sync`**: Run pipeline and return results immediately (blocking)
- **`execute_pipeline_async`**: Start background execution and return a job ID
- **`check_job_status`**: Monitor async job progress and retrieve results
- **`list_jobs`**: View all tracked jobs in the current session
- **`cancel_job`**: Stop a running async job
- **`run_info`**: Inspect any pipeline run folder on disk
- **`list_historical_runs`**: Browse previous pipeline executions
- **`load_outputs`**: Load results from completed runs

Every tool returns JSON-serializable dictionaries so AI clients can surface rich status information without additional parsing.

---

## Execution Modes

### Synchronous Execution

Blocks until completion and returns results:

```python
# Client calls:
execute_pipeline_sync(
    inputs={"series": [2.0, 2.1, 2.2, 2.05, 2.15, 2.08, 2.12, 2.18, 50.0]},
    parallel=False,
)
# Returns:
# {
#   "clean_series": {"output": [2.0, 2.1, 2.2, 2.05, 2.15, 2.08, 2.12, 2.18, 50.0], "shape": null},
#   "summary": {
#     "output": {
#       "count": 9,
#       "mean": 7.431111111111111,
#       "median": 2.12,
#       "min": 2.0,
#       "max": 50.0,
#       "std": 15.050490907050504,
#       "range": 48.0
#     },
#     "shape": null
#   },
#   "anomalies": {
#     "output": [
#       {"index": 8, "value": 50.0, "z_score": 2.828405342509274}
#     ],
#     "shape": null
#   }
# }
```

### Asynchronous Execution

Returns immediately with a job ID for tracking:

```python
# Start job:
execute_pipeline_async(
    inputs={
        "series": [18, 17.5, 18.2, 17.8, 55.0, 18.1, 17.9],
        "z_threshold": 2.0,
    }
)
# Returns: {"job_id": "9d41...", "run_folder": "runs/job_9d41..."}

# Check status:
check_job_status(job_id="uuid-string")
# Returns: {"status": "running", "progress": {...}, ...}

# Cancel if needed:
cancel_job(job_id="uuid-string")
# Returns: {"status": "cancelled", "job_id": "uuid-string"}
```

By default, asynchronous runs are saved under `runs/job_<job_id>`; pass `run_folder` explicitly to change the destination.

---

## Client Configuration

Configure your AI client to connect to the server. For example, in Cursor IDE's `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "series-analyzer": {
      "url": "http://127.0.0.1:8000/series"
    }
  }
}
```

Or for Claude Desktop's `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "series-analyzer": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"]
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

## AI Agent Example

This example shows how to create an AI agent using [Agno](https://github.com/agno-agi/agno) that can use your pipeline as a tool. The pipeline, *Series Analyzer*, sanitizes numeric inputs, computes descriptive statistics, and flags anomalies based on z-scores:

Create `mcp_server.py` with inline dependencies using [uv script](https://docs.astral.sh/uv/guides/scripts/):

```{literalinclude} mcp_server.py
:language: python
```

Create `mcp_agent.py` to connect to the server:

```{literalinclude} mcp_agent.py
:language: python
```

Create a `.env` file with your credentials (for example `ANTHROPIC_API_KEY=your-api-key`) and run:

```bash
uv run --script mcp_agent.py
```

The `.env` file is loaded automatically, so no manual exporting is required. `uv` manages the dependencies declared in the script, and the agent will automatically discover and use the `execute_pipeline_sync` tool from your MCP server.

---


## Dependencies

Install MCP support with:

```bash
pip install "pipefunc[mcp]"
```

This installs:

- **fastmcp** – MCP server framework
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

The MCP server handles parameter validation, type coercion, job tracking, and result formatting automatically based on your pipeline's type annotations and docstrings.
