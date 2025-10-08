#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["agno", "anthropic", "python-dotenv", "mcp"]
# ///

"""Example Agno agent that consumes the Series Analyzer MCP server."""

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.tools.mcp import MCPTools
from dotenv import load_dotenv

load_dotenv()


agent = Agent(
    model=Claude(id="claude-sonnet-4-5"),
    tools=[
        MCPTools(command="uv run --script mcp_server.py", transport="stdio"),
    ],
    instructions=[
        "You are a data-analysis assistant.",
        "When a user shares numbers, call the Series Analyzer MCP tool to compute statistics and flag anomalies.",
        "Explain your findings using the returned summary and anomalies list.",
    ],
    markdown=True,
)


if __name__ == "__main__":
    agent.print_response(
        "Analyze the series [3, 3.2, 3.1, 10.5, 3.0, 2.9, 9.8]. Highlight anomalies and summarize the data. Use the tool to get the results.",
        stream=True,
    )
