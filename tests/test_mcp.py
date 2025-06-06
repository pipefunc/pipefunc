"""Test suite for MCP server functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pipefunc import Pipeline, pipefunc

if TYPE_CHECKING:
    from pipefunc.typing import Array

# Check for optional dependencies
fastmcp = pytest.importorskip("fastmcp", reason="fastmcp not available")
pydantic = pytest.importorskip("pydantic", reason="pydantic not available")


@pytest.fixture
def simple_pipeline():
    """Create a simple pipeline for testing."""

    @pipefunc(output_name="result")
    def add_numbers(x: float, y: float) -> float:
        return x + y

    return Pipeline([add_numbers])


@pytest.fixture
def complex_pipeline():
    """Create a complex pipeline with mapspecs for testing."""

    @pipefunc(output_name="values", mapspec="x[i] -> values[i]")
    def compute_values(x: float) -> float:
        return x * 2.0

    @pipefunc(output_name="sum_result")
    def sum_values(values: Array) -> float:
        return float(sum(values))

    return Pipeline([compute_values, sum_values])


# Test MCP server building functionality.


@pytest.mark.asyncio
async def test_build_mcp_server_simple(simple_pipeline):
    """Test building MCP server for simple pipeline."""
    from fastmcp import Client

    from pipefunc.mcp import build_mcp_server

    mcp = build_mcp_server(simple_pipeline)
    assert mcp is not None
    async with Client(mcp) as client:
        result = await client.call_tool("execute_pipeline", {"input": {"x": 1, "y": 2}})
        assert result[0].text == "{'result': {'output': 3.0, 'shape': None}}"


# Test MCP internal helper functions


def test_get_pipeline_documentation(simple_pipeline):
    """Test pipeline documentation generation."""
    from pipefunc.mcp import _get_pipeline_documentation

    docs = _get_pipeline_documentation(simple_pipeline)
    assert isinstance(docs, str)
    assert len(docs) > 0


def test_get_pipeline_info_summary(simple_pipeline):
    """Test pipeline info summary generation."""
    from pipefunc.mcp import _get_pipeline_info_summary

    summary = _get_pipeline_info_summary("Test Pipeline", simple_pipeline)
    assert isinstance(summary, str)
    assert "Pipeline Name: Test Pipeline" in summary
    assert "Required Inputs:" in summary
    assert "Outputs:" in summary


def test_get_mapspec_section_simple(simple_pipeline):
    """Test mapspec section for simple pipeline."""
    from pipefunc.mcp import _get_mapspec_section

    section = _get_mapspec_section(simple_pipeline)
    assert isinstance(section, str)
    assert "None (This pipeline processes single values only)" in section


def test_get_mapspec_section_complex(complex_pipeline):
    """Test mapspec section for complex pipeline."""
    from pipefunc.mcp import _get_mapspec_section

    section = _get_mapspec_section(complex_pipeline)
    assert isinstance(section, str)
    assert "mapspecs define how arrays are processed" in section


def test_get_input_format_section_simple(simple_pipeline):
    """Test input format section for simple pipeline."""
    from pipefunc.mcp import _get_input_format_section

    section = _get_input_format_section(simple_pipeline)
    assert isinstance(section, str)
    assert "Single values only" in section


def test_get_input_format_section_complex(complex_pipeline):
    """Test input format section for complex pipeline."""
    from pipefunc.mcp import _get_input_format_section

    section = _get_input_format_section(complex_pipeline)
    assert isinstance(section, str)
    assert "element-wise mapping" in section


def test_format_tool_description():
    """Test tool description formatting."""
    from pipefunc.mcp import _format_tool_description

    description = _format_tool_description(
        pipeline_info="Test info",
        mapspec_section="Test mapspec",
        input_format="Test format",
        documentation="Test docs",
    )
    assert isinstance(description, str)
    assert "Test info" in description
    assert "Test mapspec" in description
    assert "Test format" in description
    assert "Test docs" in description


# Test MCP error handling and edge cases.


def test_empty_pipeline_handling():
    """Test handling of empty pipeline."""
    from pipefunc.mcp import _get_pipeline_info_summary

    empty_pipeline = Pipeline([])
    summary = _get_pipeline_info_summary("Empty", empty_pipeline)

    assert isinstance(summary, str)
    assert "Pipeline Name: Empty" in summary


def test_pipeline_with_no_outputs():
    """Test handling of pipeline with no outputs."""
    from pipefunc.mcp import _get_pipeline_info_summary

    # Create a function with minimal configuration
    @pipefunc(output_name="dummy_output")
    def dummy_func(x: int) -> int:
        return x

    pipeline = Pipeline([dummy_func])
    summary = _get_pipeline_info_summary("Test", pipeline)
    assert isinstance(summary, str)
    assert "dummy_output" in summary


def test_pipeline_constants_template_access():
    """Test that MCP constants and templates are accessible."""
    from pipefunc.mcp import (
        _MAPSPEC_INPUT_FORMAT,
        _NO_MAPSPEC_INPUT_FORMAT,
        _PIPEFUNC_INSTRUCTIONS,
        _PIPELINE_DESCRIPTION_TEMPLATE,
    )

    # Test that all constants are strings
    assert isinstance(_PIPEFUNC_INSTRUCTIONS, str)
    assert isinstance(_PIPELINE_DESCRIPTION_TEMPLATE, str)
    assert isinstance(_NO_MAPSPEC_INPUT_FORMAT, str)
    assert isinstance(_MAPSPEC_INPUT_FORMAT, str)

    # Test that they contain expected content
    assert "MCP server" in _PIPEFUNC_INSTRUCTIONS
    assert "PIPELINE INFORMATION" in _PIPELINE_DESCRIPTION_TEMPLATE
    assert "Single values only" in _NO_MAPSPEC_INPUT_FORMAT
    assert "element-wise mapping" in _MAPSPEC_INPUT_FORMAT
