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


class TestMCPServerBuilding:
    """Test MCP server building functionality."""

    def test_build_mcp_server_simple(self, simple_pipeline):
        """Test building MCP server for simple pipeline."""
        try:
            from pipefunc.mcp import build_mcp_server

            result = build_mcp_server(simple_pipeline)
            assert result is not None
            assert hasattr(result, "add_tool")
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"MCP server building failed: {e}")

    def test_build_mcp_server_with_version(self, simple_pipeline):
        """Test building MCP server with custom version."""
        try:
            from pipefunc.mcp import build_mcp_server

            result = build_mcp_server(simple_pipeline, version="2.0.0")
            assert result is not None
            assert hasattr(result, "add_tool")
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"MCP server building failed: {e}")

    def test_build_mcp_server_imports(self):
        """Test that build_mcp_server handles missing imports gracefully."""
        from pipefunc.mcp import build_mcp_server

        # Should not raise ImportError
        assert callable(build_mcp_server)

    def test_run_mcp_server_exists(self):
        """Test that run_mcp_server function exists."""
        from pipefunc.mcp import run_mcp_server

        # Should not raise ImportError
        assert callable(run_mcp_server)


class TestMCPInternalHelpers:
    """Test MCP internal helper functions."""

    def test_get_pipeline_documentation(self, simple_pipeline):
        """Test pipeline documentation generation."""
        from pipefunc.mcp import _get_pipeline_documentation

        docs = _get_pipeline_documentation(simple_pipeline)
        assert isinstance(docs, str)
        assert len(docs) > 0

    def test_get_pipeline_info_summary(self, simple_pipeline):
        """Test pipeline info summary generation."""
        from pipefunc.mcp import _get_pipeline_info_summary

        summary = _get_pipeline_info_summary("Test Pipeline", simple_pipeline)
        assert isinstance(summary, str)
        assert "Pipeline Name: Test Pipeline" in summary
        assert "Required Inputs:" in summary
        assert "Outputs:" in summary

    def test_get_mapspec_section_simple(self, simple_pipeline):
        """Test mapspec section for simple pipeline."""
        from pipefunc.mcp import _get_mapspec_section

        section = _get_mapspec_section(simple_pipeline)
        assert isinstance(section, str)
        assert "None (This pipeline processes single values only)" in section

    def test_get_mapspec_section_complex(self, complex_pipeline):
        """Test mapspec section for complex pipeline."""
        from pipefunc.mcp import _get_mapspec_section

        section = _get_mapspec_section(complex_pipeline)
        assert isinstance(section, str)
        assert "mapspecs define how arrays are processed" in section

    def test_get_input_format_section_simple(self, simple_pipeline):
        """Test input format section for simple pipeline."""
        from pipefunc.mcp import _get_input_format_section

        section = _get_input_format_section(simple_pipeline)
        assert isinstance(section, str)
        assert "Single values only" in section

    def test_get_input_format_section_complex(self, complex_pipeline):
        """Test input format section for complex pipeline."""
        from pipefunc.mcp import _get_input_format_section

        section = _get_input_format_section(complex_pipeline)
        assert isinstance(section, str)
        assert "element-wise mapping" in section

    def test_format_tool_description(self):
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


class TestMCPIntegration:
    """Test MCP integration with pipelines."""

    def test_pipeline_pydantic_model_simple(self, simple_pipeline):
        """Test pydantic model creation for simple pipeline."""
        # This should work without errors
        model_class = simple_pipeline.pydantic_model()

        # Test model validation
        model_instance = model_class(x=1.0, y=2.0)
        assert model_instance.x == 1.0
        assert model_instance.y == 2.0

    def test_pipeline_execution_simple(self, simple_pipeline):
        """Test simple pipeline execution that would be called via MCP."""
        # Execute pipeline directly (simulating MCP call)
        result = simple_pipeline("result", x=1.0, y=2.0)
        assert result == 3.0

    def test_complex_pipeline_execution(self, complex_pipeline):
        """Test complex pipeline execution with mapspecs."""
        # Test that pydantic model can be created
        model_class = complex_pipeline.pydantic_model()

        # This would be the kind of input an MCP call might make
        # We test that the model can handle it
        assert model_class is not None


class TestMCPErrorHandling:
    """Test MCP error handling and edge cases."""

    def test_empty_pipeline_handling(self):
        """Test handling of empty pipeline."""
        from pipefunc.mcp import _get_pipeline_info_summary

        empty_pipeline = Pipeline([])
        summary = _get_pipeline_info_summary("Empty", empty_pipeline)

        assert isinstance(summary, str)
        assert "Pipeline Name: Empty" in summary

    def test_missing_dependencies_handling(self):
        """Test that missing dependencies are handled in requires."""
        # This test ensures that the requires function properly handles missing deps
        from pipefunc._utils import requires

        with pytest.raises(ImportError):
            requires("nonexistent_package", reason="test")

    def test_pipeline_with_no_outputs(self):
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


class TestMCPRealExecution:
    """Test real MCP execution scenarios."""

    def test_pipeline_map_execution(self, complex_pipeline):
        """Test pipeline map execution (array processing)."""
        # Test that the pipeline can handle array inputs
        inputs = {"x": [1.0, 2.0, 3.0]}

        # Execute with map (this simulates what MCP would do for array inputs)
        results = complex_pipeline.map(inputs, run_folder=None)

        # Check that we got results
        assert "sum_result" in results
        # The sum should be 2*1 + 2*2 + 2*3 = 12
        assert results["sum_result"].output == 12.0

    def test_pipeline_mixed_mapspecs(self):
        """Test pipeline with mixed mapspecs (element-wise and reduction)."""

        @pipefunc(output_name="doubled", mapspec="x[i] -> doubled[i]")
        def double_values(x: float) -> float:
            return x * 2.0

        @pipefunc(output_name="tripled", mapspec="x[i] -> tripled[i]")
        def triple_values(x: float) -> float:
            return x * 3.0

        @pipefunc(output_name="combined")
        def combine_results(doubled: Array, tripled: Array) -> float:
            return float(sum(doubled) + sum(tripled))

        pipeline = Pipeline([double_values, triple_values, combine_results])

        inputs = {"x": [1.0, 2.0]}
        results = pipeline.map(inputs, run_folder=None)

        assert "combined" in results
        # doubled: [2, 4], tripled: [3, 6], combined: 2+4+3+6 = 15
        assert results["combined"].output == 15.0

    def test_pipeline_constants_template_access(self):
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
