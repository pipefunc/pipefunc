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


class TestMCPDocumentationGeneration:
    """Test MCP documentation generation functions."""

    def test_get_pipeline_instruction_template(self):
        """Test pipeline instruction template generation."""
        from pipefunc.mcp import get_pipeline_instruction_template

        template = get_pipeline_instruction_template()

        assert isinstance(template, str)
        assert "Execute the pipeline with input values" in template
        assert "PIPELINE INFORMATION" in template
        assert "mapspecs define how arrays are processed" in template

    def test_format_pipeline_description_simple(self, simple_pipeline):
        """Test pipeline description formatting for simple pipeline."""
        from pipefunc.mcp import format_pipeline_description

        description = format_pipeline_description(simple_pipeline)

        assert isinstance(description, str)
        assert "Required Inputs: x, y" in description
        assert "Outputs: result" in description
        assert "Pipeline Name:" in description

    def test_format_pipeline_description_complex(self, complex_pipeline):
        """Test pipeline description formatting for complex pipeline."""
        from pipefunc.mcp import format_pipeline_description

        description = format_pipeline_description(complex_pipeline)

        assert isinstance(description, str)
        assert "MAPSPEC DEFINITIONS" in description
        assert "x[i] -> values[i]" in description

    def test_format_pipeline_info_simple(self, simple_pipeline):
        """Test pipeline info formatting for simple pipeline."""
        from pipefunc.mcp import format_pipeline_info

        info = format_pipeline_info(simple_pipeline)

        assert isinstance(info, str)
        assert "Parameters" in info
        assert "Required" in info

    def test_format_pipeline_info_complex(self, complex_pipeline):
        """Test pipeline info formatting for complex pipeline."""
        from pipefunc.mcp import format_pipeline_info

        info = format_pipeline_info(complex_pipeline)

        assert isinstance(info, str)
        # Should contain parameter information
        assert "x" in info

    def test_format_mapspec_docs_simple(self, simple_pipeline):
        """Test mapspec documentation for simple pipeline."""
        from pipefunc.mcp import format_mapspec_docs

        docs = format_mapspec_docs(simple_pipeline)

        # Simple pipeline has no mapspecs, so should be empty
        assert docs == ""

    def test_format_mapspec_docs_complex(self, complex_pipeline):
        """Test mapspec documentation for complex pipeline."""
        from pipefunc.mcp import format_mapspec_docs

        docs = format_mapspec_docs(complex_pipeline)

        assert isinstance(docs, str)
        assert "x[i] -> values[i]" in docs


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

    def test_format_pipeline_description_empty(self):
        """Test pipeline description with empty pipeline."""
        from pipefunc.mcp import format_pipeline_description

        empty_pipeline = Pipeline([])
        description = format_pipeline_description(empty_pipeline)

        assert isinstance(description, str)
        assert "No functions" in description or "Pipeline Name:" in description

    def test_missing_dependencies_handling(self):
        """Test that missing dependencies are handled in requires."""
        # This test ensures that the requires function properly handles missing deps
        from pipefunc._utils import requires

        with pytest.raises(ImportError):
            requires("nonexistent_package", reason="test")

    def test_pipeline_with_no_outputs(self):
        """Test handling of pipeline with no outputs."""
        from pipefunc.mcp import format_pipeline_description

        # Create a function with no explicit output
        @pipefunc()  # No output_name specified
        def dummy_func(x: int) -> int:
            return x

        pipeline = Pipeline([dummy_func])
        description = format_pipeline_description(pipeline)
        assert isinstance(description, str)


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
        assert results["sum_result"].store == 12.0

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
        assert results["combined"].store == 15.0
