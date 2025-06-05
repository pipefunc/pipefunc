from __future__ import annotations

from pipefunc._pipeline._autodoc import PipelineDocumentation, format_pipeline_docs
from pipefunc._pipeline._base import Pipeline
from pipefunc._utils import requires

_PIPELINE_DESCRIPTION_TEMPLATE = """\
Execute the pipeline with input values. This method works for both single values and arrays/lists.

PIPELINE INFORMATION:
{pipeline_info}

MAPSPEC DEFINITIONS
{mapspec_section}

INPUT FORMAT:
{input_format}

DETAILED PIPELINE DOCUMENTATION:
{documentation}
"""

_NO_MAPSPEC_INPUT_FORMAT = """\
Single values only:
  {"a": 5, "b": 10, "x": 2}
  → Each parameter gets a single value

This will execute the pipeline once with these specific values and return the result.
"""

_MAPSPEC_INPUT_FORMAT = """\

1. Simple element-wise mapping:
   {"x": [1, 2, 3, 4]}
   → If function has mapspec "x[i] -> y[i]", this will process each x value independently

2. Cross-product of inputs:
   {"a": [1, 2], "b": [10, 20]}
   → If functions have mapspecs like "a[i], b[j] -> result[i, j]", this creates all combinations

3. Zipped inputs (same index):
   {"x": [1, 2, 3], "y": [4, 5, 6]}
   → If function has mapspec "x[i], y[i] -> z[i]", this pairs x[0] with y[0], x[1] with y[1], etc.

4. Mixed single values and arrays:
   {"data": [1, 2, 3, 4], "multiplier": 10}
   → Arrays are mapped over, single values are used for all iterations

Key concepts:
- mapspec defines how inputs map to outputs (e.g., "x[i] -> y[i]" means element-wise)
- Arrays with same index letter (like [i]) are processed together
- Arrays with different indices (like [i] and [j]) create cross-products
- Single values work regardless of mapspecs
"""


def _get_pipeline_documentation(pipeline: Pipeline) -> str:
    """Generate formatted pipeline documentation tables using Rich."""
    requires("rich", "griffe", reason="mcp", extras="autodoc")
    from rich.console import Console

    doc = PipelineDocumentation.from_pipeline(pipeline)
    tables = format_pipeline_docs(doc, print_table=False)
    assert tables is not None

    console = Console(no_color=True)
    with console.capture() as capture:
        for table in tables:
            console.print(table, "\n")
    return capture.get()


def _get_pipeline_info_summary(pipeline_name: str, pipeline: Pipeline) -> str:
    """Generate a summary of pipeline information."""
    info = pipeline.info()
    assert info is not None

    def _format(key: str) -> str:
        return ", ".join(info[key]) if info[key] else "None"

    lines = [
        f"Pipeline Name: {pipeline_name}",
        f"Required Inputs: {_format('required_inputs')}",
        f"Optional Inputs: {_format('optional_inputs')}",
        f"Outputs: {_format('outputs')}",
        f"Intermediate Outputs: {_format('intermediate_outputs')}",
    ]
    return "\n".join(lines)


def _get_mapspec_section(pipeline: Pipeline) -> str:
    """Generate mapspec information section."""
    mapspecs = pipeline.mapspecs_as_strings
    if not mapspecs:
        return "None (This pipeline processes single values only)"

    lines = [
        "The following mapspecs define how arrays are processed:",
    ]

    for i, mapspec in enumerate(mapspecs, 1):
        lines.append(f"  {i}. {mapspec}")

    lines.extend(
        [
            "",
            "Mapspec Legend:",
            "- Parameters with [i], [j], etc. represent array dimensions",
            "- Same index letter (e.g., [i]) means elements are processed together (zipped)",
            "- Different indices (e.g., [i] and [j]) create cross-products",
            "- Parameters without indices are used as single values for all iterations",
        ],
    )

    return "\n".join(lines)


def _get_input_format_section(pipeline: Pipeline) -> str:
    """Generate input format examples section."""
    mapspecs = pipeline.mapspecs_as_strings
    return _MAPSPEC_INPUT_FORMAT if mapspecs else _NO_MAPSPEC_INPUT_FORMAT


def _format_tool_description(
    pipeline_info: str,
    mapspec_section: str,
    input_format: str,
    documentation: str,
) -> str:
    """Format a complete tool description using the template."""
    return _PIPELINE_DESCRIPTION_TEMPLATE.format(
        pipeline_info=pipeline_info,
        mapspec_section=mapspec_section,
        input_format=input_format,
        documentation=documentation,
    )


def build_mcp_server(pipeline_name: str, pipeline: Pipeline):
    requires("mcp", "rich", "griffe", reason="mcp", extras="mcp")
    from fastmcp import Context, FastMCP

    # Generate all pipeline information sections
    documentation = _get_pipeline_documentation(pipeline)
    pipeline_info = _get_pipeline_info_summary(pipeline_name, pipeline)
    mapspec_section = _get_mapspec_section(pipeline)
    input_format = _get_input_format_section(pipeline)

    # Format description using the template
    description = _format_tool_description(
        pipeline_info=pipeline_info,
        mapspec_section=mapspec_section,
        input_format=input_format,
        documentation=documentation,
    )

    Model = pipeline.pydantic_model()  # noqa: N806
    Model.model_rebuild()  # Ensure all type references are resolved
    mcp = FastMCP(name=pipeline_name, version="0.1.0")

    @mcp.tool(
        name="execute_pipeline",
        description=description,
    )
    def execute_pipeline(
        ctx: Context,
        input: Model,
        parallel: bool = True,
        run_folder: str | None = None,
    ) -> str:
        """Execute pipeline with input values (works for both single values and arrays)."""
        result = pipeline.map(
            inputs=input,
            parallel=parallel,
            run_folder=run_folder,
        )
        # Convert ResultDict to a more readable format
        output = {}
        for key, result_obj in result.items():
            output[key] = {
                "output": result_obj.output.tolist()
                if hasattr(result_obj.output, "tolist")
                else result_obj.output,
                "shape": getattr(result_obj.output, "shape", None),
            }
        return str(output)

    return mcp


def run_mcp_server(pipeline_name: str, pipeline: Pipeline):
    mcp = build_mcp_server(pipeline_name, pipeline)
    mcp.run(transport="stdio")
