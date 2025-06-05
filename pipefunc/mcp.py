from __future__ import annotations

from pipefunc._pipeline._autodoc import PipelineDocumentation, format_pipeline_docs
from pipefunc._pipeline._base import Pipeline
from pipefunc._pipeline._pydantic import maybe_pydantic_model_to_dict
from pipefunc._utils import requires

_PIPELINE_DESCRIPTION_TEMPLATE = """\
{method_description}

PIPELINE INFORMATION:
{pipeline_info}

{mapspec_section}

DETAILED PIPELINE DOCUMENTATION:
{documentation}
"""

_RUN_PIPELINE_DESCRIPTION = """\
Execute the pipeline SEQUENTIALLY with single input values.

Use this method when:
- No mapspec is used in the entire pipeline!
- You have single values (not arrays/lists) for each input parameter
- You want sequential execution (one step after another)
- You need a single result from the pipeline

Input format: Provide single values for each parameter, e.g.:
{"a": 5, "b": 10, "x": 2}

This will execute the pipeline once with these specific values and return the result.
"""

_MAP_PIPELINE_DESCRIPTION = """\
Execute the pipeline in PARALLEL with arrays/lists of input values using mapspec.

Use this method when:
- You have arrays/lists of values for input parameters
- You want parallel execution across parameter combinations
- You need results for multiple input combinations
- Your pipeline functions use mapspec (e.g., "x[i] -> y[i]")

Input format examples:

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
- Functions without mapspec receive entire arrays or single values

Returns: A dictionary with all pipeline outputs, where each output contains the computed results.
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

    lines = [
        f"Pipeline Name: {pipeline_name}",
        f"Required Inputs: {', '.join(info['required_inputs']) if info['required_inputs'] else 'None'}",
        f"Optional Inputs: {', '.join(info['optional_inputs']) if info['optional_inputs'] else 'None'}",
        f"Outputs: {', '.join(info['outputs'])}",
        f"Intermediate Outputs: {', '.join(info['intermediate_outputs']) if info['intermediate_outputs'] else 'None'}",
    ]
    return "\n".join(lines)


def _get_mapspec_section(pipeline: Pipeline) -> str:
    """Generate mapspec information section."""
    mapspecs = pipeline.mapspecs_as_strings
    if not mapspecs:
        return "MAPSPEC: None (This pipeline uses sequential processing only)"

    lines = [
        "MAPSPEC DEFINITIONS:",
        "The following mapspecs define how arrays are processed in parallel:",
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


def _format_tool_description(
    method_description: str,
    pipeline_info: str,
    mapspec_section: str,
    documentation: str,
) -> str:
    """Format a complete tool description using the template."""
    return _PIPELINE_DESCRIPTION_TEMPLATE.format(
        method_description=method_description,
        pipeline_info=pipeline_info,
        mapspec_section=mapspec_section,
        documentation=documentation,
    )


def build_mcp_server(pipeline_name: str, pipeline: Pipeline):
    requires("mcp", "rich", "griffe", reason="mcp", extras="mcp")
    from fastmcp import Context, FastMCP

    # Generate all pipeline information sections
    documentation = _get_pipeline_documentation(pipeline)
    pipeline_info = _get_pipeline_info_summary(pipeline_name, pipeline)
    mapspec_section = _get_mapspec_section(pipeline)

    # Format descriptions using the template
    run_description = _format_tool_description(
        method_description=_RUN_PIPELINE_DESCRIPTION,
        pipeline_info=pipeline_info,
        mapspec_section="NOTE: Sequential execution ignores mapspecs and processes single values only.",
        documentation=documentation,
    )

    map_description = _format_tool_description(
        method_description=_MAP_PIPELINE_DESCRIPTION,
        pipeline_info=pipeline_info,
        mapspec_section=mapspec_section,
        documentation=documentation,
    )

    Model = pipeline.pydantic_model()  # noqa: N806
    mcp = FastMCP(name=pipeline_name, version="0.1.0")

    @mcp.tool(
        name="run_pipeline",
        description=run_description,
    )
    def run_pipeline(
        ctx: Context,
        input: Model,
    ) -> str:
        """Run pipeline sequentially with single input values."""
        kwargs = maybe_pydantic_model_to_dict(input)
        result = pipeline.run(pipeline.unique_leaf_node, kwargs=kwargs)
        return str(result)

    @mcp.tool(
        name="map_pipeline",
        description=map_description,
    )
    def map_pipeline(
        ctx: Context,
        input: Model,
        parallel: bool = True,
        run_folder: str | None = None,
    ) -> str:
        """Run pipeline in parallel with arrays of input values using mapspec."""
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
