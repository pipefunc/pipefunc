from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pipefunc.typing import type_as_string

if TYPE_CHECKING:
    import rich

    from pipefunc import Pipeline

    from ._types import OUTPUT_TYPE


@dataclass
class PipelineDoc:
    descriptions: dict[OUTPUT_TYPE, str]
    parameters: dict[str, list[str]]
    returns: dict[OUTPUT_TYPE, str]
    defaults: dict[str, Any]
    annotations: dict[str, Any]


def format_pipeline_docs(
    pipeline: Pipeline,
    *,
    print_table: bool = True,
) -> tuple[rich.Table, rich.Table, rich.Table] | None:
    """Formats pipeline documentation into three separate rich tables.

    Parameters
    ----------
    pipeline
        The pipeline to format documentation for.
    print_table
        Whether to print the table to the console.

    Returns
    -------
        Tuple containing three rich Tables: descriptions, parameters, returns if
        print_table is False. Otherwise, prints the tables to the console.

    """
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    doc = pipeline.doc()
    # Descriptions Table
    table_desc = Table(
        title="[bold red]Function Output Descriptions[/]",
        box=None,
        expand=True,
        show_lines=True,
    )
    table_desc.add_column("Output Name", style="bold red", no_wrap=True)
    table_desc.add_column("Description", style="green")
    for output_name, desc in sorted(doc.descriptions.items()):
        table_desc.add_row(Text.from_markup(f"{output_name}"), desc)

    # Parameters Table
    table_params = Table(
        title="[bold yellow]Parameters[/]",
        box=None,
        expand=True,
        show_lines=True,
    )
    table_params.add_column("Parameter", style="bold yellow", no_wrap=True)
    table_params.add_column("Required", style="bold yellow", no_wrap=True)
    table_params.add_column("Description", style="green")
    for param, param_descs in sorted(doc.parameters.items()):
        parameters_text = "\n".join(f"- {d}" for d in param_descs)
        default = doc.defaults.get(param)
        annotation = type_as_string(doc.annotations.get(param))
        default_text = f" [italic bold](default: {default})[/]" if param in doc.defaults else ""
        annotation_text = (
            f" [italic bold](type: {annotation})[/]" if param in doc.annotations else ""
        )
        table_params.add_row(
            Text.from_markup(f"{param}"),
            # "✅" if optional, "❌" if required
            Text.from_markup("❌" if param in doc.defaults else "✅"),
            Text.from_markup(
                parameters_text + default_text + annotation_text,
            ),
        )

    # Returns Table
    table_returns = Table(
        title="[bold magenta]Return Values[/]",
        box=None,
        expand=True,
        show_lines=True,
    )
    table_returns.add_column("Output Name", style="bold magenta", no_wrap=True)
    table_returns.add_column("Description", style="green")
    for output_name, ret_desc in sorted(doc.returns.items()):
        table_returns.add_row(Text.from_markup(f"{output_name}"), ret_desc)

    if print_table:
        console = Console()
        console.print(table_desc)
        console.print(table_params)
        console.print(table_returns)
        return None
    return table_desc, table_params, table_returns
