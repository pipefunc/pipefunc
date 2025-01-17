from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pipefunc.typing import type_as_string

if TYPE_CHECKING:
    from rich.table import Table

    from ._types import OUTPUT_TYPE


@dataclass
class PipelineDoc:
    descriptions: dict[OUTPUT_TYPE, str]
    parameters: dict[str, list[str]]
    returns: dict[OUTPUT_TYPE, str]
    defaults: dict[str, Any]
    annotations: dict[str, Any]


def format_pipeline_docs(
    doc: PipelineDoc,
    *,
    borders: bool = False,
    skip_optional: bool = False,
    function_description_table: bool = True,
    function_parameters_table: bool = True,
    function_returns_table: bool = True,
    print_table: bool = True,
) -> tuple[Table, ...] | None:
    """Formats pipeline documentation into three separate rich tables.

    Parameters
    ----------
    doc
        The pipeline documentation object.
    skip_optional
        Whether to skip optional parameters.
    borders
        Whether to include borders in the tables.
    print_table
        Whether to print the table to the console.

    Returns
    -------
        Tuple containing three rich Tables: descriptions, parameters, returns if
        print_table is False. Otherwise, prints the tables to the console.

    """
    from rich.box import HEAVY_HEAD
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    green = "#00aa00"
    bold_red = "bold #ff0000"
    bold_yellow = "bold #aaaa00"
    bold_magenta = "bold #ff00ff"
    box = None if not borders else HEAVY_HEAD

    # Descriptions Table
    table_desc = Table(
        title=f"[{bold_red}]Function Output Descriptions[/]",
        box=box,
        expand=True,
        show_lines=True,
    )
    table_desc.add_column("Output Name", style=bold_red, no_wrap=True)
    table_desc.add_column("Description", style=green)
    for output_name, desc in sorted(doc.descriptions.items()):
        table_desc.add_row(Text.from_markup(f"{output_name}"), desc)

    # Parameters Table
    table_params = Table(
        title=f"[{bold_yellow}]Parameters[/]",
        box=box,
        expand=True,
        show_lines=True,
    )
    table_params.add_column("Parameter", style=bold_yellow, no_wrap=True)
    if not skip_optional:
        table_params.add_column("Required", style=bold_yellow, no_wrap=True)
    table_params.add_column("Description", style=green)
    for param, param_descs in sorted(doc.parameters.items()):
        parameters_str = "\n".join(f"- {d}" for d in param_descs)
        if skip_optional and param in doc.defaults:
            continue
        default = doc.defaults.get(param)
        annotation = type_as_string(doc.annotations.get(param))
        default_str = f" [italic bold](default: {default})[/]" if param in doc.defaults else ""
        annotation_str = (
            f" [italic bold](type: {annotation})[/]" if param in doc.annotations else ""
        )
        param_text = Text.from_markup(f"{param}")
        default_col = Text.from_markup("❌" if param in doc.defaults else "✅")
        desc_text = Text.from_markup(
            parameters_str + default_str + annotation_str,
        )
        table_params.add_row(
            *([param_text, desc_text] if skip_optional else [param_text, default_col, desc_text]),
        )

    # Returns Table
    table_returns = Table(
        title=f"[{bold_magenta}]Return Values[/]",
        box=box,
        expand=True,
        show_lines=True,
    )
    table_returns.add_column("Output Name", style=bold_magenta, no_wrap=True)
    table_returns.add_column("Description", style=green)
    for output_name, ret_desc in sorted(doc.returns.items()):
        table_returns.add_row(Text.from_markup(f"{output_name}"), ret_desc)

    if print_table:
        console = Console()
        console.print(table_desc)
        console.print(table_params)
        console.print(table_returns)
        return None
    return table_desc, table_params, table_returns
