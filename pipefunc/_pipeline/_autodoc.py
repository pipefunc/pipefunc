from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rich.box import HEAVY_HEAD
from rich.console import Console
from rich.table import Table
from rich.text import Text

from pipefunc.typing import type_as_string

if TYPE_CHECKING:
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
    """Formats pipeline documentation into rich tables.

    Parameters
    ----------
    doc
        The pipeline documentation object.
    borders
        Whether to include borders in the tables.
    skip_optional
        Whether to skip optional parameters.
    function_description_table
        Whether to generate the function description table.
    function_parameters_table
        Whether to generate the function parameters table.
    function_returns_table
        Whether to generate the function returns table.
    print_table
        Whether to print the table to the console.

    Returns
    -------
        Tuple containing rich Tables for descriptions, parameters, and returns if
        print_table is False.
        Otherwise, prints the tables to the console.

    """
    green = "#00aa00"
    bold_red = "bold #ff0000"
    bold_yellow = "bold #aaaa00"
    bold_magenta = "bold #ff00ff"
    box = None if not borders else HEAVY_HEAD

    tables: list[Table] = []

    if function_description_table:
        table_desc = _create_description_table(doc, box, bold_red, green)
        tables.append(table_desc)

    if function_parameters_table:
        table_params = _create_parameters_table(
            doc,
            box,
            bold_yellow,
            green,
            skip_optional,
        )
        tables.append(table_params)

    if function_returns_table:
        table_returns = _create_returns_table(doc, box, bold_magenta, green)
        tables.append(table_returns)

    if print_table:
        console = Console()
        for table in tables:
            console.print(table)
        return None
    return tuple(tables)


def _create_description_table(
    doc: PipelineDoc,
    box: Any,
    bold_red: str,
    green: str,
) -> Table:
    """Creates the description table."""
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
    return table_desc


def _create_parameters_table(
    doc: PipelineDoc,
    box: Any,
    bold_yellow: str,
    green: str,
    skip_optional: bool,  # noqa: FBT001
) -> Table:
    """Creates the parameters table."""
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
            "\n".join(f"- {d}" for d in param_descs) + default_str + annotation_str,
        )
        table_params.add_row(
            *([param_text, desc_text] if skip_optional else [param_text, default_col, desc_text]),
        )
    return table_params


def _create_returns_table(
    doc: PipelineDoc,
    box: Any,
    bold_magenta: str,
    green: str,
) -> Table:
    """Creates the returns table."""
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
    return table_returns
