from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pipefunc.typing import type_as_string

if TYPE_CHECKING:
    from rich.table import Table
    from rich.text import Text

    from ._types import OUTPUT_TYPE


@dataclass
class PipelineDoc:
    descriptions: dict[OUTPUT_TYPE, str]
    parameters: dict[str, list[str]]
    returns: dict[OUTPUT_TYPE, str]
    defaults: dict[str, Any]
    p_annotations: dict[str, Any]
    r_annotations: dict[str, Any]
    root_args: list[str]


class RichStyle:
    # Colors from the Python REPL:
    # https://github.com/python/cpython/blob/13c4def692228f09df0b30c5f93bc515e89fc77f/Lib/_colorize.py#L8-L19
    GREEN = "#00aa00"
    BOLD_RED = "bold #ff0000"
    BOLD_YELLOW = "bold #aaaa00"
    BOLD_MAGENTA = "bold #ff00ff"


def format_pipeline_docs(
    doc: PipelineDoc,
    *,
    borders: bool = False,
    skip_optional: bool = False,
    skip_intermediate: bool = True,
    description_table: bool = True,
    parameters_table: bool = True,
    returns_table: bool = True,
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
    skip_intermediate
        Whether to skip intermediate outputs and only show root parameters.
    description_table
        Whether to generate the function description table.
    parameters_table
        Whether to generate the function parameters table.
    returns_table
        Whether to generate the function returns table.
    print_table
        Whether to print the table to the console.

    Returns
    -------
        Tuple containing rich Tables for descriptions, parameters, and returns if
        print_table is False.
        Otherwise, prints the tables to the console.

    """
    from rich.box import HEAVY_HEAD
    from rich.console import Console

    box = None if not borders else HEAVY_HEAD

    tables: list[Table] = []

    if description_table:
        table_desc = _create_description_table(doc, box)
        tables.append(table_desc)

    if parameters_table:
        table_params = _create_parameters_table(doc, box, skip_optional, skip_intermediate)
        tables.append(table_params)

    if returns_table:
        table_returns = _create_returns_table(doc, box)
        tables.append(table_returns)

    if print_table:
        console = Console()
        for table in tables:
            console.print(table)
        return None
    return tuple(tables)


def _create_description_table(doc: PipelineDoc, box: Any) -> Table:
    """Creates the description table."""
    from rich.table import Table
    from rich.text import Text

    table_desc = Table(
        title=f"[{RichStyle.BOLD_RED}]Function Output Descriptions[/]",
        box=box,
        expand=True,
        show_lines=True,
    )
    table_desc.add_column("Output Name", style=RichStyle.BOLD_RED, no_wrap=True)
    table_desc.add_column("Description", style=RichStyle.GREEN)
    for output_name, desc in sorted(doc.descriptions.items()):
        table_desc.add_row(Text.from_markup(f"{output_name}"), desc)
    return table_desc


def _create_parameters_table(
    doc: PipelineDoc,
    box: Any,
    skip_optional: bool,  # noqa: FBT001
    skip_intermediate: bool,  # noqa: FBT001
) -> Table:
    """Creates the parameters table."""
    from rich.table import Table

    table_params = Table(
        title=f"[{RichStyle.BOLD_YELLOW}]Parameters[/]",
        box=box,
        expand=True,
        show_lines=True,
    )
    table_params.add_column("Parameter", style=RichStyle.BOLD_YELLOW, no_wrap=True)
    if not skip_optional:
        table_params.add_column(
            "Required",
            style=RichStyle.BOLD_YELLOW,
            no_wrap=True,
            justify="center",
        )
    table_params.add_column("Description", style=RichStyle.GREEN)
    for param, param_descs in sorted(doc.parameters.items()):
        if skip_optional and param in doc.defaults:
            continue
        if skip_intermediate and param not in doc.root_args:
            continue
        table_params.add_row(
            *_create_parameter_row(
                param,
                param_descs,
                doc.defaults,
                doc.p_annotations,
                skip_optional,
            ),
        )
    return table_params


def _create_parameter_row(
    param: str,
    param_descs: list[str],
    defaults: dict[str, Any],
    p_annotations: dict[str, Any],
    skip_optional: bool,  # noqa: FBT001
) -> list[Text]:
    """Creates a row for the parameters table."""
    from rich.text import Text

    default = defaults.get(param)
    annotation = type_as_string(p_annotations.get(param))
    default_str = f" [italic bold](default: {default})[/]" if param in defaults else ""
    annotation_str = f" [italic bold](type: {annotation})[/]" if param in p_annotations else ""
    param_text = Text.from_markup(f"{param}")
    default_col = Text.from_markup("❌" if param in defaults else "✅")
    if len(param_descs) == 1:
        param_desc_text = param_descs[0]
    else:
        param_desc_text = "\n".join(f"{i}. {d}" for i, d in enumerate(param_descs, start=1))
    desc_text = Text.from_markup(param_desc_text + default_str + annotation_str)
    return [param_text, desc_text] if skip_optional else [param_text, default_col, desc_text]


def _create_returns_table(doc: PipelineDoc, box: Any) -> Table:
    """Creates the returns table."""
    from rich.table import Table
    from rich.text import Text

    table_returns = Table(
        title=f"[{RichStyle.BOLD_MAGENTA}]Return Values[/]",
        box=box,
        expand=True,
        show_lines=True,
    )
    table_returns.add_column("Output Name", style=RichStyle.BOLD_MAGENTA, no_wrap=True)
    table_returns.add_column("Description", style=RichStyle.GREEN)
    for output_name, ret_desc in sorted(doc.returns.items()):
        table_returns.add_row(Text.from_markup(f"{output_name}"), ret_desc)
    return table_returns
