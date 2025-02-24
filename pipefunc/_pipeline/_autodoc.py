from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pipefunc._utils import at_least_tuple, parse_function_docstring
from pipefunc.typing import type_as_string

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rich.table import Table
    from rich.text import Text

    from pipefunc import Pipeline

    from ._types import OUTPUT_TYPE


@dataclass
class PipelineDocumentation:
    descriptions: dict[OUTPUT_TYPE, str]
    parameters: dict[str, list[str]]
    returns: dict[OUTPUT_TYPE, str]
    function_names: dict[OUTPUT_TYPE, str]
    defaults: dict[str, Any]
    p_annotations: dict[str, Any]
    r_annotations: dict[str, Any]
    topological_order: list[OUTPUT_TYPE]
    root_args: list[str]

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline) -> PipelineDocumentation:
        """Generates a PipelineDocumentation object from a pipeline."""
        descriptions: dict[OUTPUT_TYPE, str] = {}
        returns: dict[OUTPUT_TYPE, str] = {}
        parameters: dict[str, list[str]] = defaultdict(list)

        for f in pipeline.sorted_functions:
            doc = parse_function_docstring(f.func)
            if doc.description:
                descriptions[f.output_name] = doc.description
            if doc.returns:
                returns[f.output_name] = doc.returns
            for p, v in doc.parameters.items():
                p_renamed = f.renames.get(p, p)
                if p_renamed in f.bound:
                    continue
                if v not in parameters[p_renamed]:
                    parameters[p_renamed].append(v)

        # Add emdash to dicts where docs are missing
        info = pipeline.info()
        assert info is not None
        for p in info["inputs"]:
            if p not in parameters:
                parameters[p].append("—")
        for f in pipeline.functions:
            if f.output_name not in returns:
                returns[f.output_name] = "—"
            if f.output_name not in descriptions:
                descriptions[f.output_name] = "—"

        return cls(
            descriptions=descriptions,
            parameters=dict(parameters),
            returns=returns,
            function_names={f.output_name: f.func.__name__ for f in pipeline.functions},
            defaults=pipeline.defaults,
            p_annotations=pipeline.parameter_annotations,
            r_annotations=pipeline.output_annotations,
            topological_order=[f.output_name for f in pipeline.sorted_functions],
            root_args=pipeline.topological_generations.root_args,
        )


class RichStyle:
    # Colors from the Python REPL:
    # https://github.com/python/cpython/blob/13c4def692228f09df0b30c5f93bc515e89fc77f/Lib/_colorize.py#L8-L19
    GREEN = "#00aa00"
    BOLD_RED = "bold #ff0000"
    BOLD_YELLOW = "bold #aaaa00"
    BOLD_MAGENTA = "bold #ff00ff"


def format_pipeline_docs(
    doc: PipelineDocumentation,
    *,
    borders: bool = False,
    skip_optional: bool = False,
    skip_intermediate: bool = True,
    description_table: bool = True,
    parameters_table: bool = True,
    returns_table: bool = True,
    order: Literal["topological", "alphabetical"] = "topological",
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
    order
        The order in which to display the functions in the documentation.
        Options are:

        * ``topological``: Display functions in topological order.
        * ``alphabetical``: Display functions in alphabetical order (using ``output_name``).

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

    if parameters_table:
        table_params = _create_parameters_table(doc, box, skip_optional, skip_intermediate)
        tables.append(table_params)

    if description_table:
        table_desc = _create_description_table(doc, box, order)
        tables.append(table_desc)

    if returns_table:
        table_returns = _create_returns_table(doc, box, order)
        tables.append(table_returns)

    if print_table:
        console = Console()
        for table in tables:
            console.print(table, "\n")
        return None
    return tuple(tables)


def _create_description_table(
    doc: PipelineDocumentation,
    box: Any,
    order: Literal["topological", "alphabetical"],
) -> Table:
    """Creates the description table."""
    from rich.table import Table

    table_desc = Table(
        title=f"[{RichStyle.BOLD_RED}]Function Output Descriptions[/]",
        box=box,
        expand=True,
        show_lines=True,
    )
    table_desc.add_column("Function Name", style=RichStyle.BOLD_RED, no_wrap=True)
    table_desc.add_column("Output Name", style=RichStyle.BOLD_MAGENTA, no_wrap=True)
    table_desc.add_column("Description", style=RichStyle.GREEN)
    for output_name in _sort(doc.descriptions, doc.topological_order, order):
        desc = doc.descriptions[output_name]
        name = doc.function_names[output_name]
        table_desc.add_row(name, _output_name_text(output_name), desc)
    return table_desc


def _sort(
    output_names: Iterable[OUTPUT_TYPE],
    topological_order: list[OUTPUT_TYPE],
    order: Literal["topological", "alphabetical"],
) -> list[OUTPUT_TYPE]:
    """Sorts the output names based on the specified order."""
    if order == "alphabetical":
        return sorted(output_names, key=at_least_tuple)
    return [n for n in topological_order if n in output_names]


def _output_name_text(output_name: OUTPUT_TYPE) -> Text:
    """Creates a Text object for the output name."""
    from rich.text import Text

    return Text.from_markup(f"{', '.join(at_least_tuple(output_name))}")


def _create_parameters_table(
    doc: PipelineDocumentation,
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


def _create_returns_table(
    doc: PipelineDocumentation,
    box: Any,
    order: Literal["topological", "alphabetical"],
) -> Table:
    """Creates the returns table."""
    from rich.table import Table

    table_returns = Table(
        title=f"[{RichStyle.BOLD_MAGENTA}]Return Values[/]",
        box=box,
        expand=True,
        show_lines=True,
    )
    table_returns.add_column("Output Name", style=RichStyle.BOLD_MAGENTA, no_wrap=True)
    table_returns.add_column("Description", style=RichStyle.GREEN)

    for output_name in _sort(doc.returns, doc.topological_order, order):
        desc = doc.returns[output_name]
        desc_text = f"{desc}"
        output_tuple = at_least_tuple(output_name)
        for name in output_tuple:
            annotation = type_as_string(doc.r_annotations[name])
            if len(output_tuple) > 1:
                annotation_str = f" [italic bold](type {name}: {annotation})[/]"
            else:
                annotation_str = f" [italic bold](type: {annotation})[/]"
            desc_text += annotation_str
        table_returns.add_row(_output_name_text(output_name), desc_text)

    return table_returns
