#!/usr/bin/env python
"""Check for discrepancies in parameter descriptions between functions."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import pipefunc._pipeline
import pipefunc._pipeline._autodoc
import pipefunc._pipeline._cli

if TYPE_CHECKING:
    from collections.abc import Callable


def extract_param_descriptions(func: Callable[..., Any]) -> dict[str, str]:
    """Extract parameter descriptions from a function's docstring.

    Parameters
    ----------
    func
        The function to extract parameter descriptions from.

    Returns
    -------
    A dictionary where keys are parameter names and values are their descriptions.

    """
    doc = inspect.getdoc(func)

    if not doc:
        return {}

    param_dict = {}
    in_params_section = False
    current_param = None
    current_description: list[str] = []

    for line in doc.split("\n"):
        stripped_line = line.strip()
        if stripped_line == "Parameters":
            in_params_section = True
        elif in_params_section and (
            stripped_line in ("See Also", "Returns", "Raises", "Notes", "Examples")
        ):
            break
        elif in_params_section:
            if stripped_line.startswith("---"):
                continue
            if stripped_line == "":
                # Empty line inside of a parameter description, e.g., for lists
                continue
            if not line.startswith("    "):  # Parameter names are not indented
                if current_param:
                    param_dict[current_param] = " ".join(current_description).strip()
                current_param = stripped_line
                current_description = []
            elif current_param is not None:
                current_description.append(stripped_line)

    if current_param:
        param_dict[current_param] = " ".join(current_description).strip()

    return param_dict


class ParameterDiscrepancyError(Exception):
    """Exception raised for discrepancies in parameter descriptions."""


class MissingParameterError(Exception):
    """Exception raised for parameters missing in one of the functions."""


def test_module() -> None:
    """Tests whether this module works as intended."""

    def starts_ends_with(s: str, start: str, end: str) -> bool:
        return s.startswith(start) and s.endswith(end)

    p = extract_param_descriptions(extract_param_descriptions)
    assert p.keys() == {"func"}
    assert starts_ends_with(p["func"], "The function", "descriptions from.")
    p = extract_param_descriptions(compare_param_descriptions)
    assert p.keys() == {"func1", "func2", "allow_missing", "allow_discrepancy"}
    assert p["func1"] == "The first function to compare."
    assert p["func2"] == "The second function to compare."
    assert starts_ends_with(p["allow_missing"], "If True, allow any", "missing parameters.")
    assert starts_ends_with(p["allow_discrepancy"], "If True, allow any", "discrepancies.")

    def func_with_spacing() -> None:
        """Example

        Parameters
        ----------
        param1
            The first parameter.

            - First line
            - Second line

            Yo end of list.
        param2
            The second parameter.

        """

    p = extract_param_descriptions(func_with_spacing)
    assert p.keys() == {"param1", "param2"}
    assert starts_ends_with(p["param1"], "The first parameter.", "Yo end of list.")
    assert p["param2"] == "The second parameter."


def compare_param_descriptions(
    func1: Callable,
    func2: Callable,
    *,
    allow_missing: list[str] | bool = False,
    allow_discrepancy: list[str] | bool = False,
) -> None:
    """Compare parameter descriptions between two functions.

    Parameters
    ----------
    func1
        The first function to compare.
    func2
        The second function to compare.
    allow_missing
        If True, allow any missing parameters. If a list, allow missing parameters
        specified in the list. If False, raise exceptions for all missing parameters.
    allow_discrepancy
        If True, allow any discrepancies in parameter descriptions. If a list, allow
        discrepancies for parameters specified in the list. If False, raise exceptions
        for all discrepancies.

    Raises
    ------
    ExceptionGroup
        An exception group containing all discrepancies and missing parameters.

    """
    params1 = extract_param_descriptions(func1)
    params2 = extract_param_descriptions(func2)

    all_params = set(params1.keys()) | set(params2.keys())
    exceptions: list[Exception] = []

    for param in all_params:
        if param in params1 and param in params2:
            if params1[param] != params2[param] and not (
                allow_discrepancy is True
                or (isinstance(allow_discrepancy, list) and param in allow_discrepancy)
            ):
                exceptions.append(
                    ParameterDiscrepancyError(
                        f"Discrepancy in parameter '{param}':\n"
                        f"  {func1.__name__}: {params1[param]}\n"
                        f"  {func2.__name__}: {params2[param]}",
                    ),
                )
        elif param in params1:
            if not (
                allow_missing is True
                or (isinstance(allow_missing, list) and param in allow_missing)
            ):
                exceptions.append(
                    MissingParameterError(
                        f"Parameter '{param}' is in {func1.__name__} but not in {func2.__name__}",
                    ),
                )
        elif not (
            allow_missing is True or (isinstance(allow_missing, list) and param in allow_missing)
        ):
            exceptions.append(
                MissingParameterError(
                    f"Parameter '{param}' is in {func2.__name__} but not in {func1.__name__}",
                ),
            )

    if exceptions:
        msg = "Parameter description discrepancies"
        raise ExceptionGroup(msg, exceptions)  # type: ignore[name-defined]  # noqa: F821


if __name__ == "__main__":
    test_module()

    import pipefunc
    import pipefunc._plotting
    from pipefunc.map import run_map, run_map_async
    from pipefunc.map._run_eager import run_map_eager
    from pipefunc.map._run_eager_async import run_map_eager_async

    # @pipefunc and PipeFunc
    compare_param_descriptions(
        pipefunc.PipeFunc,
        pipefunc.pipefunc,
        # In PipeFunc "wrapped function" and in @pipefunc "decorated function"
        allow_discrepancy=["output_name", "profile", "cache"],
        allow_missing=["func"],
    )
    compare_param_descriptions(
        pipefunc.PipeFunc.update_bound,
        pipefunc.NestedPipeFunc.update_bound,
    )

    # map vs map_async
    compare_param_descriptions(
        pipefunc.Pipeline.map,
        run_map,
        allow_missing=["pipeline", "scheduling_strategy"],
    )
    compare_param_descriptions(
        run_map_eager,
        run_map,
    )
    compare_param_descriptions(
        run_map_eager_async,
        run_map_async,
    )
    compare_param_descriptions(
        pipefunc.Pipeline.map_async,
        run_map_async,
        allow_missing=["parallel", "pipeline", "scheduling_strategy"],
        allow_discrepancy=["show_progress"],
    )
    compare_param_descriptions(
        pipefunc.Pipeline.map,
        pipefunc.Pipeline.map_async,
        allow_missing=["parallel"],
        allow_discrepancy=["show_progress", "executor"],
    )

    # plotting
    compare_param_descriptions(
        pipefunc._plotting.visualize_graphviz,
        pipefunc.Pipeline.visualize_graphviz,
        allow_missing=["defaults", "graph"],
    )
    compare_param_descriptions(
        pipefunc._plotting.visualize_holoviews,
        pipefunc.Pipeline.visualize_holoviews,
        allow_missing=["graph"],
    )
    compare_param_descriptions(
        pipefunc._plotting.visualize_matplotlib,
        pipefunc.Pipeline.visualize_matplotlib,
        allow_missing=[
            "output_name",
            "color_combinable",
            "conservatively_combine",
            "graph",
            "func_node_colors",
        ],
    )

    # Pipeline and VariantsPipeline
    compare_param_descriptions(
        pipefunc.Pipeline,
        pipefunc.VariantPipeline,
        allow_missing=["default_variant"],
        allow_discrepancy=["functions"],
    )

    # print_documentation and format_pipeline_docs
    compare_param_descriptions(
        pipefunc.Pipeline.print_documentation,
        pipefunc._pipeline._autodoc.format_pipeline_docs,
        allow_missing=["doc", "print_table"],
    )

    # CLI
    compare_param_descriptions(
        pipefunc.Pipeline.cli,
        pipefunc._pipeline._cli.cli,
        allow_missing=["pipeline"],
    )
