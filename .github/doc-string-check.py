"""Check for discrepancies in parameter descriptions between functions."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

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
        elif stripped_line in ("Returns", "Raises"):
            break  # Stop processing when we hit Returns or Raises section
        elif in_params_section:
            if stripped_line.startswith("---"):
                continue
            if stripped_line == "":
                if current_param:
                    param_dict[current_param] = " ".join(current_description).strip()
                    current_param = None
                    current_description = []
            elif not line.startswith("    "):  # Parameter names are not indented
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
    import pipefunc
    import pipefunc._plotting
    import pipefunc.map._run

    # map vs map_async
    compare_param_descriptions(
        pipefunc.Pipeline.map,
        pipefunc.map._run.run,
        allow_missing=["pipeline"],
    )
    compare_param_descriptions(
        pipefunc.Pipeline.map_async,
        pipefunc.map._run.run_async,
        allow_missing=["parallel", "pipeline"],
        allow_discrepancy=["with_progress"],
    )
    compare_param_descriptions(
        pipefunc.Pipeline.map,
        pipefunc.Pipeline.map_async,
        allow_missing=["parallel"],
        allow_discrepancy=["with_progress"],
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
