"""Testing utilities for the pipefunc package."""

from __future__ import annotations

import contextlib
import unittest.mock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

    from pipefunc import Pipeline


@contextlib.contextmanager
def patch(pipeline: Pipeline, func_name: str) -> Generator[unittest.mock.MagicMock, None, None]:
    """Patch a function within a Pipeline for testing purposes.

    This function provides a context manager to temporarily replace the function of a
    specified `~pipefunc.PipeFunc` instance within the pipeline.

    Parameters
    ----------
    pipeline
        The Pipeline instance to be patched.
    func_name
        The name of the function to be patched. This can be either a simple function
        name or a fully qualified name including the module path.
        If a dot is present in `func_name`, the function will attempt to match the full
        module path and function name. Otherwise, it will use only the function name.

    Yields
    ------
    mock
        A MagicMock object that can be used to set return values or side effects.

    Raises
    ------
    ValueError
        If no function with the given name is found in the pipeline.

    Examples
    --------
    >>> @pipefunc(output_name="c")
    ... def f() -> Any:
    ...     raise ValueError("test")

    >>> pipeline = Pipeline([f])

    >>> with patch(pipeline, "f") as mock:
    ...     mock.return_value = 1
    ...     print(pipeline("c"))  # Prints 1

    """
    target_func = None
    for f in pipeline.functions:
        if isinstance(f.func, unittest.mock.MagicMock):
            continue
        full_name = f"{f.func.__module__}.{f.func.__name__}"
        # Check for full match if there's a dot in func_name, otherwise just use func_name
        if ("." in func_name and full_name == func_name) or f.__name__ == func_name:
            target_func = f
            break

    if target_func is None:
        msg = f"No function named '{func_name}' found in the pipeline."
        raise ValueError(msg)

    with unittest.mock.patch.object(target_func, "func") as mock:
        yield mock
