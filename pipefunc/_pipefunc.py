"""PipeFunc: A Python library for defining, managing, and executing function pipelines.

This module provides an implementation of a Pipeline class, which allows you
to easily define, manage, and execute a sequence of functions, where each
function may depend on the outputs of other functions. The pipeline is
represented as a directed graph, where nodes represent functions and edges
represent dependencies between functions. The Pipeline class provides methods
for adding functions to the pipeline, executing the pipeline for specific
output values, visualizing the pipeline as a directed graph, and profiling
the resource usage of the pipeline functions.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import os
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Tuple,
    TypeVar,
    Union,
)

import cloudpickle

from pipefunc._lazy import evaluate_lazy
from pipefunc._perf import ProfilingStats, ResourceProfiler
from pipefunc._utils import at_least_tuple

if sys.version_info < (3, 9):  # pragma: no cover
    from typing import Callable
else:
    from collections.abc import Callable

if TYPE_CHECKING:
    from pathlib import Path


T = TypeVar("T", bound=Callable[..., Any])
_OUTPUT_TYPE = Union[str, Tuple[str, ...]]


def _default_output_picker(
    output: Any,
    name: str,
    output_name: _OUTPUT_TYPE,
) -> Any:
    """Default output picker function for tuples."""
    return output[output_name.index(name)]


def _update_wrapper(wrapper, wrapped) -> None:  # noqa: ANN001
    functools.update_wrapper(wrapper, wrapped)
    # Need to manually update __wrapped__ to keep functions picklable
    del wrapper.__dict__["__wrapped__"]


class PipelineFunction(Generic[T]):
    """Function wrapper class for pipeline functions with additional attributes.

    Parameters
    ----------
    func
        The original function to be wrapped.
    output_name
        The identifier for the output of the wrapped function.
    output_picker
        A function that takes the output of the wrapped function as first argument
        and the output_name (str) as second argument, and returns the desired output.
        If None, the output of the wrapped function is returned as is.
    renames
        A dictionary mapping from original argument names to new argument names.
    profile
        Flag indicating whether the wrapped function should be profiled.
    debug
        Flag indicating whether debug information should be printed.
    cache
        Flag indicating whether the wrapped function should be cached.
    save
        Flag indicating whether the output of the wrapped function should be saved.
    save_function
        A function that takes the filename and a dict containing the inputs and output.

    Returns
    -------
        The identifier for the output of the wrapped function.

    Examples
    --------
    >>> def add_one(a, b):
    ...     return a + 1, b + 1
    >>> add_one_func = PipelineFunction(
    ...     add_one,
    ...     output_name="a_plus_one",
    ...     renames={"a": "x", "b": "y"},
    ... )
    >>> add_one_func(x=1, y=2)
    (2, 3)

    """

    def __init__(
        self,
        func: T,
        output_name: _OUTPUT_TYPE,
        *,
        output_picker: Callable[[str, Any], Any] | None = None,
        renames: dict[str, str] | None = None,
        profile: bool = False,
        debug: bool = False,
        cache: bool = False,
        save: bool | None = None,
        save_function: Callable[[str | Path, dict[str, Any]], None] | None = None,
    ) -> None:
        """Function wrapper class for pipeline functions with additional attributes."""
        _update_wrapper(self, func)
        self.func: Callable[..., Any] = func
        self.output_name: _OUTPUT_TYPE = output_name
        self.debug = debug
        self.cache = cache
        self.save_function = save_function
        self.save = save if save is not None else save_function is not None
        self.output_picker: Callable[[Any, str], Any] | None = output_picker
        if output_picker is None and isinstance(output_name, tuple):
            self.output_picker = functools.partial(
                _default_output_picker,
                output_name=self.output_name,
            )

        self._profile = profile
        self.renames: dict[str, str] = renames or {}
        self._inverse_renames: dict[str, str] = {v: k for k, v in self.renames.items()}
        parameters = inspect.signature(func).parameters
        self.parameters: list[str] = [self.renames.get(k, k) for k in parameters]
        self.defaults: dict[str, Any] = {
            self.renames.get(k, k): v.default
            for k, v in parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        self.profiling_stats: ProfilingStats | None
        self.set_profiling(enable=profile)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function with the given arguments.

        Returns
        -------
        Any
            The return value of the wrapped function.

        """
        kwargs = {self._inverse_renames.get(k, k): v for k, v in kwargs.items()}
        with self._maybe_profiler():
            args = evaluate_lazy(args)
            kwargs = evaluate_lazy(kwargs)
            result = self.func(*args, **kwargs)

        if self.debug and self.profiling_stats is not None:
            dt = self.profiling_stats.time.average
            print(
                f"Function {self.func.__name__} called with args={args},"
                f" kwargs={kwargs}, took {dt:.2e} seconds to execute.",
            )
        return result

    @property
    def profile(self) -> bool:
        """Return whether profiling is enabled for the wrapped function."""
        return self._profile

    @profile.setter
    def profile(self, enable: bool) -> None:
        """Enable or disable profiling for the wrapped function."""
        self.set_profiling(enable=enable)

    def set_profiling(self, *, enable: bool = True) -> None:
        """Enable or disable profiling for the wrapped function."""
        self._profile = enable
        if enable:
            self.profiling_stats = ProfilingStats()
        else:
            self.profiling_stats = None

    def _maybe_profiler(self) -> contextlib.AbstractContextManager:
        """Maybe get profiler.

        Get a profiler instance if profiling is enabled, otherwise
        return a dummy context manager.

        Returns
        -------
        AbstractContextManager
            A ResourceProfiler instance if profiling is enabled, or a
            nullcontext if disabled.

        """
        if self.profiling_stats is not None:
            return ResourceProfiler(os.getpid(), self.profiling_stats)
        return contextlib.nullcontext()

    def __getattr__(self, name: str) -> Any:
        """Get attributes of the wrapped function.

        Parameters
        ----------
        name
            The name of the attribute to get.

        Returns
        -------
        Any
            The value of the attribute.

        """
        return getattr(self.func, name)

    def __str__(self) -> str:
        """Return a string representation of the PipelineFunction instance.

        Returns
        -------
        str
            A string representation of the PipelineFunction instance.

        """
        params = ", ".join(self.parameters)
        outputs = ", ".join(at_least_tuple(self.output_name))
        return f"{self.func.__name__}({params}) → {outputs}"

    def __repr__(self) -> str:
        """Return a string representation of the PipelineFunction instance.

        Returns
        -------
        str
            A string representation of the PipelineFunction instance.

        """
        return f"PipelineFunction({self.func.__name__})"

    def __getstate__(self) -> dict:
        """Prepare the state of the current object for pickling.

        The state includes all picklable instance variables.
        For non-picklable instance variable,  they are transformed
        into a picklable form or ignored.

        Returns
        -------
        state : dict
            A dictionary containing the picklable state of the object.

        """
        state = self.__dict__.copy()
        state["func"] = cloudpickle.dumps(state.pop("func"))
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore the state of the current object from the provided state.

        It also handles restoring non-picklable instance variable
        into their original form.

        Parameters
        ----------
        state : dict
            A dictionary containing the picklable state of the object.

        """
        self.__dict__.update(state)
        self.func = cloudpickle.loads(self.func)


def pipefunc(
    output_name: _OUTPUT_TYPE,
    *,
    output_picker: Callable[[Any, str], Any] | None = None,
    renames: dict[str, str] | None = None,
    profile: bool = False,
    debug: bool = False,
    cache: bool = False,
    save: bool | None = None,
    save_function: Callable[[str | Path, dict[str, Any]], None] | None = None,
) -> Callable[[Callable[..., Any]], PipelineFunction]:
    """A decorator for tagging pipeline functions with a return identifier.

    Parameters
    ----------
    output_name
        The identifier for the output of the decorated function.
    output_picker
        A function that takes the output of the wrapped function as first argument
        and the output_name (str) as second argument, and returns the desired output.
        If None, the output of the wrapped function is returned as is.
    renames
        A dictionary mapping from original argument names to new argument names.
    profile
        Flag indicating whether the decorated function should be profiled.
    debug
        Flag indicating whether debug information should be printed.
    cache
        Flag indicating whether the decorated function should be cached.
    save
        Flag indicating whether the output of the wrapped function should be saved.
    save_function
        A function that takes the filename and a dict containing the inputs and output.

    Returns
    -------
    Callable[[Callable[..., Any]], PipelineFunction]
        A decorator function that takes the original function and output_name a
        PipelineFunction instance with the specified return identifier.

    """

    def decorator(f: Callable[..., Any]) -> PipelineFunction:
        """Wraps the original function in a PipelineFunction instance.

        Parameters
        ----------
        f
            The original function to be wrapped.

        Returns
        -------
        PipelineFunction
            The wrapped function with the specified return identifier.

        """
        return PipelineFunction(
            f,
            output_name,
            output_picker=output_picker,
            renames=renames,
            profile=profile,
            debug=debug,
            cache=cache,
            save=save,
            save_function=save_function,
        )

    return decorator
