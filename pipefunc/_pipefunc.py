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
from typing import TYPE_CHECKING, Any, Generic, Tuple, TypeVar, Union

import cloudpickle

from pipefunc._lazy import evaluate_lazy
from pipefunc._perf import ProfilingStats, ResourceProfiler
from pipefunc._utils import at_least_tuple, format_function_call
from pipefunc.map._mapspec import MapSpec

if sys.version_info < (3, 9):  # pragma: no cover
    from typing import Callable
else:
    from collections.abc import Callable

if TYPE_CHECKING:
    from pathlib import Path

    if sys.version_info < (3, 10):  # pragma: no cover
        from typing_extensions import TypeAlias
    else:
        from typing import TypeAlias


T = TypeVar("T", bound=Callable[..., Any])
_OUTPUT_TYPE: TypeAlias = Union[str, Tuple[str, ...]]
MAX_PARAMS_LEN = 15


def _default_output_picker(
    output: Any,
    name: str,
    output_name: _OUTPUT_TYPE,
) -> Any:
    """Default output picker function for tuples."""
    return output[output_name.index(name)]


class PipeFunc(Generic[T]):
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
    save_function
        A function that takes the filename and a dict containing the inputs and output.
        If provided, the result will be saved.
    mapspec
        This is a specification for mapping that dictates how input values should
        be merged together. If None, the default behavior is that the input directly
        maps to the output.

    Returns
    -------
        The identifier for the output of the wrapped function.

    Examples
    --------
    >>> def add_one(a, b):
    ...     return a + 1, b + 1
    >>> add_one_func = PipeFunc(
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
        save_function: Callable[[str | Path, dict[str, Any]], None] | None = None,
        mapspec: str | MapSpec | None = None,
    ) -> None:
        """Function wrapper class for pipeline functions with additional attributes."""
        self.func: Callable[..., Any] = func
        self.output_name: _OUTPUT_TYPE = output_name
        self.debug = debug
        self.cache = cache
        self.save_function = save_function
        self.mapspec = MapSpec.from_string(mapspec) if isinstance(mapspec, str) else mapspec
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
        self._validate_mapspec()

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

        if self.debug:
            func_str = format_function_call(self.func.__name__, (), kwargs)
            msg = (
                f"Function returning '{self.output_name}' was invoked"
                f" as `{func_str}` and returned `{result}`."
            )
            if self.profiling_stats is not None:
                dt = self.profiling_stats.time.average
                msg += f" The execution time was {dt:.2e} seconds on average."
            print(msg)
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
        """Return a string representation of the PipeFunc instance.

        Returns
        -------
        str
            A string representation of the PipeFunc instance.

        """
        outputs = ", ".join(at_least_tuple(self.output_name))
        return f"{self.func.__name__}(...) → {outputs}"

    def __repr__(self) -> str:
        """Return a string representation of the PipeFunc instance.

        Returns
        -------
        str
            A string representation of the PipeFunc instance.

        """
        return f"PipeFunc({self.func.__name__})"

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
        state = {k: v for k, v in self.__dict__.items() if k != "func"}
        state["func"] = cloudpickle.dumps(self.func)
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

    def _validate_mapspec(self) -> None:
        if self.mapspec is None:
            return

        if not isinstance(self.mapspec, MapSpec):  # pragma: no cover
            msg = (
                "The 'mapspec' argument should be an instance of MapSpec,"
                f" not {type(self.mapspec)}."
            )
            raise TypeError(msg)

        mapspec_input_names = {x.name for x in self.mapspec.inputs}
        input_names = set(self.parameters)
        if extra := mapspec_input_names - input_names:
            msg = (
                f"The input of the function `{self.__name__}` should match"
                f" the input of the MapSpec `{self.mapspec}`:"
                f" `{extra} not in {input_names}`."
            )
            raise ValueError(msg)

        mapspec_output_names = {x.name for x in self.mapspec.outputs}
        output_names = set(at_least_tuple(self.output_name))
        if mapspec_output_names != output_names:
            msg = (
                f"The output of the function `{self.__name__}` should match"
                f" the output of the MapSpec `{self.mapspec}`:"
                f" `{mapspec_output_names} != {output_names}`."
            )
            raise ValueError(msg)


def pipefunc(
    output_name: _OUTPUT_TYPE,
    *,
    output_picker: Callable[[Any, str], Any] | None = None,
    renames: dict[str, str] | None = None,
    profile: bool = False,
    debug: bool = False,
    cache: bool = False,
    save_function: Callable[[str | Path, dict[str, Any]], None] | None = None,
    mapspec: str | MapSpec | None = None,
) -> Callable[[Callable[..., Any]], PipeFunc]:
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
    save_function
        A function that takes the filename and a dict containing the inputs and output.
        If provided, the result will be saved.
    mapspec
        This is a specification for mapping that dictates how input values should
        be merged together. If None, the default behavior is that the input directly
        maps to the output.

    Returns
    -------
    Callable[[Callable[..., Any]], PipeFunc]
        A decorator function that takes the original function and output_name a
        PipeFunc instance with the specified return identifier.

    """

    def decorator(f: Callable[..., Any]) -> PipeFunc:
        """Wraps the original function in a PipeFunc instance.

        Parameters
        ----------
        f
            The original function to be wrapped.

        Returns
        -------
        PipeFunc
            The wrapped function with the specified return identifier.

        """
        return PipeFunc(
            f,
            output_name,
            output_picker=output_picker,
            renames=renames,
            profile=profile,
            debug=debug,
            cache=cache,
            save_function=save_function,
            mapspec=mapspec,
        )

    return decorator
