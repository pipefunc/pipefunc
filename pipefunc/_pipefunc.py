"""PipeFunc: A Python library for defining, managing, and executing function pipelines.

This module implements the `PipeFunc` class, which is a function wrapper class for
pipeline functions with additional attributes. It also provides a decorator `pipefunc`
that wraps a function in a `PipeFunc` instance.
These `PipeFunc` objects are used to construct a `pipefunc.Pipeline`.
"""

from __future__ import annotations

import contextlib
import datetime
import functools
import inspect
import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar, Union

import cloudpickle

from pipefunc._perf import ProfilingStats, ResourceProfiler
from pipefunc._utils import at_least_tuple, clear_cached_properties, format_function_call
from pipefunc.lazy import evaluate_lazy
from pipefunc.map._mapspec import ArraySpec, MapSpec, mapspec_axes
from pipefunc.resources import Resources

if TYPE_CHECKING:
    from pathlib import Path


T = TypeVar("T", bound=Callable[..., Any])
_OUTPUT_TYPE: TypeAlias = Union[str, tuple[str, ...]]
MAX_PARAMS_LEN = 15


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
        and the ``output_name`` (str) as second argument, and returns the desired output.
        If ``None``, the output of the wrapped function is returned as is.
    renames
        A dictionary mapping from original argument names to new argument names.
    defaults
        Set defaults for parameters. Overwrites any current defaults. Must be in terms
        of the renamed argument names.
    bound
        Bind arguments to the function. These are arguments that are fixed. Even when
        providing different values, the bound values will be used. Must be in terms of
        the renamed argument names.
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
        be merged together. If ``None``, the default behavior is that the input directly
        maps to the output.
    resources
        A dictionary or `Resources` instance containing the resources required
        for the function. This can be used to specify the number of CPUs, GPUs,
        memory, wall time, queue, partition, and any extra job scheduler
        arguments. This is *not* used by the `pipefunc` directly but can be
        used by job schedulers to manage the resources required for the
        function.

    Returns
    -------
        The identifier for the output of the wrapped function.

    Examples
    --------
    >>> def add_one(a, b):
    ...     return a + 1, b + 1
    >>> add_one_func = PipeFunc(
    ...     add_one,
    ...     output_name="c",
    ...     renames={"a": "x", "b": "y"},
    ... )
    >>> add_one_func(x=1, y=2)
    (2, 3)
    >>> add_one_func.update_defaults({"x": 1, "y": 1})
    >>> add_one_func()
    (2, 2)

    """

    def __init__(
        self,
        func: T,
        output_name: _OUTPUT_TYPE,
        *,
        output_picker: Callable[[str, Any], Any] | None = None,
        renames: dict[str, str] | None = None,
        defaults: dict[str, Any] | None = None,
        bound: dict[str, Any] | None = None,
        profile: bool = False,
        debug: bool = False,
        cache: bool = False,
        save_function: Callable[[str | Path, dict[str, Any]], None] | None = None,
        mapspec: str | MapSpec | None = None,
        resources: dict | Resources | None = None,
    ) -> None:
        """Function wrapper class for pipeline functions with additional attributes."""
        self.func: Callable[..., Any] = func
        self.output_name: _OUTPUT_TYPE = output_name
        self.debug = debug
        self.cache = cache
        self.save_function = save_function
        self.mapspec = _maybe_mapspec(mapspec)
        self._output_picker: Callable[[Any, str], Any] | None = output_picker
        self._profile = profile
        self._renames: dict[str, str] = renames or {}
        self._defaults: dict[str, Any] = defaults or {}
        self._bound: dict[str, Any] = bound or {}
        self.resources = _maybe_resources(resources)
        self.profiling_stats: ProfilingStats | None
        self.set_profiling(enable=profile)
        self._validate_mapspec()
        self._validate_names()

    @property
    def renames(self) -> dict[str, str]:
        """Return the renames for the function arguments.

        See Also
        --------
        update_renames
            Update the ``renames`` via this method.

        """
        # Is a property to prevent users mutating the renames directly
        return self._renames

    @property
    def bound(self) -> dict[str, Any]:
        """Return the bound arguments for the function. These are arguments that are fixed.

        See Also
        --------
        update_bound
            Update the ``bound`` parameters via this method.

        """
        # Is a property to prevent users mutating `bound` directly
        return self._bound

    @functools.cached_property
    def parameters(self) -> tuple[str, ...]:
        return tuple(self._renames.get(k, k) for k in self.original_parameters)

    @property
    def original_parameters(self) -> dict[str, inspect.Parameter]:
        """Return the original (before renames) parameters of the wrapped function.

        Returns
        -------
            A mapping of the original parameters of the wrapped function to their
            respective `inspect.Parameter` objects.

        """
        return dict(inspect.signature(self.func).parameters)

    @functools.cached_property
    def defaults(self) -> dict[str, Any]:
        """Return the defaults for the function arguments.

        Returns
        -------
            A dictionary of default values for the keyword arguments.

        See Also
        --------
        update_defaults
            Update the ``defaults`` via this method.

        """
        parameters = self.original_parameters
        defaults = {}
        for original_name, v in parameters.items():
            new_name = self._renames.get(original_name, original_name)
            if new_name in self._defaults:
                defaults[new_name] = self._defaults[new_name]
            elif v.default is not inspect.Parameter.empty and new_name not in self._bound:
                defaults[new_name] = v.default
        return defaults

    @functools.cached_property
    def _inverse_renames(self) -> dict[str, str]:
        return {v: k for k, v in self._renames.items()}

    @functools.cached_property
    def output_picker(self) -> Callable[[Any, str], Any] | None:
        """Return the output picker function for the wrapped function.

        The output picker function takes the output of the wrapped function as first
        argument and the ``output_name`` (str) as second argument, and returns the
        desired output.
        """
        if self._output_picker is None and isinstance(self.output_name, tuple):
            return functools.partial(_default_output_picker, output_name=self.output_name)
        return self._output_picker

    def update_defaults(self, defaults: dict[str, Any], *, overwrite: bool = False) -> None:
        """Update defaults to the provided keyword arguments.

        Parameters
        ----------
        defaults
            A dictionary of default values for the keyword arguments.
        overwrite
            Whether to overwrite the existing defaults. If ``False``, the new
            defaults will be added to the existing defaults.

        """
        self._validate_update(defaults, "defaults", self.parameters)
        if overwrite:
            self._defaults = defaults.copy()
        else:
            self._defaults = dict(self._defaults, **defaults)
        clear_cached_properties(self, PipeFunc)

    def update_renames(
        self,
        renames: dict[str, str],
        *,
        update_from: Literal["current", "original"] = "current",
        overwrite: bool = False,
    ) -> None:
        """Update renames to function arguments for the wrapped function.

        Parameters
        ----------
        renames
            A dictionary of renames for the function arguments.
        update_from
            Whether to update the renames from the current parameter names (`PipeFunc.parameters`)
            or from the original parameter names (`PipeFunc.original_parameters`).
        overwrite
            Whether to overwrite the existing renames. If ``False``, the new
            renames will be added to the existing renames.

        """
        assert update_from in ("current", "original")
        self._validate_update(
            renames,
            "renames",
            self.parameters if update_from == "current" else self.original_parameters.keys(),  # type: ignore[arg-type]
        )
        if update_from == "current":
            renames = {
                self._inverse_renames.get(k, k): v
                for k, v in renames.items()
                if k in self.parameters
            }
        old_inverse = self._inverse_renames.copy()
        bound_original = {old_inverse.get(k, k): v for k, v in self._bound.items()}
        if overwrite:
            self._renames = renames.copy()
        else:
            self._renames = dict(self._renames, **renames)

        # Update defaults with new renames
        new_defaults = {}
        for name, value in self._defaults.items():
            if original_name := old_inverse.get(name):
                name = self._renames.get(original_name, original_name)  # noqa: PLW2901
            new_defaults[name] = value
        self._defaults = new_defaults

        # Update bound with new renames
        new_bound = {}
        for name, value in bound_original.items():
            new_name = self._renames.get(name, name)
            new_bound[new_name] = value
        self._bound = new_bound

        clear_cached_properties(self, PipeFunc)

    def update_bound(self, bound: dict[str, Any], *, overwrite: bool = False) -> None:
        """Update the bound arguments for the function that are fixed.

        Parameters
        ----------
        bound
            A dictionary of bound arguments for the function.
        overwrite
            Whether to overwrite the existing bound arguments. If ``False``, the new
            bound arguments will be added to the existing bound arguments.

        """
        self._validate_update(bound, "bound", self.parameters)
        if overwrite:
            self._bound = bound.copy()
        else:
            self._bound = dict(self._bound, **bound)

        clear_cached_properties(self, PipeFunc)

    def _validate_update(
        self,
        update: dict[str, Any],
        name: str,
        parameters: tuple[str, ...],
    ) -> None:
        if extra := set(update) - set(parameters):
            msg = (
                f"Unexpected `{name}` arguments: `{extra}`."
                f" The allowed arguments are: `{parameters}`."
                f" The provided arguments are: `{update}`."
            )
            raise ValueError(msg)

        for key in update:
            _validate_identifier(key, name)

    def _validate_names(self) -> None:
        if common := set(self._defaults) & set(self._bound):
            msg = (
                f"The following parameters are both defaults and bound: `{common}`."
                " This is not allowed."
            )
            raise ValueError(msg)

        self._validate_update(self._renames, "renames", self.original_parameters)  # type: ignore[arg-type]
        self._validate_update(self._defaults, "defaults", self.parameters)
        self._validate_update(self._bound, "bound", self.parameters)
        if not isinstance(self.output_name, str | tuple):
            msg = (
                f"The output name should be a string or a tuple of strings,"
                f" not {type(self.output_name)}."
            )
            raise TypeError(msg)
        for name in at_least_tuple(self.output_name):
            _validate_identifier("output_name", name)

    def copy(self) -> PipeFunc:
        return PipeFunc(
            self.func,
            self.output_name,
            output_picker=self._output_picker,
            renames=self._renames,
            defaults=self._defaults,
            bound=self._bound,
            profile=self._profile,
            debug=self.debug,
            cache=self.cache,
            save_function=self.save_function,
            mapspec=self.mapspec,
            resources=self.resources,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function with the given arguments.

        Returns
        -------
            The return value of the wrapped function.

        """
        if extra := set(kwargs) - set(self.parameters):
            msg = (
                f"Unexpected keyword arguments: `{extra}`."
                f" The allowed arguments are: `{self.parameters}`."
                f" The provided arguments are: `{kwargs}`."
            )
            raise ValueError(msg)

        kwargs = self.defaults | kwargs | self._bound
        kwargs = {self._inverse_renames.get(k, k): v for k, v in kwargs.items()}

        with self._maybe_profiler():
            args = evaluate_lazy(args)
            kwargs = evaluate_lazy(kwargs)
            result = self.func(*args, **kwargs)

        if self.debug:
            func_str = format_function_call(self.func.__name__, (), kwargs)
            now = datetime.datetime.now()  # noqa: DTZ005
            msg = (
                f"{now} - Function returning '{self.output_name}' was invoked"
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
            A `ResourceProfiler` instance if profiling is enabled, or a
            `nullcontext` if disabled.

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
            The value of the attribute.

        """
        return getattr(self.func, name)

    def __str__(self) -> str:
        """Return a string representation of the PipeFunc instance.

        Returns
        -------
            A string representation of the PipeFunc instance.

        """
        outputs = ", ".join(at_least_tuple(self.output_name))
        return f"{self.func.__name__}(...) → {outputs}"

    def __repr__(self) -> str:
        """Return a string representation of the PipeFunc instance.

        Returns
        -------
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
        state
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

        mapspec_input_names = set(self.mapspec.input_names)
        if extra := mapspec_input_names - set(self.parameters):
            msg = (
                f"The input of the function `{self.__name__}` should match"
                f" the input of the MapSpec `{self.mapspec}`:"
                f" `{extra} not in {self.parameters}`."
            )
            raise ValueError(msg)

        mapspec_output_names = set(self.mapspec.output_names)
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
    defaults: dict[str, Any] | None = None,
    bound: dict[str, Any] | None = None,
    profile: bool = False,
    debug: bool = False,
    cache: bool = False,
    save_function: Callable[[str | Path, dict[str, Any]], None] | None = None,
    mapspec: str | MapSpec | None = None,
    resources: dict | Resources | None = None,
) -> Callable[[Callable[..., Any]], PipeFunc]:
    """A decorator that wraps a function in a PipeFunc instance.

    Parameters
    ----------
    output_name
        The identifier for the output of the decorated function.
    output_picker
        A function that takes the output of the wrapped function as first argument
        and the ``output_name`` (str) as second argument, and returns the desired output.
        If ``None``, the output of the wrapped function is returned as is.
    renames
        A dictionary mapping from original argument names to new argument names.
    defaults
        Set defaults for parameters. Overwrites any current defaults. Must be in terms
        of the renamed argument names.
    bound
        Bind arguments to the function. These are arguments that are fixed. Even when
        providing different values, the bound values will be used. Must be in terms of
        the renamed argument names.
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
        be merged together. If ``None``, the default behavior is that the input directly
        maps to the output.
    resources
        A dictionary or `Resources` instance containing the resources required
        for the function. This can be used to specify the number of CPUs, GPUs,
        memory, wall time, queue, partition, and any extra job scheduler
        arguments. This is *not* used by the `pipefunc` directly but can be
        used by job schedulers to manage the resources required for the
        function.

    Returns
    -------
        A decorator function that takes the original function and ``output_name`` and
        creates a `PipeFunc` instance with the specified return identifier.

    See Also
    --------
    PipeFunc
        A function wrapper class for pipeline functions with additional attributes.

    Examples
    --------
    >>> @pipefunc(output_name="c")
    ... def add(a, b):
    ...     return a + b
    >>> add(a=1, b=2)
    3
    >>> add.update_renames({"a": "x", "b": "y"})
    >>> add(x=1, y=2)
    3

    """

    def decorator(f: Callable[..., Any]) -> PipeFunc:
        """Wraps the original function in a PipeFunc instance.

        Parameters
        ----------
        f
            The original function to be wrapped.

        Returns
        -------
            The wrapped function with the specified return identifier.

        """
        return PipeFunc(
            f,
            output_name,
            output_picker=output_picker,
            renames=renames,
            defaults=defaults,
            bound=bound,
            profile=profile,
            debug=debug,
            cache=cache,
            save_function=save_function,
            mapspec=mapspec,
            resources=resources,
        )

    return decorator


class NestedPipeFunc(PipeFunc):
    """Combine multiple `PipeFunc` instances into a single function with an internal `Pipeline`.

    Parameters
    ----------
    pipefuncs
        A sequence of at least 2 `PipeFunc` instances to combine into a single function.
    output_name
        The identifier for the output of the wrapped function. If ``None``, it is automatically
        constructed from all the output names of the `PipeFunc` instances.
    mapspec
        `~pipefunc.map.MapSpec` for the joint function. If ``None``, the mapspec is inferred
        from the individual `PipeFunc` instances. None of the `MapsSpec` instances should
        have a reduction and all should use identical axes.
    resources
        Same as the `PipeFunc` class. However, if it is ``None`` here, it is inferred from
        from the `PipeFunc` instances. Specifically, it takes the maximum of the resources.

    Attributes
    ----------
    pipefuncs
        List of `PipeFunc` instances (copies of input) that are used in the internal ``pipeline``.
    pipeline
        The `Pipeline` instance that manages the `PipeFunc` instances.

    Notes
    -----
    The `NestedPipeFunc` class is a subclass of the `PipeFunc` class that allows you to
    combine multiple `PipeFunc` instances into a single function that has an internal
    `~pipefunc.Pipeline` instance.

    """

    def __init__(
        self,
        pipefuncs: Sequence[PipeFunc],
        *,
        output_name: _OUTPUT_TYPE | None = None,
        mapspec: str | MapSpec | None = None,
        resources: dict | Resources | None = None,
    ) -> None:
        from pipefunc import Pipeline

        _validate_pipefuncs(pipefuncs)
        self.pipefuncs: list[PipeFunc] = [f.copy() for f in pipefuncs]
        self.pipeline = Pipeline(self.pipefuncs)  # type: ignore[arg-type]
        _validate_single_leaf_node(self.pipeline.leaf_nodes)
        _validate_output_name(output_name, self._all_outputs)
        self.output_name: _OUTPUT_TYPE = output_name or self._all_outputs
        self.debug = False  # The underlying PipeFuncs will handle this
        self.cache = any(f.cache for f in self.pipefuncs)
        self.save_function = None
        self._output_picker = None
        self._profile = False
        self._renames: dict[str, str] = {}
        self._defaults: dict[str, Any] = {
            k: v for k, v in self.pipeline.defaults.items() if k in self.parameters
        }
        self._bound: dict[str, Any] = {}
        self.resources = _maybe_max_resources(resources, self.pipefuncs)
        self.profiling_stats = None
        self.mapspec = self._combine_mapspecs() if mapspec is None else _maybe_mapspec(mapspec)
        for f in self.pipefuncs:
            f.mapspec = None  # MapSpec is handled by the NestedPipeFunc
        self._validate_mapspec()
        self._validate_names()

    def copy(self) -> NestedPipeFunc:
        # Pass the mapspec to the new instance because we set
        # the child mapspecs to None in the __init__
        return NestedPipeFunc(self.pipefuncs, output_name=self.output_name, mapspec=self.mapspec)

    def _combine_mapspecs(self) -> MapSpec | None:
        mapspecs = [f.mapspec for f in self.pipefuncs]
        if all(m is None for m in mapspecs):
            return None
        _validate_combinable_mapspecs(mapspecs)
        axes = mapspec_axes(mapspecs)  # type: ignore[arg-type]
        return MapSpec(
            tuple(ArraySpec(n, axes[n]) for n in sorted(self.parameters)),
            tuple(ArraySpec(n, axes[n]) for n in sorted(at_least_tuple(self.output_name))),
        )

    @functools.cached_property
    def original_parameters(self) -> dict[str, Any]:
        parameters = set(self._all_inputs) - set(self._all_outputs)
        return {k: inspect.Parameter(k, inspect.Parameter.KEYWORD_ONLY) for k in sorted(parameters)}

    @functools.cached_property
    def _all_outputs(self) -> tuple[str, ...]:
        outputs: set[str] = set()
        for f in self.pipefuncs:
            outputs.update(at_least_tuple(f.output_name))
        return tuple(sorted(outputs))

    @functools.cached_property
    def _all_inputs(self) -> tuple[str, ...]:
        inputs: set[str] = set()
        for f in self.pipefuncs:
            inputs.update(f.parameters)
        return tuple(sorted(inputs))

    @functools.cached_property
    def func(self) -> Callable[..., tuple[Any, ...]]:  # type: ignore[override]
        func = self.pipeline.func(self.pipeline.unique_leaf_node.output_name)
        return _NestedFuncWrapper(func.call_full_output, self.output_name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pipefuncs={self.pipefuncs})"


def _maybe_max_resources(
    resources: dict | Resources | None,
    pipefuncs: list[PipeFunc],
) -> Resources | None:
    if isinstance(resources, Resources):
        return resources
    if resources is not None:
        return Resources.from_dict(resources)
    resources_list = [f.resources for f in pipefuncs if f.resources is not None]
    if len(resources_list) == 1:
        return resources_list[0]
    if not resources_list:
        return None
    return Resources.combine_max(resources_list)


def _maybe_resources(resources: dict | Resources | None) -> Resources | None:
    if resources is None:
        return None
    if isinstance(resources, Resources):
        return resources
    return Resources.from_dict(resources)


class _NestedFuncWrapper:
    """Wrapper class for nested functions.

    Takes a function that returns a dictionary and returns a tuple of values in the
    order specified by the output_name.
    """

    def __init__(self, func: Callable[..., dict[str, Any]], output_name: _OUTPUT_TYPE) -> None:
        self.func: Callable[..., dict[str, Any]] = func
        self.output_name: _OUTPUT_TYPE = output_name
        self.__name__ = f"NestedPipeFunc_{'_'.join(at_least_tuple(output_name))}"

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        result_dict = self.func(*args, **kwds)
        if isinstance(self.output_name, str):
            return result_dict[self.output_name]
        return tuple(result_dict[name] for name in self.output_name)


def _validate_identifier(name: str, value: Any) -> None:
    if not value.isidentifier():
        msg = f"The `{name}` should contain/be valid Python identifier(s), not `{value}`."
        raise ValueError(msg)


def _validate_pipefuncs(pipefuncs: Sequence[PipeFunc]) -> None:
    if not all(isinstance(f, PipeFunc) for f in pipefuncs):
        msg = "All elements in `pipefuncs` should be instances of `PipeFunc`."
        raise TypeError(msg)

    if len(pipefuncs) < 2:  # noqa: PLR2004
        msg = "The provided `pipefuncs` should have at least two `PipeFunc`s."
        raise ValueError(msg)


def _validate_single_leaf_node(leaf_nodes: list[PipeFunc]) -> None:
    if len(leaf_nodes) > 1:
        msg = f"The provided `pipefuncs` should have only one leaf node, not {len(leaf_nodes)}."
        raise ValueError(msg)


def _validate_output_name(output_name: _OUTPUT_TYPE | None, all_outputs: tuple[str, ...]) -> None:
    if output_name is None:
        return
    if not all(x in all_outputs for x in at_least_tuple(output_name)):
        msg = f"The provided `{output_name=}` should be a subset of the combined output names: {all_outputs}."
        raise ValueError(msg)


def _validate_combinable_mapspecs(mapspecs: list[MapSpec | None]) -> None:
    if any(m is None for m in mapspecs):
        msg = "Cannot combine a mix of None and MapSpec instances."
        raise ValueError(msg)
    assert len(mapspecs) > 1

    first = mapspecs[0]
    assert first is not None
    for m in mapspecs:
        assert m is not None
        if m.input_indices != set(m.output_indices):
            msg = "Cannot combine MapSpecs with different input and output mappings."
            raise ValueError(msg)
        if m.input_indices != first.input_indices:
            msg = "Cannot combine MapSpecs with different input mappings."
            raise ValueError(msg)
        if m.output_indices != first.output_indices:
            msg = "Cannot combine MapSpecs with different output mappings."
            raise ValueError(msg)


def _default_output_picker(
    output: Any,
    name: str,
    output_name: _OUTPUT_TYPE,
) -> Any:
    """Default output picker function for tuples."""
    return output[output_name.index(name)]


def _maybe_mapspec(mapspec: str | MapSpec | None) -> MapSpec | None:
    return MapSpec.from_string(mapspec) if isinstance(mapspec, str) else mapspec
