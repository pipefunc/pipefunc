"""PipeFunc: A Python library for defining, managing, and executing function pipelines.

This module implements the `PipeFunc` class, which is a function wrapper class for
pipeline functions with additional attributes. It also provides a decorator `pipefunc`
that wraps a function in a `PipeFunc` instance.
These `PipeFunc` objects are used to construct a `pipefunc.Pipeline`.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import functools
import getpass
import inspect
import os
import platform
import traceback
import warnings
import weakref
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, get_args, get_origin

import cloudpickle

from pipefunc._profile import ProfilingStats, ResourceProfiler
from pipefunc._utils import (
    assert_complete_kwargs,
    at_least_tuple,
    clear_cached_properties,
    format_function_call,
    get_local_ip,
    is_classmethod,
    is_pydantic_base_model,
    requires,
)
from pipefunc.lazy import evaluate_lazy
from pipefunc.map._mapspec import ArraySpec, MapSpec, mapspec_axes
from pipefunc.map._run import _EVALUATED_RESOURCES
from pipefunc.resources import Resources
from pipefunc.typing import NoAnnotation, safe_get_type_hints

if TYPE_CHECKING:
    from pathlib import Path

    import pydantic

    from pipefunc import Pipeline
    from pipefunc._pipeline._types import OUTPUT_TYPE
    from pipefunc.map._types import ShapeTuple

T = TypeVar("T", bound=Callable[..., Any])

MAX_PARAMS_LEN = 15


class PipeFunc(Generic[T]):
    """Function wrapper class for pipeline functions with additional attributes.

    Parameters
    ----------
    func
        The original function to be wrapped.
    output_name
        The identifier for the output of the wrapped function.
        Provide a tuple of strings for multiple outputs.
    output_picker
        A function that takes the output of the wrapped function as first argument
        and the ``output_name`` (str) as second argument, and returns the desired output.
        If ``None``, the output of the wrapped function is returned as is.
    renames
        A dictionary for renaming function arguments and outputs. The keys are the
        original names (as defined in the function signature or the ``output_name``),
        and the values are the new names to be used. This allows you to change how
        the function is called without modifying its internal logic. For example,
        ``{"old_name": "new_name"}`` would allow the function to be called with
        ``new_name`` instead of ``old_name``. If renaming the ``output_name``, include it
        in this dictionary as well.
    defaults
        Set defaults for parameters. Overwrites any current defaults. Must be in terms
        of the renamed argument names.
    bound
        Bind arguments to the function. These are arguments that are fixed. Even when
        providing different values, the bound values will be used. Must be in terms of
        the renamed argument names.
    profile
        Flag indicating whether the wrapped function should be profiled.
        Profiling is only available for sequential execution.
    debug
        Flag indicating whether debug information should be printed.
    cache
        Flag indicating whether the wrapped function should be cached.
    mapspec
        This is a specification for mapping that dictates how input values should
        be merged together. If ``None``, the default behavior is that the input directly
        maps to the output.
    internal_shape
        The shape of the output produced by this function *when it is used within a
        ``mapspec`` context*. Can be an int or a tuple of ints, or "?" for unknown
        dimensions, or a tuple with a mix of both. If not provided, the shape will be
        inferred from the first execution of the function. If provided, the shape will be
        validated against the actual shape of the output. This parameters is required only
        when a `mapspec` like `... -> out[i]` is used, indicating that the shape cannot be
        derived from the inputs. In case there are multiple outputs, provide the shape for
        one of the outputs. This works because the shape of all outputs are required to be
        identical.
    post_execution_hook
        A callback function that is invoked after the function is executed.
        The callback signature is ``hook(func: PipeFunc, result: Any, kwargs: dict) -> None``.
        This hook can be used for logging, visualization of intermediate results,
        debugging, statistics collection, or other side effects. The hook is executed
        synchronously after the function returns but before the result is passed to
        the next function in the pipeline. Keep the hook lightweight to avoid impacting performance.
    resources
        A dictionary or `Resources` instance containing the resources required
        for the function. This can be used to specify the number of CPUs, GPUs,
        memory, wall time, queue, partition, and any extra job scheduler
        arguments. This is *not* used by the `pipefunc` directly but can be
        used by job schedulers to manage the resources required for the
        function. Alternatively, provide a callable that receives a dict with the
        input values and returns a `Resources` instance.
    resources_variable
        If provided, the resources will be passed as the specified argument name to the function.
        This requires that the function has a parameter with the same name. For example,
        if ``resources_variable="resources"``, the function will be called as
        ``func(..., resources=Resources(...))``. This is useful when the function handles internal
        parallelization.
    resources_scope
        Determines how resources are allocated in relation to the mapspec:

        - "map": Allocate resources for the entire mapspec operation (default).
        - "element": Allocate resources for each element in the mapspec.

        If no mapspec is defined, this parameter is ignored.
    scope
        If provided, *all* parameter names and output names of the function will
        be prefixed with the specified scope followed by a dot (``'.'``), e.g., parameter
        ``x`` with scope ``foo`` becomes ``foo.x``. This allows multiple functions in a
        pipeline to have parameters with the same name without conflict. To be selective
        about which parameters and outputs to include in the scope, use the
        `PipeFunc.update_scope` method.

        When providing parameter values for functions that have scopes, they can
        be provided either as a dictionary for the scope, or by using the
        ``f'{scope}.{name}'`` notation. For example,
        a `PipeFunc` instance with scope "foo" and "bar", the parameters
        can be provided as: ``func(foo=dict(a=1, b=2), bar=dict(a=3, b=4))``
        or ``func(**{"foo.a": 1, "foo.b": 2, "bar.a": 3, "bar.b": 4})``.
    variant
        Identifies this function as an alternative implementation in a
        `VariantPipeline` and specifies which variant groups it belongs to.
        When multiple functions share the same `output_name`, variants allow
        selecting which implementation to use during pipeline execution.

        Can be specified in two formats:
        - A string (e.g., ``"fast"``): Places the function in the default unnamed
          group (None) with the specified variant name. Equivalent to ``{None: "fast"}``.
        - A dictionary (e.g., ``{"algorithm": "fast", "optimization": "level1"}``):
          Assigns the function to multiple variant groups simultaneously, with a
          specific variant name in each group.

        Functions with the same `output_name` but different variant specifications
        represent alternative implementations. The {meth}`VariantPipeline.with_variant`
        method selects which variants to use for execution. For example, you might
        have "preprocessing" variants ("v1"/"v2") independent from "computation"
        variants ("fast"/"accurate"), allowing you to select specific combinations
        like ``{"preprocessing": "v1", "computation": "fast"}``.
    variant_group
        DEPRECATED in v0.58.0: Use `variant` instead.

    Returns
    -------
        A `PipeFunc` instance that wraps the original function with the specified return identifier.

    Attributes
    ----------
    error_snapshot
        If an error occurs while calling the function, this attribute will contain
        an `ErrorSnapshot` instance with information about the error.

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
        output_name: OUTPUT_TYPE,
        *,
        output_picker: Callable[[Any, str], Any] | None = None,
        renames: dict[str, str] | None = None,
        defaults: dict[str, Any] | None = None,
        bound: dict[str, Any] | None = None,
        profile: bool = False,
        debug: bool = False,
        cache: bool = False,
        mapspec: str | MapSpec | None = None,
        internal_shape: int | Literal["?"] | ShapeTuple | None = None,
        post_execution_hook: Callable[[PipeFunc, Any, dict[str, Any]], None] | None = None,
        resources: dict
        | Resources
        | Callable[[dict[str, Any]], Resources | dict[str, Any]]
        | None = None,
        resources_variable: str | None = None,
        resources_scope: Literal["map", "element"] = "map",
        scope: str | None = None,
        variant: str | dict[str | None, str] | None = None,
        variant_group: str | None = None,  # deprecated
    ) -> None:
        """Function wrapper class for pipeline functions with additional attributes."""
        self._pipelines: weakref.WeakSet[Pipeline] = weakref.WeakSet()
        self.func: Callable[..., Any] = func
        self.__name__ = _get_name(func)
        self._output_name: OUTPUT_TYPE = output_name
        self.debug = debug
        self.cache = cache
        self.mapspec = _maybe_mapspec(mapspec)
        self.internal_shape: int | Literal["?"] | ShapeTuple | None = internal_shape
        self.post_execution_hook = post_execution_hook
        self._output_picker: Callable[[Any, str], Any] | None = output_picker
        self.profile = profile
        self._renames: dict[str, str] = renames or {}
        self._defaults: dict[str, Any] = defaults or {}
        self._bound: dict[str, Any] = bound or {}
        self.resources = Resources.maybe_from_dict(resources)
        self.resources_variable = resources_variable
        self.resources_scope: Literal["map", "element"] = resources_scope
        _maybe_variant_group_error(variant_group, variant)
        self.variant = _ensure_variant(variant)
        self.profiling_stats: ProfilingStats | None
        if scope is not None:
            self.update_scope(scope, inputs="*", outputs="*")
        self._validate()
        self.error_snapshot: ErrorSnapshot | None = None

    @property
    def renames(self) -> dict[str, str]:
        """Return the renames for the function arguments and output name.

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
    def output_name(self) -> OUTPUT_TYPE:
        """Return the output name(s) of the wrapped function.

        Returns
        -------
            The output name(s) of the wrapped function.

        """
        return _rename_output_name(self._output_name, self._renames)

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
        parameters = dict(inspect.signature(self.func).parameters)
        if self.resources_variable is not None:
            del parameters[self.resources_variable]
        return parameters

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

        # Handle dataclass case
        if dataclasses.is_dataclass(self.func):
            fields = dataclasses.fields(self.func)
            for f in fields:
                new_name = self._renames.get(f.name, f.name)
                if new_name in self._defaults:
                    defaults[new_name] = self._defaults[new_name]
                elif f.default_factory is not dataclasses.MISSING:
                    defaults[new_name] = f.default_factory()
                elif f.default is not dataclasses.MISSING:
                    defaults[new_name] = f.default
            return defaults

        # Handle pydantic case
        if is_pydantic_base_model(self.func):
            return _pydantic_defaults(self.func, self._renames, self._defaults)

        # Handle regular function case
        for original_name, v in parameters.items():
            new_name = self._renames.get(original_name, original_name)
            if new_name in self._defaults:
                defaults[new_name] = self._defaults[new_name]
            elif v.default is not inspect.Parameter.empty and new_name not in self._bound:
                defaults[new_name] = v.default
        return defaults

    @functools.cached_property
    def _inverse_renames(self) -> dict[str, str]:
        """Renames from current name to original name."""
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
        self._clear_internal_cache()
        self._validate()

    def update_renames(
        self,
        renames: dict[str, str],
        *,
        update_from: Literal["current", "original"] = "current",
        overwrite: bool = False,
    ) -> None:
        """Update renames to function arguments and ``output_name`` for the wrapped function.

        When renaming the ``output_name`` and if it is a tuple of strings, the
        renames must be provided as individual strings in the tuple.

        Parameters
        ----------
        renames
            A dictionary of renames for the function arguments or ``output_name``.
        update_from
            Whether to update the renames from the ``"current"`` parameter names
            (`PipeFunc.parameters`) or from the ``"original"`` parameter names as
            in the function signature (`PipeFunc.original_parameters`). If also updating
            the ``output_name``, original means the name that was provided to the
            `PipeFunc` instance.
        overwrite
            Whether to overwrite the existing renames. If ``False``, the new
            renames will be added to the existing renames.

        """
        assert update_from in ("current", "original")
        assert all(isinstance(k, str) for k in renames.keys())  # noqa: SIM118
        assert all(isinstance(v, str) for v in renames.values())
        allowed_parameters = tuple(
            self.parameters + at_least_tuple(self.output_name)
            if update_from == "current"
            else tuple(self.original_parameters) + at_least_tuple(self._output_name),
        )
        self._validate_update(renames, "renames", allowed_parameters)
        if update_from == "current":
            # Convert to `renames` in terms of original names
            renames = {
                self._inverse_renames.get(k, k): v
                for k, v in renames.items()
                if k in allowed_parameters
            }
        old_inverse = self._inverse_renames.copy()
        bound_original = {old_inverse.get(k, k): v for k, v in self._bound.items()}
        defaults_original = {old_inverse.get(k, k): v for k, v in self._defaults.items()}
        if overwrite:
            self._renames = renames.copy()
        else:
            self._renames = dict(self._renames, **renames)

        # Update `defaults`
        new_defaults = {}
        for name, value in defaults_original.items():
            name = self._renames.get(name, name)  # noqa: PLW2901
            new_defaults[name] = value
        self._defaults = new_defaults

        # Update `bound`
        new_bound = {}
        for name, value in bound_original.items():
            new_name = self._renames.get(name, name)
            new_bound[new_name] = value
        self._bound = new_bound

        # Update `mapspec`
        if self.mapspec is not None:
            self.mapspec = self.mapspec.rename(old_inverse).rename(self._renames)

        self._clear_internal_cache()
        self._validate()

    def update_scope(
        self,
        scope: str | None,
        inputs: set[str] | Literal["*"] | None = None,
        outputs: set[str] | Literal["*"] | None = None,
        exclude: set[str] | None = None,
    ) -> None:
        """Update the scope for the `PipeFunc` by adding (or removing) a prefix to the input and output names.

        This method updates the names of the specified inputs and outputs by adding the provided
        scope as a prefix. The scope is added to the names using the format `f"{scope}.{name}"`.
        If an input or output name already starts with the scope prefix, it remains unchanged.
        If their is an existing scope, it is replaced with the new scope.

        Internally, simply calls `PipeFunc.update_renames` with  ``renames={name: f"{scope}.{name}", ...}``.

        When providing parameter values for functions that have scopes, they can
        be provided either as a dictionary for the scope, or by using the
        ``f'{scope}.{name}'`` notation. For example,
        a `PipeFunc` instance with scope "foo" and "bar", the parameters
        can be provided as: ``func(foo=dict(a=1, b=2), bar=dict(a=3, b=4))``
        or ``func(**{"foo.a": 1, "foo.b": 2, "bar.a": 3, "bar.b": 4})``.

        Parameters
        ----------
        scope
            The scope to set for the inputs and outputs. If ``None``, the scope of inputs and outputs is removed.
        inputs
            Specific input names to include, or "*" to include all inputs. If None, no inputs are included.
        outputs
            Specific output names to include, or "*" to include all outputs. If None, no outputs are included.
        exclude
            Names to exclude from the scope. This can include both inputs and outputs. Can be used with `inputs`
            or `outputs` being "*" to exclude specific names.

        Examples
        --------
        >>> f.update_scope("my_scope", inputs="*", outputs="*")  # Add scope to all inputs and outputs
        >>> f.update_scope("my_scope", "*", "*", exclude={"output1"}) # Add to all except "output1"
        >>> f.update_scope("my_scope", inputs="*", outputs={"output2"})  # Add scope to all inputs and "output2"
        >>> f.update_scope(None, inputs="*", outputs="*")  # Remove scope from all inputs and outputs

        """
        if scope is not None and (
            scope in self.unscoped_parameters or scope in at_least_tuple(self.output_name)
        ):
            msg = f"The provided `{scope=}` cannot be identical to the function input parameters or output name."
            raise ValueError(msg)

        if exclude is None:
            exclude = set()

        if inputs == "*":
            inputs = set(self.parameters)
        elif inputs is None:
            inputs = set()
        else:
            inputs = set(inputs)  # Ensure it is a set

        if outputs == "*":
            outputs = set(at_least_tuple(self.output_name))
        elif outputs is None:
            outputs = set()
        else:
            outputs = set(outputs)  # Ensure it is a set

        all_parameters = (inputs | outputs) - exclude
        assert all_parameters
        renames = {name: _prepend_name_with_scope(name, scope) for name in all_parameters}
        self.update_renames(renames, update_from="current")

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
        self._clear_internal_cache()
        self._validate()

    def _clear_internal_cache(self, *, clear_pipelines: bool = True) -> None:
        clear_cached_properties(self, PipeFunc)
        if clear_pipelines:
            for pipeline in self._pipelines:
                pipeline._clear_internal_cache()

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

        for key, value in update.items():
            _validate_identifier(name, key)
            if name == "renames":
                _validate_identifier(name, value)

    def _validate(self) -> None:
        self._validate_names()
        self._validate_mapspec()

    def _validate_names(self) -> None:
        if common := set(self._defaults) & set(self._bound):
            msg = (
                f"The following parameters are both defaults and bound: `{common}`."
                " This is not allowed."
            )
            raise ValueError(msg)
        if not isinstance(self._output_name, str | tuple):
            msg = (
                f"The `output_name` should be a string or a tuple of strings,"
                f" not {type(self._output_name)}."
            )
            raise TypeError(msg)
        if self.resources_variable is not None:
            try:
                self.original_parameters  # noqa: B018
            except KeyError as e:
                msg = (
                    f"The `resources_variable={self.resources_variable!r}`"
                    " should be a parameter of the function."
                )
                raise ValueError(msg) from e
        if overlap := set(self.parameters) & set(at_least_tuple(self.output_name)):
            msg = (
                "The `output_name` cannot be the same as any of the input"
                f" parameter names. The overlap is: {overlap}"
            )
            raise ValueError(msg)
        if len(self._renames) != len(self._inverse_renames):
            inverse_renames = defaultdict(list)
            for k, v in self._renames.items():
                inverse_renames[v].append(k)
            violations = {k: v for k, v in inverse_renames.items() if len(v) > 1}
            violation_details = "; ".join(f"`{k}: {v}`" for k, v in violations.items())
            msg = (
                f"The `renames` should be a one-to-one mapping. Found violations where "
                f"multiple keys map to the same value: {violation_details}."
            )
            raise ValueError(msg)
        self._validate_update(
            self._renames,
            "renames",
            tuple(self.original_parameters) + at_least_tuple(self._output_name),  # type: ignore[arg-type]
        )
        self._validate_update(self._defaults, "defaults", self.parameters)
        self._validate_update(self._bound, "bound", self.parameters)
        for name in at_least_tuple(self.output_name):
            _validate_identifier("output_name", name)

    def copy(self, **update: Any) -> PipeFunc:
        """Create a copy of the `PipeFunc` instance, optionally updating the attributes."""
        kwargs = {
            "func": self.func,
            "output_name": self._output_name,
            "output_picker": self._output_picker,
            "renames": self._renames,
            "defaults": self._defaults,
            "bound": self._bound,
            "profile": self._profile,
            "debug": self.debug,
            "cache": self.cache,
            "mapspec": self.mapspec,
            "internal_shape": self.internal_shape,
            "post_execution_hook": self.post_execution_hook,
            "resources": self.resources,
            "resources_variable": self.resources_variable,
            "resources_scope": self.resources_scope,
            "variant": self.variant,
            "variant_group": None,  # deprecated
        }
        assert_complete_kwargs(kwargs, PipeFunc, skip={"self", "scope"})
        kwargs.update(update)
        return PipeFunc(**kwargs)  # type: ignore[arg-type,type-var]

    @functools.cached_property
    def _evaluate_lazy(self) -> bool:
        """Return whether the function should evaluate lazy arguments."""
        # This is a cached property because it is slow and otherwise called multiple times.
        # We assume that once it is set, it does not change during the lifetime of the object.
        return any(p.lazy for p in self._pipelines)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the wrapped function with the given arguments.

        Returns
        -------
            The return value of the wrapped function.

        """
        evaluated_resources = kwargs.pop(_EVALUATED_RESOURCES, None)
        kwargs = self._flatten_scopes(kwargs)
        if extra := set(kwargs) - set(self.parameters):
            msg = (
                f"Unexpected keyword arguments: `{extra}`."
                f" The allowed arguments are: `{self.parameters}`."
                f" The provided arguments are: `{kwargs}`."
            )
            raise ValueError(msg)

        if args:  # Put positional arguments into kwargs
            for p, v in zip(self.parameters, args):
                if p in kwargs:
                    msg = f"Multiple values provided for parameter `{p}`."
                    raise ValueError(msg)
                kwargs[p] = v
            args = ()
        kwargs = self.defaults | kwargs | self._bound
        kwargs = {self._inverse_renames.get(k, k): v for k, v in kwargs.items()}

        with self._maybe_profiler():
            if self._evaluate_lazy:
                args = evaluate_lazy(args)
                kwargs = evaluate_lazy(kwargs)
            _maybe_update_kwargs_with_resources(
                kwargs,
                self.resources_variable,
                evaluated_resources,
                self.resources,
            )
            try:
                result = self.func(*args, **kwargs)
            except Exception as e:
                print(
                    f"An error occurred while calling the function `{self.__name__}`"
                    f" with the arguments `{args=}` and `{kwargs=}`.",
                )
                self.error_snapshot = ErrorSnapshot(self.func, e, args, kwargs)
                raise

        if self.debug:
            _default_debug_printer(self, result, kwargs)
        if self.post_execution_hook is not None:
            self.post_execution_hook(self, result, kwargs)
        return result

    @property
    def profile(self) -> bool:
        """Return whether profiling is enabled for the wrapped function."""
        return self._profile

    @profile.setter
    def profile(self, enable: bool) -> None:
        """Enable or disable profiling for the wrapped function."""
        self._profile = enable
        if enable:
            requires("psutil", reason="profile", extras="profiling")
            self.profiling_stats = ProfilingStats()
        else:
            self.profiling_stats = None

    @functools.cached_property
    def parameter_scopes(self) -> set[str]:
        """Return the scopes of the function parameters.

        These are constructed from the parameter names that contain a dot.
        So if the parameter is ``foo.bar``, the scope is ``foo``.
        """
        return {k.split(".", 1)[0] for k in self.parameters if "." in k}

    @functools.cached_property
    def unscoped_parameters(self) -> tuple[str, ...]:
        """Return the parameters with the scope stripped off."""
        return tuple(name.split(".", 1)[-1] for name in self.parameters)

    def _flatten_scopes(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Flatten the scopes of the function parameters.

        Flattens `{scope: {name: value}}` to `{f"{scope}.{name}": value}`.

        Examples
        --------
        >>> f_c(x={"a": 1, "b": 1})
        >>> f_c(**{"x.a": 1, "x.b": 1})
        >>> f_c(x=dict(a=1), **{"x.b": 1})

        """
        if not self.parameter_scopes:
            return kwargs

        requires_flattening = self.parameter_scopes & kwargs.keys()
        if not requires_flattening:
            return kwargs

        new_kwargs = {}
        for k, v in kwargs.items():
            if k in self.parameter_scopes:
                new_kwargs.update({f"{k}.{name}": value for name, value in v.items()})
            else:
                new_kwargs[k] = v
        return new_kwargs

    @functools.cached_property
    def parameter_annotations(self) -> dict[str, Any]:
        """Return the type annotations of the wrapped function's parameters."""
        func = self.func
        if not is_pydantic_base_model(func):
            if inspect.isclass(func):
                func = func.__init__
            elif not inspect.isfunction(func) and not is_classmethod(func):
                func = func.__call__  # type: ignore[operator]
        type_hints = safe_get_type_hints(func, include_extras=True)
        return {self.renames.get(k, k): v for k, v in type_hints.items() if k != "return"}

    @functools.cached_property
    def output_annotation(self) -> dict[str, Any]:
        """Return the type annotation of the wrapped function's output."""
        func = self.func
        if inspect.isclass(func) and isinstance(self.output_name, str):
            return {self.output_name: func}
        if not inspect.isfunction(func) and not is_classmethod(func):
            func = func.__call__  # type: ignore[operator]
        if self._output_picker is None:
            hint = safe_get_type_hints(func, include_extras=True).get("return", NoAnnotation)
        else:
            # We cannot determine the output type if a custom output picker
            # is used, however, if the output is a tuple and the _default_output_picker
            # is used, we can determine the output type.
            hint = NoAnnotation
        if not isinstance(self.output_name, tuple):
            return {self.output_name: hint}
        if get_origin(hint) is tuple:
            return dict(zip(self.output_name, get_args(hint)))
        return {name: NoAnnotation for name in self.output_name}

    @functools.cached_property
    def requires_mapping(self) -> bool:
        return self.mapspec is not None and bool(self.mapspec.inputs)

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

    def __str__(self) -> str:
        """Return a string representation of the PipeFunc instance.

        Returns
        -------
            A string representation of the PipeFunc instance.

        """
        outputs = ", ".join(at_least_tuple(self.output_name))
        return f"{self.__name__}(...) → {outputs}"

    def __repr__(self) -> str:
        """Return a string representation of the PipeFunc instance.

        Returns
        -------
            A string representation of the PipeFunc instance.

        """
        return f"PipeFunc({self.__name__})"

    def __getstate__(self) -> dict:
        """Prepare the state of the current object for pickling.

        The state includes all picklable instance variables.
        For non-picklable instance variable,  they are transformed
        into a picklable form or ignored.

        Returns
        -------
            A dictionary containing the picklable state of the object.

        """
        state = {
            k: v for k, v in self.__dict__.items() if k not in ("func", "_pipelines", "resources")
        }
        state["func"] = cloudpickle.dumps(self.func)
        state["resources"] = (
            cloudpickle.dumps(self.resources) if self.resources is not None else None
        )
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
        self._pipelines = weakref.WeakSet()
        self.func = cloudpickle.loads(self.func)
        self.resources = cloudpickle.loads(self.resources) if self.resources is not None else None

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

        if bound_inputs := self._bound.keys() & mapspec_input_names:
            msg = (
                f"The bound arguments cannot be part of the MapSpec input names."
                f" The violating bound arguments are: `{bound_inputs}`."
                " Because bound arguments might have the same name in different"
                " functions and the MapSpec input names are unique, the bound"
                " arguments cannot be part of the MapSpec input names."
            )
            # This *can* be implemented but requires a lot of work
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

    @functools.cached_property
    def _cache_id(self) -> str:
        """Return a unique identifier for the function used in cache keys."""
        name = "-".join(at_least_tuple(self.output_name))
        if hasattr(self.func, "__pipefunc_hash__"):
            pipefunc_hash = self.func.__pipefunc_hash__()
            return f"{name}-{pipefunc_hash}"
        return name


def pipefunc(
    output_name: OUTPUT_TYPE,
    *,
    output_picker: Callable[[Any, str], Any] | None = None,
    renames: dict[str, str] | None = None,
    defaults: dict[str, Any] | None = None,
    bound: dict[str, Any] | None = None,
    profile: bool = False,
    debug: bool = False,
    cache: bool = False,
    mapspec: str | MapSpec | None = None,
    internal_shape: int | Literal["?"] | ShapeTuple | None = None,
    post_execution_hook: Callable[[PipeFunc, Any, dict[str, Any]], None] | None = None,
    resources: dict
    | Resources
    | Callable[[dict[str, Any]], Resources | dict[str, Any]]
    | None = None,
    resources_variable: str | None = None,
    resources_scope: Literal["map", "element"] = "map",
    scope: str | None = None,
    variant: str | dict[str | None, str] | None = None,
    variant_group: str | None = None,  # deprecated
) -> Callable[[Callable[..., Any]], PipeFunc]:
    """A decorator that wraps a function in a PipeFunc instance.

    Parameters
    ----------
    output_name
        The identifier for the output of the decorated function.
        Provide a tuple of strings for multiple outputs.
    output_picker
        A function that takes the output of the wrapped function as first argument
        and the ``output_name`` (str) as second argument, and returns the desired output.
        If ``None``, the output of the wrapped function is returned as is.
    renames
        A dictionary for renaming function arguments and outputs. The keys are the
        original names (as defined in the function signature or the ``output_name``),
        and the values are the new names to be used. This allows you to change how
        the function is called without modifying its internal logic. For example,
        ``{"old_name": "new_name"}`` would allow the function to be called with
        ``new_name`` instead of ``old_name``. If renaming the ``output_name``, include it
        in this dictionary as well.
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
    mapspec
        This is a specification for mapping that dictates how input values should
        be merged together. If ``None``, the default behavior is that the input directly
        maps to the output.
    internal_shape
        The shape of the output produced by this function *when it is used within a
        ``mapspec`` context*. Can be an int or a tuple of ints, or "?" for unknown
        dimensions, or a tuple with a mix of both. If not provided, the shape will be
        inferred from the first execution of the function. If provided, the shape will be
        validated against the actual shape of the output. This parameters is required only
        when a `mapspec` like `... -> out[i]` is used, indicating that the shape cannot be
        derived from the inputs. In case there are multiple outputs, provide the shape for
        one of the outputs. This works because the shape of all outputs are required to be
        identical.
    post_execution_hook
        A callback function that is invoked after the function is executed.
        The callback signature is ``hook(func: PipeFunc, result: Any, kwargs: dict) -> None``.
        This hook can be used for logging, visualization of intermediate results,
        debugging, statistics collection, or other side effects. The hook is executed
        synchronously after the function returns but before the result is passed to
        the next function in the pipeline. Keep the hook lightweight to avoid impacting performance.
    resources
        A dictionary or `Resources` instance containing the resources required
        for the function. This can be used to specify the number of CPUs, GPUs,
        memory, wall time, queue, partition, and any extra job scheduler
        arguments. This is *not* used by the `pipefunc` directly but can be
        used by job schedulers to manage the resources required for the
        function. Alternatively, provide a callable that receives a dict with the
        input values and returns a `Resources` instance.
    resources_variable
        If provided, the resources will be passed as the specified argument name to the function.
        This requires that the function has a parameter with the same name. For example,
        if ``resources_variable="resources"``, the function will be called as
        ``func(..., resources=Resources(...))``. This is useful when the function handles internal
        parallelization.
    resources_scope
        Determines how resources are allocated in relation to the mapspec:

        - "map": Allocate resources for the entire mapspec operation (default).
        - "element": Allocate resources for each element in the mapspec.

        If no mapspec is defined, this parameter is ignored.
    scope
        If provided, *all* parameter names and output names of the function will
        be prefixed with the specified scope followed by a dot (``'.'``), e.g., parameter
        ``x`` with scope ``foo`` becomes ``foo.x``. This allows multiple functions in a
        pipeline to have parameters with the same name without conflict. To be selective
        about which parameters and outputs to include in the scope, use the
        `PipeFunc.update_scope` method.

        When providing parameter values for functions that have scopes, they can
        be provided either as a dictionary for the scope, or by using the
        ``f'{scope}.{name}'`` notation. For example,
        a `PipeFunc` instance with scope "foo" and "bar", the parameters
        can be provided as: ``func(foo=dict(a=1, b=2), bar=dict(a=3, b=4))``
        or ``func(**{"foo.a": 1, "foo.b": 2, "bar.a": 3, "bar.b": 4})``.
    variant
        Identifies this function as an alternative implementation in a
        `VariantPipeline` and specifies which variant groups it belongs to.
        When multiple functions share the same `output_name`, variants allow
        selecting which implementation to use during pipeline execution.

        Can be specified in two formats:
        - A string (e.g., ``"fast"``): Places the function in the default unnamed
          group (None) with the specified variant name. Equivalent to ``{None: "fast"}``.
        - A dictionary (e.g., ``{"algorithm": "fast", "optimization": "level1"}``):
          Assigns the function to multiple variant groups simultaneously, with a
          specific variant name in each group.

        Functions with the same `output_name` but different variant specifications
        represent alternative implementations. The {meth}`VariantPipeline.with_variant`
        method selects which variants to use for execution. For example, you might
        have "preprocessing" variants ("v1"/"v2") independent from "computation"
        variants ("fast"/"accurate"), allowing you to select specific combinations
        like ``{"preprocessing": "v1", "computation": "fast"}``.
    variant_group
        DEPRECATED in v0.58.0: Use `variant` instead.

    Returns
    -------
        A wrapped function that takes the original function and ``output_name`` and
        creates a `PipeFunc` instance with the specified return identifier.

    See Also
    --------
    PipeFunc
        A function wrapper class for pipeline functions. Has the same functionality
        as the `pipefunc` decorator but can be used to wrap existing functions.

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
            mapspec=mapspec,
            internal_shape=internal_shape,
            post_execution_hook=post_execution_hook,
            resources=resources,
            resources_variable=resources_variable,
            resources_scope=resources_scope,
            variant=variant,
            variant_group=variant_group,  # deprecated
            scope=scope,
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
        constructed from all the output names of the `PipeFunc` instances. Must be a subset of
        the output names of the `PipeFunc` instances.
    function_name
        The name of the nested function, if ``None`` the name will be set
        to ``"NestedPipeFunc_{output_name[0]}_{output_name[...]}"``.
    mapspec
        `~pipefunc.map.MapSpec` for the joint function. If ``None``, the mapspec is inferred
        from the individual `PipeFunc` instances. None of the `MapsSpec` instances should
        have a reduction and all should use identical axes.
    resources
        Same as the `PipeFunc` class. However, if it is ``None`` here, it is inferred from
        from the `PipeFunc` instances. Specifically, it takes the maximum of the resources.
        Unlike the `PipeFunc` class, the `resources` argument cannot be a callable.
    bound
        Same as the `PipeFunc` class. Bind arguments to the functions. These are arguments
        that are fixed. Even when providing different values, the bound values will be
        used. Must be in terms of the renamed argument names.
    variant
        Same as the `PipeFunc` class.
        Identifies this function as an alternative implementation in a
        `VariantPipeline` and specifies which variant groups it belongs to.
        When multiple functions share the same `output_name`, variants allow
        selecting which implementation to use during pipeline execution.

        Can be specified in two formats:
        - A string (e.g., ``"fast"``): Places the function in the default unnamed
          group (None) with the specified variant name. Equivalent to ``{None: "fast"}``.
        - A dictionary (e.g., ``{"algorithm": "fast", "optimization": "level1"}``):
          Assigns the function to multiple variant groups simultaneously, with a
          specific variant name in each group.

        Functions with the same `output_name` but different variant specifications
        represent alternative implementations. The {meth}`VariantPipeline.with_variant`
        method selects which variants to use for execution. For example, you might
        have "preprocessing" variants ("v1"/"v2") independent from "computation"
        variants ("fast"/"accurate"), allowing you to select specific combinations
        like ``{"preprocessing": "v1", "computation": "fast"}``.
    variant_group
        DEPRECATED in v0.58.0: Use `variant` instead.

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
        pipefuncs: list[PipeFunc],
        output_name: OUTPUT_TYPE | None = None,
        function_name: str | None = None,
        *,
        renames: dict[str, str] | None = None,
        mapspec: str | MapSpec | None = None,
        resources: dict | Resources | None = None,
        bound: dict[str, Any] | None = None,
        variant: str | dict[str | None, str] | None = None,
        variant_group: str | None = None,  # deprecated
    ) -> None:
        from pipefunc import Pipeline

        self._pipelines: weakref.WeakSet[Pipeline] = weakref.WeakSet()
        _validate_nested_pipefunc(pipefuncs, resources)
        self.resources = _maybe_max_resources(resources, pipefuncs)
        functions = [f.copy(resources=self.resources) for f in pipefuncs]
        self.pipeline = Pipeline(functions)  # type: ignore[arg-type]
        _validate_single_leaf_node(self.pipeline.leaf_nodes)
        _validate_output_name(output_name, self._all_outputs)
        self._output_name: OUTPUT_TYPE = output_name or self._all_outputs
        self.function_name = function_name
        self.debug = False  # The underlying PipeFuncs will handle this
        self.cache = any(f.cache for f in self.pipeline.functions)
        _maybe_variant_group_error(variant_group, variant)
        self.variant: dict[str | None, str] = _ensure_variant(variant)
        self._output_picker = None
        self._profile = False
        self._renames: dict[str, str] = renames or {}
        self._defaults: dict[str, Any] = {
            k: v for k, v in self.pipeline.defaults.items() if k in self.parameters
        }
        self._bound: dict[str, Any] = bound or {}
        self.resources_variable = None  # not supported in NestedPipeFunc
        self.profiling_stats = None
        self.post_execution_hook = None
        self.internal_shape = None
        self.mapspec = self._combine_mapspecs() if mapspec is None else _maybe_mapspec(mapspec)
        for f in self.pipeline.functions:
            f.mapspec = None  # MapSpec is handled by the NestedPipeFunc
        self._validate()

    def copy(self, **update: Any) -> NestedPipeFunc:
        # Pass the mapspec to the new instance because we set
        # the child mapspecs to None in the __init__
        kwargs = {
            "pipefuncs": self.pipeline.functions,
            "output_name": self._output_name,
            "function_name": self.function_name,
            "renames": self._renames,
            "bound": self._bound,
            "mapspec": self.mapspec,
            "resources": self.resources,
            "variant": self.variant,
            "variant_group": None,  # deprecated
        }
        assert_complete_kwargs(kwargs, NestedPipeFunc, skip={"self"})
        kwargs.update(update)
        f = NestedPipeFunc(**kwargs)  # type: ignore[arg-type]
        f._defaults = self._defaults.copy()
        f._bound = self._bound.copy()
        return f

    def _combine_mapspecs(self) -> MapSpec | None:
        mapspecs = [f.mapspec for f in self.pipeline.functions]
        if all(m is None for m in mapspecs):
            return None
        _validate_combinable_mapspecs(mapspecs)
        axes = mapspec_axes(mapspecs)  # type: ignore[arg-type]
        return MapSpec(
            tuple(ArraySpec(n, axes[n]) for n in sorted(self.parameters) if n in axes),
            tuple(ArraySpec(n, axes[n]) for n in sorted(at_least_tuple(self.output_name))),
            _is_generated=True,
        )

    @functools.cached_property
    def original_parameters(self) -> dict[str, Any]:
        parameters = set(self._all_inputs) - set(self._all_outputs)
        return {
            k: inspect.Parameter(
                k,
                inspect.Parameter.KEYWORD_ONLY,
                # TODO: Do we need defaults here?
                # default=...,  # noqa: ERA001
            )
            for k in sorted(parameters)
        }

    @functools.cached_property
    def output_annotation(self) -> dict[str, Any]:
        return {
            name: self.pipeline[name].output_annotation[name]
            for name in at_least_tuple(self._output_name)
        }

    @functools.cached_property
    def parameter_annotations(self) -> dict[str, Any]:
        """Return the type annotations of the wrapped function's parameters."""
        annotations = self.pipeline.parameter_annotations
        return {p: annotations[p] for p in self.parameters if p in annotations}

    @functools.cached_property
    def _all_outputs(self) -> tuple[str, ...]:
        outputs: set[str] = set()
        for f in self.pipeline.functions:
            outputs.update(at_least_tuple(f.output_name))
        return tuple(sorted(outputs))

    @functools.cached_property
    def _all_inputs(self) -> tuple[str, ...]:
        inputs: set[str] = set()
        for f in self.pipeline.functions:
            parameters_excluding_bound = set(f.parameters) - set(f._bound)
            inputs.update(parameters_excluding_bound)
        return tuple(sorted(inputs))

    @functools.cached_property
    def func(self) -> Callable[..., tuple[Any, ...]]:  # type: ignore[override]
        func = self.pipeline.func(self.pipeline.unique_leaf_node.output_name)
        return _NestedFuncWrapper(func.call_full_output, self._output_name, self.function_name)

    @functools.cached_property
    def __name__(self) -> str:  # type: ignore[override]
        return self.func.__name__

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(pipefuncs={self.pipeline.functions})"


def _maybe_max_resources(
    resources: dict | Resources | None,
    pipefuncs: list[PipeFunc],
) -> Resources | None:
    if isinstance(resources, Resources) or callable(resources):
        return resources
    if isinstance(resources, dict):
        return Resources.from_dict(resources)
    resources_list = [f.resources for f in pipefuncs if f.resources is not None]
    assert not any(callable(f.resources) for f in pipefuncs)
    if len(resources_list) == 1:
        return resources_list[0]  # type: ignore[return-value]
    if not resources_list:
        return None
    return Resources.combine_max(resources_list)  # type: ignore[arg-type]


class _NestedFuncWrapper:
    """Wrapper class for nested functions.

    Takes a function that returns a dictionary and returns a tuple of values in the
    order specified by the output_name.
    """

    def __init__(
        self,
        func: Callable[..., dict[str, Any]],
        output_name: OUTPUT_TYPE,
        function_name: str | None = None,
    ) -> None:
        self.func: Callable[..., dict[str, Any]] = func
        self.output_name: OUTPUT_TYPE = output_name
        if function_name is not None:
            self.__name__ = function_name
        else:
            self.__name__ = f"NestedPipeFunc_{'_'.join(at_least_tuple(output_name))}"

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        result_dict = self.func(*args, **kwds)
        if isinstance(self.output_name, str):
            return result_dict[self.output_name]
        return tuple(result_dict[name] for name in self.output_name)


def _timestamp() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat()


@dataclass
class ErrorSnapshot:
    """A snapshot that represents an error in a function call."""

    function: Callable[..., Any]
    exception: Exception
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    traceback: str = field(init=False)
    timestamp: str = field(default_factory=_timestamp)
    user: str = field(default_factory=getpass.getuser)
    machine: str = field(default_factory=platform.node)
    ip_address: str = field(default_factory=get_local_ip)
    current_directory: str = field(default_factory=os.getcwd)

    def __post_init__(self) -> None:
        tb = traceback.format_exception(
            type(self.exception),
            self.exception,
            self.exception.__traceback__,
        )
        self.traceback = "".join(tb)

    def __str__(self) -> str:
        args_repr = ", ".join(repr(a) for a in self.args)
        kwargs_repr = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        func_name = f"{self.function.__module__}.{self.function.__qualname__}"

        return (
            "ErrorSnapshot:\n"
            "--------------\n"
            f"- 🛠 Function: {func_name}\n"
            f"- 🚨 Exception type: {type(self.exception).__name__}\n"
            f"- 💥 Exception message: {self.exception}\n"
            f"- 📋 Args: ({args_repr})\n"
            f"- 🗂 Kwargs: {{{kwargs_repr}}}\n"
            f"- 🕒 Timestamp: {self.timestamp}\n"
            f"- 👤 User: {self.user}\n"
            f"- 💻 Machine: {self.machine}\n"
            f"- 📡 IP Address: {self.ip_address}\n"
            f"- 📂 Current Directory: {self.current_directory}\n"
            "\n"
            "🔁 Reproduce the error by calling `error_snapshot.reproduce()`.\n"
            "📄 Or see the full stored traceback using `error_snapshot.traceback`.\n"
            "🔍 Inspect `error_snapshot.args` and `error_snapshot.kwargs`.\n"
            "💾 Or save the error to a file using `error_snapshot.save_to_file(filename)`"
            " and load it using `ErrorSnapshot.load_from_file(filename)`."
        )

    def reproduce(self) -> Any | None:
        """Attempt to recreate the error by calling the function with stored arguments."""
        return self.function(*self.args, **self.kwargs)

    def save_to_file(self, filename: str | Path) -> None:
        """Save the error snapshot to a file using cloudpickle."""
        with open(filename, "wb") as f:  # noqa: PTH123
            cloudpickle.dump(self, f)

    @classmethod
    def load_from_file(cls, filename: str | Path) -> ErrorSnapshot:
        """Load an error snapshot from a file using cloudpickle."""
        with open(filename, "rb") as f:  # noqa: PTH123
            return cloudpickle.load(f)

    def _ipython_display_(self) -> None:  # pragma: no cover
        from IPython.display import HTML, display

        display(HTML(f"<pre>{self}</pre>"))


def _validate_identifier(name: str, value: Any) -> None:
    if "." in value:
        scope, value = value.split(".", 1)
        _validate_identifier(name, scope)
        _validate_identifier(name, value)
        return
    if not value.isidentifier():
        msg = f"The `{name}` should contain/be valid Python identifier(s), not `{value}`."
        raise ValueError(msg)


def _validate_nested_pipefunc(
    pipefuncs: Sequence[PipeFunc],
    resources: dict | Resources | None,
) -> None:
    if not all(isinstance(f, PipeFunc) for f in pipefuncs):
        msg = "All elements in `pipefuncs` should be instances of `PipeFunc`."
        raise TypeError(msg)

    if len(pipefuncs) < 2:  # noqa: PLR2004
        msg = "The provided `pipefuncs` should have at least two `PipeFunc`s."
        raise ValueError(msg)

    if resources is None and any(callable(f.resources) for f in pipefuncs):
        msg = (
            "A `NestedPipeFunc` cannot have nested functions with callable `resources`."
            " Provide `NestedPipeFunc(..., resources=...)` instead."
        )
        raise ValueError(msg)

    if callable(resources):
        msg = (
            "`NestedPipeFunc` cannot have callable `resources`."
            " Provide a `Resources` instance instead or do not nest the `PipeFunc`s."
        )
        raise TypeError(msg)


def _validate_single_leaf_node(leaf_nodes: list[PipeFunc]) -> None:
    if len(leaf_nodes) > 1:
        msg = f"The provided `pipefuncs` should have only one leaf node, not {len(leaf_nodes)}."
        raise ValueError(msg)


def _validate_output_name(output_name: OUTPUT_TYPE | None, all_outputs: tuple[str, ...]) -> None:
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
            msg = (
                f"Cannot combine MapSpecs with different input and output mappings. Mapspec: `{m}`"
            )
            raise ValueError(msg)
        if m.input_indices != first.input_indices:
            msg = f"Cannot combine MapSpecs with different input mappings. Mapspec: `{m}`"
            raise ValueError(msg)
        if m.output_indices != first.output_indices:
            msg = f"Cannot combine MapSpecs with different output mappings. Mapspec: `{m}`"
            raise ValueError(msg)


def _default_output_picker(output: Any, name: str, output_name: OUTPUT_TYPE) -> Any:
    """Default output picker function for tuples."""
    return output[output_name.index(name)]


def _rename_output_name(
    original_output_name: OUTPUT_TYPE,
    renames: dict[str, str],
) -> OUTPUT_TYPE:
    if isinstance(original_output_name, str):
        return renames.get(original_output_name, original_output_name)
    return tuple(renames.get(name, name) for name in original_output_name)


def _prepend_name_with_scope(name: str, scope: str | None) -> str:
    if scope is None:
        return name.split(".", 1)[1] if "." in name else name
    if name.startswith(f"{scope}."):
        return name
    if "." in name:
        old_scope, name = name.split(".", 1)
        warnings.warn(
            f"Parameter '{name}' already has a scope '{old_scope}', replacing it with '{name}'.",
            stacklevel=3,
        )
    return f"{scope}.{name}"


def _maybe_mapspec(mapspec: str | MapSpec | None) -> MapSpec | None:
    """Return either a MapSpec or None, depending on the input."""
    return MapSpec.from_string(mapspec) if isinstance(mapspec, str) else mapspec


def _maybe_update_kwargs_with_resources(
    kwargs: dict[str, Any],
    resources_variable: str | None,
    evaluated_resources: Resources | None,
    resources: Resources | Callable[[dict[str, Any]], Resources] | None,
) -> None:
    if resources_variable:
        if evaluated_resources is not None:
            kwargs[resources_variable] = evaluated_resources
        elif callable(resources):
            kwargs[resources_variable] = resources(kwargs)
        else:
            kwargs[resources_variable] = resources


def _default_debug_printer(func: PipeFunc, result: Any, kwargs: dict[str, Any]) -> None:
    func_str = format_function_call(func.__name__, (), kwargs)
    now = datetime.datetime.now()  # noqa: DTZ005
    msg = (
        f"{now} - Function returning '{func.output_name}' was invoked"
        f" as `{func_str}` and returned `{result}`."
    )
    if func.profiling_stats is not None:
        dt = func.profiling_stats.time.average
        msg += f" The execution time was {dt:.2e} seconds on average."
    print(msg)


def _get_name(func: Callable[..., Any]) -> str:
    if isinstance(func, PipeFunc):
        return _get_name(func.func)
    if inspect.ismethod(func):
        qualname = func.__qualname__
        if "." in qualname:
            *_, class_name, method_name = qualname.split(".")
            return f"{class_name}.{method_name}"
        return qualname  # pragma: no cover
    if inspect.isfunction(func) or hasattr(func, "__name__"):
        return func.__name__
    return type(func).__name__


def _pydantic_defaults(
    func: type[pydantic.BaseModel],
    renames: dict[str, Any],
    defaults: dict[str, Any],
) -> dict[str, Any]:
    import pydantic

    defaults = defaults.copy()  # Make a copy to avoid modifying the original
    if pydantic.__version__.split(".", 1)[0] == "1":  # pragma: no cover
        msg = "Pydantic version 1 defaults cannot be extracted."
        warnings.warn(msg, UserWarning, stacklevel=2)
        return {}
    from pydantic_core import PydanticUndefined

    for name, field_ in func.model_fields.items():
        new_name = renames.get(name, name)
        if new_name in defaults:
            defaults[new_name] = defaults[new_name]
        elif field_.default_factory is not None:
            defaults[new_name] = field_.default_factory()
        elif field_.default is not PydanticUndefined:
            defaults[new_name] = field_.default
    return defaults


def _ensure_variant(variant: str | dict[str | None, str] | None) -> dict[str | None, str]:
    """Ensure that the variant is in the correct format."""
    # Convert string variant to dict with None as group
    if isinstance(variant, str):
        return {None: variant}
    return variant or {}


def _maybe_variant_group_error(
    variant_group: str | None,
    variant: str | dict[str | None, str] | None,
) -> None:
    if variant_group is not None:  # TODO: remove in 2025-09
        msg = (
            "The `variant_group` parameter has been removed in v0.58.0."
            f" Use the `variant = {{{variant_group!r}: {variant!r}}}` parameter instead."
        )
        raise ValueError(msg)
