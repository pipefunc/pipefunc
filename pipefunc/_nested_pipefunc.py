from __future__ import annotations

import functools
import inspect
import weakref
from typing import TYPE_CHECKING, Any, Literal

from pipefunc._pipefunc import PipeFunc, _ensure_variant, _maybe_mapspec, _maybe_variant_group_error
from pipefunc._utils import assert_complete_kwargs, at_least_tuple
from pipefunc.map._mapspec import ArraySpec, MapSpec, mapspec_axes
from pipefunc.resources import Resources

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pipefunc import Pipeline
    from pipefunc._pipeline._types import OUTPUT_TYPE


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
    resources_scope
        Same as the `PipeFunc` class.
        Determines how resources are allocated in relation to the mapspec:

        - "map": Allocate resources for the entire mapspec operation (default).
        - "element": Allocate resources for each element in the mapspec.

        If no mapspec is defined, this parameter is ignored.
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
        resources_scope: Literal["map", "element"] = "map",
        bound: dict[str, Any] | None = None,
        variant: str | dict[str | None, str] | None = None,
        variant_group: str | None = None,  # deprecated
    ) -> None:
        from pipefunc import Pipeline

        self._pipelines: weakref.WeakSet[Pipeline] = weakref.WeakSet()
        _validate_nested_pipefunc(pipefuncs, resources)
        self.resources = _maybe_max_resources(resources, pipefuncs)
        self.resources_scope = resources_scope
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
            "resources_scope": self.resources_scope,
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
