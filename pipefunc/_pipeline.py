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

import functools
import inspect
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias

import networkx as nx

from pipefunc._cache import DiskCache, HybridCache, LRUCache, SimpleCache
from pipefunc._pipefunc import NestedPipeFunc, PipeFunc, _maybe_mapspec
from pipefunc._profile import print_profiling_stats
from pipefunc._simplify import _func_node_colors, _identify_combinable_nodes, simplified_pipeline
from pipefunc._utils import (
    assert_complete_kwargs,
    at_least_tuple,
    clear_cached_properties,
    handle_error,
)
from pipefunc.exceptions import UnusedParametersError
from pipefunc.lazy import _LazyFunction, task_graph
from pipefunc.map._mapspec import (
    ArraySpec,
    MapSpec,
    mapspec_axes,
    mapspec_dimensions,
    validate_consistent_axes,
)
from pipefunc.map._run import run
from pipefunc.resources import Resources
from pipefunc.typing import Array, is_object_array_type, is_type_compatible

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from concurrent.futures import Executor
    from pathlib import Path

    import holoviews as hv

    from pipefunc._profile import ProfilingStats
    from pipefunc.map._run import Result


_OUTPUT_TYPE: TypeAlias = str | tuple[str, ...]
_CACHE_KEY_TYPE: TypeAlias = tuple[_OUTPUT_TYPE, tuple[tuple[str, Any], ...]]

_empty = inspect.Parameter.empty


class Pipeline:
    """Pipeline class for managing and executing a sequence of functions.

    Parameters
    ----------
    functions
        A list of functions that form the pipeline. Note that the functions
        are copied when added to the pipeline using `PipeFunc.copy`.
    lazy
        Flag indicating whether the pipeline should be lazy.
    debug
        Flag indicating whether debug information should be printed.
        If ``None``, the value of each PipeFunc's debug attribute is used.
    profile
        Flag indicating whether profiling information should be collected.
        If ``None``, the value of each PipeFunc's profile attribute is used.
    cache_type
        The type of cache to use.
    cache_kwargs
        Keyword arguments passed to the cache constructor.
    validate_type_annotations
        Flag indicating whether type validation should be performed. If ``True``,
        the type annotations of the functions are validated during the pipeline
        initialization. If ``False``, the type annotations are not validated.
    scope
        If provided, *all* parameter names and output names of the pipeline functions will
        be prefixed with the specified scope followed by a dot (``'.'``), e.g., parameter
        ``x`` with scope ``foo`` becomes ``foo.x``. This allows multiple functions in a
        pipeline to have parameters with the same name without conflict. To be selective
        about which parameters and outputs to include in the scope, use the
        `Pipeline.update_scope` method.

        When providing parameter values for pipelines that have scopes, they can
        be provided either as a dictionary for the scope, or by using the
        ``f'{scope}.{name}'`` notation. For example,
        a `Pipeline` instance with scope "foo" and "bar", the parameters
        can be provided as:
        ``pipeline(output_name, foo=dict(a=1, b=2), bar=dict(a=3, b=4))`` or
        ``pipeline(output_name, **{"foo.a": 1, "foo.b": 2, "bar.a": 3, "bar.b": 4})``.
    default_resources
        Default resources to use for the pipeline functions. If ``None``,
        the resources are not set. Either a dict or a `pipefunc.resources.Resources`
        instance can be provided. If provided, the resources in the `PipeFunc`
        instances are updated with the default resources.

    """

    def __init__(
        self,
        functions: list[PipeFunc | tuple[PipeFunc, str | MapSpec]],
        *,
        lazy: bool = False,
        debug: bool | None = None,
        profile: bool | None = None,
        cache_type: Literal["lru", "hybrid", "disk", "simple"] | None = None,
        cache_kwargs: dict[str, Any] | None = None,
        validate_type_annotations: bool = True,
        scope: str | None = None,
        default_resources: dict[str, Any] | Resources | None = None,
    ) -> None:
        """Pipeline class for managing and executing a sequence of functions."""
        self.functions: list[PipeFunc] = []
        self.lazy = lazy
        self._debug = debug
        self._profile = profile
        self._default_resources: Resources | None = Resources.maybe_from_dict(default_resources)  # type: ignore[assignment]
        self.validate_type_annotations = validate_type_annotations
        for f in functions:
            if isinstance(f, tuple):
                f, mapspec = f  # noqa: PLW2901
            else:
                mapspec = None
            self.add(f, mapspec=mapspec)
        self._cache_type = cache_type
        self._cache_kwargs = cache_kwargs
        if cache_type is None and any(f.cache for f in self.functions):
            cache_type = "lru"
        self.cache = _create_cache(cache_type, lazy, cache_kwargs)
        if scope is not None:
            self.update_scope(scope, "*", "*")

    @property
    def profile(self) -> bool | None:
        """Flag indicating whether profiling information should be collected."""
        return self._profile

    @profile.setter
    def profile(self, value: bool | None) -> None:
        """Set the profiling flag for the pipeline and all functions."""
        self._profile = value
        if value is not None:
            for f in self.functions:
                f.profile = value

    @property
    def debug(self) -> bool | None:
        """Flag indicating whether debug information should be printed."""
        return self._debug

    @debug.setter
    def debug(self, value: bool | None) -> None:
        """Set the debug flag for the pipeline and all functions."""
        self._debug = value
        if value is not None:
            for f in self.functions:
                f.debug = value

    def add(
        self,
        f: PipeFunc | Callable,
        mapspec: str | MapSpec | None = None,
    ) -> PipeFunc:
        """Add a function to the pipeline.

        Always creates a copy of the `PipeFunc` instance to avoid side effects.

        Parameters
        ----------
        f
            The function to add to the pipeline.
        profile
            Flag indicating whether profiling information should be collected.
        mapspec
            This is a specification for mapping that dictates how input values should
            be merged together. If ``None``, the default behavior is that the input directly
            maps to the output.

        """
        if isinstance(f, PipeFunc):
            resources = Resources.maybe_with_defaults(f.resources, self._default_resources)
            f: PipeFunc = f.copy(  # type: ignore[no-redef]
                resources=resources,
                mapspec=f.mapspec if mapspec is None else _maybe_mapspec(mapspec),
            )
        elif callable(f):
            f = PipeFunc(
                f,
                output_name=f.__name__,
                mapspec=mapspec,
                resources=self._default_resources,
            )
        else:
            msg = f"`f` must be a `PipeFunc` or callable, got {type(f)}"
            raise TypeError(msg)

        self.functions.append(f)
        f._pipelines.add(self)

        if self.profile is not None:
            f.profile = self.profile

        if self.debug is not None:
            f.debug = self.debug

        self._clear_internal_cache()  # reset cache
        self._validate()
        return f

    def drop(self, *, f: PipeFunc | None = None, output_name: _OUTPUT_TYPE | None = None) -> None:
        """Drop a function from the pipeline.

        Parameters
        ----------
        f
            The function to drop from the pipeline.
        output_name
            The name of the output to drop from the pipeline.

        """
        if (f is not None and output_name is not None) or (f is None and output_name is None):
            msg = "Either `f` or `output_name` should be provided."
            raise ValueError(msg)
        if f is not None:
            if f not in self.functions:
                msg = (
                    f"The function `{f}` is not in the pipeline."
                    " Remember that the `PipeFunc` instances are copied on `Pipeline` initialization."
                )
                if f.output_name in self.output_to_func:
                    msg += (
                        f" However, the function with the same output name `{f.output_name!r}` exists in the"
                        f" pipeline, you can access that function via `pipeline[{f.output_name!r}]`."
                    )
                raise ValueError(msg)
            self.functions.remove(f)
        elif output_name is not None:
            f = self.output_to_func[output_name]
            self.drop(f=f)
        self._clear_internal_cache()
        self._validate()

    def replace(self, new: PipeFunc, old: PipeFunc | None = None) -> None:
        """Replace a function in the pipeline with another function.

        Parameters
        ----------
        new
            The function to add to the pipeline.
        old
            The function to replace in the pipeline. If None, ``old`` is
            assumed to be the function with the same output name as ``new``.

        """
        if old is None:
            self.drop(output_name=new.output_name)
        else:
            self.drop(f=old)
        self.add(new)
        self._clear_internal_cache()
        self._validate()

    @functools.cached_property
    def output_to_func(self) -> dict[_OUTPUT_TYPE, PipeFunc]:
        """Return a mapping from output names to functions.

        The mapping includes functions with multiple outputs both as individual
        outputs and as tuples of outputs. For example, if a function has the
        output name ``("a", "b")``, the mapping will include both ``"a"``,
        ``"b"``, and ``("a", "b")`` as keys.

        See Also
        --------
        __getitem__
            Shortcut for accessing the function corresponding to a specific output name.

        """
        output_to_func: dict[_OUTPUT_TYPE, PipeFunc] = {}
        for f in self.functions:
            output_to_func[f.output_name] = f
            if isinstance(f.output_name, tuple):
                for name in f.output_name:
                    output_to_func[name] = f
        return output_to_func

    def __getitem__(self, output_name: _OUTPUT_TYPE) -> PipeFunc:
        """Return the function corresponding to a specific output name.

        See Also
        --------
        output_to_func
            The mapping from output names to functions.

        """
        if output_name not in self.output_to_func:
            available = list(self.output_to_func.keys())
            msg = f"No function with output name `{output_name!r}` in the pipeline, only `{available}`."
            raise KeyError(msg)
        return self.output_to_func[output_name]

    def __contains__(self, output_name: _OUTPUT_TYPE) -> bool:
        """Check if the pipeline contains a function with a specific output name."""
        return output_name in self.output_to_func

    @functools.cached_property
    def node_mapping(self) -> dict[_OUTPUT_TYPE, PipeFunc | str]:
        """Return a mapping from node names to nodes.

        Returns
        -------
            A mapping from node names to nodes.

        """
        mapping: dict[_OUTPUT_TYPE, PipeFunc | str] = {}
        for node in self.graph.nodes:
            if isinstance(node, PipeFunc):
                if isinstance(node.output_name, tuple):
                    for name in node.output_name:
                        mapping[name] = node
                mapping[node.output_name] = node
            elif isinstance(node, str):
                mapping[node] = node
            else:
                assert isinstance(node, _Bound | _Resources)
        return mapping

    @functools.cached_property
    def graph(self) -> nx.DiGraph:
        """Create a directed graph representing the pipeline.

        Returns
        -------
            A directed graph with nodes representing functions and edges
            representing dependencies between functions.

        """
        _check_consistent_defaults(self.functions, output_to_func=self.output_to_func)
        g = nx.DiGraph()
        for f in self.functions:
            g.add_node(f)
            assert f.parameters is not None
            for arg in f.parameters:
                if arg in self.output_to_func:  # is function output
                    if arg in f._bound:
                        bound = _Bound(arg, f.output_name)
                        g.add_edge(bound, f)
                    else:
                        edge = (self.output_to_func[arg], f)
                        if edge not in g.edges:
                            g.add_edge(*edge, arg=arg)
                        else:
                            # edge already exists because of multiple outputs
                            assert isinstance(edge[0].output_name, tuple)
                            current = g.edges[edge]["arg"]
                            g.edges[edge]["arg"] = (*at_least_tuple(current), arg)
                else:  # noqa: PLR5501
                    if arg in f._bound:
                        bound = _Bound(arg, f.output_name)
                        g.add_edge(bound, f)
                    else:
                        if arg not in g:
                            # Add the node only if it doesn't exist
                            g.add_node(arg)
                        g.add_edge(arg, f, arg=arg)
            if f.resources_variable is not None:
                g.add_edge(_Resources(f.resources_variable, f.output_name), f)
        return g

    def func(self, output_name: _OUTPUT_TYPE) -> _PipelineAsFunc:
        """Create a composed function that can be called with keyword arguments.

        Parameters
        ----------
        output_name
            The identifier for the return value of the composed function.

        Returns
        -------
            The composed function that can be called with keyword arguments.

        """
        if f := self._internal_cache.func.get(output_name):
            return f
        root_args = self.root_args(output_name)
        assert isinstance(root_args, tuple)
        f = _PipelineAsFunc(self, output_name, root_args=root_args)
        self._internal_cache.func[output_name] = f
        return f

    @functools.cached_property
    def _internal_cache(self) -> _PipelineInternalCache:
        return _PipelineInternalCache()

    def _clear_internal_cache(self) -> None:
        clear_cached_properties(self)

    def __call__(self, __output_name__: _OUTPUT_TYPE | None = None, /, **kwargs: Any) -> Any:
        """Call the pipeline for a specific return value.

        Parameters
        ----------
        __output_name__
            The identifier for the return value of the pipeline.
            Is None by default, in which case the unique leaf node is used.
            This parameter is positional-only and the strange name is used
            to avoid conflicts with the ``output_name`` argument that might be
            passed via ``kwargs``.
        kwargs
            Keyword arguments to be passed to the pipeline functions.

        Returns
        -------
            The return value of the pipeline.

        """
        if __output_name__ is None:
            __output_name__ = self.unique_leaf_node.output_name
        return self.run(__output_name__, kwargs=kwargs)

    def _get_func_args(
        self,
        func: PipeFunc,
        flat_scope_kwargs: dict[str, Any],
        all_results: dict[_OUTPUT_TYPE, Any],
        full_output: bool,  # noqa: FBT001
        used_parameters: set[str | None],
    ) -> dict[str, Any]:
        # Used in _run
        func_args = {}
        for arg in func.parameters:
            if arg in func._bound:
                value = func._bound[arg]
            elif arg in flat_scope_kwargs:
                value = flat_scope_kwargs[arg]
            elif arg in self.output_to_func:
                value = self._run(
                    output_name=arg,
                    flat_scope_kwargs=flat_scope_kwargs,
                    all_results=all_results,
                    full_output=full_output,
                    used_parameters=used_parameters,
                )
            elif arg in self.defaults:
                value = self.defaults[arg]
            else:
                msg = f"Missing value for argument `{arg}` in `{func}`."
                raise ValueError(msg)
            func_args[arg] = value
            used_parameters.add(arg)
        return func_args

    def _current_cache(self) -> LRUCache | HybridCache | DiskCache | SimpleCache | None:
        """Return the cache used by the pipeline."""
        if not isinstance(self.cache, SimpleCache) and (tg := task_graph()) is not None:
            return tg.cache
        return self.cache

    def _run(
        self,
        *,
        output_name: _OUTPUT_TYPE,
        flat_scope_kwargs: dict[str, Any],
        all_results: dict[_OUTPUT_TYPE, Any],
        full_output: bool,
        used_parameters: set[str | None],
    ) -> Any:
        if output_name in all_results:
            return all_results[output_name]
        func = self.output_to_func[output_name]
        assert func.parameters is not None

        cache = self._current_cache()
        use_cache = (func.cache and cache is not None) or task_graph() is not None
        root_args = self.root_args(output_name)
        result_from_cache = False
        if use_cache:
            assert cache is not None
            cache_key = _compute_cache_key(
                func.output_name,
                self._func_defaults(func) | flat_scope_kwargs | func._bound,
                root_args,
            )
            return_now, result_from_cache = _get_result_from_cache(
                func,
                cache,
                cache_key,
                output_name,
                all_results,
                full_output,
                used_parameters,
                self.lazy,
            )
            if return_now:
                return all_results[output_name]

        func_args = self._get_func_args(
            func,
            flat_scope_kwargs,
            all_results,
            full_output,
            used_parameters,
        )

        if result_from_cache:
            assert full_output
            return all_results[output_name]

        start_time = time.perf_counter()
        r = _execute_func(func, func_args, self.lazy)
        if use_cache and cache_key is not None:
            assert cache is not None
            _update_cache(cache, cache_key, r, start_time)
        _update_all_results(func, r, output_name, all_results, self.lazy)
        return all_results[output_name]

    def run(
        self,
        output_name: _OUTPUT_TYPE,
        *,
        full_output: bool = False,
        kwargs: dict[str, Any],
    ) -> Any:
        """Execute the pipeline for a specific return value.

        Parameters
        ----------
        output_name
            The identifier for the return value of the pipeline.
        full_output
            Whether to return the outputs of all function executions
            as a dictionary mapping function names to their return values.
        kwargs
            Keyword arguments to be passed to the pipeline functions.

        Returns
        -------
            The return value of the pipeline or a dictionary mapping function
            names to their return values if ``full_output`` is ``True``.

        """
        if p := self.mapspec_names & set(self.func_dependencies(output_name)):
            inputs = self.mapspec_names & set(self.root_args(output_name))
            msg = (
                f"Cannot execute pipeline to get `{output_name}` because `{p}`"
                f" (depends on `{inputs=}`) have `MapSpec`(s). Use `Pipeline.map` instead."
            )
            raise RuntimeError(msg)

        if output_name in kwargs:
            msg = f"The `output_name='{output_name}'` argument cannot be provided in `kwargs={kwargs}`."
            raise ValueError(msg)

        flat_scope_kwargs = self._flatten_scopes(kwargs)

        all_results: dict[_OUTPUT_TYPE, Any] = flat_scope_kwargs.copy()  # type: ignore[assignment]
        used_parameters: set[str | None] = set()

        self._run(
            output_name=output_name,
            flat_scope_kwargs=flat_scope_kwargs,
            all_results=all_results,
            full_output=full_output,
            used_parameters=used_parameters,
        )

        # if has None, result was from cache, so we don't know which parameters were used
        if None not in used_parameters and (
            unused := flat_scope_kwargs.keys() - set(used_parameters)
        ):
            unused_str = ", ".join(sorted(unused))
            msg = f"Unused keyword arguments: `{unused_str}`. {kwargs=}, {used_parameters=}"
            raise UnusedParametersError(msg)

        return all_results if full_output else all_results[output_name]

    def map(
        self,
        inputs: dict[str, Any],
        run_folder: str | Path | None = None,
        internal_shapes: dict[str, int | tuple[int, ...]] | None = None,
        *,
        output_names: set[_OUTPUT_TYPE] | None = None,
        parallel: bool = True,
        executor: Executor | None = None,
        storage: str = "file_array",
        persist_memory: bool = True,
        cleanup: bool = True,
        fixed_indices: dict[str, int | slice] | None = None,
        auto_subpipeline: bool = False,
    ) -> dict[str, Result]:
        """Run a pipeline with `MapSpec` functions for given ``inputs``.

        Parameters
        ----------
        inputs
            The inputs to the pipeline. The keys should be the names of the input
            parameters of the pipeline functions and the values should be the
            corresponding input data, these are either single values for functions without ``mapspec``
            or lists of values or `numpy.ndarray`s for functions with ``mapspec``.
        run_folder
            The folder to store the run information. If ``None``, a temporary folder
            is created.
        internal_shapes
            The shapes for intermediary outputs that cannot be inferred from the inputs.
            You will receive an exception if the shapes cannot be inferred and need to be provided.
        output_names
            The output(s) to calculate. If ``None``, the entire pipeline is run and all outputs are computed.
        parallel
            Whether to run the functions in parallel.
        executor
            The executor to use for parallel execution. If ``None``, a `ProcessPoolExecutor`
            is used. Only relevant if ``parallel=True``.
        storage
            The storage class to use for the file arrays. Can use any registered storage class.
        persist_memory
            Whether to write results to disk when memory based storage is used.
            Does not have any effect when file based storage is used.
            Can use any registered storage class. See `pipefunc.map.storage_registry`.
        cleanup
            Whether to clean up the ``run_folder`` before running the pipeline.
        fixed_indices
            A dictionary mapping axes names to indices that should be fixed for the run.
            If not provided, all indices are iterated over.
        auto_subpipeline
            If ``True``, a subpipeline is created with the specified ``inputs``, using
            `Pipeline.subpipeline`. This allows to provide intermediate results in the ``inputs`` instead
            of providing the root arguments. If ``False``, all root arguments must be provided,
            and an exception is raised if any are missing.

        """
        return run(
            self,
            inputs,
            run_folder,
            internal_shapes=internal_shapes,
            output_names=output_names,
            parallel=parallel,
            executor=executor,
            storage=storage,
            persist_memory=persist_memory,
            cleanup=cleanup,
            fixed_indices=fixed_indices,
            auto_subpipeline=auto_subpipeline,
        )

    def arg_combinations(self, output_name: _OUTPUT_TYPE) -> set[tuple[str, ...]]:
        """Return the arguments required to compute a specific output.

        Parameters
        ----------
        output_name
            The identifier for the return value of the pipeline.

        Returns
        -------
            A set of tuples containing possible argument combinations.
            The tuples are sorted in lexicographical order.

        """
        if r := self._internal_cache.arg_combinations.get(output_name):
            return r
        head = self.node_mapping[output_name]
        arg_set: set[tuple[str, ...]] = set()
        _compute_arg_mapping(self.graph, head, head, [], [], arg_set)  # type: ignore[arg-type]
        self._internal_cache.arg_combinations[output_name] = arg_set
        return arg_set

    def root_args(self, output_name: _OUTPUT_TYPE) -> tuple[str, ...]:
        """Return the root arguments required to compute a specific output."""
        if r := self._internal_cache.root_args.get(output_name):
            return r
        arg_combos = self.arg_combinations(output_name)
        root_args = next(
            args for args in arg_combos if all(isinstance(self.node_mapping[n], str) for n in args)
        )
        self._internal_cache.root_args[output_name] = root_args
        return root_args

    def func_dependencies(self, output_name: _OUTPUT_TYPE | PipeFunc) -> list[_OUTPUT_TYPE]:
        """Return the functions required to compute a specific output.

        See Also
        --------
        func_predecessors

        """
        return _traverse_graph(output_name, "predecessors", self.graph, self.node_mapping)

    def func_dependents(self, name: _OUTPUT_TYPE | PipeFunc) -> list[_OUTPUT_TYPE]:
        """Return the functions that depend on a specific input/output.

        See Also
        --------
        func_successors

        """
        return _traverse_graph(name, "successors", self.graph, self.node_mapping)

    @functools.cached_property
    def defaults(self) -> dict[str, Any]:
        return {
            arg: value
            for func in self.functions
            for arg, value in func.defaults.items()
            if arg not in func._bound and arg not in self.output_to_func
        }

    def _func_defaults(self, func: PipeFunc) -> dict[str, Any]:
        """Retrieve defaults for a function, including those set by other functions."""
        if r := self._internal_cache.func_defaults.get(func.output_name):
            return r
        defaults = func.defaults.copy()
        for arg in func.parameters:
            if arg in self.defaults:
                pipeline_default = self.defaults[arg]
                if arg in defaults:
                    assert defaults[arg] == pipeline_default
                    continue
                defaults[arg] = self.defaults[arg]
        self._internal_cache.func_defaults[func.output_name] = defaults
        return defaults

    def update_defaults(self, defaults: dict[str, Any], *, overwrite: bool = False) -> None:
        """Update defaults to the provided keyword arguments.

        Automatically traverses the pipeline graph to find all functions that
        that the defaults can be applied to.

        If `overwrite` is `False`, the new defaults will be added to the existing
        defaults. If `overwrite` is `True`, the existing defaults will be replaced
        with the new defaults.

        Parameters
        ----------
        defaults
            A dictionary of default values for the keyword arguments.
        overwrite
            Whether to overwrite the existing defaults. If ``False``, the new
            defaults will be added to the existing defaults.

        """
        unused = set(defaults.keys())
        for f in self.functions:
            update = {k: v for k, v in defaults.items() if k in f.parameters if k not in f.bound}
            unused -= set(update.keys())
            if overwrite or update:
                f.update_defaults(update, overwrite=overwrite)
        self._clear_internal_cache()
        if unused:
            unused_str = ", ".join(sorted(unused))
            msg = f"Unused keyword arguments: `{unused_str}`. These are not settable defaults."
            raise ValueError(msg)
        self._validate()

    def update_renames(
        self,
        renames: dict[str, str],
        *,
        update_from: Literal["current", "original"] = "current",
        overwrite: bool = False,
    ) -> None:
        """Update the renames for the pipeline.

        Automatically traverses the pipeline graph to find all functions that
        the renames can be applied to.

        Parameters
        ----------
        renames
            A dictionary mapping old parameter names to new parameter and output names.
        update_from
            Whether to update the renames from the current parameter names (`PipeFunc.parameters`)
            or from the original parameter names (`PipeFunc.original_parameters`).
        overwrite
            Whether to overwrite the existing renames. If ``False``, the new
            renames will be added to the existing renames.

        """
        unused = set(renames.keys())
        for f in self.functions:
            parameters = tuple(
                f.parameters + at_least_tuple(f.output_name)
                if update_from == "current"
                else tuple(f.original_parameters) + at_least_tuple(f._output_name),
            )
            update = {k: v for k, v in renames.items() if k in parameters}
            unused -= set(update.keys())
            f.update_renames(update, overwrite=overwrite, update_from=update_from)
        self._clear_internal_cache()
        if unused:
            unused_str = ", ".join(sorted(unused))
            msg = f"Unused keyword arguments: `{unused_str}`. These are not settable renames."
            raise ValueError(msg)
        self._validate()

    def update_scope(
        self,
        scope: str | None,
        inputs: set[str] | Literal["*"] | None = None,
        outputs: set[str] | Literal["*"] | None = None,
        exclude: set[str] | None = None,
    ) -> None:
        """Update the scope for the pipeline by adding (or removing) a prefix to the input and output names.

        This method updates the names of the specified inputs and outputs by adding the provided
        scope as a prefix. The scope is added to the names using the format ``f"{scope}.{name}"``.
        If an input or output name already starts with the scope prefix, it remains unchanged.
        If their is an existing scope, it is replaced with the new scope.

        ``inputs`` are the root arguments of the pipeline. Inputs to functions
        which are outputs of other functions are considered to be outputs.

        Internally, simply calls `PipeFunc.update_renames` with  ``renames={name: f"{scope}.{name}", ...}``.

        When providing parameter values for pipelines that have scopes, they can
        be provided either as a dictionary for the scope, or by using the
        ``f'{scope}.{name}'`` notation. For example,
        a `Pipeline` instance with scope "foo" and "bar", the parameters
        can be provided as:
        ``pipeline(output_name, foo=dict(a=1, b=2), bar=dict(a=3, b=4))`` or
        ``pipeline(output_name, **{"foo.a": 1, "foo.b": 2, "bar.a": 3, "bar.b": 4})``.

        Parameters
        ----------
        scope
            The scope to set for the inputs and outputs. If ``None``, the scope of inputs and outputs is removed.
        inputs
            Specific input names to include, or ``"*"`` to include all inputs.
            The inputs are *only* the root arguments of the pipeline.
            If ``None``, no inputs are included.
        outputs
            Specific output names to include, or ``"*"`` to include all outputs.
            If ``None``, no outputs are included.
        exclude
            Names to exclude from the scope. Both inputs and outputs can be excluded.
            Can be used with ``inputs`` or ``outputs`` being ``"*"`` to exclude specific names.

        Examples
        --------
        >>> pipeline.update_scope("my_scope", inputs="*", outputs="*")  # Add scope to all inputs and outputs
        >>> pipeline.update_scope("my_scope", "*", "*", exclude={"output1"}) # Add to all except "output1"
        >>> pipeline.update_scope("my_scope", inputs="*", outputs={"output2"})  # Add scope to all inputs and "output2"
        >>> pipeline.update_scope(None, inputs="*", outputs="*")  # Remove scope from all inputs and outputs

        """
        _validate_scopes(self.functions, scope)
        all_inputs = set(self.topological_generations.root_args)
        all_outputs = self.all_output_names
        if inputs == "*":
            inputs = all_inputs
        if outputs == "*":
            outputs = all_outputs
        if exclude is None:
            exclude = set()

        for f in self.functions:
            parameters = set(f.parameters)
            f_inputs = (
                (set(inputs) & parameters & all_inputs) - exclude
                if isinstance(inputs, set)
                else inputs
            )
            all_names = set(at_least_tuple(f.output_name)) | parameters
            f_outputs = (
                (set(outputs) & all_names & all_outputs) - exclude
                if isinstance(outputs, set)
                else outputs
            )
            if f_inputs or f_outputs:
                f.update_scope(scope, inputs=f_inputs, outputs=f_outputs, exclude=exclude)
        self._clear_internal_cache()
        self._validate()

    def _flatten_scopes(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        flat_scope_kwargs = kwargs
        for f in self.functions:
            flat_scope_kwargs = f._flatten_scopes(flat_scope_kwargs)
        return flat_scope_kwargs

    @functools.cached_property
    def all_arg_combinations(self) -> dict[_OUTPUT_TYPE, set[tuple[str, ...]]]:
        """Compute all possible argument mappings for the pipeline.

        Returns
        -------
            A dictionary mapping function names to sets of tuples containing
            possible argument combinations.

        """
        return {
            node.output_name: self.arg_combinations(node.output_name)
            for node in self.graph.nodes
            if isinstance(node, PipeFunc)
        }

    @functools.cached_property
    def all_root_args(self) -> dict[_OUTPUT_TYPE, tuple[str, ...]]:
        """Return the root arguments required to compute all outputs."""
        return {
            node.output_name: self.root_args(node.output_name)
            for node in self.graph.nodes
            if isinstance(node, PipeFunc)
        }

    @functools.cached_property
    def mapspec_names(self) -> set[str]:
        return {
            name
            for mapspec in self.mapspecs()
            for name in mapspec.input_names + mapspec.output_names
        }

    def mapspecs(self, *, ordered: bool = True) -> list[MapSpec]:
        """Return the MapSpecs for all functions in the pipeline."""
        functions = self.sorted_functions if ordered else self.functions
        return [f.mapspec for f in functions if f.mapspec]

    @functools.cached_property
    def mapspecs_as_strings(self) -> list[str]:
        """Return the MapSpecs for all functions in the pipeline as strings."""
        return [str(mapspec) for mapspec in self.mapspecs(ordered=True)]

    @functools.cached_property
    def mapspec_dimensions(self: Pipeline) -> dict[str, int]:
        """Return the number of dimensions for each array parameter in the pipeline."""
        return mapspec_dimensions(self.mapspecs())

    @functools.cached_property
    def mapspec_axes(self: Pipeline) -> dict[str, tuple[str, ...]]:
        """Return the axes for each array parameter in the pipeline."""
        return mapspec_axes(self.mapspecs())

    def _validate(self) -> None:
        """Validate the pipeline."""
        _validate_scopes(self.functions)
        _check_consistent_defaults(self.functions, output_to_func=self.output_to_func)
        self._validate_mapspec()
        if self.validate_type_annotations:
            _check_consistent_type_annotations(self.graph)

    def _validate_mapspec(self) -> None:
        """Validate the MapSpecs for all functions in the pipeline."""
        for f in self.functions:
            if f.mapspec and at_least_tuple(f.output_name) != f.mapspec.output_names:
                msg = (
                    f"The output_name of the function `{f}` does not match the output_names"
                    f" in the MapSpec: `{f.output_name}` != `{f.mapspec.output_names}`."
                )
                raise ValueError(msg)
        validate_consistent_axes(self.mapspecs(ordered=False))
        self._autogen_mapspec_axes()

    @functools.cached_property
    def unique_leaf_node(self) -> PipeFunc:
        """Return the unique leaf node of the pipeline graph."""
        leaf_nodes = self.leaf_nodes
        if len(leaf_nodes) != 1:  # pragma: no cover
            msg = (
                "The pipeline has multiple leaf nodes. Please specify the output_name"
                " argument to disambiguate.",
            )
            raise ValueError(msg)
        return leaf_nodes[0]

    @functools.cached_property
    def topological_generations(self) -> Generations:
        """Return the functions in the pipeline grouped by topological generation.

        This method uses `networkx.topological_generations` on the pipeline graph to group
        functions by their dependency order. The result includes:
        - Root arguments: Initial inputs to the pipeline.
        - Function generations: Subsequent groups of functions in topological order.

        Nullary functions (those without parameters) are handled specially to ensure
        they're included in the generations rather than treated as root arguments.
        """
        nullary_functions = [f for f in self.functions if not f.parameters]
        if nullary_functions:
            # Handle nullary functions by adding placeholder edges.
            # This ensures they're included in the generations rather than as root arguments.
            graph = self.graph.copy()
            for i, f in enumerate(nullary_functions):
                graph.add_edge(i, f)
        else:
            graph = self.graph

        generations = list(nx.topological_generations(graph))
        if not generations:
            return Generations([], [])

        root_args: list[str] = []
        function_lists: list[list[PipeFunc]] = []
        for i, generation in enumerate(generations):
            generation_functions: list[PipeFunc] = []
            for x in generation:
                if i == 0 and isinstance(x, str):
                    root_args.append(x)
                elif i == 0 and isinstance(x, _Bound | _Resources | int):
                    # Skip special first-generation nodes that aren't root arguments
                    pass
                else:
                    assert isinstance(x, PipeFunc)
                    generation_functions.append(x)
            if generation_functions:
                function_lists.append(generation_functions)

        return Generations(root_args, function_lists)

    @functools.cached_property
    def sorted_functions(self) -> list[PipeFunc]:
        """Return the functions in the pipeline in topological order."""
        return [f for gen in self.topological_generations.function_lists for f in gen]

    @functools.cached_property
    def all_output_names(self) -> set[str]:
        return {name for f in self.functions for name in at_least_tuple(f.output_name)}

    def _autogen_mapspec_axes(self) -> set[PipeFunc]:
        """Generate `MapSpec`s for functions that return arrays with ``internal_shapes``."""
        root_args = self.topological_generations.root_args
        mapspecs = self.mapspecs(ordered=False)
        non_root_inputs = _find_non_root_axes(mapspecs, root_args)
        output_names = {at_least_tuple(f.output_name) for f in self.functions}
        multi_output_mapping = {n: names for names in output_names for n in names if len(names) > 1}
        _replace_none_in_axes(mapspecs, non_root_inputs, multi_output_mapping)  # type: ignore[arg-type]
        return _create_missing_mapspecs(self.functions, non_root_inputs)  # type: ignore[arg-type]

    def add_mapspec_axis(self, *parameter: str, axis: str) -> None:
        """Add a new axis to ``parameter``'s `MapSpec`.

        Parameters
        ----------
        parameter
            The parameter to add an axis to.
        axis
            The axis to add to the `MapSpec` of all functions that depends on
            ``parameter``. Provide a new axis name to add a new axis or an
            existing axis name to zip the parameter with the existing axis.

        """
        self._autogen_mapspec_axes()
        for p in parameter:
            _add_mapspec_axis(p, dims={}, axis=axis, functions=self.sorted_functions)
        self._clear_internal_cache()
        self._validate()

    def _func_node_colors(
        self,
        *,
        conservatively_combine: bool = False,
        output_name: _OUTPUT_TYPE | None = None,
    ) -> list[str]:
        if output_name is None:
            output_name = self.unique_leaf_node.output_name
        combinable_nodes = _identify_combinable_nodes(
            self.output_to_func[output_name],
            self.graph,
            self.all_root_args,
            conservatively_combine=conservatively_combine,
        )
        return _func_node_colors(self.functions, combinable_nodes)

    def visualize(
        self,
        figsize: tuple[int, int] = (10, 10),
        filename: str | Path | None = None,
        *,
        color_combinable: bool = False,
        conservatively_combine: bool = False,
        output_name: _OUTPUT_TYPE | None = None,
    ) -> None:
        """Visualize the pipeline as a directed graph.

        Parameters
        ----------
        figsize
            The width and height of the figure in inches.
        filename
            The filename to save the figure to.
        color_combinable
            Whether to color combinable nodes differently.
        conservatively_combine
            Argument as passed to `Pipeline.simply_pipeline`.
        output_name
            Argument as passed to `Pipeline.simply_pipeline`.

        """
        from pipefunc._plotting import visualize

        if color_combinable:
            func_node_colors = self._func_node_colors(
                conservatively_combine=conservatively_combine,
                output_name=output_name,
            )
        else:
            func_node_colors = None
        visualize(
            self.graph,
            figsize=figsize,
            filename=filename,
            func_node_colors=func_node_colors,
        )

    def visualize_holoviews(self, *, show: bool = False) -> hv.Graph | None:
        """Visualize the pipeline as a directed graph using HoloViews.

        Parameters
        ----------
        show
            Whether to show the plot. Uses `bokeh.plotting.show(holoviews.render(plot))`.
            If ``False`` the `holoviews.Graph` object is returned.

        """
        from pipefunc._plotting import visualize_holoviews

        return visualize_holoviews(self.graph, show=show)

    def print_profiling_stats(self) -> None:
        """Display the resource usage report for each function in the pipeline."""
        if not self.profiling_stats:
            msg = "Profiling is not enabled."
            raise ValueError(msg)
        print_profiling_stats(self.profiling_stats)

    def simplified_pipeline(
        self,
        output_name: _OUTPUT_TYPE | None = None,
        *,
        conservatively_combine: bool = False,
    ) -> Pipeline:
        """Simplify pipeline with combined function nodes.

        Generate a simplified version of the pipeline where combinable function
        nodes have been merged into single function nodes.

        This method identifies combinable nodes in the pipeline's execution
        graph (i.e., functions that share the same root arguments) and merges
        them into single function nodes. This results in a simplified pipeline
        where each key function only depends on nodes that cannot be further
        combined.

        Parameters
        ----------
        output_name
            The name of the output from the pipeline function we are starting
            the simplification from. If ``None``, the unique tip of the pipeline
            graph is used (if there is one).
        conservatively_combine
            If True, only combine a function node with its predecessors if all
            of its predecessors have the same root arguments as the function
            node itself. If False, combine a function node with its predecessors
            if any of its predecessors have the same root arguments as the
            function node.

        Returns
        -------
            The simplified version of the pipeline.

        """
        if output_name is None:
            output_name = self.unique_leaf_node.output_name
        return simplified_pipeline(
            functions=self.functions,
            graph=self.graph,
            all_root_args=self.all_root_args,
            node_mapping=self.node_mapping,
            output_name=output_name,
            conservatively_combine=conservatively_combine,
        )

    @functools.cached_property
    def leaf_nodes(self) -> list[PipeFunc]:
        """Return the leaf nodes in the pipeline's execution graph."""
        return [node for node in self.graph.nodes() if self.graph.out_degree(node) == 0]

    @functools.cached_property
    def root_nodes(self) -> list[PipeFunc]:
        """Return the root nodes in the pipeline's execution graph."""
        return [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]

    @property
    def profiling_stats(self) -> dict[str, ProfilingStats]:
        """Return the profiling data for each function in the pipeline."""
        return {f.__name__: f.profiling_stats for f in self.functions if f.profiling_stats}

    def __str__(self) -> str:
        """Return a string representation of the pipeline."""
        pipeline_str = "Pipeline:\n"
        for node in self.graph.nodes:
            if isinstance(node, PipeFunc):
                fn = node
                input_args = self.all_arg_combinations[fn.output_name]
                pipeline_str += f"  {fn.output_name} = {fn.__name__}({', '.join(fn.parameters)})\n"
                pipeline_str += f"    Possible input arguments: {input_args}\n"
        return pipeline_str

    def copy(self, **update: Any) -> Pipeline:
        """Return a copy of the pipeline.

        Parameters
        ----------
        update
            Keyword arguments passed to the `Pipeline` constructor instead of the
            original values.

        """
        kwargs = {
            "functions": self.functions,
            "lazy": self.lazy,
            "debug": self._debug,
            "profile": self._profile,
            "cache_type": self._cache_type,
            "cache_kwargs": self._cache_kwargs,
            "default_resources": self._default_resources,
            "validate_type_annotations": self.validate_type_annotations,
        }
        assert_complete_kwargs(kwargs, Pipeline.__init__, skip={"self", "scope"})
        kwargs.update(update)
        return Pipeline(**kwargs)  # type: ignore[arg-type]

    def nest_funcs(
        self,
        output_names: set[_OUTPUT_TYPE] | Literal["*"],
        new_output_name: _OUTPUT_TYPE | None = None,
    ) -> NestedPipeFunc:
        """Replaces a set of output names with a single nested function inplace.

        Parameters
        ----------
        output_names
            The output names to nest in a `NestedPipeFunc`. Can also be ``"*"`` to nest all functions
            in the pipeline into a single `NestedPipeFunc`.
        new_output_name
            The identifier for the output of the wrapped function. If ``None``, it is automatically
            constructed from all the output names of the `PipeFunc` instances.

        Returns
        -------
            The newly added `NestedPipeFunc` instance.

        """
        if output_names == "*":
            funcs = self.functions.copy()
        else:
            funcs = [self.output_to_func[output_name] for output_name in output_names]

        for f in funcs:
            self.drop(f=f)
        nested_func = NestedPipeFunc(funcs, output_name=new_output_name)
        self.add(nested_func)
        return nested_func

    def join(self, *pipelines: Pipeline | PipeFunc) -> Pipeline:
        """Join multiple pipelines into a single new pipeline.

        The new pipeline has no `default_resources` set, instead, each function has a
        `Resources` attribute that is created via
        ``Resources.maybe_with_defaults(f.resources, pipeline.default_resources)``.

        Parameters
        ----------
        pipelines
            The pipelines to join. Can also be individual `PipeFunc` instances.

        Returns
        -------
            A new pipeline containing all functions from the original pipelines.

        """
        functions = []
        for pipeline in [self, *pipelines]:
            if isinstance(pipeline, Pipeline):
                for f in pipeline.functions:
                    f_new = f.copy(resources=f.resources)
                    functions.append(f_new)
            elif isinstance(pipeline, PipeFunc):
                functions.append(pipeline.copy())
            else:
                msg = "Only `Pipeline` or `PipeFunc` instances can be joined."
                raise TypeError(msg)

        return self.copy(functions=functions, default_resources=None)

    def __or__(self, other: Pipeline | PipeFunc) -> Pipeline:
        """Combine two pipelines using the ``|`` operator.

        See Also
        --------
        join
            The method that is called when using the ``|`` operator.

        Examples
        --------
        >>> pipeline1 = Pipeline([f1, f2])
        >>> pipeline2 = Pipeline([f3, f4])
        >>> combined_pipeline = pipeline1 | pipeline2

        """
        return self.join(other)

    def _connected_components(self) -> list[set[PipeFunc | str]]:
        """Return the connected components of the pipeline graph."""
        return list(nx.connected_components(self.graph.to_undirected()))

    def split_disconnected(self: Pipeline, **pipeline_kwargs: Any) -> tuple[Pipeline, ...]:
        """Split disconnected components of the pipeline into separate pipelines.

        Parameters
        ----------
        pipeline_kwargs
            Keyword arguments to pass to the `Pipeline` constructor.

        Returns
        -------
            Tuple of fully connected `Pipeline` objects.

        """
        connected_components = self._connected_components()
        pipefunc_lists = [
            [x.copy() for x in xs if isinstance(x, PipeFunc)] for xs in connected_components
        ]
        if len(pipefunc_lists) == 1:
            msg = "Pipeline is fully connected, no need to split."
            raise ValueError(msg)
        return tuple(Pipeline(pfs, **pipeline_kwargs) for pfs in pipefunc_lists)  # type: ignore[arg-type]

    def _axis_in_root_arg(
        self,
        axis: str,
        output_name: _OUTPUT_TYPE,
        root_args: tuple[str, ...] | None = None,
        visited: set[_OUTPUT_TYPE] | None = None,
        result: set[bool] | None = None,
    ) -> bool:
        if root_args is None:
            root_args = self.root_args(output_name)
        if visited is None:
            visited = set()
        if result is None:
            result = set()
        if output_name in visited:
            return None  # type: ignore[return-value]

        visited.add(output_name)
        visited.update(at_least_tuple(output_name))

        func = self.output_to_func[output_name]
        assert func.mapspec is not None
        if axis not in func.mapspec.output_indices:  # pragma: no cover
            msg = f"Axis `{axis}` not in output indices for `{output_name=}`"
            raise ValueError(msg)

        if axis not in func.mapspec.input_indices:
            # Axis was in output but not in input
            result.add(False)  # noqa: FBT003

        axes = self.mapspec_axes
        for name in func.mapspec.input_names:
            if axis not in axes[name]:
                continue
            if name in root_args:
                if axis in axes[name]:
                    result.add(True)  # noqa: FBT003
            else:
                self._axis_in_root_arg(axis, name, root_args, visited, result)

        return all(result)

    def independent_axes_in_mapspecs(self, output_name: _OUTPUT_TYPE) -> set[str]:
        """Return the axes that are both in the output and in the root arguments.

        Identifies axes that are cross-products and can be computed independently.
        """
        func = self.output_to_func[output_name]
        if func.mapspec is None:
            return set()
        return {
            axis
            for axis in func.mapspec.output_indices
            if self._axis_in_root_arg(axis, output_name)
        }

    def subpipeline(
        self,
        inputs: set[str] | None = None,
        output_names: set[_OUTPUT_TYPE] | None = None,
    ) -> Pipeline:
        """Create a new pipeline containing only the nodes between the specified inputs and outputs.

        Parameters
        ----------
        inputs
            Set of input names to include in the subpipeline. If ``None``, all root nodes of the
            original pipeline will be used as inputs.
        output_names
            Set of output names to include in the subpipeline. If ``None``, all leaf nodes of the
            original pipeline will be used as outputs.

        Returns
        -------
            A new pipeline containing only the nodes and connections between the specified
            inputs and outputs.

        Notes
        -----
        The subpipeline is created by copying the original pipeline and then removing the nodes
        that are not part of the path from the specified inputs to the specified outputs. The
        resulting subpipeline will have the same behavior as the original pipeline for the
        selected inputs and outputs.

        If ``inputs`` is provided, the subpipeline will use those nodes as the new root nodes. If
        ``output_names`` is provided, the subpipeline will use those nodes as the new leaf nodes.

        Examples
        --------
        >>> @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
        ... def f(x: int) -> int:
        ...     return x
        ...
        >>> @pipefunc(output_name="z")
        ... def g(y: np.ndarray) -> int:
        ...     return sum(y)
        ...
        >>> pipeline = Pipeline([f, g])
        >>> inputs = {"x": [1, 2, 3]}
        >>> results = pipeline.map(inputs, "tmp_path")
        >>> partial = pipeline.subpipeline({"y"})
        >>> r = partial.map({"y": results["y"].output}, "tmp_path")
        >>> assert len(r) == 1
        >>> assert r["z"].output == 6

        """
        if inputs is None and output_names is None:
            msg = "At least one of `inputs` or `output_names` should be provided."
            raise ValueError(msg)

        pipeline = self.copy()

        input_nodes: set[str | PipeFunc] = (
            set(pipeline.topological_generations.root_args)
            if inputs is None
            else {pipeline.node_mapping[n] for n in inputs}
        )
        output_nodes: set[PipeFunc] = (
            set(pipeline.leaf_nodes)
            if output_names is None
            else {pipeline.node_mapping[n] for n in output_names}  # type: ignore[misc]
        )
        between = _find_nodes_between(pipeline.graph, input_nodes, output_nodes)
        drop = [f for f in pipeline.functions if f not in between]
        for f in drop:
            pipeline.drop(f=f)

        if inputs is not None:
            new_root_args = set(pipeline.topological_generations.root_args)
            if not new_root_args.issubset(inputs):
                outputs = {f.output_name for f in pipeline.functions}
                msg = (
                    f"Cannot construct a partial pipeline with `{outputs=}`"
                    f" and `{inputs=}`, it would require `{new_root_args}`."
                )
                raise ValueError(msg)

        return pipeline


class Generations(NamedTuple):
    root_args: list[str]
    function_lists: list[list[PipeFunc]]


@dataclass(frozen=True, slots=True, eq=True)
class _Bound:
    name: str
    output_name: _OUTPUT_TYPE


@dataclass(frozen=True, slots=True, eq=True)
class _Resources:
    name: str
    output_name: _OUTPUT_TYPE


class _PipelineAsFunc:
    """Wrapper class for a pipeline function.

    Parameters
    ----------
    pipeline
        The pipeline to which the function belongs.
    output_name
        The identifier for the return value of the pipeline function.
    root_args
        The names of the pipeline function's root inputs.

    """

    __slots__ = ["pipeline", "output_name", "root_args", "_call_with_root_args"]

    def __init__(
        self,
        pipeline: Pipeline,
        output_name: _OUTPUT_TYPE,
        root_args: tuple[str, ...],
    ) -> None:
        """Initialize the function wrapper."""
        self.pipeline = pipeline
        self.output_name = output_name
        self.root_args = root_args
        self._call_with_root_args: Callable[..., Any] | None = None

    @property
    def call_with_root_args(self) -> Callable[..., Any]:
        if self._call_with_root_args is None:
            self._call_with_root_args = self._create_call_with_root_args_method()
        return self._call_with_root_args

    def __call__(self, **kwargs: Any) -> Any:
        """Call the pipeline function with the given arguments.

        Parameters
        ----------
        kwargs
            Keyword arguments to be passed to the pipeline function.

        Returns
        -------
            The return value of the pipeline function.

        """
        return self.pipeline.run(output_name=self.output_name, kwargs=kwargs)

    def call_full_output(self, **kwargs: Any) -> dict[str, Any]:
        """Call the pipeline function with the given arguments and return all outputs.

        Parameters
        ----------
        kwargs
            Keyword arguments to be passed to the pipeline function.

        Returns
        -------
            The return value of the pipeline function.

        """
        return self.pipeline.run(self.output_name, full_output=True, kwargs=kwargs)

    def call_with_dict(self, kwargs: dict[str, Any]) -> Any:
        """Call the pipeline function with the given arguments.

        Parameters
        ----------
        kwargs
            Keyword arguments to be passed to the pipeline function.

        Returns
        -------
            The return value of the pipeline function.

        """
        return self(**kwargs)

    def __getstate__(self) -> dict:
        """Prepare the state of the current object for pickling."""
        state = {slot: getattr(self, slot) for slot in self.__slots__}
        state["_call_with_root_args"] = None  # don't pickle the execute method
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore the state of the current object from the provided state."""
        for slot in self.__slots__:
            setattr(self, slot, state[slot])
        # Initialize _call_with_root_args if necessary
        self._call_with_root_args = None

    def _create_call_with_parameters_method(
        self,
        parameters: tuple[str, ...],
    ) -> Callable[..., Any]:
        sig = inspect.signature(self.__call__)
        new_params = [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD) for name in parameters
        ]
        new_sig = sig.replace(parameters=new_params)

        def call(*args: Any, **kwargs: Any) -> Any:
            """Call the pipeline function with the root arguments."""
            bound = new_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return self(**bound.arguments)

        call.__signature__ = new_sig  # type: ignore[attr-defined]
        return call

    def _create_call_with_root_args_method(self) -> Callable[..., Any]:
        return self._create_call_with_parameters_method(self.root_args)


def _update_all_results(
    func: PipeFunc,
    r: Any,
    output_name: _OUTPUT_TYPE,
    all_results: dict[_OUTPUT_TYPE, Any],
    lazy: bool,  # noqa: FBT001
) -> None:
    if isinstance(func.output_name, tuple) and not isinstance(output_name, tuple):
        # Function produces multiple outputs, but only one is requested
        assert func.output_picker is not None
        for name in func.output_name:
            all_results[name] = (
                _LazyFunction(func.output_picker, args=(r, name))
                if lazy
                else func.output_picker(r, name)
            )
    else:
        all_results[func.output_name] = r


def _valid_key(key: Any) -> Any:
    if isinstance(key, dict):
        return tuple(sorted(key.items()))
    if isinstance(key, list):
        return tuple(key)
    if isinstance(key, set):
        return tuple(sorted(key))
    return key


def _update_cache(
    cache: LRUCache | HybridCache | DiskCache | SimpleCache,
    cache_key: _CACHE_KEY_TYPE,
    r: Any,
    start_time: float,
) -> None:
    # Used in _run
    if isinstance(cache, HybridCache):
        duration = time.perf_counter() - start_time
        cache.put(cache_key, r, duration)
    else:
        cache.put(cache_key, r)


def _get_result_from_cache(
    func: PipeFunc,
    cache: LRUCache | HybridCache | DiskCache | SimpleCache,
    cache_key: _CACHE_KEY_TYPE | None,
    output_name: _OUTPUT_TYPE,
    all_results: dict[_OUTPUT_TYPE, Any],
    full_output: bool,  # noqa: FBT001
    used_parameters: set[str | None],
    lazy: bool = False,  # noqa: FBT002, FBT001
) -> tuple[bool, bool]:
    # Used in _run
    result_from_cache = False
    if cache_key is not None and cache_key in cache:
        r = cache.get(cache_key)
        _update_all_results(func, r, output_name, all_results, lazy)
        result_from_cache = True
        if not full_output:
            used_parameters.add(None)  # indicate that the result was from cache
            return True, result_from_cache
    return False, result_from_cache


def _check_consistent_defaults(
    functions: list[PipeFunc],
    output_to_func: dict[_OUTPUT_TYPE, PipeFunc],
) -> None:
    """Check that the default values for shared arguments are consistent."""
    arg_defaults = {}
    for f in functions:
        for arg, default_value in f.defaults.items():
            if arg in f._bound or arg in output_to_func:
                continue
            if arg not in arg_defaults:
                arg_defaults[arg] = default_value
            elif default_value != arg_defaults[arg]:
                msg = (
                    f"Inconsistent default values for argument '{arg}' in"
                    " functions. Please make sure the shared input arguments have"
                    " the same default value or are set only for one function."
                )
                raise ValueError(msg)


def _create_cache(
    cache_type: Literal["lru", "hybrid", "disk", "simple"] | None,
    lazy: bool,  # noqa: FBT001
    cache_kwargs: dict[str, Any] | None,
) -> LRUCache | HybridCache | DiskCache | SimpleCache | None:
    if cache_type is None:
        return None
    if cache_kwargs is None:
        cache_kwargs = {}
    if cache_type == "lru":
        cache_kwargs.setdefault("shared", not lazy)
        return LRUCache(**cache_kwargs)
    if cache_type == "hybrid":
        if lazy:
            warnings.warn(
                "Hybrid cache uses function evaluation duration which"
                " is not measured correctly when using `lazy=True`.",
                UserWarning,
                stacklevel=2,
            )
        cache_kwargs.setdefault("shared", not lazy)
        return HybridCache(**cache_kwargs)
    if cache_type == "disk":
        cache_kwargs.setdefault("lru_shared", not lazy)
        return DiskCache(**cache_kwargs)
    if cache_type == "simple":
        return SimpleCache()

    msg = f"Invalid cache type: {cache_type}."
    raise ValueError(msg)


def _execute_func(func: PipeFunc, func_args: dict[str, Any], lazy: bool) -> Any:  # noqa: FBT001
    if lazy:
        return _LazyFunction(func, kwargs=func_args)
    try:
        return func(**func_args)
    except Exception as e:
        handle_error(e, func, func_args)
        # handle_error raises but mypy doesn't know that
        raise  # pragma: no cover


def _compute_cache_key(
    output_name: _OUTPUT_TYPE,
    kwargs: dict[str, Any],
    root_args: tuple[str, ...],
) -> _CACHE_KEY_TYPE | None:
    """Compute the cache key for a specific output name.

    The cache key is a tuple consisting of the output name and a tuple of
    root input keys and their corresponding values. Root inputs are the
    inputs that are not derived from any other function in the pipeline.

    If any of the root inputs required for the output_name are not available
    in kwargs, the cache key computation is skipped, and the method returns
    None. This can happen when a non-root input is directly provided as an
    input to another function, in which case the result should not be
    cached.

    Parameters
    ----------
    output_name
        The identifier for the return value of the pipeline.
    kwargs
        Keyword arguments to be passed to the pipeline functions.
    root_args
        The names of the pipeline function's root inputs.

    Returns
    -------
        A tuple containing the output name and a tuple of root input keys
        and their corresponding values, or None if the cache key computation
        is skipped.

    """
    cache_key_items = []
    for k in root_args:
        if k not in kwargs:
            # This means the computation was run with non-root inputs
            # i.e., the output of a function was directly provided as an input to
            # another function. In this case, we don't want to cache the result.
            return None
        key = _valid_key(kwargs[k])
        cache_key_items.append((k, key))

    return output_name, tuple(cache_key_items)


def _names(nodes: Iterable[PipeFunc | str]) -> tuple[str, ...]:
    names: list[str] = []
    for n in nodes:
        if isinstance(n, PipeFunc):
            names.extend(at_least_tuple(n.output_name))
        else:
            assert isinstance(n, str)
            names.append(n)
    return tuple(sorted(names))


def _sort_key(node: PipeFunc | str) -> str:
    if isinstance(node, PipeFunc):
        if isinstance(node.output_name, tuple):
            return ",".join(node.output_name)
        return node.output_name
    return node


def _unique(nodes: Iterable[PipeFunc | str]) -> tuple[PipeFunc | str, ...]:
    return tuple(sorted(set(nodes), key=_sort_key))


def _filter_funcs(funcs: Iterable[PipeFunc | str]) -> list[PipeFunc]:
    return [f for f in funcs if isinstance(f, PipeFunc)]


def _compute_arg_mapping(
    graph: nx.DiGraph,
    node: PipeFunc,
    head: PipeFunc,
    args: list[PipeFunc | str],
    replaced: list[PipeFunc | str],
    arg_set: set[tuple[str, ...]],
) -> None:
    preds = [
        n
        for n in graph.predecessors(node)
        if n not in replaced and not isinstance(n, _Bound | _Resources)
    ]
    deps = _unique(args + preds)
    deps_names = _names(deps)
    if deps_names in arg_set:
        return
    arg_set.add(deps_names)

    for func in _filter_funcs(deps):
        new_args = [dep for dep in deps if dep != func]
        _compute_arg_mapping(graph, func, head, new_args, [*replaced, node], arg_set)


def _axes_from_dims(p: str, dims: dict[str, int], axis: str) -> tuple[str | None, ...]:
    n = dims.get(p, 1) - 1
    return n * (None,) + (axis,)


def _add_mapspec_axis(p: str, dims: dict[str, int], axis: str, functions: list[PipeFunc]) -> None:
    # Modify the MapSpec of functions that depend on `p` to include the new axis
    for f in functions:
        if p not in f.parameters or p in f._bound:
            continue
        if f.mapspec is None:
            axes = _axes_from_dims(p, dims, axis)
            input_specs = [ArraySpec(p, axes)]
            output_specs = [ArraySpec(name, (axis,)) for name in at_least_tuple(f.output_name)]
        else:
            existing_inputs = set(f.mapspec.input_names)
            if p in existing_inputs:
                input_specs = [
                    s.add_axes(axis) if s.name == p and axis not in s.axes else s
                    for s in f.mapspec.inputs
                ]
            else:
                axes = _axes_from_dims(p, dims, axis)
                input_specs = [*f.mapspec.inputs, ArraySpec(p, axes)]
            output_specs = [
                s.add_axes(axis) if axis not in s.axes else s for s in f.mapspec.outputs
            ]
        f.mapspec = MapSpec(tuple(input_specs), tuple(output_specs), _is_generated=True)
        for o in output_specs:
            dims[o.name] = len(o.axes)
            _add_mapspec_axis(o.name, dims, axis, functions)


def _find_non_root_axes(
    mapspecs: list[MapSpec],
    root_args: list[str],
) -> dict[str, list[str | None]]:
    non_root_inputs: dict[str, list[str | None]] = {}
    for mapspec in mapspecs:
        for spec in mapspec.inputs:
            if spec.name not in root_args:
                if spec.name not in non_root_inputs:
                    non_root_inputs[spec.name] = spec.rank * [None]  # type: ignore[assignment]
                for i, axis in enumerate(spec.axes):
                    if axis is not None:
                        non_root_inputs[spec.name][i] = axis
    return non_root_inputs


def _replace_none_in_axes(
    mapspecs: list[MapSpec],
    non_root_inputs: dict[str, list[str]],
    multi_output_mapping: dict[str, tuple[str, ...]],
) -> None:
    all_axes_names = {
        axis.name for mapspec in mapspecs for axis in mapspec.inputs + mapspec.outputs
    }

    i = 0
    axis_template = "unnamed_{}"
    for name, axes in non_root_inputs.items():
        for j, axis in enumerate(axes):
            if axis is None:
                while (new_axis := axis_template.format(i)) in all_axes_names:
                    i += 1
                non_root_inputs[name][j] = new_axis
                all_axes_names.add(new_axis)
                if name in multi_output_mapping:
                    # If output is a tuple, update its axes with the new axis.
                    for output_name in multi_output_mapping[name]:
                        non_root_inputs[output_name][j] = new_axis
    assert not any(None in axes for axes in non_root_inputs.values())


def _create_missing_mapspecs(
    functions: list[PipeFunc],
    non_root_inputs: dict[str, set[str]],
) -> set[PipeFunc]:
    # Mapping from output_name to PipeFunc for functions without a MapSpec
    outputs_without_mapspec: dict[str, PipeFunc] = {
        name: func
        for func in functions
        if func.mapspec is None
        for name in at_least_tuple(func.output_name)
    }

    missing: set[str] = non_root_inputs.keys() & outputs_without_mapspec.keys()
    func_with_new_mapspecs = set()
    for p in missing:
        func = outputs_without_mapspec[p]
        if func in func_with_new_mapspecs:
            continue  # already added a MapSpec because of multiple outputs
        axes = tuple(non_root_inputs[p])
        outputs = tuple(ArraySpec(x, axes) for x in at_least_tuple(func.output_name))
        func.mapspec = MapSpec(inputs=(), outputs=outputs, _is_generated=True)
        func_with_new_mapspecs.add(func)
        print(f"Autogenerated MapSpec for `{func}`: `{func.mapspec}`")
    return func_with_new_mapspecs


def _traverse_graph(
    start: _OUTPUT_TYPE | PipeFunc,
    direction: Literal["predecessors", "successors"],
    graph: nx.DiGraph,
    node_mapping: dict[_OUTPUT_TYPE, PipeFunc | str],
) -> list[_OUTPUT_TYPE]:
    visited = set()

    def _traverse(x: _OUTPUT_TYPE | PipeFunc) -> list[_OUTPUT_TYPE]:
        results = set()
        if isinstance(x, str | tuple):
            x = node_mapping[x]
        for neighbor in getattr(graph, direction)(x):
            if isinstance(neighbor, PipeFunc):
                output_name = neighbor.output_name
                if output_name not in visited:
                    visited.add(output_name)
                    results.add(output_name)
                    results.update(_traverse(neighbor))
        return results  # type: ignore[return-value]

    return sorted(_traverse(start), key=at_least_tuple)


def _find_nodes_between(
    graph: nx.DiGraph,
    input_nodes: set[Any],
    output_nodes: set[Any],
) -> set[Any]:
    reachable_from_inputs = set()
    for input_node in input_nodes:
        reachable_from_inputs.update(nx.descendants(graph, input_node))
    reachable_to_outputs = set()
    for output_node in output_nodes:
        reachable_to_outputs.update(nx.ancestors(graph, output_node))
    reachable_to_outputs.update(output_nodes)
    return reachable_from_inputs & reachable_to_outputs


def _validate_scopes(functions: list[PipeFunc], new_scope: str | None = None) -> None:
    all_scopes = {scope for f in functions for scope in f.parameter_scopes}
    if new_scope is not None:
        all_scopes.add(new_scope)
    all_parameters = {p for f in functions for p in f.parameters + at_least_tuple(f.output_name)}
    if overlap := all_scopes & all_parameters:
        overlap_str = ", ".join(overlap)
        msg = f"Scope(s) `{overlap_str}` are used as both parameter and scope."
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class _PipelineInternalCache:
    arg_combinations: dict[_OUTPUT_TYPE, set[tuple[str, ...]]] = field(default_factory=dict)
    root_args: dict[_OUTPUT_TYPE, tuple[str, ...]] = field(default_factory=dict)
    func: dict[_OUTPUT_TYPE, _PipelineAsFunc] = field(default_factory=dict)
    func_defaults: dict[_OUTPUT_TYPE, dict[str, Any]] = field(default_factory=dict)


def _check_consistent_type_annotations(graph: nx.DiGraph) -> None:
    """Check that the type annotations for shared arguments are consistent."""
    for node in graph.nodes:
        if not isinstance(node, PipeFunc):
            continue
        deps = nx.descendants_at_distance(graph, node, 1)
        output_types = node.output_annotation
        for dep in deps:
            assert isinstance(dep, PipeFunc)
            for parameter_name, input_type in dep.parameter_annotations.items():
                if parameter_name not in output_types:
                    continue
                if _mapspec_is_generated(node, dep):
                    # NOTE: We cannot check the type-hints for auto-generated MapSpecs
                    continue
                if _mapspec_with_internal_shape(node, dep, parameter_name):
                    # NOTE: We cannot verify the type hints because the output
                    # might be any iterable instead of an Array as returned by
                    # a map operation.
                    continue
                output_type = output_types[parameter_name]
                if _axis_is_reduced(node, dep, parameter_name) and not is_object_array_type(
                    output_type,
                ):
                    output_type = Array[output_type]  # type: ignore[valid-type]
                if not is_type_compatible(output_type, input_type):
                    msg = (
                        f"Inconsistent type annotations for:"
                        f"\n  - Argument `{parameter_name}`"
                        f"\n  - Function `{node.__name__}(...)` returns:\n      `{output_type}`."
                        f"\n  - Function `{dep.__name__}(...)` expects:\n      `{input_type}`."
                        "\nPlease make sure the shared input arguments have the same type."
                        "\nNote that the output type displayed above might be wrapped in"
                        " `pipefunc.typing.Array` if using `MapSpec`s."
                        " Disable this check by setting `validate_type_annotations=False`."
                    )
                    raise TypeError(msg)


def _axis_is_reduced(f_out: PipeFunc, f_in: PipeFunc, parameter_name: str) -> bool:
    """Whether the output was the result of a map, and the input takes the entire result."""
    output_mapspec_names = f_out.mapspec.output_names if f_out.mapspec else ()
    input_mapspec_names = f_in.mapspec.input_names if f_in.mapspec else ()
    if f_in.mapspec:
        input_spec_axes = next(
            (s.axes for s in f_in.mapspec.inputs if s.name == parameter_name),
            None,
        )
    else:
        input_spec_axes = None
    return parameter_name in output_mapspec_names and (
        parameter_name not in input_mapspec_names
        or (input_spec_axes is not None and None in input_spec_axes)
    )


def _mapspec_is_generated(f_out: PipeFunc, f_in: PipeFunc) -> bool:
    if f_out.mapspec is None or f_in.mapspec is None:
        return False
    return f_out.mapspec._is_generated or f_in.mapspec._is_generated


def _mapspec_with_internal_shape(f_out: PipeFunc, f_in: PipeFunc, parameter_name: str) -> bool:
    """Whether the output was not from a map operation but returned an array with internal shape."""
    # NOTE: The only relevant case is where f_in uses elements of f_out as input
    # whereas, if f_in requires the entire output of f_out, it is not relevant.
    # This is all in the context of type annotations.
    if (
        f_out.mapspec is None
        or f_in.mapspec is None
        or parameter_name not in f_out.mapspec.output_names
        or parameter_name not in f_in.mapspec.input_names
    ):
        return False
    output_spec = next(s for s in f_out.mapspec.outputs if s.name == parameter_name)
    all_inputs_in_outputs = f_out.mapspec.input_indices.issuperset(output_spec.indices)
    return not all_inputs_in_outputs
