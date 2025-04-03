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
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import networkx as nx

from pipefunc._pipefunc import ErrorSnapshot, NestedPipeFunc, PipeFunc, _maybe_mapspec
from pipefunc._profile import print_profiling_stats
from pipefunc._utils import (
    assert_complete_kwargs,
    at_least_tuple,
    clear_cached_properties,
    handle_error,
    is_installed,
    is_running_in_ipynb,
    requires,
)
from pipefunc.cache import DiskCache, HybridCache, LRUCache, SimpleCache
from pipefunc.exceptions import UnusedParametersError
from pipefunc.lazy import _LazyFunction, task_graph
from pipefunc.map._mapspec import (
    MapSpec,
    mapspec_axes,
    mapspec_dimensions,
    validate_consistent_axes,
)
from pipefunc.map._run import AsyncMap, run_map, run_map_async
from pipefunc.map._run_eager import run_map_eager
from pipefunc.map._run_eager_async import run_map_eager_async
from pipefunc.resources import Resources

from ._autodoc import PipelineDocumentation, format_pipeline_docs
from ._cache import compute_cache_key, create_cache, get_result_from_cache, update_cache
from ._cli import cli
from ._mapspec import (
    add_mapspec_axis,
    create_missing_mapspecs,
    find_non_root_axes,
    replace_none_in_axes,
)
from ._pydantic import pipeline_to_pydantic
from ._simplify import _func_node_colors, _identify_combinable_nodes, simplified_pipeline
from ._validation import (
    validate_consistent_defaults,
    validate_consistent_type_annotations,
    validate_scopes,
    validate_unique_output_names,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from concurrent.futures import Executor
    from pathlib import Path

    import graphviz
    import holoviews as hv
    import IPython.display
    import ipywidgets
    import pydantic
    from rich.table import Table

    from pipefunc._plotting import GraphvizStyle
    from pipefunc._profile import ProfilingStats
    from pipefunc.map._result import ResultDict
    from pipefunc.map._types import UserShapeDict

    from ._types import OUTPUT_TYPE, StorageType


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
        Profiling is only available for sequential execution.
    cache_type
        The type of cache to use. See the notes below for more *important* information.
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

    Notes
    -----
    Important note about caching: The caching behavior differs between ``pipeline.map`` and
    ``pipeline.run`` / ``pipeline(...)``.

    1. For ``pipeline.run`` and ``pipeline(...)`` ("calling the pipeline as a function"):

    - The cache key is computed based solely on the root arguments provided to the pipeline.
    - Only the root arguments need to be hashable.
    - The root arguments uniquely determine the output across the entire pipeline, allowing
      caching to be simple and effective when computing the final result.

    2. For ``pipeline.map``:

    - The cache key is computed based on the input values of each `PipeFunc`.
    - So a `PipeFunc` with ``cache=True`` must have hashable input values.
    - When using ``pipeline.map(..., parallel=True)``, the cache itself will be serialized,
      so one must use a cache that supports shared memory, such as `~pipefunc.cache.LRUCache`
      with ``shared=True`` or uses a disk cache like `~pipefunc.cache.DiskCache`.

    For both methods:

    - The `pipefunc.cache.to_hashable` function is used to attempt to ensure that input values are hashable,
      which is a requirement for storing results in a cache.
    - This function works for many common types but is not guaranteed to work for all types.
    - If `~pipefunc.cache.to_hashable` cannot make a value hashable, it falls back to using the serialized representation of the value.

    The key difference is that ``pipeline.run``'s output is uniquely determined by the root arguments,
    while ``pipeline.map`` is not because it may contain reduction operations as described by `~pipefunc.map.MapSpec`.

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
        self.cache = create_cache(cache_type, lazy, cache_kwargs)
        if scope is not None:
            self.update_scope(scope, "*", "*")

    def info(self, *, print_table: bool = False) -> dict[str, Any] | None:
        """Return information about inputs and outputs of the Pipeline.

        Parameters
        ----------
        print_table
            Whether to print a rich-formatted table to the console. Requires the `rich` package.

        Returns
        -------
        dict or None
            If `print_table` is False, returns a dictionary containing information about
            the inputs and outputs of the Pipeline, with the following keys:

            - ``inputs``: The input arguments of the Pipeline.
            - ``outputs``: The output arguments of the Pipeline.
            - ``intermediate_outputs``: The intermediate output arguments of the Pipeline.
            - ``required_inputs``: The required input arguments of the Pipeline.
            - ``optional_inputs``: The optional input arguments of the Pipeline (see `Pipeline.defaults`).

            If `print_table` is True, prints a rich-formatted table to the console and returns None.

        See Also
        --------
        defaults
            A dictionary with input name to default value mappings.
        leaf_nodes
            The leaf nodes of the pipeline as `PipeFunc` objects.
        root_args
            The root arguments (inputs) required to compute the output of the pipeline.
        print_documentation
            Print formatted documentation of the pipeline to the console.

        """
        inputs = self.root_args()
        outputs = tuple(sorted(n for f in self.leaf_nodes for n in at_least_tuple(f.output_name)))
        intermediate_outputs = tuple(sorted(self.all_output_names - set(outputs)))
        required_inputs = tuple(sorted(arg for arg in inputs if arg not in self.defaults))
        optional_inputs = tuple(sorted(arg for arg in inputs if arg in self.defaults))
        info = {
            "required_inputs": required_inputs,
            "optional_inputs": optional_inputs,
            "inputs": inputs,
            "intermediate_outputs": intermediate_outputs,
            "outputs": outputs,
        }
        if not print_table:
            return info
        _ = _rich_info_table(info, prints=True)
        return None

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

    def add(self, f: PipeFunc | Callable, mapspec: str | MapSpec | None = None) -> PipeFunc:
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

        validate_unique_output_names(f.output_name, self.output_to_func)
        self.functions.append(f)
        f._pipelines.add(self)

        if self.profile is not None:
            f.profile = self.profile

        if self.debug is not None:
            f.debug = self.debug

        self._clear_internal_cache()  # reset cache
        self.validate()
        return f

    def drop(self, *, f: PipeFunc | None = None, output_name: OUTPUT_TYPE | None = None) -> None:
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
        self.validate()

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
        self.validate()

    @functools.cached_property
    def output_to_func(self) -> dict[OUTPUT_TYPE, PipeFunc]:
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
        output_to_func: dict[OUTPUT_TYPE, PipeFunc] = {}
        for f in self.functions:
            output_to_func[f.output_name] = f
            if isinstance(f.output_name, tuple):
                for name in f.output_name:
                    output_to_func[name] = f
        return output_to_func

    def __getitem__(self, output_name: OUTPUT_TYPE) -> PipeFunc:
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

    def __contains__(self, output_name: OUTPUT_TYPE) -> bool:
        """Check if the pipeline contains a function with a specific output name."""
        return output_name in self.output_to_func

    @functools.cached_property
    def node_mapping(self) -> dict[OUTPUT_TYPE, PipeFunc | str]:
        """Return a mapping from node names to nodes.

        Returns
        -------
            A mapping from node names to nodes.

        """
        mapping: dict[OUTPUT_TYPE, PipeFunc | str] = {}
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
        validate_consistent_defaults(self.functions, output_to_func=self.output_to_func)
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

    def func(self, output_name: OUTPUT_TYPE | list[OUTPUT_TYPE]) -> _PipelineAsFunc:
        """Create a composed function that can be called with keyword arguments.

        Parameters
        ----------
        output_name
            The identifier for the return value of the composed function.

        Returns
        -------
            The composed function that can be called with keyword arguments.

        """
        key = tuple(output_name) if isinstance(output_name, list) else output_name
        if f := self._internal_cache.func.get(key):
            return f
        root_args = _root_args(self, output_name)
        assert isinstance(root_args, tuple)
        f = _PipelineAsFunc(self, output_name, root_args=root_args)
        self._internal_cache.func[key] = f
        return f

    @functools.cached_property
    def _internal_cache(self) -> _PipelineInternalCache:
        return _PipelineInternalCache()

    def _clear_internal_cache(self) -> None:
        clear_cached_properties(self)
        for f in self.functions:
            # `clear_pipelines=False` to avoid infinite recursion
            f._clear_internal_cache(clear_pipelines=False)

    def __call__(self, __output_name__: OUTPUT_TYPE | None = None, /, **kwargs: Any) -> Any:
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
        all_results: dict[OUTPUT_TYPE, Any],
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
        output_name: OUTPUT_TYPE,
        flat_scope_kwargs: dict[str, Any],
        all_results: dict[OUTPUT_TYPE, Any],
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
            cache_key = compute_cache_key(
                func._cache_id,
                self._func_defaults(func) | flat_scope_kwargs | func._bound,
                root_args,
            )
            return_now, result_from_cache = get_result_from_cache(
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
            update_cache(cache, cache_key, r, start_time)
        _update_all_results(func, r, output_name, all_results, self.lazy)
        return all_results[output_name]

    def _validate_run_output_name(
        self,
        kwargs: dict[str, Any],
        output_name: OUTPUT_TYPE | list[OUTPUT_TYPE],
    ) -> None:
        if isinstance(output_name, list):
            for name in output_name:
                self._validate_run_output_name(kwargs, name)
            return
        if output_name in kwargs:
            msg = f"The `output_name='{output_name}'` argument cannot be provided in `kwargs={kwargs}`."
            raise ValueError(msg)
        if output_name not in self.output_to_func:
            available = ", ".join(k for k in self.output_to_func if isinstance(k, str))
            msg = (
                f"No function with output name `{output_name}` in the pipeline, only `{available}`."
            )
            raise ValueError(msg)

        if p := self.mapspec_names & set(self.func_dependencies(output_name)):
            inputs = self.mapspec_names & set(self.root_args(output_name))
            msg = (
                f"Cannot execute pipeline to get `{output_name}` because `{p}`"
                f" (depends on `{inputs=}`) have `MapSpec`(s). Use `Pipeline.map` instead."
            )
            raise RuntimeError(msg)

    def run(
        self,
        output_name: OUTPUT_TYPE | list[OUTPUT_TYPE],
        *,
        full_output: bool = False,
        kwargs: dict[str, Any],
        allow_unused: bool = False,
    ) -> Any:
        """Execute the pipeline for a specific return value.

        Parameters
        ----------
        output_name
            The identifier for the return value of the pipeline. Can be a single
            output name or a list of output names.
        full_output
            Whether to return the outputs of all function executions
            as a dictionary mapping function names to their return values.
        kwargs
            Keyword arguments to be passed to the pipeline functions.
        allow_unused
            Whether to allow unused keyword arguments. If ``False``, an error
            is raised if any keyword arguments are unused. If ``True``, unused
            keyword arguments are ignored.

        Returns
        -------
            A dictionary mapping function names to their return values
            if ``full_output`` is ``True``. Otherwise, the return value is the
            return value of the pipeline function specified by ``output_name``.
            If ``output_name`` is a list, the return value is a tuple of the
            return values of the pipeline functions.

        """
        self._validate_run_output_name(kwargs, output_name)
        flat_scope_kwargs = self._flatten_scopes(kwargs)

        all_results: dict[OUTPUT_TYPE, Any] = flat_scope_kwargs.copy()  # type: ignore[assignment]
        used_parameters: set[str | None] = set()
        output_names = [output_name] if not isinstance(output_name, list) else output_name
        for _output_name in output_names:
            self._run(
                output_name=_output_name,
                flat_scope_kwargs=flat_scope_kwargs,
                all_results=all_results,
                full_output=full_output,
                used_parameters=used_parameters,
            )

        # if has None, result was from cache, so we don't know which parameters were used
        if (
            not allow_unused
            and None not in used_parameters
            and (unused := flat_scope_kwargs.keys() - set(used_parameters))
        ):
            unused_str = ", ".join(sorted(unused))
            msg = f"Unused keyword arguments: `{unused_str}`. {kwargs=}, {used_parameters=}"
            raise UnusedParametersError(msg)

        if full_output:
            return all_results
        if isinstance(output_name, list):
            return tuple(all_results[k] for k in output_name)
        return all_results[output_name]

    def map(
        self,
        inputs: dict[str, Any] | pydantic.BaseModel,
        run_folder: str | Path | None = None,
        internal_shapes: UserShapeDict | None = None,
        *,
        output_names: set[OUTPUT_TYPE] | None = None,
        parallel: bool = True,
        executor: Executor | dict[OUTPUT_TYPE, Executor] | None = None,
        chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int]] | None = None,
        storage: StorageType = "file_array",
        persist_memory: bool = True,
        cleanup: bool = True,
        fixed_indices: dict[str, int | slice] | None = None,
        auto_subpipeline: bool = False,
        show_progress: bool = False,
        return_results: bool = True,
        scheduling_strategy: Literal["generation", "eager"] = "generation",
    ) -> ResultDict:
        """Run a pipeline with `MapSpec` functions for given ``inputs``.

        Parameters
        ----------
        inputs
            The inputs to the pipeline. The keys should be the names of the input
            parameters of the pipeline functions and the values should be the
            corresponding input data, these are either single values for functions without ``mapspec``
            or lists of values or `numpy.ndarray`s for functions with ``mapspec``.
        run_folder
            The folder to store the run information. If ``None``, either a temporary folder
            is created or no folder is used, depending on whether the storage class requires serialization.
        internal_shapes
            The shapes for intermediary outputs that cannot be inferred from the inputs.
            If not provided, the shapes will be inferred from the first execution of the function.
            If provided, the shapes will be validated against the actual shapes of the outputs.
            The values can be either integers or "?" for unknown dimensions.
            The ``internal_shape`` can also be provided via the ``PipeFunc(..., internal_shape=...)`` argument.
            If a `PipeFunc` has an ``internal_shape`` argument *and* it is provided here, the provided value is used.
        output_names
            The output(s) to calculate. If ``None``, the entire pipeline is run and all outputs are computed.
        parallel
            Whether to run the functions in parallel. Is ignored if provided ``executor`` is not ``None``.
        executor
            The executor to use for parallel execution. Can be specified as:

            1. ``None``: A `concurrent.futures.ProcessPoolExecutor` is used (only if ``parallel=True``).
            2. A `concurrent.futures.Executor` instance: Used for all outputs.
            3. A dictionary: Specify different executors for different outputs.

               - Use output names as keys and `~concurrent.futures.Executor` instances as values.
               - Use an empty string ``""`` as a key to set a default executor.

            If parallel is ``False``, this argument is ignored.
        chunksizes
            Controls batching of `~pipefunc.map.MapSpec` computations for parallel execution.
            Reduces overhead by grouping multiple function calls into single tasks.
            Can be specified as:

            - None: Automatically determine optimal chunk sizes (default)
            - int: Same chunk size for all outputs
            - dict: Different chunk sizes per output where:
                - Keys are output names (or ``""`` for default)
                - Values are either integers or callables
                - Callables take total execution count and return chunk size

            **Examples:**

            >>> chunksizes = None  # Auto-determine optimal chunk sizes
            >>> chunksizes = 100  # All outputs use chunks of 100
            >>> chunksizes = {"out1": 50, "out2": 100}  # Different sizes per output
            >>> chunksizes = {"": 50, "out1": lambda n: n // 20}  # Default and dynamic
        storage
            The storage class to use for storing intermediate and final results.
            Can be specified as:

            1. A string: Use a single storage class for all outputs.
            2. A dictionary: Specify different storage classes for different outputs.

               - Use output names as keys and storage class names as values.
               - Use an empty string ``""`` as a key to set a default storage class.

            Available storage classes are registered in `pipefunc.map.storage_registry`.
            Common options include ``"file_array"``, ``"dict"``, and ``"shared_memory_dict"``.
        persist_memory
            Whether to write results to disk when memory based storage is used.
            Does not have any effect when file based storage is used.
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
        show_progress
            Whether to display a progress bar. Only works if ``parallel=True``.
        return_results
            Whether to return the results of the pipeline. If ``False``, the pipeline is run
            without keeping the results in memory. Instead the results are only kept in the set
            ``storage``. This is useful for very large pipelines where the results do not fit into memory.
        scheduling_strategy
            Strategy for scheduling pipeline function execution:

            - "generation" (default): Executes functions in strict topological generations,
              waiting for all functions in a generation to complete before starting the next.
              Provides predictable execution order but may not maximize parallelism.

            - "eager": Dynamically schedules functions as soon as their dependencies are met,
              without waiting for entire generations to complete. Can improve performance
              by maximizing parallel execution, especially for complex dependency graphs
              with varied execution times.

        See Also
        --------
        map_async
            The asynchronous version of this method.

        Returns
        -------
            A `ResultDict` containing the results of the pipeline. The values are of type `Result`,
            use `Result.output` to get the actual result.

        """
        if scheduling_strategy == "generation":
            run_map_func = run_map
        elif scheduling_strategy == "eager":
            run_map_func = run_map_eager
        else:  # pragma: no cover
            msg = f"Invalid scheduling type: {scheduling_strategy}"
            raise ValueError(msg)
        return run_map_func(
            self,
            inputs,
            run_folder,
            internal_shapes=internal_shapes,
            output_names=output_names,
            parallel=parallel,
            executor=executor,
            chunksizes=chunksizes,
            storage=storage,
            persist_memory=persist_memory,
            cleanup=cleanup,
            fixed_indices=fixed_indices,
            auto_subpipeline=auto_subpipeline,
            show_progress=show_progress,
            return_results=return_results,
        )

    def map_async(
        self,
        inputs: dict[str, Any] | pydantic.BaseModel,
        run_folder: str | Path | None = None,
        internal_shapes: UserShapeDict | None = None,
        *,
        output_names: set[OUTPUT_TYPE] | None = None,
        executor: Executor | dict[OUTPUT_TYPE, Executor] | None = None,
        chunksizes: int | dict[OUTPUT_TYPE, int | Callable[[int], int]] | None = None,
        storage: StorageType = "file_array",
        persist_memory: bool = True,
        cleanup: bool = True,
        fixed_indices: dict[str, int | slice] | None = None,
        auto_subpipeline: bool = False,
        show_progress: bool = False,
        return_results: bool = True,
        scheduling_strategy: Literal["generation", "eager"] = "generation",
    ) -> AsyncMap:
        """Asynchronously run a pipeline with `MapSpec` functions for given ``inputs``.

        Returns immediately with an `AsyncRun` instance with a `task` attribute that can be awaited.

        Parameters
        ----------
        inputs
            The inputs to the pipeline. The keys should be the names of the input
            parameters of the pipeline functions and the values should be the
            corresponding input data, these are either single values for functions without ``mapspec``
            or lists of values or `numpy.ndarray`s for functions with ``mapspec``.
        run_folder
            The folder to store the run information. If ``None``, either a temporary folder
            is created or no folder is used, depending on whether the storage class requires serialization.
        internal_shapes
            The shapes for intermediary outputs that cannot be inferred from the inputs.
            If not provided, the shapes will be inferred from the first execution of the function.
            If provided, the shapes will be validated against the actual shapes of the outputs.
            The values can be either integers or "?" for unknown dimensions.
            The ``internal_shape`` can also be provided via the ``PipeFunc(..., internal_shape=...)`` argument.
            If a `PipeFunc` has an ``internal_shape`` argument *and* it is provided here, the provided value is used.
        output_names
            The output(s) to calculate. If ``None``, the entire pipeline is run and all outputs are computed.
        executor
            The executor to use for parallel execution. Can be specified as:

            1. ``None``: A `concurrent.futures.ProcessPoolExecutor` is used (only if ``parallel=True``).
            2. A `concurrent.futures.Executor` instance: Used for all outputs.
            3. A dictionary: Specify different executors for different outputs.

               - Use output names as keys and `~concurrent.futures.Executor` instances as values.
               - Use an empty string ``""`` as a key to set a default executor.
        chunksizes
            Controls batching of `~pipefunc.map.MapSpec` computations for parallel execution.
            Reduces overhead by grouping multiple function calls into single tasks.
            Can be specified as:

            - None: Automatically determine optimal chunk sizes (default)
            - int: Same chunk size for all outputs
            - dict: Different chunk sizes per output where:
                - Keys are output names (or ``""`` for default)
                - Values are either integers or callables
                - Callables take total execution count and return chunk size

            **Examples:**

            >>> chunksizes = None  # Auto-determine optimal chunk sizes
            >>> chunksizes = 100  # All outputs use chunks of 100
            >>> chunksizes = {"out1": 50, "out2": 100}  # Different sizes per output
            >>> chunksizes = {"": 50, "out1": lambda n: n // 20}  # Default and dynamic
        storage
            The storage class to use for storing intermediate and final results.
            Can be specified as:

            1. A string: Use a single storage class for all outputs.
            2. A dictionary: Specify different storage classes for different outputs.

               - Use output names as keys and storage class names as values.
               - Use an empty string ``""`` as a key to set a default storage class.

            Available storage classes are registered in `pipefunc.map.storage_registry`.
            Common options include ``"file_array"``, ``"dict"``, and ``"shared_memory_dict"``.
        persist_memory
            Whether to write results to disk when memory based storage is used.
            Does not have any effect when file based storage is used.
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
        show_progress
            Whether to display a progress bar.
        return_results
            Whether to return the results of the pipeline. If ``False``, the pipeline is run
            without keeping the results in memory. Instead the results are only kept in the set
            ``storage``. This is useful for very large pipelines where the results do not fit into memory.
        scheduling_strategy
            Strategy for scheduling pipeline function execution:

            - "generation" (default): Executes functions in strict topological generations,
              waiting for all functions in a generation to complete before starting the next.
              Provides predictable execution order but may not maximize parallelism.

            - "eager": Dynamically schedules functions as soon as their dependencies are met,
              without waiting for entire generations to complete. Can improve performance
              by maximizing parallel execution, especially for complex dependency graphs
              with varied execution times.

        See Also
        --------
        map
            The synchronous version of this method.

        Returns
        -------
            An `AsyncRun` instance that contains ``run_info``, ``progress`` and ``task``.
            The ``task`` can be awaited to get the final result of the pipeline.


        """
        if scheduling_strategy == "generation":
            run_map_func = run_map_async
        elif scheduling_strategy == "eager":
            run_map_func = run_map_eager_async
        else:  # pragma: no cover
            msg = f"Invalid scheduling type: {scheduling_strategy}"
            raise ValueError(msg)

        return run_map_func(
            self,
            inputs,
            run_folder,
            internal_shapes=internal_shapes,
            output_names=output_names,
            executor=executor,
            chunksizes=chunksizes,
            storage=storage,
            persist_memory=persist_memory,
            cleanup=cleanup,
            fixed_indices=fixed_indices,
            auto_subpipeline=auto_subpipeline,
            show_progress=show_progress,
            return_results=return_results,
        )

    def arg_combinations(self, output_name: OUTPUT_TYPE) -> set[tuple[str, ...]]:
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

    def root_args(self, output_name: OUTPUT_TYPE | None = None) -> tuple[str, ...]:
        """Return the root arguments required to compute a specific (or all) output(s).

        Parameters
        ----------
        output_name
            The identifier for the return value of the pipeline. If ``None``,
            the root arguments for all outputs are returned.

        Returns
        -------
            A tuple containing the root arguments required to compute the output.
            The tuple is sorted in alphabetical order.

        """
        if r := self._internal_cache.root_args.get(output_name):
            return r

        if output_name is None:
            root_args = tuple(sorted(self.topological_generations.root_args))
        else:
            all_root_args = set(self.topological_generations.root_args)
            ancestors = nx.ancestors(self.graph, self.output_to_func[output_name])
            root_args_set = {n for n in self.graph.nodes if n in all_root_args and n in ancestors}
            root_args = tuple(sorted(root_args_set))

        self._internal_cache.root_args[output_name] = root_args
        return root_args

    def func_dependencies(self, output_name: OUTPUT_TYPE | PipeFunc) -> list[OUTPUT_TYPE]:
        """Return the functions required to compute a specific output.

        See Also
        --------
        func_dependents

        """
        return _traverse_graph(output_name, "predecessors", self.graph, self.node_mapping)

    def func_dependents(self, name: OUTPUT_TYPE | PipeFunc) -> list[OUTPUT_TYPE]:
        """Return the functions that depend on a specific input/output.

        See Also
        --------
        func_dependencies

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
        self.validate()

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
        self.validate()

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

        Raises
        ------
        ValueError
            If no function's scope was updated, e.g., when both ``inputs=None`` and ``outputs=None``.

        Examples
        --------
        >>> pipeline.update_scope("my_scope", inputs="*", outputs="*")  # Add scope to all inputs and outputs
        >>> pipeline.update_scope("my_scope", "*", "*", exclude={"output1"}) # Add to all except "output1"
        >>> pipeline.update_scope("my_scope", inputs="*", outputs={"output2"})  # Add scope to all inputs and "output2"
        >>> pipeline.update_scope(None, inputs="*", outputs="*")  # Remove scope from all inputs and outputs

        """
        validate_scopes(self.functions, scope)
        all_inputs = set(self.topological_generations.root_args)
        all_outputs = self.all_output_names
        if inputs == "*":
            inputs = all_inputs
        if outputs == "*":
            outputs = all_outputs
        if exclude is None:
            exclude = set()
        changed_any = False
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
                changed_any = True
                f.update_scope(scope, inputs=f_inputs, outputs=f_outputs, exclude=exclude)
        if not changed_any:
            msg = "No function's scope was updated. Ensure `inputs` and/or `outputs` are specified correctly."
            raise ValueError(msg)
        self._clear_internal_cache()
        self.validate()

    def _flatten_scopes(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        flat_scope_kwargs = kwargs
        for f in self.functions:
            flat_scope_kwargs = f._flatten_scopes(flat_scope_kwargs)
        return flat_scope_kwargs

    @functools.cached_property
    def parameter_annotations(self) -> dict[str, Any]:
        """Return the parameter annotations for the pipeline.

        The parameter annotations are computed by traversing the pipeline graph in topological order
        and collecting the annotations from the functions. If there are conflicting annotations
        for the same parameter, a warning is issued and the first encountered annotation is used.
        """
        annotations: dict[str, Any] = {}
        for f in self.sorted_functions:
            for p, v in f.parameter_annotations.items():
                if p in annotations and annotations[p] != v:
                    msg = (
                        f"Conflicting annotations for parameter `{p}`: `{annotations[p]}` != `{v}`."
                    )
                    warnings.warn(msg, stacklevel=2)
                    continue
                annotations[p] = v
        return annotations

    @functools.cached_property
    def output_annotations(self) -> dict[str, Any]:
        """Return the (final and intermediate) output annotations for the pipeline."""
        annotations: dict[str, Any] = {}
        for f in self.sorted_functions:
            annotations.update(f.output_annotation)
        return annotations

    @functools.cached_property
    def all_arg_combinations(self) -> dict[OUTPUT_TYPE, set[tuple[str, ...]]]:
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
    def all_root_args(self) -> dict[OUTPUT_TYPE, tuple[str, ...]]:
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

    def validate(self) -> None:
        """Validate the pipeline (checks its scopes, renames, defaults, mapspec, type hints).

        This is automatically called when the pipeline is created and when calling state
        updating methods like {method}`~Pipeline.update_renames` or
        {method}`~Pipeline.update_defaults`. Should be called manually after e.g.,
        manually updating `pipeline.validate_type_annotations` or changing some other attributes.
        """
        validate_scopes(self.functions)
        validate_consistent_defaults(self.functions, output_to_func=self.output_to_func)
        self._validate_mapspec()
        if self.validate_type_annotations:
            validate_consistent_type_annotations(self.graph)

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
                " argument to disambiguate."
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
        non_root_inputs = find_non_root_axes(mapspecs, root_args)
        output_names = {at_least_tuple(f.output_name) for f in self.functions}
        multi_output_mapping = {n: names for names in output_names for n in names if len(names) > 1}
        replace_none_in_axes(mapspecs, non_root_inputs, multi_output_mapping)  # type: ignore[arg-type]
        return create_missing_mapspecs(self.functions, non_root_inputs)  # type: ignore[arg-type]

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
            add_mapspec_axis(p, dims={}, axis=axis, functions=self.sorted_functions)
        self._clear_internal_cache()
        self.validate()

    def _func_node_colors(
        self,
        *,
        conservatively_combine: bool = False,
        output_name: OUTPUT_TYPE | None = None,
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
        *,
        backend: Literal["matplotlib", "graphviz", "graphviz_widget", "holoviews"] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Visualize the pipeline as a directed graph.

        If running in a Jupyter notebook and *not* in VS Code a widget-based backend
        will be used if available.

        Parameters
        ----------
        backend
            The plotting backend to use. If ``None``, the best backend available
            will be used in the following order: Graphviz (widget), Graphviz,
            Matplotlib, and HoloViews.
        kwargs
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
            The output of the plotting function.

        See Also
        --------
        visualize_graphviz
            Create a directed graph using Graphviz (``backend="graphviz"``).
        visualize_graphviz_widget
            Create a directed graph using Graphviz and ipywidgets (``backend="graphviz_widget"``).
        visualize_matplotlib
            Create a directed graph using Matplotlib (``backend="matplotlib"``).
        visualize_holoviews
            Create a directed graph using HoloViews (``backend="holoviews"``).

        """
        if backend is None:  # pragma: no cover
            if os.getenv("READTHEDOCS") is not None:
                # Set a default visualization backend in the docs
                # until AnyWidget shares JS code: https://github.com/manzt/anywidget/pull/628
                # https://github.com/manzt/anywidget/issues/613
                # TODO: Remove this.
                backend = "graphviz"
            elif is_installed("graphviz"):
                if is_installed("graphviz_anywidget") and is_running_in_ipynb():
                    backend = "graphviz_widget"
                else:
                    backend = "graphviz"
            elif is_installed("matplotlib"):
                backend = "matplotlib"
            elif is_installed("holoviews"):
                backend = "holoviews"
            else:
                msg = (
                    "No plotting backends are installed."
                    " Install 'graphviz', 'matplotlib', or 'holoviews' to visualize the pipeline."
                    " To install all backends, run `pip install 'pipefunc[plotting]'`."
                )
                raise ImportError(msg)
        if backend == "graphviz":
            return self.visualize_graphviz(**kwargs)
        if backend == "graphviz_widget":
            return self.visualize_graphviz_widget(**kwargs)
        if backend == "matplotlib":
            return self.visualize_matplotlib(**kwargs)
        if backend == "holoviews":
            return self.visualize_holoviews(**kwargs)
        msg = f"Invalid backend: {backend}. Must be 'graphviz_widget', 'graphviz', 'matplotlib', or 'holoviews'."  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover

    def visualize_graphviz(
        self,
        *,
        figsize: tuple[int, int] | int | None = None,
        collapse_scopes: bool | Sequence[str] = False,
        min_arg_group_size: int | None = None,
        filename: str | Path | None = None,
        style: GraphvizStyle | None = None,
        orient: Literal["TB", "LR", "BT", "RL"] = "LR",
        graphviz_kwargs: dict[str, Any] | None = None,
        show_legend: bool = True,
        include_full_mapspec: bool = False,
        return_type: Literal["graphviz", "html"] | None = None,
    ) -> graphviz.Digraph | IPython.display.HTML:
        """Visualize the pipeline as a directed graph using Graphviz.

        Parameters
        ----------
        figsize
            The width and height of the figure in inches.
            If a single integer is provided, the figure will be a square.
            If ``None``, the size will be determined automatically.
        collapse_scopes
            Whether to collapse scopes in the graph.
            If ``True``, scopes are collapsed into a single node.
            If a sequence of scope names, only the specified scopes are collapsed.
        min_arg_group_size
            Minimum number of parameters to combine into a single node. Only applies to
            parameters used exclusively by one PipeFunc. If None, no grouping is performed.
        filename
            The filename to save the figure to, if provided.
        style
            Style for the graph visualization.
        orient
            Graph orientation: 'TB', 'LR', 'BT', 'RL'.
        graphviz_kwargs
            Graphviz-specific keyword arguments for customizing the graph's appearance.
        show_legend
            Whether to show the legend in the graph visualization.
        include_full_mapspec
            Whether to include the full mapspec as a separate line in the `PipeFunc` labels.
        return_type
            The format to return the visualization in.
            If ``'html'``, the visualization is returned as a `IPython.display.html`,
            if ``'graphviz'``, the `graphviz.Digraph` object is returned.
            If ``None``, the format is ``'html'`` if running in a Jupyter notebook,
            otherwise ``'graphviz'``.

        Returns
        -------
        graphviz.Digraph
            The resulting Graphviz Digraph object.

        """
        from pipefunc._plotting import visualize_graphviz

        return visualize_graphviz(
            self.graph,
            self.defaults,
            figsize=figsize,
            collapse_scopes=collapse_scopes,
            min_arg_group_size=min_arg_group_size,
            filename=filename,
            style=style,
            orient=orient,
            graphviz_kwargs=graphviz_kwargs,
            show_legend=show_legend,
            include_full_mapspec=include_full_mapspec,
            return_type=return_type,
        )

    def visualize_graphviz_widget(
        self,
        *,
        collapse_scopes: bool | Sequence[str] = False,
        orient: Literal["TB", "LR", "BT", "RL"] = "LR",
        graphviz_kwargs: dict[str, Any] | None = None,
    ) -> ipywidgets.VBox:
        """Create an interactive visualization of the pipeline as a directed graph.

        Creates a widget that allows interactive exploration of the pipeline graph.
        The widget provides the following interactions:

        - Zoom: Use mouse scroll
        - Pan: Click and drag
        - Node selection: Click on nodes to highlight connected nodes
        - Multi-select: Shift-click on nodes to select multiple routes
        - Search: Use the search box to highlight matching nodes
        - Reset view: Press Escape

        Requires the `graphviz-anywidget` package to be installed, which is maintained
        by the pipefunc authors, see https://github.com/pipefunc/graphviz-anywidget

        Parameters
        ----------
        collapse_scopes
            Whether to collapse scopes in the graph.
            If ``True``, scopes are collapsed into a single node.
            If a sequence of scope names, only the specified scopes are collapsed.
        orient
            Graph orientation, controlling the main direction of the graph flow.
            Options are:
            - 'TB': Top to bottom
            - 'LR': Left to right
            - 'BT': Bottom to top
            - 'RL': Right to left
        graphviz_kwargs
            Graphviz-specific keyword arguments for customizing the graph's appearance.

        Returns
        -------
        ipywidgets.VBox
            Interactive widget containing the graph visualization.

        """
        requires(
            "graphviz_anywidget",
            "graphviz",
            reason="visualize_graphviz_widget",
            extras="plotting",
        )
        import graphviz
        from graphviz_anywidget import graphviz_widget

        graph = self.visualize_graphviz(
            collapse_scopes=collapse_scopes,
            orient=orient,
            graphviz_kwargs=graphviz_kwargs,
            return_type="graphviz",
        )
        assert isinstance(graph, graphviz.Digraph)
        dot_source = graph.source
        return graphviz_widget(dot_source)

    def visualize_matplotlib(
        self,
        figsize: tuple[int, int] | int = (10, 10),
        filename: str | Path | None = None,
        *,
        color_combinable: bool = False,
        conservatively_combine: bool = False,
        output_name: OUTPUT_TYPE | None = None,
    ) -> None:
        """Visualize the pipeline as a directed graph.

        Parameters
        ----------
        figsize
            The width and height of the figure in inches.
            If a single integer is provided, the figure will be a square.
        filename
            The filename to save the figure to.
        color_combinable
            Whether to color combinable nodes differently.
        conservatively_combine
            Argument as passed to `Pipeline.simplify_pipeline`.
        output_name
            Argument as passed to `Pipeline.simplify_pipeline`.

        """
        from pipefunc._plotting import visualize_matplotlib

        if color_combinable:
            func_node_colors = self._func_node_colors(
                conservatively_combine=conservatively_combine,
                output_name=output_name,
            )
        else:
            func_node_colors = None
        visualize_matplotlib(
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
        output_name: OUTPUT_TYPE | None = None,
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

    @property
    def error_snapshot(self) -> ErrorSnapshot | None:
        """Return an error snapshot for the pipeline.

        This value is `None` if no errors have occurred during
        the pipeline execution.
        """
        for f in self.functions:
            if f.error_snapshot:
                return f.error_snapshot
        return None

    def nest_funcs(
        self,
        output_names: set[OUTPUT_TYPE] | Literal["*"],
        new_output_name: OUTPUT_TYPE | None = None,
        function_name: str | None = None,
    ) -> NestedPipeFunc:
        """Replaces a set of output names with a single nested function inplace.

        Parameters
        ----------
        output_names
            The output names to nest in a `NestedPipeFunc`. Can also be ``"*"`` to nest all functions
            in the pipeline into a single `NestedPipeFunc`.
        new_output_name
            The identifier for the output of the wrapped function. If ``None``, it is automatically
            constructed from all the output names of the `PipeFunc` instances. Must be a subset of
            the output names of the `PipeFunc` instances.
        function_name
            The name of the nested function, if ``None`` the name will be set
            to ``"NestedPipeFunc_{output_name[0]}_{output_name[...]}"``.

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
        nested_func = NestedPipeFunc(
            funcs,
            output_name=new_output_name,
            function_name=function_name,
        )
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
        output_name: OUTPUT_TYPE,
        root_args: tuple[str, ...] | None = None,
        visited: set[OUTPUT_TYPE] | None = None,
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

    def independent_axes_in_mapspecs(self, output_name: OUTPUT_TYPE) -> set[str]:
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
        output_names: set[OUTPUT_TYPE] | None = None,
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

    def _repr_mimebundle_(
        self,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> dict[str, str]:  # pragma: no cover
        """Display the pipeline widget."""
        if is_running_in_ipynb() and is_installed("rich"):
            info = self.info()
            assert isinstance(info, dict)
            table = _rich_info_table(info)
            return table._repr_mimebundle_(include=include, exclude=exclude)
        # Return a plaintext representation of the object
        return {"text/plain": repr(self)}

    def print_documentation(
        self,
        *,
        borders: bool = False,
        skip_optional: bool = False,
        skip_intermediate: bool = True,
        description_table: bool = True,
        parameters_table: bool = True,
        returns_table: bool = True,
        order: Literal["topological", "alphabetical"] = "topological",
    ) -> None:
        """Print the documentation for the pipeline as a table formatted with Rich.

        Parameters
        ----------
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
        order
            The order in which to display the functions in the documentation.
            Options are:

            * ``topological``: Display functions in topological order.
            * ``alphabetical``: Display functions in alphabetical order (using ``output_name``).

        See Also
        --------
        info
            Returns the input and output information for the pipeline.

        """
        requires("rich", "griffe", reason="print_doc", extras="autodoc")
        doc = PipelineDocumentation.from_pipeline(self)
        format_pipeline_docs(
            doc,
            skip_optional=skip_optional,
            skip_intermediate=skip_intermediate,
            borders=borders,
            description_table=description_table,
            parameters_table=parameters_table,
            returns_table=returns_table,
            order=order,
        )

    def pydantic_model(self, model_name: str = "InputModel") -> type[pydantic.BaseModel]:
        """Generate a Pydantic model for pipeline root input parameters.

        Inspects the pipeline to extract defaults, type annotations, and docstrings to
        create a model that validates and coerces input data (e.g., from JSON) to the
        correct types. This is useful for ensuring that inputs meet the pipeline's
        requirements and for generating a CLI.

        **Multidimensional Array Handling:**
        Array inputs specified via mapspecs are annotated as nested lists because Pydantic
        cannot directly coerce JSON arrays into NumPy arrays. After validation, these
        lists are converted to NumPy ndarrays.

        Parameters
        ----------
        model_name
            Name for the generated Pydantic model class.

        Returns
        -------
        type[pydantic.BaseModel]
            A dynamically generated Pydantic model class for validating pipeline inputs. It:

            - Validates and coerces input data to the expected types.
            - Annotates multidimensional arrays as nested lists and converts them to NumPy arrays.
            - Facilitates CLI creation by ensuring proper input validation.

        Examples
        --------
        >>> from pipefunc import Pipeline, pipefunc
        >>> @pipefunc("foo")
        ... def foo(x: int, y: int = 1) -> int:
        ...     return x + y
        >>> pipeline = Pipeline([foo])
        >>> InputModel = pipeline.pydantic_model()
        >>> inputs = {"x": "10", "y": "2"}
        >>> model = InputModel(**inputs)
        >>> model.x, model.y
        (10, 2)
        >>> results = pipeline.map(model)  # Equivalent to `pipeline.map(inputs)`

        Notes
        -----
        - If available, detailed parameter descriptions are extracted from docstrings using griffe.
        - This method is especially useful for CLI generation, ensuring that user inputs are properly
          validated and converted before pipeline execution.

        See Also
        --------
        cli
            Automatically construct a command-line interface using argparse.
        print_documentation
            Print the pipeline documentation as a table formatted with Rich.

        """
        return pipeline_to_pydantic(self, model_name)

    def cli(self: Pipeline, description: str | None = None) -> None:
        """Automatically construct a command-line interface using argparse.

        This method creates an `argparse.ArgumentParser` instance, adds arguments for each
        root parameter in the pipeline using a Pydantic model, sets default values if they exist,
        parses the command-line arguments, and runs one of three subcommands:

        - ``cli``: Supply individual input parameters as command-line options.
        - ``json``: Load all input parameters from a JSON file.
        - ``docs``: Display the pipeline documentation (using `pipeline.print_documentation`).

        Mapping options (prefixed with `--map-`) are available for the `cli` and `json` subcommands to control
        parallel execution, storage method, and cleanup behavior.

        Usage Examples:

        **CLI mode:**
            ``python cli-example.py cli --x 2 --y 3 --map-parallel false --map-cleanup true``

        **JSON mode:**
            ``python cli-example.py json --json-file inputs.json --map-parallel false --map-cleanup true``

        **Docs mode:**
            ``python cli-example.py docs``

        Parameters
        ----------
        pipeline
            The PipeFunc pipeline instance to be executed.
        description
            A custom description for the CLI help message. If not provided, a default description is used.

        Raises
        ------
        ValueError
            If an invalid subcommand is specified.
        FileNotFoundError
            If the JSON input file does not exist (in JSON mode).
        json.JSONDecodeError
            If the JSON input file is not formatted correctly.

        Examples
        --------
        >>> if __name__ == "__main__":
        ...     pipeline = create_my_pipeline()
        ...     pipeline.cli()

        See Also
        --------
        pydantic_model
            Generate a Pydantic model for pipeline root input parameters.
        print_documentation
            Print the pipeline documentation as a table formatted with Rich.

        """
        cli(self, description=description)


class Generations(NamedTuple):
    root_args: list[str]
    function_lists: list[list[PipeFunc]]


@dataclass(frozen=True, slots=True, eq=True)
class _Bound:
    name: str
    output_name: OUTPUT_TYPE


@dataclass(frozen=True, slots=True, eq=True)
class _Resources:
    name: str
    output_name: OUTPUT_TYPE


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

    __slots__ = ["_call_with_root_args", "output_name", "pipeline", "root_args"]

    def __init__(
        self,
        pipeline: Pipeline,
        output_name: OUTPUT_TYPE | list[OUTPUT_TYPE],
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
    output_name: OUTPUT_TYPE,
    all_results: dict[OUTPUT_TYPE, Any],
    lazy: bool,  # noqa: FBT001
) -> None:
    if isinstance(func.output_name, tuple):
        assert func.output_picker is not None
        for name in func.output_name:
            all_results[name] = (
                _LazyFunction(func.output_picker, args=(r, name))
                if lazy
                else func.output_picker(r, name)
            )
        if isinstance(output_name, tuple):
            # Also assign the full name because `_run` will need it
            # This duplicates the result but it's a small overhead
            all_results[func.output_name] = r
    else:
        all_results[func.output_name] = r


def _execute_func(func: PipeFunc, func_args: dict[str, Any], lazy: bool) -> Any:  # noqa: FBT001
    if lazy:
        return _LazyFunction(func, kwargs=func_args)
    try:
        return func(**func_args)
    except Exception as e:
        handle_error(e, func, func_args)
        # handle_error raises but mypy doesn't know that
        raise  # pragma: no cover


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


def _traverse_graph(
    start: OUTPUT_TYPE | PipeFunc,
    direction: Literal["predecessors", "successors"],
    graph: nx.DiGraph,
    node_mapping: dict[OUTPUT_TYPE, PipeFunc | str],
) -> list[OUTPUT_TYPE]:
    visited = set()

    def _traverse(x: OUTPUT_TYPE | PipeFunc) -> list[OUTPUT_TYPE]:
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


@dataclass(frozen=True, slots=True)
class _PipelineInternalCache:
    arg_combinations: dict[OUTPUT_TYPE, set[tuple[str, ...]]] = field(default_factory=dict)
    root_args: dict[OUTPUT_TYPE | None, tuple[str, ...]] = field(default_factory=dict)
    func: dict[OUTPUT_TYPE | tuple[OUTPUT_TYPE, ...], _PipelineAsFunc] = field(default_factory=dict)
    func_defaults: dict[OUTPUT_TYPE, dict[str, Any]] = field(default_factory=dict)


def _rich_info_table(info: dict[str, Any], *, prints: bool = False) -> Table:
    """Create a rich table from a dictionary of information."""
    requires("rich", reason="print_table=True", extras="rich")
    import rich.table

    table = rich.table.Table(title="Pipeline Info", box=rich.box.DOUBLE)
    table.add_column("Category", style="dim", width=20)
    table.add_column("Items")

    for category, items in info.items():
        styles = {"required_inputs": "bold green", "optional_inputs": "bold yellow"}
        table.add_row(category, ", ".join(items), style=styles.get(category))
    if prints:
        console = rich.get_console()
        console.print(table)
    return table


def _root_args(pipeline: Pipeline, output_name: OUTPUT_TYPE | list[OUTPUT_TYPE]) -> tuple[str, ...]:
    if not isinstance(output_name, list):
        return pipeline.root_args(output_name)
    root_args: list[str] = []
    for name in output_name:
        root_args.extend(pipeline.root_args(name))
    # deduplicate while preserving order
    return tuple(dict.fromkeys(root_args))
