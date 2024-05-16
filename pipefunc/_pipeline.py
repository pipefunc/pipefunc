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
import sys
import time
import warnings
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Literal,
    Tuple,
    Union,
)

import networkx as nx
from tabulate import tabulate

from pipefunc._cache import DiskCache, HybridCache, LRUCache, SimpleCache
from pipefunc._lazy import _LazyFunction, task_graph
from pipefunc._pipefunc import PipeFunc
from pipefunc._plotting import visualize, visualize_holoviews
from pipefunc._simplify import _combine_nodes, _get_signature, _wrap_dict_to_tuple
from pipefunc._utils import at_least_tuple, generate_filename_from_dict, handle_error
from pipefunc.exceptions import UnusedParametersError
from pipefunc.map._mapspec import MapSpec

if sys.version_info < (3, 10):  # pragma: no cover
    from typing_extensions import TypeAlias
else:
    from typing import TypeAlias

if TYPE_CHECKING:
    if sys.version_info < (3, 9):  # pragma: no cover
        from typing import Callable
    else:
        from collections.abc import Callable

    from pathlib import Path

    import holoviews as hv

    from pipefunc._perf import ProfilingStats
    from pipefunc.map._run import Result

_OUTPUT_TYPE: TypeAlias = Union[str, Tuple[str, ...]]
_CACHE_KEY_TYPE: TypeAlias = Tuple[_OUTPUT_TYPE, Tuple[Tuple[str, Any], ...]]


class _Function:
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
        Any
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
        Any
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
        Any
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


class Pipeline:
    """Pipeline class for managing and executing a sequence of functions.

    Parameters
    ----------
    functions
        A list of functions that form the pipeline.
    lazy
        Flag indicating whether the pipeline should be lazy.
    debug
        Flag indicating whether debug information should be printed.
        If None, the value of each PipeFunc's debug attribute is used.
    profile
        Flag indicating whether profiling information should be collected.
        If None, the value of each PipeFunc's profile attribute is used.
    cache_type
        The type of cache to use.
    cache_kwargs
        Keyword arguments passed to

    """

    def __init__(
        self,
        functions: list[PipeFunc | tuple[PipeFunc, str | MapSpec]],
        *,
        lazy: bool = False,
        debug: bool | None = None,
        profile: bool | None = None,
        cache_type: Literal["lru", "hybrid", "disk", "simple"] | None = "lru",
        cache_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Pipeline class for managing and executing a sequence of functions."""
        self.functions: list[PipeFunc] = []
        self.lazy = lazy
        self._debug = debug
        self._profile = profile
        self.output_to_func: dict[_OUTPUT_TYPE, PipeFunc] = {}
        for f in functions:
            if isinstance(f, tuple):
                f, mapspec = f  # noqa: PLW2901
            else:
                mapspec = None
            self.add(f, mapspec=mapspec)
        self._init_internal_cache()
        self._cache_type = cache_type
        self._cache_kwargs = cache_kwargs
        self.cache = _create_cache(cache_type, lazy, cache_kwargs)

    def _init_internal_cache(self) -> None:
        # Internal Pipeline cache
        self._arg_combinations: dict[_OUTPUT_TYPE, set[tuple[str, ...]]] = {}
        self._root_args: dict[_OUTPUT_TYPE, tuple[str, ...]] = {}
        self._func: dict[_OUTPUT_TYPE, _Function] = {}
        with contextlib.suppress(AttributeError):
            del self.graph
        with contextlib.suppress(AttributeError):
            del self.root_nodes
        with contextlib.suppress(AttributeError):
            del self.leaf_nodes
        with contextlib.suppress(AttributeError):
            del self.unique_leaf_node
        with contextlib.suppress(AttributeError):
            del self.map_parameters
        with contextlib.suppress(AttributeError):
            del self.defaults
        with contextlib.suppress(AttributeError):
            del self.node_mapping
        with contextlib.suppress(AttributeError):
            del self.all_arg_combinations
        with contextlib.suppress(AttributeError):
            del self.all_root_args
        with contextlib.suppress(AttributeError):
            del self.topological_generations

    def _current_cache(self) -> LRUCache | HybridCache | DiskCache | SimpleCache | None:
        """Return the cache used by the pipeline."""
        if not isinstance(self.cache, SimpleCache) and (tg := task_graph()) is not None:
            return tg.cache
        return self.cache

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

        Parameters
        ----------
        f
            The function to add to the pipeline.
        profile
            Flag indicating whether profiling information should be collected.
        mapspec
            This is a specification for mapping that dictates how input values should
            be merged together. If None, the default behavior is that the input directly
            maps to the output.

        """
        if not isinstance(f, PipeFunc):
            f = PipeFunc(f, output_name=f.__name__)
        elif mapspec is not None:
            msg = (
                "Initializing the `Pipeline` using `MapSpec`s and"
                " `PipeFunc`s modifies the `PipeFunc`s inplace."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)

        if mapspec is not None:
            if isinstance(mapspec, str):
                mapspec = MapSpec.from_string(mapspec)
            f.mapspec = mapspec
            f._validate_mapspec()

        self.functions.append(f)

        self.output_to_func[f.output_name] = f
        if isinstance(f.output_name, tuple):
            for name in f.output_name:
                self.output_to_func[name] = f

        if self.profile is not None:
            f.set_profiling(enable=self.profile)

        if self.debug is not None:
            f.debug = self.debug

        self._init_internal_cache()  # reset cache
        return f

    def drop(
        self,
        *,
        f: PipeFunc | None = None,
        output_name: _OUTPUT_TYPE | None = None,
    ) -> None:
        """Drop a function from the pipeline.

        Parameters
        ----------
        f
            The function to drop from the pipeline.
        output_name
            The name of the output to drop from the pipeline.

        """
        if (f is not None and output_name is not None) or (f is None and output_name is None):
            msg = "One of `f` or `output_name` should be provided."
            raise ValueError(msg)
        if f is not None:
            self.functions.remove(f)
            if isinstance(f.output_name, tuple):
                for name in f.output_name:
                    del self.output_to_func[name]
            else:
                del self.output_to_func[f.output_name]
        elif output_name is not None:
            f = self.output_to_func[output_name]
            self.drop(f=f)
        self._init_internal_cache()

    @functools.cached_property
    def graph(self) -> nx.DiGraph:
        """Create a directed graph representing the pipeline.

        Returns
        -------
        nx.DiGraph
            A directed graph with nodes representing functions and edges
            representing dependencies between functions.

        """
        _check_consistent_defaults(self.functions)
        g = nx.DiGraph()
        for f in self.functions:
            g.add_node(f)
            assert f.parameters is not None
            for arg in f.parameters:
                if arg in self.output_to_func:  # is function output
                    edge = (self.output_to_func[arg], f)
                    if edge not in g.edges:
                        g.add_edge(*edge, arg=arg)
                    else:
                        # tuple output of function, and the edge already exists
                        assert isinstance(edge[0].output_name, tuple)
                        current = g.edges[edge]["arg"]
                        g.edges[edge]["arg"] = (*at_least_tuple(current), arg)

                else:
                    if arg not in g:  # Add the node only if it doesn't exist
                        default_value = f.defaults.get(arg, inspect.Parameter.empty)
                        g.add_node(arg, default_value=default_value)
                    g.add_edge(arg, f, arg=arg)
        return g

    def func(self, output_name: _OUTPUT_TYPE) -> _Function:
        """Create a composed function that can be called with keyword arguments.

        Parameters
        ----------
        output_name
            The identifier for the return value of the composed function.

        Returns
        -------
        Callable[..., Any]
            The composed function that can be called with keyword arguments.

        """
        if f := self._func.get(output_name):
            return f
        root_args = self.root_args(output_name)
        assert isinstance(root_args, tuple)
        f = _Function(self, output_name, root_args=root_args)
        self._func[output_name] = f
        return f

    def __call__(
        self,
        __output_name__: _OUTPUT_TYPE | None = None,
        /,
        **kwargs: Any,
    ) -> Any:
        """Call the pipeline for a specific return value.

        Parameters
        ----------
        __output_name__
            The identifier for the return value of the pipeline.
            Is None by default, in which case the unique leaf node is used.
            This parameter is positional-only and the strange name is used
            to avoid conflicts with the `output_name` argument that might be
            passed via `kwargs`.
        kwargs
            Keyword arguments to be passed to the pipeline functions.

        Returns
        -------
        Any
            The return value of the pipeline.

        """
        if __output_name__ is None:
            __output_name__ = self.unique_leaf_node.output_name
        return self.func(__output_name__)(**kwargs)

    def _get_func_args(
        self,
        func: PipeFunc,
        kwargs: dict[str, Any],
        all_results: dict[_OUTPUT_TYPE, Any],
        full_output: bool,  # noqa: FBT001
        used_parameters: set[str | None],
    ) -> dict[str, Any]:
        # Used in _execute_pipeline
        func_args = {}
        for arg in func.parameters:
            if arg in kwargs:
                func_args[arg] = kwargs[arg]
            elif arg in func.defaults:
                func_args[arg] = func.defaults[arg]
            else:
                func_args[arg] = self._execute_pipeline(
                    output_name=arg,
                    kwargs=kwargs,
                    all_results=all_results,
                    full_output=full_output,
                    used_parameters=used_parameters,
                )
        used_parameters.update(func_args)
        return func_args

    def _execute_pipeline(
        self,
        *,
        output_name: _OUTPUT_TYPE,
        kwargs: Any,
        all_results: dict[_OUTPUT_TYPE, Any],
        full_output: bool,
        used_parameters: set[str | None],
    ) -> Any:
        func = self.output_to_func[output_name]
        assert func.parameters is not None

        cache = self._current_cache()
        use_cache = (func.cache and cache is not None) or task_graph() is not None

        root_args = self.root_args(output_name)
        result_from_cache = False
        if use_cache:
            assert cache is not None
            cache_key = _compute_cache_key(func.output_name, kwargs, root_args)
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
            kwargs,
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
        _save_results(func, r, output_name, all_results, root_args, self.lazy)
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
        Any
            The return value of the pipeline or a dictionary mapping function
            names to their return values if full_output is True.

        """
        if p := self.map_parameters & set(self.func_dependencies(output_name)):
            msg = (
                f"Cannot execute pipeline to get `{output_name}` because `{p}`"
                f" have `MapSpec`(s). Use `Pipeline.map` instead."
            )
            raise RuntimeError(msg)

        if output_name in kwargs:
            msg = f"The `output_name='{output_name}'` argument cannot be provided in `kwargs={kwargs}`."
            raise ValueError(msg)

        all_results: dict[_OUTPUT_TYPE, Any] = kwargs.copy()  # type: ignore[assignment]
        used_parameters: set[str | None] = set()

        self._execute_pipeline(
            output_name=output_name,
            kwargs=kwargs,
            all_results=all_results,
            full_output=full_output,
            used_parameters=used_parameters,
        )

        # if has None, result was from cache, so we don't know which parameters were used
        if None not in used_parameters and (unused := set(kwargs) - set(used_parameters)):
            unused_str = ", ".join(sorted(unused))
            msg = f"Unused keyword arguments: `{unused_str}`. {kwargs=}, {used_parameters=}"
            raise UnusedParametersError(msg)

        return all_results if full_output else all_results[output_name]

    def map(
        self,
        inputs: dict[str, Any],
        run_folder: str | Path,
        manual_shapes: dict[str, int | tuple[int, ...]] | None = None,
        *,
        cleanup: bool = True,
        parallel: bool = True,
    ) -> list[Result]:
        from pipefunc.map import run

        return run(self, inputs, run_folder, manual_shapes, cleanup=cleanup, parallel=parallel)

    @functools.cached_property
    def node_mapping(self) -> dict[_OUTPUT_TYPE, PipeFunc | str]:
        """Return a mapping from node names to nodes.

        Returns
        -------
        Dict[_OUTPUT_TYPE, PipeFunc | str]
            A mapping from node names to nodes.

        """
        mapping: dict[_OUTPUT_TYPE, PipeFunc | str] = {}
        for node in self.graph.nodes:
            if isinstance(node, PipeFunc):
                if isinstance(node.output_name, tuple):
                    for name in node.output_name:
                        mapping[name] = node
                mapping[node.output_name] = node
            else:
                assert isinstance(node, str)
                mapping[node] = node
        return mapping

    def arg_combinations(self, output_name: _OUTPUT_TYPE) -> set[tuple[str, ...]]:
        """Return the arguments required to compute a specific output.

        Parameters
        ----------
        output_name
            The identifier for the return value of the pipeline.

        Returns
        -------
        Set[Tuple[str, ...]]
            A set of tuples containing possible argument combinations.
            The tuples are sorted in lexicographical order.

        """
        if r := self._arg_combinations.get(output_name):
            return r
        head = self.node_mapping[output_name]
        arg_set: set[tuple[str, ...]] = set()
        _compute_arg_mapping(self.graph, head, head, [], [], arg_set)  # type: ignore[arg-type]
        self._arg_combinations[output_name] = arg_set
        return arg_set

    def root_args(self, output_name: _OUTPUT_TYPE) -> tuple[str, ...]:
        """Return the root arguments required to compute a specific output."""
        if r := self._root_args.get(output_name):
            return r
        arg_combos = self.arg_combinations(output_name)
        root_args = next(
            args for args in arg_combos if all(isinstance(self.node_mapping[n], str) for n in args)
        )
        self._root_args[output_name] = root_args
        return root_args

    def func_dependencies(self, output_name: _OUTPUT_TYPE) -> list[_OUTPUT_TYPE]:
        """Return the functions required to compute a specific output."""

        def _predecessors(x: _OUTPUT_TYPE | PipeFunc) -> list[_OUTPUT_TYPE]:
            preds = set()
            if isinstance(x, (str, tuple)):
                x = self.node_mapping[x]
            for pred in self.graph.predecessors(x):
                if isinstance(pred, PipeFunc):
                    preds.add(pred.output_name)
                    for p in _predecessors(pred):
                        preds.add(p)
            return preds  # type: ignore[return-value]

        return sorted(_predecessors(output_name), key=at_least_tuple)

    @functools.cached_property
    def all_arg_combinations(self) -> dict[_OUTPUT_TYPE, set[tuple[str, ...]]]:
        """Compute all possible argument mappings for the pipeline.

        Returns
        -------
        Dict[_OUTPUT_TYPE, Set[Tuple[str, ...]]]
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
    def map_parameters(self) -> set[str]:
        map_parameters: set[str] = set()
        for func in self.functions:
            if func.mapspec:
                map_parameters.update(func.mapspec.parameters)
                for output in func.mapspec.outputs:
                    map_parameters.add(output.name)
        return map_parameters

    @functools.cached_property
    def defaults(self) -> dict[str, Any]:
        defaults = {}
        for func in self.functions:
            defaults.update(func.defaults)
        return defaults

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
    def topological_generations(self) -> tuple[list[str], list[list[PipeFunc]]]:
        generations = list(nx.topological_generations(self.graph))
        assert all(isinstance(x, str) for x in generations[0])
        assert all(isinstance(x, PipeFunc) for gen in generations[1:] for x in gen)
        return generations[0], generations[1:]

    def _func_node_colors(
        self,
        *,
        conservatively_combine: bool = False,
        output_name: _OUTPUT_TYPE | None = None,
    ) -> list[str]:
        if output_name is None:
            output_name = self.unique_leaf_node.output_name

        func_node_colors = []
        combinable_nodes = self._identify_combinable_nodes(
            output_name=output_name,
            conservatively_combine=conservatively_combine,
        )
        combinable_nodes = _combine_nodes(combinable_nodes)
        node_sets = [{k, *v} for k, v in combinable_nodes.items()]
        color_index = len(node_sets)  # for non-combinable nodes
        for node in self.graph.nodes:
            if isinstance(node, PipeFunc):
                i = next(
                    (i for i, nodes in enumerate(node_sets) if node in nodes),
                    None,
                )
                if i is not None:
                    func_node_colors.append(f"C{i}")
                else:
                    func_node_colors.append(f"C{color_index}")
                    color_index += 1
        return func_node_colors

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

    def visualize_holoviews(self) -> hv.Graph:
        """Visualize the pipeline as a directed graph using HoloViews."""
        return visualize_holoviews(self.graph)

    def resources_report(self) -> None:
        """Display the resource usage report for each function in the pipeline."""
        if not self.profiling_stats:
            msg = "Profiling is not enabled."
            raise ValueError(msg)

        headers = [
            "Function",
            "Avg CPU Usage (%)",
            "Max Memory Usage (MB)",
            "Avg Time (s)",
            "Total Time (%)",
            "Number of Calls",
        ]
        table_data = []

        for func_name, stats in self.profiling_stats.items():
            row = [
                func_name,
                f"{stats.cpu.average:.2f}",
                f"{stats.memory.max / (1024 * 1024):.2f}",
                f"{stats.time.average:.2e}",
                stats.time.average * stats.time.num_executions,
                stats.time.num_executions,
            ]
            table_data.append(row)

        total_time = sum(row[4] for row in table_data)  # type: ignore[misc]
        if total_time > 0:
            for row in table_data:
                row[4] = f"{row[4] / total_time * 100:.2f}"  # type: ignore[operator]

        print("Resource Usage Report:")
        print(tabulate(table_data, headers, tablefmt="grid"))

    def _identify_combinable_nodes(
        self,
        output_name: _OUTPUT_TYPE,
        *,
        conservatively_combine: bool = False,
    ) -> dict[PipeFunc, set[PipeFunc]]:
        """Identify which function nodes can be combined into a single function.

        This method identifies the PipeFuncs in the execution graph that
        can be combined into a single function. The criterion for combinability
        is that the functions share the same root arguments.

        Parameters
        ----------
        output_name
            The name of the output from the pipeline function we are starting
            the search from. It is used to get the starting function in the
            pipeline.
        conservatively_combine
            If True, only combine a function node with its predecessors if all
            of its predecessors have the same root arguments as the function
            node itself. If False, combine a function node with its predecessors
            if any of its predecessors have the same root arguments as the
            function node.

        Returns
        -------
        dict[PipeFunc, set[PipeFunc]]
            A dictionary where each key is a PipeFunc that can be
            combined with others. The value associated with each key is a set of
            PipeFuncs that can be combined with the key function.

        Notes
        -----
        This function works by performing a depth-first search through the
        pipeline's execution graph. Starting from the PipeFunc
        corresponding to the `output_name`, it goes through each predecessor in
        the graph (functions that need to be executed before the current one).
        For each predecessor function, it recursively checks if it can be
        combined with others by comparing their root arguments.

        If a function's root arguments are identical to the head function's root
        arguments, it is considered combinable and added to the set of
        combinable functions for the head. If `conservatively_combine=True` and
        all predecessor functions are combinable, the head function and its set
        of combinable functions are added to the `combinable_nodes` dictionary.
        If `conservatively_combine=False` and any predecessor function is
        combinable, the head function and its set of combinable functions are
        added to the `combinable_nodes` dictionary.

        The function 'head' in the nested function `_recurse` represents the
        current function being checked in the execution graph.

        """
        # Nested function _recurse performs the depth-first search and updates the
        # `combinable_nodes` dictionary.

        def _recurse(head: PipeFunc) -> None:
            head_args = self.root_args(head.output_name)
            funcs = set()
            i = 0
            for node in self.graph.predecessors(head):
                if isinstance(node, (tuple, str)):  # node is root_arg
                    continue
                i += 1
                _recurse(node)
                node_args = self.root_args(node.output_name)
                if node_args == head_args:
                    funcs.add(node)
            if funcs and (not conservatively_combine or i == len(funcs)):
                combinable_nodes[head] = funcs

        combinable_nodes: dict[PipeFunc, set[PipeFunc]] = {}
        func = self.node_mapping[output_name]
        assert isinstance(func, PipeFunc)
        _recurse(func)
        return combinable_nodes

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
            the simplification from. It is used to get the starting function in the
            pipeline. If None, the unique tip of the pipeline graph is used (if
            there is one).
        conservatively_combine
            If True, only combine a function node with its predecessors if all
            of its predecessors have the same root arguments as the function
            node itself. If False, combine a function node with its predecessors
            if any of its predecessors have the same root arguments as the
            function node.

        Returns
        -------
        Pipeline
            The simplified version of the pipeline.

        Notes
        -----
        The pipeline simplification process works in the following way:

        1.  Identify combinable function nodes in the execution graph by
            checking if they share the same root arguments.
        2.  Simplify the dictionary of combinable nodes by replacing any nodes
            that can be combined with their dependencies.
        3.  Generate the set of nodes to be skipped (those that will be merged).
        4.  Get the input and output signatures for the combined nodes.
        5.  Create new pipeline functions for the combined nodes, and add them
            to the list of new functions.
        6.  Add the remaining (non-combinable) functions to the list of new
            functions.
        7.  Generate a new pipeline with the new functions.

        This process can significantly simplify complex pipelines, making them
        easier to understand and potentially improving performance by simplifying
        function calls.

        """
        if output_name is None:
            output_name = self.unique_leaf_node.output_name
        combinable_nodes = self._identify_combinable_nodes(
            output_name,
            conservatively_combine=conservatively_combine,
        )
        if not combinable_nodes:
            warnings.warn(
                "No combinable nodes found, the pipeline cannot be simplified.",
                UserWarning,
                stacklevel=2,
            )
        # Simplify the combinable_nodes dictionary by replacing any nodes that
        # can be combined with their own dependencies, so that each key in the
        # dictionary only depends on nodes that cannot be further combined.
        combinable_nodes = _combine_nodes(combinable_nodes)
        skip = set.union(*combinable_nodes.values()) if combinable_nodes else set()
        in_sig, out_sig = _get_signature(combinable_nodes, self.graph)
        m = self.node_mapping
        predecessors = [m[o] for o in self.func_dependencies(output_name)]
        head = self.node_mapping[output_name]
        new_functions = []
        for f in self.functions:
            if f != head and f not in predecessors:
                continue
            if f in combinable_nodes:
                inputs = tuple(sorted(in_sig[f]))
                outputs = tuple(sorted(out_sig[f]))
                if len(outputs) == 1:
                    outputs = outputs[0]  # type: ignore[assignment]
                funcs = [f, *combinable_nodes[f]]
                mini_pipeline = Pipeline(funcs)  # type: ignore[arg-type]
                func = mini_pipeline.func(f.output_name).call_full_output
                f_combined = _wrap_dict_to_tuple(func, inputs, outputs)
                f_combined.__name__ = f"combined_{f.__name__}"
                f_pipefunc = PipeFunc(
                    f_combined,
                    outputs,
                    profile=f.profile,
                    cache=f.cache,
                    save_function=f.save_function,
                )
                # Disable saving for all functions that are being combined
                for f_ in funcs:
                    f_.save_function = None
                f_pipefunc.parameters = list(inputs)
                new_functions.append(f_pipefunc)
            elif f not in skip:
                new_functions.append(f)
        return Pipeline(new_functions)  # type: ignore[arg-type]

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

    def copy(self) -> Pipeline:
        """Return a copy of the pipeline."""
        return Pipeline(
            self.functions,  # type: ignore[arg-type]
            lazy=self.lazy,
            debug=self._debug,
            profile=self._profile,
            cache_type=self._cache_type,
            cache_kwargs=self._cache_kwargs,
        )


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
    # Used in _execute_pipeline
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
    # Used in _execute_pipeline
    result_from_cache = False
    if cache_key is not None and cache_key in cache:
        r = cache.get(cache_key)
        _update_all_results(func, r, output_name, all_results, lazy)
        result_from_cache = True
        if not full_output:
            used_parameters.add(None)  # indicate that the result was from cache
            return True, result_from_cache
    return False, result_from_cache


def _check_consistent_defaults(functions: list[PipeFunc]) -> None:
    """Check that the default values for shared arguments are consistent."""
    arg_defaults = defaultdict(set)
    for f in functions:
        for arg, default_value in f.defaults.items():
            arg_defaults[arg].add(default_value)
            if len(arg_defaults[arg]) > 1:
                msg = (
                    f"Inconsistent default values for argument '{arg}' in"
                    " functions. Please make sure the shared input arguments have"
                    " the same default value or are set only for one function.",
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
        raise  # handle_error raises but mypy doesn't know that


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
    _CACHE_KEY_TYPE | None
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


def _save_results(
    func: PipeFunc,
    r: Any,
    output_name: _OUTPUT_TYPE,
    all_results: dict[_OUTPUT_TYPE, Any],
    root_args: tuple[str, ...],
    lazy: bool,  # noqa: FBT001
) -> None:
    # Used in _execute_pipeline
    if func.save_function is None:
        return
    to_save = {k: all_results[k] for k in root_args}
    filename = generate_filename_from_dict(to_save)  # type: ignore[arg-type]
    filename = func.__name__ / filename
    to_save[output_name] = all_results[output_name]  # type: ignore[index]
    if lazy:
        lazy_save = _LazyFunction(func.save_function, args=(filename, to_save), add_to_graph=False)
        r.add_delayed_callback(lazy_save)
    else:
        func.save_function(filename, to_save)  # type: ignore[arg-type]


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


def _unique(
    nodes: Iterable[PipeFunc | str],
) -> tuple[PipeFunc | str, ...]:
    return tuple(sorted(set(nodes), key=_sort_key))


def _filter_funcs(
    funcs: Iterable[PipeFunc | str],
) -> list[PipeFunc]:
    return [f for f in funcs if isinstance(f, PipeFunc)]


def _compute_arg_mapping(
    graph: nx.DiGraph,
    node: PipeFunc,
    head: PipeFunc,
    args: list[PipeFunc | str],
    replaced: list[PipeFunc | str],
    arg_set: set[tuple[str, ...]],
) -> None:
    preds = [n for n in graph.predecessors(node) if n not in replaced]
    deps = _unique(args + preds)
    deps_names = _names(deps)
    if deps_names in arg_set:
        return
    arg_set.add(deps_names)

    for func in _filter_funcs(deps):
        new_args = [dep for dep in deps if dep != func]
        _compute_arg_mapping(graph, func, head, new_args, [*replaced, node], arg_set)
