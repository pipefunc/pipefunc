
from __future__ import annotations

import functools
import inspect
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias

import networkx as nx

from pipefunc._pipefunc import ErrorSnapshot, NestedPipeFunc, PipeFunc, _maybe_mapspec
from pipefunc._profile import print_profiling_stats
from pipefunc._simplify import _func_node_colors, _identify_combinable_nodes, simplified_pipeline
from pipefunc._utils import (
    assert_complete_kwargs,
    at_least_tuple,
    clear_cached_properties,
    handle_error,
)
from pipefunc.cache import DiskCache, HybridCache, LRUCache, SimpleCache, to_hashable
from pipefunc.exceptions import UnusedParametersError
from pipefunc.lazy import _LazyFunction, task_graph
from pipefunc.map._mapspec import (
    ArraySpec,
    MapSpec,
    mapspec_axes,
    mapspec_dimensions,
    validate_consistent_axes,
)
from pipefunc.map._run import AsyncMap, run_map, run_map_async
from pipefunc.resources import Resources
from pipefunc.typing import (
    Array,
    NoAnnotation,
    Unresolvable,
    is_object_array_type,
    is_type_compatible,
)

if TYPE_CHECKING:
    from collections import OrderedDict
    from collections.abc import Callable, Iterable
    from concurrent.futures import Executor
    from pathlib import Path

    import graphviz
    import holoviews as hv
    import IPython.display

    from pipefunc._profile import ProfilingStats
    from pipefunc.map._result import Result


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
        cache_kwargs.setdefault("cache_dir", tempfile.gettempdir())
        return DiskCache(**cache_kwargs)
    if cache_type == "simple":
        return SimpleCache()

    msg = f"Invalid cache type: {cache_type}."
    raise ValueError(msg)


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
        key = to_hashable(kwargs[k])
        cache_key_items.append((k, key))

    return output_name, tuple(cache_key_items)



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