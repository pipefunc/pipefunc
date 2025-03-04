from __future__ import annotations

import tempfile
import time
import warnings
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from pipefunc.cache import DiskCache, HybridCache, LRUCache, SimpleCache, to_hashable

from ._types import OUTPUT_TYPE

if TYPE_CHECKING:
    from pipefunc._pipefunc import PipeFunc


_CACHE_KEY_TYPE: TypeAlias = tuple[OUTPUT_TYPE, tuple[tuple[str, Any], ...]]


def create_cache(
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


def compute_cache_key(
    cache_id: str,
    kwargs: dict[str, Any],
    root_args: tuple[str, ...],
) -> _CACHE_KEY_TYPE | None:
    """Compute the cache key for a specific output name.

    The cache key is a tuple consisting of the output name and a tuple of
    root input keys and their corresponding values. Root inputs are the
    inputs that are not derived from any other function in the pipeline.

    If any of the root inputs are not available in kwargs, the cache key computation is
    skipped, and the method returns None. This can happen when a non-root input is
    directly provided as an input to another function, in which case the result should not
    be cached.

    Parameters
    ----------
    cache_id
        A hashable identifier for the PipeFunc instance.
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

    return cache_id, tuple(cache_key_items)


def update_cache(
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


def get_result_from_cache(
    func: PipeFunc,
    cache: LRUCache | HybridCache | DiskCache | SimpleCache,
    cache_key: _CACHE_KEY_TYPE | None,
    output_name: OUTPUT_TYPE,
    all_results: dict[OUTPUT_TYPE, Any],
    full_output: bool,  # noqa: FBT001
    used_parameters: set[str | None],
    lazy: bool = False,  # noqa: FBT002, FBT001
) -> tuple[bool, bool]:
    from ._base import _update_all_results

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
