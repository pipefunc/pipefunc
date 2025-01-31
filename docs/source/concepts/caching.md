---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Caching

```{try-notebook}

```

```{contents} ToC
:depth: 2
```

Caching is a crucial technique for optimizing the performance of your pipelines.
By storing the results of expensive function calls and reusing them when the same inputs occur again, you can avoid redundant computations and significantly speed up execution.

## Enabling Caching

Caching is disabled by default for each `PipeFunc` but can be enabled using the `cache=True` parameter of the `@pipefunc` decorator or the `PipeFunc` constructor.
When caching is enabled, the result of the function is stored in the pipeline's cache after the first call.
The actual cache lives in the `Pipeline` object and is shared across all functions in the pipeline.
It's configured at the `Pipeline` level using the `cache_type` and `cache_kwargs` parameters.
When you create a `Pipeline`, you can specify the type of cache to use:

```python
from pipefunc import pipefunc, Pipeline

@pipefunc("output", cache=True)
def my_function(input1, input2):
    # ... expensive computation ...
    return result

@pipefunc("another_output", cache=True)
def another_function(x, y):
    # ... some computation ...
    return result

# Enable caching for the entire pipeline using an LRU cache
pipeline = Pipeline([my_function, another_function], cache_type="lru", cache_kwargs={"max_size": 256})
```

Individual functions have caching enabled or disabled using the `cache` parameter of the `@pipefunc` decorator or the `PipeFunc` constructor.
If `cache=True` the function will use the pipeline's cache.
If `cache=False` (default) caching is disabled for that function.

```python
@pipefunc("output", cache=False)  # Disable caching for this function
def my_function(input1, input2):
    # ... inexpensive computation ...
    return result
```

## Cache Types

`pipefunc` supports several cache types, each with different characteristics and trade-offs:

- **`lru` (Least Recently Used)**:
  Keeps a fixed number of the most recently used items in memory.
  When the cache is full, the least recently used item is discarded.
  This is a good general-purpose cache.
  Use the `shared=True` option in `cache_kwargs` when using `pipeline.map` with `parallel=True`.

- **`hybrid`**:
  Uses a combination of Least Frequently Used (LFU) and Least Computationally Expensive (LCE) strategies to determine which items to evict from the cache.
  This is useful when the computation time of functions varies significantly.
  Use the `shared=True` option in `cache_kwargs` when using `pipeline.map` with `parallel=True`.

- **`disk`**:
  Stores cached results on disk using `pickle` or `cloudpickle`.
  Useful for caching large objects that do not fit in memory or for persisting the cache across sessions.
  The `with_lru_cache=True` option in `cache_kwargs` can be used to combine `disk` with an in-memory LRU cache to avoid reading from disk too often.

- **`simple`**:
  A basic cache that stores all results in memory without any eviction strategy.
  Useful for debugging or when you know that the cache will not grow too large.

You can specify the cache type using the `cache_type` parameter when creating a `Pipeline`.

## Cache Keys

The cache key is computed based on the input values of each `PipeFunc`.
When using `pipeline.run` or calling the pipeline as a function, the cache key is computed based solely on the root arguments provided to the pipeline. This means that _*only*_ the root arguments need to be "hashable" (see [section](#handling-unhashable-objects) below) for caching to work.
When using `pipeline.map`, the cache key is computed based on the input values of each `PipeFunc`. That means that _*all*_ arguments to each cached function must be "hashable" (see [section](#handling-unhashable-objects) below) for caching to work.

By default, `pipefunc` uses the `pipefunc.cache.to_hashable` function to convert non-hashable input arguments into a hashable representation that can be used as a cache key.
This function handles most built-in Python types and some common third-party types like NumPy arrays and pandas Series/DataFrames.
Not all types can be converted to a hashable representation.

## Shared Memory Caching

When using `pipeline.map` with `parallel=True`, it is essential to use a cache that supports shared memory.
The `lru` and `hybrid` cache types support shared memory when created with `shared=True` in the `cache_kwargs`.
This ensures that multiple processes can access and update the same cache.

```python
from pipefunc import Pipeline, pipefunc

# Enable shared memory caching for the pipeline
pipeline = Pipeline(
    [my_function, another_function],
    cache_type="lru",
    cache_kwargs={"max_size": 256, "shared": True},
)
```

## Handling Unhashable Objects

If your function arguments are not hashable, `pipefunc` will attempt to convert them into a hashable representation using the `pipefunc.cache.to_hashable` function.
This function handles most built-in Python types and some common third-party types like NumPy arrays and pandas Series/DataFrames.

If the object cannot be made hashable, `pipefunc` will raise an `UnhashableError` by default.
You can change this behavior using the `unhashable_action` parameter of the `pipefunc.cache.memoize` function:

- `"error"`: Raise an `UnhashableError` (default).
- `"warning"`: Log a warning and skip caching for that call.
- `"ignore"`: Silently skip caching for that call.

```python
from pipefunc import Pipeline, pipefunc
from pipefunc.cache import memoize

# Configure memoize to warn on unhashable objects
memoize_decorator = functools.partial(memoize, unhashable_action="warning")

@pipefunc("output", memoize=memoize_decorator)
def my_function(input1, input2):
    # ... expensive computation ...
    return result

pipeline = Pipeline([my_function], cache_type="lru")
```

## Clearing the Cache

You can clear the pipeline's cache using the `clear()` method of the cache object:

```python
pipeline.cache.clear()
```

## Important Notes

- The caching behavior differs between `pipeline.map` and `pipeline.run`/`pipeline(...)`.
- When using `pipeline.map` with `parallel=True`, the cache itself will be serialized, so one must use a cache that supports shared memory, such as `~pipefunc.cache.LRUCache` with `shared=True` or a disk cache like `~pipefunc.cache.DiskCache`.
- The `pipefunc.cache.to_hashable` function is used to attempt to ensure that input values are hashable, which is a requirement for storing results in a cache.
- This function works for many common types but is not guaranteed to work for all types.
- If `~pipefunc.cache.to_hashable` cannot make a value hashable, it falls back to using the `str` representation of the value.
- Caution ⛔️: Using `str` representations can lead to unexpected behavior if they are not unique for different function calls!

By understanding and utilizing `pipefunc`'s caching mechanisms effectively, you can significantly improve the performance of your pipelines, especially when dealing with computationally expensive functions or large datasets.
