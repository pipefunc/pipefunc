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

Caching can dramatically speed up pipelines by storing and reusing the results of expensive function calls, avoiding redundant computations.
By storing the results of expensive function calls and reusing them when the same inputs occur again, you can significantly improve performance.

## Enabling Caching

Caching is disabled by default for each {class}`~pipefunc.PipeFunc` but can be enabled using the `cache=True` parameter of the `@pipefunc` decorator or the {class}`~pipefunc.PipeFunc` constructor.
When caching is enabled, the result of the function is stored in the pipeline's cache after the first call.
The actual cache lives in the {class}`~pipefunc.Pipeline` object and is shared across all functions in the pipeline.
It's configured at the {class}`~pipefunc.Pipeline` level using the `cache_type` and `cache_kwargs` parameters.
When you create a {class}`~pipefunc.Pipeline`, you can specify the type of cache to use:

```{code-cell} ipython3
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

Individual functions have caching enabled or disabled using the `cache` parameter of the `@pipefunc` decorator or the {class}`~pipefunc.PipeFunc` constructor.
If `cache=True` the function will use the pipeline's cache.
If `cache=False` (default) caching is disabled for that function.

## Cache Types

`pipefunc` supports several cache types, each with different characteristics and trade-offs:

- **`lru` (Least Recently Used)**:
  Keeps a fixed number of the most recently used items in memory.
  When the cache is full, the least recently used item is discarded.
  This is a good general-purpose cache.
  Use the `shared=True` option in `cache_kwargs` when using `pipeline.map` with `parallel=True`.
  This creates a {class}`~pipefunc.cache.LRUCache` instance.

- **`hybrid`**:
  Uses a combination of Least Frequently Used (LFU) and Least Computationally Expensive (LCE) strategies to determine which items to evict from the cache.
  This is useful when the computation time of functions varies significantly.
  Use the `shared=True` option in `cache_kwargs` when using `pipeline.map` with `parallel=True`.
  This creates a {class}`~pipefunc.cache.HybridCache` instance.

- **`disk`**:
  Stores cached results on disk using `pickle` or `cloudpickle`.
  Useful for caching large objects that do not fit in memory or for persisting the cache across sessions.
  The `with_lru_cache=True` option in `cache_kwargs` can be used to combine `disk` with an in-memory LRU cache to avoid reading from disk too often.
  This creates a {class}`~pipefunc.cache.DiskCache` instance.

- **`simple`**:
  A basic cache that stores all results in memory without any eviction strategy.
  Useful for debugging or when you know that the cache will not grow too large.
  This creates a {class}`~pipefunc.cache.SimpleCache` instance.

You can specify the cache type using the `cache_type` parameter when creating a {class}`~pipefunc.Pipeline`.

## Cache Keys

The cache key is computed based on the input values of each {class}`~pipefunc.PipeFunc`.
When using `pipeline.run` or calling the pipeline as a function, the cache key is computed based solely on the root arguments provided to the pipeline. This means that _*only*_ the root arguments need to be "hashable" (see [section](#handling-unhashable-objects) below) for caching to work.
When using `pipeline.map`, the cache key is computed based on the input values of each {class}`~pipefunc.PipeFunc`. That means that _*all*_ arguments to each cached function must be "hashable" (see [section](#handling-unhashable-objects) below) for caching to work.

By default, `pipefunc` uses the `pipefunc.cache.to_hashable` function to convert non-hashable input arguments into a hashable representation that can be used as a cache key.
This function handles most built-in Python types and some common third-party types like NumPy arrays and pandas Series/DataFrames.
If `to_hashable` cannot create a hashable representation of the input, it will attempt to serialize it to a string using `cloudpickle`.
If that also fails, it will raise an {class}`pipefunc.cache.UnhashableError`.

## Parallelization and Caching

When using `pipeline.map` with `parallel=True`, the pipeline will execute functions in parallel using multiple processes.
In this scenario, it is essential to use a cache that is safe for parallel access.
`pipefunc` offers two primary options for this: shared memory caches and disk-based caches.

**Shared Memory Caches:**

The `lru` and `hybrid` cache types support shared memory when created with `shared=True` in the `cache_kwargs`.
This ensures that all processes can safely access and update the same cache in memory without conflicts.

**Disk-Based Caches:**

The `disk` type provides an alternative approach where each process stores its cached data in separate files on disk.
This allows for caching large datasets that exceed available memory but comes with slower access times due to disk I/O.

**Example with Shared Memory Cache:**

```{code-cell} ipython3
from pipefunc import Pipeline, pipefunc

# Enable shared memory caching for the pipeline using an LRU cache
pipeline = Pipeline(
    [my_function, another_function],
    cache_type="lru",
    cache_kwargs={"max_size": 256, "shared": True},
)

# The pipeline can now be safely used with `pipeline.map(..., parallel=True)`
```

**Example with Disk Cache:**

```{code-cell} ipython3
from pipefunc import Pipeline, pipefunc

# Enable disk caching for the pipeline
pipeline = Pipeline(
    [my_function, another_function],
    cache_type="disk",
    cache_kwargs={"cache_dir": "my_cache_dir"},
)

# The pipeline can now be safely used with `pipeline.map(..., parallel=True)`
```

**Choosing the Right Cache Type:**

- **Shared Memory Caches (`lru`, `hybrid` with `shared=True`):**

  - Generally faster access times as data is stored in memory.
  - Suitable for smaller datasets that can fit in memory.
  - Requires careful consideration of `max_size` to avoid excessive memory consumption.

- **Disk-Based Caches (`disk`):**
  - Ideal for very large datasets that exceed available memory.
  - Cache survives across pipeline runs and process terminations.
  - Can handle larger cache sizes without impacting memory usage.
  - Slower access times compared to shared memory caches due to disk I/O.
  - The `with_lru_cache=True` option can be used to mitigate this by adding an in-memory LRU cache.

**Important Considerations:**

- Using a shared cache or a disk cache with `parallel=False` is generally not recommended as it adds unnecessary overhead without providing any benefits.
- When using `cache_type=disk`, ensure that the specified `cache_dir` has sufficient disk space and that the I/O performance is adequate for your needs.

## Handling Unhashable Objects

If your function arguments are not hashable, `pipefunc` will attempt to convert them into a hashable representation using the {func}`pipefunc.cache.to_hashable` function.
This function handles most built-in Python types and some common third-party types like NumPy arrays and pandas Series/DataFrames.
If it cannot make the object hashable, it will attempt to serialize it using `cloudpickle`.
Finally, if that fails, it will raise an {class}`pipefunc.cache.UnhashableError`.

## Clearing the Cache

You can clear the pipeline's cache using the `clear()` method of the cache object:

```{code-cell} ipython3
pipeline.cache.clear()
```

## Important Notes

- The caching behavior differs between `pipeline.map` and `pipeline.run`/`pipeline(...)`.
- When using `pipeline.map` with `parallel=True`, the cache itself will be serialized, so one must use a cache that supports shared memory, such as {class}`~pipefunc.cache.LRUCache` with `shared=True` or a disk cache like {class}`~pipefunc.cache.DiskCache`.
- The {func}`pipefunc.cache.to_hashable` function is used to attempt to ensure that input values are hashable, which is a requirement for storing results in a cache.
- This function works for many common types but is not guaranteed to work for all types.

By understanding and utilizing `pipefunc`'s caching mechanisms effectively, you can significantly improve the performance of your pipelines, especially when dealing with computationally expensive functions or large datasets.

## Advanced: Caching Stateful Functions

When caching stateful functions, you need to be careful about the cache key because the function's internal state can affect the result, even if the input arguments are the same.
By default, `pipefunc` computes the [cache key](#cache-keys) based on the function's input arguments.
However, this is insufficient for stateful functions where the internal state can change the output.

To address this, `pipefunc` provides a mechanism to customize how the cache key is generated for stateful functions using the special `__pipefunc_hash__` method.

### The `__pipefunc_hash__` Method

If a function (or a callable object) defines a `__pipefunc_hash__` method, `pipefunc` will call this method to obtain a string representation of the function's state, which will be included in the cache key.
This allows you to incorporate relevant parts of the function's state into the cache key, ensuring that the cached results are invalidated when the state changes.
It's crucial that this string uniquely represents the state of the object, as any collisions will lead to incorrect cache behavior.

**Example:**

```{code-cell} ipython3
from pipefunc import PipeFunc, Pipeline

class MyStatefulFunction:
    def __init__(self, value: int):
        self.value = value

    def __call__(self, x: int) -> int:
        return self.value + x

    def __pipefunc_hash__(self) -> str:
        # Include the relevant state in the hash
        return str(self.value)

func = MyStatefulFunction(1)
pfunc = PipeFunc(func, "out", cache=True)
pipeline = Pipeline([pfunc], cache_type="disk")

# Call the function to populate the cache
result1 = pipeline(x=1)
print(f"{pipeline.cache.cache=}")  # Print the cache
```

In this example, `MyStatefulFunction` has an internal state `value`.
The `__pipefunc_hash__` method returns a string representation of this state.
When the function is called through the {class}`~pipefunc.PipeFunc` instance, `pipefunc` will automatically call `__pipefunc_hash__` to get the state representation and include it in the cache key.

**Note:**

- The `__pipefunc_hash__` method should return a string that uniquely identifies the relevant state of the function.
- If you don't define `__pipefunc_hash__` for a stateful function, only the input arguments will be used for cache key computation, which might lead to incorrect cached results.
- The `__pipefunc_hash__` method is only relevant when caching is enabled for the {class}`~pipefunc.PipeFunc` instance (i.e., `cache=True`).

By using the `__pipefunc_hash__` method, you can ensure that `pipefunc`'s caching mechanism correctly handles stateful functions and invalidates cached results when the function's state changes.

## The `@memoize` Decorator

The `pipefunc.cache` module also provides a {func}`pipefunc.cache.memoize` decorator for general-purpose function memoization, independent of {class}`~pipefunc.PipeFunc` and {class}`~pipefunc.Pipeline`.
This decorator allows you to cache the results of any function using the available cache types (e.g., `LRUCache`, `HybridCache`, `SimpleCache`, `DiskCache`).

```{code-cell} ipython3
from pipefunc.cache import memoize, LRUCache, HybridCache, SimpleCache, DiskCache

# Use a shared LRU cache
lru_cache = LRUCache(max_size=256, shared=True)

@memoize(cache=lru_cache)
def expensive_function(arg1, arg2):
    # ... expensive computation ...
    return result

# Use a HybridCache with specific weights
hybrid_cache = HybridCache(max_size=100, access_weight=0.7, duration_weight=0.3, shared=True)

@memoize(cache=hybrid_cache)
def another_function(x, y):
    # ... some computation ...
    return result

# Use a SimpleCache
simple_cache = SimpleCache()

@memoize(cache=simple_cache)
def simple_function(a, b):
    # ... simple computation ...
    return a + b

# Use a DiskCache with cloudpickle serialization and LRU caching
disk_cache = DiskCache(cache_dir="cache_dir", use_cloudpickle=True, with_lru_cache=True, lru_cache_size=128)

@memoize(cache=disk_cache)
def disk_cached_function(data):
    # ... function that processes large data ...
    return processed_data
```

You can customize the behavior of `@memoize` using the following parameters:

- `cache`: The cache instance to use. If `None`, a `SimpleCache` is used.
- `key_func`: A custom function to generate cache keys. If `None`, the default key generation using `pipefunc.cache.try_to_hashable` is used.
- `fallback_to_pickle`: If `True` (default), unhashable objects will be pickled using `cloudpickle` as a last resort.
- `unhashable_action`: Determines the behavior when encountering unhashable objects:
  - `"error"`: Raise an {class}`~pipefunc.cache.UnhashableError` (default).
  - `"warning"`: Log a warning and skip caching for that call.
  - `"ignore"`: Silently skip caching for that call.

```{code-cell} ipython3
from pipefunc.cache import memoize, UnhashableError

def custom_key_func(*args, **kwargs):
    # Custom logic to generate a hashable key from the function arguments
    return hash(str(args) + str(kwargs))

@memoize(key_func=custom_key_func, unhashable_action="warning")
def my_function(data):
    # ... function that takes unhashable arguments ...
    return result
```

Use the `@memoize` decorator to easily add caching to any function, even outside the context of {class}`~pipefunc.PipeFunc` and {class}`~pipefunc.Pipeline`.
