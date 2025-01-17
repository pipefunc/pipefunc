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

# Overhead and Efficiency

```{try-notebook}
```

```{contents} ToC
:depth: 2
```

## What is the overhead / efficiency / performance of `pipefunc`?

```{note}
**tl;dr**: About 15 µs per iteration on a MacBook Pro M2 for a simple pipeline with three functions.
```

To benchmark the performance of `pipefunc`, you can measure the execution time of a pipeline using different input sizes.
Below is a simple benchmarking code example that evaluates the performance of a pipeline with three functions and a 3D sweep of 125,000 iterations.
In this example, the function bodies are trivial, as the goal is to measure the overhead introduced by `pipefunc`, rather than the computational time of complex operations.

```{code-cell} ipython3
import time
from pipefunc import Pipeline, pipefunc

@pipefunc(output_name="c", mapspec="a[k], b[k] -> c[k]")
def f(a, b):
    return 1

@pipefunc(output_name="d", mapspec="c[k], x[i], y[j] -> d[i, j, k]")
def g(b, c, x, y):
    return 1

@pipefunc(output_name="e", mapspec="d[i, j, :] -> e[i, j]")
def h(d):
    return 1

pipeline = Pipeline([f, g, h])
N = 50
iterations = N**3  # Actually N + N^3 + N^2 iterations
lst = list(range(N))

t_start = time.monotonic()
pipeline.map(
    inputs={"a": lst, "b": lst, "x": lst, "y": lst},
    storage="dict",  # store results in memory
    parallel=False,  # run sequentially to measure overhead without parallelism
)
t = time.monotonic() - t_start
print(f"Time: {t / iterations * 1e6:.2f} µs per iteration")
```

This code sets up a simple pipeline with three functions, each utilizing a `mapspec` to handle multi-dimensional inputs.
The performance is measured by the time it takes to process `N**3` iterations through the pipeline, where `N` is the size of each input list.

For the provided example, you might expect an output similar to `Time: 14.93 µs per iteration` on a MacBook Pro M2.
The number reported above might be slower because it is running on ReadTheDocs' hosted hardware.
It's important to note that this benchmark avoids parallel computations and caches results in memory (using a `dict`) to focus on the overhead introduced by `pipefunc` instead of parallelization and serialization overhead.
Results can vary depending on your hardware and current system load.

By using this benchmark as a baseline, you can assess performance changes after modifying your pipeline or optimizing your function logic.
To further analyze performance, consider profiling individual functions using the `profile` option in `Pipeline`.
This will provide insights into resource usage, including CPU and memory consumption, helping you identify potential bottlenecks.

For context, consider that submitting a function to a `ThreadPoolExecutor` or `ProcessPoolExecutor` typically introduces an overhead of around 1-2 ms per function call (100x slower than the overhead of `pipefunc`), or that serializing results to disk can add an overhead of 1-100 ms per function call (100x to 10,000x slower).
