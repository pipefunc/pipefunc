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

# Parallelism and Execution

```{try-notebook}
```

```{contents} ToC
:depth: 2
```

(run-vs-map)=
## What is the difference between `pipeline.run` and `pipeline.map`?

These methods are used to execute the pipeline but have different use cases:

- `pipeline.run(output_name, kwargs)` is used to execute the pipeline as a function and is fully sequential. It allows going from any input arguments to any output arguments. It does **not** support map-reduce operations. Internally, it keeps all intermediate results in memory in a dictionary.
- `pipeline.map(inputs, ...)` is used to execute the pipeline in parallel. It supports map-reduce operations and is optimized for parallel execution. Internally, it puts all intermediate results in a {class}`~pipefunc.map.StorageBase` (there are implementations for storing on disk or in memory).

```{note}
Internally, the `pipeline.run` method is called when using the pipeline as a function, the following are equivalent:

- `pipeline.run(output_name, kwargs)`
- `pipeline(output_name, **kwargs)`
- `f = pipeline.func(output_name)` and then `f(**kwargs)`

```

Here is a table summarizing the differences between `pipeline.run` and `pipeline.map`:

| Feature                                                | `pipeline.run` and `pipeline(...)`                                               | `pipeline.map`                                                                                                         |
| ------------------------------------------------------ | -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Execution mode                                         | Sequential                                                                       | Parallel (any {class}`~concurrent.futures.Executor` class) or sequential                                               |
| Map-reduce support (via {class}`pipefunc.map.MapSpec`) | No                                                                               | Yes                                                                                                                    |
| Input arguments                                        | Can provide _any_ input arguments for any function in the pipeline               | Requires the root arguments (use {class}`~pipefunc.Pipeline.subpipeline` to get a subgraph)                            |
| Output arguments                                       | Can request the output of any function in the pipeline                           | Calculates _all_ function nodes in the entire pipeline (use {class}`~pipefunc.Pipeline.subpipeline` to get a subgraph) |
| Intermediate results storage                           | In-memory dictionary                                                             | Configurable storage ({class}`~pipefunc.map.StorageBase`), e.g., on disk, cloud, or in (shared-)memory                 |
| Use case                                               | Executing the pipeline as a regular function, going from any input to any output | Optimized for parallel execution and map-reduce operations                                                             |
| Calling syntax                                         | `pipeline.run(output_name, kwargs)` or `pipeline(output_name, **kwargs)`         | `pipeline.map(inputs, ...)`                                                                                            |

In summary, `pipeline.run` is used for sequential execution and allows flexible input and output arguments, while `pipeline.map` is optimized for parallel execution and map-reduce operations but requires structured inputs and outputs based on the `mapspec` of the functions.


## Mixing executors and storage backends for I/O-bound and CPU-bound work

You can mix different executors and storage backends in a pipeline.

Imagine that some `PipeFunc`s are trivial to execute, some are CPU-bound and some are I/O-bound.
You can mix different executors and storage backends in a pipeline.

Let's consider an example where we have two `PipeFunc`s, `f` and `g`.
`f` is I/O-bound and `g` is CPU-bound.
We can use a `ThreadPoolExecutor` for `f` and a `ProcessPoolExecutor` for `g`.
We will store the results of `f` in memory and store the results of `g` in a file.

```{code-cell} ipython3
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from pipefunc import Pipeline, pipefunc
import threading
import multiprocessing

@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def f(x):
    time.sleep(1)  # Simulate I/O-bound work
    return threading.current_thread().name

@pipefunc(output_name="z", mapspec="x[i] -> z[i]")
def g(x):
    np.linalg.eig(np.random.rand(10, 10))  # CPU-bound work
    return multiprocessing.current_process().name

pipeline = Pipeline([f, g])
inputs = {"x": [1, 2, 3]}

executor = {
    "y": ThreadPoolExecutor(max_workers=2),
    "": ProcessPoolExecutor(max_workers=2),  # empty string means default executor
}
storage = {
    "z": "file_array",
    "": "dict",  # empty string means default storage
}
results = pipeline.map(inputs, run_folder="run_folder", executor=executor, storage=storage)

# Get the results to check the thread and process names
thread_names = results["y"].output.tolist()
process_names = results["z"].output.tolist()
print(f"thread_names: {thread_names}")
print(f"process_names: {process_names}")
```

In both `executor` and `storage` you can use the special key `""` to apply the default executor or storage.

```{note}
The `executor` supports any executor that is compliant with the `concurrent.futures.Executor` interface.
That includes:

- `concurrent.futures.ProcessPoolExecutor`
- `concurrent.futures.ThreadPoolExecutor`
- `ipyparallel.Client().executor()`
- `dask.distributed.Client().get_executor()`
- `mpi4py.futures.MPIPoolExecutor()`
- `loky.get_reusable_executor()`
- `executorlib.SingleNodeExecutor`, `executorlib.SlurmClusterExecutor`, `executorlib.SlurmJobExecutor`, `executorlib.FluxClusterExecutor`, `executorlib.FluxJobExecutor`

```

## How to use post-execution hooks?

Post-execution hooks allow you to execute custom code after a function completes. This is useful for logging, debugging, or collecting statistics about function execution.

You can set a post-execution hook in two ways:

1. When creating a `PipeFunc` using the `post_execution_hook` parameter
2. When using the `@pipefunc` decorator

The hook function receives three arguments:

- The `PipeFunc` instance
- The return value of the function
- A dictionary of the input arguments

Here's an example:

```{code-cell} ipython3
from pipefunc import pipefunc, Pipeline

def my_hook(func, result, kwargs):
    print(f"Function {func.__name__} returned {result} with inputs {kwargs}")

@pipefunc(output_name="c", post_execution_hook=my_hook)
def f(a, b):
    return a + b

# The hook will print after each execution
f(a=1, b=2)  # Prints: Function f returned 3 with inputs {'a': 1, 'b': 2}

# Hooks also work in pipelines and with map operations
@pipefunc(output_name="d")
def g(c):
    return c * 2

pipeline = Pipeline([f, g])
pipeline(a=1, b=2)  # Hook is called when f executes in the pipeline
```

Post-execution hooks are particularly useful for:

- Debugging: Print intermediate results and inputs
- Logging: Record function execution details
- Profiling: Collect timing or resource usage statistics
- Validation: Check results or inputs meet certain criteria
- Monitoring: Track pipeline progress

Note that hooks are executed synchronously after the function returns but before the result is passed to the next function in the pipeline.
They should be kept lightweight to avoid impacting performance.

## Running multiple `map` calls concurrently

In some scenarios, you might need to run `pipeline.map` multiple times with different sets of inputs or even with different pipelines.
`pipefunc` provides a convenient way to manage and execute these concurrent map operations, giving you control over the degree of parallelism.
This is particularly useful when dealing with tasks that have varying computational requirements or when you want to orchestrate a series of related but independent parameter sweeps.

The core functions for this are {func}`~pipefunc.helpers.launch_maps` and {func}`~pipefunc.helpers.gather_maps`.

**The workflow is as follows:**

1.  Create a list of `AsyncMap` runners by calling {meth}`~pipefunc.Pipeline.map_async` with `start=False`. This prepares the map operations without immediately executing them.
2.  Pass these runners to `launch_maps` or `gather_maps` to execute them.

- {func}`~pipefunc.helpers.launch_maps`: A non-blocking function ideal for interactive environments like Jupyter. It starts the execution in the background and returns an `asyncio.Task` that you can `await` later.
- {func}`~pipefunc.helpers.gather_maps`: A blocking `async` function that runs the maps and waits for all of them to complete before returning.

Let's see an example:

```{code-cell} ipython3
from pipefunc import Pipeline, pipefunc
from pipefunc.helpers import launch_maps

@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def double_it(x: int) -> int:
    return 2 * x

pipeline = Pipeline([double_it])

# Define two different sets of inputs with different sizes
inputs1 = {"x": [1, 2, 3]}
inputs2 = {"x": [4, 5, 6, 7]}

# 1. Prepare the runners
runners = [
    pipeline.map_async(inputs1, start=False),
    pipeline.map_async(inputs2, start=False),
]

# 2. Launch the maps concurrently
# This will run at most 2 maps at the same time.
task = launch_maps(*runners, max_concurrent=2)
```

In a Jupyter notebook, `launch_maps` will automatically display a tabbed widget to monitor the progress of each map operation.

To get the results, you can `await` the task in a later cell:

```{code-cell} ipython3
# In a new cell
results = await task
print(results[0]["y"].output)
print(results[1]["y"].output)
```

### Controlling Concurrency

The `max_concurrent` parameter in `launch_maps` and `gather_maps` controls how many of the `map` operations are allowed to run at the same time. For example, if you have 10 map operations to run but set `max_concurrent=3`, only three will execute in parallel at any given time.

### Sequential Execution

If you want to run the maps one after another, simply set `max_concurrent=1`.

```python
task = launch_maps(*runners, max_concurrent=1)
```

This is useful when subsequent map operations might depend on the resources freed up by preceding ones, or when you want to avoid overloading a system.

### Why run maps concurrently?

This feature is beneficial in several situations:

- **Heterogeneous Workloads**: When you have map operations with different input sizes or computational costs.
- **Resource Constraints**: When the design of your `pipefunc`s requires that all maps produce arrays of the same shape, but your tasks naturally have different input sizes. Running them as separate map calls allows you to handle this.
- **Complex Workflows**: For orchestrating multiple, independent parameter sweeps as part of a larger computational experiment.

By using `launch_maps`, you can manage these complex scenarios with simple, readable code while retaining fine-grained control over the execution.
