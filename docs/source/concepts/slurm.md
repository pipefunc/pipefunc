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

# SLURM Integration with PipeFunc

PipeFunc integrates with SLURM on a high level without requiring you to write `sbatch` scripts or manage job submission manually.

PipeFunc allows you to submit jobs to a SLURM cluster using the {class}`adaptive_scheduler.SlurmExecutor` and set its resources for individual functions using the `resources` parameter in `@pipefunc(..., resources=...)`.
The `resources` parameter can be a dictionary or a {class}`~pipefunc.resources.Resources` object.
This tutorial explains how to:

- **Submit jobs using SlurmExecutor.**
- **Specify Resources statically or dynamically.**
- **Control resource allocation scopes** so that resources apply either to the entire mapspec or to each individual iteration.
- **Have functions manage their own parallelization** when running a SLURM job in sequential mode.
- **Use a dictionary of executors** to run some functions via SLURM while others run on a {class}`concurrent.futures.ThreadPoolExecutor`.

## tl;dr

```python
from pipefunc import Pipeline, pipefunc
from adaptive_scheduler import SlurmExecutor

@pipefunc(output_name="y", resources={"cpus": 2})  # SLURM job with 2 CPUs.
def double_it(x: int) -> int:
    return 2 * x

pipeline = Pipeline([double_it])
runner = pipeline.map_async(
    {"x": 1},
    run_folder="my_run_folder",
    executor=SlurmExecutor(),  # Run on SLURM.
)
result = await runner.task
```

## Overview of SLURM Integration

When running pipelines on a SLURM cluster, each function (or each iteration, see [next section](#resource-allocation-scopes)) is submitted as a separate job with the desired compute resources.
The two sources of resource specifications are:

1. **SlurmExecutor Defaults:**
   When you instantiate a SlurmExecutor (e.g. `SlurmExecutor(cores_per_node=2)`), you set default values. These defaults serve as a fallback for any function that does not override the resources.

2. **Function-Specific Resources:**
   When decorating your function with PipeFunc, you can pass `resources={"cpus": 2, ...}` (or a {class}`~pipefunc.resources.Resources` object). These values will overwrite the executor defaults for that function’s job submission.

This design ensures flexibility: you have a cluster-wide default configuration that can be tailored on a per-function basis.

:::{admonition} `resources=Resouces(...)` vs `resources={...}`
:class: important

The {class}`~pipefunc.resources.Resources` object provided (either as a dict or via `Resources(...)`) must have keys matching those defined by the Resources class (such as `"cpus"`, `"memory"`, etc.).
:::

---

## Resource Allocation Scopes

PipeFunc supports two modes for SLURM job submission when using mapspec.
The **resources_scope** parameter (in `@pipefunc(..., resources_scope=...)`) controls how the {class}`~pipefunc.resources.Resources` provided are applied:

- **Map Scope (`resources_scope=="map"`):**
  The entire mapspec operation for the function is submitted as a single SLURM job using the provided resources.

- **Element Scope (`resources_scope=="element"`):**
  A separate SLURM job is submitted for each iteration of the mapspec.

This distinction is essential when planning job submissions and balancing overhead versus resource granularity.

---

## Using SlurmExecutor

To run your pipeline on a SLURM cluster, instantiate a `SlurmExecutor` and pass it as the executor in your `Pipeline.map_async` call. For example:

```python

from pipefunc import Pipeline, pipefunc
from adaptive_scheduler import SlurmExecutor

@pipefunc(output_name="y", mapspec="x[i] -> y[i]", resources={"cpus": 2})
def double_it(x: int) -> int:
    return 2 * x

@pipefunc(
    output_name="z",
    mapspec="y[i] -> z[i]",
    # Dynamically set resources based on input "x".
    # Note: only parameters available in the function signature can be used,
    # so here we rely on "y" which is the input to this function.
    resources=lambda kwargs: {"cpus": int(kwargs["y"] % 3) + 1},
    resources_scope="element",  # One SLURM job per iteration in the mapspec.
)
def add_one(y: int) -> int:
    return y + 1

@pipefunc(output_name="z_sum")
def sum_it(z):
    return sum(z)

pipeline = Pipeline([double_it, add_one, sum_it])
inputs = {"x": range(10)}
```

### Submitting with SLURM

Pass a `SlurmExecutor` to your asynchronous mapping call:

```python
# Create a SlurmExecutor.
# The cores_per_node parameter specifies default resources.
executor = SlurmExecutor(cores_per_node=2)

# Submit the pipeline.
runner = pipeline.map_async(
    inputs,
    run_folder="my_run_folder",
    executor=executor,  # This uses SLURM to run the jobs.
    show_progress=True,
)
result = await runner.task  # Await asynchronous execution.
```

In this example, functions with `resources_scope=="element"` submit one job per iteration while those with `resources_scope=="map"` submit one job for the entire mapspec operation.
The resources specified on the function override the executor’s defaults for that job.

---

## Setting Resources Statistically and Dynamically

### Static Resources

You can specify a fixed resource requirement by passing a dictionary or a {class}`~pipefunc.resources.Resources` instance when decorating your function:

```python
@pipefunc(output_name="geo", resources={"cpus": 4, "memory": "8GB"})
def make_geometry(x: float, y: float) -> Geometry:
    # Create and return a geometry object.
    return Geometry(x, y)
```

Every time `make_geometry` is executed, SLURM will allocate 4 CPUs and 8GB of memory.

### Dynamic Resources

If your resource needs depend on the input values, pass a callable that returns a resource dictionary.
For instance:

```python
@pipefunc(
    output_name="electrostatics",
    mapspec="V_left[i], V_right[j] -> electrostatics[i, j]",
    # Dynamically set resources based on inputs: e.g., more voltage difference requires more CPUs.
    resources=lambda kw: {"cpus": abs(kw["V_left"] - kw["V_right"]) + 1},
    resources_scope="element",
)
def run_electrostatics(mesh, materials, V_left: float, V_right: float):
    # Run the simulation.
    return Electrostatics(mesh, materials, [V_left, V_right])
```

Here, for each iteration of the simulation, the lambda inspects the voltage values and returns a dictionary specifying the number of CPUs needed.

---

## Functions Managing Their Own Parallelization

Sometimes you want the SLURM job to have resources allocated while allowing your function to control its own parallelism internally.
You can achieve this by running the SLURM job in sequential mode.
In this mode, the SLURM job is submitted with the allocated {class}`~pipefunc.resources.Resources`, but the function is executed sequentially; it is then up to your function to manage internal parallelization (for example, by creating its own process pool).

```python
from adaptive_scheduler import SlurmExecutor

# Create a SlurmExecutor in "sequential" mode.
executor = SlurmExecutor(cores_per_node=2, executor_type="sequential")

@pipefunc(
    output_name="data_processed",
    resources={"cpus": 2, "memory": "4GB"},
    resources_variable="resources",
)
def process_data(data, resources):
    # The SLURM job will be allocated 2 CPUs and 4GB memory.
    # The Resources object is available as `resources`.
    print(f"Allocated Resources: {resources}")
    # The function can now manage its own parallelism (e.g. via a process pool).
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=resources.cpus) as executor:
        return list(executor.map(myfunc, data))

pipeline = Pipeline([process_data])
result = pipeline(data=[1, 2, 3])
```

Inside `process_data`, the {class}`~pipefunc.resources.Resources` object is injected via `resources_variable` so you can adapt the internal behavior based on the allocated resources.

---

## Using a Dictionary of Executors

For pipelines where different functions have very different runtimes, you can provide a dictionary of executors so that fast functions run with minimal overhead (using a {class}`~concurrent.futures.ThreadPoolExecutor`) while slower, resource‐intensive functions run on SLURM.

For example:

```python
from concurrent.futures import ThreadPoolExecutor
from adaptive_scheduler import SlurmExecutor

@pipefunc(output_name="a", mapspec="x[i] -> a[i]", resources={"cpus": 2})
def slow_function(x: int) -> int:
    # Simulate a slow, resource-intensive computation.
    import time
    time.sleep(1)
    return x * 10

@pipefunc(output_name="b", mapspec="a[i] -> b[i]")
def fast_function(a: int) -> int:
    # Simulate a fast computation that does not require heavy resources.
    return a + 5

pipeline = Pipeline([slow_function, fast_function])
inputs = {"x": range(5)}

# Create a dictionary of executors:
# - Run slow_function (output "a") using SlurmExecutor.
# - Run fast_function (output "b") using ThreadPoolExecutor.
# - The empty key ("") is the default if a function's output name is not explicitly specified.
executors = {
    "a": SlurmExecutor(),
    "b": ThreadPoolExecutor(),
    "": SlurmExecutor(cores_per_node=1),  # default executor
}

runner = pipeline.map_async(inputs, run_folder="executor_example", executor=executors)
result = await runner.task

print("Result from slow_function (a):", result["a"].output)
print("Result from fast_function (b):", result["b"].output)
```

In this example, the dictionary `executors` assigns:

- **Output "a"** (from `slow_function`) to a SLURM job.
- **Output "b"** (from `fast_function`) to a {class}`~concurrent.futures.ThreadPoolExecutor` for lower overhead.

This hybrid approach allows you to optimize the overall pipeline execution by assigning appropriate executors based on the function’s runtime characteristics.

---

## Summary

- **SlurmExecutor vs. Function Resources:**
  - **SlurmExecutor defaults** (e.g. `cores_per_node`) set cluster-wide defaults.
  - **Function-specific Resources** (via `resources={...}` or `Resources()`) override these defaults for that function.
- **Resource Allocation Scopes:**
  - `"map"` scope applies the provided resources to the entire mapspec operation (one job per function).
  - `"element"` scope applies the resources to each iteration (one job per element).
- **Setting Resources:**
  - **Statically:** Pass a fixed dict or Resources instance.
  - **Dynamically:** Pass a callable that returns the resource dict (using only parameters available in the function signature).
- **Functions Managing Their Own Parallelization:**
  Use `executor_type="sequential"` in SlurmExecutor so that while SLURM allocates the requested resources, your function can handle internal parallelism. The `resources_variable` parameter injects the allocated Resources object.
- **Dictionary of Executors:**
  You can pass a dict to the `executor` argument so that different functions run on different executors (for example, fast functions can use a ThreadPoolExecutor, while heavy ones run on SLURM).
