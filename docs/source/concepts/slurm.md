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
- **Have functions manage their own parallelization** when running a SLURM job.
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

### Splitting a `map` into Multiple Jobs

By default, `resources_scope="map"` creates a single SLURM job that processes all items in the input iterable, using internal parallelization (e.g., a `ProcessPoolExecutor` if `executor_type="process-pool"` (default)) based on the `cpus` requested.

However, for very large iterables, you might want to split the work into a fixed number of SLURM jobs instead of one giant job.
You can achieve this by setting the `size_per_learner` argument when creating the `SlurmExecutor`.
This will chunk the input iterable and create one job per chunk.

**Example:**

Suppose you want to process 100 items, but you want to split them into two SLURM jobs, each handling 50 items.

```python
from pipefunc import Pipeline, pipefunc
from adaptive_scheduler import SlurmExecutor

@pipefunc(
    output_name="y",
    mapspec="x[i] -> y[i]",
    resources={"cpus": 8},
    resources_scope="map",  # Important: this must be "map" (default)
)
def process_item(x):
    return x + 1

pipeline = Pipeline([process_item])

# By setting size_per_learner=50, adaptive_scheduler will create
# two jobs for the 100 items in the input.
executor = SlurmExecutor(size_per_learner=50)

runner = pipeline.map_async(
    inputs={"x": range(100)},
    run_folder="my_run_folder",
    executor=executor,
)
# result = await runner.task
```

In this scenario:

- `pipefunc` sees a `map` scope and tells `adaptive_scheduler` to process `range(100)`.
- The `SlurmExecutor`'s `size_per_learner=50` argument instructs it to break the 100 items into chunks of 50.
- This results in **two** SLURM jobs being submitted.
- Each job is allocated 8 CPUs and will process its 50 items using an internal `ProcessPoolExecutor` with 8 workers.

This gives you fine-grained control over job submission without changing your function's code.

---

## Functions Managing Their Own Parallelization

Sometimes you want the SLURM job to have resources allocated while allowing your function to control its own parallelism internally.
You can achieve this by specifying `parallelization_mode="internal"` in {class}`~pipefunc.resources.Resources`.
In this mode, the SLURM job is submitted with the allocated resources, but the function is executed sequentially; it is then up to your function to manage internal parallelization (for example, by creating its own process pool).

```python
from adaptive_scheduler import SlurmExecutor

executor = SlurmExecutor()

@pipefunc(
    output_name="data_processed",
    resources={"cpus": 2, "memory": "4GB"},
    resources_variable="resources",
    parallelization_mode="internal",  # Manage parallelism internally.
)
def process_data(data, resources):
    # The SLURM job will be allocated 2 CPUs and 4GB memory.
    # The Resources object is available as `resources`.
    print(f"Allocated Resources: {resources}")
    # The function can now manage its own parallelism (e.g. via a process pool).
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(resources.cpus) as ex:
        # Do some printing in parallel (not smart, but just to show the parallelization)
        list(ex.map(print, data))
    return sum(data)

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

## Handling Large Inputs with `SlurmExecutor`

When using `pipeline.map_async` with `SlurmExecutor`, especially for operations involving large input data, directly passing these inputs can be inefficient.
Each distributed task (Slurm job) might end up pickling and transferring a copy of the entire large input, even if it only needs a small part or if the same large input is needed by many jobs.
This can lead to:

- Slow task submission and increased overhead.
- High memory usage on the node submitting the jobs.

To mitigate this, `pipefunc` offers utilities to pre-serialize large inputs to a shared disk location.
The lightweight reference objects are then passed to `pipeline.map_async`.

### 1. Large Array-Like Inputs (for `MapSpec` iteration)

For large array-like inputs (e.g., lists of objects, large NumPy arrays) that are processed element-wise a `MapSpec`, use {class}`~pipefunc.helpers.FileArray.from_data`.

**How it works:**

1.  **Pre-serialize:** Save your large input data to a directory using `FileArray.from_data(data, folder)`. This creates a `FileArray` structure where your data is stored on disk, with each element (or chunk) typically as a separate file.
2.  **Pass Lightweight Object:** Pass the `FileArray` instance (small, containing path and shape metadata) in the `inputs` dictionary to `pipeline.map_async`.
3.  **Efficient Worker Access:** When a `MapSpec` like `x[i] -> y[i]` processes this `FileArray` (where `x` is the `FileArray`), `pipefunc`'s internal `MapSpec` machinery invokes `FileArray[(i,)]`. This method efficiently loads only the specific element `i` from disk _on the worker node_.

**Example:**

```python
from pathlib import Path
import numpy as np
from pipefunc import Pipeline, pipefunc
from pipefunc.helpers import FileArray # Import FileArray
from adaptive_scheduler import SlurmExecutor # Assuming SlurmExecutor is used

large_array_data = np.arange(1_000_000)
shared_input_folder = Path("/mnt/shared/my_large_array_data") # Accessible by Slurm nodes
shared_input_folder.mkdir(parents=True, exist_ok=True)
file_array_input = FileArray.from_data(large_array_data, shared_input_folder)

@pipefunc(output_name="y", mapspec="x[i] -> y[i]", resources={"cpus": 1}, resources_scope="element")
def process_element(x: int) -> int:
    return x * 2

pipeline = Pipeline([process_element])
run_folder = Path("/mnt/shared/my_run_output_slurm") # Also shared

runner = pipeline.map_async(
    inputs={"x": file_array_input},
    run_folder=run_folder,
    executor=SlurmExecutor(),
)
# results = await runner.task
```

### 2. Single Large Non-Iterable Inputs

For single, potentially large, non-iterable inputs (e.g., a large configuration dictionary, a pre-trained model object) that need to be passed to multiple jobs without re-serializing each time, use {class}`~pipefunc.helpers.FileValue.from_data`.

**How it works:**

1.  **Pre-serialize:** Save your single large object to a file using `FileValue.from_data(data, path)`.
2.  **Pass Lightweight Object:** Pass the `FileValue` instance (small, containing the file path) in the `inputs` dictionary.
3.  **Efficient Worker Access:** `pipefunc` will automatically load the object from the referenced file on the worker node just before it's needed by your function.

**Example:**

```python
from pathlib import Path
from pipefunc import Pipeline, pipefunc, resources
from pipefunc.helpers import FileValue # Import FileValue
from adaptive_scheduler import SlurmExecutor

large_config_object = {"param_a": "value_a", "data": list(range(100_000))}
shared_file_path = Path("/mnt/shared/my_large_config.pkl") # Accessible by Slurm nodes
file_ref_input = FileValue.from_data(large_config_object, shared_file_path)

@pipefunc(output_name="y", resources={"cpus": 1})
def use_config(config: dict, task_id: int) -> str:
    # config will be the loaded large_config_object
    return f"Task {task_id} processed config with {len(config.get('data', []))} items."

pipeline = Pipeline([use_config])
run_folder = Path("/mnt/shared/my_single_obj_run")

# Example: Pass the same large config to multiple "tasks" simulated by a mapspec on task_id
# In a real scenario, 'config' might be a non-mapspec'd input to several parallel mapspec functions.
pipeline.add_mapspec_axis("task_id", axis="i") # Make task_id a mapspec input

runner = pipeline.map_async(
    inputs={"config": file_ref_input, "task_id": range(5)},
    run_folder=run_folder,
    executor=SlurmExecutor(cores_per_node=1),
)
# results = await runner.task
```

**Important Considerations for Both Methods:**

- **Shared Filesystem:** The `folder` (for `FileArray.from_data`) or `path` (for `FileValue.from_data`) and the `run_folder` for `pipeline.map_async` **must be on a filesystem accessible by all Slurm worker nodes** with the _exact same path_.
- **Non-MapSpec Functions:**
  - If a `FileArray` is passed as an input to a function _not_ iterated by a `MapSpec`, `FileArray.to_array()` will load the entire array into memory on the worker.
  - `FileValue` is primarily for non-MapSpec'd inputs or inputs where the entire object is needed per MapSpec iteration.
- **Data Locality & Filesystem Performance:** While serialization overhead is reduced, data still needs to be read from disk by the workers. The performance of your shared filesystem will be a factor.

By using these techniques, you can significantly improve the performance and scalability of your distributed `pipefunc` workflows on SLURM when dealing with large inputs.

---

## Summary

- **SlurmExecutor vs. Function Resources:**
  - **SlurmExecutor defaults** (e.g. `cores_per_node`) set cluster-wide defaults.
  - **Function-specific Resources** (via `resources={...}` or `Resources()`) override these defaults for that function.
- **Resource Allocation Scopes:**
  - `"map"` scope applies the provided resources to the entire mapspec operation (one job per function).
  - `"element"` scope applies the resources to each iteration (one job per element).
- **Setting Resources:**
  - **Statically:** Pass a fixed dict or `Resources` instance.
  - **Dynamically:** Pass a callable that returns the resource dict (using only parameters available in the function signature).
- **Functions Managing Their Own Parallelization:**
  Use `parallelization_mode="internal"` in {class}`~pipefunc.resources.Resources` so that while SLURM allocates the requested resources, your function can handle internal parallelism. The `resources_variable` parameter injects the allocated `Resources` object.
- **Dictionary of Executors:**
  You can pass a dict to the `executor` argument so that different functions run on different executors (for example, fast functions can use a ThreadPoolExecutor, while heavy ones run on SLURM).
- **Handling Large Input Arrays:**
  Use `FileArray.from_data()` to pre-serialize large input arrays to disk. This allows you to pass a lightweight `FileArray` object to `pipeline.map_async`, which efficiently loads only the required elements on the worker nodes.

---

## Alternative SLURM Submission Using Adaptive Learners

```{note}
We will assume familiarity with the `adaptive` and `adaptive_scheduler` packages in this section.
```

In addition to using the {class}`adaptive_scheduler.SlurmExecutor` for SLURM job submissions (as described above), PipeFunc also supports an alternative method based on creating Adaptive Learners first and then submitting to SLURM via Adaptive Scheduler.
This approach is particularly well suited for _really_ big sweeps where communication overhead between SLURM jobs and the local notebook kernel might become a bottleneck.

### How It Works

Instead of submitting each function call via an executor that mimics local execution, you can use the {func}`pipefunc.map.adaptive.create_learners` function to convert your pipeline into a dictionary of {class}`adaptive.SequenceLearner` objects.
These learners are then submitted to SLURM.

The key advantages (👍) of this approach include:

- **Independent Data Handling:**
  Each learner is responsible for its own data, meaning that the simulation outputs and intermediate results are handled entirely within the SLURM jobs themselves. This reduces the communication overhead that can occur when large amounts of data are transferred back and forth.

- **Scalability for Huge Sweeps:**
  This approach minimizes central communication between the notebook that starts the jobs and the jobs themselves, and allows the jobs to scale more efficiently.

- **Independent Branch Execution:**
  When using the `split_independent_axes=True` option, independent branches in the computational graph can progress on their own. This means that parts of your pipeline that do not depend on each other are not forced to wait, further reducing overall computation time.

The disadvantages (👎) of this approach include:

- **Complexity:**
  The setup is more complex than using the `SlurmExecutor` directly, as you need to manage the Adaptive Learners and the SLURM submission separately, instead of changing _only_ the executor in `pipeline.map_async`.

- **No Dynamic `internal_shapes`:**
  The {func}`~pipefunc.map.adaptive.create_learners` function does not support dynamic `internal_shapes` (as introduced [here](../tutorial.md#dynamic-output-shapes-and-internal-shapes)), so you need to manually specify `internal_shapes` if needed.

### Using Adaptive Learners with SLURM

Below is an example of how you can integrate this approach into your workflow.
In this case, the pipeline is first converted into a dictionary of Adaptive Learners using `create_learners`.
Then, these learners can be submitted to a SLURM cluster via your favorite SLURM submission mechanism (e.g., using Adaptive Scheduler).

```{code-cell} ipython3
from pipefunc import Pipeline, pipefunc
from pipefunc.map.adaptive import create_learners
import numpy as np

# Example pipeline definition
@pipefunc(output_name="double", mapspec="x[i] -> double[i]", resources={"cpus": 2})
def double_it(x: int) -> int:
    return 2 * x

@pipefunc(output_name="half", mapspec="x[i] -> half[i]", resources={"memory": "8GB"})
def half_it(x: int) -> int:
    return x // 2

@pipefunc(output_name="sum")
def take_sum(half: np.ndarray, double: np.ndarray) -> int:
    return sum(half + double)

# Create a pipeline with three functions
pipeline_adapt = Pipeline([double_it, half_it, take_sum])

# Define the input parameters for the sweep
inputs = {"x": [0, 1, 2, 3]}

# Create Adaptive Learners from the pipeline.
# The `split_independent_axes=True` option allows independent branches
# in the computational graph to run separately.
learners_dict = create_learners(
    pipeline_adapt,
    inputs,
    run_folder="my_run_folder",
    split_independent_axes=True,
)

# Now, learners_dict is a dictionary of adaptive learners that can be submitted to SLURM.
# For example, you can extract SLURM submission parameters from the learners:
kwargs = learners_dict.to_slurm_run(
    returns="kwargs",  # Returns a dictionary of SLURM submission keyword arguments
    default_resources={"cpus": 2, "memory": "8GB"},
)

# The resulting `kwargs` can be passed to `slurm_run`
# adaptive_scheduler.slurm_run(**kwargs)
```

### Choosing Between the Two Methods

- **Using `SlurmExecutor`:**
  This method is ideal when you want your SLURM job submission to closely mimic local execution. The results will be organized in the same data structures as you would expect from a local run, and it is straightforward to use with your existing pipelines.

- **Using Adaptive Learners (`create_learners`):**
  This alternative approach is better suited for extremely large sweeps or when you have independent branches in your computational graph. It offloads all data handling to the SLURM jobs themselves, thereby minimizing communication overhead.

By choosing the method that best fits your computational workload and cluster environment, you can optimize both performance and resource utilization when running large-scale simulations on SLURM.
