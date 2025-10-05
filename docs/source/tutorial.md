---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  display_name: .venv
  language: python
  name: python3
---

# Tutorial for Pipefunc Package

The pipefunc package is a Python library designed to simplify the creation and execution of function pipelines.
It allows you to define functions as pipeline steps, automatically managing dependencies and execution order.
In this tutorial, we will guide you through the key features of pipefunc, including sequential and parallel execution, map-reduce operations, and advanced functionalities.

+++

This page is [a Jupyter notebook](https://github.com/pipefunc/pipefunc/blob/main/example.ipynb), executed and rendered in [the official documentation](https://pipefunc.readthedocs.io/en/latest/tutorial).

```{try-notebook} example.ipynb
```

+++

## High level overview

1. The pipefunc package allows to create reusable and callable pipelines.
1. A `Pipeline` contains a list of `PipeFunc` objects.
1. At its core, these `PipeFunc` objects only contain a function and an output name.
1. You can create a `PipeFunc` object directly or using the `@pipefunc` decorator.
1. The `Pipeline` will automatically connect all functions based on the output names and function inputs.

+++

---

## Building a Simple Pipeline

Let's start by importing `pipefunc` and `Pipeline` from the `pipefunc` module.

```{code-cell} ipython3
from pipefunc import PipeFunc, Pipeline, pipefunc
```

We then define some functions using the `@pipefunc` decorator.
The `@pipefunc` decorator turns these functions into pipeline steps.
For each function, we specify an `output_name` which will be used to refer to the output of that function in the pipeline.

```{code-cell} ipython3
@pipefunc(output_name="c")
def f(a, b):
    return a + b


@pipefunc(output_name="d")
def g(b, c, x=1):  # "c" is the output of f
    return b * c * x


@pipefunc(output_name="e")
def h(c, d, x=1):  # "d" is the output of g
    return c * d * x
```

We now have three functions `f`, `g`, and `h`, which we can use to build a pipeline.
We create a `Pipeline` object passing the list of functions.
We can also enable debugging, profiling, and caching for the entire pipeline:

```{code-cell} ipython3
pipeline = Pipeline(
    [f, g, h],
    debug=True,  # optionally print debug information
    profile=True,  # optionally profile the pipeline
    cache_type="hybrid",  # optionally cache the pipeline
)
```

Now, we have a pipeline that only requires `a` and `b` as inputs and uses the outputs of the functions and automatically passes them as inputs to the next function.

Don't want to use the `@pipefunc` decorator? No problem! You can create a `PipeFunc` object directly:

```{code-cell} ipython3
@pipefunc(output_name="c")
def f(a, b):
    return a + b


# is equivalent to


def f(a, b):
    return a + b


f = PipeFunc(f, output_name="c")
```


---

## Visualizing the Pipeline

You can visualize your pipeline using the `visualize()` method, and print the nodes in the graph using the `graph.nodes` attribute.

+++

??? note "Interactive visualization with [`graphviz-anywidget`](https://github.com/pipefunc/graphviz-anywidget)"
    In a live Jupyter notebook, the output below allows interaction with the pipeline visualization.

    You will be able to zoom by scrolling, pan by dragging the image, and click on nodes to highlight all connected nodes. Click Escape to reset the view.

```{code-cell} ipython3
print("Graph nodes:", pipeline.graph.nodes)
pipeline.visualize()
```

---

## Executing the Pipeline

There are two ways to execute the pipeline:

1. Call the pipeline as a function (***sequentially***) and get a specific output:
   - `pipeline(output_name, **kwargs)`
   - `pipeline.run(output_name, kwargs)`
2. Evaluate the entire pipeline (***parallel***) including map-reduce operations:
   - `pipeline.map(kwargs)`

We start with calling the pipeline directly and then introduce the `map` method.

See [this documentation page](./concepts/execution-and-parallelism.md#run-vs-map) for more information on the difference between `run` and `map`.

+++

### Using `pipeline(...)` (Sequential Execution)

If the pipeline has a unique leaf node (single final output), then we can directly call the pipeline object with the input arguments.

```{code-cell} ipython3
pipeline(a=1, b=2)
```

```{code-cell} ipython3
# The above returns the output for:
pipeline.unique_leaf_node
```

We can also specify the desired output as the first argument of the pipeline function:

```{code-cell} ipython3
print("`e` is:", pipeline("e", a=1, b=2))
print("`d` is:", pipeline("d", a=1, b=2))
```

### Using `pipeline.run(...)` (Sequential Execution)

Similar to calling the `pipeline` object, we can use the `run` method to execute the pipeline.

> Note: The `pipeline(...)` call is just a wrapper around the `run` method.

```{code-cell} ipython3
result = pipeline.run("e", kwargs={"a": 1, "b": 2})
print(result)
```

or get _*all*_ function outputs and inputs by specifying `full_output=True`:

```{code-cell} ipython3
result = pipeline.run("e", kwargs={"a": 1, "b": 2}, full_output=True)
print(result)
```

### Using `pipeline.map(...)` (Parallel Execution)

+++

`pipeline.map` allows you to execute your pipeline over a set of inputs in parallel.

> **Note:** The `mapspec` argument in the `@pipefunc` decorator defines how inputs are mapped to outputs.

> **Note:** ⚠️ The mapping computation of the pipeline is done in parallel using the `concurrent.futures.ProcessPoolExecutor` whenever `pipeline.map(..., parallel=True)` (default).

```{code-cell} ipython3
@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def double_it(x: int) -> int:
    assert isinstance(x, int)
    return 2 * x


pipeline_double = Pipeline([double_it])

inputs = {"x": [1, 2, 3, 4, 5]}
run_folder = "my_run_folder"  # save the results in this folder
result = pipeline_double.map(inputs, run_folder)
print(result["y"].output)
```

**Syntax of `mapspec`:**

```
input1[i], input2[j] -> output[i, j]
```

- **`i` and `j`** are indices over which the function maps.
- **`input1[i]`** means the function will receive `input1` at index `i`.
- **`output[i, j]`** means the function will produce `output` with indices `i` and `j`.

+++

Instead of defining `mapspec` manually, you can use the `add_mapspec_axis` method on the pipeline object:

```{code-cell} ipython3
# Take `pipeline` defined above and add a 2D mapspec
pipeline2 = pipeline.copy()
pipeline2.debug = False  # Turn off debugging print statements
pipeline2.add_mapspec_axis("a", axis="i")
pipeline2.add_mapspec_axis("b", axis="j")
run_folder = "my_run_folder"
result = pipeline2.map({"a": [1, 2], "b": [3, 4]}, run_folder, show_progress=True)
result["e"].output  # This is now a 2D array
```

The methods above will automatically generate the `mapspec` for you, which is now:

```{code-cell} ipython3
pipeline2.mapspecs_as_strings
```

The `pipeline.map` method is powerful and can handle complex map-reduce operations, which we will demonstrate next.

+++

#### Map-reduce or fan-in / fan-out operations

The script below demonstrates a two-step pipeline: doubling each integer in an input list, followed by summing all the doubled values.

```{code-cell} ipython3
import numpy as np

from pipefunc import Pipeline, pipefunc
from pipefunc.typing import Array


@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def double_it(x: int) -> int:
    assert isinstance(x, int)
    return 2 * x


@pipefunc(output_name="sum")  # no mapspec, so receives y[:] as input
def take_sum(y: Array[int]) -> int:
    assert isinstance(y, np.ndarray)
    return sum(y)


pipeline_map = Pipeline([double_it, take_sum])
pipeline_map.visualize()
```

??? note "What is `mapspec`?"
    In `double_it`, `mapspec="x[i] -> y[i]"` specifies that each element `i` of the input array `x` is independently processed to produce the corresponding element `i` in the output array `y`.
    Because `take_sum` does not have a `mapspec`, it receives the entire array `y` for aggregation.

+++

Note that the mapspecs are present in the plot. For example, `x` is now `x[i]`.

```{code-cell} ipython3
inputs = {"x": [0, 1, 2, 3]}
run_folder = "my_run_folder"
results = pipeline_map.map(inputs, run_folder=run_folder)
```

Check the results in the resulting dict

```{code-cell} ipython3
assert results["y"].output.tolist() == [0, 2, 4, 6]
assert results["sum"].output == 12
```

Or load the outputs from disk

```{code-cell} ipython3
from pipefunc.map import load_outputs

assert load_outputs("y", run_folder=run_folder).tolist() == [0, 2, 4, 6]
assert load_outputs("sum", run_folder=run_folder) == 12
```

Or also load from disk but as an `xarray.Dataset`:

```{code-cell} ipython3
from pipefunc.map import load_xarray_dataset

load_xarray_dataset(run_folder=run_folder)
```

## Advanced features

Below are some advanced features of the `pipefunc` package.
You will find more features in the [FAQ](https://pipefunc.readthedocs.io/en/latest/faq/).

+++

---

### Working with Resources Report

The `print_profiling_stats()` method of the `pipeline` provides useful information on the performance of the functions in the pipeline such as CPU usage, memory usage, average time, and the number of times each function was called.
This feature is only available if `profile=True` when creating the pipeline.

```{code-cell} ipython3
# This will print the number of times each function was called
# CPU, memory, and time usage is also reported
pipeline.print_profiling_stats()
```

This report can be beneficial in performance tuning and identifying bottlenecks in your pipeline. You can identify which functions are consuming the most resources and adjust your pipeline accordingly.

You can also look all the stats directly with:

```{code-cell} ipython3
pipeline.profiling_stats
```

 ---

### Handling Multiple Outputs

Functions can return multiple results at once.
The `output_name` argument allows you to specify multiple outputs by providing a tuple of strings.
By default, this assumes the output is a `tuple`.
However, if you provide a `output_picker` function, you can return any type of object.
As long as the output name can be used to get the desired output from the returned object.

```{code-cell} ipython3
from pipefunc import Pipeline, pipefunc


# Returns 2 outputs as a tuple: 'c' and 'const'.
@pipefunc(output_name=("c", "const"))
def add_ab(a, b):
    return (a + b, 1)


def get_dict_output(output, key):
    return output[key]


# Function that returns a dictionary, output_picker is used
# to pick out "d" and "e".
@pipefunc(output_name=("d", "e"), output_picker=get_dict_output)
def mul_bc(b, c, x=1):
    return {"d": b * c, "e": x}


# Function returns an object with attributes 'g' and 'h'.
# output_picker is used to pick out 'g' and 'h'.
@pipefunc(output_name=("g", "h"), output_picker=getattr)
def calc_cde(c, d, e, x):
    from types import SimpleNamespace

    return SimpleNamespace(g=c * d * x, h=c + e)


# Define a function add_gh with a single output 'i'.
@pipefunc(output_name="i")
def add_gh(e, g):
    return e + g


# Create a pipeline with the defined functions and visualize it.
pipeline_multiple = Pipeline([add_ab, mul_bc, calc_cde, add_gh])
pipeline_multiple.visualize()
```

```{code-cell} ipython3
pipeline_multiple(a=1, b=2, x=3)
```

---

### Using the `renames` Feature

The `renames` attribute in `pipefunc` allows you to rename the inputs and outputs of a function before passing them to the next step in the pipeline.
This can be particularly useful when the same function is used multiple times in a pipeline, or when you want to provide more meaningful names to the inputs and outputs.

In the example below, we demonstrate how to use the `renames` attribute to rename the inputs of a function before they are passed to the next step in the pipeline.

> ⚠️ Instead of using the `@pipefunc` decorator (which creates `pipefunc.PipeFunc` object), we will create `PipeFunc` objects directly and specify the `renames` attribute.

```{code-cell} ipython3
from pipefunc import PipeFunc, Pipeline


def prod(a, b):
    return a * b


def subtract(a, b):
    return a - b


# We're going to use these functions multiple times in the pipeline
functions = [
    PipeFunc(prod, output_name="prod1"),
    PipeFunc(prod, output_name="prod2", renames={"a": "x", "b": "y"}),
    PipeFunc(subtract, output_name="delta1", renames={"a": "prod1", "b": "prod2"}),
    PipeFunc(subtract, output_name="delta2", renames={"a": "prod2", "b": "prod1"}),
    PipeFunc(prod, output_name="result", renames={"a": "delta1", "b": "delta2"}),
]
pipeline_renames = Pipeline(functions)

inputs = {"a": 1, "b": 2, "x": 3, "y": 4}
results = pipeline_renames("result", **inputs)

# Output the results
print("Results:", results)

pipeline_renames.visualize()
```

**Explanation**:

1. **Function Definitions**:

   - `prod(a, b)`: Multiples two numbers and returns the result.
   - `subtract(a, b)`: Subtracts `b` from `a` and returns the result.

2. **Pipeline Construction**:

We are just using the `prod` and `subtract` functions multiple times, but change the names of the inputs and outputs to create a pipeline from it.

+++

One can also apply the renames afterwards using the `update_renames` method. Or even to the entire pipeline, like:

```{code-cell} ipython3
pipeline_renames2 = pipeline_renames.copy()
pipeline_renames2.update_renames(
    {
        "a": "aa",
        "b": "bb",
        "x": "xx",
        "y": "yy",
        "result": "final_result",  # Rename the `output_name` of the last function
    },
    update_from="current",  # update from the current renames, not the original
)
pipeline_renames2(aa=1, bb=2, xx=3, yy=4)
```

Also check out these `Pipeline` methods:

- `Pipeline.update_defaults`
- `Pipeline.update_bound`

and these `PipeFunc` methods:

- `PipeFunc.update_renames`
- `PipeFunc.update_defaults`
- `PipeFunc.update_bound`

---

+++

### Custom Parallelism

By default when `pipeline.map(..., parallel=True)` is used, the pipeline is executed in parallel using the `concurrent.futures.ProcessPoolExecutor`. However, you can also specify a custom executor to control the parallelism of the pipeline execution.

It works with any custom executor that has the `concurrent.futures.Executor` interface, so for example it works with:

- `concurrent.futures.ProcessPoolExecutor`
- `concurrent.futures.ThreadPoolExecutor`
- `ipyparallel.Client().executor()`
- `dask.distributed.Client().get_executor()`
- `mpi4py.futures.MPIPoolExecutor()`
- `loky.get_reusable_executor()`
- `executorlib.SingleNodeExecutor`, `executorlib.SlurmClusterExecutor`, `executorlib.SlurmJobExecutor`, `executorlib.FluxClusterExecutor`, `executorlib.FluxJobExecutor`

To just change the number of cores while using the default executor, use

```{code-cell} ipython3
import datetime
import time
from concurrent.futures import ProcessPoolExecutor

from pipefunc import Pipeline, pipefunc


@pipefunc(output_name="double", mapspec="x[i] -> double[i]")
def double_it(x: int) -> int:
    print(f"{datetime.datetime.now()} - Running double_it for x={x}")
    time.sleep(1)
    return 2 * x


@pipefunc(output_name="half", mapspec="x[i] -> half[i]")
def half_it(x: int) -> int:
    print(f"{datetime.datetime.now()} - Running half_it for x={x}")
    time.sleep(1)
    return x // 2


@pipefunc(output_name="sum")
def take_sum(half: np.ndarray, double: np.ndarray) -> int:
    print(f"{datetime.datetime.now()} - Running take_sum")
    return sum(half + double)


pipeline_parallel = Pipeline([double_it, half_it, take_sum])
inputs = {"x": [0, 1, 2, 3]}
run_folder = "my_run_folder"
executor = ProcessPoolExecutor(max_workers=8)  # use 8 processes
results = pipeline_parallel.map(
    inputs,
    run_folder=run_folder,
    parallel=True,
    executor=executor,
    storage="shared_memory_dict",
)
print(results["sum"].output)
```

> ⚠️ In this pipeline, `double_it` and `half_it` are doubly parallel; both the map is parallel and the two functions are executed at the same time, note the timestamps and the `sleep()` calls.
> See the `visualize()` output to see the structure of the pipeline.

```{code-cell} ipython3
pipeline_parallel.visualize()
```

---

### Combining Pipelines

Different pipelines can be combined into a single pipeline using the `Pipeline.join` method or the `|` operator.

!!! note
 tl;dr
Use `pipeline1 | pipeline2` to join two pipelines.


In cases the output names and arugments do not match up, we can rename the parameters of an entire pipeline using the `update_renames` method.

```{code-cell} ipython3
from pipefunc import Pipeline, pipefunc


@pipefunc(output_name="c")
def f(a, b):
    return a + b


@pipefunc(output_name="d")
def g(b, c, x=1):
    return b + c + x


pl1 = Pipeline([f, g])


@pipefunc(output_name="e")
def h(cc, dd, xx=2):
    return cc + dd + xx


pl2 = Pipeline([h])

# We now have two pipelines, `pl1` and `pl2`, that we want to combine
# into a single pipeline. However, they have different inputs and defaults.
# Let's update the renames and defaults of `pl2` to match `pl1`.
pl2_ = pl2.copy()
pl2_.update_renames({"cc": "c", "dd": "d", "xx": "x"})
pl2_.update_defaults({"x": 1})
combined_pipeline = pl1 | pl2_  # or use `pl1.combine(pl2_)`

combined_pipeline.visualize()
```

```{code-cell} ipython3
# The combined pipeline can now be used as a single pipeline
result = combined_pipeline(a=2, b=3, x=2)
print(result)  # Output: 17
```

Just to see another quick example of combining pipelines (even though it makes no sense to combine these pipelines):

```{code-cell} ipython3
pipeline_silly = pipeline_renames | combined_pipeline
pipeline_silly.visualize()
```

```{code-cell} ipython3
# e.g., if we want to get the output of `result` in the `pipeline` (not the leaf node!):
pipeline_silly("result", a=1, b=2, y=3)
```

---

### Caching Results

To enable caching, simply set the `cache` attribute to `True` for each function.
This can be useful to avoid recomputing results when calling the same function with the same arguments multiple times.

!!! note

Some cache types support shared memory, which means that the cache can be shared between different processes when running in parallel.


```{code-cell} ipython3
@pipefunc(output_name="y", cache=True)
def my_function(a: int, b: int) -> int:
    time.sleep(1)  # Pretend this is a slow function
    print("Function is called!")
    return a + b


# multiple cache_type options are available, e.g., "lru", "hybrid", "disk", and "simple"
pipeline_cache = Pipeline([my_function], cache_type="lru")

# lets call the function 10 times with the same arguments
for _ in range(10):
    pipeline_cache(a=2, b=3)
```

```{code-cell} ipython3
print(f"Cache object: {pipeline_cache.cache}")
pipeline_cache.cache.cache
```

The cache is populated _**even when using parallel execution**_. To see the cache, you can use the `cache` attribute on the pipeline.

!!! note

If calling the pipeline like a function (in contrast to using `pipeline.map`) keys of the cache are always in terms of the root arguments of the pipeline. When using `pipeline.map`, the keys are in terms of the arguments of the function.

The key is constructed from the function name and the (root) arguments passed to the function. If the arguments are not hashable, the `pipefunc.cache.to_hashable` function is used to *attempt* to convert them to a hashable form.


One can also enable caching after the pipeline is created by setting the `cache` attribute to `True` for each function.

```python
for f in pipeline.functions:
    f.cache = True
```

+++

---

### Function Argument Combinations

As we showed in the first example, we can call the functions in the pipeline by either providing the root inputs or by providing the output of the previous function ourselves.

To see all the possible combinations of arguments that can be passed to each function, you can use the `all_arg_combinations` property. This will return a dictionary, with function output names as keys and sets of argument tuples as values.

```{code-cell} ipython3
all_args = pipeline.all_arg_combinations
assert all_args == {
    # means we can call `pipeline("c", a=1, b=2)`
    "c": {("a", "b")},
    # means we can call `pipeline("d", a=1, b=2, x=3)` or `pipeline("d", b=2, c=3, x=4)`
    "d": {("a", "b", "x"), ("b", "c", "x")},
    # means we can call `pipeline("e", a=1, b=2, x=3)` or `pipeline("e", b=2, d=3, x=4)`, etc.
    "e": {("a", "b", "x"), ("a", "b", "d", "x"), ("b", "c", "x"), ("c", "d", "x")},
}
# We can get root arguments for a specific function
assert pipeline.root_args("e") == ("a", "b", "x")
```

---

### More `mapspec` Examples

This section shows additional `mapspec` examples.

+++

#### Cross-product of two inputs

This example shows how to compute the outer product of two input vectors (`x` and `y`) and then aggregate the resulting matrix along rows, and finally reduce the computation to a single `float` by taking the `norm` of the resulting `aggregated` vector.

```{code-cell} ipython3
from pipefunc import Pipeline, pipefunc


@pipefunc(output_name="z", mapspec="x[i], y[j] -> z[i, j]")
def multiply_elements(x: int, y: int) -> int:
    """Multiply two integers."""
    return x * y


@pipefunc(output_name="aggregated", mapspec="z[i, :] -> aggregated[i]")
def aggregate_rows(z: np.ndarray) -> np.ndarray:
    """Sum the elements of each row in matrix z."""
    return np.sum(z)


@pipefunc(output_name="norm")
def compute_norm(aggregated: np.ndarray) -> float:
    """Compute the Euclidean norm of the vector aggregated."""
    return np.linalg.norm(aggregated)


pipeline_norm = Pipeline([multiply_elements, aggregate_rows, compute_norm])
inputs = {"x": [1, 2, 3], "y": [4, 5, 6]}
results = pipeline_norm.map(inputs, run_folder="my_run_folder")
print("Norm of the aggregated sums:", results["norm"].output)
```

```{code-cell} ipython3
pipeline_norm.visualize()
```

**Explanation**:

1. **Matrix Creation (`multiply_elements`)**:

   - Each combination of elements from arrays `x` and `y` is multiplied to form the matrix `z`. The `mapspec` `"x[i], y[j] -> z[i, j]"` ensures that every pair of elements is processed to generate a 2D matrix.

2. **Row Aggregation (`aggregate_rows`)**:

   - The matrix `z` is then processed row by row to sum the values, creating an aggregated result for each row. The `mapspec` `"z[i, :] -> aggregated[i]"` directs the pipeline to apply the summation across each row, transforming a 2D array into a 1D array of row sums.

3. **Vector Norm Calculation (`compute_norm`)**:
   - Finally, the norm of the aggregated vector is computed, providing a single scalar value that quantifies the magnitude of the vector formed from row sums. This step does not require a `mapspec` as it takes the entire output from the previous step and produces a single output.

+++


#### Dynamic Output Shapes and `internal_shapes`

In most cases, `pipefunc` automatically infers the output shape of each function based on the `mapspec` and the input shapes.
However, use the `internal_shapes` argument if **a function returns an iterable/array that the next function will iterate over using a `mapspec`.**
The most common case is when the `mapspec` of the first function is `... -> output1[i]` and the `mapspec` of the second function is `output1[i] -> output2[i]`.

**How to use `internal_shapes`:**

1. Provide a tuple in `@pipefunc(internal_shape=(...))` representing the shape of the output of that function. You can use `?` for unknown dimensions.
2. Provide a dictionary in `pipeline.map(internal_shapes={...})` where keys are function output names, and values are tuples representing the shape *added* by that function. You can use `?` for unknown dimensions.
3. Or omit `internal_shapes` and let `pipefunc` infer the shapes automatically (missing out on some consistency checks).

**Minimal example:**

```python
@pipefunc(output_name="x", internal_shape=(10, 20))  # or `internal_shape=("?", "?")`
def generate_ints() -> np.ndarray:
    return np.ones((10, 20))

# or

pipeline.map(..., internal_shapes={"x": (10, 20)})  # or `internal_shapes={"x": ("?", "?")}`
```

**Full example:**

We generate a list of integers with a length determined by an input parameter `n`.

```{code-cell} ipython3
from pipefunc import Pipeline, pipefunc
from pipefunc.typing import Array


@pipefunc(output_name="x")
def generate_ints(n: int) -> list[int]:
    """Generate a list of integers from 0 to n-1."""
    return list(range(n))


@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def double_it(x: int) -> int:
    """Double the input integer."""
    return 2 * x


@pipefunc(output_name="sum")
def take_sum(y: Array[int]) -> int:
    """Sum a list of integers."""
    return sum(y)


pipeline_sum = Pipeline([generate_ints, double_it, take_sum])
```

Here, `generate_ints` creates a list of length `n`.
In the function `double_it`, we map over the resulting list and double each element.
Note that PipeFunc automatically generated the `mapspec="... -> x[i]"` for `generate_ints`, which means that the output is an array with index `i` that can be mapped over in the `double_it` function.

We indicate that the output is a 1D array with an unknown number of elements by doing either:

1. setting the `internal_shape` argument of the `generate_ints` decorator to `@pipefunc(output_name="x", internal_shapes="?")`, or
2. by providing a dictionary to the `internal_shapes` argument in `pipeline.map`:

Using option 2:

```{code-cell} ipython3
inputs = {"n": 4}
results = pipeline_sum.map(
    inputs,
    run_folder="my_run_folder",
    internal_shapes={"x": ("?",)},  # Or if we know the shape of the output `{"x": (4,)}`
)
print("Sum of doubled integers:", results["sum"].output)
```

Or we can omit the `internal_shapes` argument and let `pipefunc` infer the shapes automatically:

```{code-cell} ipython3
results = pipeline_sum.map(inputs, run_folder="my_run_folder")
print("Sum of doubled integers:", results["sum"].output)
```



#### Zipped inputs

This pipeline processes zipped inputs `x` and `y` with independent `z` to compute a function across all combinations, producing a 2D matrix `r`.

```{code-cell} ipython3
from pipefunc import Pipeline, pipefunc


@pipefunc(output_name="r")
def process_elements(x: int, y: int, z: int) -> float:
    return x * y + z


pipeline_proc = Pipeline([(process_elements, "x[a], y[a], z[b] -> r[a, b]")])

inputs = {"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8]}

results = pipeline_proc.map(inputs, run_folder="my_run_folder")
output_matrix = results["r"].output
print("Output Matrix:\n", output_matrix)
```

**Explanation**:

- **Function `process_elements`**:

  - Takes three inputs: `x`, `y`, and `z`. For each pair `(x[a], y[a])`, the function is applied with each `z[b]`.

- **Pipeline Definition**:

  - The `mapspec` `"x[a], y[a], z[b] -> r[a, b]"` specifies how elements from the inputs are to be combined. It states that each element from the paired inputs `x` and `y` (indexed by `a`) should be processed with each element from `z` (indexed by `b`), resulting in a 2D output array `r`.

- **Outputs**:
  - The output `r` is a 2-dimensional matrix where the dimensions are determined by the lengths of `x`/`y` and `z`. Each element of this matrix represents the computation result for a specific combination of inputs.

+++

---

### Nesting Pipelines for Modularity and Reusability

`pipefunc` allows you to create modular and reusable pipeline components by nesting pipelines within each other using the `pipefunc.NestedPipeFunc` class or the ``pipefunc.Pipeline.nest_funcs`` method. This is particularly useful for:

- **Encapsulating** a sequence of steps that logically belong together.
- **Reusing** a part of a pipeline in multiple projects or within a larger pipeline.
- **Abstracting** away internal details of a complex sub-process.
- **Selectively avoid returning** intermediate results when using `pipeline.map` (e.g., to prevent serializing large objects and passing it around).

**Creating Nested Pipelines:**

You can manually create a ``pipefunc.NestedPipeFunc`` by passing a list of functions to its constructor.
However, a potentially more convenient way is to use the `pipefunc.Pipeline.nest_funcs` method, which allows you to combine existing functions within a pipeline into a nested one:

```{code-cell} ipython3
from pipefunc import Pipeline, pipefunc


@pipefunc(output_name="c")
def f1(a, b):
    return a + b


@pipefunc(output_name="d")
def f2(c):
    return c * 2


@pipefunc(output_name="e")
def f3(d, x):
    return d + x


pipeline = Pipeline([f1, f2, f3])

# Nest f1 and f2 into a single NestedPipeFunc
nested_pipeline = pipeline.copy()
nested_func = nested_pipeline.nest_funcs(
    {"c", "d"},
    new_output_name="d",  # Only returns "d" and not "c"
    function_name="f1_f2",
)
nested_pipeline.visualize()
```

This creates a `nested_pipeline` where `f1` and `f2` are combined into a `NestedPipeFunc` named `f1_f2`.
The new nested function only returns `"d"` and not `"c"`.
The `new_output_name` must be a subset of the outputs of the nested pipeline.
You can optionally specify the name of the function using the `function_name` argument.

**Inspecting the Nested Pipeline:**

The `nested_func` object contains its own internal pipeline, accessible via the `pipeline` attribute:

```{code-cell} ipython3
nested_func.pipeline.visualize()
```

**Using the Nested Pipeline:**

You can now use the `nested_pipeline` like any other pipeline. When executed, the `NestedPipeFunc` will run its internal pipeline, taking the required inputs and producing the specified output.

```{code-cell} ipython3
result = nested_pipeline(a=1, b=2, x=3)
print(f"{result=}")
nested_result = nested_func(a=1, b=2)
print(f"{nested_result=}")
```

**Limitations with `mapspec`:**

While `NestedPipeFunc` offers powerful modularity, there are limitations when using it with `mapspec`:

- **No Map-Reduce Operations:** The `mapspec` of functions within a `NestedPipeFunc` **cannot** contain reductions (e.g., `x[i, j] -> y[i]`).
- **No Dynamic Axis Generation:** The `mapspec` **cannot** dynamically generate new axes (e.g., `... -> out[i]`). In other words, it cannot return an output with an `internal_shape`.
- **Allowed `mapspec`s:** You can use mapspecs that do not reduce or create new axes.
- **Bound arguments:** The `bound` arguments do not appear as parameters to the nested pipeline. To update the bound arguments, use `nested_func.pipeline["output_name"].update_bound(...)`.

These limitations stem from the fact that the nested pipeline is treated as a single unit, and its internal operations are not directly exposed to the outer pipeline's mapping logic.

**Benefits of Nesting (Despite Limitations):**

- **Modularity:** Create self-contained, reusable pipeline components.
- **Abstraction:** Hide internal complexity behind a well-defined interface.
- **Reusability:** Easily integrate nested pipelines into other projects or larger workflows.
- **Clarity:** Improve the overall structure and readability of your pipelines.
- **Control over intermediate results:** When using `pipeline.map`, use `nest_funcs` to avoid returning intermediate results.

+++

## Full Examples


### Example: Physics based example

This section has been moved to the [Physics based example](./examples/physics-simulation.md) page.

### Example: Sensor Data Processing Pipeline

This section has been moved to the [Sensor Data Processing Pipeline](./examples/sensor-data-processing.md) page.

### Example: Image Processing Workflow Example with `mapspec`

This section has been moved to the [Image Processing Workflow Example with `mapspec`](./examples/image-processing.md) page.

### Example: Natural Language Processing Pipeline for Text Summarization

This section has been moved to the [Natural Language Processing Pipeline for Text Summarization](./examples/nlp-text-summarization.md) page.

### Example: Weather Simulation and Analysis Pipeline with `xarray`

This section has been moved to the [Weather Simulation and Analysis Pipeline with `xarray`](./examples/weather-simulation.md) page.
