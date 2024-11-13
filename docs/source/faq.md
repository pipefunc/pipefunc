---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# ❓ FAQ: Frequently Asked Questions

```{contents} ToC – Questions
:depth: 3
```

Missing something or is something unclear? Please [open an issue](https://github.com/pipefunc/pipefunc/issues/new)!

## How to handle defaults?

You can provide defaults in

- The original function definition (the normal way)
- The `pipefunc` decorator `@`{class}`pipefunc.pipefunc``(..., defaults={...})`
- Update the defaults of a `PipeFunc` object (a wrapped function) via {class}`pipefunc.PipeFunc.update_defaults``({...})`
- Update the defaults of an entire pipeline via {class}`pipefunc.Pipeline.update_defaults``({...})`

Some examples in code:

```{code-cell} ipython3
from pipefunc import pipefunc, Pipeline

@pipefunc(output_name="y", defaults={"x": 2})
def f(a, x):
    return a * x
```

This function `f` has a default value for `x` set to 2.

:::{admonition} Do the same by constructing a <code>PipeFunc</code> object directly
:class: note, dropdown

   ```python
   from pipefunc import PipeFunc

   def f(a, x):
       return a * x

   f_func = PipeFunc(f, output_name="y", defaults={"x": 2})
   ```

:::

We can also update the defaults of the function afterwards:

```{code-cell} ipython3
f.update_defaults({"x": 3})
```

If a default value is provided in the function signature, it will be used.
However, we can override it by updating the defaults.
We can also update the defaults of the entire pipeline.

```{code-cell} ipython3
@pipefunc(output_name="z", defaults={"b": 2})  # override `b=1` default
def g(y, b=1):
    return y + b

pipeline = Pipeline([f, g])
pipeline.update_defaults({"a": 1, "b": 3, "x": 1})  # override `b=2` default
```

We can check the defaults of the pipeline:

```{code-cell} ipython3
pipeline.defaults  # all parameters now have defaults
```

Now, when we call the pipeline, we don't need to provide any arguments:

```{code-cell} ipython3
pipeline()
```

To undo the defaults, you can use `overwrite=True`:

```{code-cell} ipython3
g.update_defaults({}, overwrite=True)
print(g.defaults)  # remaining original default from signature
```

This will remove the defaults that pipefunc has set for the function `g`, and leave the original defaults in the function signature (`b=1`).

## How to bind parameters to a fixed value?

Instead of using defaults, you can bind parameters to a fixed value using the `bound` argument.

See:

- The `pipefunc` decorator `@`{class}`pipefunc.pipefunc``(..., bound={...})`
- Update the bound arguments of a `PipeFunc` object (a wrapped function) via {class}`pipefunc.PipeFunc.update_bound``({...})`

```{code-cell} ipython3
@pipefunc(output_name="y", bound={"x": 2})  # x is now fixed to 2
def f(a, x):
    return a + x

f(a=1, x=999)  # x is ignored and replaced by the bound value
```

:::{admonition} Do the same by constructing a <code>PipeFunc</code> object directly
:class: note, dropdown

   ```python
   from pipefunc import PipeFunc

   def f(a, x):
      return a + x

   f_func = PipeFunc(f, output_name="y", bound={"x": 2})
   ```

:::

Bound arguments show as red hexagons in the pipeline visualization.

```{code-cell} ipython3
pipeline = Pipeline([f])
pipeline.visualize()
```

We can update the bound arguments with

```{code-cell} ipython3
f.update_bound({"x": 3})
```

or remove them with

```{code-cell} ipython3
f.update_bound({}, overwrite=True)
f(a=1, x=999)  # no longer fixed
```

## How to rename inputs and outputs?

The `renames` attribute in `@`{class}`~pipefunc.pipefunc` and {class}`~pipefunc.PipeFunc` allows you to rename the inputs and outputs of a function before passing them to the next step in the pipeline.
This can be particularly useful when:

- The same function is used multiple times in a pipeline
- You want to provide more meaningful names to the outputs
- You need to avoid name collisions between functions

There are a few ways to specify renames:

1. Via the `@pipefunc` decorator:

```{code-cell} ipython3
@pipefunc(output_name="prod", renames={"a": "x", "b": "y"})
def multiply(a, b):
    return a * b
```

This renames the `a` input to `x` and the `b` input to `y`.

2. By creating a `PipeFunc` object directly and specifying the `renames` attribute:

```{code-cell} ipython3
from pipefunc import PipeFunc

def add(a, b):
    return a + b

add_func = PipeFunc(add, output_name="sum", renames={"a": "x", "b": "y"})
```

3. By updating the renames of an existing `PipeFunc` object:

```{code-cell} ipython3
add_func.update_renames({"x": "c", "y": "d"}, update_from="current")
```

This updates the current renames `{"a": "x", "b": "y"}` to `{"a": "c", "b": "d"}`.


4. By updating the renames of an entire pipeline:

```{code-cell} ipython3
pipeline = Pipeline([add_func])
pipeline.update_renames({"a": "aa", "b": "bb"}, update_from="original")
```

When specifying renames, you can choose to update from the original argument names (`update_from="original"`) or from the current renamed arguments (`update_from="current"`).

:::{admonition} We can also update the <code>output_name</code>
:class: note, dropdown

   ```{code-cell} ipython3
   @pipefunc(output_name=("i", "j"))
   def f(a, b):
       return a, b

   # renames must be in terms of individual output strings
   f.update_renames({"i": "ii"}, update_from="current")
   assert f.output_name == ("ii", "j")
   ```

:::


Some key things to note:

- Renaming inputs does not change the actual parameter names in the function definition, it just changes how the pipeline passes arguments between functions.
- Renaming allows using the same function multiple times in a pipeline with different input/output names each time.
- Renames are applied in the order they are defined when a function is used multiple times in a pipeline.
- You can use `pipeline.visualize()` to see a graph of the pipeline with the renamed arguments.

Proper use of renames can make your pipelines more readable and maintainable by providing clear, context-specific names for the data flowing between functions.

## How to handle multiple outputs?

Functions in a pipeline can return multiple outputs.

By default, `pipefunc` assumes that a function returns a single output or a tuple of outputs.
For any other return type, you need to specify an `output_picker` function.

If `output_picker` is not specified, `pipefunc` assumes that the function returns a single output or a tuple of outputs.
In this case, the `output_name` should be a single string or a tuple of strings with the same length as the returned tuple.

Here are a few ways to handle multiple outputs:

1. Return a tuple of outputs and specify the `output_name` as a tuple of strings:

```{code-cell} ipython3
@pipefunc(output_name=("mean", "std"))
def mean_and_std(data):
    return np.mean(data), np.std(data)
```

This will automatically unpack the tuple and assign each output to the corresponding name in `output_name`.

2. Return a dictionary, custom object, or any other type and specify the `output_name` as a tuple of strings along with an `output_picker` function:

```{code-cell} ipython3
def output_picker(dct, output_name):
    return dct[output_name]

@pipefunc(output_name=("mean", "std"), output_picker=output_picker)
def mean_and_std(data):
    return {"mean": np.mean(data), "std": np.std(data)}
```

The `output_picker` function takes the returned object as the first argument and the `output_name` as the second argument.
It should return the output corresponding to the given name.

Another example with a custom object and an explicit `output_picker` function:

```{code-cell} ipython3
from dataclasses import dataclass

@dataclass
class MeanStd:
    mean: float
    std: float

def pick_mean_std(obj, output_name):
    return getattr(obj, output_name)

@pipefunc(output_name=("mean", "std"), output_picker=pick_mean_std)
def mean_and_std(data):
    return MeanStd(np.mean(data), np.std(data))
```

Here, the `pick_mean_std` function is defined to extract the `mean` and `std` attributes from the returned `MeanStd` object.

Note that the `output_picker` function is called once for each output name specified in `output_name`.
This allows you to handle cases where the returned object has a different structure than the desired output names.

When a function has multiple outputs, subsequent functions in the pipeline can access any of those outputs by name:

```{code-cell} ipython3
@pipefunc(output_name="normalized")
def normalize(data, mean, std):
    return (data - mean) / std
```

This function takes `mean` and `std` as separate inputs, which will be automatically wired from the outputs of `mean_and_std`.

Some key things to note:

- If there are multiple outputs, all must be explicitly named in `output_name`, even if some outputs are not used by subsequent functions.
- You can use `pipeline.visualize()` to see how the multiple outputs are connected in the pipeline graph.

Handling multiple outputs allows for more modular and reusable functions in your pipelines.
It's particularly useful when a function computes multiple related values that might be used independently by different downstream functions.
This way, you can avoid recomputing the same values multiple times and can mix and match the outputs as needed.

## How does type checking work in `pipefunc`?

`pipefunc` supports type checking for function arguments and outputs using Python type hints.
It ensures that the output of one function matches the expected input types of the next function in the pipeline.
This is crucial for maintaining data integrity and catching errors early in pipeline-based workflows.

### Basic type checking

Here's an example of `pipefunc` raising a `TypeError` when the types don't match:

```{code-cell} ipython3
:tags: [raises-exception]

# All type hints that are not relevant for this example are omitted!

@pipefunc(output_name="y")
def f(a) -> int:  # output 'y' is expected to be an `int`
    return 2 * a

@pipefunc(output_name="z")
def g(y: str):  # here 'y' is expected to be a `str`
    return y.upper()

# Creating the `Pipeline` will raise a `TypeError`
pipeline = Pipeline([f, g])
```

In this example, function `f` outputs an `int`, but function `g` expects a `str` input.
When we try to create the pipeline, it will raise a `TypeError` due to this type mismatch.

```{note}
`pipefunc` only checks the type hints during pipeline construction, not during function execution.
However, *soon* we will add runtime type checking as an option.
```

To turn off this type checking, you can set the `validate_type_annotations` argument to `False` in the `Pipeline` constructor:

```{code-cell} ipython3
pipeline = Pipeline([f, g], validate_type_annotations=False)
```

Note that while disabling type checking allows the pipeline to run, it may lead to runtime errors or unexpected results if the types are not compatible.

### Type checking for Pipelines with `MapSpec` and reductions

When a pipeline contains a reduction operation (using `MapSpec`s), the type checking is more complex.

The results of a ND map operation are always stored in a numpy object array, which means that the original types are preserved in the elements of this array.
This means the type hints for the function should be `numpy.ndarray[Any, np.dtype[numpy.object_]]`.
Unfortunately, it is not possible to statically check the types of the elements in the object array (e.g., with `mypy`).
We can however, check the types of the elements at runtime.
To do this, we can use the {class}`~pipefunc.typing.Array` type hint from `pipefunc.typing`.
This `Array` generic contains the correct `numpy.ndarray` type hint for object arrays, but is annotated with the element type using {class}`typing.Annotated`.
When using e.g., `Array[int]`, the type hint is `numpy.ndarray[Any, np.dtype[numpy.object_]]` with the element type `int` in the metadata of `Annotated`.
MyPy will ensure the numpy array type, however, `PipeFunc` will ensure both the numpy object array and its element type.

Use it like this:

```{code-cell} ipython3
import numpy as np
from pipefunc import Pipeline, pipefunc
from pipefunc.typing import Array

@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def double_it(x: int) -> int:
    assert isinstance(x, int)
    return 2 * x

@pipefunc(output_name="sum")
def take_sum(y: Array[int]) -> int:
    # y is a numpy object array of integers
    # the original types are always preserved!
    assert isinstance(y, np.ndarray)
    assert isinstance(y.dtype, object)
    assert isinstance(y[0], int)
    return sum(y)

pipeline_map = Pipeline([double_it, take_sum])
pipeline_map.map({"x": [1, 2, 3]})
```

For completeness, this is the type hint for `Array[int]`:

```{code-cell} ipython3
from pipefunc.typing import Array
Array[int]
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
| Input arguments                                        | Can provide *any* input arguments for any function in the pipeline               | Requires the root arguments (use {class}`~pipefunc.Pipeline.subpipeline` to get a subgraph)                            |
| Output arguments                                       | Can request the output of any function in the pipeline                           | Calculates *all* function nodes in the entire pipeline (use {class}`~pipefunc.Pipeline.subpipeline` to get a subgraph) |
| Intermediate results storage                           | In-memory dictionary                                                             | Configurable storage ({class}`~pipefunc.map.StorageBase`), e.g., on disk, cloud, or in (shared-)memory                 |
| Use case                                               | Executing the pipeline as a regular function, going from any input to any output | Optimized for parallel execution and map-reduce operations                                                             |
| Calling syntax                                         | `pipeline.run(output_name, kwargs)` or `pipeline(output_name, **kwargs)`         | `pipeline.map(inputs, ...)`                                                                                            |

In summary, `pipeline.run` is used for sequential execution and allows flexible input and output arguments, while `pipeline.map` is optimized for parallel execution and map-reduce operations but requires structured inputs and outputs based on the `mapspec` of the functions.

## How to use parameter scopes (namespaces)?

Parameter scopes, also known as namespaces, allow you to group related parameters together and avoid naming conflicts when multiple functions in a pipeline have parameters with the same name.
You can set a scope for a `PipeFunc` using the `scope` argument in the `@pipefunc` decorator or `PipeFunc` constructor, or update the scope of an existing `PipeFunc` or `Pipeline` using the `update_scope` method.

Here are a few ways to use parameter scopes:

1. Set the scope when defining a `PipeFunc`:

```{code-cell} ipython3
@pipefunc(output_name="y", scope="foo")
def f(a, b):
    return a + b

print(f.renames)  # Output: {'a': 'foo.a', 'b': 'foo.b', 'y': 'foo.y'}
```

This sets the scope "foo" for all parameters and the output name of the function `f`.
The actual parameter names become `foo.a` and `foo.b`, and the output name becomes `foo.y`.

2. Update the scope of an existing `PipeFunc`:

```{code-cell} ipython3
from pipefunc import PipeFunc

def g(a, b, y):
    return a * b + y

g_func = PipeFunc(g, output_name="z", renames={"y": "foo.y"})
print(g_func.parameters)  # Output: ('a', 'b', 'foo.y')
print(g_func.output_name)  # Output: 'z'

g_func.update_scope("bar", inputs={"a"}, outputs="*")
print(g_func.parameters)  # Output: ('bar.a', 'b', 'foo.y')
print(g_func.output_name)  # Output: 'bar.z'
```

This updates the scope of the outputs of `g_func` to "bar".
The parameter names become `bar.a`, `b`, and `foo.y`, and the output name becomes `bar.z`.

3. Update the scope of an entire `Pipeline`:

```{code-cell} ipython3
pipeline = Pipeline([f, g_func])
# all outputs except foo.y, so only bar.z, which becomes baz.z
pipeline.update_scope("baz", inputs=None, outputs="*", exclude={"foo.y"})
```

This updates the scope of all outputs of the pipeline to "baz", except for the output `foo.y` which keeps its existing scope.
The parameters are now `foo.a`, `foo.b`, `bar.a`, `b`, and the output names are `foo.y` and `baz.z`.
Noting that if a parameter is already in a scope, it will be replaced by the new scope.

When providing parameter values for functions or pipelines with scopes, you can either use a nested dictionary structure or the dot notation:

```{code-cell} ipython3
pipeline(foo=dict(a=1, b=2), bar=dict(a=3), b=4)
# or
pipeline(**{"foo.a": 1, "foo.b": 2, "bar.a": 3, "b": 4})
```

Some key things to note:

- Setting a scope prefixes parameter and output names with `{scope}.`, e.g., `x` becomes `foo.x` if the scope is "foo".
- You can selectively include or exclude certain inputs/outputs when updating the scope using the `inputs`, `outputs` and `exclude` arguments.
- Updating the scope of a pipeline updates the scopes of its functions, propagating the changes.
- Applying a scope to a parameter that is already in a scope will *replace* the existing scope with the new one.
- Using scopes makes it possible to use the same parameter names in different contexts without conflicts.
- Scopes are purely a naming mechanism and do not affect the actual function execution.

Parameter scopes are a convenient way to organize complex pipelines and make them more readable by grouping related parameters together.
They also help avoid naming conflicts and make it easier to reason about the data flow between functions.

To illustrate how `update_scope` works under the hood, consider this example:

```{code-cell} ipython3
@pipefunc(output_name="y")
def f(a, b):
    return a + b

@pipefunc(output_name="z")
def g(y, c):
    return y * c

pipeline = Pipeline([f, g])
pipeline2 = pipeline.copy()
```

Now, let's update the scope of the pipeline using `update_scope`:

```{code-cell} ipython3
pipeline.update_scope("my_scope", inputs="*", outputs="*")
```

This is equivalent to applying the following renames:

```{code-cell} ipython3
pipeline2.update_renames({"a": "my_scope.a", "b": "my_scope.b", "y": "my_scope.y", "c": "my_scope.c", "z": "my_scope.z"})
```

After applying the scope, the parameter names and output names of the functions in the pipeline are prefixed with `my_scope.`.
We can confirm this by inspecting the `PipeFunc` objects:

:::{admonition} Get the <code>PipeFunc</code> objects using <code>pipeline[output_name]</code>
:class: note, dropdown

   The functions passed to the `Pipeline` constructor are copied using `PipeFunc.copy()`, so the original functions are not modified.
   Therefore, to get the `PipeFunc` objects from the pipeline, you can use `pipeline[output_name]` to retrieve the functions by their output names.

:::

```{code-cell} ipython3
f = pipeline["my_scope.y"]
print(f.parameters)  # Output: ('my_scope.a', 'my_scope.b')
print(f.output_name)  # Output: 'my_scope.y'
g = pipeline["my_scope.z"]
print(g.parameters)  # Output: ('my_scope.y', 'my_scope.c')
print(g.output_name)  # Output: 'my_scope.z'
```

or see the `renames` attribute:

```{code-cell} ipython3
for f in pipeline.functions:
    print(f.__name__, f.renames)

for f in pipeline2.functions:
    print(f.__name__, f.renames)
```

So, `update_scope` is really just a convenience method that automatically generates the appropriate renames based on the provided scope and applies them to the `PipeFunc` or `Pipeline`.
Internally, it calls `update_renames` with the generated renames, making it easier to manage parameter and output names in complex pipelines.

It's worth noting that while `update_scope` affects the external names (i.e., how the parameters and outputs are referred to in the pipeline), it doesn't change the actual parameter names in the original function definitions.
The mapping between the original names and the scoped names is handled by the `PipeFunc` wrapper.

## How to inspect the `Resources` inside a `PipeFunc`?

```{note}
Using `resouces` requires an execution environment that supports resource management.
Currently, only the Adaptive Scheduler execution environment supports resource management.
In all cases, you can set and inspect the `Resources` object, but whether these resources are actually used depends on the execution environment.
```

When the `resources_variable` argument is provided, you can access the `resources` object inside the function to inspect the `Resources` associated with the `PipeFunc`.

Here's an example:

```{code-cell} ipython3
from pipefunc import pipefunc, Pipeline

@pipefunc(
    output_name="c",
    resources={"memory": "1GB", "cpus": 2},
    resources_variable="resources",
)
def f(a, b, resources):
    print(f"Inside the function `f`, resources.memory: {resources.memory}")
    print(f"Inside the function `f`, resources.cpus: {resources.cpus}")
    return a + b

result = f(a=1, b=1)
print(f"Result: {result}")
```

In this example, the `resources` argument is passed to the function `f` via the `resources_variable` parameter.
Inside the function, you can access the attributes of the `Resources` instance using `resources.memory` and `resources.cpus`.

As you can see, the function `f` has access to the `resources` object and can inspect its attributes directly.

Similarly, when using a `Pipeline`, you can inspect the `Resources` inside the functions:

```{code-cell} ipython3
@pipefunc(output_name="d", resources={"gpus": 4}, resources_variable="resources")
def g(c, resources):
    print(f"Inside the function `g`, resources.gpus: {resources.gpus}")
    return c * 2

pipeline = Pipeline([f, g])
result = pipeline(a=1, b=1)
print(f"Pipeline result: {result}")
```

In this case, the `Pipeline` consists of two functions, `f` and `g`, both of which have access to their respective `resources` objects.

The function `f` can inspect `resources.memory` and `resources.cpus`, while the function `g` can inspect `resources.gpus`.

By using the `resources_variable` argument, you can pass the `Resources` instance directly to the functions, allowing them to inspect the resource information as needed.

## How to set the `Resources` dynamically, based on the input arguments?

You can set the `Resources` for a `PipeFunc` dynamically based on the input arguments by providing a callable to the `resources` argument.
This ensures lazy evaluation of the resources, allowing you to determine the resources at runtime based on the input arguments.
The callable should take a dictionary of input arguments and return a `Resources` instance.

```{note}
This becomes a powerful feature when combined with the `resources_variable` argument, but we first demonstrate it without using `resources_variable`.
See the next example for how to use it in combination with `resources_variable`.
```

Here's an example that uses a function to determine the resources for a `PipeFunc`:

```{code-cell} ipython3
from pipefunc import pipefunc, Pipeline
from pipefunc.resources import Resources

def resources_func(kwargs):
    gpus = kwargs["x"] + kwargs["y"]
    print(f"Inside the resources function, gpus: {gpus}")
    return Resources(gpus=gpus)

@pipefunc(output_name="out1", resources=resources_func)
def f(x, y):
    return x * y

result = f(x=2, y=3)
print(f"Result: {result}")
```

In this case, `f.resources` is a callable that takes a dictionary of input arguments and returns a `Resources` instance with `gpus` set to the sum of `x` and `y`.

The `f.resources` callable is invoked with the dictionary of input arguments to determine the resources for that specific execution.

```{code-cell} ipython3
print(f"Resources: {f.resources({"x": 2, "y": 3})}")
```

You can also set the `Resources` dynamically in a `Pipeline`:

```{code-cell} ipython3
pipeline = Pipeline([f])
result = pipeline(x=2, y=3)
print(f"Pipeline result: {result}")
```

Now, let's see an example that uses both a `resources` callable and the `resources_variable` argument:

```{code-cell} ipython3
def resources_with_cpu(kwargs):
    cpus = kwargs["out1"] + kwargs["z"]
    return Resources(cpus=cpus)

@pipefunc(
    output_name="out2",
    resources=resources_with_cpu,
    resources_variable="resources",
)
def g(out1, z, resources):
    print(f"Inside the function `g`, resources.cpus: {resources.cpus}")
    return out1 * z

result = g(out1=2, z=3)
print(f"Result: {result}")
```

In this case, `g.resources` is a callable that takes a dictionary of input arguments and returns a `Resources` instance with `cpus` set to the sum of `out1` and `z`
The resulting `Resources` instance is then passed to the function `g` via the `resources` parameter.

The `resources` callable dynamically creates a `Resources` instance based on the input arguments, and the function `g` can access the `cpus` attribute of the `resources` object inside the function.

Combining both functions into a pipeline

```{code-cell} ipython3
pipeline = Pipeline([f, g])
result = pipeline(x=2, y=3, z=1)
print(f"Result: {result}")
```

By using a callable for `resources`, you can dynamically determine the resources based on the input arguments.
Additionally, by using the `resources_variable` argument, you can pass the dynamically created `Resources` instance directly to the function, allowing it to access and utilize the resource information as needed.

## How to use `adaptive` with `pipefunc`?

```{note}
We will assume familiarity with the `adaptive` and `adaptive_scheduler` packages in this section.
```

There are plans to integrate `adaptive` with `pipeline.map` to enable adaptive sweeps over parameter spaces.
Currently, using `adaptive` with `pipefunc` is a bit more cumbersome, but it is still possible.
See [this tutorial](adaptive.md) for a detailed example of how to use `adaptive` with `pipefunc`.

## SLURM integration via [Adaptive Scheduler](https://adaptive-scheduler.readthedocs.io/) integration

PipeFunc can also be used with the `adaptive_scheduler` package to run the pipeline on a cluster.
This allows you to run the pipeline on a cluster (e.g., with SLURM) without having to worry about the details of submitting jobs and managing resources.

Here's an example of how to use `pipefunc` with `adaptive_scheduler`:

```{code-cell} ipython3
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from pipefunc import Pipeline, pipefunc
from pipefunc.map.adaptive import create_learners
from pipefunc.resources import Resources


# Pass in a `Resources` object that specifies the resources needed for each function
@pipefunc(output_name="double", mapspec="x[i] -> double[i]", resources=Resources(cpus=5))
def double_it(x: int) -> int:
    return 2 * x


# Or specify the resources as a dictionary
@pipefunc(output_name="half", mapspec="x[i] -> half[i]", resources={"memory": "8GB"})
def half_it(x: int) -> int:
    return x // 2


# Specify delayed resources that are used inside the function; "internal" parallelization
@pipefunc(
    output_name="sum",
    resources=lambda kw: {"cpus": len(kw["half"]), "parallelization_mode": "internal"},
    resources_variable="resources",
)
def take_sum(half: np.ndarray, double: np.ndarray, resources: Resources) -> int:
    with ProcessPoolExecutor(resources.cpus) as executor:
        # Do some printing in parallel (not smart, but just to show the parallelization)
        list(executor.map(print, range(resources.cpus)))
    return sum(half + double)


pipeline_adapt = Pipeline([double_it, half_it, take_sum])
```

We now have a pipeline with three functions, each with different resource requirements.
So far nothing is Adaptive-Scheduler specific.
We could call `pipeline_adapt.map(...)` to run this pipeline in parallel on your local machine.However, to run it on a cluster, we need to use `adaptive_scheduler`.
We can convert the pipeline to a dictionary of `adaptive.SequenceLearner`s objects using `create_learners` and then submit these to the cluster using `adaptive_scheduler`.

```{code-cell} ipython3
inputs = {"x": [0, 1, 2, 3]}
run_folder = "my_run_folder"
learners_dict = create_learners(
    pipeline_adapt,
    inputs,
    run_folder=run_folder,
    split_independent_axes=True,  # Split up into as many independent jobs as possible
)
kwargs = learners_dict.to_slurm_run(
    returns="kwargs",  # or "run_manager" to return a `adaptive_scheduler.RunManager` object
    default_resources={"cpus": 2, "memory": "8GB"},
)
# kwargs can be passed to `adaptive_scheduler.slurm_run(**kwargs)`
```

## What is the `ErrorSnapshot` feature in `pipefunc`?

The {class}`~pipefunc.ErrorSnapshot` feature captures detailed information about errors occurring during the execution of a `PipeFunc`. It aids in debugging by storing snapshots of error states, including the function, exception details, arguments, timestamp, and environment. This snapshot can be used to reproduce the error and examine the error context.

**Key Features:**

- **Error Details**: Captures function name, exception, arguments, traceback, user, machine, timestamp, and directory.
- **Reproduction**: Offers a `reproduce()` method to recreate the error with the stored arguments.
- **Persistence**: Allows saving and loading snapshots using `save_to_file` and `load_from_file`.

**Usage:**

1. **Accessing Snapshots**:
   ```python
   result = my_pipefunc_or_pipeline(args)
   if my_pipefunc_or_pipeline.error_snapshot:
       print(my_pipefunc_or_pipeline.error_snapshot)
   ```

2. **Reproducing Errors**:
   ```python
   error_snapshot = my_pipefunc_or_pipeline.error_snapshot
   if error_snapshot:
       error_snapshot.reproduce()
   ```

3. **Saving and Loading**:
   ```python
   error_snapshot.save_to_file("snapshot.pkl")
   loaded_snapshot = ErrorSnapshot.load_from_file("snapshot.pkl")
   ```

**Example:**

```{code-cell} ipython3
@pipefunc(output_name="c")
def faulty_function(a, b):
    # Simulate an error
    raise ValueError("Intentional error")

try:
    faulty_function(a=1, b=2)
except Exception:
    snapshot = faulty_function.error_snapshot
    print(snapshot)
```

{class}`~pipefunc.ErrorSnapshot` is very useful for debugging complex pipelines, making it easy to replicate and understand issues as they occur.

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

## How to mock functions in a pipeline for testing?

When mocking a function within a `Pipeline` for testing purposes, you can use the {class}`pipefunc.testing.patch` utility.
This is particularly useful for replacing the implementation of a function with a mock during tests, allowing you to control outputs and side effects.

:::{admonition} Why not `unittest.mock.patch`?
The plain use of `unittest.mock.patch` is insufficient for `Pipeline` objects due to internal management of functions.
**Wrapped Functions**: A `Pipeline` contains `PipeFunc` instances that store a reference to the original function in a `func` attribute.
This structure means the function isn't directly accessible by name for patching, as `unittest.mock.patch` would typically require.
:::

See this example for how to use {class}`pipefunc.testing.patch` to mock functions in a pipeline:

```{code-cell} ipython3
from pipefunc.testing import patch
from pipefunc import Pipeline, pipefunc, PipeFunc
import random

my_first = PipeFunc(random.randint, output_name="rnd", defaults={"a": 0, "b": 10})


@pipefunc(output_name="result")
def my_second(rnd):
    raise RuntimeError("This function should be mocked")


pipeline = Pipeline([my_first, my_second])

# Patch a single function
with patch(pipeline, "my_second") as mock:
    mock.return_value = 5
    print(pipeline())

# Patch multiple functions
with patch(pipeline, "random.randint") as mock1, patch(pipeline, "my_second") as mock2:
    mock1.return_value = 3
    mock2.return_value = 5
    print(pipeline())
```

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

```

## Get a function handle for a specific pipeline output (`pipeline.func`)

We can get a handle for each function using the `func` method on the pipeline, passing the output name of the function we want.

```{code-cell} ipython3

@pipefunc(output_name="c")
def f(a, b):
    return a + b


@pipefunc(output_name="d")
def g(b, c, x=1):
    return b * c * x


@pipefunc(output_name="e")
def h(c, d, x=1):
    return c * d * x

pipeline = Pipeline([f, g, h])
func_d = pipeline.func("d")
func_e = pipeline.func("e")
```

We can now use these handles as if they were the original functions.
The pipeline will automatically ensure that the functions are called in the correct order, passing the output of one function as the input to the next.

```{code-cell} ipython3
c = f(a=2, b=3)  # call the wrapped function directly
assert c == 5
```

```{code-cell} ipython3
assert (
    g(b=3, c=5)
    == func_d(a=2, b=3)  # We can call func_d with different arguments
    == func_d(b=3, c=5)
    == 15
)
assert func_e(c=c, d=15, x=1) == func_e(a=2, b=3, x=1) == func_e(a=2, b=3, d=15, x=1) == 75
```

The functions returned by `pipeline.func` have several additional methods

**Using the call_full_output Method**

The `call_full_output()` method can be used to call the function and get all the outputs from the pipeline as a dictionary.

```{code-cell} ipython3
func_e = pipeline.func("e")
func_e.call_full_output(a=2, b=3, x=1)
```

**Direct Calling with Root Arguments (as positional arguments)**

You can directly call the functions in the pipeline with the root arguments using the `call_with_root_args()` method. It automatically executes all the dependencies of the function in the pipeline with the given root arguments.

```{code-cell} ipython3
func_e = pipeline.func("e")
func_e.call_with_root_args(1, 2, 1)  # note these are now positional args
```

This executes the function `g` with the root arguments `a=1, b=2, x=1`.

For more information about this method, you can use the Python built-in `help` function or the `?` command.

```{code-cell} ipython3
help(func_e.call_with_root_args)
```

This shows the signature and the doc-string of the `call_with_root_args` method.

## `dataclasses` and `pydantic.BaseModel` as `PipeFunc`

`PipeFunc` can be used with `dataclasses` and `pydantic.BaseModel` classes as `PipeFunc`s.

Suppose we have a `dataclass` and a `pydantic.BaseModel` class:

```python
from pipefunc import PipeFunc, Pipeline
from dataclasses import dataclass
from pydantic import BaseModel

@dataclass
class InputDataClass:
    a: int
    b: int

class PydanticModel(BaseModel):
    x: int
    y: int

# We can use these classes as PipeFuncs

pf1 = PipeFunc(InputDataClass, output_name="dataclass")
pf2 = PipeFunc(PydanticModel, output_name="pydantic")

pipeline = Pipeline([pf1, pf2])
result = pipeline.map(inputs={"a": 1, "b": 2, "x": 3, "y": 4}, parallel=False)
assert result["dataclass"].output == InputDataClass(a=1, b=2)
assert result["pydantic"].output == PydanticModel(x=3, y=4)
```

:::{admonition} Careful with `default_factory`!
:class: warning
When using `dataclasses` or `pydantic.BaseModel` with `dataclasses.field(..., default_factory=...)` or `pydantic.Field(..., default_factory=...)`, the default value will be computed only once when the `PipeFunc` class is defined.
So if you are using mutable defaults, make sure to not mutate the value in the function body!
This is the same behavior as with regular Python functions.
:::

## Parameter Sweeps

The `pipefunc.sweep` module provides a convenient way to contruct parameter sweeps.
It was developed before `pipeline.map` which can perform sweep operations in parallel.
However, by itself {class}`pipefunc.sweep.Sweep` might still be useful for cases where you have a pipeline that has no `mapspec`.

```{code-cell} ipython3
from pipefunc.sweep import Sweep

combos = {
    "a": [0, 1, 2],
    "b": [0, 1, 2],
    "c": [0, 1, 2],
}
# This means a Cartesian product of all the values in the lists
# while zipping ("a", "b").
sweep = Sweep(combos, dims=[("a", "b"), "c"])
sweep.list()[:10]  # show the first 10 combinations
```

The function `set_cache_for_sweep` then enables caching for nodes in the pipeline that are expected to be executed two or more times during the parameter sweep.

```python
from pipefunc.sweep import set_cache_for_sweep

set_cache_for_sweep(output_name, pipeline, sweep, min_executions=2, verbose=True)
```

We can now run the sweep using e.g.,

```python
results = [
    pipeline.run(output_name, kwargs=combo, full_output=True) for combo in sweep.list()
]
```
