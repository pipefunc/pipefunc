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

There are plans to integrate `adaptive` with `pipeline.map` to enable adaptive sweeps over parameter spaces.
Currently, using `adaptive` with `pipefunc` is a bit more cumbersome, but it is still possible.
See [this tutorial](adaptive.md) for a detailed example of how to use `adaptive` with `pipefunc`.
