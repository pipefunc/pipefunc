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

# Resource Management

```{try-notebook}
```

```{contents} ToC
:depth: 2
```

## How to inspect the `Resources` inside a `PipeFunc`?

```{note}
Using `resouces` requires an execution environment that supports resource management.
Currently, only the Adaptive Scheduler execution environment supports resource management, see [SLURM integration](./slurm.md).
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
