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

# Parameter Scopes

```{try-notebook}
```

```{contents} ToC
:depth: 2
```

## How to use parameter scopes (namespaces)?

Parameter scopes, also known as namespaces, allow you to group related parameters together and avoid naming conflicts when multiple functions in a pipeline have parameters with the same name.
You can set a scope for a `PipeFunc` using the `scope` argument in the `@pipefunc` decorator or `PipeFunc` constructor, or update the scope of an existing `PipeFunc` or `Pipeline` using the `update_scope` method.

Here are a few ways to use parameter scopes:

1. Set the scope when defining a `PipeFunc`:

```{code-cell} ipython3
from pipefunc import pipefunc, Pipeline

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
- Applying a scope to a parameter that is already in a scope will _replace_ the existing scope with the new one.
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
