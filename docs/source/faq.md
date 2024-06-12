---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: adaptive
  language: python
  name: python3
---

# ❓ FAQ: Frequently Asked Questions

## How to handle defaults?

You can provide defaults in

- The original function definition (the normal way)
- The `pipefunc` decorator `@pipefunc(..., defaults={...})` (see {class}`pipefunc.pipefunc`)
- Update the defaults of a `PipeFunc` object (a wrapped function) via `PipeFunc.update_defaults({...})` (see {class}`pipefunc.PipeFunc.update_defaults`)
- Update the defaults of an entire pipeline via `Pipeline.update_defaults({...})` (see {class}`pipefunc.Pipeline.update_defaults`)

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

- The `pipefunc` decorator `@pipefunc(..., bound={...})` (see {class}`pipefunc.pipefunc`)
- Update the bound arguments of a `PipeFunc` object (a wrapped function) via `PipeFunc.update_bound({...})` (see {class}`pipefunc.PipeFunc.update_bound`)

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

The `renames` attribute in `@`{class}`~pipefunc.pipefunc` and {class}`~pipefunc.PipeFunc` allows you to rename the inputs and outputs of a function before passing them to the next step in the pipeline. This can be particularly useful when:

- The same function is used multiple times in a pipeline
- You want to provide more meaningful names to the outputs
- You need to avoid name collisions between functions

There are a few ways to specify renames:

1. Via the `@pipefunc` decorator:

   ```python
   @pipefunc(output_name="y", renames={"a": "x", "b": "y"})
   def multiply(a, b):
       return a * b
   ```

   This renames the `a` input to `x` and the `b` input to `y`.

2. By creating a `PipeFunc` object directly and specifying the `renames` attribute:

   ```python
   def add(a, b):
       return a + b

   add_func = PipeFunc(add, output_name="sum", renames={"a": "x", "b": "y"})
   ```

3. By updating the renames of an existing `PipeFunc` object:

   ```python
   add_func.update_renames({"x": "c", "y": "d"}, update_from="current")
   ```

   This updates the current renames `{"a": "x", "b": "y"}` to `{"a": "c", "b": "d"}`.

4. By updating the renames of an entire pipeline:

   ```python
   pipeline.update_renames({"a": "aa", "b": "bb"}, update_from="current")
   ```

When specifying renames, you can choose to update from the original argument names (`update_from="original"`) or from the current renamed arguments (`update_from="current"`).

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

   ```python
   @pipefunc(output_name=("mean", "std"))
   def mean_and_std(data):
       return np.mean(data), np.std(data)
   ```

   This will automatically unpack the tuple and assign each output to the corresponding name in `output_name`.

2. Return a dictionary, custom object, or any other type and specify the `output_name` as a tuple of strings along with an `output_picker` function:

   ```python
   def output_picker(dct, output_name):
       return dct[output_name]

   @pipefunc(output_name=("mean", "std"), output_picker=output_picker)
   def mean_and_std(data):
       return {"mean": np.mean(data), "std": np.std(data)}
   ```

   The `output_picker` function takes the returned object as the first argument and the `output_name` as the second argument. It should return the output corresponding to the given name.

   Another example with a custom object and an explicit `output_picker` function:

   ```python
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

```python
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

## What is the difference between `pipeline.run` and `pipeline.map`?

These methods are used to execute the pipeline but have different use cases:

- `pipeline.run(output_name, kwargs)` is used to execute the pipeline as a function and is fully sequential. It allows going from any input arguments to any output arguments. It does **not** support map-reduce operations. Internally, it keeps all intermediate results in memory in a dictionary.
- `pipeline.map(inputs, ...)` is used to execute the pipeline in parallel. It supports map-reduce operations and is optimized for parallel execution. Internally, it puts all intermediate results in a {class}`~pipefunc.map.StorageBase` (there are implementations for storing on disk or in memory).

```{note}
When calling the pipeline as a function, like `pipeline("output_name", **kwargs)`, it is equivalent to calling `pipeline.run("output_name", **kwargs)`.
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
