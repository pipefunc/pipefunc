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

# Function Inputs and Outputs

```{try-notebook}
```

```{contents} ToC
:depth: 2
```

## How to handle defaults?

You can provide defaults in

- The original function definition (the normal way)
- The `pipefunc` decorator `@`{class}` pipefunc.pipefunc``(..., defaults={...}) `
- Update the defaults of a `PipeFunc` object (a wrapped function) via {class}` pipefunc.PipeFunc.update_defaults``({...}) `
- Update the defaults of an entire pipeline via {class}` pipefunc.Pipeline.update_defaults``({...}) `

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

## How to bind parameters to a fixed value?

Instead of using defaults, you can bind parameters to a fixed value using the `bound` argument.

See:

- The `pipefunc` decorator `@`{class}` pipefunc.pipefunc``(..., bound={...}) `
- Update the bound arguments of a `PipeFunc` object (a wrapped function) via {class}` pipefunc.PipeFunc.update_bound``({...}) `

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

## How to collect results as a step in my `Pipeline`?

Sometimes you might need to collect specific inputs and/or outputs of different `PipeFunc`s within your pipeline.
You can achieve this by using {class}`pipefunc.helpers.collect_kwargs` to create a `PipeFunc` that gathers these values into a dictionary.

:::{admonition} Using `pipeline.map` automatically collects all results
:class: note, dropdown
When using `pipeline.map`, all results are automatically collected and returned as a dictionary of `Result` objects.
These `Result` objects contain the `kwargs` and `output` of each function in the pipeline.
:::

Here's an example:

```{code-cell} ipython3
from pipefunc import Pipeline, pipefunc, PipeFunc
from pipefunc.helpers import collect_kwargs

@pipefunc(output_name="out1")
def f1(in1):
    return in1

@pipefunc(output_name="out2")
def f2(in2, out1):
    return in2 + out1

# Creates a function with signature `aggregate(in1, out1, out2) -> dict[str, Any]`
agg = collect_kwargs(("in1", "out1", "out2"), function_name="aggregate")
f3 = PipeFunc(agg, output_name="result_dict")

pipeline = Pipeline([f1, f2, f3])
result = pipeline(in1=1, in2=2)
assert result == {"in1": 1, "out1": 1, "out2": 3}  # same parameters as in `collect_kwargs`

pipeline.visualize(backend="graphviz")
```

## `PipeFunc`s with Multiple Outputs of Different Shapes

**Question:** How can I use `PipeFunc` to return multiple outputs with different shapes when using `mapspec`? It seems like `mapspec` requires all outputs to have the same dimensions.

**Answer:**

You're correct that `pipefunc` currently has a limitation where multiple outputs within a single `PipeFunc` using `mapspec` must share the same indices and therefore the same shape.
In the future we might remove this requirement.

**Workaround:**

The recommended solution is to encapsulate your multiple outputs within a single container object (like a `dataclass`, `NamedTuple`, or even a dictionary) and return that container from your `PipeFunc`. Then, create separate `PipeFunc`s that extract the individual outputs from the container.

**Example:**

Let's say you have a function that processes some data and needs to return two lists, "complete" and "incomplete", which will likely have different lengths. Here's how you can structure it using a `dataclass` and subsequent functions to access each list:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipefunc import Pipeline, pipefunc
from pipefunc.typing import Array

@dataclass
class Status:
    complete: list[int]
    incomplete: list[int]

@pipefunc("status")
def get_status(mock_complete: list[int], mock_incomplete: list[int]) -> Status:
    return Status(mock_complete, mock_incomplete)

@pipefunc("incomplete")
def get_incomplete(status: Status) -> list[int]:
    return status.incomplete

@pipefunc("complete")
def get_complete(status: Status) -> list[int]:
    return status.complete

@pipefunc("loaded", mapspec="complete[i] -> loaded[i]")
def load_complete(complete: int) -> int:
    # Pretend we loaded something
    return complete

@pipefunc("executed", mapspec="incomplete[j] -> executed[j]")
def run_incomplete(incomplete: int) -> int:
    # Pretend we executed something
    return incomplete

@pipefunc("result")
def combine(loaded: Array[int], executed: Array[int]) -> list[int]:
    return list(loaded) + list(executed)

pipeline = Pipeline(
    [
        get_status,
        get_incomplete,
        get_complete,
        load_complete,
        run_incomplete,
        combine,
    ]
)
result = pipeline.map(
    {"mock_complete": [0], "mock_incomplete": [1, 2, 3]},
    internal_shapes={"incomplete": ("?",), "complete": ("?",)},
    parallel=False,
)

print(result["result"].output)
```

**Explanation:**

1. **`Status` Dataclass:** We define a `Status` dataclass to hold the `complete` and `incomplete` lists as a single object.
2. **`get_status` Function:** This function now returns a `Status` object. Because it does not have a `mapspec` it will only run once.
3. **`get_incomplete` and `get_complete` Functions:** These helper functions extract the individual lists from the `Status` object.
4. **`load_complete` and `run_incomplete` Functions:** These functions can now use `mapspec` to iterate over the `complete` and `incomplete` lists, respectively.
5. **`combine` Function:** This function now takes `completed` and `executed` and combines them with the `complete` list.
6. **`pipeline.map`:** We call `pipeline.map` as before, but now we only need to specify the `internal_shapes` of the lists, not the shape of the status. The `internal_shapes` argument is only needed when you return a list, and it cannot be inferred from the inputs.

This pattern provides a clean and manageable way to work with functions that logically produce multiple outputs of varying shapes within the current capabilities of `pipefunc`.
