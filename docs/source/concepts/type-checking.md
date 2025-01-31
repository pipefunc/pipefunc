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

# Type Checking

```{try-notebook}
```

```{contents} ToC
:depth: 2
```

## How does type checking work in `pipefunc`?

`pipefunc` supports type checking for function arguments and outputs using Python type hints.
It ensures that the output of one function matches the expected input types of the next function in the pipeline.
This is crucial for maintaining data integrity and catching errors early in pipeline-based workflows.

### Basic type checking

Here's an example of `pipefunc` raising a `TypeError` when the types don't match:

```{code-cell} ipython3
:tags: [raises-exception]
from pipefunc import Pipeline, pipefunc

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
