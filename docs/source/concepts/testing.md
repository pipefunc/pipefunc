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

# Testing

```{try-notebook}
```

```{contents} ToC
:depth: 2
```

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
