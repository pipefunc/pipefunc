---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: pipefunc
  language: python
  name: python3
---

# Basic Usage

```{try-notebook}
```

This example demonstrates a simple pipeline using the {func}`@pipefunc <pipefunc.pipefunc>` decorator and the {class}`~pipefunc.Pipeline` class. It showcases a basic workflow with sequential execution.

## Code

```{code-cell} ipython3
from pipefunc import pipefunc, Pipeline

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
pipeline.visualize()
```

```{code-cell} ipython3
pipeline  # Display the pipeline
```

```{code-cell} ipython3
result = pipeline("e", a=2, b=3)  # Or: pipeline.run("e", kwargs={"a": 2, "b": 3})
print(result)
```

## Explanation

1. **Define "Pipeable" Functions:** We define three functions, `f`, `g`, and `h`. Each is decorated with `@pipefunc`, making it usable within a `Pipeline`. `output_name` assigns a name to each function's output.
2. **Create Pipeline:** A `Pipeline` object is created using `Pipeline([f, g, h])`. The order of functions in this list does **not** affect execution order.
3. **Visualize Pipeline:** The pipeline is visualized using `pipeline.visualize()`. This shows the function dependencies.
4. **Execute Pipeline:** The pipeline is run using `pipeline("e", a=2, b=3)`.
   - `"e"` indicates that we want the output of function `h` (which has `output_name="e"`).
   - `a=2, b=3` provide input arguments.
   - The pipeline automatically determines the correct execution order based on function dependencies.
5. **Sequential Execution:** In this example, the functions are executed sequentially based on their dependencies:
   - `f(a=2, b=3)` produces `c=5`.
   - `g(b=3, c=5, x=1)` produces `d=15`.
   - `h(c=5, d=15, x=1)` produces `e=75`.

**Features Demonstrated:**

- {func}`@pipefunc <pipefunc.pipefunc>`: Decorator to make a function "pipeable."
- {class}`~pipefunc.Pipeline`: Class to create and manage a pipeline of functions.
- Visualization using {meth}`~pipefunc.Pipeline.visualize`.
- Sequential execution using `pipeline()` (or equivalently, {meth}`~pipefunc.Pipeline.run`).

**Further Exploration:**

- For more details on creating pipelines, see the [main tutorial](../tutorial.md).
- Already familiar with the basics? Check out the [Physics Based Example](./physics-simulation.md) for an example using some awesome `pipefunc` features!
