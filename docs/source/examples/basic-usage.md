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
result = pipeline("e", a=2, b=3)  # Or: pipeline.run("e", kwargs={"a": 2, "b": 3})
print(result)
```

## Explanation

1. **Function Definition:** We define three simple functions, `f`, `g`, and `h`. Each function is decorated with `@pipefunc`, which makes it a "pipeable" function that can be used within a `Pipeline`. The `output_name` argument specifies the name of the output produced by each function.
2. **Pipeline Creation:** We create a `Pipeline` object, passing a list of the pipeable functions: `[f, g, h]`. The order of the functions in the list is unimportant.
3. **Pipeline Execution:** We execute the pipeline using `pipeline("e", a=2, b=3)`.
   - `"e"` specifies that we want the output of function `h` (which has `output_name="e"`).
   - `a=2, b=3` are the input arguments to the pipeline.
   - The pipeline automatically determines the execution order based on the dependencies between the functions and their inputs/outputs.
4. **Sequential Execution:** In this basic example, the functions are executed sequentially:
   - First, `f` is called with `a=2` and `b=3`, producing `c=5`.
   - Next, `g` is called with `b=3`, `c=5` (the output of `f`), and the default `x=1`, producing `d=15`.
   - Finally, `h` is called with `c=5`, `d=15`, and `x=1`, producing `e=75`.

**Features Demonstrated:**

- {func}`@pipefunc <pipefunc.pipefunc>`: Decorator to make a function "pipeable."
- {class}`~pipefunc.Pipeline`: Class to create and manage a pipeline of functions.
- Sequential execution using `pipeline()` (or equivalently, {meth}`~pipefunc.Pipeline.run`).

**Further Exploration:**

- For more details on creating pipelines, see the [main tutorial](../tutorial.md).
- To learn about parallel execution, refer to the {ref}`execution and parallelism <execution-and-parallelism>` concept.
