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

# Simplifying Pipelines

```{try-notebook}
```

```{contents} ToC
:depth: 2
```

This section is about {meth}`pipefunc.Pipeline.simplified_pipeline`, which is a convenient way to simplify a pipeline by merging multiple nodes into a single node (creating a {class}`pipefunc.NestedPipeFunc`).
Consider the following pipeline (look at the `visualize()` output to see the structure of the pipeline):

```{code-cell} ipython3
from pipefunc import Pipeline


def f1(a, b, c, d):
    return a + b + c + d


def f2(a, b, e):
    return a + b + e


def f3(a, b, f1):
    return a + b + f1


def f4(f1, f3):
    return f1 + f3


def f5(f1, f4):
    return f1 + f4


def f6(b, f5):
    return b + f5


def f7(a, f2, f6):
    return a + f2 + f6


# If the functions are not decorated with @pipefunc,
# they will be wrapped and the output_name will be the function name
pipeline_complex = Pipeline([f1, f2, f3, f4, f5, f6, f7])
pipeline_complex("f7", a=1, b=2, c=3, d=4, e=5)
pipeline_complex.visualize_matplotlib(
    color_combinable=True,
)  # combinable functions have the same color
```

In the example code above, the complex pipeline composed of multiple functions (`f1`, `f2`, `f3`, `f4`, `f5`, `f6`, `f7`) can be simplified by merging the nodes `f1`, `f3`, `f4`, `f5`, `f6` into a single node.
This merging process simplifies the pipeline and allows to reduce the number of functions that need to be cached/saved.

The method `reduced_pipeline` from the `Pipeline` class is used to generate this simplified version of the pipeline.

```{code-cell} ipython3
simplified_pipeline_complex = pipeline_complex.simplified_pipeline("f7")
simplified_pipeline_complex.visualize()  # A `NestedPipeFunc` will have a red edge
```

However, simplifying a pipeline comes with a trade-off. The simplification process removes intermediate nodes that may be necessary for debugging or inspection.

For instance, if a developer wants to monitor the output of `f3` while processing the pipeline, they would not be able to do so in the simplified pipeline as `f3` has been merged into a {class}`pipefunc.NestedPipeFunc`.
+++

The simplified pipeline now contains a {class}`pipefunc.NestedPipeFunc` object, which is a subclass of {class}`~pipefunc.PipeFunc` but contains an internal pipeline.

```{code-cell} ipython3
simplified_pipeline_complex.functions
```

```{code-cell} ipython3
nested_func = simplified_pipeline_complex.functions[-1]
print(f"{nested_func.parameters=}, {nested_func.output_name=}, {nested_func(a=1, b=2, c=3, d=4)=}")
nested_func.pipeline.visualize()
```
