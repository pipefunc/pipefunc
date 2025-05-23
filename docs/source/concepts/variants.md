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

# Variants

```{try-notebook}
```

```{contents} ToC
:depth: 2
```

## What is `VariantPipeline` and how to use it?

{class}`pipefunc.VariantPipeline` allows creating pipelines with alternative implementations (variants) of functions. This is useful when you want to experiment with different implementations without creating separate pipelines.

Here's a simple example:

```{code-cell} ipython3
from pipefunc import VariantPipeline, pipefunc

@pipefunc(output_name="c", variant="A")
def f(a, b):
    return a + b

@pipefunc(output_name="c", variant="B")
def f_alt(a, b):
    return a - b

@pipefunc(output_name="d")
def g(b, c):
    return b * c

# Create pipeline with default variant
pipeline = VariantPipeline([f, f_alt, g], default_variant="A")

# Get a regular Pipeline with variant A
pipeline_A = pipeline.with_variant()  # uses default variant
result_A = pipeline_A(a=2, b=3)  # (2 + 3) * 3 = 15

# Get a regular Pipeline with variant B
pipeline_B = pipeline.with_variant(select="B")
result_B = pipeline_B(a=2, b=3)  # (2 - 3) * 3 = -3
```

For more complex cases, you can group variants using a dictionary:

```{code-cell} ipython3
@pipefunc(output_name="c", variant={"method": "add"})
def process_A(a, b):
    return a + b

@pipefunc(output_name="b", variant={"method": "sub"})
def process_B1(a):
    return a

@pipefunc(output_name="c", variant={"method": "sub"})
def process_B2(a, b):
    return a - b

@pipefunc(output_name="d", variant={"analysis": "mul"})
def analyze_A(b, c):
    return b * c

@pipefunc(output_name="d", variant={"analysis": "div"})
def analyze_B(b, c):
    return b / c

pipeline = VariantPipeline(
    [process_A, process_B1, process_B2, analyze_A, analyze_B],
    default_variant={"method": "add", "analysis": "mul"}
)

# Select specific variants for each group
sub_div_pipeline = pipeline.with_variant(
    select={"method": "sub", "analysis": "div"}
)
```

Here, we see that the `variant={"method": "add"}` in for `process_A` and `variant={"method": "sub"}` for `process_B1` and `process_B2` define alternative pipelines.

You can visualize the pipelines using the `visualize` method:

```{code-cell} ipython3
pipeline.visualize(backend="graphviz")
```

This will include dropdowns for each variant group, allowing you to select the specific variant you want to visualize.

:::{admonition} The interactive widgets do not work in the documentation
They are only functional in live Jupyter environments.
:::

You can inspect available variants using `variants_mapping()`:

```{code-cell} ipython3
pipeline.variants_mapping()
```

Variants in the same group can have different output names:

```{code-cell} ipython3
@pipefunc(output_name="stats_result", variant={"analysis": "stats"})
def analyze_stats(data):
    # Perform statistical analysis
    return ...

@pipefunc(output_name="ml_result", variant={"analysis": "ml"})
def analyze_ml(data):
    # Perform machine learning analysis
    return ...

# The output name to use depends on which variant is selected
pipeline = VariantPipeline([analyze_stats, analyze_ml])
pipeline_stats = pipeline.with_variant(select={"analysis": "stats"})
result = pipeline_stats("stats_result", data={...})

pipeline_ml = pipeline.with_variant(select={"analysis": "ml"})
result = pipeline_ml("ml_result", data={...})
```

Key features:

- Define multiple implementations of a function using the `variant` parameter
- Group related variants using dictionary keys in the `variant` parameter
- Specify defaults with `default_variant`
- Get a regular `Pipeline` when variants are selected
- No changes required to your existing functions

The `with_variant()` method returns either:

- A regular `Pipeline` if all variants are resolved
- Another `VariantPipeline` if some variants remain unselected

Also check out {class}`pipefunc.VariantPipeline.from_pipelines` to create a `VariantPipeline` from multiple `Pipeline` objects without having to specify `variant` for each function.

This makes `VariantPipeline` ideal for:

- A/B testing different implementations
- Experimenting with algorithm variations
- Managing multiple processing options
- Creating flexible, configurable pipelines
