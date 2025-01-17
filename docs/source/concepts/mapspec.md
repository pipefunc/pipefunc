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

# Understanding `mapspec`

```{try-notebook}
```

```{contents} ToC
:depth: 2
```

`mapspec` is a powerful string-based syntax within `pipefunc` that defines how data is mapped between functions in a pipeline, especially when dealing with arrays or lists of inputs.
It allows you to express element-wise operations, reductions, and even the creation of new dimensions, enabling **parallel computations**.

:::{admonition} Go through the main tutorial first!
:class: warning, dropdown

We recommend going through the main [pipefunc tutorial](../tutorial) before diving into `mapspec`, as it provides a comprehensive overview of the `pipefunc` library and its core concepts.

:::

## Basic Syntax

The general format of a `mapspec` string is:

```python
"input1[index1, index2, ...], input2[index3, ...] -> output1[index4, ...], output2[index4, ...]"
```

**Components:**

- **Inputs:** `input1`, `input2`, etc. are the names of input arguments to the `@pipefunc`-decorated function.
- **Outputs:** `output1`, `output2`, etc. are the names of outputs produced by the function. These names must match the `output_name` specified in the `@pipefunc` decorator.
- **Indices:** `index1`, `index2`, etc. are single-letter indices (typically `i`, `j`, `k`, `a`, `b`, etc.) that represent dimensions or elements within the input and output arrays.
- **`->`:** The arrow separates the input side from the output side.
- **`...`:** The ellipsis is a special index that represents implicit inputs for functions that produce a dynamic number of outputs.

**Assumptions:**

- `mapspec` assumes that inputs and outputs are array-like objects that can be indexed using the specified indices (e.g., NumPy arrays, lists of lists).

## Common `mapspec` Patterns

Let's explore common `mapspec` patterns with examples and Mermaid diagrams to illustrate the mappings.

### 1. Element-wise Operations

**Pattern:** `x[i] -> y[i]`

**Description:** This pattern applies a function element by element. Each element `x[i]` from the input `x` is used to compute the corresponding element `y[i]` in the output `y`.

**Example:** Doubling each element of an array.

```{code-cell} ipython3
from pipefunc import pipefunc, Pipeline
import numpy as np

@pipefunc("y", mapspec="x[i] -> y[i]")
def double(x):
    return 2 * x

pipeline = Pipeline([double])
result = pipeline.map({"x": np.array([1, 2, 3, 4])})
print(result["y"].output)
```

**Diagram:**

```{mermaid}
graph LR
    subgraph "Element-wise Operation (x[i] -> y[i])"
        direction LR
        %% Style definitions
        classDef xNodes fill:#fff3d4,stroke:#d68a00,stroke-width:2px,color:#000
        classDef yNodes fill:#f0f0ff,stroke:#0000cc,stroke-width:2px,color:#000

        A1["x[0] = 1"]:::xNodes --i--> B1["y[0] = 2"]:::yNodes
        A2["x[1] = 2"]:::xNodes --i--> B2["y[1] = 4"]:::yNodes
        A3["x[2] = 3"]:::xNodes --i--> B3["y[2] = 6"]:::yNodes
        A4["x[3] = 4"]:::xNodes --i--> B4["y[3] = 8"]:::yNodes
    end
```

### 2. Multi-dimensional Mapping

**Pattern:** `x[i], y[j] -> z[i, j]`

**Description:** This pattern creates a multi-dimensional output `z` by combining elements from multiple inputs `x` and `y` based on their indices.

**Example:** Computing the outer product of two vectors.

```{code-cell} ipython3
from pipefunc import pipefunc, Pipeline
import numpy as np

@pipefunc("z", mapspec="x[i], y[j] -> z[i, j]")
def outer_product(x, y):
    return x * y

pipeline = Pipeline([outer_product])
result = pipeline.map({"x": np.array([1, 2, 3]), "y": np.array([4, 5])})
print(result["z"].output)
```

**Diagram:**

```{mermaid}
graph LR
    subgraph "Multi-dimensional Mapping (x[i], y[j] -> z[i,j])"
        direction LR
        %% Style definitions
        classDef xNodes fill:#fff3d4,stroke:#d68a00,stroke-width:2px,color:#000
        classDef yNodes fill:#d4f3e6,stroke:#2d8659,stroke-width:2px,color:#000
        classDef zNodes fill:#f0f0ff,stroke:#0000cc,stroke-width:2px,color:#000

        A["x[0] = 1"]:::xNodes;
        B["x[1] = 2"]:::xNodes;
        C["x[2] = 3"]:::xNodes;
        D["y[0] = 4"]:::yNodes;
        E["y[1] = 5"]:::yNodes;
        A --"i"--> F["z[0,0] = 4"]:::zNodes;
        A --"i"--> G["z[0,1] = 5"]:::zNodes;
        B --"i"--> H["z[1,0] = 8"]:::zNodes;
        B --"i"--> I["z[1,1] = 10"]:::zNodes;
        C --"i"--> J["z[2,0] = 12"]:::zNodes;
        C --"i"--> K["z[2,1] = 15"]:::zNodes;
        D --"j"--> F;
        E --"j"--> G;
        D --"j"--> H;
        E --"j"--> I;
        D --"j"--> J;
        E --"j"--> K;

        %% Style for i connections (orange, solid)
        linkStyle 0,1,2,3,4,5 stroke:#d68a00,stroke-width:2px
        %% Style for j connections (green, dashed)
        linkStyle 6,7,8,9,10,11 stroke:#2d8659,stroke-width:2px,stroke-dasharray: 5 5
    end
```

### 3. Reductions

**Pattern:** `x[i, :] -> y[i]`

**Description:** This pattern reduces a dimension in the output by combining elements across a particular index.

**Example:** Summing the rows of a matrix.

```{code-cell} ipython3
from pipefunc import pipefunc, Pipeline
import numpy as np

@pipefunc("y", mapspec="x[i, :] -> y[i]")
def sum_rows(x):
    return np.sum(x) # sum across the rows

pipeline = Pipeline([sum_rows])
result = pipeline.map({"x": np.array([[1, 2, 3], [4, 5, 6]])})
print(result["y"].output)
```

**Diagram:**

```{mermaid}
graph LR
    subgraph "Reduction across j (x[i, :] -> y[i])"
        direction LR
        %% Style definitions
        classDef xNodes fill:#fff3d4,stroke:#d68a00,stroke-width:2px,color:#000
        classDef yNodes fill:#f0f0ff,stroke:#0000cc,stroke-width:2px,color:#000

        A["x[0,0] = 1"]:::xNodes
        B["x[0,1] = 2"]:::xNodes
        C["x[0,2] = 3"]:::xNodes
        D["x[1,0] = 4"]:::xNodes
        E["x[1,1] = 5"]:::xNodes
        F["x[1,2] = 6"]:::xNodes
        G["y[0] = 6"]:::yNodes
        H["y[1] = 15"]:::yNodes

        A --"j"--> G
        B --"j"--> G
        C --"j"--> G
        D --"j"--> H
        E --"j"--> H
        F --"j"--> H

        %% Style for j connections
        linkStyle 0,1,2,3,4,5 stroke:#2d8659,stroke-width:2px,stroke-dasharray: 5 5
    end
```

### 4. Dynamic Axis Generation

**Pattern:** `... -> x[i]`

**Description:** This pattern generates a new axis (dimension) in the output `x`. The ellipsis (`...`) indicates that the function conceptually takes some implicit input and produces an output with an unknown or dynamic number of elements.

**Example:** Creating a list of items.

```{code-cell} ipython3
from pipefunc import pipefunc, Pipeline

@pipefunc("x", mapspec="... -> x[i]")
def generate_items(n):
    return list(range(n))

pipeline = Pipeline([generate_items])
result = pipeline.map({"n": 5}, internal_shapes={"x": (5,)})  # internal_shapes is optional
print(result["x"].output)
```

**Diagram:**

```{mermaid}
graph LR
    subgraph "Dynamic Axis Generation (... -> x[i])"
        direction LR
        %% Style definitions
        classDef implicitNode fill:#e6e6e6,stroke:#666,stroke-width:2px,color:#000
        classDef xNodes fill:#f0f0ff,stroke:#0000cc,stroke-width:2px,color:#000

        A["(implicit input)"]:::implicitNode
        A --i--> B["x[0] = 0"]:::xNodes
        A --i--> C["x[1] = 1"]:::xNodes
        A --i--> D["x[2] = 2"]:::xNodes
        A --i--> E["x[3] = 3"]:::xNodes
        A --i--> F["x[4] = 4"]:::xNodes

        %% Style for i connections
        linkStyle 0,1,2,3,4 stroke:#666,stroke-width:2px
    end
```

### 5. Zipped Inputs

**Pattern:** `x[a], y[a], z[b] -> r[a, b]`

**Description:** This pattern processes elements from multiple lists `x`, `y` (zipped together), and `z` independently, combining them based on their indices.

**Example:**

```{code-cell} ipython3
from pipefunc import pipefunc, Pipeline
import numpy as np

@pipefunc("r", mapspec="x[a], y[a], z[b] -> r[a, b]")
def process_zipped(x, y, z):
    return x * y + z

pipeline = Pipeline([process_zipped])
result = pipeline.map(
    {"x": np.array([1, 2, 3]), "y": np.array([4, 5, 6]), "z": np.array([7, 8])},
)
print(result["r"].output)
```

**Diagram:**

```{mermaid}
graph LR
    subgraph "Zipped Inputs (x[a], y[a], z[b] -> r[a,b])"
        direction LR
        %% Style definitions
        classDef xNodes fill:#fff3d4,stroke:#d68a00,stroke-width:2px,color:#000
        classDef yNodes fill:#d4f3e6,stroke:#2d8659,stroke-width:2px,color:#000
        classDef zNodes fill:#ffe6e6,stroke:#cc0000,stroke-width:2px,color:#000
        classDef rNodes fill:#f0f0ff,stroke:#0000cc,stroke-width:2px,color:#000

        A["x[0] = 1"]:::xNodes
        B["x[1] = 2"]:::xNodes
        C["x[2] = 3"]:::xNodes
        D["y[0] = 4"]:::yNodes
        E["y[1] = 5"]:::yNodes
        F["y[2] = 6"]:::yNodes
        G["z[0] = 7"]:::zNodes
        H["z[1] = 8"]:::zNodes

        I["r[0,0] = 11"]:::rNodes
        J["r[0,1] = 12"]:::rNodes
        K["r[1,0] = 17"]:::rNodes
        L["r[1,1] = 18"]:::rNodes
        M["r[2,0] = 25"]:::rNodes
        N["r[2,1] = 26"]:::rNodes

        A --"a"--> I & J
        B --"a"--> K & L
        C --"a"--> M & N
        D --"a"--> I & J
        E --"a"--> K & L
        F --"a"--> M & N
        G --"b"--> I & K & M
        H --"b"--> J & L & N

        %% Style for a connections (orange, solid)
        linkStyle 0,1,2,3,4,5,6,7,8,9,10,11 stroke:#d68a00,stroke-width:2px
        %% Style for b connections (red, dashed)
        linkStyle 12,13,14,15,16,17 stroke:#cc0000,stroke-width:2px,stroke-dasharray: 5 5
    end
```

## `pipeline.add_mapspec_axis()` method

The `pipeline.add_mapspec_axis()` method offers a streamlined way to dynamically introduce or alter dimensions (axes) within your pipeline's `mapspec` without manually editing each function's `mapspec` string.
It automatically propagates these dimensional changes across selected functions, making it ideal for handling different multi-dimensional sweeps for different simulations.

**Example 1: Adding Axes to a Pipeline with No Initial `mapspec`**

Let's start with a simple pipeline that performs basic arithmetic operations without any `mapspec` defined:

```{code-cell} ipython3
from pipefunc import Pipeline, pipefunc

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

Initially, this pipeline processes single values. Now, let's say we want to introduce dimensions to our inputs and process arrays of data. We can use `add_mapspec_axis()` to add axes to `a` and `b` (zipping them together) and another independent axis to `x`.

```{code-cell} ipython3
# Add a zipped axis to "a" and "b"
pipeline.add_mapspec_axis("a", "b", axis="i")

# Add an independent axis to "x"
pipeline.add_mapspec_axis("x", axis="j")

# Check the generated mapspec strings
print(pipeline.mapspecs_as_strings)
pipeline.visualize()
```

**Explanation:**

1. **No Initial `mapspec`:** The functions `f`, `g`, and `h` initially operate on single values.
2. **`add_mapspec_axis("a", "b", axis="i")`:** This adds a new dimension indexed by `i` to both `a` and `b`, and since they are zipped, they will share the same index `i`. The `mapspec` strings are updated accordingly. For example, `f` now has `mapspec="a[i], b[i] -> c[i]"`.
3. **`add_mapspec_axis("x", axis="j")`:** This adds another dimension indexed by `j` to `x`. The `mapspec` of `g` and `h` are updated to include `x[j]`.
4. **Resulting `mapspec`:** The functions in the pipeline now have `mapspec` strings that reflect the added dimensions:
   - `f`: `"a[i], b[i] -> c[i]"`
   - `g`: `"b[i], c[i], x[j] -> d[i, j]"`
   - `h`: `"c[i], d[i, j], x[j] -> e[i, j]"`

Now, the pipeline can process 1D arrays of `a`, `b` and `x` values. The `i` index will iterate through the zipped `a` and `b` arrays, and the `j` index will iterate through the `x` array. The output `e` will be a 2D array with shape `(len(a), len(x))`.

**Running the Pipeline:**

```{code-cell} ipython3
import numpy as np

result = pipeline.map({"a": [1, 2], "b": [3, 4], "x": [5, 6]})
print(result["e"].output)
```

This will produce a 2x2 output array `e` where each element `e[i, j]` is the result of the pipeline operations on `a[i]`, `b[i]`, and `x[j]`.

**Example 2: Adding an Axis to a Variable Not Initially in `mapspec`**

Consider this pipeline, which involves doubling an input array `x` and then summing the results, with an additional parameter `b` not initially involved in the `mapspec`:

```{code-cell} ipython3
import numpy as np
from pipefunc import Pipeline, pipefunc
from pipefunc.typing import Array

@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def double_it(x: int, b: int) -> int:
    return 2 * x + b

@pipefunc(output_name="sum")  # no mapspec, so receives y[:] as input
def take_sum(y: Array[int]) -> int:
    assert isinstance(y, np.ndarray)
    return sum(y)

pipeline_map = Pipeline([double_it, take_sum])
pipeline_map.visualize()
```

Now, let's say we want to perform this operation for multiple values of `b`, effectively adding a new dimension to our computation. We can use `add_mapspec_axis()` to add an axis to `b`:

```{code-cell} ipython3
# Add an axis to "b"
pipeline_map.add_mapspec_axis("b", axis="j")

# Check the generated mapspec strings
print(pipeline_map.mapspecs_as_strings)
pipeline_map.visualize()
```

**Explanation:**

1. **Initial `mapspec`:** The `double_it` function has `mapspec="x[i] -> y[i]"`, indicating an element-wise operation on `x`. The `take_sum` function has no `mapspec`, so it receives the entire `y` array.
2. **`add_mapspec_axis("b", axis="j")`:** This adds a new dimension indexed by `j` to `b`. The `mapspec` strings are updated:
   - `double_it`: `"x[i], b[j] -> y[i, j]"`
   - `take_sum`: `"y[:, j] -> sum[j]"`
3. **New `mapspec` Behavior:** The pipeline now expects a 1D array of `b` values. The `double_it` function will iterate through `x` with index `i` and `b` with index `j`, producing a 2D output array `y` with shape `(len(x), len(b))`. The `take_sum` function will then sum the `y` array along the `i` axis, for each value of `j`, resulting in a 1D output array `sum` with shape `(len(b),)`.

**Running the Pipeline:**

```{code-cell} ipython3
result = pipeline_map.map({"x": np.array([1, 2, 3]), "b": np.array([10, 20])})
print(result["y"].output)
print(result["sum"].output)
```

This will produce:

- A 2D array `y` where each element `y[i, j]` is `2 * x[i] + b[j]`.
- A 1D array `sum` where each element `sum[j]` is the sum of `y` values along the `i` axis for the corresponding `b[j]`.

**Key Takeaway:**

{meth}`~pipefunc.Pipeline.add_mapspec_axis` simplifies introducing or modifying dimensions, especially when dealing with pipelines that have many functions or high-dimensional data.
It allows for easy extension of your pipeline's capabilities to handle multi-dimensional data by automatically managing `mapspec` changes, making your code more concise and adaptable.

## Tips and Best Practices

- **Start Simple:** Begin with basic element-wise mappings and gradually move to more complex patterns.
- **Visualize:** Use the `pipeline.visualize()` method and the diagrams shown above to understand how data flows through your pipeline.
- **Use Descriptive Indices:** Choose index names that are meaningful in the context of your data (e.g., `row`, `col`, `channel`, `time`).
- **Modularize:** Break down complex mappings into smaller, more manageable functions.
- **Test Thoroughly:** Verify that your `mapspec` strings produce the expected output shapes and values, especially when dealing with reductions or dynamic axis generation.

## Conclusion

`mapspec` is a powerful tool for defining data mappings in `pipefunc` pipelines.
By understanding its syntax and common patterns, you can create efficient and expressive parallel computations.
