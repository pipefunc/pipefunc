# Understanding `mapspec`

`mapspec` is a powerful string-based syntax within `pipefunc` that defines how data is mapped between functions in a pipeline, especially when dealing with arrays or lists of inputs.
It allows you to express element-wise operations, reductions, and even the creation of new dimensions, enabling efficient parallel and vectorized computations.

## Basic Syntax

The general format of a `mapspec` string is:

```
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

**Pattern:** `x[i, j] -> y[i]`

**Description:** This pattern reduces a dimension in the output by combining elements across a particular index.

**Example:** Summing the rows of a matrix.

```{code-cell} ipython3
from pipefunc import pipefunc, Pipeline
import numpy as np

@pipefunc("y", mapspec="x[i, j] -> y[i]")
def sum_rows(x):
    return np.sum(x) # sum across the columns

pipeline = Pipeline([sum_rows])
result = pipeline.map({"x": np.array([[1, 2, 3], [4, 5, 6]])})
print(result["y"].output)
```

**Diagram:**

```{mermaid}
graph LR
    subgraph "Reduction across j (x[i,j] -> y[i])"
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
result = pipeline.map({"x": np.array([1, 2, 3]), "y": np.array([4, 5, 6]), "z": np.array([7, 8])})
print(result["r"].output)
```

**Diagram:**

```{mermaid}
graph TD
    subgraph "Zipped Inputs (x[a], y[a], z[b] -> r[a,b])"
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
        K["r[1,0] = 18"]:::rNodes
        L["r[1,1] = 19"]:::rNodes
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

## Limitations within `NestedPipeFunc`

When using `mapspec` within functions nested inside a {class}`~pipefunc.NestedPipeFunc` (created using {meth}`~pipefunc.Pipeline.nest_funcs`), the following restrictions apply:

- **No Reductions:** You cannot use `mapspec` patterns that reduce dimensions (e.g., `x[i, j] -> y[i]`).
- **No Dynamic Axis Generation:** You cannot use `mapspec` patterns that dynamically generate new axes (e.g., `... -> x[i]`) or have an `internal_shape`.

These limitations exist because a nested pipeline is treated as a single unit, and its internal `mapspec` operations are not directly visible to the outer pipeline.

## Tips and Best Practices

- **Start Simple:** Begin with basic element-wise mappings and gradually move to more complex patterns.
- **Visualize:** Use the `pipeline.visualize()` method and the diagrams shown above to understand how data flows through your pipeline.
- **Use Descriptive Indices:** Choose index names that are meaningful in the context of your data (e.g., `row`, `col`, `channel`, `time`).
- **Modularize:** Break down complex mappings into smaller, more manageable functions.
- **Test Thoroughly:** Verify that your `mapspec` strings produce the expected output shapes and values, especially when dealing with reductions or dynamic axis generation.

## Conclusion

`mapspec` is a powerful tool for defining data mappings in `pipefunc` pipelines. By understanding its syntax and common patterns, you can create efficient and expressive parallel computations. Remember to consider the limitations when using `mapspec` within nested pipelines, and use the visualization tools and diagrams to help you design and debug your workflows.
