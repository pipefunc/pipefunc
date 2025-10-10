---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Error Handling

```{try-notebook}
```

```{contents} ToC
:depth: 2
```

## What is the `ErrorSnapshot` feature in `pipefunc`?

The {class}`~pipefunc.ErrorSnapshot` feature captures detailed information about errors occurring during the execution of a `PipeFunc`. It aids in debugging by storing snapshots of error states, including the function, exception details, arguments, timestamp, and environment. This snapshot can be used to reproduce the error and examine the error context.

**Key Features:**

- **Error Details**: Captures function name, exception, arguments, traceback, user, machine, timestamp, and directory.
- **Reproduction**: Offers a `reproduce()` method to recreate the error with the stored arguments.
- **Persistence**: Allows saving and loading snapshots using `save_to_file` and `load_from_file`.

**Usage:**

1. **Accessing Snapshots**:

   ```python
   result = my_pipefunc_or_pipeline(args)
   if my_pipefunc_or_pipeline.error_snapshot:
       print(my_pipefunc_or_pipeline.error_snapshot)
   ```

2. **Reproducing Errors**:

   ```python
   error_snapshot = my_pipefunc_or_pipeline.error_snapshot
   if error_snapshot:
       error_snapshot.reproduce()
   ```

3. **Saving and Loading**:
   ```python
   error_snapshot.save_to_file("snapshot.pkl")
   loaded_snapshot = ErrorSnapshot.load_from_file("snapshot.pkl")
   ```

**Example:**

```{code-cell} ipython3
from pipefunc import pipefunc

@pipefunc(output_name="c")
def faulty_function(a, b):
    # Simulate an error
    raise ValueError("Intentional error")

try:
    faulty_function(a=1, b=2)
except Exception:
    snapshot = faulty_function.error_snapshot
    print(snapshot)
```

In the same way, for a {class}`~pipefunc.Pipeline` we can also access the error snapshot of the last failed function using the `error_snapshot` attribute.

```{code-cell} ipython3
from pipefunc import Pipeline

pipeline = Pipeline([faulty_function])
try:
    pipeline(a=1, b=2)
except Exception:
    snapshot = pipeline.error_snapshot
    print(snapshot)
```

{class}`~pipefunc.ErrorSnapshot` is very useful for debugging complex pipelines, making it easy to replicate and understand issues as they occur.

## Continue Mode in Parallel and Async Runs

`Pipeline.map(..., error_handling="continue")` behaves consistently whether work
is executed sequentially, through `parallel=True` with an executor
(`ThreadPoolExecutor`, `ProcessPoolExecutor`, etc.), or via
`pipeline.map_async`. Every failing element becomes an
{class}`~pipefunc.ErrorSnapshot`, and the snapshot is written back to the correct
index even when the work was processed in a chunked executor task.

```{code-cell} ipython3
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pipefunc import Pipeline, pipefunc

@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def double_or_fail(x: int) -> int:
    if x == 13:
        raise ValueError("boom")
    return x * 2

pipeline = Pipeline([double_or_fail])
inputs = {"x": list(range(20))}

with ThreadPoolExecutor(max_workers=4) as executor:
    parallel_result = pipeline.map(
        inputs,
        parallel=True,
        executor=executor,
        chunksizes=6,
        error_handling="continue",
    )


async def run_async() -> tuple[object, object]:
    with ThreadPoolExecutor(max_workers=4) as async_executor:
        async_result = pipeline.map_async(
            inputs,
            executor=async_executor,
            chunksizes=5,
            error_handling="continue",
        )
        async_outputs = await async_result.task
    return parallel_result["y"].output[13], async_outputs["y"].output[13]


try:
    parallel_err, async_err = asyncio.run(run_async())
except RuntimeError:
    parallel_err, async_err = await run_async()
parallel_err, async_err
```

This means downstream pipeline steps can trust that per-index error metadata is
aligned, regardless of the execution strategy or chunk size.

## Basic Continue Mode Example

The same guarantees apply in simpler element-wise pipelines. When
`error_handling="continue"`, failures turn into
{class}`~pipefunc.ErrorSnapshot` objects and downstream map steps receive the
snapshots instead of hard failures.

```{code-cell} ipython3
from pipefunc import Pipeline, pipefunc

@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def may_fail(x: int) -> int:
    if x == 3:
        raise ValueError("Cannot process 3")
    return x * 2

@pipefunc(output_name="z", mapspec="y[i] -> z[i]")
def add_ten(y: int) -> int:
    return y + 10

pipeline_basic = Pipeline([may_fail, add_ten])

result_basic = pipeline_basic.map(
    {"x": [1, 2, 3, 4, 5]},
    error_handling="continue",
)

result_basic["y"].output, result_basic["z"].output
```

```{code-cell} ipython3
from pipefunc.exceptions import ErrorSnapshot, PropagatedErrorSnapshot

y_outputs = result_basic["y"].output
z_outputs = result_basic["z"].output

type(y_outputs[2]), type(z_outputs[2])
```

## ErrorSnapshot capabilities

{class}`~pipefunc.ErrorSnapshot` instances store rich debugging context.

```{code-cell} ipython3
snapshot = y_outputs[2]
print(snapshot)
print("\nReproduce raises the same exception:")
try:
    snapshot.reproduce()
except Exception as exc:  # noqa: BLE001
    print(type(exc), exc)
```

## PropagatedErrorSnapshot overview

Functions downstream of an error receive
{class}`~pipefunc.PropagatedErrorSnapshot` objects describing what inputs were
problematic.

```{code-cell} ipython3
propagated = z_outputs[2]
print(propagated)
propagated.get_root_causes()
```

## Error propagation patterns

### Element-wise operations

```{code-cell} ipython3
import numpy as np

@pipefunc(output_name="doubled", mapspec="x[i] -> doubled[i]")
def double(x: int) -> int:
    if x < 0:
        raise ValueError(f"Negative value: {x}")
    return x * 2

@pipefunc(output_name="squared", mapspec="doubled[i] -> squared[i]")
def square(doubled: int) -> int:
    return doubled**2

pipeline_elements = Pipeline([double, square])
result_elements = pipeline_elements.map(
    {"x": [1, -2, 3, -4, 5]},
    error_handling="continue",
)

result_elements["doubled"].output, result_elements["squared"].output
```

### Reduction operations

```{code-cell} ipython3
@pipefunc(output_name="matrix", mapspec="x[i], y[j] -> matrix[i, j]")
def compute(x: int, y: int) -> int:
    if x == 2 and y == 2:
        raise ValueError("Cannot compute (2, 2)")
    return x * y

@pipefunc(output_name="row_sums", mapspec="matrix[i, :] -> row_sums[i]")
def sum_row(matrix: np.ndarray) -> int:
    return int(np.sum(matrix))

pipeline_rows = Pipeline([compute, sum_row])
result_rows = pipeline_rows.map(
    {"x": [1, 2, 3], "y": [1, 2, 3]},
    error_handling="continue",
)

result_rows["matrix"].output, result_rows["row_sums"].output
```

### Full array operations

```{code-cell} ipython3
@pipefunc(output_name="values", mapspec="x[i] -> values[i]")
def process(x: int) -> int:
    if x == 3:
        raise ValueError("Bad value")
    return x * 10

@pipefunc(output_name="total")
def sum_all(values: np.ndarray) -> int:
    return int(np.sum(values))

pipeline_total = Pipeline([process, sum_all])
result_total = pipeline_total.map(
    {"x": [1, 2, 3, 4]},
    error_handling="continue",
)

result_total["values"].output, result_total["total"].output
```
