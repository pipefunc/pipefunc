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
    async_result = pipeline.map_async(
        inputs,
        executor=ThreadPoolExecutor(max_workers=4),
        chunksizes=5,
        error_handling="continue",
    )
    async_outputs = await async_result.task
    return parallel_result["y"].output[13], async_outputs["y"].output[13]


parallel_err, async_err = asyncio.run(run_async())
parallel_err, async_err
```

This means downstream pipeline steps can trust that per-index error metadata is
aligned, regardless of the execution strategy or chunk size.
