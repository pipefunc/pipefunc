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

## Overview

`pipefunc` exposes two error-handling strategies when running pipelines:

- `error_handling="raise"` (default): stop on the first failure and raise the
  underlying exception.
- `error_handling="continue"`: convert failures raised inside user-defined
  {class}`~pipefunc.PipeFunc` callables into
  {class}`~pipefunc.ErrorSnapshot` objects so the run can continue and
  downstream tasks can inspect the failure. Infrastructure issues (e.g. pickling
  the return value, shape validation, storage I/O) still abort the run regardless
  of the setting.

The sections below show how to inspect snapshots, how propagation works, and
why the behaviour is consistent across synchronous, parallel, and async runs.

## Capturing error snapshots

When a {class}`~pipefunc.PipeFunc` fails, it stores an
{class}`~pipefunc.ErrorSnapshot` on the callable itself and on the owning
pipeline. The attribute reflects the most recent failure on that
{class}`PipeFunc`. When multiple errors occur (for example in threaded or
process-based executors) each failing invocation also returns its own snapshot,
so the per-element objects in the result record the correct kwargs and
exceptions even if the shared attribute is overwritten by later failures.

```{code-cell} ipython3
from pipefunc import Pipeline, pipefunc
from pipefunc.exceptions import ErrorSnapshot

@pipefunc(output_name="c")
def faulty_function(a: int, b: int) -> int:
    raise ValueError("Intentional error")

pipeline = Pipeline([faulty_function])

try:
    pipeline(a=1, b=2)
except Exception:  # noqa: BLE001
    func_snapshot = faulty_function.error_snapshot
    pipeline_snapshot = pipeline.error_snapshot

func_snapshot is pipeline_snapshot, isinstance(func_snapshot, ErrorSnapshot)
```

`ErrorSnapshot` captures the function, arguments, traceback, and useful helper
methods such as {meth}`ErrorSnapshot.reproduce`:

```{code-cell} ipython3
print(func_snapshot)

try:
    func_snapshot.reproduce()
except Exception as exc:  # noqa: BLE001
    type(exc), exc
```

## Continue mode walkthrough

One small example covers the core concepts: failing elements turn into
`ErrorSnapshot` objects, and downstream functions see
{class}`~pipefunc.PropagatedErrorSnapshot` placeholders when their inputs
contain errors.

```{code-cell} ipython3
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

y_outputs = result_basic["y"].output
z_outputs = result_basic["z"].output

y_outputs, z_outputs
```

The error for `x == 3` hydrates two complementary objects:

```{code-cell} ipython3
from pipefunc.exceptions import PropagatedErrorSnapshot

type(y_outputs[2]), type(z_outputs[2])
```

```{code-cell} ipython3
snapshot = y_outputs[2]
print(snapshot)
snapshot.kwargs
```

Downstream code can walk back to the original failures:

```{code-cell} ipython3
propagated = z_outputs[2]
propagated.get_root_causes()
```

## Parallel and async consistency

`continue` mode produces the same per-index snapshots even when work is chunked
across executors or awaited asynchronously.

```{code-cell} ipython3
from concurrent.futures import ThreadPoolExecutor

inputs = {"x": list(range(10))}
parallel_outputs = pipeline_basic.map(
    inputs,
    parallel=True,
    executor=ThreadPoolExecutor(),
    chunksizes=4,
    error_handling="continue",
)

async_result = pipeline_basic.map_async(
    inputs,
    executor=ThreadPoolExecutor(),
    chunksizes=4,
    error_handling="continue",
)
results = await async_result.task
async_outputs = results["y"].output
```

## Recap

- `error_handling="raise"` aborts immediately; `"continue"` records
  `ErrorSnapshot` objects while allowing the run to finish.
- `ErrorSnapshot` instances preserve everything needed to reproduce the
  exception locally or offline.
- {class}`~pipefunc.PropagatedErrorSnapshot` highlights which downstream inputs
  contained errors and lets you walk back to the root causes.
- The semantics are identical for sequential, threaded, process-based, and
  async execution so downstream code can rely on consistent error metadata.

```{note}
Reduction limitation: For reductions where a function receives an array that
contains one or more errors (reason: `array_contains_errors`),
{meth}`~pipefunc.PropagatedErrorSnapshot.get_root_causes` currently returns
an empty list. Root-cause enumeration is only available when the upstream input
parameter itself is a single error ("full" case). You can still use the
`reason` and `error_info` metadata to detect which inputs contained errors.
```
