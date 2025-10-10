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
