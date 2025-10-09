# Error Handling in pipefunc

## Overview

The `error_handling` parameter in `Pipeline.map()` allows pipelines to continue processing even when some function calls fail. When set to `"continue"`, errors are captured as `ErrorSnapshot` objects instead of being raised, and downstream functions that depend on error results are skipped with `PropagatedErrorSnapshot` objects.

## Basic Usage

```python
from pipefunc import Pipeline, pipefunc
import numpy as np

@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def may_fail(x: int) -> int:
    if x == 3:
        raise ValueError("Cannot process 3")
    return x * 2

@pipefunc(output_name="z", mapspec="y[i] -> z[i]")
def add_ten(y: int) -> int:
    return y + 10

# Create pipeline
pipeline = Pipeline([may_fail, add_ten])

# Run with error handling
result = pipeline.map(
    {"x": [1, 2, 3, 4, 5]},
    error_handling="continue"  # Default is "raise"
)

# Access results
y = result["y"].output  # np.array([2, 4, ErrorSnapshot, 8, 10], dtype=object)
z = result["z"].output  # np.array([12, 14, PropagatedErrorSnapshot, 18, 20], dtype=object)

# Check for errors
for i, val in enumerate(y):
    if isinstance(val, ErrorSnapshot):
        print(f"Error at index {i}: {val.exception}")
```

## Error Types

### ErrorSnapshot

Represents a function call that failed with an exception:

```python
from pipefunc.exceptions import ErrorSnapshot

# ErrorSnapshot contains:
# - function: The function that failed
# - exception: The exception that was raised
# - args: Arguments passed to the function
# - kwargs: Keyword arguments passed to the function
# - traceback: Full traceback as a string
# - timestamp: When the error occurred
# - user, machine, ip_address: System information

# You can reproduce the error:
error_snapshot.reproduce()  # Will raise the same exception

# Or inspect it:
print(error_snapshot)  # Pretty-printed error information
print(error_snapshot.traceback)  # Full traceback

# Save/load for debugging:
error_snapshot.save_to_file("error.pkl")
loaded = ErrorSnapshot.load_from_file("error.pkl")
```

### PropagatedErrorSnapshot

Represents a function that was skipped because its inputs contained errors:

```python
from pipefunc.exceptions import PropagatedErrorSnapshot

# PropagatedErrorSnapshot contains:
# - error_info: Information about which parameters had errors
# - skipped_function: The function that was skipped
# - reason: Why it was skipped ("input_is_error", "array_contains_errors")
# - attempted_kwargs: The non-error arguments that would have been passed

# Get root cause errors:
root_errors = propagated_error.get_root_causes()  # List of original ErrorSnapshots
```

## Error Propagation Patterns

### Element-wise Operations

When using element-wise mapspecs, errors propagate individually:

```python
@pipefunc(output_name="doubled", mapspec="x[i] -> doubled[i]")
def double(x: int) -> int:
    if x < 0:
        raise ValueError(f"Negative value: {x}")
    return x * 2

@pipefunc(output_name="squared", mapspec="doubled[i] -> squared[i]")
def square(doubled: int) -> int:
    return doubled ** 2

pipeline = Pipeline([double, square])
result = pipeline.map({"x": [1, -2, 3, -4, 5]}, error_handling="continue")

# doubled = [2, ErrorSnapshot, 6, ErrorSnapshot, 10]
# squared = [4, PropagatedErrorSnapshot, 36, PropagatedErrorSnapshot, 100]
```

### Reduction Operations

When a function receives an array slice containing errors, it's skipped entirely:

```python
@pipefunc(output_name="matrix", mapspec="x[i], y[j] -> matrix[i, j]")
def compute(x: int, y: int) -> int:
    if x == 2 and y == 2:
        raise ValueError("Cannot compute (2, 2)")
    return x * y

@pipefunc(output_name="row_sums", mapspec="matrix[i, :] -> row_sums[i]")
def sum_row(matrix: np.ndarray) -> int:
    return np.sum(matrix)

pipeline = Pipeline([compute, sum_row])
result = pipeline.map({"x": [1, 2, 3], "y": [1, 2, 3]}, error_handling="continue")

# matrix = [[1, 2, 3],
#           [2, ErrorSnapshot, 6],
#           [3, 6, 9]]
#
# row_sums = [6,                        # sum([1, 2, 3])
#             PropagatedErrorSnapshot,  # skipped: row contains error
#             18]                       # sum([3, 6, 9])
```

### Full Array Operations

Functions that receive entire arrays are skipped if any element contains an error:

```python
@pipefunc(output_name="values", mapspec="x[i] -> values[i]")
def process(x: int) -> int:
    if x == 3:
        raise ValueError("Bad value")
    return x * 10

@pipefunc(output_name="total")
def sum_all(values: np.ndarray) -> int:
    return np.sum(values)

pipeline = Pipeline([process, sum_all])
result = pipeline.map({"x": [1, 2, 3, 4]}, error_handling="continue")

# values = [10, 20, ErrorSnapshot, 40]
# total = PropagatedErrorSnapshot  # Entire function skipped
```

## Parallel and Async Execution

`error_handling="continue"` behaves the same in sequential runs, threaded
executors, process pools, and `pipeline.map_async`. When a chunk of parallel work
fails, each element in that chunk is recorded as an `ErrorSnapshot`, so the
result arrays keep the right index-to-error mapping.

```python
from concurrent.futures import ThreadPoolExecutor

inputs = {"x": list(range(12))}

with ThreadPoolExecutor(max_workers=4) as executor:
    parallel_result = pipeline.map(
        inputs,
        parallel=True,
        executor=executor,
        chunksizes=6,
        error_handling="continue",
    )

async_result = pipeline.map_async(
    inputs,
    executor=ThreadPoolExecutor(max_workers=4),
    chunksizes=4,
    error_handling="continue",
)

# In both runs, `parallel_result["y"].output[13]` and `await async_result.task`
# contain an ErrorSnapshot describing `x == 13` without shifting other indices.
```

## Working with Mixed Results

When `error_handling="continue"`, arrays use object dtype to store mixed types:

```python
result = pipeline.map(inputs, error_handling="continue")
output = result["output_name"].output

# Filter out errors to get valid results
valid_results = [val for val in output if not isinstance(val, (ErrorSnapshot, PropagatedErrorSnapshot))]

# Count errors
error_count = sum(1 for val in output if isinstance(val, (ErrorSnapshot, PropagatedErrorSnapshot)))

# Get error details
for i, val in enumerate(output):
    if isinstance(val, ErrorSnapshot):
        print(f"Error at {i}: {val.exception}")
    elif isinstance(val, PropagatedErrorSnapshot):
        print(f"Skipped at {i}: {val.reason}")
        # Get root causes
        for root_error in val.get_root_causes():
            print(f"  Caused by: {root_error.exception}")
```

## Storage Backend Compatibility

Error handling works with different storage backends:

```python
# Dict storage (default) - preserves error objects
result = pipeline.map(inputs, error_handling="continue", storage="dict")

# File storage - may not preserve error objects (stores as None)
result = pipeline.map(inputs, error_handling="continue", storage="file_array")

# Zarr storage - may not preserve error objects
result = pipeline.map(inputs, error_handling="continue", storage="zarr_memory")
```

Note: Some storage backends may not fully support storing Python objects like ErrorSnapshot. In these cases, errors may be stored as None or raise warnings.

## Best Practices

1. **Check for errors before processing results**: Always verify if outputs contain errors before using them.

2. **Use type casting carefully**: When using `result.type_cast()`, be aware that arrays with errors cannot be cast to numeric types.

3. **Error recovery**: For critical pipelines, implement error recovery logic:
   ```python
   if isinstance(result, ErrorSnapshot):
       # Try alternative processing or use default value
       result = default_value
   ```

4. **Logging errors**: Collect and log all errors for debugging:
   ```python
   errors = []
   for name, res in result.items():
       output = res.output
       if isinstance(output, np.ndarray):
           for i, val in enumerate(output):
               if isinstance(val, ErrorSnapshot):
                   errors.append((name, i, val))
   ```

5. **Partial results**: Even with errors, you can often extract useful partial results:
   ```python
   # Get all successful computations
   successful = result["output"].output[~pd.isna(result["output"].output)]
   ```

## Performance Considerations

- Arrays with `error_handling="continue"` use object dtype, which is slower than numeric arrays
- Error checking adds overhead to each function call
- Consider using `error_handling="raise"` (default) for production when errors should halt execution

## Debugging

To debug errors after a pipeline run:

```python
# Find all errors in results
for output_name, result in results.items():
    if isinstance(result.output, np.ndarray):
        errors = [(i, e) for i, e in enumerate(result.output)
                  if isinstance(e, ErrorSnapshot)]
        for i, error in errors:
            print(f"\n{output_name}[{i}] failed:")
            print(error)  # Pretty-printed error info
            # print(error.traceback)  # Full traceback if needed

# Save problematic error for later analysis
error.save_to_file(f"error_{output_name}_{i}.pkl")
```
