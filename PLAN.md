# Error Handling Feature for map() Method

## Overview

We are adding error handling options to the `map` method in pipefunc. Currently, error behavior depends on the execution mode:

- **With `parallel=True`** (default): All tasks for a function are submitted to executors, so even if some fail, others continue executing. However, when iterating over results, the first exception encountered causes the pipeline to stop, preventing downstream functions from running even for successfully computed points.

- **With `parallel=False`**: Execution stops immediately at the first exception, preventing both the current function and all downstream functions from processing any remaining data points.

In both cases, partial results that were successfully computed cannot flow to downstream functions, even though those functions might be able to process the successful results.

## Design Goals

1. **Continue on Error**: Allow the pipeline to continue processing even when some individual function calls fail during mapping operations
2. **Error Collection**: Collect and store errors as `ErrorSnapshot` objects instead of immediately raising them
3. **Partial Results**: Enable subsequent pipeline steps to process successfully computed results even when some inputs failed
4. **User Control**: Provide an `error_handling` parameter with options "raise" (current behavior) or "continue"

## Current Implementation Status

Based on the git diff, the following has been implemented:

1. Added `error_handling: Literal["raise", "continue"] = "raise"` parameter to:
   - `Pipeline.map()`
   - `run_map()` and related functions
   - `RunInfo` dataclass
   - `handle_pipefunc_error()` function

2. Modified error handling logic to:
   - Return `None` instead of raising when `error_handling="continue"`
   - Store `ErrorSnapshot` objects in place of results when errors occur
   - Pass `error_handling` parameter through the execution chain

## Implementation Plan

### Phase 1: Core Error Handling (Partially Complete)
- [x] Add `error_handling` parameter to map() method signature
- [x] Thread parameter through RunInfo and execution functions
- [x] Modify `handle_pipefunc_error()` to support continue mode
- [x] Return ErrorSnapshot objects instead of raising in continue mode

### Phase 2: Array/Storage Handling (TODO)
- [ ] Handle ErrorSnapshot objects in array storage operations
- [ ] Ensure proper shape handling when some elements are errors
- [ ] Update `_update_array()` to handle mixed results/errors
- [ ] Handle ErrorSnapshot serialization for different storage backends

### Phase 3: Downstream Processing (TODO)
- [ ] Allow downstream functions to receive partial results
- [ ] Handle mapspec operations with missing/error values
- [ ] Implement strategies for reduction operations with errors
- [ ] Define behavior for functions that depend on error results

### Phase 4: Testing & Documentation (TODO)
- [ ] Write comprehensive tests for error handling scenarios
- [ ] Test parallel execution with errors
- [ ] Test various mapspec patterns with errors
- [ ] Update documentation and examples

## Key Technical Challenges

1. **Array Storage**: How to store ErrorSnapshot objects in numpy arrays or zarr arrays
   - Option 1: Use object dtype arrays (allows mixed data/error storage)
   - Option 2: Store errors separately with indices
   - Option 3: Use masked arrays or sentinel values
   - **Recommendation**: Use object dtype arrays for flexibility

2. **Shape Consistency**: Maintaining consistent array shapes when some elements are errors
   - ErrorSnapshot objects can be stored directly in object arrays
   - Maintain shape consistency by storing errors in-place

3. **Error Propagation Strategy** (NEW):
   - When a function receives an ErrorSnapshot as input, it should:
     a. Detect the error input early (before executing the function)
     b. Create a new ErrorSnapshot that references the original error
     c. Return immediately without executing the function logic
   - This creates an error chain that can be traced back to the source

4. **Downstream Dependencies**: How to handle functions that depend on error results
   - **Proposed Solution**: Early break-out with error propagation
   - Create a `PropagatedErrorSnapshot` that contains:
     - Reference to the original ErrorSnapshot
     - Information about which input parameter had the error
     - The function that would have been called
   - This maintains the causal chain of errors

5. **Reduction Operations**: How to handle reduce operations when some inputs are errors
   - For reductions: filter out ErrorSnapshot objects by default
   - Provide option for custom error handling in reduce operations
   - Consider adding error_count or error_indices to results

## Next Steps

1. Implement array storage handling for ErrorSnapshot objects
2. Test basic error continuity with simple pipelines
3. Handle downstream propagation of partial results
4. Write comprehensive test suite
5. Update documentation with examples

## Example Usage

### Example 1: Element-wise operations with errors

```python
@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def may_fail(x: int) -> int:
    if x == 3:
        raise ValueError("Cannot process 3")
    return x * 2

@pipefunc(output_name="z", mapspec="y[i] -> z[i]")
def process_y(y: int) -> int:
    # When y[i] is an ErrorSnapshot, this won't be called
    # Instead, a PropagatedErrorSnapshot is returned
    return y + 10

@pipefunc(output_name="sum")
def sum_valid(z: np.ndarray) -> int:
    # z will be an object array potentially containing ErrorSnapshots
    # Since this function receives the entire array (no mapspec),
    # if ANY element is an ErrorSnapshot, this function won't be called
    # Instead, sum will be a PropagatedErrorSnapshot
    return sum(z)

pipeline = Pipeline([may_fail, process_y, sum_valid])
result = pipeline.map({"x": [1, 2, 3, 4, 5]}, error_handling="continue")
# y = [2, 4, ErrorSnapshot, 8, 10] (object array)
# z = [12, 14, PropagatedErrorSnapshot, 18, 20] (object array)
# sum = PropagatedErrorSnapshot (because z contains errors)
```

### Example 2: Reduction with partial errors

```python
@pipefunc(output_name="matrix", mapspec="x[i], y[j] -> matrix[i, j]")
def compute(x: int, y: int) -> int:
    if x == 2 and y == 3:
        raise ValueError("Cannot compute for x=2, y=3")
    return x * y

@pipefunc(output_name="row_sums", mapspec="matrix[i, :] -> row_sums[i]")
def sum_rows(matrix: np.ndarray) -> int:
    # This function is called once per row (for each i)
    # If matrix[i, :] contains ANY ErrorSnapshot objects,
    # this specific call won't execute
    # Instead, row_sums[i] will be a PropagatedErrorSnapshot
    return sum(matrix)

pipeline = Pipeline([compute, sum_rows])
result = pipeline.map({"x": [1, 2, 3], "y": [2, 3, 4]}, error_handling="continue")
# matrix will be:
# [[2, 3, 4],
#  [4, ErrorSnapshot, 8],  # matrix[1, 1] is an error
#  [6, 9, 12]]
#
# row_sums will be:
# [9,                      # sum([2, 3, 4])
#  PropagatedErrorSnapshot, # skipped because matrix[1, :] contains an error
#  27]                     # sum([6, 9, 12])
```

## Proposed Implementation Approach

### 1. Early Error Detection (Before Execution)

```python
def _submit_func(func, kwargs, ...):
    # Check inputs for ErrorSnapshot objects BEFORE execution
    error_info = {}

    for param_name, value in kwargs.items():
        if isinstance(value, ErrorSnapshot):
            # Simple case: entire parameter is an error
            error_info[param_name] = {"type": "full", "error": value}
        elif isinstance(value, np.ndarray) and value.dtype == object:
            # Array case: check for ErrorSnapshot objects within the array
            error_mask = np.array([isinstance(v, ErrorSnapshot) for v in value.flat])
            if error_mask.any():
                error_info[param_name] = {
                    "type": "partial",
                    "shape": value.shape,
                    "error_indices": np.where(error_mask.reshape(value.shape)),
                    "error_count": error_mask.sum()
                }

    if error_info and run_info.error_handling == "continue":
        # Check if this function call should be skipped
        should_skip = False
        skip_reason = ""

        for param_name, info in error_info.items():
            if info["type"] == "full":
                # Entire parameter is an error - always skip
                should_skip = True
                skip_reason = "input_is_error"
                break
            elif info["type"] == "partial":
                # Array contains some errors
                # For ANY function that receives an array/slice, skip if it contains errors
                # This includes reductions, aggregations, or any operation on the array
                should_skip = True
                skip_reason = "array_contains_errors"
                break

        if should_skip:
            propagated_error = PropagatedErrorSnapshot(
                error_info=error_info,
                skipped_function=func,
                reason=skip_reason,
                attempted_kwargs={k: v for k, v in kwargs.items()
                                if k not in error_info},
            )
            return propagated_error

    # Normal execution path - create and submit task
    ...
```

This handles:
- **Full errors**: When an entire parameter is an ErrorSnapshot → Always skip function
- **Partial errors**: When an array contains some ErrorSnapshot elements → Always skip function
- **Design decision**: ANY function that receives an array/slice with errors will be skipped entirely
  - This ensures data integrity (no partial sums or incomplete operations)
  - Functions only run with complete, valid data
  - Clear error propagation (no ambiguity about missing values)
- **Element-wise handling**: Individual elements with errors are handled by the mapping infrastructure

### 2. PropagatedErrorSnapshot Class

```python
@dataclass
class PropagatedErrorSnapshot:
    """Represents a function that was skipped due to upstream errors."""
    error_info: dict[str, dict[str, Any]]  # parameter -> error details
    skipped_function: PipeFunc | Callable
    reason: str  # "requires_full_array", "all_inputs_failed", etc.
    attempted_kwargs: dict[str, Any]  # kwargs that were not errors
    timestamp: str = field(default_factory=_timestamp)

    def get_root_causes(self) -> list[ErrorSnapshot]:
        """Extract all original ErrorSnapshot objects."""
        root_causes = []
        for param, info in self.error_info.items():
            if info["type"] == "full":
                error = info["error"]
                if isinstance(error, PropagatedErrorSnapshot):
                    root_causes.extend(error.get_root_causes())
                else:
                    root_causes.append(error)
            elif info["type"] == "partial":
                # Would need to extract from the array
                pass
        return root_causes

    def __str__(self) -> str:
        func_name = getattr(self.skipped_function, '__name__', str(self.skipped_function))
        error_summary = []
        for param, info in self.error_info.items():
            if info["type"] == "full":
                error_summary.append(f"{param} (complete failure)")
            else:
                error_summary.append(f"{param} ({info['error_count']} errors in array)")

        return (
            f"PropagatedErrorSnapshot: Function '{func_name}' was skipped\n"
            f"Reason: {self.reason}\n"
            f"Errors in: {', '.join(error_summary)}"
        )
```

### 3. Storage Handling

- Use object dtype for arrays when error_handling="continue"
- Ensure zarr/disk storage can serialize ErrorSnapshot objects
- Consider adding metadata to indicate which indices contain errors

### 4. Mapspec-aware Error Handling

The error handling behavior depends on the mapspec pattern:

#### Element-wise operations (`x[i] -> y[i]`):
- Each element is processed individually
- If `x[i]` is an ErrorSnapshot, then `y[i]` becomes a PropagatedErrorSnapshot
- Other elements continue processing normally

```python
# Given: x = [1, 2, ErrorSnapshot, 4, 5]
# With mapspec "x[i] -> y[i]"
# Result: y = [f(1), f(2), PropagatedErrorSnapshot, f(4), f(5)]
```

#### Reduction operations (`x[i, :] -> y[i]` or `x[:, j] -> y[j]`):
- The function is called once per output element
- Each call receives a slice of the input
- If that slice contains ANY ErrorSnapshot, the output element becomes a PropagatedErrorSnapshot
- Other slices without errors process normally

```python
# Given: matrix = [[1, 2, 3],
#                   [4, ErrorSnapshot, 6],
#                   [7, 8, 9]]
# With mapspec "matrix[i, :] -> row_sums[i]"
# Result: row_sums = [6, PropagatedErrorSnapshot, 24]
```

#### Complete reduction (`x[:] -> y` or no mapspec with array input):
- The function receives the entire array
- If ANY element contains an ErrorSnapshot, the entire function is skipped
- The output becomes a PropagatedErrorSnapshot

```python
# Given: array = [1, 2, ErrorSnapshot, 4, 5]
# With no mapspec (receives entire array)
# Result: y = PropagatedErrorSnapshot
```
