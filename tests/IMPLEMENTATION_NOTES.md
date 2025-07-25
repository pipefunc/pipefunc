# Implementation Notes for Error Handling Feature

## Current Issues Found Through Testing

### 1. MapSpec Error Handling Issue
When a function with mapspec (e.g., `x[i] -> y[i]`) encounters an error:
- Current behavior: `_run_iteration` returns an `ErrorSnapshot` object
- `_pick_output` wraps it in a tuple: `(ErrorSnapshot,)`
- `_process_task` expects lists from mapspec operations to chain together
- Error: `TypeError: 'ErrorSnapshot' object is not iterable` when trying to use `itertools.chain`

### Root Cause
In mapspec operations, each chunk processor returns a list of results. When an error occurs, we're returning a single ErrorSnapshot instead of a list containing the ErrorSnapshot at the appropriate position.

### Fix Needed
The `_run_iteration_and_process` function needs to be modified to:
1. When an error occurs, determine which element in the batch caused the error
2. Return a list with successful results and ErrorSnapshot at the failed position
3. Ensure the list has the correct length matching the chunk size

## Test Results Summary

### Working Tests
None yet - all tests fail due to the mapspec issue above.

### Tests Written
1. **Basic Tests** (`test_error_handling_basic.py`):
   - Single function with errors
   - Sequential vs parallel execution
   - Multiple errors in one function
   - Default error_handling="raise" behavior
   - 2D mapspec with errors
   - Caching with errors

2. **Advanced Tests** (`test_error_handling.py`):
   - Error propagation through pipelines
   - 1D and 2D reductions with partial errors
   - Complex multi-step pipelines
   - Tests requiring `PropagatedErrorSnapshot` (currently skipped)

## Implementation Priority

1. **Fix MapSpec Error Handling** (Critical)
   - Modify `_run_iteration_and_process` to return lists with errors in correct positions
   - Ensure array storage can handle object dtype

2. **Implement PropagatedErrorSnapshot** (High)
   - Add class to `exceptions.py`
   - Implement early error detection in `_submit_func`

3. **Array Storage Updates** (High)
   - Ensure arrays use object dtype when error_handling="continue"
   - Handle mixed data types in arrays

4. **Storage Backend Support** (Medium)
   - Ensure ErrorSnapshot serialization works with zarr/disk storage

## Key Design Decisions

1. **Error Propagation Rules**:
   - Element-wise operations: Process each element independently
   - Array/slice operations: Skip if ANY element has an error
   - Complete reductions: Skip if array contains any errors

2. **Early Detection**:
   - Check for errors before submitting tasks to executors
   - Return PropagatedErrorSnapshot without executing functions

3. **Storage Strategy**:
   - Use object dtype arrays to store mixed values/errors
   - Maintain array shapes with errors in-place
