# Development Notes for Error Handling Feature

## Error Handling Implementation

### Overview

Successfully implemented the `error_handling="continue"` feature for the `map()` method in pipefunc. This allows pipelines to continue processing even when some function calls fail, collecting errors as `ErrorSnapshot` objects instead of immediately raising exceptions.

### Key Components Implemented

1. **PropagatedErrorSnapshot Class** (exceptions.py)
   - New class to represent functions that were skipped due to upstream errors
   - Tracks error information, skipped function, reason, and attempted kwargs
   - Provides clear string representation for debugging

2. **Early Error Detection**
   - For mapspec operations: Added in `_run_iteration()` to check inputs before execution
   - For non-mapspec operations: Added in `compute_fn()` within `_execute_single()`
   - Functions skip execution when they receive ErrorSnapshot inputs
   - Creates PropagatedErrorSnapshot for skipped functions

3. **Array Storage with Object Dtype**
   - Arrays now use object dtype when `error_handling="continue"`
   - Allows mixed storage of values and ErrorSnapshot objects
   - Maintains array shape consistency

4. **Error Propagation Rules**
   - Element-wise operations: Each element processed independently
   - Reduction operations: Skip if input slice contains any errors
   - Full array operations: Skip if array contains any errors

### Test Coverage

All 10 comprehensive tests pass, covering:
- Simple pipelines without mapspec
- Element-wise operations with errors
- Pipeline error propagation
- 1D and 2D reductions with partial errors
- Full array reductions with errors
- Complex multi-step pipelines
- Sequential execution mode
- Multiple errors in same function
- Default "raise" behavior preserved

## Parallel Execution Fix

### Problem

When `error_handling="continue"` was implemented, tests failed with `parallel=True` due to pickling errors. Python's default `pickle` module cannot properly serialize function references when they're stored as attributes in dataclasses that get passed between processes.

### Solution

Added custom `__getstate__` and `__setstate__` methods to both `ErrorSnapshot` and `PropagatedErrorSnapshot` classes to use `cloudpickle` for serializing function references. This allows proper serialization of error objects containing function references during parallel execution.

### Results

- All error handling tests now pass with both `parallel=False` and `parallel=True`
- Error propagation works correctly across process boundaries
- No changes needed to the core error handling logic

## Code Refactoring

### Overview

Refactored the error handling code to follow the DRY (Don't Repeat Yourself) principle by extracting common operations into reusable functions.

### Changes Made

1. **Created `pipefunc/_error_handling.py`** with common utilities:
   - `check_for_error_inputs()`: Checks if inputs contain ErrorSnapshot objects
   - `create_propagated_error()`: Creates PropagatedErrorSnapshot instances
   - `handle_error_inputs()`: Combined error checking and propagation
   - `cloudpickle_function_state()`: Helper for pickling state dictionaries
   - `cloudunpickle_function_state()`: Helper for unpickling state dictionaries

2. **Refactored duplicate code** in `pipefunc/map/_run.py`
   - Replaced ~60 lines of duplicated error detection code
   - Both `_run_iteration()` and `compute_fn()` now use common functions

3. **Improved pickling methods** in ErrorSnapshot and PropagatedErrorSnapshot classes

4. **Fixed missing parameters** - Added error_handling parameter to prepare_run calls

### Benefits

- Code Reuse: Eliminated ~60 lines of duplicated code
- Maintainability: Changes only need to be made in one place
- Consistency: Ensures uniform error handling behavior
- Testability: Common functions can be unit tested independently
- Readability: Clearer main code flow

## Known Limitations

1. **Storage Backends**: ErrorSnapshot serialization for zarr/disk storage not yet implemented
2. **Performance**: Object dtype arrays may have performance implications for large datasets
