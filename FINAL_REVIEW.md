# Final Review: Error Handling "Continue" Mode PR

This document provides a comprehensive review of the error handling "continue" mode feature for pipefunc, including bugs that were fixed, issues that were investigated but found to NOT be bugs, and regression test verification.

## Overview

The PR adds a new `error_handling` parameter to `run_map` and `run_map_async` in the pipefunc library. This parameter accepts:
- `"raise"` (default) - existing behavior, raises exceptions immediately
- `"continue"` (new) - collects errors as `ErrorSnapshot` objects and propagates them through the pipeline

---

## Part 1: Confirmed Bugs That Were Fixed

Three bugs were identified, verified to exist in the old code, and fixed with proper regression tests.

### Bug 1: Resume with Different `error_handling` Mode Silently Allowed

**Location**: `pipefunc/map/_run_info.py`

**Problem**: When resuming a run that was executed with `error_handling="continue"`, switching to `error_handling="raise"` was silently allowed. The `_compare_to_previous_run_info()` function validated shapes, mapspecs, inputs, and defaults but **never checked** if `error_handling` matched.

**Impact**:
- ErrorSnapshots stored from a continue-mode run would be loaded and passed to downstream functions as if they were valid results
- Users expected "raise" mode to re-execute failed computations, but they were silently skipped
- Downstream functions would receive ErrorSnapshot objects and fail with confusing `AttributeError` or `TypeError`

**Verification** (with old code):
```python
# First run with error_handling="continue" - stores ErrorSnapshots
pipeline.map({"x": [0, 1, 2]}, error_handling="continue", run_folder=tmp_path)

# Resume with error_handling="raise" - WAS SILENTLY ALLOWED (BUG)
result = pipeline.map({"x": [0, 1, 2]}, error_handling="raise", resume=True, run_folder=tmp_path)
# result["y"].output[1] was an ErrorSnapshot instead of raising!
```

**Fix**: Added validation in `_compare_to_previous_run_info()`:
```python
def _compare_to_previous_run_info(
    ...
    error_handling: Literal["raise", "continue"] = "raise",
) -> None:
    ...
    # Validate error_handling mode matches - critical for correct behavior
    if error_handling != old.error_handling:
        msg = (
            f"`error_handling='{error_handling}'` does not match previous run "
            f"(`error_handling='{old.error_handling}'`), cannot use `resume=True`. "
            "Resuming with a different error handling mode could lead to incorrect results."
        )
        raise ValueError(msg)
```

Also updated the call site in `RunInfo.create()` to pass `error_handling` to the validation function.

**Regression Test**: `tests/integration/map/test_error_handling.py::test_resume_rejects_different_error_handling_mode`
- Old code: ❌ FAILED (`DID NOT RAISE <class 'ValueError'>`)
- Fixed code: ✅ PASSED

---

### Bug 2: `scan_inputs_for_errors` Crashes on 0-D Object Arrays Containing Errors

**Location**: `pipefunc/_error_handling.py`

**Problem**: For 0-dimensional numpy arrays with `dtype=object` that contain an `ErrorSnapshot`, the code called `np.where(error_mask.reshape(array.shape))` which raises `ValueError: Calling nonzero on 0d arrays is not allowed`.

**Important Detail**: The bug only manifests when the 0-D array **contains** an ErrorSnapshot. A 0-D array with a regular value (e.g., `np.array(42, dtype=object)`) does not crash because the `error_mask` is all False and `error_mask.any()` returns False before reaching `np.where()`.

**Verification** (with old code):
```python
from pipefunc._error_handling import scan_inputs_for_errors
from pipefunc.exceptions import ErrorSnapshot
import numpy as np

error = ErrorSnapshot(function=lambda x: x, exception=ValueError("test"), args=(1,), kwargs={})
arr_with_error = np.array(error, dtype=object)  # 0-D array containing ErrorSnapshot

scan_inputs_for_errors({"x": arr_with_error})
# CRASHED: ValueError: Calling nonzero on 0d arrays is not allowed
```

**Fix**: Added special handling for 0-D arrays before the `np.where()` call:
```python
if array is not None:
    if array.dtype != object:
        continue

    # Handle 0-D arrays specially (np.where doesn't work on 0-D arrays)
    if array.ndim == 0:
        item = array.item()
        if isinstance(item, (ErrorSnapshot, PropagatedErrorSnapshot)):
            error_info[param_name] = ErrorInfo.from_full_error(item)
        continue

    # ... rest of the function for N-D arrays
```

**Regression Test**: `tests/unit/error_handling/test_error_info_and_snapshots.py::test_scan_inputs_for_errors_0d_object_array`
- Old code: ❌ FAILED (`ValueError: Calling nonzero on 0d arrays is not allowed`)
- Fixed code: ✅ PASSED

---

### Bug 3: `output_picker` Exceptions Bypass Continue Mode

**Location**: `pipefunc/map/_run.py`

**Problem**: The `_pick_output` function was called **after** the user function returned, outside the error handling context. If `output_picker` raised an exception, the pipeline crashed even with `error_handling="continue"`.

**Old Code**:
```python
def _pick_output(func: PipeFunc, output: Any) -> tuple[Any, ...]:
    output_names = at_least_tuple(func.output_name)
    if _is_error_snapshot(output):
        return tuple(output for _ in output_names)
    return tuple(
        (func.output_picker(output, output_name) if func.output_picker is not None else output)
        # ^^^ This could raise, and it was NOT caught!
        for output_name in output_names
    )
```

**Verification** (with old code):
```python
def bad_picker(result, key):
    if key == "bad":
        raise RuntimeError("picker boom")
    return result[key]

@pipefunc(output_name=("good", "bad"), mapspec="x[i] -> good[i], bad[i]", output_picker=bad_picker)
def my_func(x: int) -> dict:
    return {"good": x * 2, "bad": x * 3}

pipeline = Pipeline([my_func])
pipeline.map({"x": [1, 2, 3]}, error_handling="continue", parallel=False)
# CRASHED: RuntimeError: picker boom (should have returned ErrorSnapshot)
```

**Fix**: Added error handling to `_pick_output()` via a new helper function:
```python
def _pick_single_output(
    func: PipeFunc,
    output: Any,
    output_name: str,
    error_handling: Literal["raise", "continue"],
) -> Any:
    """Pick a single output, with error handling for output_picker failures."""
    assert func.output_picker is not None
    try:
        return func.output_picker(output, output_name)
    except Exception as e:
        if error_handling == "continue":
            return ErrorSnapshot(
                func.output_picker,
                e,
                args=(output, output_name),
                kwargs={},
            )
        raise


def _pick_output(
    func: PipeFunc,
    output: Any,
    error_handling: Literal["raise", "continue"] = "raise",
) -> tuple[Any, ...]:
    output_names = at_least_tuple(func.output_name)
    if _is_error_snapshot(output):
        return tuple(output for _ in output_names)
    if func.output_picker is None:
        return tuple(output for _ in output_names)

    return tuple(
        _pick_single_output(func, output, output_name, error_handling)
        for output_name in output_names
    )
```

Also fixed `_dump_single_output()` which had the same issue for non-mapspec functions with multiple outputs.

**Regression Test**: `tests/integration/map/test_error_handling.py::test_output_picker_exception_in_continue_mode`
- Old code: ❌ FAILED (`RuntimeError: picker boom`)
- Fixed code: ✅ PASSED

**Non-Regression Test**: `tests/integration/map/test_error_handling.py::test_output_picker_exception_raises_in_raise_mode`
- Old code: ✅ PASSED (raised because there was no error handling at all)
- Fixed code: ✅ PASSED (raised because `error_handling="raise"` explicitly re-raises)
- Note: This test documents expected behavior but is NOT a regression test since it passes on both versions.

---

## Part 2: Issues Investigated and Found to NOT Be Bugs

The following issues were identified in the initial review but upon investigation were found to be either by design, not reproducible, or not actually bugs.

### Non-Issue 1: Double Dumping of Error Objects

**Claimed Problem**: ErrorSnapshot objects are dumped to storage twice (once in worker process, once in main process), causing inefficient I/O.

**Investigation**: Created test `test_error_objects_dump_count_with_mock` that patches `FileArray.dump` to count calls.

**Finding**: ✅ CONFIRMED as inefficiency but NOT incorrect behavior.

The ErrorSnapshot IS dumped twice because:
1. First dump in `_update_array` with `in_post_process=False` (during execution)
2. Second dump in `_update_array` with `in_post_process=True` (during post-processing)

The XOR logic `(array.dump_in_subprocess ^ in_post_process)` is bypassed for error objects because `is_error_object=True` causes immediate dump regardless of XOR result.

**Conclusion**: This is an **inefficiency** (double serialization/write), but not incorrect behavior. The second write simply overwrites the first with identical data. Marked as `xfail` test to document the behavior.

**Test**: `tests/integration/map/test_error_handling_potential_issues.py::test_error_objects_dump_count_with_mock` (xfail)

---

### Non-Issue 2: Map-Scope Resources Skipped for Valid Elements

**Claimed Problem**: When ANY map-level input contains an error, resources are skipped entirely, which could affect valid elements that still need to execute.

**Investigation**: Created test `test_map_scope_resources_with_partial_errors` to track resource function calls.

**Finding**: ❌ NOT A BUG - This is by design.

When map-level inputs contain errors:
- For `resources_scope="map"` with `resources_variable=None`: Resources are skipped because the resource function can't be called meaningfully when the input array contains errors
- For `resources_scope="map"` with `resources_variable` set: The resources function IS called but receives the error-containing array; if it can't handle errors, it will fail

The code comment explains the design:
```python
if func.resources_scope == "map":
    if error_infos.map and func.resources_variable is None:
        return _RESOURCES_SKIPPED
```

**Test**: `tests/integration/map/test_error_handling_potential_issues.py::test_map_scope_resources_with_partial_errors`

---

### Non-Issue 3: Resource Error Uses Wrong Function Reference

**Claimed Problem**: When resource evaluation fails, the ErrorSnapshot stores `func.resources` (the resources callable) instead of the actual user function, making `error.reproduce()` call the wrong thing.

**Investigation**: Created tests `test_resource_evaluation_error_function_reference` and `test_resource_error_reproduce_behavior`.

**Finding**: ❌ NOT A BUG - This is correct behavior.

Resource evaluation errors create a `PropagatedErrorSnapshot` (not direct `ErrorSnapshot`). The inner `ErrorSnapshot` stores the resources callable because:
1. The error occurred in resource evaluation, not in the actual function
2. `reproduce()` correctly re-raises the resource evaluation error
3. The user can see from the error structure that it was a resource issue

The `PropagatedErrorSnapshot` wraps the resource error with:
- `skipped_function`: The actual PipeFunc that was skipped
- `reason`: `"input_is_error"`
- `error_info["__pipefunc_internal_resource_error__"]`: The actual ErrorSnapshot from resource evaluation

**Test**: `tests/integration/map/test_error_handling_potential_issues.py::test_resource_evaluation_error_function_reference`

---

### Non-Issue 4: SLURM Filtering Only for `resources_scope="element"`

**Claimed Problem**: `should_filter_error_indices()` only returns `True` for `resources_scope="element"`, so map-scope resources might submit pointless jobs for indices with upstream errors.

**Investigation**: Created tests `test_slurm_filtering_with_map_scope_resources` and `test_slurm_filtering_function_behavior`.

**Finding**: ❓ INCONCLUSIVE - Cannot properly test without SLURM environment.

The test showed that `should_filter_error_indices()` returns `False` for ALL test cases because `is_slurm_executor()` returns `False` for the mock executor. The mock doesn't properly simulate a SLURM executor.

To properly verify this would require either:
1. Using the actual SLURM executor class
2. Properly mocking `is_slurm_executor()` to return `True`
3. Testing in a real SLURM environment

**Code Logic**: The code does show that map-scope resources don't filter error indices:
```python
def should_filter_error_indices(func, executor, error_handling) -> bool:
    return (
        func.resources_scope == "element"  # <-- Only element scope
        and is_slurm_executor(executor)
        and error_handling == "continue"
    )
```

**Test**: `tests/integration/map/test_error_handling_potential_issues.py::test_slurm_filtering_with_map_scope_resources`

---

## Part 3: Additional Tests Added (Not Regression Tests)

These tests were added for completeness but don't test bug fixes:

### `test_empty_inputs_continue_mode`
Tests that empty input arrays work correctly with `error_handling="continue"`. This is a general functionality test, not a regression test.

### `test_scan_inputs_for_errors_0d_numeric_array`
Tests that 0-D numeric arrays are skipped (they can't contain errors). This documents expected behavior.

### `test_output_picker_exception_raises_in_raise_mode`
Tests that output_picker exceptions still raise in `error_handling="raise"` mode. This passes on both old and new code because:
- Old code: Raised (no error handling existed at all)
- New code: Raised (explicit re-raise for raise mode)

---

## Part 4: Summary

### Bugs Fixed: 3

| Bug | Location | Regression Test |
|-----|----------|-----------------|
| Resume validation missing `error_handling` check | `_run_info.py` | `test_resume_rejects_different_error_handling_mode` |
| 0-D object array crash in `scan_inputs_for_errors` | `_error_handling.py` | `test_scan_inputs_for_errors_0d_object_array` |
| `output_picker` exceptions bypass continue mode | `_run.py` | `test_output_picker_exception_in_continue_mode` |

### Issues Investigated but NOT Bugs: 4

| Issue | Finding |
|-------|---------|
| Double dumping of error objects | Inefficiency, not incorrectness (xfail test) |
| Map-scope resources skipped | By design |
| Resource error function reference | Correct behavior |
| SLURM filtering for map-scope | Inconclusive (needs SLURM env) |

### Verification Method

All fixes were verified by:
1. Checking out the old code (commit `361d4e0d`)
2. Running the regression tests against old code → Tests FAILED
3. Restoring the fixed code (commit `eb7bd3de`)
4. Running the regression tests against fixed code → Tests PASSED

---

## Remaining TODOs / Open Risks

### Investigated and Assessed (Not Issues)

1. **Performance: StorageBase numeric array loading** ✅ ALREADY OPTIMIZED
   - The code at `_error_handling.py:72-82` already checks `dtype` BEFORE calling `to_array()` on StorageBase
   - If dtype is not object, it skips loading entirely

2. **Global `func.error_snapshot` mutation (thread-safety)** ✅ NOT AN ISSUE
   - In `error_handling="continue"` mode, the **returned snapshots** (stored in result arrays) are correct per-thread
   - The `func.error_snapshot` attribute is just a debug convenience for interactive use
   - Test `test_parallel_error_snapshot_race` validates correct per-thread snapshots are returned

3. **Deep propagation pickling recursion** ✅ NOT AN ISSUE
   - Python's default recursion limit is 1000
   - Practical pipelines rarely exceed 10-20 stages deep

4. **Partial array errors in `get_root_causes`** ✅ DOCUMENTED BEHAVIOR
   - `get_root_causes()` explicitly ignores `type=="partial"` entries
   - Docstring states: "For array-containing errors (reductions), this currently returns an empty list"
   - Users can access `error_info` metadata directly for partial errors

### Still Unvalidated / Future Work

5. **SLURM map-scope filtering**
   - `should_filter_error_indices` only gates `resources_scope=="element"`
   - Map-scope resources plus propagated errors may still submit useless jobs
   - Needs real SlurmExecutor or faithful mock to validate
   - Risk: Medium - only affects SLURM users with map-scope resources

6. **ErrorSnapshot double-dump inefficiency**
   - Error objects bypass the XOR guard in `_update_array`
   - Written in both executor and post-process phases
   - Currently inefficient but not incorrect
   - Test: `test_error_objects_dump_count_with_mock` (xfail) documents this
   - Risk: Low - inefficiency only, not correctness issue

7. **Untested/under-tested edge cases**
   - Nested pipelines in continue mode
   - Mixed storages (FileArray + SharedMemoryArray) with errors
   - Async cancellation/timeout under continue mode
   - Memory footprint for very large error-bearing arrays

---

## Part 5: Files Changed

### Source Files Modified:
- `pipefunc/_error_handling.py` - Added 0-D array handling (+8 lines)
- `pipefunc/map/_run.py` - Added output_picker error handling (+50 lines, -7 lines)
- `pipefunc/map/_run_info.py` - Added error_handling validation (+81 lines, -39 lines refactored)

### Test Files Added/Modified:
- `tests/integration/map/test_error_handling.py` - Added 4 new tests (+115 lines)
- `tests/unit/error_handling/test_error_info_and_snapshots.py` - Added 2 new tests (+50 lines)
- `tests/integration/map/test_error_handling_potential_issues.py` - New file for investigating potential issues

### Commits:
- `eb7bd3de` - fix: address error handling bugs found in PR review
