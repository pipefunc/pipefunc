# Error Handling Regressions and Gaps (tracking)

This note summarizes concrete issues found in the new error-handling work, with
clear expectations and planned regression tests that fail today and will pass
after fixes. It serves as an anchor if the chat context is lost.

## 1) Cache cross-mode leak (bug)

- Symptom: Results cached under `error_handling="continue"` (e.g., an
  `ErrorSnapshot`) are reused when the same function is invoked with
  `error_handling="raise"`.
- Cause: The map-mode cache key omits the error-handling mode; it is currently
  `(func._cache_id, to_hashable(kwargs))`. When `raise` is requested later, the
  cached `ErrorSnapshot` is returned instead of executing user code and raising.
- Expected: With `error_handling="raise"`, an exception should be raised even if
  a previous `continue` run cached a snapshot for the same inputs.
- Fix sketch: include the error-handling mode in the cache key, e.g.
  `(func._cache_id, error_handling, to_hashable(kwargs))`, or raise when a
  cached value is an `ErrorSnapshot` and mode is `"raise"`.
- Test: `test_cache_does_not_mask_raise_mode_after_continue_cache` (added).

## 2) PropagatedErrorSnapshot.get_root_causes for partial-array errors (gap)

- Background: `PropagatedErrorSnapshot.get_root_causes()` walks upstream to
  gather original `ErrorSnapshot`s. It works when a parameter itself is an
  error ("full"), e.g. element-wise propagation (`y[i] -> z[i]`).
- Symptom: For reductions where a parameter is an array containing one or more
  `ErrorSnapshot`s ("partial" case), `get_root_causes()` currently returns an
  empty list, because it doesn’t traverse indices in arrays.
- Why this matters: The docs suggest that downstream code “can walk back to the
  root causes.” That expectation naturally includes reductions over arrays that
  contain errors.
- Expected: Calling `get_root_causes()` on a propagated error from a reduction
  should surface the underlying element-level `ErrorSnapshot`(s).
- Fix sketch (directional): Either store a compact reference to the source
  storage and error indices to resolve lazily, or embed a bounded set of root
  errors; then have `get_root_causes()` resolve them.
- Test: `test_get_root_causes_in_reduction_returns_upstream_errors` (added).

## 3) Reason string consistency (addressed)

- Now constrained to a small Literal domain: `{"input_is_error", "array_contains_errors"}`.
- The legacy placeholder `"input_contains_errors"` is removed from internals.
- Existing tests that assert specific reasons remain valid.

## 4) Container-recursive error scanning (enhancement, not a failing bug)

- Today: `scan_inputs_for_errors` detects `ErrorSnapshot`s in `StorageBase`
  arrays and object-`numpy.ndarray`s. It does not recurse into arbitrary
  containers (e.g., `dict` of arrays containing errors).
- Impact: For non-canonical container-shaped arguments, downstream functions may
  execute and raise (becoming `ErrorSnapshot`) rather than receiving a
  `PropagatedErrorSnapshot`.
- Status: Design choice vs. convenience; not asserted by docs. No regression
  test added.
