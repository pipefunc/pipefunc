# Error Handling — Open Items (concise)

This list tracks only unresolved items.

## Reduction root causes (known limitation)

- Current: reductions return `PropagatedErrorSnapshot` with
  `reason == "array_contains_errors"`; `.get_root_causes()` returns an empty
  list. Future work: lazy resolution from indexed sources, if needed.

## Container‑recursive error scanning (enhancement)

- Today: `scan_inputs_for_errors` detects errors in `StorageBase` arrays and
  object `numpy.ndarray`s, but not inside arbitrary containers. Optional.
