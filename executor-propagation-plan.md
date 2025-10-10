# Executor Propagation Handling (Oct 2025)

## Problem Snapshot
- When map-scope functions run with `error_handling="continue"`, upstream errors propagate.
- Despite that, `_maybe_parallel_map` still calls `_submit`, so Slurm (and other executors) receive jobs that immediately bail out.
- On clusters this can queue work in the wrong partition or stall jobs indefinitely.

## Observed Behaviour
- `tests/map/test_adaptive_slurm_executor.py::test_map_scope_all_error_inputs_skip_executor_submission` documents the regression.
- The test patches `_submit` and shows four executor submissions even though every element already has an error snapshot.

## Why It Happens
- Resource evaluation short-circuits on error inputs, but `_maybe_parallel_map` still hands every index to the executor.
- There was no single source of truth describing which indices had propagated errors, so earlier fixes duplicated checks or ignored the metadata.

## Recent Prep Work
- `_select_kwargs` plus `_maybe_eval_resources_in_selected` now compute the per-index kwargs and cached `error_info` once so every caller reuses the same metadata.
- `_raise_and_set_error_snapshot` reuses the cached slice, preventing redundant recomputation.
- `handle_error_inputs` accepts precomputed error info so early error detection stays cheap.

## Plan of Attack
1. Extend `_maybe_parallel_map` to reuse the precomputed `error_info` for each index.
2. If all indices are already errors, resolve them in-process (using `process_index(i)`) and skip `_submit` entirely.
3. Ensure progress/status trackers still receive accurate updates when short-circuiting.
4. Add mixed-case tests (some error, some healthy) to confirm only healthy indices hit the executor.

## Open Questions
- Should we reuse the original executor when no overrides exist, or leave it untouched when we skip submission altogether?
- Do we want to stash the per-index kwargs/error metadata somewhere to avoid recomputing when we do submit work?

## Next Steps
- Implement the short-circuit logic behind a small helper so both sync/async map runners can share it.
- Update the new failing test to assert the expected zero submissions once the fix lands.
