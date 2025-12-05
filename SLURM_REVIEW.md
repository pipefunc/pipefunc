# SLURM Map-Scope Error Filtering Validation

## Findings

1.  **Missing Filtering Logic**: The function `should_filter_error_indices` does **not** exist in `pipefunc/map/_run.py` in the current codebase. Consequently, no error filtering logic is applied before job submission.
2.  **Executor Detection**: `is_slurm_executor()` correctly identifies `adaptive_scheduler.SlurmExecutor` instances.
3.  **Inefficient Job Submission**:
    *   In a test pipeline where index 1 of an upstream function `a` fails (returning an `ErrorSnapshot`), the downstream function `b` (with `resources_scope="map"`) still receives a job submission for index 1.
    *   **Observation**: 3 jobs were submitted for function `b` (indices 0, 1, 2), whereas only 2 (indices 0, 2) should have been submitted if filtering were active.
    *   **Impact**: The job for index 1 will inevitably fail or return a propagated error immediately upon execution, wasting cluster queue slots and startup overhead.

## Conclusion

The current implementation leads to inefficient resource usage on SLURM clusters when `continue_on_error=True`. Downstream jobs are submitted even when their inputs are `ErrorSnapshot` objects.

## Recommendation

Implement logic to filter out indices that correspond to upstream errors before submitting jobs to the SLURM executor. This should likely be done in `pipefunc/map/_run.py` within `_map_slurm_executor_kwargs` or `_maybe_parallel_map`, inspecting the inputs for `ErrorSnapshot` instances when `continue_on_error` is enabled.
