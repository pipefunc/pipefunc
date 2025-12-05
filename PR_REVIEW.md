# PR Review: Error Handling Feature

## Summary
This PR introduces comprehensive error handling capabilities (`error_handling="continue"` vs `"raise"`) to the `pipefunc` pipeline execution. It allows pipelines to gracefully handle failures by propagating `ErrorSnapshot` objects instead of crashing, which is a significant enhancement for long-running or complex pipelines.

## Strengths
-   **Architecture**: The core logic is cleanly separated into `pipefunc/_error_handling.py`, while integration points in `pipefunc/map/_run.py` are well-defined.
-   **Testing**: The PR includes an impressive suite of tests (`tests/integration/map/test_error_handling.py`, `tests/unit/error_handling/`), covering unit logic, integration scenarios, and regressions.
-   **Refactoring**: The recent refactoring in `pipefunc/map/_run.py` (introducing `_prepare_execution_environment`) effectively deduplicates logic between standard execution, single item execution, and resource evaluation.
-   **Edge Cases**: Handling of tuple outputs with errors (`_default_output_picker` update) and resource evaluation failures demonstrates attention to detail.

## Performance Note (Resolved)
The earlier concern about scanning non-object `StorageBase` arrays was addressed in the current code: dtype is checked before `to_array()`, and guarded again afterward, so numeric storages are skipped without loading or scanning.

## Minor Observations
1.  **Untracked Files**: There are untracked files in the directory (`AGENTS.md`, `notebooks/`, `zarrv3.md`). Ensure these are not intended to be part of the PR or add them to `.gitignore` if they are personal artifacts.
2.  **Magic Strings**: The key `_EVALUATED_RESOURCES` is used in multiple files. It is defined in `pipefunc/map/_run.py` and imported elsewhere, which is good practice, but ensure strict usage of the constant to avoid typos.

## Conclusion
The PR is in excellent shape. The feature is well-implemented and thoroughly tested. With the performance optimization for `StorageBase` scanning, it should be ready to merge.

## Outstanding TODOs / Open Risks

### Investigated - Not Issues
- **Global `func.error_snapshot` mutation:** ✅ Not an issue. The returned snapshots (stored in result arrays) are correct per-thread. The `func.error_snapshot` attribute is just a debug convenience. Test `test_parallel_error_snapshot_race` validates this.
- **Deep propagation pickling recursion:** ✅ Not an issue. Python's recursion limit is 1000; practical pipelines rarely exceed 10-20 stages.
- **Partial array errors in `get_root_causes`:** ✅ Documented behavior. The docstring explicitly states empty list for partial errors; users can access `error_info` directly.

### Still Unvalidated / Future Work
- **SLURM map-scope filtering:** `should_filter_error_indices` only handles `resources_scope=="element"`. Needs real SlurmExecutor or faithful mock to verify.
- **ErrorSnapshot double-dump inefficiency:** Error objects bypass the XOR guard and are written twice. Inefficient but not incorrect; documented with xfail test.
- **Untested scenarios:** Nested pipelines in continue mode, mixed storages with errors, async cancellation/timeouts, memory impact for large error-bearing arrays.
