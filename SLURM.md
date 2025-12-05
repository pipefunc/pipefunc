# SLURM Map-Scope Error Filtering Validation

## Context
In the `pipefunc` library, the function `should_filter_error_indices` determines whether to filter out error indices locally before submitting jobs to an executor (specifically SLURM). Historically, this filtering only occurred for `resources_scope="element"`.

## Investigation
We investigated whether `resources_scope="map"` functions with upstream errors were causing pointless job submissions to SLURM.

### Findings
1.  **Verification**: Using a mock-based reproduction script, we confirmed that `should_filter_error_indices` returned `False` for functions with `resources_scope="map"`, even when `error_handling="continue"` was active and the executor was identified as a SLURM executor.
2.  **Behavior**: This resulted in indices corresponding to upstream errors being routed to the remote executor rather than being handled locally. This meant SLURM jobs were being submitted for inputs that were already known to be `ErrorSnapshot` or `PropagatedErrorSnapshot` objects.

## Conclusion & Fix
The restriction to `resources_scope="element"` was unnecessary. If upstream errors exist and we are in "continue" mode, we should avoid submitting those specific indices to the SLURM cluster regardless of whether resources are defined at the "map" or "element" level.

### Changes Applied
*   **Code**: Modified `pipefunc/map/_adaptive_scheduler_slurm_executor.py` to remove the `func.resources_scope == "element"` check from `should_filter_error_indices`.
*   **Logic**: Now, any function using a SLURM executor with `error_handling="continue"` will filter out indices that have upstream errors before submission.
*   **Testing**: Added a regression test `test_map_scope_filters_error_indices_with_mock_slurm` to ensure this behavior is preserved.
