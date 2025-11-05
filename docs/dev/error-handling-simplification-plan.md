## Open items only

- Unify scheduler submit/process loops (generation + eager) behind one engine.
- Consider lazy root‑cause resolution for reductions (documented limitation).

## Impacted Files and Concrete Changes

- `pipefunc/map/_run.py`
  - Add: `ErrorContext`, `maybe_propagate_before_call`, `maybe_wrap_exception`,
    `eval_resources`, and `ErrorStub`.
  - Replace most of the logic in `_run_iteration`, `_execute_single`,
    `_prepare_kwargs_for_execution`, and scattered pre/post guards with the
    unified flow shown above.
  - Simplify `_update_array` and `_update_result_array` by using `ErrorStub`.
  - Normalize reasons to the `Reason` Literal.

- `pipefunc/map/_run_eager.py`
  - Replace custom processing loops with a thin wrapper over the unified submit /
    process functions; keep only readiness policy differences.

- `pipefunc/exceptions.py`
  - Extend `PropagatedErrorSnapshot` with `source_output_name`, `error_indices`,
    `run_folder`; implement lazy `get_root_causes()`.

- `pipefunc/_error_handling.py`
  - Keep `ErrorInfo`, `scan_inputs_for_errors`. Optionally add a trivial shallow
    recursion into built‑ins (list/tuple/dict) later; default to current fast
    path for now.

- `pipefunc/_pipefunc_utils.py`
  - Continue to attach `ErrorSnapshot` for debug printing; now routed through
    `maybe_wrap_exception`.

- Cache key logic
  - In `_get_or_set_cache` (or its replacement), include `mode` for map flows;
    add re‑raise guard for cached snapshots in `raise` mode.

## Backward Compatibility and Behavior

- Public API unchanged: the `error_handling` flag and docs remain as in this PR.
- Continue/raise semantics unchanged; only internal structure simplified.
- Snapshots pickling: preserved (we only add optional fields). Existing pickles
  should load.

## Testing Strategy

- Keep all new tests. Ensure the two regressions pass:
  - `test_cache_does_not_mask_raise_mode_after_continue_cache`
  - `test_get_root_causes_in_reduction_returns_upstream_errors`
- Add a small contract test asserting `reason in {"input_is_error", "array_contains_errors"}`.
- Add a test proving `return_results=False` still persists real snapshots and
  memory stays low (reusing current coverage).
- Re‑run async and eager parity tests (already present) — expect identical
  semantics.

## Performance Considerations

- Early propagation check avoids executor submission and function calls; unchanged
  from current behavior but made more predictable.
- Chunking and progress overhead unchanged.
- Lazy root‑cause resolution only loads error elements on demand, reducing memory
  pressure compared to storing all roots up front.

## Estimated Code Size Change

- Remove duplicated guards and resource checks: ~120–200 LOC.
- Scheduler processing unification: ~50–90 LOC.
- ErrorStub simplification: ~20–40 LOC.
- Additions for lazy roots: ~80–120 LOC.
- Net: ~200–400 fewer lines, with clearer flow.

## Migration Plan (Phased)

1) Correctness fixes first:
   - Add `error_handling` to cache key and re‑raise guard in `raise` mode.
   - Implement lazy root‑cause resolution for reductions.

2) Refactor internals (no behavior change):
   - Introduce `ErrorContext`, guard helpers, and `ResourcesEval`.
   - Swap call‑sites in `_run.py` and `_run_eager.py` to use the unified flow.

3) Cleanups:
   - Replace `_ErrorMarker`/`_InternalShape` with `ErrorStub`.
   - Normalize reasons (Literal) and add the small reason-set test.

4) Optional enhancements (future):
   - Shallow container recursion in `scan_inputs_for_errors` (list/tuple/dict).

## Risks and Mitigations

- Risk: subtle divergence between generation and eager code paths.
  - Mitigation: one submit/process implementation, shared by both.
- Risk: snapshot pickling compatibility.
  - Mitigation: keep optional fields and version comments; tests for legacy
    loading already exist and should be extended if needed.
- Risk: regressions hidden by cache.
  - Mitigation: mode‑aware cache key and re‑raise guard; test locks this in.

## Documentation Updates

- Expand the error‑handling concept page with:
  - The two concrete reasons and when they appear.
  - A brief note on lazy root‑cause resolution for reductions.
  - Clarify that infra/shape/storage errors still raise even in continue mode.

---
This plan keeps public semantics intact, shrinks the moving parts, and creates a
single, simple mental model for error flow. It can be implemented incrementally
with the tests added in this branch guarding correctness.
