# Error Handling Simplification Plan (Detailed)

This document lays out a concrete plan to simplify the new error‑handling
implementation while preserving behavior and semantics. It is intentionally
high‑detail to serve as a blueprint for a follow‑up refactor PR.

Status terms used below:
- Current: behavior/structure as implemented in this branch.
- Target: the simplified/clean design.
- Impacted files: primary modules to change.

## Goals

- Reduce branching and parameter threading across the map runners.
- Create single, obvious hook points for error handling and resource evaluation.
- Make eager and generation scheduling share a single execution engine.
- Preserve loud failures for infra/programming errors; keep user-code failures
  controllably soft in `error_handling="continue"`.
- Keep the public API stable; limit changes to internals.

## High‑Level Approach

1) Introduce a tiny `ErrorContext` and two guard helpers used everywhere:
   - `maybe_propagate_before_call(...)` checks inputs, returns a
     `PropagatedErrorSnapshot` or `None` (proceed).
   - `maybe_wrap_exception(...)` maps a thrown user exception to either
     `ErrorSnapshot` (continue) or re‑raise (raise mode).

2) Introduce a single `ResourcesEval` tri‑state helper for resource evaluation
   that is used in both single and mapspec paths (and controls executor
   submission): `evaluated(Resources) | skipped | error(ErrorSnapshot)`.

3) Unify the execution engine for generation and eager schedulers: one
   "submit → collect → dump → postprocess" path (sync + async variants) used by
   both, with only the readiness policy differing between schedulers.

4) Normalize the error reason domain (Literal/Enum) and document it.

5) Keep the lightweight marker path for `return_results=False`, but replace the
   tuple‑mixing with a single `ErrorStub` that carries just enough metadata to
   count/propagate errors without holding heavy kwargs in memory.

6) Implement lazy root‑cause resolution for reductions: store compact references
   in `PropagatedErrorSnapshot` so `.get_root_causes()` can resolve upstream
   `ErrorSnapshot`s even when the error originated from array elements.

7) Fix cache cross‑mode by including `error_handling` in the map‑cache key and
   (defensively) re‑raise when a cached `ErrorSnapshot` is encountered under
   `error_handling="raise"`.

## Current Pain Points (Where We Simplify)

- Error flow and resource evaluation are woven through multiple call sites:
  `_run_iteration`, `_execute_single`, `_prepare_kwargs_for_execution`,
  `_maybe_eval_resources`, submission wrappers, and postprocessors.
- Eager vs generation schedulers duplicate parts of submit/collect/postprocess.
- `return_results=False` path introduces `_ErrorMarker` and `_InternalShape`
  mixing that leaks into result piping and dumping logic.
- Reason strings are a free‑form set.
- `get_root_causes()` is a no‑op for partial/array cases.

## Proposed Types and Helpers (Target Design)

### 1) ErrorContext

```
@dataclass(frozen=True)
class ErrorContext:
    mode: Literal["raise", "continue"]
    # Precomputed error info for current call (element‑level for mapspec case)
    error_info: dict[str, ErrorInfo] | None
    # Resource evaluation result placeholder (set later)
    resources: Resources | None = None
    resources_state: Literal["unknown", "evaluated", "skipped", "error"] = "unknown"
```

- Created at the earliest point that we have the kwargs available.
- For mapspec: `error_info` comes from scanning the element kwargs.
- For non‑mapspec: `error_info` is also computed once per call.

### 2) Guard: maybe_propagate_before_call

```
def maybe_propagate_before_call(
    ctx: ErrorContext,
    func: PipeFunc,
    kwargs: dict[str, Any],
) -> PropagatedErrorSnapshot | None:
    # If ctx.mode == "continue" and ctx.error_info indicates any errors in inputs,
    # return a PropagatedErrorSnapshot with normalized reasons.
```

- This folds the existing `propagate_input_errors` path and the reason
  computation into one helper.

### 3) Guard: maybe_wrap_exception

```
def maybe_wrap_exception(
    ctx: ErrorContext,
    func: PipeFunc,
    kwargs: dict[str, Any],
    exc: BaseException,
) -> ErrorSnapshot:
    # If ctx.mode=="continue": return ErrorSnapshot and attach to func/pipeline.
    # Else: re‑raise (after attaching a snapshot and printing guidance for debugging).
```

- Consolidates repeated try/except branches and guarantees consistent
  attachment/printing semantics.

### 4) Resources Evaluation Helper

```
@dataclass(frozen=True)
class ResourcesEval:
    state: Literal["evaluated", "skipped", "error"]
    resources: Resources | None
    snapshot: ErrorSnapshot | None


def eval_resources(
    *,
    func: PipeFunc,
    map_scope_kwargs: dict[str, Any] | None,
    element_kwargs: dict[str, Any] | None,
    mode: Literal["raise", "continue"],
) -> ResourcesEval:
    # Decides scope (map vs element), skips on error inputs (continue mode),
    # captures ErrorSnapshot on failure.
```

- Submission code simply checks: if `state == "error"`, do not submit to the
  executor; if `state == "skipped"`, call target without resources; else call
  with resources or pass variable.

### 5) Reason Domain

```
Reason = Literal["input_is_error", "array_contains_errors"]
```

- Map directly to two user‑visible concepts with stable names.

### 6) ErrorStub for return_results=False

```
@dataclass(frozen=True)
class ErrorStub:
    propagated: bool  # True if PropagatedErrorSnapshot, else ErrorSnapshot
```

- Used internally in place of `_ErrorMarker` and `_InternalShape.from_outputs()`.
- Keeps result array updates simpler; dumping still persists the real error
  objects for correctness.

### 7) Lazy Root Causes for Reductions

Add fields to `PropagatedErrorSnapshot` to represent the partial/array case:

```
@dataclass
class PropagatedErrorSnapshot:
    error_info: dict[str, ErrorInfo]
    skipped_function: Callable[..., Any]
    reason: Reason
    attempted_kwargs: dict[str, Any]
    timestamp: str = field(default_factory=_timestamp)
    # New fields for partial arrays (reduction cases):
    source_output_name: str | None = None
    error_indices: tuple[np.ndarray, ...] | None = None
    run_folder: str | None = None
```

- On creation in the map reduction bridge, fill `source_output_name` and
  `error_indices` (from `ErrorInfo`) and set `run_folder` from `RunInfo` if any.
- Implement:

```
def get_root_causes(self) -> list[ErrorSnapshot]:
    # If full error(s), flatten recursively as today.
    # If partial: lazily load only the indicated indices from storage and
    # collect underlying ErrorSnapshot objects.
```

- This keeps snapshots small, conforms with docs promises, and avoids keeping
  the full array in memory.

### 8) Map Cache Key

- Include `error_handling` in the map cache key. Also, if a cached value is an
  `ErrorSnapshot` and `error_handling=="raise"`, immediately re‑raise the stored
  exception after attaching a fresh snapshot.

## Execution Flow (Target)

Below is the single path used by both single and mapspec calls; mapspec adds an
outer loop over indices/chunks.

```
# 1. Build kwargs (single or element‑selected), then load data (FileValue/StorageBase).
kwargs = _select_kwargs_if_needed(...)
kwargs = maybe_load_data(kwargs)

# 2. Prepare context and resource evaluation
error_info = scan_inputs_for_errors(kwargs)  # selective / fast path as today
ctx = ErrorContext(mode=mode, error_info=error_info)
res = eval_resources(func=func, map_scope_kwargs=..., element_kwargs=kwargs, mode=mode)
if res.state == "error":
    # create propagated snapshot from ctx.error_info and res.snapshot
    return create_propagated_error(...)

# 3. Early propagation guard
if snap := maybe_propagate_before_call(ctx, func, kwargs):
    return snap

# 4. Cache lookup (map or single)
if cache:
    cache_key = (func._cache_id, mode, to_hashable(kwargs))
    if cache_key in cache:
        val = cache.get(cache_key)
        if isinstance(val, ErrorSnapshot) and mode == "raise":
            # Re‑raise to keep fail‑loud semantics
            raise maybe_wrap_exception(ctx, func, kwargs, val.exception)
        return val

# 5. Execute user code
try:
    call_kwargs = inject_resources_if_needed(kwargs, res)
    result = func(**call_kwargs)
except Exception as exc:  # noqa: BLE001
    snap = maybe_wrap_exception(ctx, func, kwargs, exc)
    if mode == "continue":
        result = snap
    else:
        raise  # dead code

# 6. Cache put (if enabled)
if cache:
    cache.put(cache_key, result)

# 7. Dump/storage update and progress accounting (shared helper)
_update_result_arrays_and_storage(..., result)
```

This template is used:
- For single (non‑mapspec) calls (once per function)
- For mapspec elements (per index) and chunked submission with the executor.
- For reductions, the same path applies; the only difference is the construction
  of `ErrorInfo` and, when propagating, the population of `source_output_name`,
  `error_indices`, and `run_folder`.

## Scheduler Unification

- Keep the readiness policies separate:
  - Generation: submit after an entire generation is ready (as today).
  - Eager: submit a function as soon as its PipeFunc predecessors complete.
- Share everything else:
  - A single `_submit_function` wrapper that constructs the element/single tasks
    (using the unified flow above), registers futures, and integrates with
    progress.
  - A single `_process_completed_function` that drains futures, calls the
    postprocessor/dumper, updates `RunInfo` shapes, and updates outputs.
- Async variants reuse the same structure with `asyncio.wrap_future`.

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

