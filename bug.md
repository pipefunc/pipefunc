# Chunked Map Error Overwrite Bug

## Summary
When a `Pipeline.map_async`/`Pipeline.map` run uses chunked execution (non-trivial
`chunksizes`) together with `error_handling="continue"`, a failure in the tail
of a chunk causes the framework to synthesize error outputs for **every** index
in that chunk. Those synthetic errors are then written back into the storage
array and the final `ResultDict`, overwriting successful results that were
produced earlier in the chunk.

## How It Happens
1. `_maybe_parallel_map` assigns a chunk of indices to an executor future via
   `_submit` (using `_process_chunk`).
2. Each index in the chunk executes `process_index` sequentially inside the
   worker. Successful indices finish and `_run_iteration_and_process` writes
   their outputs into the backing `StorageBase` before moving to the next index.
3. A later index in the same chunk raises, so the worker surfaces the exception;
   the future returned to the driver fails.
4. In the driver, `_result(..., chunk_indices=...)` catches the exception and
   calls `_error_outputs_for_chunk`, which currently fabricates an
   `ErrorSnapshot` for **every** index listed in `chunk_indices`.
5. `_output_from_mapspec_task` then dumps those fabricated errors straight back
   into the store, replacing any real outputs that had already been written.

The net effect: every element in the chunk is reported as an error even if only
the last element actually failed.

## Current Patch
The quick fix we pushed earlier dodges the overwrite by inspecting the store and
reusing whatever was already persisted. It works in practice for the regression,
but it keeps the responsibility spread across storage lookups and doesn’t make
the error signalling explicit.

## Proposed Proper Fix (Sketch)
Goal: represent partial successes directly in the object returned from the
executor so the driver knows exactly which entries succeeded and which failed.

High-level steps:

1. **Wrap `_process_chunk`:** instead of `list(map(process_index, chunk))`, run a
   manual loop that records per-index results. Pseudocode:

   ```python
   def _process_chunk(chunk, process_index):
       outputs = []
       for idx in chunk:
           try:
               outputs.append((idx, process_index(idx), None))
           except Exception as exc:
               outputs.append((idx, None, exc))
               raise ChunkFailure(outputs)  # custom exception carrying partial results
       return outputs
   ```

   where `ChunkFailure` subclasses `Exception` and holds the partial outputs so
   they are not lost.

2. **Update `_result`:** when catching exceptions from the future, detect
   `ChunkFailure`. For entries with `value is not None`, hand that value through
   unchanged; for entries with `exc` set, create an `ErrorSnapshot`.

3. **Adapt `_error_outputs_for_chunk`:** it should now accept the structured
   data coming from `ChunkFailure`, not fabricate errors blindly.

4. **Storage writes:** `_output_from_mapspec_task` will receive the real values
   for successful indices and snapshots only for the failing ones, so it can
   write each item exactly once. No extra store lookups necessary.

5. **Async path:** mirror the same behaviour in `_result_async` so the async map
   runner shares the fix.

6. **Tests:** add a regression test that forces chunking (e.g. `chunksizes=2`)
   with a ProcessPool executor so that the executor raises after the second
   element. Assert that successful indices retain their numerical values while
   only the failing index holds an `ErrorSnapshot`.

## Status
- ✅ Root cause identified (fabricating snapshots for whole chunk).
- ✅ Minimal patch implemented (reuse store value when present).
- ☐ Structural fix (ChunkFailure + structured return) still outstanding.
- ☐ Regression tests for the structured approach (once implemented).
