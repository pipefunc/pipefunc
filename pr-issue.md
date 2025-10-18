## Regression: Loading Legacy ErrorSnapshot Pickles Fails
- Branch `continue-on-errors2` introduced pickling helpers that unconditionally call `cloudpickle.loads()` on the stored function.
- Snapshots saved before v0.88 stored the live callable, so re-loading them now raises `TypeError: a bytes-like object is required`.
- This surfaced because run folders (`run_folder` / `map_async`) have been persisting `ErrorSnapshot` objects all along; the API contract requires `ErrorSnapshot.load_from_file` to keep working.

## Fix Implemented in Branch
- `cloudunpickle_function_state` now only deserializes when the stored value is `bytes`.
- Applied the same guard when restoring nested errors in `PropagatedErrorSnapshot`.
- Added regression tests that synthesize pre-v0.88 pickles for both `ErrorSnapshot` and `PropagatedErrorSnapshot`.

## Open UX Discussion (follow-up issue?)
- On `main`, any pipeline that uses a `run_folder` (explicitly or via `pipeline.map_async`) ends up writing `ErrorSnapshot` objects to disk because the default storage (`file_array`) pickles every element.
- Users may be surprised that transient errors become persisted artefacts.
- Potential next steps: document the behaviour, or add a storage policy knob (`persist_errors=False` / `errors="none|full"`).
