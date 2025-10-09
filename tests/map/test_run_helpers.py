from __future__ import annotations

import asyncio
from concurrent.futures import Future

import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.exceptions import ErrorSnapshot
from pipefunc.map._prepare import prepare_run
from pipefunc.map._run import (
    _count_errors_in_result,
    _entry_contains_error,
    _InternalShape,
    _raise_and_set_error_snapshot,
    _result,
    _result_async,
    _run_iteration,
)


@pipefunc(output_name="y")
def _always_fail(x: int) -> int:
    msg = "boom"
    raise ValueError(msg)


@pipefunc(output_name="ym", mapspec="x[i] -> ym[i]")
def _maybe_fail_mapping(x: int) -> int:
    if x == 13:
        msg = "boom"
        raise ValueError(msg)
    return x * 2


@pytest.fixture
def non_mapping_setup():
    pipeline = Pipeline([_always_fail])
    prepared = prepare_run(
        pipeline=pipeline,
        inputs={"x": [0]},
        run_folder=None,
        internal_shapes=None,
        output_names=None,
        parallel=False,
        executor=None,
        chunksizes=None,
        storage=None,
        cleanup=True,
        fixed_indices=None,
        auto_subpipeline=False,
        show_progress=False,
        error_handling="continue",
        in_async=False,
    )
    func = pipeline.functions[0]
    kwargs = {"x": 0}
    return func, kwargs, prepared.run_info


@pytest.fixture
def mapping_setup():
    pipeline = Pipeline([_maybe_fail_mapping])
    prepared = prepare_run(
        pipeline=pipeline,
        inputs={"x": list(range(3))},
        run_folder=None,
        internal_shapes=None,
        output_names=None,
        parallel=False,
        executor=None,
        chunksizes=None,
        storage=None,
        cleanup=True,
        fixed_indices=None,
        auto_subpipeline=False,
        show_progress=False,
        error_handling="continue",
        in_async=False,
    )
    func = pipeline.functions[0]
    kwargs = {"x": list(range(3))}
    return func, kwargs, prepared.run_info


def test_run_iteration_raises_in_raise_mode():
    func = Pipeline([_always_fail]).functions[0]
    with pytest.raises(ValueError, match="boom"):
        _run_iteration(func, {"x": 0}, cache=None, error_handling="raise")


def test_entry_contains_error_variants():
    snapshot = ErrorSnapshot(_always_fail.func, ValueError("x"), args=(), kwargs={})
    assert _entry_contains_error(snapshot)
    assert not _entry_contains_error(_InternalShape((1,)))
    assert not _entry_contains_error((1, 2))
    assert _count_errors_in_result([snapshot, 1]) == 1


def test_result_future_without_chunk_indices(non_mapping_setup):
    func, kwargs, run_info = non_mapping_setup
    fut = Future()
    fut.set_exception(ValueError("boom"))
    snapshot = _result(fut, func, kwargs, run_info)
    assert isinstance(snapshot, ErrorSnapshot)
    assert snapshot.kwargs["x"] == 0


def test_result_future_with_chunk_indices(mapping_setup):
    func, kwargs, run_info = mapping_setup
    fut = Future()
    fut.set_exception(ValueError("boom"))
    outputs = _result(fut, func, kwargs, run_info, chunk_indices=(0, 2))
    assert isinstance(outputs, list)
    assert all(isinstance(entry[0], ErrorSnapshot) for entry in outputs)


@pytest.mark.asyncio
async def test_result_async_branches(mapping_setup):
    func, kwargs, run_info = mapping_setup
    loop = asyncio.get_running_loop()
    fut = Future()
    fut.set_exception(ValueError("boom"))
    snapshot = await _result_async(fut, loop, func, kwargs, run_info)
    assert isinstance(snapshot, ErrorSnapshot)

    fut_chunked = Future()
    fut_chunked.set_exception(ValueError("boom"))
    outputs = await _result_async(fut_chunked, loop, func, kwargs, run_info, chunk_indices=(0,))
    assert isinstance(outputs, list)
    assert isinstance(outputs[0][0], ErrorSnapshot)


def test_raise_and_set_error_snapshot_with_index(mapping_setup):
    func, kwargs, run_info = mapping_setup
    snapshot = _raise_and_set_error_snapshot(ValueError("boom"), func, kwargs, run_info, index=1)
    assert isinstance(snapshot, ErrorSnapshot)
    assert snapshot.kwargs["x"] == 1
