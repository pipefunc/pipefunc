"""Regression tests for continue-mode error handling in chunked map runs."""

from __future__ import annotations

import sys
import threading
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.exceptions import ErrorSnapshot

if TYPE_CHECKING:
    from collections.abc import Sequence

ERROR_INPUT = 13
INPUT_VALUES = list(range(30))


@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def _double_or_fail(x: int) -> int:
    """Return ``2 * x`` except raise on the designated ``ERROR_INPUT``."""

    if x == ERROR_INPUT:
        msg = f"boom at {x}"
        raise ValueError(msg)
    return x * 2


def _assert_expected_outputs(raw_outputs: Sequence) -> None:
    """Validate that the outputs align per-index and capture the real kwargs."""

    outputs = list(raw_outputs)
    assert len(outputs) == len(INPUT_VALUES)
    for idx, source_value in enumerate(INPUT_VALUES):
        entry = outputs[idx]
        if idx == ERROR_INPUT:
            assert isinstance(entry, ErrorSnapshot)
            assert entry.kwargs.get("x") == ERROR_INPUT
        else:
            assert entry == source_value * 2


def test_parallel_continue_preserves_alignment_with_chunk_errors() -> None:
    """Parallel map with chunking should keep per-index ordering and metadata."""

    pipeline = Pipeline([_double_or_fail])
    with ThreadPoolExecutor(max_workers=4) as executor:
        result = pipeline.map(
            {"x": INPUT_VALUES},
            error_handling="continue",
            parallel=True,
            executor=executor,
            chunksizes=10,
        )

    outputs = result["y"].output
    _assert_expected_outputs(outputs)
    assert all(entry is not None for entry in outputs)


class _NoAttrFuture(Future):
    """Future that rejects setting arbitrary attributes (simulates wrapped futures)."""

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_pipefunc_chunk_indices":
            msg = "chunk metadata attribute not supported"
            raise AttributeError(msg)
        super().__setattr__(name, value)


class _ProxyExecutor(Executor):
    """Executor wrapper returning futures without custom attributes."""

    def __init__(self, max_workers: int) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, fn, /, *args, **kwargs):  # type: ignore[override]
        inner = self._pool.submit(fn, *args, **kwargs)
        proxy: _NoAttrFuture = _NoAttrFuture()

        def _transfer(source: Future) -> None:
            exc = source.exception()
            if exc is None:
                proxy.set_result(source.result())
            else:
                proxy.set_exception(exc)

        inner.add_done_callback(_transfer)
        return proxy

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:  # noqa: FBT002
        self._pool.shutdown(wait=wait, cancel_futures=cancel_futures)


def test_parallel_continue_without_future_attrs() -> None:
    """Chunk metadata is preserved even when futures reject dynamic attributes."""

    pipeline = Pipeline([_double_or_fail])
    executor = _ProxyExecutor(max_workers=3)
    try:
        result = pipeline.map(
            {"x": INPUT_VALUES},
            error_handling="continue",
            parallel=True,
            executor=executor,
            chunksizes=8,
        )
    finally:
        executor.shutdown()

    outputs = result["y"].output
    _assert_expected_outputs(outputs)


@pytest.mark.asyncio
async def test_map_async_continue_returns_per_index_outputs() -> None:
    """Async continue-mode should surface chunk failures without crashing."""

    pipeline = Pipeline([_double_or_fail])
    with ThreadPoolExecutor(max_workers=4) as executor:
        async_run = pipeline.map_async(
            {"x": INPUT_VALUES},
            error_handling="continue",
            chunksizes=6,
            executor=executor,
            start=True,
        )
        result = await async_run.task

    outputs = result["y"].output
    _assert_expected_outputs(outputs)


@pipefunc(output_name="ynon")
def _single_fail(x: int) -> int:
    msg = "boom"
    raise ValueError(msg)


def test_parallel_continue_non_mapspec_executor() -> None:
    """Non-mapspec futures should still record snapshot outputs by index."""

    pipeline = Pipeline([_single_fail])
    with ThreadPoolExecutor(max_workers=2) as executor:
        result = pipeline.map(
            {"x": 1},
            error_handling="continue",
            parallel=True,
            executor=executor,
            show_progress="headless",
        )

    snapshot = result["ynon"].output
    assert isinstance(snapshot, ErrorSnapshot)
    assert snapshot.kwargs["x"] == 1


@pytest.mark.asyncio
async def test_map_async_continue_non_mapspec_executor() -> None:
    """Async non-mapspec futures also surface snapshots without chunk metadata."""

    pipeline = Pipeline([_single_fail])
    with ThreadPoolExecutor(max_workers=2) as executor:
        async_map = pipeline.map_async(
            {"x": 1},
            error_handling="continue",
            executor=executor,
            start=True,
        )
        result = await async_map.task

    snapshot = result["ynon"].output
    assert isinstance(snapshot, ErrorSnapshot)


@pipefunc(output_name="yplain", mapspec="x[i] -> yplain[i]")
def _identity(x: int) -> int:
    return x


def test_parallel_raise_mode_sequences() -> None:
    pipeline = Pipeline([_double_or_fail])
    with pytest.raises(ValueError, match="boom at 13"):
        pipeline.map({"x": [ERROR_INPUT]}, error_handling="raise", parallel=False)


def test_parallel_internal_shape_progress() -> None:
    pipeline = Pipeline([_identity])
    with ThreadPoolExecutor(max_workers=2) as executor:
        pipeline.map(
            {"x": list(range(4))},
            error_handling="continue",
            parallel=True,
            executor=executor,
            chunksizes=2,
            return_results=False,
            show_progress="headless",
        )


def test_parallel_raise_mode_with_executor() -> None:
    """Parallel execution with raise mode should propagate exceptions."""
    pipeline = Pipeline([_double_or_fail])
    with ThreadPoolExecutor(max_workers=2) as executor:  # noqa: SIM117
        with pytest.raises(ValueError, match="boom at 13"):
            pipeline.map(
                {"x": list(range(20))},
                error_handling="raise",
                parallel=True,
                executor=executor,
                chunksizes=5,
            )


@pytest.mark.asyncio
async def test_async_raise_mode() -> None:
    """Async execution with raise mode should propagate exceptions."""
    pipeline = Pipeline([_double_or_fail])
    with ThreadPoolExecutor(max_workers=2) as executor:
        async_run = pipeline.map_async(
            {"x": list(range(20))},
            error_handling="raise",
            executor=executor,
            chunksizes=5,
            start=True,
        )
        with pytest.raises(ValueError, match="boom at 13"):
            await async_run.task


@pipefunc(output_name=("multi1", "multi2"), mapspec="x[i] -> multi1[i], multi2[i]")
def _tuple_with_errors(x: int) -> tuple[int, int]:
    """Return tuple output, raise on ERROR_INPUT."""
    if x == ERROR_INPUT:
        msg = f"boom at {x}"
        raise ValueError(msg)
    return (x * 2, x * 3)


def test_tuple_output_error_counting() -> None:
    """Tuple outputs with errors should be counted correctly in progress tracking."""
    pipeline = Pipeline([_tuple_with_errors])
    with ThreadPoolExecutor(max_workers=2) as executor:
        result = pipeline.map(
            {"x": [1, 2, ERROR_INPUT, 6]},
            error_handling="continue",
            parallel=True,
            executor=executor,
            chunksizes=2,
            show_progress="headless",  # Triggers status tracking
        )

    # Both outputs should have ErrorSnapshot at index 2
    assert isinstance(result["multi1"].output[2], ErrorSnapshot)
    assert isinstance(result["multi2"].output[2], ErrorSnapshot)
    assert result["multi1"].output[0] == 2
    assert result["multi2"].output[0] == 3


@pipefunc(output_name="unpicklable")
def _return_lock(x: int) -> threading.Lock:
    """Returns unpicklable object to trigger serialization error."""
    return threading.Lock()


@pipefunc(output_name="unpicklable_map", mapspec="x[i] -> unpicklable_map[i]")
def _return_lock_map(x: int) -> threading.Lock:
    """Returns unpicklable object with mapspec."""
    return threading.Lock()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="ProcessPoolExecutor pickling behavior differs on Windows",
)
def test_serialization_error_non_mapspec_continue() -> None:
    """Infrastructure failures (serialization) should create ErrorSnapshots for non-mapspec."""
    pipeline = Pipeline([_return_lock])
    with ProcessPoolExecutor(max_workers=1) as executor:
        result = pipeline.map(
            {"x": 1},
            error_handling="continue",
            parallel=True,
            executor=executor,
        )

    snapshot = result["unpicklable"].output
    assert isinstance(snapshot, ErrorSnapshot)
    # Should contain PicklingError or TypeError in the exception
    assert "pickle" in str(snapshot.exception).lower() or "lock" in str(snapshot.exception).lower()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="ProcessPoolExecutor pickling behavior differs on Windows",
)
def test_serialization_error_mapspec_continue() -> None:
    """Infrastructure failures (serialization) should create per-element ErrorSnapshots for mapspec."""
    pipeline = Pipeline([_return_lock_map])
    with ProcessPoolExecutor(max_workers=1) as executor:
        result = pipeline.map(
            {"x": [1, 2, 3, 4]},
            error_handling="continue",
            parallel=True,
            executor=executor,
            chunksizes=2,
        )

    outputs = result["unpicklable_map"].output
    # All should be ErrorSnapshots due to serialization failure
    assert all(isinstance(out, ErrorSnapshot) for out in outputs)
    # Check that each has correct kwargs
    for i, snapshot in enumerate(outputs, start=1):
        assert snapshot.kwargs["x"] == i


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="ProcessPoolExecutor pickling behavior differs on Windows",
)
async def test_async_serialization_error_continue() -> None:
    """Async map should handle infrastructure failures gracefully."""
    pipeline = Pipeline([_return_lock_map])
    with ProcessPoolExecutor(max_workers=1) as executor:
        async_run = pipeline.map_async(
            {"x": [1, 2, 3]},
            error_handling="continue",
            executor=executor,
            chunksizes=2,
            start=True,
        )
        result = await async_run.task

    outputs = result["unpicklable_map"].output
    assert all(isinstance(out, ErrorSnapshot) for out in outputs)
    # Each snapshot should have correct per-element kwargs
    for i, snapshot in enumerate(outputs, start=1):
        assert snapshot.kwargs["x"] == i


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="ProcessPoolExecutor pickling behavior differs on Windows",
)
async def test_async_serialization_error_non_mapspec_continue() -> None:
    """Async map should handle non-mapspec infrastructure failures gracefully."""
    pipeline = Pipeline([_return_lock])
    with ProcessPoolExecutor(max_workers=1) as executor:
        async_run = pipeline.map_async(
            {"x": 1},
            error_handling="continue",
            executor=executor,
            start=True,
        )
        result = await async_run.task

    snapshot = result["unpicklable"].output
    assert isinstance(snapshot, ErrorSnapshot)
    # Should contain PicklingError or TypeError in the exception
    assert "pickle" in str(snapshot.exception).lower() or "lock" in str(snapshot.exception).lower()
