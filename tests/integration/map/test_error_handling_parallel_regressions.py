"""Regression tests for continue-mode error handling in chunked map runs."""

from __future__ import annotations

from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc._widgets.progress_base import ProgressTrackerBase
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


@pipefunc(output_name="yfail", mapspec="x[i] -> yfail[i]")
def _always_fail(x: int) -> int:
    """Always raise to exercise continue-mode progress accounting."""

    msg = "boom"
    raise ValueError(msg)


def _capture_headless_tracker(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    from pipefunc.map import _prepare as prepare_mod

    captured: dict[str, object] = {}
    original = prepare_mod.init_tracker

    def fake_init_tracker(*args, **kwargs):  # type: ignore[override]
        args = list(args)
        if len(args) >= 3:
            args[2] = "headless"
        elif "show_progress" in kwargs:
            kwargs["show_progress"] = "headless"
        else:
            args.append("headless")
        tracker = original(*args, **kwargs)
        captured["tracker"] = tracker
        return tracker

    monkeypatch.setattr(prepare_mod, "init_tracker", fake_init_tracker)
    return captured


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


def test_chunk_continue_handles_mid_chunk_failure() -> None:
    """Chunked continue-mode should preserve earlier successes when a later element fails."""

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def fail_on_second(x: int) -> int:
        if x == 2:
            msg = "boom at index 2"
            raise ValueError(msg)
        return x * 10

    pipeline = Pipeline([fail_on_second])

    inputs = {"x": [0, 1, 2, 3, 4]}
    with ThreadPoolExecutor(max_workers=2) as executor:
        result = pipeline.map(
            inputs,
            error_handling="continue",
            parallel=True,
            executor=executor,
            chunksizes=4,
        )

    outputs = list(result["y"].output)
    assert outputs[0] == 0
    assert outputs[1] == 10
    assert isinstance(outputs[2], ErrorSnapshot)
    assert outputs[3] == 30
    assert outputs[4] == 40


def test_continue_headless_counts_failures_without_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Continue-mode must surface failures in progress when results are dropped."""

    captured = _capture_headless_tracker(monkeypatch)
    pipeline = Pipeline([_always_fail])
    inputs = {"x": [0, 1, 2, 3]}

    with ThreadPoolExecutor(max_workers=2) as executor:
        pipeline.map(
            inputs,
            error_handling="continue",
            parallel=True,
            executor=executor,
            chunksizes=2,
            return_results=False,
            show_progress=True,
        )

    tracker_obj = captured.get("tracker")
    assert isinstance(tracker_obj, ProgressTrackerBase)
    status = tracker_obj.progress_dict["yfail"]
    assert status.n_total == len(inputs["x"])
    assert status.n_failed == len(inputs["x"])
    assert status.n_completed == 0


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
