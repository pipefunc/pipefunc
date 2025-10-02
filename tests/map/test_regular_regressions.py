from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.map._progress import Status

if TYPE_CHECKING:
    from pipefunc._widgets.progress_headless import HeadlessProgressTracker

has_xarray = importlib.util.find_spec("xarray") is not None


def _regular_pipeline() -> Pipeline:
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double(x: int) -> int:
        return 2 * x

    return Pipeline([double])


def test_regular_arrays_remain_plain_ndarrays() -> None:
    pipeline = _regular_pipeline()
    inputs = {"x": [1, 2, 3]}

    result = pipeline.map(inputs=inputs, storage="dict", parallel=False)

    y_values = result["y"].output
    assert not isinstance(y_values, np.ma.MaskedArray)
    assert isinstance(y_values, np.ndarray)
    assert y_values.tolist() == [2, 4, 6]

    repeat = pipeline.map(inputs=inputs, storage="dict", parallel=False)
    np.testing.assert_array_equal(repeat["y"].output, y_values)


def _capture_headless_tracker(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from pipefunc.map import _prepare as prepare_mod

    captured: dict[str, Any] = {}
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


def test_regular_progress_counts_all_entries(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured = _capture_headless_tracker(monkeypatch)
    pipeline = _regular_pipeline()
    inputs = {"x": [0, 1, 2, 3]}

    pipeline.map(
        inputs=inputs,
        storage="file_array",
        run_folder=tmp_path,
        parallel=False,
        show_progress=True,
    )

    tracker_obj = captured.get("tracker")
    assert tracker_obj is not None

    tracker = cast("HeadlessProgressTracker", tracker_obj)
    status = tracker.progress_dict["y"]
    assert isinstance(status, Status)
    assert status.n_total == len(inputs["x"])
    assert status.n_completed == len(inputs["x"])
    assert status.progress == pytest.approx(1.0)


@pytest.mark.skipif(not has_xarray, reason="xarray is not installed")
def test_regular_to_xarray_has_no_mask_attrs() -> None:
    pipeline = _regular_pipeline()
    inputs = {"x": [4, 5]}

    result = pipeline.map(inputs=inputs, storage="dict", parallel=False)
    ds = result.to_xarray(type_cast=False)

    da = ds["y"]
    assert "_mask" not in da.attrs
    np.testing.assert_array_equal(da.to_numpy(), np.array([8, 10]))

    df = ds.to_dataframe().reset_index(drop=True)
    assert all(not column.startswith("__mask_") for column in df.columns)
