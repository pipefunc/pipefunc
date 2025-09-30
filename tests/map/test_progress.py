from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pipefunc import Pipeline, pipefunc
from pipefunc.map._progress import Status
from pipefunc.map._run import _update_status_if_needed

if TYPE_CHECKING:
    import pytest


def _capture_headless_tracker(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from pipefunc.map import _prepare as prepare_mod

    captured: dict[str, Any] = {}
    original = prepare_mod.init_tracker

    def fake_init_tracker(store, functions, show_progress, in_async):  # type: ignore[override]
        tracker = original(store, functions, "headless", in_async)
        captured["tracker"] = tracker
        return tracker

    monkeypatch.setattr(prepare_mod, "init_tracker", fake_init_tracker)
    return captured


def test_progress_counts_only_real_irregular_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = _capture_headless_tracker(monkeypatch)

    @pipefunc(output_name="words", mapspec="text[i] -> words[i, j*]")
    def split_text(text: str) -> list[str]:
        return text.split()

    @pipefunc(output_name="lengths", mapspec="words[i, j*] -> lengths[i, j*]")
    def word_lengths(words: str) -> int:
        return len(words)

    pipeline = Pipeline([split_text, word_lengths])

    inputs = {"text": ["Hello world", "Python is great", "A"]}

    results = pipeline.map(
        inputs=inputs,
        internal_shapes={"words": (3,), "lengths": (3,)},
        storage="dict",
        parallel=False,
        show_progress=True,
    )

    lengths = results["lengths"].output
    assert isinstance(lengths, np.ma.MaskedArray)
    assert lengths.count() == 6

    tracker = captured.get("tracker")
    assert tracker is not None

    lengths_status = tracker.progress_dict["lengths"]
    assert isinstance(lengths_status, Status)
    assert lengths_status.n_total == 6
    assert lengths_status.n_completed == 6
    assert lengths_status.progress == 1.0


def test_update_status_counts_existing_in_total() -> None:
    status = Status(n_total=None)

    _update_status_if_needed(status, existing=[0, 1], missing=[2])
    status.mark_complete()

    assert status.n_total == 3
    assert status.progress == 1.0


def test_progress_overcounts_existing_entries(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured = _capture_headless_tracker(monkeypatch)

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([double])
    inputs = {"x": [1, 2, 3]}

    pipeline.map(
        inputs=inputs,
        storage="file_array",
        parallel=False,
        run_folder=tmp_path,
    )

    missing_path = tmp_path / "outputs" / "y" / "__1__.pickle"
    missing_path.unlink()

    pipeline.map(
        inputs=inputs,
        storage="file_array",
        parallel=False,
        run_folder=tmp_path,
        show_progress=True,
    )

    tracker = captured.get("tracker")
    assert tracker is not None

    status = tracker.progress_dict["y"]
    assert isinstance(status, Status)
    assert status.n_total == len(inputs["x"])  # expected total tasks
    assert status.progress <= 1.0
