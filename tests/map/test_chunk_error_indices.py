"""Regression tests for preserving chunk indices in error snapshots."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from pipefunc import Pipeline, pipefunc


@pipefunc(output_name="y", mapspec="x[i] -> y[i]")
def _fail_on_two(x: int) -> int:
    if x == 2:
        msg = "boom"
        raise ValueError(msg)
    return x * 2


def test_chunk_error_uses_correct_index() -> None:
    pipeline = Pipeline([_fail_on_two])
    inputs = {"x": [0, 1, 2, 3]}

    with (
        ThreadPoolExecutor(max_workers=2) as executor,
        pytest.raises(
            ValueError,
            match="boom",
        ),
    ):
        pipeline.map(
            inputs,
            parallel=True,
            executor=executor,
            chunksizes=2,
        )

    snapshot = pipeline.error_snapshot
    assert snapshot is not None
    assert snapshot.kwargs["x"] == 2
