from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pipefunc import Pipeline, Sweep, pipefunc
from pipefunc.map._run import load_outputs, map_shapes

if TYPE_CHECKING:
    from pathlib import Path


def test_simple_sweep(tmp_path: Path) -> None:
    @pipefunc(output_name="z", mapspec="x[i] -> z[i]")
    def add(x: int, y: int) -> int:
        assert isinstance(x, int)
        assert isinstance(y, int)
        return x + y

    @pipefunc(output_name="sum")
    def take_sum(z: np.ndarray[Any, np.dtype[np.int_]]) -> int:
        assert isinstance(z, np.ndarray)
        return sum(z)

    pipeline = Pipeline([add, take_sum])

    inputs = {"x": [1, 2, 3], "y": 2}
    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False)
    assert results["sum"].output == 12
    assert results["sum"].output_name == "sum"
    assert load_outputs("sum", run_folder=tmp_path) == 12
    shapes, masks = map_shapes(pipeline, inputs)
    assert shapes == {"x": (3,), "z": (3,)}
    assert all(all(mask) for mask in masks.values())

    sweep = Sweep({"y": [42, 69]}, constants={"x": [1, 2, 3]})
    for i, combo in enumerate(sweep.generate()):
        run_folder = tmp_path / f"sweep_{i}"
        results = pipeline.map(combo, run_folder=run_folder, parallel=False)
