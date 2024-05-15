from __future__ import annotations

from typing import TYPE_CHECKING

import adaptive
import numpy as np

from pipefunc import Pipeline, pipefunc
from pipefunc.map._adaptive import make_learners

if TYPE_CHECKING:
    from pathlib import Path


def test_basic(tmp_path: Path) -> None:
    @pipefunc(output_name="z")
    def add(x: int, y: int) -> int:
        assert isinstance(x, int)
        assert isinstance(y, int)
        return x + y

    @pipefunc(output_name="prod")
    def take_sum(z: np.ndarray) -> int:
        assert isinstance(z, np.ndarray)
        return np.prod(z)

    pipeline = Pipeline(
        [
            (add, "x[i], y[j] -> z[i, j]"),
            take_sum,
        ],
    )

    inputs = {"x": [1, 2, 3], "y": [1, 2, 3]}
    learners_dicts = make_learners(
        pipeline,
        inputs,
        run_folder=tmp_path,
        return_output=True,
    )
    flat_learners = {
        k: v for learner_dict in learners_dicts for k, v in learner_dict.items()
    }
    assert len(flat_learners) == 2
    adaptive.runner.simple(flat_learners["z"])
    assert flat_learners["z"].data == {
        0: [2],
        1: [3],
        2: [4],
        3: [3],
        4: [4],
        5: [5],
        6: [4],
        7: [5],
        8: [6],
    }
    adaptive.runner.simple(flat_learners["prod"])
    assert flat_learners["prod"].data == {0: 172800}


def test_simple_from_step(tmp_path: Path) -> None:
    @pipefunc(output_name="x")
    def generate_seeds(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        assert isinstance(x, int)
        return x * 2

    @pipefunc(output_name="sum")
    def take_sum(y: list[int]) -> int:
        return sum(y)

    pipeline = Pipeline(
        [
            generate_seeds,
            (double_it, "x[i] -> y[i]"),
            take_sum,
        ],
    )
    inputs = {"n": 4}
    learners_dicts = make_learners(
        pipeline,
        inputs,
        run_folder=tmp_path,
        manual_shapes={"x": 4},  # 4 should become (4,)
        return_output=True,
    )
    flat_learners = {
        k: v for learner_dict in learners_dicts for k, v in learner_dict.items()
    }
    assert len(flat_learners) == 3
    adaptive.runner.simple(flat_learners["x"])
    assert flat_learners["x"].data == {0: [0, 1, 2, 3]}
    adaptive.runner.simple(flat_learners["y"])
    assert flat_learners["y"].data == {0: [0], 1: [2], 2: [4], 3: [6]}
    adaptive.runner.simple(flat_learners["sum"])
    assert flat_learners["sum"].data == {0: 12}
