from __future__ import annotations

from typing import TYPE_CHECKING

import adaptive
import numpy as np
import pytest

from pipefunc import Pipeline, Sweep, pipefunc
from pipefunc.map import load_outputs
from pipefunc.map.adaptive import create_learners, create_learners_from_sweep, flatten_learners

if TYPE_CHECKING:
    from pathlib import Path

# Tests with create_learners


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

    pipeline = Pipeline([(add, "x[i], y[j] -> z[i, j]"), take_sum])

    inputs = {"x": [1, 2, 3], "y": [1, 2, 3]}
    learners_dicts = create_learners(
        pipeline,
        inputs,
        run_folder=tmp_path,
        return_output=True,
    )
    flat_learners = flatten_learners(learners_dicts)
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
    learners_dicts = create_learners(
        pipeline,
        inputs,
        run_folder=tmp_path,
        manual_shapes={"x": 4},  # 4 should become (4,)
        return_output=True,
    )
    flat_learners = flatten_learners(learners_dicts)
    assert len(flat_learners) == 3
    adaptive.runner.simple(flat_learners["x"])
    assert flat_learners["x"].data == {0: [0, 1, 2, 3]}
    adaptive.runner.simple(flat_learners["y"])
    assert flat_learners["y"].data == {0: [0], 1: [2], 2: [4], 3: [6]}
    adaptive.runner.simple(flat_learners["sum"])
    assert flat_learners["sum"].data == {0: 12}


@pytest.mark.parametrize("return_output", [True, False])
def test_create_learners_loading_data(tmp_path: Path, return_output: bool) -> None:  # noqa: FBT001
    counters = {"add": 0, "take_sum": 0}

    @pipefunc(output_name="z")
    def add(x: int, y: int) -> int:
        counters["add"] += 1
        assert isinstance(x, int)
        assert isinstance(y, int)
        return x + y

    @pipefunc(output_name="prod")
    def take_sum(z: np.ndarray) -> int:
        counters["take_sum"] += 1
        assert isinstance(z, np.ndarray)
        return np.prod(z)

    pipeline = Pipeline([(add, "x[i], y[j] -> z[i, j]"), take_sum])
    inputs = {"x": [1, 2], "y": [3, 4]}

    # First run, should create the files
    learners_dicts = create_learners(
        pipeline,
        inputs,
        run_folder=tmp_path,
        return_output=return_output,
        cleanup=True,
    )
    flat_learners = flatten_learners(learners_dicts)
    for learner in flat_learners.values():
        adaptive.runner.simple(learner)
    assert counters["add"] == 4
    assert counters["take_sum"] == 1

    # Second run, should load the files (not run the functions again)
    learners_dicts = create_learners(
        pipeline,
        inputs,
        run_folder=tmp_path,
        return_output=return_output,
        cleanup=False,
    )
    flat_learners = flatten_learners(learners_dicts)
    for learner in flat_learners.values():
        adaptive.runner.simple(learner)
    assert counters["add"] == 4
    assert counters["take_sum"] == 1


# Tests with create_learners_from_sweep


def test_create_learners_from_sweep(tmp_path: Path) -> None:
    counters = {"add": 0, "take_sum": 0}

    @pipefunc(output_name="z")
    def add(x: int, y: int) -> int:
        assert isinstance(x, int)
        assert isinstance(y, int)
        counters["add"] += 1
        return x + y

    @pipefunc(output_name="prod")
    def take_sum(z: np.ndarray) -> int:
        assert isinstance(z, np.ndarray)
        counters["take_sum"] += 1
        return np.prod(z)

    pipeline = Pipeline([(add, "x[i], y[j] -> z[i, j]"), take_sum])
    sweep = Sweep({"y": [[1, 2], [3, 4, 5]]}, constants={"x": [1, 2]})
    learners, folders = create_learners_from_sweep(
        pipeline,
        sweep,
        run_folder=tmp_path,
        cleanup=True,
        parallel=False,  # otherwise the counters won't be set
    )
    for learner in learners:
        adaptive.runner.simple(learner)
    for folder in folders:
        assert load_outputs("prod", run_folder=folder) > 0
    assert counters["add"] == 10
    assert counters["take_sum"] == 2

    # Run again, should load the data
    learners, folders = create_learners_from_sweep(
        pipeline,
        sweep,
        run_folder=tmp_path,
        cleanup=False,
        parallel=False,  # otherwise the counters won't be set
    )
    for learner in learners:
        adaptive.runner.simple(learner)
    assert counters["add"] == 10
    assert counters["take_sum"] == 2

    # Run again, now cleaning up the data
    learners, folders = create_learners_from_sweep(
        pipeline,
        sweep,
        run_folder=tmp_path,
        cleanup=True,
        parallel=False,  # otherwise the counters won't be set
    )
    for learner in learners:
        adaptive.runner.simple(learner)
    assert counters["add"] == 20
    assert counters["take_sum"] == 4
