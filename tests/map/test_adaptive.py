from __future__ import annotations

from typing import TYPE_CHECKING

import adaptive
import numpy as np
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.map import load_outputs
from pipefunc.map._run_info import RunInfo
from pipefunc.map.adaptive import create_learners, create_learners_from_sweep, flatten_learners
from pipefunc.sweep import Sweep

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
    adaptive.runner.simple(flat_learners["z"][0])
    assert flat_learners["z"][0].data == {
        0: (2,),
        1: (3,),
        2: (4,),
        3: (3,),
        4: (4,),
        5: (5,),
        6: (4,),
        7: (5,),
        8: (6,),
    }
    adaptive.runner.simple(flat_learners["prod"][0])
    assert flat_learners["prod"][0].data == {0: 172800}


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
        internal_shapes={"x": 4},  # 4 should become (4,)
        return_output=True,
    )
    flat_learners = flatten_learners(learners_dicts)
    assert len(flat_learners) == 3
    assert sum(len(learners) for learners in flat_learners.values()) == 3
    adaptive.runner.simple(flat_learners["x"][0])
    assert flat_learners["x"][0].data == {0: [0, 1, 2, 3]}
    adaptive.runner.simple(flat_learners["y"][0])
    assert flat_learners["y"][0].data == {0: (0,), 1: (2,), 2: (4,), 3: (6,)}
    adaptive.runner.simple(flat_learners["sum"][0])
    assert flat_learners["sum"][0].data == {0: 12}


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
    for learners in flat_learners.values():
        for learner in learners:
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
    for learners in flat_learners.values():
        for learner in learners:
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


def test_basic_with_fixed_indices(tmp_path: Path) -> None:
    @pipefunc(output_name="z", mapspec="x[i], y[j] -> z[i, j]")
    def add(x: int, y: int) -> tuple[int, int]:
        assert isinstance(x, int)
        assert isinstance(y, int)
        return x, y

    pipeline = Pipeline([add])

    inputs = {"x": [1, 2, 3], "y": [1, 2, 3]}
    learners_dicts = create_learners(
        pipeline,
        inputs,
        run_folder=tmp_path,
        return_output=True,
        fixed_indices={"i": 0},
    )
    flat_learners = flatten_learners(learners_dicts)
    assert len(flat_learners) == 1
    adaptive.runner.simple(flat_learners["z"][0])
    assert flat_learners["z"][0].data == {0: ((1, 1),), 1: ((1, 2),), 2: ((1, 3),)}
    run_info = RunInfo.load(run_folder=tmp_path)
    store = run_info.init_store()
    assert store["z"].to_array().tolist() == [
        [(1, 1), (1, 2), (1, 3)],
        [None, None, None],
        [None, None, None],
    ]

    with pytest.raises(ValueError, match="Got extra `fixed_indices`: `{'not_exist'}`"):
        create_learners(
            pipeline,
            inputs,
            run_folder=tmp_path,
            return_output=True,
            fixed_indices={"not_exist": 0},
        )


def test_basic_with_split_independent_axes(tmp_path: Path) -> None:
    @pipefunc(output_name="z", mapspec="x[i], y[i, j] -> z[i, j]")
    def add(x: int, y: int) -> tuple[int, int]:
        assert isinstance(x, int)
        assert isinstance(y, np.int_)
        return x, y

    pipeline = Pipeline([add])

    inputs = {"x": [1, 2, 3], "y": np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])}
    learners_dicts = create_learners(
        pipeline,
        inputs,
        run_folder=tmp_path,
        return_output=True,
        split_independent_axes=True,
    )
    flat_learners = flatten_learners(learners_dicts)
    assert len(flat_learners) == 1
    assert len(flat_learners["z"]) == 12
    for learner in flat_learners["z"][:-3]:
        adaptive.runner.simple(learner)
    assert flat_learners["z"][0].data == {0: ((1, 1),)}
    run_info = RunInfo.load(run_folder=tmp_path)
    store = run_info.init_store()
    assert store["z"].to_array().tolist() == [
        [(1, 1), (1, 2), (1, 3), (1, 4)],
        [(2, 1), (2, 2), (2, 3), (2, 4)],
        [(3, 1), None, None, None],  # only last 3 missing because of `[:-3]` above
    ]


def test_create_learners_split_axes_with_reduction(tmp_path: Path) -> None:
    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        return 2 * x

    @pipefunc(output_name="sum")
    def take_sum(y) -> int:
        assert isinstance(y, np.ndarray)
        return sum(y)

    pipeline = Pipeline(
        [
            (double_it, "x[i] -> y[i]"),
            take_sum,
        ],
    )
    pipeline.add_mapspec_axis("x", axis="j")

    inputs = {"x": np.array([[0, 1, 2, 3], [0, 1, 2, 3]])}
    results = pipeline.map(inputs, tmp_path, parallel=False)
    learners = create_learners(
        pipeline,
        inputs,
        tmp_path,
        return_output=True,
        split_independent_axes=True,
    )
    flat_learners = flatten_learners(learners)
    for learners_list in flat_learners.values():
        for learner in learners_list:
            adaptive.runner.simple(learner)
    assert results["sum"].output.tolist() == [0, 4, 8, 12]
    assert [learner.data for learner in flat_learners["sum"]] == [
        {0: (0,)},
        {0: (4,)},
        {0: (8,)},
        {0: (12,)},
    ]
    assert len(learners) == 4
    assert list(learners.keys()) == [
        (("j", 0),),
        (("j", 1),),
        (("j", 2),),
        (("j", 3),),
    ]
