from __future__ import annotations

from typing import TYPE_CHECKING

import adaptive
import numpy as np
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.map._load import load_outputs
from pipefunc.map._run_info import RunInfo
from pipefunc.map._storage_array._base import StorageBase
from pipefunc.map.adaptive import (
    LearnersDict,
    create_learners,
    create_learners_from_sweep,
    to_adaptive_learner,
)
from pipefunc.sweep import Sweep
from pipefunc.typing import Array  # noqa: TC001

if TYPE_CHECKING:
    from pathlib import Path

# Tests with create_learners


@pytest.mark.parametrize("storage", ["dict", "file_array"])
def test_basic(tmp_path: Path, storage: str) -> None:
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
    pipeline.update_scope("foo", outputs={"z"})

    inputs = {"x": [1, 2, 3], "y": [1, 2, 3]}
    learners = create_learners(
        pipeline,
        inputs,
        storage=storage,
        run_folder=tmp_path if storage == "file_array" else None,
        return_output=True,
    )
    learners.simple_run()
    flat_learners = learners.flatten()
    assert len(flat_learners) == 2
    assert flat_learners["foo.z"][0].data == {0: 2, 1: 3, 2: 4, 3: 3, 4: 4, 5: 5, 6: 4, 7: 5, 8: 6}
    adaptive.runner.simple(flat_learners["prod"][0])
    assert flat_learners["prod"][0].data == {0: 172800}


@pytest.mark.parametrize("storage", ["dict", "file_array"])
def test_simple_from_step(tmp_path: Path, storage: str) -> None:
    @pipefunc(output_name="x")
    def generate_seeds(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        assert isinstance(x, int)
        return x * 2

    @pipefunc(output_name="sum")
    def take_sum(y: Array[int]) -> int:
        return sum(y)

    pipeline = Pipeline(
        [
            generate_seeds,
            (double_it, "x[i] -> y[i]"),
            take_sum,
        ],
    )
    inputs = {"n": 4}
    learners = create_learners(
        pipeline,
        inputs,
        run_folder=tmp_path if storage == "file_array" else None,
        storage=storage,
        internal_shapes={"x": 4},  # 4 should become (4,)
        return_output=True,
    )
    flat_learners = learners.flatten()
    assert len(flat_learners) == 3
    assert sum(len(learners) for learners in flat_learners.values()) == 3
    adaptive.runner.simple(flat_learners["x"][0])
    assert flat_learners["x"][0].data == {0: [0, 1, 2, 3]}
    adaptive.runner.simple(flat_learners["y"][0])
    assert flat_learners["y"][0].data == {0: 0, 1: 2, 2: 4, 3: 6}
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
    learners = create_learners(
        pipeline,
        inputs,
        run_folder=tmp_path,
        return_output=return_output,
        cleanup=True,
    )
    flat_learners = learners.flatten()
    for learner_list in flat_learners.values():
        for learner in learner_list:
            adaptive.runner.simple(learner)
    assert counters["add"] == 4
    assert counters["take_sum"] == 1

    # Second run, should load the files (not run the functions again)
    learners = create_learners(
        pipeline,
        inputs,
        run_folder=tmp_path,
        return_output=return_output,
        cleanup=False,
    )
    flat_learners = learners.flatten()
    for learner_list in flat_learners.values():
        for learner in learner_list:
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
    learners = create_learners(
        pipeline,
        inputs,
        run_folder=tmp_path,
        return_output=True,
        fixed_indices={"i": 0},
    )
    flat_learners = learners.flatten()
    assert len(flat_learners) == 1
    adaptive.runner.simple(flat_learners["z"][0])
    assert flat_learners["z"][0].data == {0: (1, 1), 1: (1, 2), 2: (1, 3)}
    run_info = RunInfo.load(run_folder=tmp_path)
    store = run_info.init_store()
    assert isinstance(store["z"], StorageBase)
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
    learners = create_learners(
        pipeline,
        inputs,
        run_folder=tmp_path,
        return_output=True,
        split_independent_axes=True,
    )
    flat_learners = learners.flatten()
    assert len(flat_learners) == 1
    assert len(flat_learners["z"]) == 12
    for learner in flat_learners["z"][:-3]:
        adaptive.runner.simple(learner)
    assert flat_learners["z"][0].data == {0: (1, 1)}
    run_info = RunInfo.load(run_folder=tmp_path)
    store = run_info.init_store()
    assert isinstance(store["z"], StorageBase)
    assert store["z"].to_array().tolist() == [
        [(1, 1), (1, 2), (1, 3), (1, 4)],
        [(2, 1), (2, 2), (2, 3), (2, 4)],
        [(3, 1), None, None, None],  # only last 3 missing because of `[:-3]` above
    ]


@pytest.mark.parametrize("storage", ["dict", "file_array"])
def test_create_learners_split_axes_with_reduction(tmp_path: Path, storage: str) -> None:
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
    results = pipeline.map(inputs, tmp_path, parallel=False, storage="dict")
    learners = create_learners(
        pipeline,
        inputs,
        tmp_path if storage == "file_array" else None,
        storage=storage,
        return_output=True,
        split_independent_axes=True,
    )
    flat_learners = learners.flatten()
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


@pytest.mark.parametrize("storage", ["dict", "file_array"])
def test_internal_shapes(storage: str, tmp_path: Path) -> None:
    @pipefunc(output_name="y", mapspec="x[i, j] -> y[i, j]")
    def f(x):
        return x

    @pipefunc(output_name="r", mapspec="y[i, j] -> r[i, j, k]")
    def g(y, z) -> int:
        return z

    pipeline = Pipeline([f, g])

    inputs = {"x": np.array([[0, 1, 2, 3], [0, 1, 2, 3]]), "z": np.arange(5)}
    internal_shapes = {"r": 5}
    results = pipeline.map(
        inputs,
        tmp_path if storage == "file_array" else None,
        internal_shapes,  # type: ignore[arg-type]
        parallel=False,
        storage=storage,
    )
    learners = create_learners(
        pipeline,
        inputs,
        tmp_path / "learners",
        internal_shapes=internal_shapes,  # type: ignore[arg-type]
        return_output=True,
        split_independent_axes=True,
    )
    assert results
    assert learners
    learners.simple_run()
    if storage == "file_array":
        r_map = load_outputs("r", run_folder=tmp_path)
        r_adap = load_outputs("r", run_folder=tmp_path / "learners")
        assert r_map.tolist() == r_adap.tolist()


def test_learners_dict_no_run_info():
    learners_dict = LearnersDict()
    with pytest.raises(ValueError, match="`run_info` must be provided"):
        learners_dict.to_slurm_run()


@pytest.fixture
def pipeline() -> Pipeline:
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: float, c: float) -> float:
        return 2 * x + c

    @pipefunc(output_name="sum_")
    def take_sum(y: Array[float], d: float, e: float) -> float:
        return sum(y) / d + e

    return Pipeline([double_it, take_sum])


def test_adaptive_wrapper_1d(tmp_path: Path, pipeline: Pipeline) -> None:
    run_folder_template = f"{tmp_path}/run_folder_{{}}"
    learner1d = to_adaptive_learner(
        pipeline,
        inputs={"x": [0, 1, 2, 3], "d": 1, "e": 0},
        adaptive_dimensions={"c": (0, 100)},
        adaptive_output="sum_",
        run_folder_template=run_folder_template,
        map_kwargs={"parallel": False, "storage": "dict"},
    )
    assert isinstance(learner1d, adaptive.Learner1D)
    npoints_goal = 5
    adaptive.runner.simple(learner1d, npoints_goal=npoints_goal)

    assert learner1d.to_numpy().shape == (npoints_goal, 2)
    assert len(list(tmp_path.glob("*"))) == npoints_goal


def test_adaptive_wrapper_2d(tmp_path: Path, pipeline: Pipeline) -> None:
    run_folder_template = f"{tmp_path}/run_folder_{{}}"
    learner2d = to_adaptive_learner(
        pipeline,
        inputs={"x": [0, 1, 2, 3], "e": 0},
        adaptive_dimensions={"c": (0, 100), "d": (-1, 1)},
        adaptive_output="sum_",
        run_folder_template=run_folder_template,
        map_kwargs={"parallel": False, "storage": "dict"},
    )
    assert isinstance(learner2d, adaptive.Learner2D)
    npoints_goal = 5
    adaptive.runner.simple(learner2d, npoints_goal=npoints_goal)

    assert learner2d.to_numpy().shape == (npoints_goal, 3)
    assert len(list(tmp_path.glob("*"))) == npoints_goal


def test_adaptive_wrapper_3d(tmp_path: Path, pipeline: Pipeline) -> None:
    run_folder_template = f"{tmp_path}/run_folder_{{}}"
    learner3d = to_adaptive_learner(
        pipeline,
        inputs={"x": [0, 1, 2, 3]},
        adaptive_dimensions={"c": (0, 100), "d": (-1, 1), "e": (-1, 1)},
        adaptive_output="sum_",
        run_folder_template=run_folder_template,
        map_kwargs={"parallel": False, "storage": "dict"},
    )
    assert isinstance(learner3d, adaptive.LearnerND)
    npoints_goal = 5
    adaptive.runner.simple(learner3d, npoints_goal=npoints_goal)

    assert learner3d.to_numpy().shape == (npoints_goal, 4)
    assert len(list(tmp_path.glob("*"))) == npoints_goal


def test_adaptive_wrapper_invalid(tmp_path: Path, pipeline: Pipeline) -> None:
    run_folder_template = f"{tmp_path}/run_folder_{{}}"
    with pytest.raises(ValueError, match="`adaptive_dimensions` must be a non-empty dict"):
        to_adaptive_learner(
            pipeline,
            inputs={"x": [0, 1, 2, 3]},
            adaptive_dimensions={},
            adaptive_output="sum_",
            run_folder_template=run_folder_template,
        )
    with pytest.raises(ValueError, match="cannot be in inputs"):
        to_adaptive_learner(
            pipeline,
            inputs={"x": [0, 1, 2, 3], "c": 0, "e": 0},
            adaptive_dimensions={"c": (-1, 1)},
            adaptive_output="sum_",
            run_folder_template=run_folder_template,
        )
    with pytest.raises(ValueError, match="Adaptive dimensions `{'x'}` cannot be in `MapSpec`s"):
        to_adaptive_learner(
            pipeline,
            inputs={"e": 0},
            adaptive_dimensions={"c": (-1, 1), "x": (0, 100)},
            adaptive_output="sum_",
            run_folder_template=run_folder_template,
        )


def test_adaptive_wrapper_with_heterogeneous_storage(tmp_path: Path, pipeline: Pipeline) -> None:
    run_folder_template = f"{tmp_path}/run_folder_{{}}"
    storage = {
        "": "dict",
        "sum_": "file_array",
    }
    learner = to_adaptive_learner(
        pipeline,
        inputs={"x": [0, 1, 2, 3]},
        adaptive_dimensions={"c": (0, 100), "d": (-1, 1), "e": (-1, 1)},
        adaptive_output="sum_",
        run_folder_template=run_folder_template,
        map_kwargs={"parallel": False, "storage": storage},
    )
    assert isinstance(learner, adaptive.LearnerND)
    npoints_goal = 5
    adaptive.runner.simple(learner, npoints_goal=npoints_goal)

    assert learner.to_numpy().shape == (npoints_goal, 4)
    assert len(list(tmp_path.glob("*"))) == npoints_goal


def test_adaptive_run_dynamic_internal_shape():
    @pipefunc(output_name="n")
    def f() -> int:
        return 10

    @pipefunc(output_name="y", internal_shape=("?",))
    def g(n: int, a: float) -> list[float]:
        return [a * i for i in range(n)]

    @pipefunc(output_name="z", mapspec="y[i] -> z[i]")
    def h(y: float) -> float:
        return y**2

    @pipefunc(output_name="sum")
    def i(z: Array[float]) -> float:
        return sum(z)

    pipeline = Pipeline([f, g, h, i])

    learner = to_adaptive_learner(
        pipeline,
        inputs={},
        adaptive_dimensions={"a": (0.0, 1.0)},
        adaptive_output="sum",
        map_kwargs={"parallel": False, "storage": "dict"},
    )

    adaptive.runner.simple(learner, npoints_goal=10)

    assert len(learner.data) == 10
    assert learner.to_numpy().shape == (10, 2)
