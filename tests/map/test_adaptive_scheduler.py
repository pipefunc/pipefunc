from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.map.adaptive import create_learners
from pipefunc.map.adaptive_scheduler import AdaptiveSchedulerDetails, slurm_run_setup
from pipefunc.resources import Resources

if TYPE_CHECKING:
    from pathlib import Path


def test_slurm_run_setup(tmp_path: Path) -> None:
    @pipefunc(output_name="x", mapspec="a[i] -> x[i]")
    def f1(a: int) -> int:
        return a

    @pipefunc(output_name="y")
    def f2(x: int) -> int:
        return x

    pipeline = Pipeline([f1, f2])

    inputs = {"a": list(range(10))}
    learners_dict = create_learners(pipeline, inputs, tmp_path, split_independent_axes=True)

    info = learners_dict.to_slurm_run(
        tmp_path,
        Resources(
            num_cpus_per_node=2,
            num_nodes=1,
            partition="partition-1",
            num_gpus=1,
            wall_time="1:00:00",
        ),
        returns="namedtuple",
    )
    assert isinstance(info, AdaptiveSchedulerDetails)
    assert len(info.learners) == 2
    assert len(info.fnames) == 2
    assert info.dependencies == {0: [], 1: [0]}
    assert info.nodes == (1, 1)
    assert info.cores_per_node == (2, 2)
    assert info.extra_scheduler == (
        ["--gres=gpu:1", "--time=1:00:00"],
        ["--gres=gpu:1", "--time=1:00:00"],
    )
    assert info.partition == ("partition-1", "partition-1")
    assert list(info.kwargs().keys()) == [
        "learners",
        "fnames",
        "dependencies",
        "nodes",
        "cores_per_node",
        "extra_scheduler",
        "partition",
        "exclusive",
    ]


def test_slurm_run_setup_with_resources(tmp_path: Path) -> None:
    @pipefunc(output_name="x", resources=Resources(memory="8GB"), mapspec="a[i] -> x[i]")
    def f1(a: int) -> int:
        return a

    @pipefunc(
        output_name="y",
        resources=Resources(num_cpus=2, memory="4GB", extra_args={"qos": "high"}),
    )
    def f2(x: int) -> int:
        return x

    pipeline = Pipeline([f1, f2])

    inputs = {"a": list(range(4))}
    learners_dict = create_learners(pipeline, inputs, tmp_path, split_independent_axes=True)

    with pytest.raises(ValueError, match="At least one `PipeFunc` provides `num_cpus`"):
        learners_dict.to_slurm_run(tmp_path, None, returns="namedtuple")

    # Test including defaults
    info = learners_dict.to_slurm_run(tmp_path, {"num_cpus": 8}, returns="namedtuple")
    assert isinstance(info, AdaptiveSchedulerDetails)
    assert len(info.learners) == 2
    assert len(info.fnames) == 2
    assert len(info.learners[0].sequence) == 4
    assert len(info.learners[1].sequence) == 1
    assert info.dependencies == {0: [], 1: [0]}
    assert info.nodes is None
    assert info.extra_scheduler == (["--mem=8GB"], ["--mem=4GB", "--qos=high"])
    assert info.partition is None
    assert info.cores_per_node == (8, 2)

    # Test ignoring resources
    info = learners_dict.to_slurm_run(tmp_path, None, ignore_resources=True, returns="namedtuple")
    assert isinstance(info, AdaptiveSchedulerDetails)
    assert len(info.learners) == 2
    assert info.extra_scheduler is None
    assert info.cores_per_node is None

    # Test ignoring resources with default (now using "kwargs")
    info = learners_dict.to_slurm_run(
        tmp_path,
        {"num_cpus": 8},
        ignore_resources=True,
        returns="kwargs",
    )
    assert isinstance(info, dict)
    assert len(info["learners"]) == 2
    assert "extra_scheduler" not in info
    assert info["cores_per_node"] == (8, 8)


def test_missing_resources(tmp_path: Path) -> None:
    @pipefunc(output_name="x", mapspec="a[i] -> x[i]")
    def f1(a: int) -> int:
        return a

    @pipefunc(output_name="y")
    def f2(x: int) -> int:
        return x

    pipeline = Pipeline([f1, f2])

    inputs = {"a": list(range(4))}
    learners_dict = create_learners(pipeline, inputs, tmp_path, split_independent_axes=True)
    with pytest.raises(
        ValueError,
        match="Either all `PipeFunc`s must have resources or `default_resources` must be provided.",
    ):
        learners_dict.to_slurm_run(tmp_path)


def test_slurm_run_setup_with_partial_default_resources(tmp_path: Path) -> None:
    @pipefunc(output_name="x", resources=Resources(num_cpus=2), mapspec="a[i] -> x[i]")
    def f1(a: int) -> int:
        return a

    @pipefunc(output_name="y")
    def f2(x: int) -> int:
        return x

    pipeline = Pipeline([f1, f2])

    inputs = {"a": list(range(10))}
    learners_dict = create_learners(pipeline, inputs, tmp_path, split_independent_axes=True)

    default_resources = Resources(num_cpus=4)
    info = slurm_run_setup(learners_dict, tmp_path, default_resources)
    assert isinstance(info, AdaptiveSchedulerDetails)
    assert len(info.learners) == 2
    assert len(info.fnames) == 2
    assert info.dependencies == {0: [], 1: [0]}
    assert info.nodes is None
    assert info.cores_per_node == (2, 4)
    assert info.extra_scheduler is None
    assert info.partition is None
    kwargs = info.kwargs()
    assert "partition" not in kwargs
    assert "cores_per_node" in kwargs


def test_slurm_run_setup_missing_resource(tmp_path: Path) -> None:
    @pipefunc(output_name="x", resources=Resources(partition="partition-1"), mapspec="a[i] -> x[i]")
    def f1(a: int) -> int:
        return a

    @pipefunc(output_name="y")
    def f2(x: int) -> int:
        return x

    pipeline = Pipeline([f1, f2])

    inputs = {"a": list(range(10))}
    learners_dict = create_learners(pipeline, inputs, tmp_path, split_independent_axes=True)

    with pytest.raises(
        ValueError,
        match="At least one `PipeFunc` provides `partition`.",
    ):
        slurm_run_setup(learners_dict, tmp_path, Resources(num_nodes=1))
