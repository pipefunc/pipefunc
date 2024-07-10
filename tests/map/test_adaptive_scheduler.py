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
        Resources(
            cpus_per_node=2,
            nodes=1,
            partition="partition-1",
            gpus=1,
            time="1:00:00",
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
    ]


def test_slurm_run_setup_with_resources(tmp_path: Path) -> None:
    @pipefunc(output_name="x", resources=Resources(memory="8GB"), mapspec="a[i] -> x[i]")
    def f1(a: int) -> int:
        return a

    @pipefunc(
        output_name="y",
        resources=Resources(cpus=2, memory="4GB", extra_args={"qos": "high"}),
    )
    def f2(x: int) -> int:
        return x

    pipeline = Pipeline([f1, f2])

    inputs = {"a": list(range(4))}
    learners_dict = create_learners(pipeline, inputs, tmp_path, split_independent_axes=True)

    with pytest.raises(ValueError, match="At least one `PipeFunc` provides `cpus`"):
        learners_dict.to_slurm_run(None, returns="namedtuple")

    # Test including defaults
    info = learners_dict.to_slurm_run({"cpus": 8}, returns="namedtuple")
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
    info = learners_dict.to_slurm_run({"cpus": 10}, ignore_resources=True, returns="namedtuple")
    assert isinstance(info, AdaptiveSchedulerDetails)
    assert len(info.learners) == 2
    assert info.extra_scheduler is None
    assert info.cores_per_node == (10, 10)

    # Test ignoring resources with default (now using "kwargs")
    info = learners_dict.to_slurm_run(
        {"cpus": 8},
        ignore_resources=True,
        returns="kwargs",
    )
    assert isinstance(info, dict)
    assert len(info["learners"]) == 2
    assert "extra_scheduler" not in info
    assert info["cores_per_node"] == (8, 8)

    with pytest.raises(ValueError, match="Invalid value for `returns`: not_exists"):
        learners_dict.to_slurm_run({"cpus": 8}, returns="not_exists")  # type: ignore[arg-type]


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
        learners_dict.to_slurm_run()


def test_default_resources_from_pipeline_and_to_slurm_run(tmp_path: Path) -> None:
    @pipefunc(output_name="x", mapspec="a[i] -> x[i]")
    def f1(a: int) -> int:
        return a

    @pipefunc(output_name="y")
    def f2(x: int) -> int:
        return x

    pipeline1 = Pipeline([f1], default_resources=Resources(cpus=2))
    assert isinstance(pipeline1["x"].resources, Resources)
    assert pipeline1["x"].resources.cpus == 2
    pipeline2 = Pipeline([f2])
    pipeline = pipeline1 | pipeline2
    inputs = {"a": list(range(4))}
    learners_dict = create_learners(pipeline, inputs, tmp_path, split_independent_axes=True)
    kw = learners_dict.to_slurm_run(default_resources=Resources(cpus=4))
    assert isinstance(kw, dict)
    assert kw["cores_per_node"] == (2, 4)


def test_slurm_run_setup_with_partial_default_resources(tmp_path: Path) -> None:
    @pipefunc(output_name="x", resources=Resources(cpus=2), mapspec="a[i] -> x[i]")
    def f1(a: int) -> int:
        return a

    @pipefunc(output_name="y")
    def f2(x: int) -> int:
        return x

    pipeline = Pipeline([f1, f2])

    inputs = {"a": list(range(10))}
    learners_dict = create_learners(pipeline, inputs, tmp_path, split_independent_axes=True)

    default_resources = Resources(cpus=4)
    info = slurm_run_setup(learners_dict, default_resources)
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
        slurm_run_setup(learners_dict, Resources(nodes=1))


def test_slurm_run_delayed_resources(tmp_path: Path) -> None:
    @pipefunc(
        output_name="x",
        resources=lambda kw: Resources(cpus=kw["a"]),
        resources_variable="resources",
    )
    def f1(a: int, resources: Resources):
        return a, resources

    pipeline = Pipeline([f1])
    inputs = {"a": 1}
    learners_dict = create_learners(
        pipeline,
        inputs,
        tmp_path,
        split_independent_axes=True,
        return_output=True,
    )
    info = slurm_run_setup(learners_dict, Resources(cpus=2))
    assert isinstance(info, AdaptiveSchedulerDetails)
    assert len(info.learners) == 1
    learners_dict.simple_run()
    learner_pipefunc = learners_dict[None][0][0]
    assert learner_pipefunc.learner.data == {0: (1, Resources(cpus=1))}
    assert info.cores_per_node is not None
    assert len(info.cores_per_node) == 1
    assert callable(info.cores_per_node[0])
    assert info.cores_per_node[0]() == 1

    kw = info.kwargs()
    assert len(kw["cores_per_node"]) == 1
    assert kw["cores_per_node"][0]() == 1
    assert kw.keys() == {
        "learners",
        "fnames",
        "dependencies",
        "nodes",
        "cores_per_node",
        "extra_scheduler",
        "partition",
    }
    f, *rest = kw["nodes"]
    assert len(rest) == 0
    assert f() is None
    f, *rest = kw["extra_scheduler"]
    assert len(rest) == 0
    assert f() == []
    f, *rest = kw["partition"]
    assert len(rest) == 0
    assert f() is None


def test_slurm_run_delayed_resources_with_mapspec(tmp_path: Path) -> None:
    @pipefunc(
        output_name="x",
        resources=lambda kw: Resources(cpus=kw["a"]),
        mapspec="a[i] -> x[i]",
    )
    def f1(a: int) -> int:
        return a

    @pipefunc(output_name="y")
    def f2(x: int) -> int:
        return x

    pipeline = Pipeline([f1, f2])
    inputs = {"a": list(range(10))}
    learners_dict = create_learners(pipeline, inputs, tmp_path, split_independent_axes=True)
    info = slurm_run_setup(learners_dict, Resources(cpus=2))
    assert isinstance(info, AdaptiveSchedulerDetails)
    assert len(info.learners) == 2
