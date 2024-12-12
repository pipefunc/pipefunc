from __future__ import annotations

from typing import TYPE_CHECKING

import adaptive
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.map.adaptive import create_learners
from pipefunc.map.adaptive_scheduler import AdaptiveSchedulerDetails, _or, slurm_run_setup
from pipefunc.resources import Resources
from pipefunc.typing import Array  # noqa: TC001

if TYPE_CHECKING:
    from pathlib import Path


def run(info: AdaptiveSchedulerDetails) -> None:
    for lrn in info.learners:
        adaptive.runner.simple(lrn)


def test_slurm_run_setup(tmp_path: Path) -> None:
    @pipefunc(output_name="x", mapspec="a[i] -> x[i]")
    def f1(a: int) -> int:
        return a

    @pipefunc(output_name="y")
    def f2(x: Array[int]) -> Array[int]:
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
    assert info.kwargs().keys() == {
        "learners",
        "fnames",
        "dependencies",
        "nodes",
        "cores_per_node",
        "extra_scheduler",
        "partition",
        "executor_type",
    }

    run(info)

    with pytest.raises(
        ValueError,
        match="Cannot pass `slurm_run_kwargs` when `returns='namedtuple'`.",
    ):
        learners_dict.to_slurm_run(
            default_resources=Resources(cpus_per_node=2, nodes=1),
            returns="namedtuple",
            kwargs={"will": "fail"},
        )


def test_slurm_run_setup_with_resources(tmp_path: Path) -> None:
    @pipefunc(output_name="x", resources=Resources(memory="8GB"), mapspec="a[i] -> x[i]")
    def f1(a: int) -> int:
        return a

    @pipefunc(
        output_name="y",
        resources=Resources(cpus=2, memory="4GB", extra_args={"qos": "high"}),
    )
    def f2(x: Array[int]) -> int:
        return sum(x)  # type: ignore[call-overload]

    pipeline = Pipeline([f1, f2])

    inputs = {"a": list(range(4))}
    learners_dict = create_learners(pipeline, inputs, tmp_path, split_independent_axes=True)

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
    run(info)

    # Test ignoring resources with default (now using "kwargs")
    info = learners_dict.to_slurm_run(
        {"cpus": 8},
        ignore_resources=True,
        returns="kwargs",
        log_interval=10,
        save_interval=20,
    )
    assert isinstance(info, dict)
    assert len(info["learners"]) == 2
    assert "extra_scheduler" not in info
    assert info["cores_per_node"] == (8, 8)
    assert info["log_interval"] == 10
    assert info["save_interval"] == 20

    with pytest.raises(ValueError, match="Invalid value for `returns`: not_exists"):
        learners_dict.to_slurm_run({"cpus": 8}, returns="not_exists")  # type: ignore[arg-type]


def test_missing_resources(tmp_path: Path) -> None:
    @pipefunc(output_name="x", mapspec="a[i] -> x[i]")
    def f1(a: int) -> int:
        return a

    @pipefunc(output_name="y")
    def f2(x: Array[int]) -> Array[int]:
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
    def f2(x: Array[int]) -> Array[int]:
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
    def f2(x: Array[int]) -> Array[int]:
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
    run(info)
    kwargs = info.kwargs()
    assert "partition" not in kwargs
    assert "cores_per_node" in kwargs


def test_slurm_run_delayed_resources(tmp_path: Path) -> None:
    @pipefunc(
        output_name="x",
        resources=lambda kw: Resources(cpus=kw["a"]),
        resources_variable="resources1",
    )
    def f1(a: int, resources1: Resources):
        return a, resources1

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
    run(info)

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
        "executor_type",
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
        resources=lambda kw: Resources(cpus=len(kw["a"])),
        resources_variable="resources",
        mapspec="a[i] -> x[i]",
        resources_scope="map",
    )
    def f1(a: int, resources: Resources) -> int:
        assert resources.cpus == 10
        return a

    @pipefunc(output_name="y")
    def f2(x: Array[int]) -> Array[int]:
        return x

    pipeline = Pipeline([f1, f2])
    inputs = {"a": list(range(10))}
    learners_dict = create_learners(pipeline, inputs, tmp_path, split_independent_axes=True)
    info = slurm_run_setup(learners_dict, Resources(cpus=2))
    assert isinstance(info, AdaptiveSchedulerDetails)
    assert len(info.learners) == 2

    # Run one learner iteration
    learner = info.learners[0]
    learner.function((0, learner.sequence[0]))
    run(info)

    assert isinstance(info.cores_per_node, tuple)
    assert len(info.cores_per_node) == 2
    cpn1, cpn2 = info.cores_per_node
    assert callable(cpn1)
    assert not callable(cpn2)
    assert cpn1() == 10
    assert cpn2 == 2

    assert isinstance(info.partition, tuple)
    assert len(info.partition) == 2
    p1, p2 = info.partition
    assert callable(p1)
    assert p2 is None
    assert p1() is None

    assert isinstance(info.extra_scheduler, tuple)
    assert len(info.extra_scheduler) == 2
    e1, e2 = info.extra_scheduler
    assert callable(e1)
    assert e2 == []
    assert e1() == []

    assert isinstance(info.nodes, tuple)
    assert len(info.nodes) == 2
    n1, n2 = info.nodes
    assert callable(n1)
    assert n2 is None
    assert n1() is None

    assert info.dependencies == {0: [], 1: [0]}


def test_slurm_run_delayed_resources_with_mapspec_scope(tmp_path: Path) -> None:
    @pipefunc(
        output_name="x",
        resources=lambda kw: Resources(cpus=kw["a"]),
        resources_variable="resources",
        resources_scope="element",
        mapspec="a[i] -> x[i]",
    )
    def f1(a: int, resources: Resources) -> int:
        assert isinstance(resources.cpus, int)
        return a

    @pipefunc(output_name="y")
    def f2(x: Array[int]) -> Array[int]:
        return x

    pipeline = Pipeline([f1, f2])
    inputs = {"a": list(range(1, 10))}
    learners_dict = create_learners(
        pipeline,
        inputs,
        tmp_path,
        split_independent_axes=True,
        return_output=True,
    )
    info = slurm_run_setup(learners_dict, Resources(cpus=2))
    assert isinstance(info, AdaptiveSchedulerDetails)
    assert len(info.learners) == 10
    assert info.cores_per_node[0]() == 1  # type: ignore[operator,misc,index]
    learner = info.learners[0]
    y = learner.function((0, learner.sequence[0]))
    assert y == 1
    run(info)


def test_cores_per_node_vs_cores(tmp_path: Path) -> None:
    @pipefunc(output_name="x", resources=Resources(cpus=1))
    def f1(a: int) -> int:
        return a

    @pipefunc(output_name="y", resources=Resources(cpus_per_node=2, nodes=1))
    def f2(x: int) -> int:
        return x

    pipeline = Pipeline([f1, f2])
    inputs = {"a": 1}
    learners_dict = create_learners(pipeline, inputs, tmp_path, split_independent_axes=True)
    info = slurm_run_setup(learners_dict)
    assert isinstance(info, AdaptiveSchedulerDetails)
    assert len(info.learners) == 2
    assert info.cores_per_node == (1, 2)
    assert info.nodes == (None, 1)
    run(info)


def test_cores_only(tmp_path: Path) -> None:
    @pipefunc(output_name="x", resources=Resources(cpus=1))
    def f1(a: int) -> int:
        return a

    pipeline = Pipeline([f1])
    inputs = {"a": 1}
    learners_dict = create_learners(pipeline, inputs, tmp_path, split_independent_axes=True)
    info = slurm_run_setup(learners_dict)
    assert isinstance(info, AdaptiveSchedulerDetails)
    assert len(info.learners) == 1
    assert info.cores_per_node == (1,)
    assert info.nodes is None
    run(info)


def test_or() -> None:
    assert _or(None, 1) == 1
    assert _or(1, None) == 1

    def cpus():
        return 2

    def none():
        return None

    assert _or(none, 1)() == 1  # type: ignore[operator,misc]
    assert _or(1, none)() == 1  # type: ignore[operator,misc]
    assert _or(cpus, none)() == 2  # type: ignore[operator,misc]
    assert _or(none, cpus)() == 2  # type: ignore[operator,misc]
    assert _or(1, 1) == 1  # type: ignore[operator,misc]


def test_parallelization_mode(tmp_path: Path) -> None:
    @pipefunc(
        output_name="x",
        resources=lambda _: Resources(cpus=1, parallelization_mode="internal"),
    )
    def f1(a: int) -> int:
        return a

    pipeline = Pipeline([f1])
    inputs = {"a": 1}
    learners_dict = create_learners(pipeline, inputs, tmp_path, split_independent_axes=True)
    info = slurm_run_setup(learners_dict)
    assert isinstance(info, AdaptiveSchedulerDetails)
    assert info.executor_type is not None
    assert len(info.executor_type) == 1
    assert info.executor_type[0]() == "sequential"
    run(info)


def test_slurm_run_split_all(tmp_path: Path) -> None:
    @pipefunc(
        output_name="x",
        mapspec="a[i] -> x[i]",
        resources=lambda kw: Resources(cpus_per_node=kw["a"], nodes=2),
        resources_scope="element",
        resources_variable="resources",
    )
    def f1(a: int, resources: Resources) -> int:
        assert resources.cpus_per_node in (1, 2, 3)
        return a

    @pipefunc(output_name="y", resources=lambda kw: Resources(cpus_per_node=kw["x"][0], nodes=1))
    def f2(x: Array[int]) -> Array[int]:
        return x

    pipeline = Pipeline([f1, f2])

    inputs = {"a": list(range(1, 4))}
    learners_dict = create_learners(pipeline, inputs, tmp_path, split_independent_axes=True)

    info = learners_dict.to_slurm_run(returns="namedtuple")
    assert isinstance(info, AdaptiveSchedulerDetails)
    assert len(info.learners) == 4
    assert len(info.fnames) == 4
    assert info.dependencies == {0: [], 1: [], 2: [], 3: [0, 1, 2]}
    assert info.nodes[0]() == 2  # type: ignore[operator,misc,index]
    assert info.cores_per_node[0]() == 1  # type: ignore[operator,misc,index]
    assert info.cores_per_node[1]() == 2  # type: ignore[operator,misc,index]

    run(info)


def test_slurm_run_delayed_resources_with_mapspec_scope_single_element(tmp_path: Path) -> None:
    @pipefunc(
        output_name="x",
        resources=lambda kw: Resources(cpus=kw["a"]),
        resources_variable="resources",
        resources_scope="element",
        mapspec="a[i] -> x[i]",
    )
    def f1(a: int, resources: Resources) -> int:
        assert isinstance(resources.cpus, int)
        return a

    @pipefunc(output_name="y")
    def f2(x: Array[int]) -> Array[int]:
        return x

    pipeline = Pipeline([f1, f2])
    inputs = {"a": [4]}
    learners_dict = create_learners(
        pipeline,
        inputs,
        tmp_path,
        split_independent_axes=True,
        return_output=True,
    )
    info = slurm_run_setup(learners_dict, Resources(cpus=2))
    assert isinstance(info, AdaptiveSchedulerDetails)
    assert len(info.learners) == 2
    assert info.cores_per_node[0]() == 4  # type: ignore[operator,misc,index]
    run(info)


def test_independent_axes_2(tmp_path: Path) -> None:
    @pipefunc(output_name="y", mapspec="... -> y[i]", internal_shape=(3,))
    def f(x):
        return x

    @pipefunc(output_name="r", mapspec="z[i], y[i] -> r[i]")
    def g(y, z):
        return y + z

    pipeline = Pipeline([f, g])
    inputs = {"x": [1, 2, 3], "z": [3, 4, 5]}
    r = pipeline.map(inputs, parallel=False)
    assert r["y"].output == [1, 2, 3]
    assert r["r"].output.tolist() == [4, 6, 8]
    learners_dict = create_learners(pipeline, inputs, tmp_path, split_independent_axes=True)
    info = slurm_run_setup(learners_dict, default_resources=Resources(cpus=1))
    assert isinstance(info, AdaptiveSchedulerDetails)
    assert len(info.learners) == 2
    assert len(info.fnames) == 2
    run(info)


def test_dynamic_shapes(tmp_path: Path) -> None:
    @pipefunc(output_name="y", mapspec="... -> y[i]", internal_shape=("?",))
    def f(n):
        return list(range(n))

    pipeline = Pipeline([f])
    with pytest.raises(
        ValueError,
        match="Dynamic `internal_shapes` not supported in `create_learners`.",
    ):
        create_learners(pipeline, {"n": 10}, tmp_path, split_independent_axes=True)
