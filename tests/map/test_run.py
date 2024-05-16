from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from pipefunc import PipeFunc, Pipeline, pipefunc
from pipefunc._utils import prod
from pipefunc.map._run import load_outputs, map_shapes, run

if TYPE_CHECKING:
    from pathlib import Path


def test_simple(tmp_path: Path) -> None:
    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        assert isinstance(x, int)
        return 2 * x

    @pipefunc(output_name="sum")
    def take_sum(y: np.ndarray[Any, np.dtype[np.int_]]) -> int:
        assert isinstance(y, np.ndarray)
        return sum(y)

    pipeline = Pipeline(
        [
            (double_it, "x[i] -> y[i]"),
            take_sum,
        ],
    )

    inputs = {"x": [0, 1, 2, 3]}
    results = run(pipeline, inputs, run_folder=tmp_path, parallel=False)
    assert results[-1].output == 12
    assert results[-1].output_name == "sum"
    assert load_outputs("sum", run_folder=tmp_path) == 12
    assert map_shapes(pipeline, inputs) == {"x": (4,), "y": (4,)}
    results2 = pipeline.map(inputs, run_folder=tmp_path, parallel=False)
    assert results2[-1].output == 12


def test_simple_2_dim_array(tmp_path: Path) -> None:
    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        assert isinstance(x, np.int_)
        return 2 * x

    @pipefunc(output_name="sum")
    def take_sum(y: np.ndarray) -> int:
        assert isinstance(y, np.ndarray)
        return np.sum(y, axis=0)

    pipeline = Pipeline(
        [
            (double_it, "x[i, j] -> y[i, j]"),
            take_sum,
        ],
    )

    inputs = {"x": np.arange(12).reshape(3, 4)}
    results = run(pipeline, inputs, run_folder=tmp_path, parallel=False)
    assert results[-1].output_name == "sum"
    assert results[-1].output.tolist() == [24, 30, 36, 42]
    assert load_outputs("sum", run_folder=tmp_path).tolist() == [24, 30, 36, 42]
    assert map_shapes(pipeline, inputs) == {"x": (3, 4), "y": (3, 4)}
    results2 = pipeline.map(inputs, run_folder=tmp_path, parallel=False)
    assert results2[-1].output.tolist() == [24, 30, 36, 42]


def test_simple_2_dim_array_to_1_dim(tmp_path: Path) -> None:
    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        assert isinstance(x, np.int_)
        return 2 * x

    @pipefunc(output_name="sum")
    def take_sum(y: np.ndarray) -> int:
        assert isinstance(y, np.ndarray)
        return sum(y)

    pipeline = Pipeline(
        [
            (double_it, "x[i, j] -> y[i, j]"),
            (take_sum, "y[i, :] -> sum[i]"),
        ],
    )

    inputs = {"x": np.arange(12).reshape(3, 4)}
    results = run(pipeline, inputs, run_folder=tmp_path, parallel=False)
    assert results[-1].output_name == "sum"
    assert results[-1].output.tolist() == [12, 44, 76]
    assert load_outputs("sum", run_folder=tmp_path).tolist() == [12, 44, 76]
    assert map_shapes(pipeline, inputs) == {
        "x": (3, 4),
        "y": (3, 4),
        "sum": (3,),
    }


def test_simple_2_dim_array_to_1_dim_to_0_dim(tmp_path: Path) -> None:
    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        assert isinstance(x, np.int_)
        return 2 * x

    @pipefunc(output_name="sum")
    def take_sum(y: np.ndarray) -> int:
        assert isinstance(y, np.ndarray)
        return sum(y)

    @pipefunc(output_name="prod")
    def take_prod(y: np.ndarray) -> int:
        assert isinstance(y, np.ndarray)
        return np.prod(y)

    pipeline = Pipeline(
        [
            (double_it, "x[i, j] -> y[i, j]"),
            (take_sum, "y[i, :] -> sum[i]"),
            take_prod,
        ],
    )

    inputs = {"x": np.arange(1, 13).reshape(3, 4)}
    results = run(pipeline, inputs, run_folder=tmp_path, parallel=False)
    assert results[-1].output_name == "prod"
    assert isinstance(results[-1].output, np.int_)
    assert results[-1].output == 1961990553600
    assert load_outputs("prod", run_folder=tmp_path) == 1961990553600
    assert map_shapes(pipeline, inputs) == {
        "x": (3, 4),
        "y": (3, 4),
        "sum": (3,),
    }


def run_outer_product(pipeline: Pipeline, tmp_path: Path) -> None:
    """Run the outer product test for the given pipeline."""
    # Used in the next three tests where we use alternative ways of defining the same pipeline
    inputs = {"x": [1, 2, 3], "y": [1, 2, 3]}
    results = run(pipeline, inputs, run_folder=tmp_path, parallel=False)
    assert results[0].output_name == "z"
    expected = [[2, 3, 4], [3, 4, 5], [4, 5, 6]]
    assert results[0].output.tolist() == expected
    assert load_outputs("z", run_folder=tmp_path).tolist() == expected
    assert results[1].output_name == "sum"
    assert results[1].output == 36
    assert load_outputs("sum", run_folder=tmp_path) == 36
    assert len(results) == 2
    assert map_shapes(pipeline, inputs) == {"y": (3,), "x": (3,), "z": (3, 3)}


def test_outer_product(tmp_path: Path) -> None:
    @pipefunc(output_name="z")
    def add(x: int, y: int) -> int:
        assert isinstance(x, int)
        assert isinstance(y, int)
        return x + y

    @pipefunc(output_name="sum")
    def total_sum(z: np.ndarray) -> int:
        assert isinstance(z, np.ndarray)
        return np.sum(z)

    pipeline = Pipeline(
        [
            (add, "x[i], y[j] -> z[i, j]"),
            total_sum,
        ],
    )
    run_outer_product(pipeline, tmp_path)


def test_outer_product_decorator(tmp_path: Path) -> None:
    @pipefunc(output_name="z", mapspec="x[i], y[j] -> z[i, j]")
    def add(x: int, y: int) -> int:
        assert isinstance(x, int)
        assert isinstance(y, int)
        return x + y

    @pipefunc(output_name="sum")
    def total_sum(z: np.ndarray) -> int:
        assert isinstance(z, np.ndarray)
        return np.sum(z)

    pipeline = Pipeline([add, total_sum])
    run_outer_product(pipeline, tmp_path)


def test_outer_product_functions(tmp_path: Path) -> None:
    def add(x: int, y: int) -> int:
        assert isinstance(x, int)
        assert isinstance(y, int)
        return x + y

    def total_sum(z: np.ndarray) -> int:
        assert isinstance(z, np.ndarray)
        return np.sum(z)

    pipeline = Pipeline(
        [
            PipeFunc(add, "z", mapspec="x[i], y[j] -> z[i, j]"),
            PipeFunc(total_sum, "sum"),
        ],
    )
    run_outer_product(pipeline, tmp_path)


def test_simple_from_step(tmp_path: Path) -> None:
    @pipefunc(output_name="x")
    def generate_ints(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        assert isinstance(x, int)
        return 2 * x

    @pipefunc(output_name="sum")
    def take_sum(y: list[int]) -> int:
        return sum(y)

    pipeline = Pipeline(
        [
            generate_ints,
            (double_it, "x[i] -> y[i]"),
            take_sum,
        ],
    )
    inputs = {"n": 4}
    results = run(
        pipeline,
        inputs,
        run_folder=tmp_path,
        manual_shapes={"x": (4,)},
        parallel=False,
    )
    assert results[-1].output == 12
    assert results[-1].output_name == "sum"
    assert load_outputs("sum", run_folder=tmp_path) == 12
    with pytest.raises(ValueError, match="is used in map but"):
        map_shapes(pipeline, inputs)

    assert map_shapes(pipeline, inputs, {"x": (4,)}) == {"y": (4,)}

    with pytest.raises(
        RuntimeError,
        match="Use `Pipeline.map` instead",
    ):
        pipeline("sum", n=4)
    assert pipeline("x", n=4) == list(range(4))


@pytest.mark.parametrize("output_picker", [None, dict.__getitem__])
def test_simple_multi_output(tmp_path: Path, output_picker) -> None:
    @pipefunc(output_name=("single", "double"), output_picker=output_picker)
    def double_it(x: int) -> tuple[int, int] | dict[str, int]:
        assert isinstance(x, int)
        return (x, 2 * x) if output_picker is None else {"single": x, "double": 2 * x}

    @pipefunc(output_name="sum")
    def take_sum(single: np.ndarray[Any, np.dtype[np.int_]]) -> int:
        return sum(single)

    pipeline = Pipeline(
        [
            (double_it, "x[i] -> single[i], double[i]"),
            take_sum,
        ],
    )

    inputs = {"x": [0, 1, 2, 3]}
    results = run(pipeline, inputs, run_folder=tmp_path, parallel=True)
    assert results[-1].output == 6
    assert results[-1].output_name == "sum"
    assert load_outputs("sum", run_folder=tmp_path) == 6
    assert map_shapes(pipeline, inputs) == {
        "x": (4,),
        ("single", "double"): (4,),
        "single": (4,),
        "double": (4,),
    }


def test_simple_from_step_nd(tmp_path: Path) -> None:
    @pipefunc(output_name="array")
    def generate_array(shape: tuple[int, ...]) -> np.ndarray[Any, np.dtype[np.int_]]:
        return np.arange(1, prod(shape) + 1).reshape(shape)

    @pipefunc(output_name="vector")
    def double_it(array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        assert isinstance(array, np.ndarray)
        assert array.shape == shape[1:]
        return array.sum(axis=0).sum(axis=0)

    @pipefunc(output_name="sum")
    def norm(vector: np.ndarray) -> np.float64:
        return np.linalg.norm(vector)

    pipeline = Pipeline(
        [
            generate_array,
            (double_it, "array[i, :, :] -> vector[i]"),
            norm,
        ],
    )
    inputs = {"shape": (1, 2, 3)}
    manual_shapes: dict[str, int | tuple[int, ...]] = {"array": (1, 2, 3)}
    results = run(
        pipeline,
        inputs,
        run_folder=tmp_path,
        manual_shapes=manual_shapes,  # type: ignore[arg-type]
        parallel=False,
    )
    assert results[-1].output == 21.0
    assert results[-1].output_name == "sum"
    assert load_outputs("sum", run_folder=tmp_path) == 21.0
    assert map_shapes(pipeline, inputs, manual_shapes) == {"vector": (1,)}


@dataclass(frozen=True)
class Geometry:
    x: float
    y: float


@dataclass(frozen=True)
class Mesh:
    geometry: Geometry
    mesh_size: float


@dataclass(frozen=True)
class Materials:
    geometry: Geometry
    materials: list[str]


@dataclass(frozen=True)
class Electrostatics:
    mesh: Mesh
    materials: Materials
    voltages: list[float]


@pytest.mark.parametrize("with_multiple_outputs", [False, True])
def test_pyiida_example(with_multiple_outputs: bool, tmp_path: Path) -> None:  # noqa: FBT001
    @pipefunc(output_name="geo")
    def make_geometry(x: float, y: float) -> Geometry:
        return Geometry(x, y)

    if with_multiple_outputs:

        @pipefunc(output_name=("mesh", "coarse_mesh"))
        def make_mesh(
            geo: Geometry,
            mesh_size: float,
            coarse_mesh_size: float,
        ) -> tuple[Mesh, Mesh]:
            return Mesh(geo, mesh_size), Mesh(geo, coarse_mesh_size)
    else:

        @pipefunc(output_name="mesh")
        def make_mesh(
            geo: Geometry,
            mesh_size: float,
            coarse_mesh_size: float,  # noqa: ARG001
        ) -> Mesh:
            return Mesh(geo, mesh_size)

    @pipefunc(output_name="materials")
    def make_materials(geo: Geometry) -> Materials:
        return Materials(geo, ["a", "b", "c"])

    @pipefunc(output_name="electrostatics")
    def run_electrostatics(
        mesh: Mesh,
        materials: Materials,
        V_left: float,  # noqa: N803
        V_right: float,  # noqa: N803
    ) -> Electrostatics:
        return Electrostatics(mesh, materials, [V_left, V_right])

    @pipefunc(output_name="charge")
    def get_charge(electrostatics: Electrostatics) -> float:
        # obviously not actually the charge; but we should return _some_ number that
        # is "derived" from the electrostatics.
        return sum(electrostatics.voltages)

    @pipefunc(output_name="average_charge")
    def average_charge(charge: np.ndarray) -> float:
        return np.mean(charge)

    pipeline = Pipeline(
        [
            make_geometry,
            make_mesh,
            make_materials,
            (run_electrostatics, "V_left[a], V_right[b] -> electrostatics[a, b]"),
            (get_charge, "electrostatics[i, j] -> charge[i, j]"),
            average_charge,
        ],
    )

    inputs = {
        "mesh_size": 0.01,
        "V_left": np.linspace(0, 2, 3),
        "V_right": np.linspace(-0.5, 0.5, 2),
        "x": 0.1,
        "y": 0.2,
        "coarse_mesh_size": 0.05,
    }
    assert map_shapes(pipeline, inputs) == {
        "V_right": (2,),
        "V_left": (3,),
        "electrostatics": (3, 2),
        "charge": (3, 2),
    }
    results = run(pipeline, inputs, run_folder=tmp_path, parallel=True)
    assert results[-1].output == 1.0
    assert results[-1].output_name == "average_charge"
    assert load_outputs("average_charge", run_folder=tmp_path) == 1.0


def test_validate_mapspec():
    def f(x: int) -> int:
        return x

    with pytest.raises(
        ValueError,
        match="The input of the function `f` should match the input of the MapSpec",
    ):
        PipeFunc(
            f,
            output_name="y",
            mapspec="x[i], yolo[i] -> y[i]",
        )

    with pytest.raises(
        ValueError,
        match="The output of the function `f` should match the output of the MapSpec",
    ):
        PipeFunc(
            f,
            output_name="y",
            mapspec="x[i] -> yolo[i]",
        )


def test_pipeline_with_defaults(tmp_path: Path) -> None:
    @pipefunc(output_name="z")
    def f(x: int, y: int = 1) -> int:
        return x + y

    @pipefunc(output_name="sum")
    def g(z: np.ndarray) -> int:
        return sum(z)

    pipeline = Pipeline([(f, "x[i] -> z[i]"), g])

    inputs = {"x": [0, 1, 2, 3]}
    results = run(pipeline, inputs, run_folder=tmp_path, parallel=False)
    assert results[-1].output == 10
    assert results[-1].output_name == "sum"
    assert map_shapes(pipeline, inputs) == {"x": (4,), "z": (4,)}
    sum_result = load_outputs("sum", run_folder=tmp_path)
    assert sum_result == 10
    sum_result = load_outputs("z", run_folder=tmp_path)
    assert sum_result.tolist() == [1, 2, 3, 4]  # type: ignore[union-attr]

    inputs = {"x": [0, 1, 2, 3], "y": 2}  # type: ignore[dict-item]
    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False)
    assert results[-1].output == 14


def test_pipeline_loading_existing_results(tmp_path: Path) -> None:
    counters = {"f": 0, "g": 0}

    @pipefunc(output_name="z")
    def f(x: int, y: int = 1) -> int:
        counters["f"] += 1
        return x + y

    @pipefunc(output_name="sum")
    def g(z: np.ndarray) -> int:
        counters["g"] += 1
        return sum(z)

    pipeline = Pipeline([(f, "x[i] -> z[i]"), g])
    inputs = {"x": [1, 2, 3]}

    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False, cleanup=True)
    assert results[-1].output == 9
    assert results[-1].output_name == "sum"
    assert counters["f"] == 3
    assert counters["g"] == 1

    results2 = pipeline.map(inputs, run_folder=tmp_path, parallel=False, cleanup=False)
    assert results2[-1].output == 9
    assert results2[-1].output_name == "sum"
    assert counters["f"] == 3
    assert counters["g"] == 1

    results3 = pipeline.map(inputs, run_folder=tmp_path, parallel=False, cleanup=True)
    assert results3[-1].output == 9
    assert results3[-1].output_name == "sum"
    assert counters["f"] == 6
    assert counters["g"] == 2


def test_run_info_compare(tmp_path: Path) -> None:
    @pipefunc(output_name="z", mapspec="x[i] -> z[i]")
    def f(x: int, y: int = 1) -> int:
        return x + y

    pipeline = Pipeline([f])
    inputs = {"x": [1, 2, 3]}

    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False, cleanup=True)
    assert results[-1].output.tolist() == [2, 3, 4]
    assert results[-1].output_name == "z"

    inputs = {"x": [1, 2, 3, 4]}
    with pytest.raises(ValueError, match="Shapes do not match previous run"):
        pipeline.map(inputs, run_folder=tmp_path, parallel=False, cleanup=False)
