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
    # Test `map` and a tmp run_folder
    results2 = pipeline.map(inputs, run_folder=None, parallel=False)
    assert results2[-1].output == 12

    axes = pipeline.mapspec_axes()
    assert axes == {"x": ("i",), "y": ("i",)}
    dimensions = pipeline.mapspec_dimensions()
    assert dimensions.keys() == axes.keys()
    assert all(dimensions[k] == len(v) for k, v in axes.items())


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
            (get_charge, "electrostatics[a, b] -> charge[a, b]"),
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


def test_nd_input_list(tmp_path: Path) -> None:
    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([(double_it, "x[i, j] -> y[i, j]")])

    inputs_list = {"x": [[1, 2], [3, 4]]}
    with pytest.raises(ValueError, match="Expected 2D"):
        pipeline.map(inputs_list, tmp_path, parallel=False)

    inputs_arr = {k: np.array(v) for k, v in inputs_list.items()}
    assert map_shapes(pipeline, inputs_arr) == {"x": (2, 2), "y": (2, 2)}
    results = pipeline.map(inputs_arr, tmp_path, parallel=False)
    assert results[-1].output.tolist() == [[2, 4], [6, 8]]

    pipeline.add_mapspec_axis("x", "k")
    inputs = {"x": np.arange(2**3).reshape(2, 2, 2)}
    assert map_shapes(pipeline, inputs) == {"x": (2, 2, 2), "y": (2, 2, 2)}
    results = pipeline.map(inputs, tmp_path, parallel=False)
    assert results[-1].output.tolist() == [[[0, 2], [4, 6]], [[8, 10], [12, 14]]]


def test_add_mapspec_axis(tmp_path: Path) -> None:
    @pipefunc(output_name="one", mapspec="a[i], b[j] -> one[i, j]")
    def one(a, b):
        assert isinstance(a, (int, np.float64))
        assert isinstance(b, (int, np.float64))
        return a * b

    @pipefunc(output_name="two")
    def two(one, d):
        assert isinstance(one, np.ndarray)
        return np.sum(one) / d

    @pipefunc(output_name="three")
    def three(two, d):
        return two / d

    pipeline = Pipeline([one, two, three])
    inputs = {"a": np.ones((2,)), "b": [1, 1], "d": 1}
    shapes = {"b": (2,), "a": (2,), "one": (2, 2)}
    assert map_shapes(pipeline, inputs) == shapes
    results = pipeline.map(inputs, tmp_path, parallel=False)
    assert results[-1].output == 4.0

    # Adding another axis to "one"
    pipeline.add_mapspec_axis("a", "k")
    assert str(one.mapspec) == "a[i, k], b[j] -> one[i, j, k]"
    assert str(two.mapspec) == "one[:, :, k] -> two[k]"
    assert str(three.mapspec) == "two[k] -> three[k]"

    # Run the pipeline
    inputs = {"a": np.ones((2, 3)), "b": [1, 1], "d": 1}
    shapes = {"b": (2,), "a": (2, 3), "one": (2, 2, 3), "two": (3,), "three": (3,)}
    assert map_shapes(pipeline, inputs) == shapes
    results = pipeline.map(inputs, tmp_path, parallel=False)
    assert results[-1].output.tolist() == [4.0, 4.0, 4.0]

    # Adding another axis to "d"
    pipeline.add_mapspec_axis("d", "l")
    assert str(one.mapspec) == "a[i, k], b[j] -> one[i, j, k]"
    assert str(two.mapspec) == "one[:, :, k], d[l] -> two[k, l]"
    assert str(three.mapspec) == "two[k, l], d[l] -> three[k, l]"

    # Run the pipeline
    inputs = {"a": np.ones((2, 3)), "b": [1, 1], "d": [1, 1]}
    assert pipeline.map_parameters == {"one", "a", "three", "two", "b", "d"}
    shapes = {"b": (2,), "a": (2, 3), "one": (2, 2, 3), "two": (3, 2), "three": (3, 2), "d": (2,)}
    assert map_shapes(pipeline, inputs) == shapes
    results = pipeline.map(inputs, tmp_path, parallel=False)
    assert results[-1].output.tolist() == [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]


def test_add_mapspec_axis_unused_parameter() -> None:
    @pipefunc(output_name="result", mapspec="a[i] -> result[i]")
    def func(a):
        return a

    pipeline = Pipeline([func])

    pipeline.add_mapspec_axis("unused_param", "j")

    assert str(func.mapspec) == "a[i] -> result[i]"


def test_add_mapspec_axis_complex_pipeline() -> None:
    @pipefunc(output_name=("out1", "out2"), mapspec="a[i], b[j] -> out1[i, j], out2[i, j]")
    def func1(a, b):
        return a + b, a - b

    @pipefunc(output_name="out3", mapspec="out1[i, j], c[k] -> out3[i, j, k]")
    def func2(out1, c):
        return out1 * c

    @pipefunc(output_name="out4")
    def func3(out2, out3):
        return out2 + out3

    pipeline = Pipeline([func1, func2, func3])

    pipeline.add_mapspec_axis("a", "l")

    assert str(func1.mapspec) == "a[i, l], b[j] -> out1[i, j, l], out2[i, j, l]"
    assert str(func2.mapspec) == "out1[i, j, l], c[k] -> out3[i, j, k, l]"
    assert str(func3.mapspec) == "out3[:, :, :, l], out2[:, :, l] -> out4[l]"


def test_mapspec_manual_shapes(tmp_path: Path) -> None:
    @pipefunc(output_name="x")
    def generate_ints(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: int, z: int) -> int:
        assert isinstance(x, int)
        return 2 * x + z

    @pipefunc(output_name="sum")
    def take_sum(y: list[int]) -> int:
        return sum(y)

    pipeline = Pipeline(
        [generate_ints, double_it, take_sum],
    )

    pipeline.add_mapspec_axis("z", "k")
    assert generate_ints.mapspec is None
    assert str(double_it.mapspec) == "x[i], z[k] -> y[i, k]"
    assert str(take_sum.mapspec) == "y[:, k] -> sum[k]"

    inputs = {"n": 4, "z": [1, 2]}
    manual_shapes = {"x": 4}
    results = pipeline.map(inputs, tmp_path, manual_shapes, parallel=False)  # type: ignore[arg-type]
    assert results[-1].output.tolist() == [16, 20]
    shapes = {"z": (2,), "y": (4, 2), "sum": (2,)}
    assert map_shapes(pipeline, inputs, manual_shapes) == shapes  # type: ignore[arg-type]


def test_add_mapspec_axis_multiple_axes() -> None:
    @pipefunc(output_name="result", mapspec="a[i], b[j] -> result[i, j]")
    def func(a, b):
        return a + b

    pipeline = Pipeline([func])

    pipeline.add_mapspec_axis("a", "k")
    pipeline.add_mapspec_axis("b", "l")

    assert str(func.mapspec) == "a[i, k], b[j, l] -> result[i, j, k, l]"


def test_add_mapspec_axis_parameter_in_output() -> None:
    @pipefunc(output_name="result", mapspec="a[i, j] -> result[i, j]")
    def func(a):
        return a

    pipeline = Pipeline([func])

    pipeline.add_mapspec_axis("a", "k")

    assert str(func.mapspec) == "a[i, j, k] -> result[i, j, k]"


def test_consistent_indices() -> None:
    with pytest.raises(
        ValueError,
        match="All axes should have the same name at the same index",
    ):
        Pipeline(
            [
                PipeFunc(lambda a, b: a + b, "f", mapspec="a[i], b[i] -> f[i]"),
                PipeFunc(lambda f, g: f + g, "h", mapspec="f[k], g[k] -> h[k]"),
            ],
        )

    with pytest.raises(
        ValueError,
        match="All axes should have the same length",
    ):
        Pipeline(
            [
                PipeFunc(lambda a: a, "f", mapspec="a[i] -> f[i]"),
                PipeFunc(lambda a: a, "g", mapspec="a[i, j] -> g[i, j]"),
            ],
        )


def test_consistent_indices_multiple_functions() -> None:
    pipeline = Pipeline(
        [
            PipeFunc(lambda a, b: a + b, "f", mapspec="a[i], b[j] -> f[i, j]"),
            PipeFunc(lambda f, c: f * c, "g", mapspec="f[i, j], c[k] -> g[i, j, k]"),
            PipeFunc(lambda g, d: g + d, "h", mapspec="g[i, j, k], d[l] -> h[i, j, k, l]"),
        ],
    )
    pipeline._validate_mapspec()  # Should not raise any error


def test_adding_axes_to_mapspec_less_pipeline():
    @pipefunc(output_name="c")
    def f_c(a, b):
        return a + b

    @pipefunc(output_name="d")
    def f_d(b, c, x=1):
        return b * c * x

    @pipefunc(output_name="e")
    def f_e(c, d, x=1):
        return c * d * x

    pipeline = Pipeline([f_c, f_d, f_e])
    pipeline.add_mapspec_axis("a", "i")
    pipeline.add_mapspec_axis("b", "j")
    pipeline.add_mapspec_axis("x", "k")

    assert str(f_c.mapspec) == "a[i], b[j] -> c[i, j]"
    assert str(f_d.mapspec) == "c[i, j], b[j], x[k] -> d[i, j, k]"
    assert str(f_e.mapspec) == "d[i, j, k], c[i, j], x[k] -> e[i, j, k]"

    assert pipeline.mapspecs_as_strings() == [
        "a[i], b[j] -> c[i, j]",
        "c[i, j], b[j], x[k] -> d[i, j, k]",
        "d[i, j, k], c[i, j], x[k] -> e[i, j, k]",
    ]


def test_adding_zipped_axes_to_mapspec_less_pipeline():
    @pipefunc(output_name="c")
    def f_c(a, b):
        return a + b

    @pipefunc(output_name="d")
    def f_d(b, c, x=1):
        return b * c * x

    @pipefunc(output_name="e")
    def f_e(c, d, x=1):
        return c * d * x

    pipeline = Pipeline([f_c, f_d, f_e])
    pipeline.add_mapspec_axis("a", "i")
    pipeline.add_mapspec_axis("b", "i")
    pipeline.add_mapspec_axis("x", "j")

    assert str(f_c.mapspec) == "a[i], b[i] -> c[i]"
    assert str(f_d.mapspec) == "c[i], b[i], x[j] -> d[i, j]"
    assert str(f_e.mapspec) == "d[i, j], c[i], x[j] -> e[i, j]"

    assert pipeline.mapspecs_as_strings() == [
        "a[i], b[i] -> c[i]",
        "c[i], b[i], x[j] -> d[i, j]",
        "d[i, j], c[i], x[j] -> e[i, j]",
    ]
    axes = pipeline.mapspec_axes()
    assert axes == {
        "a": ("i",),
        "b": ("i",),
        "c": ("i",),
        "x": ("j",),
        "d": ("i", "j"),
        "e": ("i", "j"),
    }
    dimensions = pipeline.mapspec_dimensions()
    assert dimensions.keys() == axes.keys()
    assert all(dimensions[k] == len(v) for k, v in axes.items())
