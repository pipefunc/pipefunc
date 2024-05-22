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
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {"x": (4,), "y": (4,)}
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
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {"x": (3, 4), "y": (3, 4)}
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
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {
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
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {
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
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {"y": (3,), "x": (3,), "z": (3, 3)}


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
            generate_ints,  # will autogen "... -> x[i]"
            (double_it, "x[i] -> y[i]"),
            take_sum,
        ],
    )
    assert pipeline.mapspecs_as_strings() == ["... -> x[i]", "x[i] -> y[i]"]
    inputs = {"n": 4}
    results = run(
        pipeline,
        inputs,
        run_folder=tmp_path,
        internal_shapes={"x": (4,)},
        parallel=False,
    )
    assert results[-1].output == 12
    assert results[-1].output_name == "sum"
    assert load_outputs("sum", run_folder=tmp_path) == 12

    shapes, masks = map_shapes(pipeline, inputs, {"x": (4,)})
    assert masks == {"x": (False,), "y": (True,)}
    assert shapes == {"x": (4,), "y": (4,)}

    with pytest.raises(ValueError, match="Internal shape for 'x' is missing."):
        map_shapes(pipeline, inputs)

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
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {
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
            generate_array,  # will autogen "... -> array[i, unnamed_0, unnamed_1]"
            (double_it, "array[i, :, :] -> vector[i]"),
            norm,
        ],
    )
    assert pipeline.mapspecs_as_strings() == [
        "... -> array[i, unnamed_0, unnamed_1]",
        "array[i, :, :] -> vector[i]",
    ]
    inputs = {"shape": (1, 2, 3)}
    internal_shapes: dict[str, int | tuple[int, ...]] = {"array": (1, 2, 3)}
    results = run(
        pipeline,
        inputs,
        run_folder=tmp_path,
        internal_shapes=internal_shapes,  # type: ignore[arg-type]
        parallel=False,
    )
    assert results[-1].output == 21.0
    assert results[-1].output_name == "sum"
    assert load_outputs("sum", run_folder=tmp_path) == 21.0
    shapes, masks = map_shapes(pipeline, inputs, internal_shapes)
    assert shapes == {"array": (1, 2, 3), "vector": (1,)}
    assert masks == {"array": (False, False, False), "vector": (True,)}


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
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {
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
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {"x": (4,), "z": (4,)}
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
    shapes, masks = map_shapes(pipeline, inputs_arr)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {"x": (2, 2), "y": (2, 2)}
    results = pipeline.map(inputs_arr, tmp_path, parallel=False)
    assert results[-1].output.tolist() == [[2, 4], [6, 8]]

    pipeline.add_mapspec_axis("x", axis="k")
    inputs = {"x": np.arange(2**3).reshape(2, 2, 2)}
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == {"x": (2, 2, 2), "y": (2, 2, 2)}
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
    expected = {"b": (2,), "a": (2,), "one": (2, 2)}
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == expected
    results = pipeline.map(inputs, tmp_path, parallel=False)
    assert results[-1].output == 4.0

    # Adding another axis to "one"
    pipeline.add_mapspec_axis("a", axis="k")
    assert str(one.mapspec) == "a[i, k], b[j] -> one[i, j, k]"
    assert str(two.mapspec) == "one[:, :, k] -> two[k]"
    assert str(three.mapspec) == "two[k] -> three[k]"

    # Run the pipeline
    inputs = {"a": np.ones((2, 3)), "b": [1, 1], "d": 1}
    expected = {"b": (2,), "a": (2, 3), "one": (2, 2, 3), "two": (3,), "three": (3,)}
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == expected
    results = pipeline.map(inputs, tmp_path, parallel=False)
    assert results[-1].output.tolist() == [4.0, 4.0, 4.0]

    # Adding another axis to "d"
    pipeline.add_mapspec_axis("d", axis="l")
    assert str(one.mapspec) == "a[i, k], b[j] -> one[i, j, k]"
    assert str(two.mapspec) == "one[:, :, k], d[l] -> two[k, l]"
    assert str(three.mapspec) == "two[k, l], d[l] -> three[k, l]"

    # Run the pipeline
    inputs = {"a": np.ones((2, 3)), "b": [1, 1], "d": [1, 1]}
    assert pipeline.map_parameters == {"one", "a", "three", "two", "b", "d"}
    expected = {"b": (2,), "a": (2, 3), "one": (2, 2, 3), "two": (3, 2), "three": (3, 2), "d": (2,)}
    shapes, masks = map_shapes(pipeline, inputs)
    assert all(all(mask) for mask in masks.values())
    assert shapes == expected
    results = pipeline.map(inputs, tmp_path, parallel=False)
    assert results[-1].output.tolist() == [[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]]


def test_add_mapspec_axis_unused_parameter() -> None:
    @pipefunc(output_name="result", mapspec="a[i] -> result[i]")
    def func(a):
        return a

    pipeline = Pipeline([func])

    pipeline.add_mapspec_axis("unused_param", axis="j")

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

    pipeline.add_mapspec_axis("a", axis="l")

    assert str(func1.mapspec) == "a[i, l], b[j] -> out1[i, j, l], out2[i, j, l]"
    assert str(func2.mapspec) == "out1[i, j, l], c[k] -> out3[i, j, k, l]"
    assert str(func3.mapspec) == "out3[:, :, :, l], out2[:, :, l] -> out4[l]"


def test_mapspec_internal_shapes(tmp_path: Path) -> None:
    @pipefunc(output_name="x", mapspec="... -> x[i]")
    def generate_ints(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: int, z: int) -> int:
        assert isinstance(x, int)
        return 2 * x + z

    @pipefunc(output_name="sum")
    def take_sum(y: list[int]) -> int:
        return sum(y)

    pipeline = Pipeline([generate_ints, double_it, take_sum])

    pipeline.add_mapspec_axis("z", axis="k")
    assert str(generate_ints.mapspec) == "... -> x[i]"
    assert str(double_it.mapspec) == "x[i], z[k] -> y[i, k]"
    assert str(take_sum.mapspec) == "y[:, k] -> sum[k]"

    inputs = {"n": 4, "z": [1, 2]}
    internal_shapes = {"x": 4}
    results = pipeline.map(inputs, tmp_path, internal_shapes, parallel=False)  # type: ignore[arg-type]
    assert results[-1].output.tolist() == [16, 20]
    expected = {"z": (2,), "x": (4,), "y": (4, 2), "sum": (2,)}
    shapes, masks = map_shapes(pipeline, inputs, internal_shapes)  # type: ignore[arg-type]
    assert masks == {"z": (True,), "x": (False,), "y": (True, True), "sum": (True,)}
    assert shapes == expected  # type: ignore[arg-type]


def test_add_mapspec_axis_multiple_axes() -> None:
    @pipefunc(output_name="result", mapspec="a[i], b[j] -> result[i, j]")
    def func(a, b):
        return a + b

    pipeline = Pipeline([func])

    pipeline.add_mapspec_axis("a", axis="k")
    pipeline.add_mapspec_axis("b", axis="l")

    assert str(func.mapspec) == "a[i, k], b[j, l] -> result[i, j, k, l]"


def test_add_mapspec_axis_parameter_in_output() -> None:
    @pipefunc(output_name="result", mapspec="a[i, j] -> result[i, j]")
    def func(a):
        return a

    pipeline = Pipeline([func])

    pipeline.add_mapspec_axis("a", axis="k")

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
    pipeline.add_mapspec_axis("a", axis="i")
    pipeline.add_mapspec_axis("b", axis="j")
    pipeline.add_mapspec_axis("x", axis="k")

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
    pipeline.add_mapspec_axis("a", axis="i")
    pipeline.add_mapspec_axis("b", axis="i")
    pipeline.add_mapspec_axis("x", axis="j")

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


def test_from_step_2_dim_array(tmp_path: Path) -> None:
    @pipefunc(output_name="x")
    def generate_ints(n: int) -> list[int]:
        return list(range(n))

    pipeline = Pipeline([(generate_ints, "... -> x[i]")])
    inputs = {"n": 4}
    internal_shapes = {"x": (4,)}
    shapes, masks = map_shapes(pipeline, inputs, internal_shapes)  # type: ignore[arg-type]
    assert shapes == {"x": (4,)}
    assert masks == {"x": (False,)}
    results = pipeline.map(inputs, tmp_path, internal_shapes, parallel=False)  # type: ignore[arg-type]
    assert results[-1].output == list(range(4))
    assert load_outputs("x", run_folder=tmp_path) == list(range(4))


def test_from_step_2_dim_array_2(tmp_path: Path) -> None:
    @pipefunc(output_name="c")
    def f(a: int, b: int) -> list[int]:
        return [a + b, a - b]

    pipeline = Pipeline([(f, "b[i] -> c[i, j]")])
    inputs = {"a": 1, "b": [1, 2]}
    internal_shapes = {"c": (2,)}
    shapes, masks = map_shapes(pipeline, inputs, internal_shapes)  # type: ignore[arg-type]
    assert shapes == {"b": (2,), "c": (2, 2)}
    assert masks == {"b": (True,), "c": (True, False)}
    results = pipeline.map(inputs, tmp_path, internal_shapes, parallel=False)  # type: ignore[arg-type]
    assert load_outputs("c", run_folder=tmp_path).tolist() == [[2, 0], [2, 0]]
    assert results[-1].output.shape == (2, 2)
    assert results[-1].output.tolist() == [[2, 0], [2, 0]]


def test_add_mapspec_axis_from_step(tmp_path: Path) -> None:
    @pipefunc(output_name="x")
    def generate_ints(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="y")
    def double_it(x: int) -> int:
        return 2 * x

    @pipefunc(output_name="side")
    def side_chain(z: int) -> int:
        return z

    @pipefunc(output_name="sum")
    def take_sum(y: list[int], z: int) -> int:
        return sum(y) + z

    pipeline = Pipeline(
        [
            (generate_ints, "... -> x[i]"),
            (double_it, "x[i] -> y[i]"),
            side_chain,
            take_sum,
        ],
    )

    inputs = {"n": 4, "z": 1}
    internal_shapes = {"x": (4,)}
    assert pipeline.mapspec_axes() == {"x": ("i",), "y": ("i",)}
    shapes, masks = map_shapes(pipeline, inputs, internal_shapes)  # type: ignore[arg-type]
    assert masks == {"x": (False,), "y": (True,)}
    assert shapes == {"x": (4,), "y": (4,)}
    results = pipeline.map(inputs, tmp_path, internal_shapes=internal_shapes, parallel=False)  # type: ignore[arg-type]
    assert results[-1].output == 13

    # Add an axis `j` to `x`
    pipeline_map = Pipeline(
        [
            (generate_ints, "n[j] -> x[i, j]"),
            (double_it, "x[i, j] -> y[i, j]"),
            side_chain,
            (take_sum, "y[:, j] -> sum[j]"),
        ],
    )
    inputs_map = {"n": [4], "z": 1}
    internal_shapes_map = {"x": (4,)}
    shapes, masks = map_shapes(pipeline_map, inputs_map, internal_shapes_map)  # type: ignore[arg-type]
    assert masks == {"n": (True,), "x": (False, True), "y": (True, True), "sum": (True,)}
    assert shapes == {"n": (1,), "x": (4, 1), "y": (4, 1), "sum": (1,)}
    results = pipeline_map.map(
        inputs_map,
        tmp_path,
        internal_shapes=internal_shapes_map,  # type: ignore[arg-type]
        parallel=False,
    )
    assert results[-1].output.tolist() == [13]

    # Do the same but with `add_mapspec_axis` on the first pipeline
    assert pipeline.mapspecs_as_strings() == ["... -> x[i]", "x[i] -> y[i]"]
    pipeline.add_mapspec_axis("n", axis="j")
    assert pipeline.mapspecs_as_strings() == [
        "n[j] -> x[i, j]",
        "x[i, j] -> y[i, j]",
        "y[:, j] -> sum[j]",
    ]
    results = pipeline.map(
        inputs_map,
        tmp_path,
        internal_shapes=internal_shapes_map,  # type: ignore[arg-type]
        parallel=False,
    )
    assert results[-1].output.tolist() == [13]


def test_return_2d_from_step(tmp_path: Path) -> None:
    @pipefunc(output_name="x")
    def generate_ints(n: int) -> np.ndarray:
        return np.ones((n, n))

    @pipefunc(output_name="y", mapspec="x[i, :] -> y[i]")
    def double_it(x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 1
        return 2 * sum(x)

    @pipefunc(output_name="sum")
    def take_sum(y: list[int]) -> int:
        return sum(y)

    pipeline = Pipeline([generate_ints, double_it, take_sum])
    r = pipeline.map({"n": 4}, tmp_path, internal_shapes={"x": (4, 4)})
    assert r[0].output.tolist() == np.ones((4, 4)).tolist()
    assert r[1].output.tolist() == [8, 8, 8, 8]
    assert r[2].output == 32


def test_growing_axis(tmp_path: Path) -> None:
    @pipefunc(output_name="x", mapspec="n[j] -> x[i, j]")
    def generate_ints(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i, j] -> y[i, j]")
    def double_it(x: int) -> int:
        return 2 * x

    @pipefunc(output_name="sum", mapspec="y[i, j] -> sum[j]")
    def take_sum(y: list[int]) -> int:
        return sum(y)

    pipeline = Pipeline([generate_ints, double_it, take_sum])
    inputs = {"n": [4, 5]}  # TODO: how to deal with this?
    # The internal_shapes becomes dynamic...
    internal_shapes = {"x": (4,)}
    results = pipeline.map(  # noqa: F841
        inputs,
        internal_shapes=internal_shapes,  # type: ignore[arg-type]
        run_folder=tmp_path,
    )
