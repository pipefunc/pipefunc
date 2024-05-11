from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pipefunc import FileArray, Pipeline, pipefunc
from pipefunc._map import run_pipeline

if TYPE_CHECKING:
    from pathlib import Path


def test_simple_fan_out_fan_in(tmp_path: Path) -> None:
    @pipefunc(output_name="result")
    def simulate(seed: int) -> int:
        assert isinstance(seed, int)
        return seed * 2

    @pipefunc(output_name="sum")
    def post_process(result: list[int]) -> int:
        return sum(result)

    pipeline = Pipeline(
        [
            (simulate, "seed[i] -> result[i]"),
            post_process,
        ],
    )

    inputs = {"seed": [0, 1, 2, 3]}
    results = run_pipeline(pipeline, inputs, run_folder=tmp_path)
    assert results[-1].output == 12
    assert results[-1].output_name == "sum"


def test_simple_fan_out_fan_in_2_dim_array(tmp_path: Path) -> None:
    @pipefunc(output_name="result")
    def simulate(seed: int) -> int:
        assert isinstance(seed, np.int_)
        return seed * 2

    @pipefunc(output_name="sum")
    def post_process(result: np.ndarray) -> int:
        assert isinstance(result, np.ndarray)
        return np.sum(result, axis=0)

    pipeline = Pipeline(
        [
            (simulate, "seed[i, j] -> result[i, j]"),
            post_process,
        ],
    )

    inputs = {"seed": np.arange(12).reshape(3, 4)}
    results = run_pipeline(pipeline, inputs, run_folder=tmp_path)
    assert results[-1].output_name == "sum"
    assert results[-1].output.tolist() == [24, 30, 36, 42]


def test_simple_fan_out_fan_in_2_dim_array_to_1_dim(tmp_path: Path) -> None:
    @pipefunc(output_name="result")
    def simulate(seed: int) -> int:
        assert isinstance(seed, np.int_)
        return seed * 2

    @pipefunc(output_name="sum")
    def post_process(result: np.ndarray) -> int:
        assert isinstance(result, np.ndarray)
        return np.sum(result)

    pipeline = Pipeline(
        [
            (simulate, "seed[i, j] -> result[i, j]"),
            (post_process, "result[i, :] -> sum[i]"),
        ],
    )

    inputs = {"seed": np.arange(12).reshape(3, 4)}
    results = run_pipeline(pipeline, inputs, run_folder=tmp_path)
    assert results[-1].output_name == "sum"
    assert results[-1].output.tolist() == [12, 44, 76]


def test_simple_fan_out_fan_in_2_dim_array_to_1_dim_to_0_dim(tmp_path: Path) -> None:
    @pipefunc(output_name="result")
    def simulate(seed: int) -> int:
        assert isinstance(seed, np.int_)
        return seed * 2

    @pipefunc(output_name="sum")
    def take_sum(result: np.ndarray) -> int:
        assert isinstance(result, np.ndarray)
        return np.sum(result)

    @pipefunc(output_name="prod")
    def take_prod(result: np.ndarray) -> int:
        assert isinstance(result, np.ndarray)
        return np.prod(result)

    pipeline = Pipeline(
        [
            (simulate, "seed[i, j] -> result[i, j]"),
            (take_sum, "result[i, :] -> sum[i]"),
            take_prod,
        ],
    )

    inputs = {"seed": np.arange(1, 13).reshape(3, 4)}
    results = run_pipeline(pipeline, inputs, run_folder=tmp_path)
    assert results[-1].output_name == "prod"
    assert isinstance(results[-1].output, np.int_)
    assert results[-1].output == 1961990553600


def test_simple_fan_out_fan_in_from_step(tmp_path: Path) -> None:
    @pipefunc(output_name="seed")
    def generate_seeds(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="result")
    def simulate(seed: int) -> int:
        assert isinstance(seed, int)
        return seed * 2

    @pipefunc(output_name="sum")
    def post_process(result: list[int]) -> int:
        return sum(result)

    pipeline = Pipeline(
        [
            generate_seeds,
            (simulate, "seed[i] -> result[i]"),
            post_process,
        ],
    )
    inputs = {"n": 4}
    results = run_pipeline(pipeline, inputs, run_folder=tmp_path)
    assert results[-1].output == 12
    assert results[-1].output_name == "sum"


@pytest.mark.parametrize("with_multiple_outputs", [False, True])
def test_pyiida_example(with_multiple_outputs: bool, tmp_path: Path) -> None:  # noqa: FBT001
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

    @pipefunc(output_name="geo")
    def make_geometry(x: float, y: float) -> Geometry:
        return Geometry(x, y)

    if with_multiple_outputs:
        # TODO: make work with multiple outputs
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
    def average_charge(charge: FileArray) -> float:
        # .to_array() is a bit dumb; it loads in _all_ the data at once, but
        # this is the simplest way, and in this example the data is not so large.
        return np.mean(charge.to_array())

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
    results = run_pipeline(pipeline, inputs, run_folder=tmp_path)
    assert results[-1].output == 1.0
    assert results[-1].output_name == "average_charge"
