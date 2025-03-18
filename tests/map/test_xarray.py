from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from pipefunc import Pipeline, pipefunc
from pipefunc.map import load_outputs
from pipefunc.map.xarray import load_xarray, xarray_dataset_from_results

if TYPE_CHECKING:
    from pathlib import Path


def test_to_xarray_1_dim(tmp_path: Path):
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([double_it])
    inputs = {"x": [1, 2, 3]}
    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False)
    output_name = results["y"].output_name
    mapspecs = pipeline.mapspecs()

    da = load_xarray(output_name, mapspecs, inputs, run_folder=tmp_path)
    expected_coords = {"x": inputs["x"]}
    expected_dims = ["i"]
    assert list(da.dims) == expected_dims
    assert da.coords["x"].to_numpy().tolist() == expected_coords["x"]
    assert da.to_numpy().tolist() == [2, 4, 6]


def test_to_xarray_2_dim(tmp_path: Path):
    @pipefunc(output_name="z", mapspec="x[i], y[j] -> z[i, j]")
    def f(x: int, y: int) -> int:
        return x + y

    pipeline = Pipeline([f])
    inputs = {"x": [1, 2, 3], "y": [4, 5]}
    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False)
    output_name = results["z"].output_name
    mapspecs = pipeline.mapspecs()

    da = load_xarray(output_name, mapspecs, inputs, run_folder=tmp_path)

    expected_coords = {"x": inputs["x"], "y": inputs["y"]}
    expected_dims = ["i", "j"]
    assert list(da.dims) == expected_dims
    assert da.coords["x"].to_numpy().tolist() == expected_coords["x"]
    assert da.coords["y"].to_numpy().tolist() == expected_coords["y"]
    assert da.to_numpy().tolist() == [[5, 6], [6, 7], [7, 8]]


def test_to_xarray_2_dim_zipped(tmp_path: Path):
    @pipefunc(output_name="r", mapspec="x[i], y[i], z[j] -> r[i, j]")
    def f(x: int, y: int, z: int) -> int:
        return x + y + z

    pipeline = Pipeline([f])
    inputs = {"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8]}
    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False)
    output_name = results["r"].output_name
    mapspecs = pipeline.mapspecs()

    da = load_xarray(output_name, mapspecs, inputs, run_folder=tmp_path)

    expected_coords = {"x": inputs["x"], "y": inputs["y"], "z": inputs["z"]}
    expected_dims = ["i", "j"]
    assert list(da.dims) == expected_dims
    assert da.coords["x:y"].to_numpy().tolist() == [(1, 4), (2, 5), (3, 6)]
    assert da.coords["z"].to_numpy().tolist() == expected_coords["z"]
    assert da.to_numpy().tolist() == [[12, 13], [14, 15], [16, 17]]


def test_to_xarray_1_dim_2_funcs(tmp_path: Path):
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def f(x: int) -> int:
        return 2 * x

    @pipefunc(output_name="z", mapspec="y[i] -> z[i]")
    def g(y: int) -> int:
        return y + 1

    pipeline = Pipeline([f, g])
    inputs = {"x": [1, 2, 3]}
    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False)

    output_name = results["z"].output_name
    mapspecs = pipeline.mapspecs()

    da = load_xarray(output_name, mapspecs, inputs, run_folder=tmp_path)

    expected_coords = {"x": inputs["x"]}
    expected_dims = ["i"]
    assert list(da.dims) == expected_dims
    assert da.coords["x"].to_numpy().tolist() == expected_coords["x"]
    assert da.to_numpy().tolist() == [3, 5, 7]


def test_to_xarray_from_step(tmp_path: Path):
    @pipefunc(output_name="x")
    def generate_ints(n: int) -> list[int]:
        """Generate a list of integers from 0 to n-1."""
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([generate_ints, double_it])
    inputs = {"n": 4}
    internal_shapes = {"x": (4,)}
    results = pipeline.map(
        inputs,
        internal_shapes=internal_shapes,  # type: ignore[arg-type]
        run_folder=tmp_path,
        parallel=False,
    )
    mapspecs = pipeline.mapspecs()
    output_name = results["y"].output_name

    da = load_xarray(
        output_name,
        mapspecs,
        inputs,
        run_folder=tmp_path,
    )

    x = load_outputs("x", run_folder=tmp_path)
    expected_coords = {"x": x}
    expected_dims = ["i"]
    assert list(da.dims) == expected_dims
    assert da.coords["x"].to_numpy().tolist() == expected_coords["x"]


def test_xarray_from_result():
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: int) -> int:
        return 2 * x

    def returns_array(a: int) -> npt.NDArray[np.int64]:
        return np.arange(a)

    def returns_custom_object(a: int) -> dict:
        return {"a": a}

    pipeline = Pipeline([double_it, returns_array, returns_custom_object])
    inputs = {"x": [1, 2, 3], "a": 10}
    results = pipeline.map(inputs, run_folder="tmp_path", parallel=False)
    ds = xarray_dataset_from_results(inputs, results, pipeline)
    assert "returns_array" in ds.coords
    assert "returns_custom_object" in ds.data_vars


def test_loop_over_list_with_elements_with_shape() -> None:
    # See PR #587
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def f(x: list[list[int]]) -> int:
        return len(x)

    pipeline = Pipeline([f])
    inputs = {"x": [[1, 2], [3, 4], [5, 6]]}
    results = pipeline.map(inputs, run_folder="tmp_path", parallel=False)
    assert results["y"].output.tolist() == [2, 2, 2]
    ds = xarray_dataset_from_results(inputs, results, pipeline)
    assert "x" in ds.coords


def test_no_inputs_to_xarray():
    @pipefunc(output_name="y")
    def f() -> int:
        return 1

    pipeline = Pipeline([f])
    results = pipeline.map({}, storage="dict", parallel=False)
    ds = results.to_xarray()
    assert "y" in ds.variables


def test_to_dataframe():
    data = {
        "player": ["Player A", "Player B", "Player C", "Player D", "Player E"],
        "at_bats": [200, 300, 330, 250, 175],
        "hits": [65, 82, 110, 92, 45],
        "home_runs": [10, 15, 20, 8, 5],
    }
    baseball_df = pd.DataFrame(data)

    @pipefunc(output_name="batting_avg", mapspec="hits[i], at_bats[i] -> batting_avg[i]")
    def calculate_batting_avg(hits: int, at_bats: int) -> float:
        """Calculate batting average from hits and at-bats."""
        return hits / at_bats if at_bats > 0 else 0.0

    @pipefunc(output_name="slugging", mapspec="hits[i], home_runs[i], at_bats[i] -> slugging[i]")
    def calculate_slugging(hits: int, home_runs: int, at_bats: int) -> float:
        """Calculate simplified slugging percentage."""
        return (hits + 3 * home_runs) / at_bats if at_bats > 0 else 0.0

    @pipefunc(output_name="category", mapspec="batting_avg[i] -> category[i]")
    def categorize_players(batting_avg: float) -> str:
        """Categorize players based on their statistics."""
        if batting_avg >= 0.300:
            return "Elite"
        if batting_avg >= 0.250:
            return "Good"
        return "Average"

    pipeline = Pipeline([calculate_batting_avg, calculate_slugging, categorize_players])
    result = pipeline.map(
        {
            "hits": baseball_df["hits"],
            "at_bats": baseball_df["at_bats"],
            "home_runs": baseball_df["home_runs"],
        },
        storage="dict",
        parallel=False,
    )
    df = result.to_dataframe()
    assert set(df.columns.tolist()) == {
        "batting_avg",
        "slugging",
        "category",
        "at_bats",
        "hits",
        "home_runs",
    }
