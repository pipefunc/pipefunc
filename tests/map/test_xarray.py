from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from pipefunc import Pipeline, pipefunc
from pipefunc.map import load_dataframe, load_outputs, load_xarray_dataset
from pipefunc.map.xarray import DimensionlessArray, load_xarray, xarray_dataset_from_results

if TYPE_CHECKING:
    from pathlib import Path

    from pipefunc.typing import Array


def test_to_xarray_1_dim(tmp_path: Path) -> None:
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: int) -> int:
        return 2 * x

    pipeline = Pipeline([double_it])
    inputs = {"x": [1, 2, 3]}
    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False, storage="dict")
    output_name = results["y"].output_name
    mapspecs = pipeline.mapspecs()

    da = load_xarray(output_name, mapspecs, inputs, run_folder=tmp_path)
    expected_coords = {"x": inputs["x"]}
    expected_dims = ["i"]
    assert list(da.dims) == expected_dims
    assert da.coords["x"].to_numpy().tolist() == expected_coords["x"]
    assert da.to_numpy().tolist() == [2, 4, 6]


def test_to_xarray_2_dim(tmp_path: Path) -> None:
    @pipefunc(output_name="z", mapspec="x[i], y[j] -> z[i, j]")
    def f(x: int, y: int) -> int:
        return x + y

    pipeline = Pipeline([f])
    inputs = {"x": [1, 2, 3], "y": [4, 5]}
    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False, storage="dict")
    output_name = results["z"].output_name
    mapspecs = pipeline.mapspecs()

    da = load_xarray(output_name, mapspecs, inputs, run_folder=tmp_path)

    expected_coords = {"x": inputs["x"], "y": inputs["y"]}
    expected_dims = ["i", "j"]
    assert list(da.dims) == expected_dims
    assert da.coords["x"].to_numpy().tolist() == expected_coords["x"]
    assert da.coords["y"].to_numpy().tolist() == expected_coords["y"]
    assert da.to_numpy().tolist() == [[5, 6], [6, 7], [7, 8]]


def test_to_xarray_2_dim_zipped(tmp_path: Path) -> None:
    @pipefunc(output_name="r", mapspec="x[i], y[i], z[j] -> r[i, j]")
    def f(x: int, y: int, z: int) -> int:
        return x + y + z

    pipeline = Pipeline([f])
    inputs = {"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8]}
    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False, storage="dict")
    output_name = results["r"].output_name
    mapspecs = pipeline.mapspecs()

    da = load_xarray(output_name, mapspecs, inputs, run_folder=tmp_path)

    expected_coords = {"x": inputs["x"], "y": inputs["y"], "z": inputs["z"]}
    expected_dims = ["i", "j"]
    assert list(da.dims) == expected_dims
    assert da.coords["x:y"].to_numpy().tolist() == [(1, 4), (2, 5), (3, 6)]
    assert da.coords["z"].to_numpy().tolist() == expected_coords["z"]
    assert da.to_numpy().tolist() == [[12, 13], [14, 15], [16, 17]]


def test_to_xarray_1_dim_2_funcs(tmp_path: Path) -> None:
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def f(x: int) -> int:
        return 2 * x

    @pipefunc(output_name="z", mapspec="y[i] -> z[i]")
    def g(y: int) -> int:
        return y + 1

    pipeline = Pipeline([f, g])
    inputs = {"x": [1, 2, 3]}
    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False, storage="dict")

    output_name = results["z"].output_name
    mapspecs = pipeline.mapspecs()

    da = load_xarray(output_name, mapspecs, inputs, run_folder=tmp_path)

    expected_coords = {"x": inputs["x"]}
    expected_dims = ["i"]
    assert list(da.dims) == expected_dims
    assert da.coords["x"].to_numpy().tolist() == expected_coords["x"]
    assert da.to_numpy().tolist() == [3, 5, 7]


def test_to_xarray_from_step(tmp_path: Path) -> None:
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
        storage="dict",
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


def test_xarray_from_result(tmp_path: Path) -> None:
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: int) -> int:
        return 2 * x

    def returns_array(a: int) -> npt.NDArray[np.int64]:
        return np.arange(a)

    def returns_custom_object(a: int) -> dict:
        return {"a": a}

    pipeline = Pipeline([double_it, returns_array, returns_custom_object])  # type: ignore[list-item]
    inputs = {"x": [1, 2, 3], "a": 10}
    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False, storage="dict")
    ds = xarray_dataset_from_results(inputs, results, pipeline)
    # Both single outputs (not part of mapspec) should be data variables, not coordinates
    assert "returns_array" in ds.data_vars
    assert "returns_custom_object" in ds.data_vars


def test_loop_over_list_with_elements_with_shape(tmp_path: Path) -> None:
    # See PR #587
    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def f(x: list[list[int]]) -> int:
        return len(x)

    pipeline = Pipeline([f])
    inputs = {"x": [[1, 2], [3, 4], [5, 6]]}
    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False, storage="dict")
    assert results["y"].output.tolist() == [2, 2, 2]
    ds = xarray_dataset_from_results(inputs, results, pipeline)
    assert "x" in ds.coords


def test_no_inputs_to_xarray() -> None:
    @pipefunc(output_name="y")
    def f() -> int:
        return 1

    pipeline = Pipeline([f])
    results = pipeline.map({}, parallel=False, storage="dict")
    ds = results.to_xarray()
    assert "y" in ds.variables


def test_to_dataframe() -> None:
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
        parallel=False,
        storage="dict",
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


def test_to_dataframe_with_single_output() -> None:
    @pipefunc(output_name="y")
    def f() -> int:
        return 1

    pipeline = Pipeline([f])
    result = pipeline.map({}, parallel=False, storage="dict")
    assert result["y"].output == 1

    # Check xarray
    ds = result.to_xarray()
    assert ds["y"].shape == ()
    assert ds["y"].to_numpy() == 1

    # Check dataframe
    df = result.to_dataframe()
    assert df.shape == (1, 1)
    assert df.columns.tolist() == ["y"]
    assert df.iloc[0]["y"] == 1


def test_2d_mapspec() -> None:
    # NotImplementedError: > 1 ndim Categorical are not supported at this time
    @pipefunc(output_name="x1")
    def x1() -> npt.NDArray[np.int_]:
        return np.array([[1, 2], [3, 4]])

    @pipefunc(output_name="x2")
    def x2() -> npt.NDArray[np.int_]:
        return np.array([[1, 2], [3, 4]])

    @pipefunc(output_name="y", mapspec="x1[i, j], x2[i, j] -> y[i, j]")
    def f(x1: npt.NDArray[np.int_], x2: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        return x1 + x2

    pipeline = Pipeline([x1, x2, f])
    results = pipeline.map({}, parallel=False, storage="dict")
    ds = results.to_xarray()
    assert "x1:x2" in ds.coords
    assert ds["x1:x2"].to_numpy().tolist() == [[(1, 1), (2, 2)], [(3, 3), (4, 4)]]
    df = results.to_dataframe()
    assert "x1" in df.columns
    assert "x2" in df.columns
    assert df.x1.tolist() == [1, 2, 3, 4]
    assert df.x2.tolist() == [1, 2, 3, 4]
    assert df.y.tolist() == [2, 4, 6, 8]


def test_2d_mapspec_with_nested_array() -> None:
    # MissingDimensionsError: cannot set variable 'x2' with 2-dimensional data without
    # explicit dimension names. Pass a tuple of (dims, data) instead.
    @pipefunc(output_name="x1")
    def x1() -> npt.NDArray[np.int_]:
        return np.array([[1, 2], [3, 4]])

    @pipefunc(output_name="x2")
    def x2() -> npt.NDArray[np.int_]:
        return np.array([[1, 2], [3, 4]])

    @pipefunc(output_name="y", mapspec="x1[i, j] -> y[i, j]")
    def f(x1: int, x2: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        return x1 + x2

    pipeline = Pipeline([x1, x2, f])

    results = pipeline.map({}, parallel=False, storage="dict")
    ds = results.to_xarray()
    assert "x1" in ds.coords
    assert "x2" in ds.data_vars
    assert ds["x1"].to_numpy().tolist() == [[1, 2], [3, 4]]
    assert ds["x2"].data.shape == ()
    assert isinstance(ds["x2"].data.item(), DimensionlessArray)
    assert ds["x2"].data.item().arr.tolist() == [[1, 2], [3, 4]]
    df = results.to_dataframe()
    assert "x1" in df.columns
    assert "x2" in df.columns
    assert df.x1.tolist() == [1, 2, 3, 4]
    assert df.x2.iloc[0].tolist() == [[1, 2], [3, 4]]
    assert df.x2.iloc[1].tolist() == [[1, 2], [3, 4]]
    assert df.x2.iloc[2].tolist() == [[1, 2], [3, 4]]
    assert df.x2.iloc[3].tolist() == [[1, 2], [3, 4]]
    assert df.y.iloc[0].tolist() == [[2, 3], [4, 5]]
    assert df.y.iloc[1].tolist() == [[3, 4], [5, 6]]
    assert df.y.iloc[2].tolist() == [[4, 5], [6, 7]]
    assert df.y.iloc[3].tolist() == [[5, 6], [7, 8]]


def test_1d_mapspec_returns_2d_array() -> None:
    @pipefunc(output_name="y", mapspec="... -> y[i]")
    def f() -> npt.NDArray[np.float64]:
        return np.ones((10, 3))

    @pipefunc(output_name="z", mapspec="y[i] -> z[i]")
    def g(y) -> int:
        return sum(y)

    pipeline = Pipeline([f, g])
    results = pipeline.map(inputs={})
    ds = results.to_xarray()
    assert "y" in ds.coords
    assert "z" in ds.data_vars


def test_1d_mapspec_returns_2d_list_of_lists() -> None:
    @pipefunc(output_name="y", mapspec="... -> y[i]")
    def f() -> list[list[float]]:
        return [[1, 2, 3] for _ in range(10)]

    @pipefunc(output_name="z", mapspec="y[i] -> z[i]")
    def g(y) -> int:
        return sum(y)

    pipeline = Pipeline([f, g])
    results = pipeline.map(inputs={})
    ds = results.to_xarray()
    assert "y" in ds.coords
    assert "z" in ds.data_vars


def test_single_output_2d_numpy_array(tmp_path: Path) -> None:
    """Test single output (not in mapspec) returning 2D numpy array."""

    @pipefunc(output_name="matrix")
    def generate_matrix() -> npt.NDArray[np.int_]:
        return np.array([[1, 2, 3], [4, 5, 6]])

    pipeline = Pipeline([generate_matrix])
    results = pipeline.map({}, run_folder=tmp_path, parallel=False, storage="dict")

    # Check raw output
    expected = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(results["matrix"].output, expected)

    # Check xarray structure
    ds = results.to_xarray()
    assert "matrix" in ds.data_vars, "matrix should be data variable"
    assert "matrix" not in ds.coords, "matrix should not be coordinate"

    # The value should be wrapped in DimensionlessArray
    assert isinstance(ds["matrix"].to_numpy().item(), DimensionlessArray)
    np.testing.assert_array_equal(ds["matrix"].to_numpy().item().arr, expected)

    # Check DataFrame

    df = load_dataframe("matrix", run_folder=tmp_path)
    assert df.shape == (1, 1), "Should have 1 row, 1 column"
    assert "matrix" in df.columns

    matrix_value = df["matrix"].iloc[0]
    assert isinstance(matrix_value, np.ndarray)
    assert matrix_value.shape == (2, 3)
    np.testing.assert_array_equal(matrix_value, expected)


def test_single_output_2d_list(tmp_path: Path) -> None:
    """Test single output (not in mapspec) returning 2D list."""

    @pipefunc(output_name="matrix")
    def generate_matrix() -> list[list[int]]:
        return [[1, 2, 3], [4, 5, 6]]

    pipeline = Pipeline([generate_matrix])
    results = pipeline.map({}, run_folder=tmp_path, parallel=False, storage="dict")

    # Check raw output
    expected = [[1, 2, 3], [4, 5, 6]]
    assert results["matrix"].output == expected

    # Check xarray structure
    ds = results.to_xarray()
    assert "matrix" in ds.data_vars, "matrix should be data variable"
    assert "matrix" not in ds.coords, "matrix should not be coordinate"

    # Check DataFrame

    df = load_dataframe("matrix", run_folder=tmp_path)
    assert df.shape == (1, 1), "Should have 1 row, 1 column"
    assert "matrix" in df.columns

    matrix_value = df["matrix"].iloc[0]
    assert isinstance(matrix_value, np.ndarray)
    assert matrix_value.shape == (2, 3)
    np.testing.assert_array_equal(matrix_value, np.array(expected, dtype=object))


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.complex64])
def test_single_output_numpy_dtype_preserved(tmp_path: Path, dtype: Any) -> None:
    """Single-output numpy arrays keep dtype in xarray Dataset and DataFrame."""

    values = np.array([1, 2, 3], dtype=dtype)
    if np.issubdtype(dtype, np.complexfloating):  # pragma: no branch - limited branch
        values = np.array([1 + 1j, 2 + 2j, 3 + 3j], dtype=dtype)

    @pipefunc(output_name="typed")
    def make_array() -> npt.NDArray:
        return values

    pipeline = Pipeline([make_array])
    results = pipeline.map({}, run_folder=tmp_path, parallel=False, storage="dict")

    ds = results.to_xarray()
    dimless = ds["typed"].to_numpy().item()
    assert isinstance(dimless, DimensionlessArray)
    np.testing.assert_array_equal(dimless.arr, values)
    assert dimless.arr.dtype == dtype

    df = load_dataframe("typed", run_folder=tmp_path)
    typed_value = df["typed"].iloc[0]
    assert isinstance(typed_value, np.ndarray)
    np.testing.assert_array_equal(typed_value, values)
    assert typed_value.dtype == dtype


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_scalar_numpy_dtype_preserved(tmp_path: Path, dtype: Any) -> None:
    """Zero-dimensional numpy outputs keep dtype when materialized."""

    scalar = np.array(5, dtype=dtype)

    @pipefunc(output_name="scalar_val")
    def make_scalar() -> npt.NDArray:
        return scalar

    pipeline = Pipeline([make_scalar])
    pipeline.map({}, run_folder=tmp_path, parallel=False, storage="dict")

    ds = load_xarray_dataset(
        "scalar_val",
        run_folder=tmp_path,
        load_intermediate=True,
    )
    dimless = ds["scalar_val"].to_numpy().item()
    assert isinstance(dimless, DimensionlessArray)
    np.testing.assert_array_equal(dimless.arr, scalar)
    assert dimless.arr.dtype == dtype

    df = load_dataframe("scalar_val", run_folder=tmp_path)
    scalar_value = df["scalar_val"].iloc[0]
    assert isinstance(scalar_value, np.ndarray)
    np.testing.assert_array_equal(scalar_value, scalar)
    assert scalar_value.dtype == dtype


@pytest.mark.parametrize("coord_dtype", [np.int8, np.float16])
def test_mapspec_coordinate_dtype_preserved(tmp_path: Path, coord_dtype: Any) -> None:
    """Coordinates derived from mapspec inputs retain their numpy dtype."""

    coord_values = np.array([1, 2, 3], dtype=coord_dtype)

    @pipefunc(output_name="axis", mapspec="... -> axis[i]")
    def axis() -> npt.NDArray:
        return coord_values

    @pipefunc(output_name="values", mapspec="axis[i] -> values[i]")
    def values(axis: Any) -> float:
        return float(axis * 2)

    pipeline = Pipeline([axis, values])
    results = pipeline.map({}, run_folder=tmp_path, parallel=False, storage="dict")

    ds = results.to_xarray()
    coord = ds.coords["axis"].to_numpy()
    np.testing.assert_array_equal(coord, coord_values)
    assert coord.dtype == coord_dtype

    df = results.to_dataframe()
    assert df["axis"].dtype == coord_dtype


def test_aggregation_to_scalar(tmp_path: Path) -> None:
    """Test aggregation/reduction from mapped output to scalar single output."""

    @pipefunc(output_name="x", mapspec="... -> x[i]")
    def generate_ints(n: int) -> list[int]:
        return list(range(n))

    @pipefunc(output_name="y", mapspec="x[i] -> y[i]")
    def double_it(x: int) -> int:
        return 2 * x

    @pipefunc(output_name="total")
    def sum_all(y: Array[int]) -> int:
        """Aggregate mapped values to scalar."""
        return int(np.sum(y))

    pipeline = Pipeline([generate_ints, double_it, sum_all])
    results = pipeline.map({"n": 5}, run_folder=tmp_path, parallel=False, storage="dict")

    # Check outputs
    assert results["x"].output == [0, 1, 2, 3, 4]
    assert results["y"].output.tolist() == [0, 2, 4, 6, 8]
    assert results["total"].output == 20  # sum([0, 2, 4, 6, 8])

    # Check xarray structure
    ds = results.to_xarray()
    assert "total" in ds.data_vars, "total should be data variable"
    assert "total" not in ds.coords, "total should not be coordinate"
    assert ds["total"].shape == (), "Scalar should have empty shape"

    # Check DataFrame
    df = load_dataframe("total", run_folder=tmp_path)
    assert df.shape == (1, 1), "Should have 1 row for scalar output"
    assert df["total"].iloc[0] == 20


def test_aggregation_to_array(tmp_path: Path) -> None:
    """Test aggregation/reduction from mapped output to array single output."""

    @pipefunc(output_name="matrices", mapspec="i[i] -> matrices[i]")
    def generate_matrices(i: int) -> npt.NDArray[np.int_]:
        """Generate different matrices."""
        return np.ones((3, 3), dtype=int) * i

    @pipefunc(output_name="row_sums")
    def compute_row_sums(matrices: Array[npt.NDArray]) -> npt.NDArray[np.int_]:
        """Aggregate matrices by computing sum of first row from each."""
        return np.array([m[0, :].sum() for m in matrices])

    pipeline = Pipeline([generate_matrices, compute_row_sums])
    inputs = {"i": [1, 2, 3]}
    results = pipeline.map(inputs, run_folder=tmp_path, parallel=False, storage="dict")

    # Check outputs
    expected_row_sums = np.array([3, 6, 9])  # [1*3, 2*3, 3*3]
    np.testing.assert_array_equal(results["row_sums"].output, expected_row_sums)

    # Check xarray structure
    ds = results.to_xarray()
    assert "row_sums" in ds.data_vars, "row_sums should be data variable"
    assert "row_sums" not in ds.coords, "row_sums should not be coordinate"

    # Check DataFrame

    df = load_dataframe("row_sums", run_folder=tmp_path)
    assert df.shape == (1, 1), "Should have 1 row for single output"

    row_sums_value = df["row_sums"].iloc[0]
    assert isinstance(row_sums_value, np.ndarray)
    assert row_sums_value.shape == (3,)
    np.testing.assert_array_equal(row_sums_value, expected_row_sums)


def test_unhashable_types() -> None:
    class Unhashable:
        def __init__(self, value):
            self.value = value

        def __hash__(self):
            msg = "Unhashable"
            raise TypeError(msg)

    @pipefunc(output_name="z", mapspec="x[i], y[i] -> z[i]")
    def f(x: Unhashable, y: int) -> int:
        return y

    pipeline = Pipeline([f])
    xs = [Unhashable(1), Unhashable(2), Unhashable(3)]
    results = pipeline.map(
        inputs={"x": xs, "y": [1, 2, 3]},
        parallel=False,
        storage="dict",
    )
    ds = results.to_xarray()
    assert "x:y" in ds.coords
    assert "z" in ds.data_vars
    df = results.to_dataframe()
    assert df.x.tolist() == xs
    assert df.y.tolist() == [1, 2, 3]
    assert df.z.tolist() == [1, 2, 3]
